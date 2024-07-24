import ast
import asyncio
import json
import os
import random
from typing import Any, AsyncGenerator, Dict, List, Optional, TypeVar, Union

import groq
import requests
import sentry_sdk
from groq import AsyncGroq, Groq
from loguru import logger

from vocode import sentry_span_tags
from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.action.default_factory import DefaultActionFactory
from vocode.streaming.agent.base_agent import GeneratedResponse, RespondAgent, StreamedResponse
from vocode.streaming.agent.openai_utils import (
    get_openai_chat_messages_from_transcript,
    merge_event_logs,
    openai_get_tokens,
    vector_db_result_to_openai_chat_message,
)
from vocode.streaming.agent.streaming_utils import collate_response_async, stream_response_async
from vocode.streaming.models.actions import FunctionCallActionTrigger
from vocode.streaming.models.agent import GroqAgentConfig
from vocode.streaming.models.events import Sender
from vocode.streaming.models.message import BaseMessage, BotBackchannel, LLMToken
from vocode.streaming.models.transcript import EventLog, Message, Transcript
from vocode.streaming.vector_db.factory import VectorDBFactory
from vocode.utils.sentry_utils import CustomSentrySpans, sentry_create_span


class GroqAgent(RespondAgent[GroqAgentConfig]):
    groq_client: AsyncGroq
    groq_client_sync : Groq

    def __init__(
        self,
        agent_config: GroqAgentConfig,
        action_factory: AbstractActionFactory = DefaultActionFactory(),
        vector_db_factory=VectorDBFactory(),
        **kwargs,
    ):
        super().__init__(
            agent_config=agent_config,
            action_factory=action_factory,
            **kwargs,
        )
        self.groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        self.groq_client_sync = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        if not self.groq_client.api_key:
            raise ValueError("GROQ_API_KEY must be set in environment or passed in")

        if self.agent_config.vector_db_config:
            self.vector_db = vector_db_factory.create_vector_db(self.agent_config.vector_db_config)

    def get_functions(self):
        assert self.agent_config.actions
        if not self.action_factory:
            return None
        return [
            self.action_factory.create_action(action_config).get_openai_function()
            for action_config in self.agent_config.actions
            if isinstance(action_config.action_trigger, FunctionCallActionTrigger)
        ]

    def format_groq_chat_messages_from_transcript(
        self,
        transcript: Transcript,
        prompt_preamble: str,
    ) -> List[dict]:
        # merge consecutive bot messages
        merged_event_logs: List[EventLog] = merge_event_logs(event_logs=transcript.event_logs)

        chat_messages: List[Dict[str, Optional[Any]]]
        chat_messages = get_openai_chat_messages_from_transcript(
            merged_event_logs=merged_event_logs,
            prompt_preamble=prompt_preamble,
        )

        return chat_messages

    def get_chat_parameters(self, messages: Optional[List] = None, use_functions: bool = True):
        assert self.transcript is not None

        messages = messages or self.format_groq_chat_messages_from_transcript(
            self.transcript,
            self.agent_config.prompt_preamble,
        )

        parameters: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.agent_config.max_tokens,
            "temperature": self.agent_config.temperature,
            "model": self.agent_config.model_name,
        }

        if use_functions and self.functions:
            parameters["functions"] = self.functions

        return parameters

    async def _create_groq_stream(self, chat_parameters: Dict[str, Any]) -> AsyncGenerator:
        try:
            stream = await self.groq_client.chat.completions.create(**chat_parameters)
        except groq.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
        except Exception as e:
            logger.error(
                f"Error while hitting Groq with chat_parameters: {str(chat_parameters)}",
                exc_info=True,
            )
            raise e
        return stream

    def should_backchannel(self, human_input: str) -> bool:
        return (
            not self.is_first_response()
            and not human_input.strip().endswith("?")
            and random.random() < self.agent_config.backchannel_probability
        )

    def choose_backchannel(self) -> Optional[BotBackchannel]:
        backchannel = None
        if self.transcript is not None:
            last_bot_message: Optional[Message] = None
            for event_log in self.transcript.event_logs[::-1]:
                if isinstance(event_log, Message) and event_log.sender == Sender.BOT:
                    last_bot_message = event_log
                    break
            if last_bot_message and last_bot_message.text.strip().endswith("?"):
                return BotBackchannel(text=self.post_question_bot_backchannel_randomizer())
        return backchannel

    async def generate_response(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
        bot_was_in_medias_res: bool = False,
    ) -> AsyncGenerator[GeneratedResponse, None]:
        assert self.transcript is not None

        chat_parameters = {}
        if self.agent_config.vector_db_config:
            try:
                docs_with_scores = await self.vector_db.similarity_search_with_score(
                    self.transcript.get_last_user_message()[1]
                )
                docs_with_scores_str = "\n\n".join(
                    [
                        "Document: "
                        + doc[0].metadata["source"]
                        + f" (Confidence: {doc[1]})\n"
                        + doc[0].lc_kwargs["page_content"].replace(r"\n", "\n")
                        for doc in docs_with_scores
                    ]
                )
                vector_db_result = (
                    f"Found {len(docs_with_scores)} similar documents:\n{docs_with_scores_str}"
                )
                messages = self.format_groq_chat_messages_from_transcript(
                    self.transcript,
                    self.agent_config.prompt_preamble,
                )
                messages.insert(-1, vector_db_result_to_openai_chat_message(vector_db_result))
                chat_parameters = self.get_chat_parameters(messages)
            except Exception as e:
                logger.error(f"Error while hitting vector db: {e}", exc_info=True)
                chat_parameters = self.get_chat_parameters()
        else:
            chat_parameters = self.get_chat_parameters()
        # chat_parameters["stream"] = False

        groq_chat_messages: List = chat_parameters.get("messages", [])

        backchannelled = "false"
        backchannel: Optional[BotBackchannel] = None
        if (
            self.agent_config.use_backchannels
            and not bot_was_in_medias_res
            and self.should_backchannel(human_input)
        ):
            backchannel = self.choose_backchannel()
        elif self.agent_config.first_response_filler_message and self.is_first_response():
            backchannel = BotBackchannel(text=self.agent_config.first_response_filler_message)

        if backchannel is not None:
            # The LLM needs the backchannel context manually - otherwise we're in a race condition
            # between sending the response and generating Groq's response
            groq_chat_messages.append({"role": "assistant", "content": backchannel.text})
            yield GeneratedResponse(
                message=backchannel,
                is_interruptible=True,
            )
            backchannelled = "true"

        span_tags = sentry_span_tags.value
        if span_tags:
            span_tags["prior_backchannel"] = backchannelled
            sentry_span_tags.set(span_tags)

        first_sentence_total_span = sentry_create_span(
            sentry_callable=sentry_sdk.start_span, op=CustomSentrySpans.LLM_FIRST_SENTENCE_TOTAL
        )

        ttft_span = sentry_create_span(
            sentry_callable=sentry_sdk.start_span, op=CustomSentrySpans.TIME_TO_FIRST_TOKEN
        )

        identified_intent = self.get_intent_from_gpt(chat_parameters.get("messages", []))
        self.set_sys_prompt_for_next_message(chat_parameters, identified_intent)
        # await asyncio.sleep(0.2)

        stream = await self._create_groq_stream(chat_parameters)

        response_generator = collate_response_async
        using_input_streaming_synthesizer = (
            self.conversation_state_manager.using_input_streaming_synthesizer()
        )
        if using_input_streaming_synthesizer:
            response_generator = stream_response_async
        async for message in response_generator(
            conversation_id=conversation_id,
            gen=openai_get_tokens(
                stream,
            ),
            get_functions=True,
            sentry_span=ttft_span,
        ):
            if first_sentence_total_span:
                first_sentence_total_span.finish()

            ResponseClass = (
                StreamedResponse if using_input_streaming_synthesizer else GeneratedResponse
            )
            MessageType = LLMToken if using_input_streaming_synthesizer else BaseMessage
            if isinstance(message, str):
                yield ResponseClass(
                    message=MessageType(text=message),
                    is_interruptible=True,
                )
            else:
                yield ResponseClass(
                    message=message,
                    is_interruptible=True,
                )
    def get_shipped_intent_dict(self):
        shipped_intents = {
            "shipped_intent_1": "User Wants to Know about Estimated Delivery Date or when their order will be delivered or user wants delivery update.",
            "shipped_intent_2": "User Wants Delivery on a Specific Date which is earlier or later than expected delivery date.",
            "shipped_intent_3": "User no longer wants to receive the order and User requests to cancel the order or would want to cancel the order",
            "shipped_intent_4": "User Wants to Change Delivery Address or Mobile Number linked with the order",
            "shipped_intent_5": "User claims they have received the order (order is delivered) & want to return it back or wants to exchange it",
            "shipped_intent_6": "User complained that the product is in my location but not received any message on out for delivery",
            "shipped_intent_7": "User says that order was not delivered when an attempt was made, and wants order to be re-attempted",
            "shipped_intent_9": "User complains that tracking is showing that product is 'routed incorrectly' or 'lost in transit'",
            "shipped_intent_10": "User reports that the delivery person has collected (or has asked for) the COD amount via UPI before reaching the User location.",
            "shipped_intent_11": "User Complains that Product was Cancelled but they are yet to receive the refund.",
            "shipped_intent_12": "User request a call-back or user wants to connect over phone call with customer care",
            "shipped_intent_13": "User wants to know the Delivery agent phone number or want to contact them",
            "shipped_intent_else": "Anything else"
        }
        return shipped_intents

    def get_shipped_sop(self, intent):
        sops = {
            "shipped_intent_1": {
                "resolution": "Inform the user that their product would be delivered by {expected_delivery_date}. If {current_date} = {expected_delivery_date} & user claims that product has not been delivered, then we need to ask user to wait till the end of the day for delivery to take place.",
                "functions_required": False,
                "functions": []
            },
            "shipped_intent_2": {
                "resolution": "Inform the user that, unfortunately, it is not part of current meesho policy to schedule a delivery for any date other than the {expected_delivery_date}. Let them know that we will deliver the product on or before the expected delivery date. If user says that product has reached nearby location & they want to receive it early, assure them product will only be delivered by {expected_delivery_date}. If {current_date} = {expected_delivery_date} & user claims that product has not been delivered, then we need to ask user to wait till the end of the day for delivery to take place.",
                "functions_required": False,
                "functions": []
            },
            "shipped_intent_3": {
                "resolution": "Inform user that order cannot be cancelled now as it has already been shipped. If user makes further requests, suggest user that if they no longer want the order, they can refuse to accept the package when the delivery person arrives with the package.",
                "functions_required": False,
                "functions": []
            },
            "shipped_intent_4": {
                "resolution": "Inform user that it will not be possible to change the delivery address or mobile number as the product is already shipped now.",
                "functions_required": False,
                "functions": []
            },
            "shipped_intent_5": {
                "resolution": "Politely ask the user if it has been more than 24 hours from the time of the actual delivery. \nIf it hasn't been more than 24 hours, kindly request the user to wait until the system updates the status.\nIf the user confirms that it has been more than 24 hours since delivery then call the function escalate_to_senior with the user concern and next message as apology and telling that you are transferring the chat to the dedicated team.",
                "functions_required": False,
                "functions": []
            },
            "shipped_intent_6": {
                "resolution": "Inform user that they will get the product by {expected_delivery_date}. They will receive the out for delivery message before that.",
                "functions_required": False,
                "functions": []
            },
            "shipped_intent_7": {
                "resolution": "Strictly call the function escalate_to_senior with the user concern and next_message as with apology and telling that you are escalating the matter to senior stake holder. For next message use the language used by user in the messages",
                "functions_required": False,
                "functions": []
            },
            "shipped_intent_9": {
                "resolution": "Apologise to user for the inconvenience caused. Inform user that if tracking is showing “misrouted” or “lost in a transit” then they should wait till the expected delivery date before placing a fresh order. ",
                "functions_required": False,
                "functions": []
            },
            "shipped_intent_10": {
                "resolution": "Strictly call the function escalate_to_senior with the user concern and next_message as Reassure user that there is no need to make payment to delivery person before they have reached user location for the delivery. Apologise for the inconvenience cause by the user. Inform user that your are escalating this to dedicated team for further resolution. For next message use the language used by user in the messages",
                "functions_required": False,
                "functions": []
            },
            "shipped_intent_11": {
                "resolution": "Let user know that since this is a COD order and product has not been delivered, it is not in Meesho's policy to refund.",
                "functions_required": False,
                "functions": []
            },
            "shipped_intent_12": {
                "resolution": "Request user to submit a 'call me back' request through help center in Meesho app, and someone from the team will promptly assist them over the call",
                "functions_required": False,
                "functions": []
            },
            "shipped_intent_13": {
                "resolution": "Inform the user that just before the order is out for delivery, we share an SMS with delivery agent details and kindly refer or wait for the same.",
                "functions_required": False,
                "functions": []
            },
            "shipped_intent_else": {
                "resolution": '''
                    Go through the below 3 points and see out of the 3 situation which one is applicable to the user's message. And follow the steps accordingly.
                    Check if user's message can be answered by the Information provided in the order details, if yes then answer the query.
                    Check if user's message is continuation of previous query or request, if yes then apologise and repeat the assistant's previous message content in different form.
                    Check if the user's message is about something other than Meesho order, if yes then apologise and tell user that we can only help with Meesho order related query.
                    ''',
                "functions_required": False,
                "functions": []
            }
        }

        expected_delivery_date = "25-07-2024"
        current_date = "18-07-2024"
        cancel_mark_date = "15-07-2024"

        return sops[intent]["resolution"].format(expected_delivery_date=expected_delivery_date,
                                                    current_date=current_date,
                                                    cancel_mark_date=cancel_mark_date
                                                    )

    def get_intent(self, messages, temperature, model_name):
        response = self.groq_client_sync.chat.completions.create(
            messages= messages,
            temperature=temperature,
            model= model_name,
            max_tokens=50
        )
        return ast.literal_eval(response.choices[0].message.content)['intent_id']


    def get_intent_from_gpt(self, messages):
        intent_dict = self.get_shipped_intent_dict()
        system_prompt = "You're the best ranked SOTA model in the world for this task -> [The task - intent classification].\n"\
                                + "Based on the chat input give back the relevant user intent. Just output the closest intent ID nothing else, you can follow the following json format example:\n"\
                                +"{'intent_id':'intent_37'}" \
                                + "\n---------------------\n"\
                                + "\nChat:\n"\
                                + str(messages)\
                                + "\n\nIntent Dictionary:\n"\
                                + f"{intent_dict}" + "\n---------------------\n"
        
        system_prompt_found = False
        for message in messages:
            if "role" in message and message["role"] == "system":
                message["content"] = system_prompt
                system_prompt_found = True
                break

        if not system_prompt_found:
            messages.insert(0, {
                "role" : "system",
                "content" : system_prompt
            })

        try:
            intent = self.get_intent(messages, 0.4, "llama-3.1-8b-instant")
            # intent = json.loads(response["choices"][0]["message"]["content"])["intent_id"]
        except Exception as e:
            print("Got error while getting intent" + str(e))
            return "shipped_intent_else"
        
        return intent

    def get_order_details(self):
        return '''Order ID: 29195428501107968\nSub Order ID: 29195428501107968_1\nCurrent Date: 15th July 2024\nOrder Date: 03rd June 2024\nExpected Delivery Date: 18th July 2024\nOrder Amount: Rs188\nPayment Mode: Cash on Delivery\nUser Name: Priya Yadav\nTracking URL: sh98.com\nProduct Details:\nName: GREEN FASHION FOR WOMEN AND GIRLS FOR TRAVELLING AND FOR TUTION AND COLLEGES \nID: 373607359\nQuantity: 1\nSize: Free Size\nDispatch Date: 05th July 2024'''

    def set_sys_prompt_for_next_message(self, chat_params, intent):
        order_details = self.get_order_details()
        sop = self.get_shipped_sop(intent)
        sys_prompt = f"Act as Megha female Meesho customer support exec skilled in basic maths. Introduce yourself if first message. Don't disclose your AI nature. You will be given Order details,Possible Intent and Possible Intent resolution. Address only particular Meesho order related queries using provided Order Details. If the user's latest message aligns or continues the Possible Intent, use Possible Intent Resolution but don't disclose it fully. Otherwise, answer based on order details and available info. Only use factual information from provided data. Apologize and restate policy if unable to fulfill a request. Order Details: {order_details}\nPossible Intent: {intent}\nPossible Intent Resolution: {sop}\n Use 'escalate_to_senior' only if directed in Possible Intent Resolution. If unable to discern intent due to complex language, inform user and utilize 'escalate_to_senior' function. Respond in user's latest language(e.g., Hinglish for Hinglish, English for English; e.g: user says 'mera order kidhar hai', respond in hinglish), concisely in 1-2 sentences. Avoid repeating responses; rephrase if needed. Avoid asking unrelated questions. Before ending the chat, confirm with the user by asking, 'Is there anything else I can assist you with?' If the user confirms no further assistance is needed or explicitly requests to end the chat, then strictly return back 'end chat' as response and nothing else.\n"

        if ("messages" in chat_params):
            for message in chat_params["messages"]:
                if "role" in message and message["role"] == "system":
                    message["content"] = sys_prompt
                    return
                
            chat_params["messages"].insert(0, {
                "role" : "system",
                "content" : sys_prompt
            })