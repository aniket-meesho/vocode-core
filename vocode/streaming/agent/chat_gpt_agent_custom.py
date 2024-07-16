import os
import ast
import json
import numpy as np
import openai
# from langchain.embeddings import OpenAIEmbeddings
# from llama_index import LangchainEmbedding
from datetime import datetime,timedelta
from qdrant_client.http import models
from llama_index import ServiceContext
from llama_index import StorageContext, load_index_from_storage
from llama_index.llms import AzureOpenAI
from functools import wraps
from app.exceptions.CustomException import CustomException
from constants import constants
from middleware.statsdmiddleware import count_this, time_this
from services.helpers.logger import get_logger
from services.llm_router.llm_router import get_llm_provider_using_routing_module_for_emb, get_llm_provider_using_routing_module
import services.llms.models.azure_models as azure_models
import services.llms.models.openai_models as openai_models
from services.llm_router.router_config import Config as RConfig
from app.config import CONFIG
from circuitbreakers.emb_cb import emb_circuit_breaker
from services.intent.intent_dict_v2 import intent_dict as intent_dict_v2, intent_v2_to_v1, intent_helper
from services.intent.intent_index_creator import shipped_intents, ordered_intents, cancelled_intents, exchanged_intents, return_intents, delivered_intents
from app.ext_client_conn import get_qdrant_client
from services.llms import llm_service
import services.sop.cancelled_sop_service as cancelled_sop_service
import services.sop.delivered_sop_service as delivered_sop_service
import services.sop.exchanged_sop_service as exchanged_sop_service
import services.sop.ordered_sop_service as ordered_sop_service
import services.sop.return_sop_service as return_sop_service
import services.sop.shipped_sop_service as shipped_sop_service

logger = get_logger()

class IntentDetector:
    def __init__(self, request_body) -> None:
        self.request_body = request_body
        self.llm_provider, self.fallback_providers, _ = get_llm_provider_using_routing_module_for_emb(self.request_body)
        self.model_type = None
        self._set_roomcode()

    def _set_roomcode(self):
        roomcode = self.request_body.get(constants.ROOM_CODE)
        if roomcode is None:
            logger.error(f"roomcode is coming as None in Intent Detection, setting it to '0'")
            roomcode = '0'
        self.roomcode = roomcode

    @staticmethod
    def convert_chat_to_string(chat_input):
        chat_string = ""
        for chat_message in chat_input:
            chat_string += chat_message.get("role") + ": " + chat_message.get("content") + "\n"
        return chat_string
    
    def get_index(self, provider, intent):
        provider_vendor = 'OPENAI' if provider == RConfig.OPENAI_PROVIDER_NAME_FOR_EMB else 'AZURE'
        storage_context = StorageContext.from_defaults(persist_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{provider_vendor}/{intent}'))
        index = load_index_from_storage(storage_context, service_context=self.get_service_context(provider))
        return index

    def set_llm_provider(self, llm_provider):
        self.llm_provider = llm_provider

    def set_fallback_providers(self, fallback_provders):
        self.fallback_providers = fallback_provders
    
    def get_llm_provider(self):
        return self.llm_provider
    
    def get_fallback_providers(self):
        return self.fallback_providers
    
    def _get_llm_emb_instance(self, llm_provider):
        if llm_provider == RConfig.OPENAI_PROVIDER_NAME:
            llm_instance, emb_instance, model_type = openai_models.get_llm_and_embedding_instance_from_openai(llm_provider, RConfig.ADA002_NAME, self.request_body)
        else:
            llm_instance, emb_instance, model_type = azure_models.get_llm_and_embedding_instance_from_azure(llm_provider, RConfig.ADA002_NAME)
        self.model_type = model_type
        return llm_instance, emb_instance
    
    def get_service_context(self, llm_provider):
        llm_instance, emb_instance = self._get_llm_emb_instance(llm_provider)
        return ServiceContext.from_defaults(llm=llm_instance, embed_model=emb_instance)
    
    #make this and try_llm_provider better, not the ideal way
    @emb_circuit_breaker
    @time_this(
        constants.STATSD_LLM_PROVIDER_LATENCY,
        tags={
          constants.USE_CASE: 'intent_detection',
        },
        tag_value_paths={
            constants.STATSD_LLM_PROVIDER_NAME: ['provider'],
            constants.STATSD_LLM_PROVIDER_MODEL_TYPE: ['model_type'],
        }
    )
    def get_query_resp(self, index, query, provider, model_type):
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        return response

    def TRY_LLM_PROVIDERS(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            chat = IntentDetector.convert_chat_to_string(args[0])
            index_name, sys_prompt = func(self, *args, **kwargs)
            query = sys_prompt + ".\n chat input - \n" + chat
            index_storage_provider = 'OPENAI' if self.llm_provider == RConfig.OPENAI_PROVIDER_NAME_FOR_EMB else 'AZURE'
            end_state_llm_provider = '--'
            try:
                count_this(constants.STATSD_LLM_REQUEST, tags={
                    "llm_provider": self.llm_provider,
                    "use_case": "intent_detection",
                    'model_type': self.model_type if self.model_type is not None else '--',
                })
                first_priority_prov = self.llm_provider
                end_state_llm_provider = first_priority_prov
                logger.info(f'first attempt with <{first_priority_prov}> for intent detection')
                index = self.get_index(first_priority_prov, index_name)
                response = self.get_query_resp(index = index, query = query, provider = index_storage_provider, model_type=self.model_type if self.model_type is not None else '--')
                logger.info(f'model tried in first attempt: <{self.model_type}>')
            except Exception as e1:
                logger.error(f'{type(e1).__name__} occurred in first attempt: {self.llm_provider} failed. Error Message: {e1}. query : {query}')
                count_this(constants.STATSD_LLM_EXCEPTION, tags={"llm_provider": self.llm_provider, "error": type(e1).__name__, "use_case": "intent_detection", 'model_type': self.model_type if self.model_type is not None else '--'})
                for llm_prov in self.fallback_providers:
                    try:
                        count_this(constants.STATSD_LLM_REQUEST, tags={
                            "llm_provider": llm_prov,
                            "use_case": "intent_detection",
                            'model_type': self.model_type if self.model_type is not None else '--'
                        })
                        end_state_llm_provider = llm_prov
                        logger.info(f'2nd attempt with <{llm_prov}> for intent detection')
                        index_storage_provider = 'OPENAI' if llm_prov == RConfig.OPENAI_PROVIDER_NAME_FOR_EMB else 'AZURE'
                        index = self.get_index(llm_prov, index_name)
                        response = self.get_query_resp(index = index, query = query, provider = index_storage_provider, model_type=self.model_type if self.model_type is not None else '--')
                        logger.info(f'model tried in 2nd attempt: <{self.model_type}>')
                        break
                    except Exception as e2:
                        end_state_llm_provider = '--'
                        logger.error(f'{type(e2).__name__} occurred in 2nd attempt onwards: {llm_prov} failed\nError Message: {e2}')
                        count_this(constants.STATSD_LLM_EXCEPTION, tags={"llm_provider": llm_prov, "error": type(e2).__name__, "use_case": "intent_detection"})
            count_this(constants.STATSD_LLM_ROUTER, tags={
                'requested': self.llm_provider,
                'selected': end_state_llm_provider,
                'model_type': self.model_type if self.model_type is not None else '--',
                'index_name': index_name,
                'use_case': 'intent_detection'
                })
            if end_state_llm_provider == '--':
                logger.error(f"Fallback failed for all providers. {self.llm_provider}")
                raise CustomException(message="Intent Detection: Fallback failed for all providers.")
            elif self.llm_provider == end_state_llm_provider:
                count_this(constants.STATSD_LLM_ROUTER_REQUESTED_PROVIDER_SELECTED, tags={
                    'requested': self.llm_provider,
                    'model_type': self.model_type if self.model_type is not None else '--',
                    'index_name': index_name,
                    'use_case': 'intent_detection'
                })
            else:
                count_this(constants.STATSD_LLM_ROUTER_ALTERNATE_PROVIDER_SELECTED, tags={
                    'requested': self.llm_provider,
                    'selected': end_state_llm_provider,
                    'model_type': self.model_type if self.model_type is not None else '--',
                    'index_name': index_name,
                    'use_case': 'intent_detection'
                })
            final_intent_detection_model = f'{index_storage_provider}_{self.model_type}'
            RConfig.set_cloud_provider_for_intent_detection(self.roomcode, final_intent_detection_model)
            return response.response.strip()
        return wrapper
    
    

    @TRY_LLM_PROVIDERS
    def get_shipped_intent(self, chat_input, intent):
        if intent is None:
            sys_prompt = "Based on the chat input give back the nearest user intent.And just output the intent nothing else, example - shipped_intent_1, shipped_intent_2"
        else:
            sys_prompt = f"Following is a part of the conversation between a user and support executive(assistant) for Meesho an ecommerce company. This is the last intent that we detected {intent}. Based on the chat input determine the intent of user's last message. If the message is continuation of previous message output the same intent.And just output the intent nothing else, example - shipped_intent_1, shipped_intent_2."
        index_name = 'shipped_intent_index'
        return index_name, sys_prompt

    @TRY_LLM_PROVIDERS
    def get_ordered_intent(self, chat_input, intent):
        if intent is None:
            sys_prompt = "Based on the chat input give back the nearest user intent.And just output the intent nothing else, example - ordered_intent_1, ordered_intent_2"
        else:
            sys_prompt = f"Following is a part of the conversation between a user and support executive(assistant) for Meesho an ecommerce company. This is the last intent that we detected {intent}. Based on the chat input determine the intent of user's last message. If the message is continuation of previous message output the same intent.And just output the intent nothing else, example - ordered_intent_1, ordered_intent_2."
        index_name = 'ordered_intent_index'
        return index_name, sys_prompt

    @TRY_LLM_PROVIDERS
    def get_cancelled_intent(self, chat_input, intent):
        if intent is None:
            sys_prompt = "Based on the chat input give back the nearest user intent.And just output the intent nothing else, example - cancelled_intent_1, cancelled_intent_2"
        else:
            sys_prompt = f"Following is a part of the conversation between a user and support executive(assistant) for Meesho an ecommerce company. This is the last intent that we detected {intent}. Based on the chat input determine the intent of user's last message. If the message is continuation of previous message output the same intent.And just output the intent nothing else, example - cancelled_intent_1, cancelled_intent_2."
        index_name = 'cancelled_intent_index'
        return index_name, sys_prompt

    @TRY_LLM_PROVIDERS
    def get_exchanged_intent(self, chat_input, intent):
        if intent is None:
            sys_prompt = "Based on the chat input give back the nearest user intent.And just output the intent nothing else, example - exchanged_intent_1, exchanged_intent_2"
        else:
            sys_prompt = f"Following is a part of the conversation between a user and support executive(assistant) for Meesho an ecommerce company. This is the last intent that we detected {intent}. Based on the chat input determine the intent of user's last message. If the message is continuation of previous message output the same intent.And just output the intent nothing else, example - exchanged_intent_1, exchanged_intent_2."
        index_name = 'exchanged_intent_index'
        return index_name, sys_prompt

    @TRY_LLM_PROVIDERS
    def get_return_intent(self, chat_input, intent):
        if intent is None:
            sys_prompt = "Based on the chat input give back the nearest user intent.And just output the intent nothing else, example - return_intent_1, return_intent_2"
        else:
            sys_prompt = f"Following is a part of the conversation between a user and support executive(assistant) for Meesho an ecommerce company. This is the last intent that we detected {intent}. Based on the chat input determine the intent of user's last message. If the message is continuation of previous message output the same intent.And just output the intent nothing else, example - return_intent_1, return_intent_2."
        index_name = 'return_intent_index'
        return index_name, sys_prompt
    
    @TRY_LLM_PROVIDERS
    def get_delivered_intent(self, chat_input, intent):
        if intent is None:
            sys_prompt = "Based on the chat input give back the nearest user intent.And just output the intent nothing else, example - deliver_intent_1, deliver_intent_2"
        else:
            sys_prompt = f"Following is a part of the conversation between a user and support executive(assistant) for Meesho an ecommerce company. This is the last intent that we detected {intent}. Based on the chat input determine the intent of user's last message. If the message is continuation of previous message output the same intent.And just output the intent nothing else, example - deliver_intent_1, deliver_intent_2."
        index_name = 'delivered_intent_index'
        return index_name, sys_prompt
    
class IntentDetector_v2:
    def __init__(self):
        pass

    def get_scenario(self, details):
        order_status = details['Order Status']
        if order_status == constants.CANCELLED:
            return cancelled_sop_service.determine_scenario(details)
        if order_status == constants.SHIPPED:
            return shipped_sop_service.determine_scenario(details)
        if order_status == constants.DELIVERED:
            return delivered_sop_service.determine_scenario(details)
        if order_status == constants.RETURN:
            return return_sop_service.determine_scenario(details)
        if order_status == constants.ORDERED:
            return ordered_sop_service.determine_scenario(details)
        if order_status == constants.EXCHANGED:
            return exchanged_sop_service.determine_scenario(details)

    def convert_chat_to_string(self, chat_input):
        chat_string = ""
        for chat_message in chat_input:
            chat_string += chat_message.get("role") + ": " + chat_message.get("content") + "\n"
        return chat_string
    
    def get_all_user_messages(self, chat_input):
        user_messages = []
        for chat_message in chat_input:
            if chat_message.get("role") == 'user':
                user_messages.append(chat_message.get("content"))
        return user_messages
    
    def fetch_sys_prompt_2(self, chat_string, shortlist_intent_dict, flag, extracted_intents = None):
        shortlist_intent_string=''
        for intent_id in shortlist_intent_dict:
            shortlist_intent_string += f"{intent_id} : {shortlist_intent_dict[intent_id]}\n"
        if flag==True:
            final_sys_prompt = "You're the best ranked SOTA model in the world for this task -> [The task - intent classification].\n"\
                            + "Based on the chat input give back the relevant user intent. Just output the closest intent ID nothing else, you can follow the following json format example:\n"\
                            +"{'intent_id':'intent_37'}" \
                            + "\n\n---------------------\n"\
                            + "Chat:\n"\
                            + str(chat_string)\
                            + "\n\nIntent Dictionary:\n"\
                            + f"{shortlist_intent_string}" + "\n---------------------\n"
        else:
            extracted_intents_string = ''
            for key in extracted_intents:
                extracted_intents_string += f"{key}, {extracted_intents[key]}\n"
            final_sys_prompt = "You're the best ranked SOTA model in the world for this task -> [The task - intent classification].\n"\
                            + "Based on the chat input give back the relevant user intent. Just output the closest intent ID from the Intent Dictionary and nothing else, you can follow the following json format example:\n"\
                            +"{'intent_id':'intent_37'}" \
                            + "\n\n---------------------\n"\
                            + "Chat:\n"\
                            + str(chat_string)\
                            + "\nExtracted user intents:\n" \
                            + f"{extracted_intents_string}" \
                            + "\nIntent Dictionary:\n"\
                            + f"{shortlist_intent_string}" + "intent_else : Anything else" + "\n---------------------\n"
        return final_sys_prompt
    
    def fetch_sys_prompt_1(self):
        system_prompt = """You are world's best intent identifier.
            User will give you an array of array of user chats.
            Each nested array is a user messages sent in a chat.
            Understand the whole conversation and find out the user intents in the chat.
            There could be a possibility that multiple lines combined makes a single statement.
            Classify it into one line intents which are unique and mutually exclusive and give the output as an array of intent names and one line description separated by ':'.
            The output should be such that it can be parsed in a python variable directly. Don't output anything else at all.
                    1. The intents should be at a chat level only.
                    2. The intent of a chat(not message) should be very granular, for the same create as many intents as needed for a chat. 
                    3. Again make sure the intents are of a chat and not each message, so understand the chat and then finalize on minimal, distinct user intents. Merge similar intents into one intent.
                    4. The intent should not be vague, it should be very definitive.
                    5. The output should strictly be as outlined above i.e. {'intent_name_1 : intent_description_1',...'intent_name_n : intent_description_n'}.
            eg. INPUT - 
            [
                "Mera parcel abhi tk nhi aaya h",
                "Kab tak aayega parcel",
                "Raat hogaya h",
                "Kitne din se to bol rhe h",
                "Abhi tk aaya nhi h",
                "Agar kal tk nhi aaya to Mai cancel kr dunge",
                "Aur kbhi order nhi karuge",
                "Ye mera last mesho order hoga",
                "Agar iss br order nhi aaya to",
                "Aur aap kya madad karoge",
                "To aap kab bhejoge mera parcel",
                "To kab aayga order",
                "To btt kro aap",
                "Maine 18 mar ko order Kiya tha",
                "Itna time nhi lgta h",
                "Apne company me btt kro",
                "Ki kb aayga",
                "Haa aap mera order bas de de"
            ]
            eg. OUTPUT(JSON) - {"Delivery status and time inquiry": "User is inquiring about the delivery status and time of their parcel.",
            "Threat": "User is threatening to cancel the order or not order again",
            "Escalation request": "User is asking to escalate",
            "Urgent delivery request": "User is urgently requesting for the delivery of their order"}

            This is to be done such that an e-commerce virtual chat assistant should be able to take unique action based on each unique intent.
            Understand and follow the eg. Input and output very carefully."""
        return system_prompt
    
    def find_closest_embeddings(self, query_embedding, group_id, client='meesho_cx'):
        qdrant_client = get_qdrant_client()
        search_params = models.SearchParams(
            hnsw_ef=128,
            exact=True  
        )
        
        filter_conditions = models.Filter(
            must=[
                models.FieldCondition(
                    key="group_id",
                    match=models.MatchValue(value=group_id)
                )
            ]
        )

        response = qdrant_client.search(
            collection_name=client,
            query_vector=query_embedding,
            limit=5,
            search_params=search_params,
            query_filter=filter_conditions,
        )
        
        return response
    
    def get_emb_response(self, llm_provider, text):
        if llm_provider == RConfig.OPENAI_PROVIDER_NAME:
            response = openai_models.get_embedding_via_api(RConfig.ADA003_LARGE_NAME, text)
        else:
            response = azure_models.get_embedding_via_api(llm_provider, RConfig.ADA003_LARGE_NAME, text)
        result = []
        for i in response['data']:
            result.append(i['embedding'])
        return result
    
    def get_embeddings(self, text, request_body):
        llm_provider, fallback_providers, _ = get_llm_provider_using_routing_module_for_emb(request_body, RConfig.ADA003_LARGE_NAME)
        try:
            result = self.get_emb_response(llm_provider, text)
            return result
        except Exception as e1:
            logger.error(f'In first attempt for ada003 large emb generation: {llm_provider} failed, error: {e1}')
            for llm_prov in fallback_providers:
                try:
                    result = self.get_emb_response(llm_prov, text)
                    return result
                except Exception as e2:
                    logger.error(f'In 2nd attempt for ada003 large emb generation: {llm_prov} failed, error: {e2}')
        return None
    
    def get_shortlisted_intents(self, extracted_intent_descriptions, group_id, request):
        extracted_intent_descriptions = list(extracted_intent_descriptions.values())
        new_embedding_list = self.get_embeddings(extracted_intent_descriptions, request)
        new_embedding_list = np.array(new_embedding_list)
        intent_set = set()
        for new_embedding in new_embedding_list:
            closest_embeddings = self.find_closest_embeddings(new_embedding, group_id)
            for result in closest_embeddings:
                intent_set.add(result.payload['intent_id'])
        shortlist_intent_dict = {key: intent_dict_v2[key] for key in intent_set if key in intent_dict_v2}
        return shortlist_intent_dict
    
    def preprocessing_intent_dict(self, order_status):
        if CONFIG.INTENT_DICTIONARY_VERSION == 'v1':
            if order_status == constants.CANCELLED:
                return cancelled_intents
            elif order_status == constants.SHIPPED:
                return shipped_intents
            elif order_status == constants.DELIVERED:
                return delivered_intents
            elif order_status == constants.RETURN:
                return return_intents
            elif order_status == constants.ORDERED:
                return ordered_intents
            elif order_status == constants.EXCHANGED:
                return exchanged_intents
            else:
                return None
        elif CONFIG.INTENT_DICTIONARY_VERSION == 'v2':
            new_intent_dict = dict()
            for key in intent_helper.keys():
                if order_status in intent_helper[key]:
                    new_intent_dict[key] = intent_dict_v2[key]
            return new_intent_dict
        else:
            return None
        
    def get_pre_intent_messages(self, system_prompt, user_messages):
        messages = [{"role": "user","content": str(user_messages)}]
        messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })
        return messages
    
    def get_intent_messages(self, system_prompt, request):
        messages = [
            {
                "role": m["role"],
                "content": m["content"]
            } for m in request["chat"]
        ]
        messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })
        return messages
    
    def remap_to_old_intent_ids(self, intent_identifier, order_status):
        for intent_id in intent_v2_to_v1[intent_identifier]:
            x = intent_id.split('_')[0]
            if x == 'exchanged':
                x = 'exchange'
            if x == 'deliver':
                x = 'delivered'
            x = x.upper()
            if x == order_status:
                return intent_id
        return None


    def get_intent(self, request, details, scenario=None):
        chat_string = self.convert_chat_to_string(request['chat'])
        intent_dict_ = self.preprocessing_intent_dict(scenario) #To change to scenario filtering eventually
        use_case  = 'intent_detection'
        request["use_case"] = use_case
        request["temperature"] = 0.2
        model_type = RConfig.GPT35_NAME
        flag = len(intent_dict_.keys())<CONFIG.INTENT_SCALEUP_THRESHOLD
        if flag:
            sys_prompt_2 = self.fetch_sys_prompt_2(chat_string, intent_dict_, flag)
        else:
            user_messages = self.get_all_user_messages(request['chat'])
            sys_prompt_1 = self.fetch_sys_prompt_1()
            request["intent_messages"] = self.get_pre_intent_messages(sys_prompt_1,user_messages)
            llm_provider, llm_fallback_providers, model_type = get_llm_provider_using_routing_module(request, details, model_type=model_type)
            response, end_state_llm_provider = llm_service.get_response_from_gpt(llm_provider, llm_fallback_providers, model_type, request, use_case, 'json')
            extracted_intent_descriptions = ast.literal_eval(response['choices'][0]['message']['content'])
            shortlisted_intents = self.get_shortlisted_intents(extracted_intent_descriptions, scenario, request) # to replace order_status with group_id = scenario eventually
            sys_prompt_2 = self.fetch_sys_prompt_2(chat_string, shortlisted_intents, flag, extracted_intents=extracted_intent_descriptions)
            
        request["intent_messages"] = self.get_intent_messages(sys_prompt_2,request)
        llm_provider, llm_fallback_providers, model_type = get_llm_provider_using_routing_module(request, details, model_type=model_type)
        response, end_state_llm_provider = llm_service.get_response_from_gpt(llm_provider, llm_fallback_providers, model_type, request, use_case, 'json')    
        intent_identifier = ast.literal_eval(response['choices'][0]['message']['content'])['intent_id']
        intent = intent_dict_[intent_identifier]
        if CONFIG.INTENT_DICTIONARY_VERSION == 'v2': # This will be removed once full scale-up/externalisation logic done
            intent_identifier = self.remap_to_old_intent_ids(intent_identifier, scenario)
        return intent, intent_identifier

    def get_intent_details(self, request, details):
        # scenario = self.get_scenario(details) # to change to new sop_v2 format
        scenario = details["Order Status"]
        intent, intent_identifier = self.get_intent(request, details, scenario)
        return intent_identifier