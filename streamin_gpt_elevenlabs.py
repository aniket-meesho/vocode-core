import asyncio
import signal

from pydantic_settings import BaseSettings, SettingsConfigDict

from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.logging import configure_pretty_logging
from vocode.streaming.agent.chat_gpt_agent_custom import ChatGPTAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig, ElevenLabsSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
    TimeEndpointingConfig,
)
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.synthesizer.eleven_labs_websocket_synthesizer import ElevenLabsWSSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber_1 import DeepgramTranscriber

configure_pretty_logging()


class Settings(BaseSettings):
    """
    Settings for the streaming conversation quickstart.
    These parameters can be configured with environment variables.
    """

    openai_api_key: str = "sk-svcacct-9FdALmoN4fticeKMRBO8T3BlbkFJ7b5iBsA9tgoxiiIigFOu"
    azure_speech_key: str = "ENTER_YOUR_AZURE_KEY_HERE"
    deepgram_api_key: str = "cd3898ec57d1581c9881355c2874f633436658c8"

    azure_speech_region: str = "eastus"

    # This means a .env file can be used to overload these settings
    # ex: "OPENAI_API_KEY=my_key" will set openai_api_key over the default above
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


async def main():
    (
        microphone_input,
        speaker_output,
    ) = create_streaming_microphone_input_and_speaker_output(
        use_default_devices=False,
    )

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=TimeEndpointingConfig(time_cutoff_seconds = 0.01),
                api_key=settings.deepgram_api_key,
                min_interrupt_confidence = 0.95
            ),
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                openai_api_key=settings.openai_api_key,
                initial_message=BaseMessage(text="Hi Welcome to meesho, How can I help you today"),
                prompt_preamble="""The AI is having a pleasant conversation about life""",
                model_name="gpt-4o-mini",
                temperature=0.7,
                max_tokens=150
            )
        ),
        # synthesizer=ElevenLabsSynthesizer(
        #     ElevenLabsSynthesizerConfig.from_output_device(speaker_output, api_key="sk_40b3c6f7619c8866657ff13b91579ad72ebfbfd8941393bb")
        # ),
        synthesizer=ElevenLabsWSSynthesizer(
            ElevenLabsSynthesizerConfig.from_output_device(speaker_output, api_key="sk_40b3c6f7619c8866657ff13b91579ad72ebfbfd8941393bb", experimental_websocket=True)
        ),
    )
    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    signal.signal(signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate()))
    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main())