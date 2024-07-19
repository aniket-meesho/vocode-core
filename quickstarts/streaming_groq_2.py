import asyncio
import logging
import os
import signal

import aiohttp
from pydantic_settings import BaseSettings, SettingsConfigDict

from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.logging import configure_pretty_logging
from vocode.streaming.agent.groq_agent_custom import GroqAgent
from vocode.streaming.models.agent import GroqAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber

configure_pretty_logging()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Settings(BaseSettings):
    """
    Settings for the streaming conversation quickstart.
    These parameters can be configured with environment variables.
    """

    openai_api_key: str = "ENTER_YOUR_OPENAI_API_KEY_HERE"
    azure_speech_key: str = "ENTER_YOUR_AZURE_KEY_HERE"
    deepgram_api_key: str = "ENTER_YOUR_DEEPGRAM_API_KEY_HERE"
    GROQ_API_KEY: str = "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"

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

    async with aiohttp.ClientSession() as session:
        microphone_input, speaker_output = create_streaming_microphone_input_and_speaker_output(
            use_default_devices=False,
            output_file='output_audio.wav'
        )
        print("Microphone and speaker initialized")

        elevenlabs_config = ElevenLabsSynthesizerConfig.from_output_device(
            output_device=speaker_output,
            api_key="sk_40b3c6f7619c8866657ff13b91579ad72ebfbfd8941393bb"
        )
    # (
    #     microphone_input,
    #     speaker_output,
    # ) = create_streaming_microphone_input_and_speaker_output(
    #     use_default_devices=False,
    # )

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=PunctuationEndpointingConfig(),
                api_key=settings.deepgram_api_key,
            ),
        ),
        agent=GroqAgent(
            GroqAgentConfig(
                api_key=settings.GROQ_API_KEY,
                initial_message=BaseMessage(text="Hi Welcome to meesho, How can I help you today"),
                prompt_preamble="""The AI is having a pleasant conversation about life""",
                model_name="llama3-70b-8192",
                temperature=0.6,
                max_tokens=250
            )
        ),
        synthesizer=ElevenLabsSynthesizer(elevenlabs_config)
    )
    async def shutdown(signal, loop):
            await conversation.terminate()
            loop.stop()

    loop = asyncio.get_running_loop()

    def signal_handler(sig):
        loop.call_soon_threadsafe(asyncio.create_task, shutdown(sig, loop))

    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, signal_handler)

    await conversation.start()
    print("Conversation started, press Ctrl+C to end")

    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        result = conversation.receive_audio(chunk)
        if asyncio.iscoroutine(result):
            await result
        elif result is not None:
            print(f"Unexpected result from receive_audio: {result}")


if __name__ == "__main__":
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
