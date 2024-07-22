import asyncio
import signal
import ssl

from pydantic_settings import BaseSettings, SettingsConfigDict

import os
import sys

# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))


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
from vocode.streaming.transcriber.deepgram_transcriber_1 import DeepgramTranscriber

configure_pretty_logging()

os.environ["GROQ_API_KEY"] = "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"

class Settings(BaseSettings):
    """
    Settings for the streaming conversation quickstart.
    These parameters can be configured with environment variables.
    """

    openai_api_key: str = "ENTER_YOUR_OPENAI_API_KEY_HERE"
    azure_speech_key: str = "ENTER_YOUR_AZURE_KEY_HERE"
    deepgram_api_key: str = "cd3898ec57d1581c9881355c2874f633436658c8"
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
        synthesizer=ElevenLabsSynthesizer(
            ElevenLabsSynthesizerConfig.from_output_device(speaker_output, api_key="sk_40b3c6f7619c8866657ff13b91579ad72ebfbfd8941393bb")
        ),
    )
    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    signal.signal(signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate()))
    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)


if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    asyncio.run(main())
