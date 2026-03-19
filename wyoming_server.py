import asyncio
import logging
import numpy as np
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize
# --- NEW IMPORTS FOR THE HANDSHAKE ---
from wyoming.info import Describe, Info, TtsProgram, TtsVoice, Attribution

from xtts_streaming_pipeline import StreamingTTSPipeline

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)

# --- CONFIGURATION ---
PORT = 10200

print("Loading XTTSv2 Pipeline on GPU...")
pipeline = StreamingTTSPipeline(
    model_dir="xtts_onnx/",
    vocab_path="xtts_onnx/vocab.json",
    mel_norms_path="xtts_onnx/mel_stats.npy",
    use_int8_gpt=False,       
)

# --- VOICE CONFIGURATION ---
VOICES = {
    "shirish": "audio_ref/me.wav",
    "amitabh": "audio_ref/amitabh.wav",
    "pi": "audio_ref/pi.wav",
    "female": "audio_ref/female_shadowheart.flac"
}

# Pre-compute and store latents for all voices on startup to eliminate latency
print("Pre-loading voice profiles...")
loaded_voices = {}
voices = []

for voice_name, file_path in VOICES.items():
    print(f"Loading {voice_name} from {file_path}...")
    # XTTS extracts the conditioning latents and speaker embeddings
    gpt_cond, speaker_emb = pipeline.get_conditioning_latents(file_path)
    loaded_voices[voice_name] = (gpt_cond, speaker_emb)
    ttsVoice = TtsVoice(name=voice_name, description=voice_name, attribution=Attribution(name="Shirish", url=""), installed=True, languages=["en"], version="1.0"),
    voices.append(ttsVoice)

print("All voices loaded and ready!")

class XTTSEventHandler(AsyncEventHandler):
    async def handle_event(self, event: Event) -> bool:
        
        # --- THE HANDSHAKE ---
        if Describe.is_type(event.type):
            _LOGGER.info("HA sent Describe. Sending Info...")
            await self.write_event(Info(
                tts=[TtsProgram(
                    name="xtts_orin",
                    description="XTTSv2 GPU Streamer",
                    attribution=Attribution(name="Shirish", url="https://github.com/coqui-ai/TTS"),
                    installed=True,
                    version="1.0",
                    voices=voices
                )]
            ).event())
            _LOGGER.info("Info sent!")
            return True

        # --- THE SYNTHESIS ---
        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            text = synthesize.text
            
            # 1. Voice Selection Logic
            voice_name = "shirish" # Default voice
            if synthesize.voice and synthesize.voice.name in loaded_voices:
                voice_name = synthesize.voice.name
                
            _LOGGER.info(f"Synthesizing: '{text}' using voice: '{voice_name}'")

            # Fetch the pre-loaded embeddings for the requested voice
            gpt_cond_latent, speaker_embedding = loaded_voices[voice_name]

            await self.write_event(AudioStart(rate=24000, width=2, channels=1).event())

            # 2. Pass the selected embeddings into the pipeline
            for chunk in pipeline.inference_stream(
                text=text,
                language="en",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                stream_chunk_size=20,
                speed=1.0,
            ):
                await asyncio.sleep(0.001) 
                chunk_int16 = (chunk * 32767).astype(np.int16).tobytes()
                
                await self.write_event(
                    AudioChunk(
                        audio=chunk_int16,
                        rate=24000,
                        width=2,
                        channels=1,
                    ).event()
                )

            await self.write_event(AudioStop().event())
            _LOGGER.info("Finished speaking.")
            return True

        return True

async def main():
    server = AsyncServer.from_uri(f"tcp://0.0.0.0:{PORT}")
    _LOGGER.info(f"Wyoming XTTS Server running on port {PORT}")
    await server.run(XTTSEventHandler)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
