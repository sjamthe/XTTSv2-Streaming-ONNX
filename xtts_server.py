##
# Server mimics OpenAI TTS API to connect to openAI clients
##

import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from xtts_streaming_pipeline import StreamingTTSPipeline
import io
import soundfile as sf

app = FastAPI()
pipeline = StreamingTTSPipeline(
    model_dir="xtts_onnx/",
    vocab_path="xtts_onnx/vocab.json",
    mel_norms_path="xtts_onnx/mel_stats.npy",
    use_int8_gpt=True,       # Use INT8-quantised GPT for faster CPU inference
    num_threads_gpt=4,        # Adjust to your CPU core count
)

class TTSRequest(BaseModel):
    model: str = "xtts"
    input: str
    voice: str = "default"

@app.post("/v1/audio/speech")
async def synthesize(req: TTSRequest):

    # 1. Voice selection logic
    if(req.voice == "stewie"):
        voice = "audio_ref/male_stewie.mp3"
    elif(req.voice == "attenborough"):
        voice = "audio_ref/david-attenborough.mp3"
    elif(req.voice == "shadowheart"):
        voice = "audio_ref/female_shadowheart.flac"
    elif(req.voice == "petergriffin"):
        voice = "audio_ref/male_petergriffin.wav"
    elif(req.voice == "shirish"):
        voice = "audio_ref/me.wav"
    else:
        voice = "audio_ref/male_old_movie.flac"

    # 2. Compute speaker conditioning
    gpt_cond_latent, speaker_embedding = pipeline.get_conditioning_latents(voice)

    # 3. Create the Async Generator for true streaming
    async def audio_stream_generator():
        for audio_chunk in pipeline.inference_stream(
            text=req.input,
            language="en",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            stream_chunk_size=20, # 20 tokens per chunk keeps it responsive
            speed=1.0,
        ):
            # Yield the raw bytes the millisecond the GPU finishes the chunk
            yield audio_chunk.tobytes()

    # 4. Return the stream directly to the client
    return StreamingResponse(audio_stream_generator(), media_type="audio/raw")
