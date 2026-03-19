import numpy as np
from xtts_streaming_pipeline import StreamingTTSPipeline

# Initialise the pipeline
pipeline = StreamingTTSPipeline(
    model_dir="xtts_onnx/",
    vocab_path="xtts_onnx/vocab.json",
    mel_norms_path="xtts_onnx/mel_stats.npy",
    use_int8_gpt=True,       # Use INT8-quantised GPT for faster CPU inference
    num_threads_gpt=4,        # Adjust to your CPU core count
)

# Compute speaker conditioning (one-time per speaker)
gpt_cond_latent, speaker_embedding = pipeline.get_conditioning_latents(
    "audio_ref/male_stewie.mp3"
)

# Stream synthesis
all_chunks = []
for audio_chunk in pipeline.inference_stream(
    text="Hello, this is a streaming text-to-speech demo.",
    language="en",
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    stream_chunk_size=20,     # AR tokens per vocoder call
    speed=1.0,                # 1.0 = normal speed
):
    all_chunks.append(audio_chunk)
    # In a real application, you would play or stream each chunk here.

# Concatenate all chunks into a single waveform
full_audio = np.concatenate(all_chunks, axis=0)

# Save to file
import soundfile as sf
sf.write("output.wav", full_audio, 24000)
