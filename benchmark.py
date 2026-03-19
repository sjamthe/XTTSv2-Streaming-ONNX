import time
import numpy as np
from xtts_streaming_pipeline import StreamingTTSPipeline

# 1. Initialize Pipeline (Using your exact server settings)
#print("Loading pipeline on CPU...")
pipeline = StreamingTTSPipeline(
    model_dir="xtts_onnx/",
    vocab_path="xtts_onnx/vocab.json",
    mel_norms_path="xtts_onnx/mel_stats.npy",
    use_int8_gpt=False,       
    num_threads_gpt=4, # Orin Nano has 6 cores, 4 is a safe allocation
)

# 2. Get speaker conditioning
# Using your specific voice file from your code
voice_file = "audio_ref/me.wav" 
#print(f"Loading voice conditioning from {voice_file}...")
gpt_cond_latent, speaker_embedding = pipeline.get_conditioning_latents(voice_file)

test_text = "This is a real-time streaming test on the Jetson Orin Nano CPU. Let's see exactly how fast it generates the very first word."

print("\nStarting benchmark...")
#print(f"Text: '{test_text}'\n")

start_time = time.time()
first_chunk_time = None
all_chunks = []

# 3. Run Inference
generator = pipeline.inference_stream(
    text=test_text,
    language="en",
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    stream_chunk_size=20,
    speed=1.0,
)

# 4. Measure the stream
for i, audio_chunk in enumerate(generator):
    if i == 0:
        # Capture the time it took to get the very first piece of audio
        first_chunk_time = time.time() - start_time
        print(f"-> FIRST AUDIO CHUNK READY in {first_chunk_time:.3f} seconds!")
    
    all_chunks.append(audio_chunk)

end_time = time.time()
total_time = end_time - start_time

# 5. Calculate Metrics
full_audio = np.concatenate(all_chunks, axis=0)
sample_rate = 24000 # Standard for XTTSv2
audio_duration = len(full_audio) / sample_rate
rtf = total_time / audio_duration

print("\n" + "="*30)
print(" BENCHMARK RESULTS (CPU) ")
print("="*30)
print(f"Time to First Byte:      {first_chunk_time:.3f} seconds")
print(f"Total Processing Time:   {total_time:.3f} seconds")
print(f"Generated Audio Length:  {audio_duration:.3f} seconds")
print(f"Real-Time Factor (RTF):  {rtf:.3f}")
print("="*30)

if rtf < 1.0 and first_chunk_time < 1.0:
    print("\nSTATUS: SUCCESS! Your CPU is fast enough for real-time streaming.")
else:
    print("\nSTATUS: WARNING! CPU is struggling. You should enable the GPU.")
