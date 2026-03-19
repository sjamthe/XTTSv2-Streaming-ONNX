---
language:
  - en 
  - es
  - fr
  - de
  - it
  - pt
  - pl
  - tr
  - ru
  - nl
  - cs
  - ar
  - zh 
  - ja
  - hu
  - ko
  - hi
tags:
  - tts
  - onnx
  - xtts
  - xttsv2
  - voice clone
  - streaming
  - gpt2
  - hifigan
  - multilingual
  - vq
  - perceiver encoder
license: apache-2.0
base_model: coqui/XTTS-v2
---

# XTTSv2 Streaming ONNX

**Streaming text-to-speech inference for [XTTSv2](https://arxiv.org/abs/2406.04904) using ONNX Runtime — no PyTorch required.**

This repository provides a complete, CPU-friendly, streaming TTS pipeline built on ONNX-exported XTTSv2 models. It replaces the original PyTorch inference path with pure Python/NumPy logic while preserving full compatibility with the XTTSv2 architecture.

---

## Features

- **Zero-shot voice cloning** from a short (≤ 6 s) reference audio clip.
- **Streaming audio output** — audio chunks are yielded as they are generated, enabling low-latency playback.
- **Pure ONNX Runtime + NumPy** — no PyTorch dependency at inference time.
- **INT8-quantised GPT model** option for reduced memory footprint and faster CPU inference.
- **Cross-fade chunk stitching** for seamless audio across vocoder boundaries.
- **Speed control** via linear interpolation of GPT latents.
- **Multilingual support** — 17 languages: English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko), Hindi (hi).

---

## Architecture Overview

XTTSv2 is composed of four main neural network components, each exported as a separate ONNX model:

| Component | ONNX File | Description |
|---|---|---|
| **Conditioning Encoder** | `conditioning_encoder.onnx` | Six 16-head attention layers + Perceiver Resampler. Compresses a reference mel-spectrogram into 32 × 1024 conditioning latents. |
| **Speaker Encoder** | `speaker_encoder.onnx` | H/ASP speaker verification network. Extracts a 512-dim speaker embedding from 16 kHz audio. |
| **GPT-2 Decoder** | `gpt_model.onnx` / `gpt_model_int8.onnx` | 30-layer, 1024-dim decoder-only transformer with KV-cache. Autoregressively predicts VQ-VAE audio codes conditioned on text tokens and conditioning latents. |
| **HiFi-GAN Vocoder** | `hifigan_vocoder.onnx` | 26M-parameter neural vocoder. Converts GPT-2 hidden states + speaker embedding into a 24 kHz waveform. |

Pre-exported embedding tables (text, mel, positional) are stored as `.npy` files in the `embeddings/` directory.

```
┌─────────────┐   mel @ 22 kHz    ┌─────────────────────┐
│  Reference   │ ───────────────► │ Conditioning Encoder │──► cond_latents [1,32,1024]
│  Audio Clip  │                  └─────────────────────┘
│              │   audio @ 16 kHz  ┌─────────────────────┐
│              │ ───────────────► │   Speaker Encoder    │──► speaker_emb  [1,512,1]
└─────────────┘                   └─────────────────────┘

┌──────────┐   BPE tokens   ┌──────────────────────────────────────────┐
│   Text   │ ─────────────► │  GPT-2 Decoder (autoregressive + KV$)   │──► latents [1,T,1024]
└──────────┘                 │  prefix = [cond | text+pos | start_mel] │
                             └──────────────────────────────────────────┘
                                           │
                                           ▼
                             ┌─────────────────────┐
                             │   HiFi-GAN Vocoder   │──► waveform @ 24 kHz
                             │  (+ speaker_emb)     │
                             └─────────────────────┘
```

---

## Repository Structure

```
.
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── xtts_streaming_pipeline.py        # Top-level streaming TTS pipeline
├── xtts_onnx_orchestrator.py         # Low-level ONNX AR loop orchestrator
├── xtts_tokenizer.py                 # BPE tokenizer wrapper
├── zh_num2words.py                   # Chinese number-to-words utility
├── xtts_onnx/                        # ONNX models & assets
│   ├── metadata.json                 # Model architecture metadata
│   ├── vocab.json                    # BPE vocabulary
│   ├── mel_stats.npy                 # Per-channel mel normalisation stats
│   ├── conditioning_encoder.onnx     # Conditioning encoder
│   ├── speaker_encoder.onnx          # H/ASP speaker encoder
│   ├── gpt_model.onnx               # GPT-2 decoder (FP32)
│   ├── gpt_model_int8.onnx          # GPT-2 decoder (INT8 quantised)
│   ├── hifigan_vocoder.onnx          # HiFi-GAN vocoder
│   └── embeddings/                   # Pre-exported embedding tables
│       ├── mel_embedding.npy         # [1026, 1024] audio code embeddings
│       ├── mel_pos_embedding.npy     # [608, 1024]  mel positional embeddings
│       ├── text_embedding.npy        # [6681, 1024] BPE text embeddings
│       └── text_pos_embedding.npy    # [404, 1024]  text positional embeddings
├── audio_ref/                        # Reference audio clips for voice cloning
└── audio_synth/                      # Directory for synthesised output
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- A C compiler may be needed for some dependencies (e.g. `tokenizers`).

### Install dependencies

```bash
pip install -r requirements.txt
```

### Clone from Hugging Face Hub

```bash
# Install Git LFS (required for large model files)
git lfs install

# Clone the repository
git clone https://huggingface.co/pltobing/XTTSv2-Streaming-ONNX
cd XTTSv2-Streaming-ONNX
```

---

## Quick Start

### Streaming TTS (command-line)

```bash
python -u xtts_streaming_pipeline.py \
    --model_dir xtts_onnx/ \
    --vocab_path xtts_onnx/vocab.json \
    --mel_norms_path xtts_onnx/mel_stats.npy \
    --ref_audio audio_ref/male_stewie.mp3 \
    --language en \
    --output output_streaming.wav
```

### Python API

```python
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
```

---

## Configuration

### SamplingConfig

Control the autoregressive token sampling behaviour:

| Parameter | Default | Description |
|---|---|---|
| `temperature` | `0.75` | Softmax temperature. Lower = more deterministic. |
| `top_k` | `50` | Keep only the top-*k* most probable tokens. |
| `top_p` | `0.85` | Nucleus sampling cumulative probability threshold. |
| `repetition_penalty` | `10.0` | Penalise previously generated tokens. |
| `do_sample` | `True` | `True` = multinomial sampling; `False` = greedy argmax. |

```python
from xtts_onnx_orchestrator import SamplingConfig

sampling = SamplingConfig(
    temperature=0.65,
    top_k=30,
    top_p=0.90,
    repetition_penalty=10.0,
    do_sample=True,
)

for chunk in pipeline.inference_stream(text, "en", cond, spk, sampling=sampling):
    ...
```

### GPTConfig

Model architecture parameters are automatically loaded from `metadata.json`. Key fields:

| Parameter | Value | Description |
|---|---|---|
| `n_layer` | 30 | Number of GPT-2 transformer layers |
| `embed_dim` | 1024 | Hidden dimension |
| `num_heads` | 16 | Number of attention heads |
| `head_dim` | 64 | Per-head dimension |
| `num_audio_tokens` | 1026 | Audio vocabulary (1024 VQ codes + start + stop) |
| `perceiver_output_len` | 32 | Conditioning latent sequence length |
| `max_gen_mel_tokens` | 605 | Maximum generated audio tokens |

---

## Module Reference

### `xtts_streaming_pipeline.py`

Top-level streaming pipeline.

| Class / Function | Description |
|---|---|
| `StreamingTTSPipeline` | Main pipeline class. Owns sessions, tokenizer, orchestrator. |
| `StreamingTTSPipeline.get_conditioning_latents()` | Extract GPT conditioning + speaker embedding from reference audio. |
| `StreamingTTSPipeline.inference_stream()` | Generator that yields audio chunks for a text segment. |
| `StreamingTTSPipeline.time_scale_gpt_latents_numpy()` | Linearly time-scale GPT latents for speed control. |
| `wav_to_mel_cloning_numpy()` | Compute normalised log-mel spectrogram (NumPy, 22 kHz). |
| `crossfade_chunks()` | Cross-fade consecutive vocoder waveform chunks. |

### `xtts_onnx_orchestrator.py`

Low-level ONNX autoregressive loop.

| Class / Function | Description |
|---|---|
| `ONNXSessionManager` | Loads and manages all ONNX sessions + embedding tables. |
| `XTTSOrchestratorONNX` | Drives the GPT-2 AR loop with KV-cache and logits processing. |
| `GPTConfig` | Model architecture hyper-parameters (from `metadata.json`). |
| `SamplingConfig` | Token sampling hyper-parameters. |
| `apply_repetition_penalty()` | NumPy repetition penalty on logits. |
| `apply_temperature()` | Temperature scaling on logits. |
| `apply_top_k()` | Top-*k* filtering on logits. |
| `apply_top_p()` | Nucleus (top-*p*) filtering on logits. |
| `numpy_softmax()` | Numerically-stable softmax in NumPy. |
| `numpy_multinomial()` | Inverse-CDF multinomial sampling. |

---

## Performance Notes

- **`stream_chunk_size`** controls the latency–quality trade-off: smaller values yield audio sooner but run the vocoder more often (on all accumulated latents).
- **Thread count** (`num_threads_gpt`) should be tuned to your CPU.  Start with the number of physical cores.
- First call to `get_conditioning_latents()` is an expensive step (resampling + mel computation + encoder inference).  Cache the results for repeated synthesis with the same speaker.

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

```
Copyright 2025 Patrick Lumbantobing, Vertox-AI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

## Acknowledgements

- [Coqui AI](https://github.com/coqui-ai/TTS) for the original XTTSv2 model and training recipe.
- [XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model](https://arxiv.org/abs/2406.04904) (Casanova et al., 2024).
- [ONNX Runtime](https://onnxruntime.ai/) for high-performance cross-platform inference.
