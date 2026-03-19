# Copyright 2025 Patrick Lumbantobing, Vertox-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
xtts_onnx_orchestrator.py
--------------------------
External autoregressive loop orchestrator for ONNX-exported XTTSv2.
Replaces stream_generator.py's sample_stream() with pure Python/NumPy logic.

This module manages:
    - Loading and wrapping all ONNX Runtime inference sessions (GPT-2 decoder,
      conditioning encoder, speaker encoder, HiFi-GAN vocoder).
    - Embedding look-ups (text, mel, positional) that were originally PyTorch
      ``nn.Embedding`` layers and are now stored as ``.npy`` files.
    - KV-cache creation and propagation across autoregressive decode steps.
    - Logits processing (repetition penalty, temperature scaling, top-k / top-p
      filtering) and multinomial sampling -- all in NumPy.

Architecture Reference (XTTSv2, Coqui AI):
    1. **Conditioning Encoder** -- six 16-head attention layers + Perceiver
       Resampler compressing a reference mel-spectrogram into 32 latent vectors
       of dimension 1024.
    2. **GPT-2 Decoder** -- a 30-layer, 1024-dim decoder-only transformer that
       autoregressively predicts VQ-VAE audio codes conditioned on text tokens
       and conditioning latents.
    3. **HiFi-GAN Vocoder** -- a 26M-parameter neural vocoder conditioned on
       GPT-2 hidden states and a 512-dim speaker embedding from an H/ASP
       speaker encoder.

See Also:
    - XTTS paper: https://arxiv.org/abs/2406.04904
    - Coqui TTS: https://github.com/coqui-ai/TTS
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Generator, Literal, Tuple

import numpy as np
import numpy.typing as npt
import onnxruntime as ort


# ─────────────────────────────────────────────────────────────────────────────
# NumPy implementations of logits processing (replaces HF LogitsProcessorList)
# ─────────────────────────────────────────────────────────────────────────────


def apply_repetition_penalty(
    scores: np.ndarray, input_ids: np.ndarray, penalty: float
) -> np.ndarray:
    """Apply repetition penalty to logits for tokens already generated.

    For each unique token previously generated:
        - If its logit is **negative**, multiply by ``penalty`` (more negative \u2192
          less likely).
        - If its logit is **positive**, divide by ``penalty`` (reduce probability
          mass).

    Follows the formulation in Keskar et al., *CTRL* (2019).

    Parameters
    ----------
    scores : np.ndarray
        Logits array of shape ``(1, vocab_size)``.
    input_ids : np.ndarray
        Token IDs generated so far, shape ``(1, seq_len)``.
    penalty : float
        Repetition penalty factor.  ``1.0`` disables the penalty.

    Returns
    -------
    np.ndarray
        Modified logits array (same object, mutated in-place).
    """
    if penalty == 1.0:
        return scores
    unique_ids = np.unique(input_ids)
    for uid in unique_ids:
        if scores[0, uid] < 0:
            scores[0, uid] *= penalty
        else:
            scores[0, uid] /= penalty
    return scores


def apply_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by ``1 / temperature``.

    Higher temperature produces a flatter distribution (more randomness);
    lower temperature produces a sharper distribution (more deterministic).

    Parameters
    ----------
    scores : np.ndarray
        Logits array of shape ``(batch, vocab_size)``.
    temperature : float
        Sampling temperature.  Must be > 0.  ``1.0`` is a no-op.

    Returns
    -------
    np.ndarray
        Scaled logits.
    """
    if temperature == 1.0:
        return scores
    return scores / temperature


def apply_top_k(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Zero-out (set to ``-inf``) all logits outside the top-*k* values.

    Parameters
    ----------
    scores : np.ndarray
        Logits of shape ``(batch, vocab_size)``.
    top_k : int
        Number of highest-scoring tokens to keep.  ``<= 0`` disables.

    Returns
    -------
    np.ndarray
        Filtered logits with all but the top-*k* set to ``-inf``.
    """
    if top_k <= 0:
        return scores
    top_k = min(top_k, scores.shape[-1])
    indices_to_remove = scores < np.sort(scores, axis=-1)[..., -top_k:][..., :1]
    scores[indices_to_remove] = -float("inf")
    return scores


def apply_top_p(scores: np.ndarray, top_p: float) -> np.ndarray:
    """Nucleus (top-*p*) filtering: keep the smallest set of tokens whose
    cumulative probability mass exceeds ``top_p``.

    Implements the algorithm from Holtzman et al., *The Curious Case of
    Neural Text Degeneration* (2020).

    Parameters
    ----------
    scores : np.ndarray
        Logits of shape ``(batch, vocab_size)``.
    top_p : float
        Cumulative probability threshold in ``(0, 1]``.  ``>= 1.0`` disables.

    Returns
    -------
    np.ndarray
        Filtered logits.
    """
    if top_p >= 1.0:
        return scores
    sorted_indices = np.argsort(-scores, axis=-1)
    sorted_scores = np.take_along_axis(scores, sorted_indices, axis=-1)

    # Softmax for cumulative probs
    exp_scores = np.exp(sorted_scores - sorted_scores.max(axis=-1, keepdims=True))
    probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    cumulative_probs = np.cumsum(probs, axis=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs - probs > top_p
    for b in range(scores.shape[0]):
        remove_indices = sorted_indices[b, sorted_indices_to_remove[b]]
        scores[b, remove_indices] = -float("inf")
    return scores


def numpy_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax implemented in pure NumPy.

    Parameters
    ----------
    x : np.ndarray
        Input array (any shape).
    axis : int, optional
        Axis along which softmax is computed.  Default ``-1``.

    Returns
    -------
    np.ndarray
        Probability distribution summing to 1 along ``axis``.
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def numpy_multinomial(probs: np.ndarray) -> int:
    """Sample one token index from a categorical probability distribution.

    Uses inverse-CDF sampling: draw ``u ~ Uniform(0, 1)`` and find the first
    index where the cumulative distribution function exceeds ``u``.

    Parameters
    ----------
    probs : np.ndarray
        Probability vector of shape ``(1, vocab_size)``.

    Returns
    -------
    int
        Sampled token index.
    """
    cumulative = np.cumsum(probs[0])
    r = np.random.random()
    return int(np.searchsorted(cumulative, r))


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GPTConfig:
    """Hyper-parameters for the GPT-2 autoregressive decoder in XTTSv2.

    These values must match the architecture used during ONNX export.  They
    can be loaded from the ``metadata.json`` shipped alongside the ONNX model
    files via :meth:`from_metadata`.

    Attributes
    ----------
    n_layer : int
        Number of transformer decoder layers (default 30).
    embed_dim : int
        Hidden / embedding dimension (default 1024).
    num_heads : int
        Number of attention heads per layer (default 16).
    head_dim : int
        Dimension of each attention head (default 64).  Satisfies
        ``num_heads * head_dim == embed_dim``.
    num_audio_tokens : int
        Size of the audio token vocabulary including start/stop sentinels
        (default 1026 = 1024 VQ codes + start + stop).
    start_audio_token : int
        Sentinel token ID marking the beginning of the audio code sequence
        (default 1024).
    stop_audio_token : int
        Sentinel token ID marking the end of the audio code sequence
        (default 1025).
    start_text_token : int
        BPE token ID for the ``[START]`` text sentinel.  Overridden at
        runtime from the tokenizer.
    stop_text_token : int
        BPE token ID for the ``[STOP]`` text sentinel.  Overridden at
        runtime from the tokenizer.
    perceiver_output_len : int
        Fixed number of conditioning latent vectors produced by the
        Perceiver Resampler (default 32).
    max_gen_mel_tokens : int
        Hard upper bound on the number of audio tokens the AR loop will
        generate before forcibly stopping (default 605).
    """

    n_layer: int = 30
    embed_dim: int = 1024
    num_heads: int = 16
    head_dim: int = 64
    num_audio_tokens: int = 1026
    start_audio_token: int = 1024
    stop_audio_token: int = 1025
    start_text_token: int = 261  # Will be set from tokenizer
    stop_text_token: int = 0  # Will be set from tokenizer
    perceiver_output_len: int = 32
    max_gen_mel_tokens: int = 605

    @classmethod
    def from_metadata(cls, metadata_path: str) -> "GPTConfig":
        """Construct a :class:`GPTConfig` from an exported ``metadata.json``.

        Parameters
        ----------
        metadata_path : str
            Path to the JSON metadata file shipped with the ONNX model
            directory.

        Returns
        -------
        GPTConfig
            Populated configuration instance.
        """
        with open(metadata_path) as f:
            meta = json.load(f)
        return cls(
            n_layer=meta["n_layer"],
            embed_dim=meta["embed_dim"],
            num_heads=meta["num_heads"],
            head_dim=meta["head_dim"],
            num_audio_tokens=meta["num_audio_tokens"],
            start_audio_token=meta["start_audio_token"],
            stop_audio_token=meta["stop_audio_token"],
            perceiver_output_len=meta["perceiver_output_len"],
        )


@dataclass
class SamplingConfig:
    """Hyper-parameters that control autoregressive token sampling.

    Attributes
    ----------
    temperature : float
        Softmax temperature.  Lower values are more deterministic.
    top_k : int
        Number of highest-probability tokens to retain before sampling.
    top_p : float
        Nucleus sampling cumulative probability cutoff.
    repetition_penalty : float
        Multiplicative penalty applied to already-generated tokens.
    do_sample : bool
        If ``True``, sample from the filtered distribution; otherwise use
        ``argmax`` (greedy decoding).
    """

    temperature: float = 0.75
    top_k: int = 50
    top_p: float = 0.85
    repetition_penalty: float = 10.0
    do_sample: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Session Manager
# ─────────────────────────────────────────────────────────────────────────────


class ONNXSessionManager:
    """Manages all ONNX Runtime sessions for XTTSv2.

    On construction the manager:
        1. Reads ``metadata.json`` from ``model_dir`` to resolve file paths.
        2. Creates ``ort.InferenceSession`` objects for the conditioning
           encoder, speaker encoder, GPT-2 decoder, and HiFi-GAN vocoder.
        3. Loads pre-exported embedding tables (text, mel, and their
           positional counterparts) from the ``embeddings/`` sub-directory.

    Parameters
    ----------
    model_dir : str
        Root directory containing the ONNX models, ``metadata.json``,
        ``embeddings/`` folder, and ``vocab.json``.
    use_int8_gpt : bool, optional
        If ``True`` (default), prefer the INT8-quantised GPT model for lower
        memory usage and faster CPU inference.
    num_threads : int, optional
        Number of intra-op threads for ONNX Runtime (default 1).

    Attributes
    ----------
    cond_encoder : ort.InferenceSession
        Conditioning encoder session (mel -> 32 x 1024 latents).
    speaker_encoder : ort.InferenceSession
        H/ASP speaker encoder session (16 kHz mel -> 512-dim embedding).
    gpt : ort.InferenceSession
        GPT-2 decoder session with KV-cache I/O.
    hifigan : ort.InferenceSession
        HiFi-GAN vocoder session (latents + speaker emb -> waveform).
    mel_embedding : np.ndarray
        Audio code embedding table, shape ``(1026, 1024)``.
    text_embedding : np.ndarray
        BPE text embedding table, shape ``(6681, 1024)``.
    mel_pos_embedding : np.ndarray
        Mel positional embedding table, shape ``(608, 1024)``.
    text_pos_embedding : np.ndarray
        Text positional embedding table, shape ``(404, 1024)``.
    metadata : dict
        Parsed contents of ``metadata.json``.
    """

    def __init__(self, model_dir: str, use_int8_gpt: bool = True, num_threads: int = 1):
        self.model_dir = model_dir
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        #providers = ["CPUExecutionProvider"]
        gpu_memory_limit_bytes = 2 * 1024 * 1024 * 1024 # limit model to 2GB on orin nano
        # Configure the CUDA provider explicitly
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }
            ),
            "CPUExecutionProvider"
        ]

        with open(os.path.join(model_dir, "metadata.json")) as f:
            self.metadata = json.load(f)

        # Resolve GPT model file -- prefer INT8-quantised variant when available
        gpt_file = (
            self.metadata["files"].get("gpt_model_int8", self.metadata["files"]["gpt_model"])
            if use_int8_gpt
            else self.metadata["files"]["gpt_model"]
        )

        logging.info(f"Loading ONNX sessions from {model_dir}...")
        self.cond_encoder = ort.InferenceSession(
            os.path.join(model_dir, self.metadata["files"]["conditioning_encoder"]),
            sess_options=opts,
            providers=providers,
        )
        self.speaker_encoder = ort.InferenceSession(
            os.path.join(model_dir, self.metadata["files"]["speaker_encoder"]), sess_options=opts, providers=providers
        )
        self.gpt = ort.InferenceSession(os.path.join(model_dir, gpt_file), sess_options=opts, providers=providers)
        self.hifigan = ort.InferenceSession(
            os.path.join(model_dir, self.metadata["files"]["hifigan_vocoder"]), sess_options=opts, providers=providers
        )

        # Load pre-exported embedding tables from the embeddings/ sub-directory
        emb_dir = os.path.join(model_dir, "embeddings")
        self.mel_embedding = np.load(os.path.join(emb_dir, "mel_embedding.npy"))  # [1026, 1024]
        self.text_embedding = np.load(os.path.join(emb_dir, "text_embedding.npy"))  # [6681, 1024]
        self.mel_pos_embedding = np.load(os.path.join(emb_dir, "mel_pos_embedding.npy"))  # [608, 1024]
        self.text_pos_embedding = np.load(os.path.join(emb_dir, "text_pos_embedding.npy"))  # [404, 1024]

        logging.info("[OK] All ONNX sessions loaded.")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator: External Autoregressive Loop
# ─────────────────────────────────────────────────────────────────────────────


class XTTSOrchestratorONNX:
    """Replaces ``stream_generator.sample_stream()`` with an external loop.

    Manages embedding lookups, KV-cache propagation, logits processing, and
    multinomial sampling -- all with ONNX Runtime and NumPy.

    The orchestrator is responsible for:
        - Computing conditioning latents from a reference mel-spectrogram via
          the conditioning encoder.
        - Computing 512-dim speaker embeddings from 16 kHz reference audio via
          the H/ASP speaker encoder.
        - Building the prefix embedding sequence:
          ``[cond_latents | text_emb + text_pos | start_audio_emb + mel_pos[0]]``
        - Running the KV-cached GPT-2 prefill + autoregressive decode loop.
        - Applying logits processing and sampling the next audio token.
        - Running the HiFi-GAN vocoder on accumulated GPT hidden states.

    Parameters
    ----------
    sessions : ONNXSessionManager
        Pre-initialised ONNX session manager.
    config : GPTConfig
        Model hyper-parameters (layers, dims, sentinel tokens, etc.).
    """

    def __init__(self, sessions: ONNXSessionManager, config: GPTConfig):
        self.sess = sessions
        self.cfg = config

    # -----------------------------------------------------------------
    # Mel-scale helper functions (used by the speaker encoder path)
    # -----------------------------------------------------------------

    def _hz_to_mel(
        self, freq: float | npt.NDArray[np.float64], mel_scale: Literal["htk", "kaldi", "slaney"]
    ) -> npt.NDArray[np.float64]:
        """Convert frequency in Hz to mel scale.

        Parameters
        ----------
        freq : float or np.ndarray
            Frequency value(s) in Hz.
        mel_scale : {"htk", "kaldi", "slaney"}
            Which mel-scale formula to use.

        Returns
        -------
        np.ndarray
            Mel-scale value(s).
        """
        if mel_scale == "htk":
            return 2595 * np.log10(1.0 + freq / 700.0)
        if mel_scale == "kaldi":
            return 1127 * np.log(1.0 + freq / 700.0)
        # Slaney: linear below 1 kHz, logarithmic above
        return np.where(
            freq < 1000,
            3 * freq / 200.0,
            15 + 27 * np.log(freq / 1000.0 + np.finfo(np.float32).eps) / np.log(6.4),
        )

    def _mel_to_hz(
        self, mels: npt.NDArray[np.float64], mel_scale: Literal["htk", "slaney"]
    ) -> npt.NDArray[np.float64]:
        """Convert mel-scale values back to Hz.

        Parameters
        ----------
        mels : np.ndarray
            Mel-scale value(s).
        mel_scale : {"htk", "slaney"}
            Which inverse formula to use.

        Returns
        -------
        np.ndarray
            Frequency value(s) in Hz.
        """
        if mel_scale == "htk":
            return 700 * (np.power(10.0, mels / 2595.0) - 1.0)
        # Slaney inverse
        return np.where(
            mels < 15,
            200 * mels / 3.0,
            1000 * np.power(6.4, ((mels - 15) / 27.0)),
        )

    def melscale_fbanks(
        self,
        n_freqs: int,
        f_min: float,
        f_max: float,
        n_mels: int,
        sample_rate: int,
        norm: Literal["slaney"] | None = None,
        mel_scale: Literal["htk", "slaney", "kaldi"] = "htk",
    ) -> npt.NDArray[np.float64]:
        """Build a mel-scale triangular filterbank matrix.

        NumPy equivalent of ``torchaudio.functional.melscale_fbanks``
        supporting HTK, Kaldi, and Slaney mel scales.

        Parameters
        ----------
        n_freqs : int
            Number of linear frequency bins (typically ``n_fft // 2 + 1``).
        f_min : float
            Minimum frequency (Hz).
        f_max : float
            Maximum frequency (Hz).  If ``<= 0``, set to ``sample_rate / 2``.
        n_mels : int
            Number of mel filterbank channels.
        sample_rate : int
            Audio sample rate in Hz.
        norm : {"slaney", None}, optional
            If ``"slaney"``, apply area normalisation (unit area per filter).
        mel_scale : {"htk", "slaney", "kaldi"}, optional
            Mel-scale variant to use (default ``"htk"``).

        Returns
        -------
        np.ndarray
            Filterbank matrix of shape ``(n_freqs, n_mels)``.
        """
        if f_max <= 0.0:
            f_max += sample_rate / 2

        all_freqs = np.linspace(0.0, sample_rate // 2, n_freqs)

        m_min = self._hz_to_mel(f_min, mel_scale=mel_scale)
        m_max = self._hz_to_mel(f_max, mel_scale=mel_scale)

        m_pts = np.linspace(m_min, m_max, n_mels + 2)

        if mel_scale == "kaldi":
            mel = self._hz_to_mel(all_freqs, mel_scale=mel_scale)
        else:
            mel = all_freqs
            m_pts = self._mel_to_hz(m_pts, mel_scale=mel_scale)

        # Triangular filters via rising / falling slopes
        up_slopes = (mel[:, None] - m_pts[:-2]) / (m_pts[1:-1] - m_pts[:-2])
        down_slopes = (m_pts[2:] - mel[:, None]) / (m_pts[2:] - m_pts[1:-1])
        fb = np.maximum(0.0, np.minimum(up_slopes, down_slopes))

        if norm == "slaney":
            # Area normalisation: scale each filter by 2 / bandwidth
            fb *= 2.0 / (m_pts[2:] - m_pts[:-2])

        return fb

    # -----------------------------------------------------------------
    # Mel spectrogram for the speaker encoder (16 kHz / Hamming / 64-mel)
    # -----------------------------------------------------------------

    def compute_mel_spectrogram_speaker_encoder(
        self,
        waveforms: npt.NDArray[np.float32],  # shape: (batch, time)
        fft_size: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        sample_rate: int = 16000,
        preemphasis: float = 0.97,
        num_mels: int = 64,
    ) -> npt.NDArray[np.float32]:
        """Compute a mel spectrogram for the H/ASP speaker encoder.

        Pure-NumPy re-implementation of the PyTorch chain::

            PreEmphasis(0.97)
            -> torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=fft_size, win_length=win_length,
                hop_length=hop_length, window_fn=torch.hamming_window,
                n_mels=num_mels)

        Parameters
        ----------
        waveforms : np.ndarray
            Mono audio waveform(s) of shape ``(batch, time)`` at 16 kHz.
        fft_size : int
            FFT window size (default 512).
        win_length : int
            Analysis window length in samples (default 400 = 25 ms @ 16 kHz).
        hop_length : int
            Hop size in samples (default 160 = 10 ms @ 16 kHz).
        sample_rate : int
            Expected sample rate (default 16000).
        preemphasis : float
            Pre-emphasis coefficient (default 0.97).  Set to 0.0 to skip.
        num_mels : int
            Number of mel channels (default 64).

        Returns
        -------
        np.ndarray
            Mel spectrogram of shape ``(batch, num_mels, num_frames)``.
        """

        # 1) Pre-emphasis with reflect padding (match torch PreEmphasis)
        if preemphasis != 0.0:
            # reflect pad 1 sample on the left along time axis
            first = waveforms[:, 0:1]
            second = waveforms[:, 1:2]
            # reflect pad: [second, first, x2, x3, ...] approximates 1-sample reflect
            pad_left = 2 * first - second
            padded = np.concatenate([pad_left, waveforms], axis=1)
            # y[n] = x[n] - a * x[n-1]
            waveforms = padded[:, 1:] - preemphasis * padded[:, :-1]

        # 2) STFT-like framing with center=True, pad_mode='reflect'
        # reflect pad fft_size//2 on both sides
        pad = fft_size // 2
        if pad > 0:
            left = waveforms[:, 1 : pad + 1][:, ::-1]
            right = waveforms[:, -pad - 1 : -1][:, ::-1]
            waveforms = np.concatenate([left, waveforms, right], axis=1)

        # 3) Create frames via sliding windows
        num_frames = 1 + (waveforms.shape[1] - fft_size) // hop_length
        frame_indices = np.arange(fft_size)[None, None, :] + hop_length * np.arange(num_frames)[None, :, None]
        strided_input = waveforms[:, :, None]
        strided_input = strided_input[
            np.arange(waveforms.shape[0])[:, None, None],
            frame_indices,
            np.zeros_like(frame_indices),
        ]  # shape: (batch, num_frames, fft_size)

        # 4) Hamming window centered in fft_size
        win = np.hamming(win_length).astype(np.float32)
        if fft_size > win_length:
            pad_left = (fft_size - win_length) // 2
            pad_right = fft_size - win_length - pad_left
            win = np.pad(win, (pad_left, pad_right))
        win = win.astype(np.float32)[None, None, :]  # broadcast over batch and frames

        strided_input = strided_input.astype(np.float32) * win

        # 5) Power spectrogram (power=2.0, onesided=True)
        spectrogram = np.abs(np.fft.rfft(strided_input, n=fft_size, axis=-1)) ** 2  # (B, T, n_freqs)

        # 6) Mel filterbank
        melscale_fbanks = self.melscale_fbanks(
            n_freqs=fft_size // 2 + 1,
            f_min=0.0,
            f_max=sample_rate // 2,
            n_mels=num_mels,
            sample_rate=sample_rate,
            norm=None,
            mel_scale="htk",
        ).astype(
            np.float32
        )  # (n_freqs, num_mels)

        mel_spectrogram = np.matmul(spectrogram, melscale_fbanks)  # (B, T, num_mels)

        # 7) Match torchaudio layout: (batch, num_mels, num_frames)
        mel_spectrogram = np.transpose(mel_spectrogram, (0, 2, 1)).astype(np.float32)

        return mel_spectrogram

    # -----------------------------------------------------------------
    # ONNX model runners
    # -----------------------------------------------------------------

    def compute_conditioning(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Run the conditioning encoder.

        The conditioning encoder consists of six 16-head attention layers
        followed by a Perceiver Resampler that always produces exactly
        ``perceiver_output_len`` (32) latent vectors.

        Parameters
        ----------
        mel_spectrogram : np.ndarray
            Reference mel spectrogram, shape ``(B, 80, T)``.

        Returns
        -------
        np.ndarray
            Conditioning latents of shape ``(B, 32, 1024)``.
        """
        result = self.sess.cond_encoder.run(None, {"mel_spectrogram": mel_spectrogram.astype(np.float32)})
        return result[0]  # [B, 32, 1024]

    def compute_speaker_embedding(self, audio_16k: np.ndarray) -> np.ndarray:
        """Run the H/ASP speaker encoder on 16 kHz audio.

        Internally computes a 64-channel mel spectrogram and passes it through
        the speaker verification network to extract a 512-dim speaker embedding.

        Parameters
        ----------
        audio_16k : np.ndarray
            Mono waveform at 16 kHz, shape ``(B, T)``.

        Returns
        -------
        np.ndarray
            Speaker embedding of shape ``(B, 512, 1)``.
        """
        logging.info(f"compute_speaker_embedding audio_16k {audio_16k} {audio_16k.shape}")
        mel_spec = self.compute_mel_spectrogram_speaker_encoder(audio_16k)
        logging.info(f"compute_speaker_embedding mel_spec {mel_spec} {mel_spec.shape}")
        result = self.sess.speaker_encoder.run(None, {"mel_spec": mel_spec})
        return result[0]  # [B, 512, 1]

    # -----------------------------------------------------------------
    # Prefix embedding construction
    # -----------------------------------------------------------------

    def compute_prefix_embedding(
        self,
        cond_latents: np.ndarray,  # [B, 32, 1024] from conditioning encoder
        text_tokens: np.ndarray,  # [T_text] token IDs (without start/stop)
    ) -> np.ndarray:
        """Build the prefix embedding fed to GPT-2 before the AR decode loop.

        The prefix is the concatenation of three segments::

            [cond_latents | text_emb + text_pos | start_audio_emb + mel_pos[0]]

        Mirrors ``gpt.compute_embeddings()`` + ``gpt_inference.forward()``
        prefill path in the original PyTorch implementation.

        Parameters
        ----------
        cond_latents : np.ndarray
            Conditioning latents from the conditioning encoder,
            shape ``(B, 32, 1024)``.
        text_tokens : np.ndarray
            BPE token IDs for the input text (**without** start/stop
            sentinels), shape ``(T_text,)``.

        Returns
        -------
        np.ndarray
            Prefix embedding tensor of shape
            ``(1, 32 + T_text + 2 + 1, 1024)``.
        """
        # Add start/stop text tokens
        tokens = np.concatenate(
            [
                [self.cfg.start_text_token],
                text_tokens,
                [self.cfg.stop_text_token],
            ]
        ).astype(np.int64)

        # Text embeddings + positional
        text_emb = self.sess.text_embedding[tokens]  # [T_text+2, 1024]
        T_text = text_emb.shape[0]
        text_pos = self.sess.text_pos_embedding[:T_text]  # [T_text+2, 1024]
        text_emb = text_emb + text_pos  # [T_text+2, 1024]

        # Cond latents: already [B, 32, 1024], squeeze batch for concat
        cond = cond_latents[0]  # [32, 1024]

        # Start audio token embedding + mel_pos at position 0
        start_audio_emb = self.sess.mel_embedding[self.cfg.start_audio_token]  # [1024]
        start_audio_pos = self.sess.mel_pos_embedding[0]  # [1024]
        start_audio_emb = start_audio_emb + start_audio_pos  # [1024]

        # Concatenate: [cond | text | start_audio]
        prefix = np.concatenate(
            [
                cond,  # [32, 1024]
                text_emb,  # [T_text+2, 1024]
                start_audio_emb[np.newaxis, :],  # [1, 1024]
            ],
            axis=0,
        )  # [32+T_text+2+1, 1024]

        return prefix[np.newaxis, :, :]  # [1, prefix_len, 1024]

    # -----------------------------------------------------------------
    # KV-cache utilities
    # -----------------------------------------------------------------

    def _create_zero_kv_cache(self, batch_size: int = 1):
        """Create zero-length KV cache for all layers.

        Each layer has a key and value tensor of shape
        ``(batch, num_heads, 0, head_dim)`` -- the sequence dimension is zero
        because no tokens have been processed yet.

        Parameters
        ----------
        batch_size : int, optional
            Batch size (default 1).

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys ``past_key_{i}`` and ``past_value_{i}``
            for ``i`` in ``range(n_layer)``.
        """
        kv = {}
        for i in range(self.cfg.n_layer):
            kv[f"past_key_{i}"] = np.zeros((batch_size, self.cfg.num_heads, 0, self.cfg.head_dim), dtype=np.float32)
            kv[f"past_value_{i}"] = np.zeros((batch_size, self.cfg.num_heads, 0, self.cfg.head_dim), dtype=np.float32)
        return kv

    # -----------------------------------------------------------------
    # Single GPT decode step
    # -----------------------------------------------------------------

    def _run_gpt_step(
        self,
        inputs_embeds: np.ndarray,  # [B, S, D]
        attention_mask: np.ndarray,  # [B, S_total]
        kv_cache: dict,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Run one GPT-2 forward pass through ONNX Runtime.

        During prefill ``inputs_embeds`` contains the entire prefix sequence.
        During decode it contains a single new token embedding.

        Parameters
        ----------
        inputs_embeds : np.ndarray
            Input embeddings, shape ``(B, S, embed_dim)`` where ``S`` is the
            number of new tokens (full prefix during prefill, 1 during decode).
        attention_mask : np.ndarray
            Float attention mask of shape ``(B, S_total)`` where ``S_total`` =
            past KV length + ``S``.
        kv_cache : dict
            Previous KV cache.  Keys: ``past_key_{i}``, ``past_value_{i}`` for
            each layer.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, dict]
            - **logits** -- shape ``(B, S, num_audio_tokens)``
            - **hidden_states** -- shape ``(B, S, embed_dim)``
            - **new_kv_cache** -- updated KV cache for the next step
        """
        feed = {
            "inputs_embeds": inputs_embeds.astype(np.float32),
            "attention_mask": attention_mask.astype(np.float32),
        }
        feed.update(kv_cache)

        # Build output names
        output_names = ["logits", "hidden_states"]
        for i in range(self.cfg.n_layer):
            output_names.extend([f"present_key_{i}", f"present_value_{i}"])

        results = self.sess.gpt.run(output_names, feed)

        logits = results[0]  # [B, S, 1026]
        hidden_states = results[1]  # [B, S, 1024]

        new_kv = {}
        for i in range(self.cfg.n_layer):
            new_kv[f"past_key_{i}"] = results[2 + 2 * i]
            new_kv[f"past_value_{i}"] = results[2 + 2 * i + 1]

        return logits, hidden_states, new_kv

    # -----------------------------------------------------------------
    # Main autoregressive generation loop
    # -----------------------------------------------------------------

    def generate_stream(
        self,
        cond_latents: np.ndarray,
        text_tokens: np.ndarray,
        sampling: SamplingConfig = SamplingConfig(),
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """External autoregressive generation loop.

        At each time-step the method:
            1. Samples an audio token from the logits (after repetition
               penalty, temperature scaling, top-k, and top-p filtering).
            2. Checks for the stop-audio sentinel.
            3. **Yields** ``(token_id, latent_vector)`` where
               ``latent_vector`` is the 1024-dim GPT hidden state used later
               by HiFi-GAN.
            4. Embeds the new token (mel embedding + positional), extends the
               attention mask, and runs a single-token GPT decode step.

        Parameters
        ----------
        cond_latents : np.ndarray
            Conditioning latents, shape ``(1, 32, 1024)``.
        text_tokens : np.ndarray
            BPE token IDs (without start/stop), shape ``(T_text,)``.
        sampling : SamplingConfig, optional
            Token sampling hyper-parameters.

        Yields
        ------
        tuple[int, np.ndarray]
            ``(token_id, latent_vector)`` per generated audio token.
            ``token_id`` is the VQ-VAE code index (0--1023).
            ``latent_vector`` is shape ``(1024,)`` for HiFi-GAN.
        """
        logging.info(f"generate_stream cond_latents {cond_latents} {cond_latents.shape}")
        logging.info(f"generate_stream text_tokens {text_tokens} {text_tokens.shape}")
        # 1. Build prefix embedding
        prefix_emb = self.compute_prefix_embedding(cond_latents, text_tokens)
        prefix_len = prefix_emb.shape[1]
        logging.info(f"generate_stream prefix_emb {prefix_emb} {prefix_emb.shape}")

        # 2. Init KV cache (zero-length)
        kv_cache = self._create_zero_kv_cache(batch_size=1)
        logging.info(f"generate_stream kv_cache {kv_cache} {kv_cache[list(kv_cache.keys())[0]].shape}")

        # 3. Prefill: run GPT on entire prefix
        attention_mask = np.ones((1, prefix_len), dtype=np.float32)
        logging.info(f"generate_stream attention_mask {attention_mask} {attention_mask.shape}")
        logits, hidden_states, kv_cache = self._run_gpt_step(prefix_emb, attention_mask, kv_cache)
        logging.info(f"generate_stream logits {logits} {logits.shape}")
        logging.info(f"generate_stream hidden_states {hidden_states} {hidden_states.shape}")
        logging.info(f"generate_stream kv_cache after {kv_cache[list(kv_cache.keys())[0]].shape}")

        # Track generated token IDs for repetition penalty (include prefix as dummy 1s)
        all_token_ids = np.ones((1, prefix_len), dtype=np.int64)
        logging.info(f"generate_stream all_token_ids {all_token_ids} {all_token_ids.shape}")

        # 4. Process first logits (from the last position of prefix)
        step_logits = logits[:, -1:, :]  # [1, 1, 1026]
        logging.info(f"generate_stream step_logits {step_logits} {step_logits.shape}")
        step_hidden = hidden_states[:, -1, :]  # [1, 1024]
        logging.info(f"generate_stream step_hidden {step_hidden} {step_hidden.shape}")

        # mel_pos index: start_audio was at position 0, so first generated token is at position 1
        mel_pos_idx = 1

        for step in range(self.cfg.max_gen_mel_tokens):
            # 5. Sample token from logits
            logging.info(f"generate_stream step_idx {step} mel_pos_idx {mel_pos_idx}")
            scores = step_logits[:, 0, :].copy()  # [1, 1026]
            logging.info(f"generate_stream scores init {scores} {scores.shape}")
            scores = apply_repetition_penalty(scores, all_token_ids, sampling.repetition_penalty)
            logging.info(f"generate_stream scores repp {scores} {scores.shape}")
            scores = apply_temperature(scores, sampling.temperature)
            logging.info(f"generate_stream scores temp {scores} {scores.shape}")
            scores = apply_top_k(scores, sampling.top_k)
            logging.info(f"generate_stream scores topk {scores} {scores.shape}")
            scores = apply_top_p(scores, sampling.top_p)
            logging.info(f"generate_stream scores topp {scores} {scores.shape}")

            if sampling.do_sample:
                probs = numpy_softmax(scores)
                next_token = numpy_multinomial(probs)
                logging.info(f"generate_stream probs {probs} {probs.shape}")
                logging.info(f"generate_stream next_token {next_token}")
            else:
                next_token = int(np.argmax(scores, axis=-1)[0])

            # 6. Check stop condition
            if next_token == self.cfg.stop_audio_token:
                break

            # 7. Yield (token, latent)
            yield next_token, step_hidden[0]  # latent: [1024]

            # 8. Prepare next step input: embed the new token
            token_emb = self.sess.mel_embedding[next_token]  # [1024]
            logging.info(f"generate_stream token_emb {token_emb} {token_emb.shape}")
            pos_emb = self.sess.mel_pos_embedding[mel_pos_idx]  # [1024]
            logging.info(f"generate_stream pos_emb {pos_emb} {pos_emb.shape}")
            next_emb = (token_emb + pos_emb)[np.newaxis, np.newaxis, :]  # [1, 1, 1024]
            logging.info(f"generate_stream next_emb {next_emb} {next_emb.shape}")

            mel_pos_idx += 1

            # 9. Extend attention mask
            current_total_len = attention_mask.shape[1] + 1
            attention_mask = np.ones((1, current_total_len), dtype=np.float32)
            logging.info(f"generate_stream attention_mask {attention_mask} {attention_mask.shape}")

            # 10. Run GPT decode step (single token)
            step_logits, step_hidden_full, kv_cache = self._run_gpt_step(next_emb, attention_mask, kv_cache)
            logging.info(f"generate_stream step_logits {step_logits} {step_logits.shape}")
            logging.info(f"generate_stream step_hidden_full {step_hidden_full} {step_hidden_full.shape}")
            logging.info(f"generate_stream kv_cache step {kv_cache[list(kv_cache.keys())[0]].shape}")
            step_hidden = step_hidden_full[:, -1, :]  # [1, 1024]
            logging.info(f"generate_stream step_hidden {step_hidden} {step_hidden.shape}")

            # 11. Track token IDs
            all_token_ids = np.concatenate([all_token_ids, np.array([[next_token]], dtype=np.int64)], axis=1)
            logging.info(f"generate_stream all_token_ids {all_token_ids} {all_token_ids.shape}")

    # -----------------------------------------------------------------
    # Vocoder
    # -----------------------------------------------------------------

    def vocoder(self, latents: np.ndarray, speaker_embedding: np.ndarray) -> np.ndarray:
        """Run HiFi-GAN vocoder to synthesise a waveform from GPT hidden states.

        Parameters
        ----------
        latents : np.ndarray
            Accumulated GPT hidden states, shape ``(1, T, 1024)``.
        speaker_embedding : np.ndarray
            512-dim speaker embedding, shape ``(1, 512, 1)``.

        Returns
        -------
        np.ndarray
            Synthesised waveform, shape ``(1, 1, T_audio)``.
        """
        result = self.sess.hifigan.run(
            None,
            {
                "latents": latents.astype(np.float32),
                "speaker_embedding": speaker_embedding.astype(np.float32),
            },
        )
        return result[0]  # [1, 1, T_audio]
