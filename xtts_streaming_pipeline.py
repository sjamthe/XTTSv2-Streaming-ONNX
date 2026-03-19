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
xtts_streaming_pipeline.py
---------------------------
Full streaming TTS pipeline matching test_xttsv2_streaming.py calling pattern.
Receives NMT text segments, yields audio chunks via ONNX-based inference.

This module provides the top-level :class:`StreamingTTSPipeline` that wires
together all components needed for XTTSv2 streaming inference:

    1. **Reference audio conditioning** -- extract a 32 x 1024 GPT
       conditioning latent and a 512-dim speaker embedding from an
       arbitrary reference audio clip.
    2. **Text tokenisation** -- BPE-encode the input text via
       :class:`VoiceBpeTokenizer`.
    3. **Autoregressive generation** -- run the GPT-2 AR loop through the
       :class:`XTTSOrchestratorONNX` to produce mel-code latents.
    4. **Vocoding & streaming** -- convert accumulated latents to waveform
       chunks with the HiFi-GAN vocoder and cross-fade consecutive chunks.

The ``__main__`` block demonstrates a simulated NMT streaming scenario
that synthesises speech for text arriving in 10-word chunks.

See Also:
    - ``xtts_onnx_orchestrator.py`` for the low-level ONNX inference.
    - XTTS paper: https://arxiv.org/abs/2406.04904
"""

import argparse
import logging
import os
import sys
import time
from typing import Generator, Optional

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy.interpolate import interp1d

from xtts_onnx_orchestrator import (GPTConfig, ONNXSessionManager,
                                    SamplingConfig, XTTSOrchestratorONNX)
from xtts_tokenizer import VoiceBpeTokenizer

LOGGING_LEVEL = ["DEBUG", "INFO", "WARNING", "ERROR"]

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def setup_logging(level: str = "WARNING") -> logging.Logger:
    """Configure root logger with timestamped format.

    Parameters
    ----------
    level : str, optional
        Logging level name (default ``"WARNING"``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger("data_processing_pipeline")
    logger.setLevel(getattr(logging, level.upper(), logging.WARNING))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


log = setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# Audio Utilities
# ─────────────────────────────────────────────────────────────────────────────


def _hz_to_mel_htk(freq: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert Hz to mel scale using the HTK formula.

    Parameters
    ----------
    freq : np.ndarray
        Frequency value(s) in Hz.

    Returns
    -------
    np.ndarray
        Corresponding mel-scale value(s).
    """
    # HTK mel formula
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz_htk(mels: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert mel-scale values back to Hz using the inverse HTK formula.

    Parameters
    ----------
    mels : np.ndarray
        Mel-scale value(s).

    Returns
    -------
    np.ndarray
        Corresponding frequency value(s) in Hz.
    """
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def _melscale_fbanks_torchaudio_style(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    norm: str | None = "slaney",
) -> npt.NDArray[np.float32]:
    """Build a mel-scale triangular filterbank in torchaudio style.

    Pure-NumPy re-implementation of ``torchaudio.functional.melscale_fbanks``
    with explicit triangular filter construction and optional Slaney-style
    area normalisation.

    Parameters
    ----------
    n_freqs : int
        Number of STFT frequency bins (typically ``n_fft // 2 + 1``).
    f_min : float
        Lowest filter centre frequency (Hz).
    f_max : float
        Highest filter centre frequency (Hz).
    n_mels : int
        Number of mel channels.
    sample_rate : int
        Audio sample rate (Hz).
    norm : {"slaney", None}, optional
        If ``"slaney"``, apply area normalisation so each filter integrates
        to unity.

    Returns
    -------
    np.ndarray
        Filterbank matrix of shape ``(n_freqs, n_mels)``.
    """
    # Frequency grid (STFT bins)
    freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=np.float64)

    # Mel points
    m_min = _hz_to_mel_htk(np.array([f_min], dtype=np.float64))[0]
    m_max = _hz_to_mel_htk(np.array([f_max], dtype=np.float64))[0]
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float64)

    # Convert mel points back to Hz
    hz_pts = _mel_to_hz_htk(m_pts)

    # Triangular filters
    fb = np.zeros((n_freqs, n_mels), dtype=np.float64)
    for i in range(n_mels):
        f_left = hz_pts[i]
        f_center = hz_pts[i + 1]
        f_right = hz_pts[i + 2]

        # rising slope
        left_mask = (freqs >= f_left) & (freqs <= f_center)
        fb[left_mask, i] = (freqs[left_mask] - f_left) / (f_center - f_left + 1e-10)

        # falling slope
        right_mask = (freqs >= f_center) & (freqs <= f_right)
        fb[right_mask, i] = (f_right - freqs[right_mask]) / (f_right - f_center + 1e-10)

    if norm == "slaney":
        # area normalization by band width (like torchaudio/librosa "slaney")
        enorm = 2.0 / (hz_pts[2:] - hz_pts[:-2])
        fb *= enorm.reshape(1, -1)

    return fb.astype(np.float32)


def wav_to_mel_cloning_numpy(
    wav: npt.NDArray[np.float32],  # shape [1, T]
    mel_norms: npt.NDArray[np.float32],  # shape [80]
    sample_rate: int = 22050,
) -> npt.NDArray[np.float32]:
    """Convert a waveform to a normalised log-mel spectrogram for voice cloning.

    Pure-NumPy approximation of the PyTorch conditioning path::

        mel_stft = torchaudio.transforms.MelSpectrogram(
            n_fft=2048, hop_length=256, win_length=1024, power=2,
            normalized=False, sample_rate=sample_rate,
            f_min=0, f_max=8000, n_mels=80, norm="slaney")
        mel = mel_stft(wav)                     # [1, 80, frames]
        mel = torch.log(torch.clamp(mel, 1e-5))
        mel = mel / mel_norms[None, :, None]

    Parameters
    ----------
    wav : np.ndarray
        Mono waveform of shape ``(1, T)`` at ``sample_rate``.
    mel_norms : np.ndarray
        Per-channel normalisation factors, shape ``(80,)``.
    sample_rate : int, optional
        Waveform sample rate (default 22050).

    Returns
    -------
    np.ndarray
        Normalised log-mel spectrogram of shape ``(1, 80, frames)``.
    """
    assert wav.ndim == 2 and wav.shape[0] == 1, "Expected wav shape [1, T]"
    n_fft = 2048
    hop_length = 256
    win_length = 1024
    n_mels = 80
    f_min = 0.0
    f_max = 8000.0

    wav = wav.astype(np.float32)

    # 1) center=True, pad_mode="reflect" around time axis
    pad = n_fft // 2
    x = wav
    # reflect pad only along time dimension
    # Left reflect
    left = x[:, 1 : pad + 1][:, ::-1]
    # Right reflect
    right = x[:, -pad - 1 : -1][:, ::-1]
    x = np.concatenate([left, x, right], axis=1)  # shape [1, T + 2*pad]

    # 2) Framing into STFT windows
    num_frames = 1 + (x.shape[1] - n_fft) // hop_length
    frame_idx = (
        np.arange(n_fft)[None, None, :] + hop_length * np.arange(num_frames)[None, :, None]
    )  # shape [1, frames, n_fft]
    x_expanded = x[:, :, None]
    frames = x_expanded[np.arange(1)[:, None, None], frame_idx, np.zeros_like(frame_idx)]  # [1, frames, n_fft]

    # 3) Window function (Hann is default in torchaudio MelSpectrogram)
    window = np.hanning(win_length).astype(np.float32)
    if n_fft > win_length:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        window = np.pad(window, (pad_left, pad_right))
    window = window[None, None, :]  # [1, 1, n_fft]
    frames = frames.astype(np.float32) * window

    # 4) STFT magnitude^2, power=2
    spec = np.fft.rfft(frames, n=n_fft, axis=-1)
    spec_power = (np.abs(spec) ** 2).astype(np.float32)  # [1, frames, n_fft//2+1]

    # 5) Mel filterbank
    n_freqs = n_fft // 2 + 1
    fbanks = _melscale_fbanks_torchaudio_style(
        n_freqs=n_freqs,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm="slaney",
    )  # [n_freqs, n_mels]

    mel_spec = np.matmul(spec_power, fbanks)  # [1, frames, n_mels]

    # 6) Match torchaudio layout: [channels, n_mels, frames]
    mel_spec = np.transpose(mel_spec, (0, 2, 1))  # [1, 80, frames]

    # 7) Log and normalization like PyTorch code
    mel_spec = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))
    mel_spec = mel_spec / mel_norms[None, :, None]

    return mel_spec.astype(np.float32)


def crossfade_chunks(
    wav_gen: np.ndarray,
    wav_gen_prev: Optional[np.ndarray],
    wav_overlap: Optional[np.ndarray],
    overlap_len: int = 1024,
) -> tuple:
    """Handle crossfading between consecutive vocoder outputs.

    Mirrors ``Xtts.handle_chunks()`` logic in the original PyTorch
    implementation.  Applies a linear fade-in / fade-out to the overlap
    region between the previous and current vocoder waveform.

    Parameters
    ----------
    wav_gen : np.ndarray
        Full vocoder output for the current chunk, shape ``(T,)``.
    wav_gen_prev : np.ndarray or None
        Full vocoder output from the *previous* chunk (``None`` on first
        call).
    wav_overlap : np.ndarray or None
        Overlap tail saved from the previous chunk (``None`` on first call).
    overlap_len : int, optional
        Number of samples in the crossfade region (default 1024).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray or None]
        ``(wav_chunk, wav_gen_prev_new, wav_overlap_new)`` where
        ``wav_chunk`` is the audio to emit, ``wav_gen_prev_new`` replaces
        ``wav_gen_prev`` for the next call, and ``wav_overlap_new`` replaces
        ``wav_overlap``.
    """
    wav_chunk = wav_gen[:-overlap_len]

    if wav_gen_prev is not None:
        wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len) : -overlap_len]

    if wav_overlap is not None:
        if overlap_len > len(wav_chunk):
            if wav_gen_prev is not None:
                wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len) :]
            else:
                wav_chunk = wav_gen[-overlap_len:]
            return wav_chunk, wav_gen, None
        else:
            fade_in = np.linspace(0.0, 1.0, overlap_len, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, overlap_len, dtype=np.float32)
            crossfade = wav_chunk[:overlap_len] * fade_in
            wav_chunk[:overlap_len] = wav_overlap * fade_out
            wav_chunk[:overlap_len] += crossfade

    wav_overlap_new = wav_gen[-overlap_len:]
    wav_gen_prev_new = wav_gen
    return wav_chunk, wav_gen_prev_new, wav_overlap_new


# ─────────────────────────────────────────────────────────────────────────────
# Streaming TTS Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class StreamingTTSPipeline:
    """Streaming XTTSv2 inference pipeline using ONNX models.

    Drop-in replacement for the PyTorch ``model.inference_stream()``.  The
    pipeline owns all ONNX sessions, the BPE tokenizer, the mel-norm
    statistics, and the :class:`XTTSOrchestratorONNX`.

    Typical usage::

        pipeline = StreamingTTSPipeline(model_dir, vocab_path, mel_norms_path)
        cond, spk = pipeline.get_conditioning_latents("ref.wav")
        for chunk in pipeline.inference_stream("Hello world", "en", cond, spk):
            play(chunk)  # float32, 24 kHz

    Parameters
    ----------
    model_dir : str
        Directory containing all ONNX models, ``metadata.json``, and the
        ``embeddings/`` sub-directory.
    vocab_path : str
        Path to the BPE ``vocab.json`` file.
    mel_norms_path : str
        Path to the ``mel_stats.npy`` normalisation file (shape ``(80,)``).
    use_int8_gpt : bool, optional
        Whether to load the INT8-quantised GPT model (default ``True``).
    num_threads_gpt : int, optional
        ONNX Runtime intra-op threads for the GPT session (default 2).
    num_threads_vocoder : int, optional
        ONNX Runtime intra-op threads for the vocoder session (default 1).
        *Currently unused -- reserved for future per-session threading.*

    Attributes
    ----------
    sessions : ONNXSessionManager
        All ONNX inference sessions and embedding tables.
    gpt_config : GPTConfig
        GPT-2 model configuration loaded from ``metadata.json``.
    tokenizer : VoiceBpeTokenizer
        BPE tokenizer for text encoding.
    orchestrator : XTTSOrchestratorONNX
        Autoregressive loop orchestrator.
    mel_norms : np.ndarray
        Per-channel mel normalisation factors, shape ``(80,)``.
    sample_rate : int
        Output audio sample rate (24000 Hz).
    """

    def __init__(
        self,
        model_dir: str,
        vocab_path: str,
        mel_norms_path: str,
        use_int8_gpt: bool = True,
        num_threads_gpt: int = 2,
        num_threads_vocoder: int = 1,
    ):
        # Load sessions
        self.sessions = ONNXSessionManager(model_dir, use_int8_gpt=use_int8_gpt, num_threads=num_threads_gpt)

        # Build config from metadata
        self.gpt_config = GPTConfig.from_metadata(os.path.join(model_dir, "metadata.json"))

        # Tokenizer
        self.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
        self.gpt_config.start_text_token = self.tokenizer.tokenizer.token_to_id("[START]")
        self.gpt_config.stop_text_token = self.tokenizer.tokenizer.token_to_id("[STOP]")

        # Orchestrator
        self.orchestrator = XTTSOrchestratorONNX(self.sessions, self.gpt_config)

        # Mel norms for conditioning
        self.mel_norms = np.load(mel_norms_path).astype(np.float32)  # shape [80]
        if self.mel_norms.ndim != 1 or self.mel_norms.shape[0] != 80:
            raise ValueError(f"Expected (80,), got {self.mel_norms.shape}")

        self.sample_rate = 24000

    # ─── One-time speaker conditioning ────────────────────────────────────

    def get_conditioning_latents(self, audio_path: str) -> tuple:
        """Compute speaker conditioning from a reference audio clip.

        Mirrors ``Xtts.get_conditioning_latents()`` in the original PyTorch
        implementation.  Produces two artefacts:

        * **GPT conditioning latent** -- a 32 x 1024 matrix extracted from
          a mel spectrogram resampled to 22050 Hz (max 6 s).
        * **Speaker embedding** -- a 512-dim vector extracted from the
          audio resampled to 16000 Hz.

        Parameters
        ----------
        audio_path : str
            Path to the reference audio file (any format supported by
            ``soundfile``).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(gpt_cond_latent, speaker_embedding)`` where
            ``gpt_cond_latent`` has shape ``(1, 32, 1024)`` and
            ``speaker_embedding`` has shape ``(1, 512, 1)``.
        """
        # Read with soundfile; returns [frames, channels] or [frames] if mono.
        audio, sr = sf.read(audio_path, always_2d=True, dtype="float32")  # [frames, channels]
        # Convert to [channels, T]
        audio = audio.T  # [channels, T]
        # If more than one channel, average to mono
        if audio.shape[0] > 1:
            audio = audio.mean(axis=0, keepdims=True)  # [1, T]
        else:
            # still keep [1, T]
            audio = audio
        log.info(f"get_conditioning_latents audio {audio} {audio.shape} {audio.dtype}")

        # GPT conditioning: mel at 22050
        audio_22k = librosa.resample(audio[0], orig_sr=sr, target_sr=22050)  # [T_resampled]
        audio_22k = audio_22k[np.newaxis, :].astype(np.float32)  # [1, T_resampled]
        log.info(f"get_conditioning_latents audio_22k {audio_22k} {audio_22k.shape} {audio_22k.dtype}")
        audio_22k = audio_22k[:, : 22050 * 6]  # 6 seconds max
        log.info(f"get_conditioning_latents audio_22k_cut {audio_22k} {audio_22k.shape} {audio_22k.dtype}")
        mel = wav_to_mel_cloning_numpy(audio_22k, self.mel_norms)  # [1, 80, T]
        log.info(f"get_conditioning_latents mel {mel} {mel.shape} {mel.dtype}")
        gpt_cond_latent = self.orchestrator.compute_conditioning(mel)  # [1, 32, 1024]
        log.info(
            f"get_conditioning_latents gpt_cond_latent {gpt_cond_latent} {gpt_cond_latent.shape} {gpt_cond_latent.dtype}"
        )

        # Speaker embedding: audio at 16kHz
        audio_16k = librosa.resample(audio[0], orig_sr=sr, target_sr=16000)  # [T_resampled]
        audio_16k = audio_16k[np.newaxis, :].astype(np.float32)  # [1, T_resampled]
        log.info(f"get_conditioning_latents audio_16k {audio_16k} {audio_16k.shape} {audio_16k.dtype}")
        speaker_embedding = self.orchestrator.compute_speaker_embedding(audio_16k)  # [1, 512, 1]
        log.info(f"get_conditioning_latents speaker_embedding {speaker_embedding.shape} {speaker_embedding.dtype}")

        return gpt_cond_latent, speaker_embedding

    # ─── Streaming inference (mirrors Xtts.inference_stream) ──────────────

    def time_scale_gpt_latents_numpy(
        self,
        gpt_latents: np.ndarray,  # shape [B, T, C]
        speed: float,
    ) -> np.ndarray:
        """Linearly time-scale GPT latents for speed control.

        Torch-free equivalent of ``torch.nn.functional.interpolate`` with
        ``mode='linear'`` and ``scale_factor=1/speed``.  Reshapes latents
        to ``(B*C, T)`` for batched 1-D interpolation via
        :func:`scipy.interpolate.interp1d`, then restores the original
        layout.

        Parameters
        ----------
        gpt_latents : np.ndarray
            GPT hidden-state latents of shape ``(B, T, C)``.
        speed : float
            Speed factor (``1.0`` = normal, ``> 1.0`` = faster / shorter,
            ``< 1.0`` = slower / longer).  Clamped to ``>= 0.05``.

        Returns
        -------
        np.ndarray
            Time-scaled latents of shape ``(B, T_out, C)``.
        """
        if speed == 1.0:
            return gpt_latents

        length_scale = 1.0 / max(speed, 0.05)  # same as original
        B, T, C = gpt_latents.shape

        # Target length similar to PyTorch interpolate with scale_factor
        T_out = int(np.floor(T * length_scale))
        if T_out < 1:
            T_out = 1

        # Original and new time indices (0..T-1 mapped to 0..T_out-1)
        x = np.linspace(0.0, T - 1, num=T, dtype=np.float32)
        x_new = np.linspace(0.0, T - 1, num=T_out, dtype=np.float32)

        # Reshape to [B*C, T] for batched 1D interpolation, then back to [B, T_out, C]
        y = gpt_latents.reshape(-1, T)  # [B*C, T]

        # Build interpolator along axis=1 (time) and evaluate at x_new
        f = interp1d(
            x,
            y,
            kind="linear",
            axis=1,
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        y_new = f(x_new)  # [B*C, T_out]

        gpt_latents_out = y_new.reshape(B, C, T_out).transpose(0, 2, 1)  # [B, T_out, C]
        return gpt_latents_out.astype(gpt_latents.dtype, copy=False)

    def inference_stream(
        self,
        text: str,
        language: str,
        gpt_cond_latent: np.ndarray,
        speaker_embedding: np.ndarray,
        stream_chunk_size: int = 20,
        overlap_wav_len: int = 1024,
        sampling: Optional[SamplingConfig] = None,
        speed: float = 1.0,
    ) -> Generator[np.ndarray, None, None]:
        """Stream audio chunks for a single text segment.

        The method drives the autoregressive token generator, accumulates
        latent vectors, and periodically runs the HiFi-GAN vocoder to emit
        crossfaded waveform chunks.

        Parameters
        ----------
        text : str
            Text string to synthesize.
        language : str
            Language code (e.g. ``"en"``, ``"ru"``, ``"zh-cn"``).  Only the
            part before ``"-"`` is used.
        gpt_cond_latent : np.ndarray
            Conditioning latent from :meth:`get_conditioning_latents`,
            shape ``(1, 32, 1024)``.
        speaker_embedding : np.ndarray
            Speaker embedding from :meth:`get_conditioning_latents`,
            shape ``(1, 512, 1)``.
        stream_chunk_size : int, optional
            Number of AR tokens to accumulate before running the vocoder
            and yielding an audio chunk (default 20).
        overlap_wav_len : int, optional
            Crossfade overlap length in samples (default 1024).
        sampling : SamplingConfig or None, optional
            Token sampling configuration.  Defaults to
            :class:`SamplingConfig` defaults.
        speed : float, optional
            Playback speed factor (default 1.0).

        Yields
        ------
        np.ndarray
            Audio chunks as 1-D float32 arrays at 24 kHz.
        """
        if sampling is None:
            sampling = SamplingConfig()

        language = language.split("-")[0]

        # Tokenize
        text = text.strip().lower()
        text_tokens = np.array(
            self.tokenizer.encode(text, lang=language),
            dtype=np.int64,
        )
        logging.debug(f"inference_stream text {text} {len(text)} {language}")
        logging.debug(f"inference_stream text_tokens {text_tokens} {text_tokens.shape}")

        # Run autoregressive generation
        all_latents = []
        last_tokens = []
        wav_gen_prev = None
        wav_overlap = None

        generator = self.orchestrator.generate_stream(gpt_cond_latent, text_tokens, sampling)

        is_end = False
        while not is_end:
            try:
                token_id, latent = next(generator)
                logging.debug(f"inference_stream token_id {token_id} latent {latent} {latent.shape}")
                last_tokens.append(token_id)
                logging.debug(f"inference_stream last_tokens {last_tokens} {len(last_tokens)}")
                all_latents.append(latent)  # [1024]
                logging.debug(f"inference_stream all_latents {len(all_latents)} {all_latents[0].shape}")
            except StopIteration:
                is_end = True

            should_voccode = is_end or (stream_chunk_size > 0 and len(last_tokens) >= stream_chunk_size)

            if should_voccode and len(all_latents) > 0:
                # Stack ALL accumulated latents (not just new ones)
                gpt_latents = np.stack(all_latents, axis=0)[np.newaxis, :, :]  # [1, N, 1024]
                logging.debug(f"inference_stream gpt_latents {gpt_latents} {gpt_latents.shape}")

                # Speed adjustment
                gpt_latents = self.time_scale_gpt_latents_numpy(gpt_latents, speed)  # [1, N_new, 1024]

                # Run vocoder
                wav_gen = self.orchestrator.vocoder(gpt_latents, speaker_embedding)
                logging.debug(f"inference_stream wav_gen {wav_gen} {wav_gen.shape}")
                wav_gen = wav_gen.squeeze()  # [T_audio]
                logging.debug(f"inference_stream wav_gen squeeze {wav_gen} {wav_gen.shape}")

                # Crossfade
                wav_chunk, wav_gen_prev, wav_overlap = crossfade_chunks(
                    wav_gen, wav_gen_prev, wav_overlap, overlap_wav_len
                )
                logging.debug(f"inference_stream wav_chunk {wav_chunk} {wav_chunk.shape}")
                logging.debug(f"inference_stream wav_gen_prev {wav_gen_prev} {wav_gen_prev.shape}")
                if wav_overlap is not None:
                    logging.debug(f"inference_stream wav_overlap {wav_overlap} {wav_overlap.shape}")
                else:
                    logging.debug(f"inference_stream wav_overlap {wav_overlap}")

                last_tokens = []
                yield wav_chunk


# ─────────────────────────────────────────────────────────────────────────────
# Example usage (mirrors test_xttsv2_streaming.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XTTSv2 ONNX Streaming TTS")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing ONNX models")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocab.json")
    parser.add_argument("--mel_norms_path", type=str, required=True, help="Path to mel_stats.pth")
    parser.add_argument("--ref_audio", type=str, required=True, help="Reference audio for voice cloning")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--use_int8_gpt", action="store_true", help="Use quantized INT8 GPT ONNX model")
    parser.add_argument("--output", type=str, default="output_streaming.wav")
    args = parser.parse_args()

    log.setLevel(getattr(logging, LOGGING_LEVEL[0], logging.WARNING))

    # Initialize pipeline
    pipeline = StreamingTTSPipeline(
        model_dir=args.model_dir,
        vocab_path=args.vocab_path,
        mel_norms_path=args.mel_norms_path,
        use_int8_gpt=args.use_int8_gpt,
    )
    log.info(pipeline)

    # Compute speaker conditioning (one-time)
    log.info("Computing speaker conditioning...")
    gpt_cond_latent, speaker_embedding = pipeline.get_conditioning_latents(args.ref_audio)

    # ── Simulated NMT text stream (replace with real NMT output) ──
    def nmt_text_stream(full_text: str, chunk_size: int = 10):
        """Simulate an NMT system emitting text in word-level chunks.

        Parameters
        ----------
        full_text : str
            Complete text to split.
        chunk_size : int, optional
            Words per chunk (default 10).

        Yields
        ------
        str
            Text chunks of up to ``chunk_size`` words.
        """
        words = full_text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            yield chunk
            time.sleep(0.1)

    text = (
        "Depending on the time, not only accuracy, but also low latency is important. "
        "If it is not instant, then the human interaction is lost. "
        "We are finally reaching a moment when the technology is fast enough "
        "for people to simply communicate, and this is a huge shift for global business."
    )

    # ── Stream synthesis ──
    all_audio = []
    log.info("Starting streaming synthesis...")
    t0 = time.time()

    for text_chunk in nmt_text_stream(text, chunk_size=10):
        if not text_chunk.strip():
            continue
        log.info(f'\n[NMT] \u2192 "{text_chunk}"')

        for i, audio_chunk in enumerate(
            pipeline.inference_stream(
                text_chunk,
                args.language,
                gpt_cond_latent,
                speaker_embedding,
                stream_chunk_size=20,
            )
        ):
            log.info(f"  Audio chunk {i}: {audio_chunk.shape[0]} samples " f"({audio_chunk.shape[0]/24000:.2f}s)")
            all_audio.append(audio_chunk)

    elapsed = time.time() - t0
    log.info(f"\nTotal synthesis time: {elapsed:.2f}s")

    if all_audio:
        full_wav = np.concatenate(all_audio, axis=0)
        sf.write(args.output, full_wav, 24000)
        log.info(f"Saved: {args.output} ({full_wav.shape[0]/24000:.2f}s audio)")
