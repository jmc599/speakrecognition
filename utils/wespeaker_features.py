from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi


WESPEAKER_SAMPLE_RATE = 16000
WESPEAKER_NUM_MEL_BINS = 80
WESPEAKER_FRAME_LENGTH_MS = 25
WESPEAKER_FRAME_SHIFT_MS = 10
WESPEAKER_DITHER = 0.0


def _load_waveform(path_text: str) -> tuple[torch.Tensor, int]:
    """Load waveform with torchaudio first, then fallback to project audio loader."""
    try:
        waveform, sample_rate = torchaudio.load(path_text)
        return waveform, int(sample_rate)
    except Exception as torchaudio_exc:
        try:
            from utils.audio_io import load_audio_mono_16k
        except Exception as import_exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to load audio with torchaudio. "
                "Please install torchcodec/PyAV compatible with your torch/torchaudio build."
            ) from import_exc

        try:
            waveform, sample_rate = load_audio_mono_16k(path_text, for_inference=False)
            return waveform, int(sample_rate)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Failed to load audio: {path_text}. "
                f"torchaudio error: {torchaudio_exc}. "
                f"fallback error: {fallback_exc}"
            ) from fallback_exc


def _to_mono_16k(waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape [C, T], got {tuple(waveform.shape)}")

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if int(sample_rate) != WESPEAKER_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=int(sample_rate),
            new_freq=WESPEAKER_SAMPLE_RATE,
        )
        sample_rate = WESPEAKER_SAMPLE_RATE

    return waveform, int(sample_rate)


def validate_wespeaker_feature(feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim != 3:
        raise ValueError(f"Expected 3D feature [1, T, 80], got shape {tuple(feature.shape)}")
    if feature.shape[0] != 1:
        raise ValueError(f"Expected batch dimension 1, got {feature.shape[0]}")
    if feature.shape[2] != WESPEAKER_NUM_MEL_BINS:
        raise ValueError(
            f"Expected mel bins {WESPEAKER_NUM_MEL_BINS}, got {feature.shape[2]}"
        )
    if feature.shape[1] <= 0:
        raise ValueError("Feature has zero frames.")
    if not np.isfinite(feature).all():
        raise ValueError("Feature contains NaN or Inf.")
    return np.ascontiguousarray(feature, dtype=np.float32)


def compute_wespeaker_fbank(wav_path: str | Path) -> np.ndarray:
    """Extract WeSpeaker-compatible fbank features (official infer_onnx settings)."""
    wav_path = str(Path(wav_path))
    waveform, sample_rate = _load_waveform(wav_path)
    waveform = waveform.to(dtype=torch.float32)
    waveform, sample_rate = _to_mono_16k(waveform, sample_rate)

    waveform = waveform * float(1 << 15)
    mat = kaldi.fbank(
        waveform,
        num_mel_bins=WESPEAKER_NUM_MEL_BINS,
        frame_length=WESPEAKER_FRAME_LENGTH_MS,
        frame_shift=WESPEAKER_FRAME_SHIFT_MS,
        dither=WESPEAKER_DITHER,
        sample_frequency=sample_rate,
        window_type="hamming",
        use_energy=False,
    )
    if mat.numel() == 0:
        raise RuntimeError(f"Empty fbank extracted from audio: {wav_path}")

    # CMN without CVN; matches WeSpeaker infer_onnx.py.
    mat = mat - torch.mean(mat, dim=0, keepdim=True)
    feature = mat.unsqueeze(0).cpu().numpy().astype(np.float32, copy=False)
    return validate_wespeaker_feature(feature)
