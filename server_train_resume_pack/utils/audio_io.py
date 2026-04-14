from pathlib import Path

import numpy as np
import torch
import torchaudio

from utils.inference_audio import preprocess_waveform_for_inference

try:
    import av
except ImportError:  # pragma: no cover
    av = None


def _load_audio_with_torchaudio(audio_path):
    errors = []
    for backend in ("soundfile", "ffmpeg"):
        try:
            waveform, sample_rate = torchaudio.load(audio_path, backend=backend)
            return waveform, sample_rate, backend
        except TypeError:
            break
        except Exception as exc:  # pragma: no cover
            errors.append(f"{backend}: {exc}")

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        return waveform, sample_rate, "auto"
    except Exception as exc:
        errors.append(f"auto: {exc}")
        raise RuntimeError("; ".join(errors))


def _load_audio_with_pyav(audio_path):
    if av is None:
        raise RuntimeError("PyAV is not installed, cannot decode this audio format.")

    container = av.open(str(audio_path))
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        raise RuntimeError(f"No audio stream found in: {audio_path}")

    resampler = av.audio.resampler.AudioResampler(
        format="fltp",
        layout="mono",
        rate=16000,
    )

    chunks = []
    for frame in container.decode(stream):
        out_frames = resampler.resample(frame)
        if out_frames is None:
            continue
        if not isinstance(out_frames, list):
            out_frames = [out_frames]
        for out_frame in out_frames:
            array = out_frame.to_ndarray()
            if array.ndim == 1:
                array = array[None, :]
            elif array.ndim == 2 and array.shape[0] > 1:
                array = array.mean(axis=0, keepdims=True)
            array = array.astype(np.float32, copy=False)
            chunks.append(array)

    if not chunks:
        raise RuntimeError(f"Decoded empty audio stream from: {audio_path}")

    waveform = torch.from_numpy(np.concatenate(chunks, axis=1))
    return waveform, 16000


def load_audio_mono_16k(audio_path, for_inference=False, return_info=False):
    audio_path = str(Path(audio_path))
    try:
        waveform, sr, backend_used = _load_audio_with_torchaudio(audio_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=16000
            )
        sample_rate = 16000
    except Exception as exc:
        try:
            waveform, sample_rate = _load_audio_with_pyav(audio_path)
            backend_used = "pyav"
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Failed to load audio: {audio_path}. "
                f"torchaudio error: {exc}. PyAV error: {fallback_exc}"
            ) from fallback_exc

    info = {
        "sample_rate": int(sample_rate),
        "for_inference": bool(for_inference),
        "audio_backend": backend_used,
    }
    if for_inference:
        processed, preprocess_info = preprocess_waveform_for_inference(
            waveform.squeeze(0).cpu().numpy(),
            sample_rate=sample_rate,
            trim_silence=True,
            normalize_volume=True,
        )
        waveform = torch.from_numpy(processed).unsqueeze(0)
        info.update(preprocess_info)

    if return_info:
        return waveform, sample_rate, info
    return waveform, sample_rate
