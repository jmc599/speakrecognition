from pathlib import Path

import numpy as np
import torch
import torchaudio

try:
    import av
except ImportError:  # pragma: no cover
    av = None


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


def load_audio_mono_16k(audio_path):
    audio_path = str(Path(audio_path))
    try:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=16000
            )
        return waveform, 16000
    except Exception as exc:
        try:
            return _load_audio_with_pyav(audio_path)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Failed to load audio: {audio_path}. "
                f"torchaudio error: {exc}. PyAV error: {fallback_exc}"
            ) from fallback_exc
