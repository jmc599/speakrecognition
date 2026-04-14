import numpy as np


def _compute_frame_activity(
    waveform,
    sample_rate=16000,
    frame_ms=25,
    hop_ms=10,
    top_db=35.0,
):
    x = np.asarray(waveform, dtype=np.float32).reshape(-1)
    frame_length = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_length = max(1, int(sample_rate * hop_ms / 1000.0))

    if x.size <= frame_length:
        starts = np.asarray([0], dtype=np.int64)
        frame = x[:frame_length]
        rms = np.asarray(
            [float(np.sqrt(np.mean(frame * frame) + 1e-12))],
            dtype=np.float32,
        )
    else:
        starts = np.arange(0, x.size - frame_length + 1, hop_length, dtype=np.int64)
        rms = np.empty(starts.shape[0], dtype=np.float32)
        for idx, start in enumerate(starts):
            frame = x[start : start + frame_length]
            rms[idx] = float(np.sqrt(np.mean(frame * frame) + 1e-12))

    peak_rms = float(rms.max()) if rms.size else 0.0
    threshold = max(peak_rms * (10.0 ** (-top_db / 20.0)), 1e-4)
    active = rms >= threshold if peak_rms > 1e-6 else np.zeros_like(rms, dtype=bool)
    return x, starts, frame_length, active, peak_rms


def detect_active_spans(
    waveform,
    sample_rate=16000,
    frame_ms=25,
    hop_ms=10,
    top_db=35.0,
    min_active_ms=200,
):
    """Return active sample spans from a mono waveform using the project's energy heuristic."""
    x, starts, frame_length, active, _ = _compute_frame_activity(
        waveform,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        top_db=top_db,
    )
    min_active_samples = max(1, int(sample_rate * min_active_ms / 1000.0))

    spans = []
    current_start = None
    current_end = None
    for idx, is_active in enumerate(active.tolist()):
        if is_active:
            frame_start = int(starts[idx])
            frame_end = min(int(x.size), frame_start + frame_length)
            if current_start is None:
                current_start = frame_start
                current_end = frame_end
            else:
                current_end = frame_end
        elif current_start is not None:
            if current_end - current_start >= min_active_samples:
                spans.append((current_start, current_end))
            current_start = None
            current_end = None

    if current_start is not None and current_end is not None:
        if current_end - current_start >= min_active_samples:
            spans.append((current_start, current_end))

    return spans


def trim_silence_edges(
    waveform,
    sample_rate=16000,
    frame_ms=25,
    hop_ms=10,
    top_db=35.0,
    pad_ms=120,
    min_active_ms=150,
):
    """Trim leading and trailing low-energy regions from a mono waveform.

    This is inference-only preprocessing intended to reduce long pauses and
    environmental silence. It does not change training data preparation.
    """
    x = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x, {
            "trim_applied": False,
            "trim_start": 0,
            "trim_end": 0,
            "original_samples": 0,
            "trimmed_samples": 0,
        }

    frame_length = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_length = max(1, int(sample_rate * hop_ms / 1000.0))
    pad_samples = max(0, int(sample_rate * pad_ms / 1000.0))
    min_active_frames = max(1, int(round(min_active_ms / max(hop_ms, 1))))

    if x.size <= frame_length:
        return x, {
            "trim_applied": False,
            "trim_start": 0,
            "trim_end": int(x.size),
            "original_samples": int(x.size),
            "trimmed_samples": int(x.size),
        }

    _, starts, frame_length, active, peak_rms = _compute_frame_activity(
        x,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        top_db=top_db,
    )
    if peak_rms <= 1e-6:
        return x, {
            "trim_applied": False,
            "trim_start": 0,
            "trim_end": int(x.size),
            "original_samples": int(x.size),
            "trimmed_samples": int(x.size),
        }
    active_idx = np.flatnonzero(active)
    if active_idx.size < min_active_frames:
        return x, {
            "trim_applied": False,
            "trim_start": 0,
            "trim_end": int(x.size),
            "original_samples": int(x.size),
            "trimmed_samples": int(x.size),
        }

    first_frame = int(active_idx[0])
    last_frame = int(active_idx[-1])
    start_sample = max(0, int(starts[first_frame]) - pad_samples)
    end_sample = min(int(x.size), int(starts[last_frame] + frame_length) + pad_samples)

    trimmed = x[start_sample:end_sample]
    if trimmed.size < max(frame_length, int(0.2 * sample_rate)):
        trimmed = x
        start_sample = 0
        end_sample = int(x.size)
        trim_applied = False
    else:
        trim_applied = start_sample > 0 or end_sample < int(x.size)

    return trimmed.astype(np.float32, copy=False), {
        "trim_applied": bool(trim_applied),
        "trim_start": int(start_sample),
        "trim_end": int(end_sample),
        "original_samples": int(x.size),
        "trimmed_samples": int(trimmed.size),
    }


def normalize_loudness(
    waveform,
    target_rms=0.08,
    peak_limit=0.98,
    max_gain_db=18.0,
):
    """Apply simple RMS loudness normalization with gain limiting."""
    x = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x, {
            "gain_applied": 1.0,
            "rms_before": 0.0,
            "rms_after": 0.0,
            "peak_after": 0.0,
        }

    rms_before = float(np.sqrt(np.mean(x * x) + 1e-12))
    if rms_before <= 1e-6:
        return x, {
            "gain_applied": 1.0,
            "rms_before": rms_before,
            "rms_after": rms_before,
            "peak_after": float(np.max(np.abs(x))) if x.size else 0.0,
        }

    max_gain = 10.0 ** (max_gain_db / 20.0)
    gain = min(float(target_rms / max(rms_before, 1e-12)), float(max_gain))
    y = x * gain

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > peak_limit and peak > 0.0:
        y = y * (peak_limit / peak)
        gain *= peak_limit / peak

    rms_after = float(np.sqrt(np.mean(y * y) + 1e-12))
    peak_after = float(np.max(np.abs(y))) if y.size else 0.0
    return y.astype(np.float32, copy=False), {
        "gain_applied": float(gain),
        "rms_before": rms_before,
        "rms_after": rms_after,
        "peak_after": peak_after,
    }


def preprocess_waveform_for_inference(
    waveform,
    sample_rate=16000,
    trim_silence=True,
    normalize_volume=True,
):
    """Inference-only waveform preprocessing."""
    x = np.asarray(waveform, dtype=np.float32).reshape(-1)
    info = {
        "original_samples": int(x.size),
        "sample_rate": int(sample_rate),
    }

    if trim_silence:
        x, trim_info = trim_silence_edges(x, sample_rate=sample_rate)
    else:
        trim_info = {
            "trim_applied": False,
            "trim_start": 0,
            "trim_end": int(x.size),
            "original_samples": int(x.size),
            "trimmed_samples": int(x.size),
        }
    info.update(trim_info)

    if normalize_volume:
        x, norm_info = normalize_loudness(x)
    else:
        norm_info = {
            "gain_applied": 1.0,
            "rms_before": float(np.sqrt(np.mean(x * x) + 1e-12)) if x.size else 0.0,
            "rms_after": float(np.sqrt(np.mean(x * x) + 1e-12)) if x.size else 0.0,
            "peak_after": float(np.max(np.abs(x))) if x.size else 0.0,
        }
    info.update(norm_info)

    return x.astype(np.float32, copy=False), info
