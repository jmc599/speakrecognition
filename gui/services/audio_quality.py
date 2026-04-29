"""Local WAV quality analysis for GUI feedback.

Pure-local, no network calls. Only numpy + soundfile.
Results are advisory only — never used for inference decisions.
"""

import numpy as np
import soundfile as sf


def analyze_audio_quality(audio_path: str) -> dict:
    data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]

    duration_s = len(data) / sr if sr > 0 else 0.0

    eps = 1e-10
    rms = float(np.sqrt(np.mean(data ** 2) + eps))
    rms_db = 20.0 * np.log10(rms + eps)

    clipping_count = int(np.sum(np.abs(data) > 0.99))
    clipping_ratio = clipping_count / max(len(data), 1)

    snr_estimate_db = _estimate_snr(data, sr)

    quality_grade = _grade(duration_s, snr_estimate_db, clipping_ratio)

    return {
        "duration_s": round(duration_s, 2),
        "rms_db": round(rms_db, 1),
        "snr_estimate_db": round(snr_estimate_db, 1),
        "clipping_ratio": round(clipping_ratio, 5),
        "quality_grade": quality_grade,
    }


def _estimate_snr(data, sr):
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    if len(data) < frame_len:
        return 0.0

    eps = 1e-10
    energies = []
    for start in range(0, len(data) - frame_len + 1, hop):
        frame = data[start : start + frame_len]
        energies.append(float(np.mean(frame ** 2)))

    if not energies:
        return 0.0

    energies = np.array(energies)
    sorted_e = np.sort(energies)
    n = len(sorted_e)

    noise_floor = float(np.mean(sorted_e[: max(n // 5, 1)])) + eps
    signal_level = float(np.mean(sorted_e[-(max(n // 5, 1)) :])) + eps

    return 10.0 * np.log10(signal_level / noise_floor)


def _grade(duration_s, snr_db, clipping_ratio):
    if snr_db >= 20.0 and clipping_ratio < 0.001 and duration_s >= 3.0:
        return "A"
    if snr_db >= 15.0 and clipping_ratio < 0.01 and duration_s >= 2.0:
        return "B"
    if snr_db >= 10.0 or duration_s < 2.0:
        return "C"
    return "D"
