from pathlib import Path
from uuid import uuid4

import numpy as np

from gui.services.audio_pipeline import normalize_audio_file
from gui.services.paths import local_tmp_dir
from utils.legacy_feature import DEFAULT_MAX_FRAMES
from utils.wespeaker_features import compute_wespeaker_fbank, validate_wespeaker_feature


def _build_legacy_feature(
    normalized_audio_path,
    max_frames=DEFAULT_MAX_FRAMES,
    num_eval=5,
):
    try:
        from export_feature_npy import build_feature
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Feature extraction dependencies are missing. "
            "Install torch/torchaudio/PyAV in the current environment."
        ) from exc

    return build_feature(
        normalized_audio_path,
        max_frames=max_frames,
        num_eval=num_eval,
        for_inference=True,
    )


def _build_wespeaker_feature(normalized_audio_path):
    feature = compute_wespeaker_fbank(normalized_audio_path)
    return validate_wespeaker_feature(feature)


def build_feature_artifacts(
    audio_path,
    prefix="feature",
    max_frames=DEFAULT_MAX_FRAMES,
    num_eval=5,
    backend_mode="local_onnx",
):
    normalized_audio_path = normalize_audio_file(audio_path, prefix=prefix)
    if backend_mode == "wespeaker_onnx":
        feature = _build_wespeaker_feature(normalized_audio_path)
    else:
        feature = _build_legacy_feature(
            normalized_audio_path,
            max_frames=max_frames,
            num_eval=num_eval,
        )

    feature_path = Path(local_tmp_dir()) / f"{prefix}_{uuid4().hex}.npy"
    np.save(feature_path, feature.astype(np.float32))
    return {
        "source_audio_path": str(audio_path),
        "normalized_audio_path": str(normalized_audio_path),
        "feature_path": str(feature_path),
        "feature_shape": tuple(feature.shape),
        "backend_mode": backend_mode,
    }
