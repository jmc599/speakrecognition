from pathlib import Path
from uuid import uuid4

import numpy as np

from gui.services.paths import local_tmp_dir
from utils.legacy_feature import LEGACY_FEATURE_SHAPE, normalize_legacy_feature_array


def _l2_normalize(x, axis=1, eps=1e-12):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _validate_model_input_shape(session):
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError("ONNX model has no inputs.")
    input_meta = inputs[0]
    shape = list(input_meta.shape)
    if len(shape) != 4:
        raise ValueError(f"Expected rank-4 ONNX input, got {shape}")

    for got, exp in zip(shape[1:], LEGACY_FEATURE_SHAPE):
        if got in (exp, str(exp), None, "None", -1):
            continue
        raise ValueError(
            f"ONNX input shape mismatch: expected [N,{LEGACY_FEATURE_SHAPE[0]},"
            f"{LEGACY_FEATURE_SHAPE[1]},{LEGACY_FEATURE_SHAPE[2]}], got {shape}. "
            "Re-export the fixed-frame ONNX model with max_frames=300."
        )
    return input_meta.name


class OnnxLocalVerifier:
    def __init__(self, config):
        self.config = config

    def test_connection(self):
        model_path = Path(self.config["local_onnx_model_path"])
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime 未安装。请先安装项目依赖。"
            ) from exc
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = _validate_model_input_shape(session)
        return {
            "mode": "local_onnx",
            "model_path": str(model_path),
            "input_name": input_name,
        }

    def embed_feature(self, feature_path, prefix="local"):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime 未安装。请先安装项目依赖。"
            ) from exc

        feature_path = Path(feature_path)
        model_path = Path(self.config["local_onnx_model_path"])
        if not feature_path.is_file():
            raise FileNotFoundError(f"Feature not found: {feature_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        features = normalize_legacy_feature_array(np.load(feature_path))

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = _validate_model_input_shape(session)
        outputs = session.run(None, {input_name: features})
        if not outputs:
            raise RuntimeError("ONNX inference returned no outputs.")

        embedding = np.asarray(outputs[0], dtype=np.float32)
        embedding = _l2_normalize(embedding, axis=1)
        embedding = embedding.mean(axis=0, keepdims=True)
        embedding = _l2_normalize(embedding, axis=1)

        local_output = Path(local_tmp_dir()) / f"{prefix}_{uuid4().hex}_embedding.npy"
        np.save(local_output, embedding.astype(np.float32))
        return {
            "embedding_path": str(local_output),
            "embedding": embedding,
            "stdout": "",
            "stderr": "",
        }
