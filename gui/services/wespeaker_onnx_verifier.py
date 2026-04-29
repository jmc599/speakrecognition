from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from gui.services.paths import local_tmp_dir
from utils.wespeaker_features import WESPEAKER_NUM_MEL_BINS, validate_wespeaker_feature


def _l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


class WespeakerOnnxVerifier:
    def __init__(self, config: dict):
        self.config = config

    @staticmethod
    def _load_ort():
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime is not installed. Install project inference dependencies first."
            ) from exc
        return ort

    @staticmethod
    def _validate_input_meta(input_meta):
        shape = list(input_meta.shape)
        if len(shape) != 3:
            raise ValueError(
                "WeSpeaker ONNX input must be rank-3 [1, T, 80], "
                f"got shape spec: {shape}"
            )

        if shape[2] not in (WESPEAKER_NUM_MEL_BINS, str(WESPEAKER_NUM_MEL_BINS)):
            raise ValueError(
                "WeSpeaker ONNX last dimension must be 80 mel bins, "
                f"got: {shape[2]}"
            )

        if input_meta.type != "tensor(float)":
            raise ValueError(
                "WeSpeaker ONNX input dtype must be float32 tensor, "
                f"got: {input_meta.type}"
            )

    def _create_session(self, model_path: Path):
        ort = self._load_ort()
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model has no input tensors.")
        self._validate_input_meta(inputs[0])
        return session, inputs[0].name

    @staticmethod
    def _normalize_output_embedding(output: np.ndarray) -> np.ndarray:
        embedding = np.asarray(output, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding[None, :]
        if embedding.ndim != 2:
            raise ValueError(f"Expected 2D embedding [N, D], got {tuple(embedding.shape)}")
        if embedding.shape[0] != 1:
            raise ValueError(
                "Expected batch size 1 for WeSpeaker embedding output, "
                f"got {embedding.shape[0]}"
            )
        if not np.isfinite(embedding).all():
            raise ValueError("Embedding output contains NaN or Inf.")
        return _l2_normalize(embedding, axis=1)

    def test_connection(self):
        model_path = Path(self.config.get("wespeaker_onnx_model_path", ""))
        if not model_path.is_file():
            raise FileNotFoundError(f"WeSpeaker ONNX model not found: {model_path}")

        session, input_name = self._create_session(model_path)

        rng = np.random.default_rng(0)
        probe_input = rng.standard_normal((1, 100, WESPEAKER_NUM_MEL_BINS)).astype(np.float32)
        outputs = session.run(None, {input_name: probe_input})
        if not outputs:
            raise RuntimeError("ONNX inference returned no outputs during probe.")

        raw_embedding = np.asarray(outputs[0], dtype=np.float32)
        if raw_embedding.ndim == 1:
            raw_embedding = raw_embedding[None, :]
        if raw_embedding.ndim != 2 or raw_embedding.shape[0] != 1:
            raise ValueError(
                "Unexpected probe embedding shape. "
                f"Expected [1, D], got {tuple(raw_embedding.shape)}"
            )
        if not np.isfinite(raw_embedding).all():
            raise ValueError("Probe embedding contains NaN or Inf.")

        output_norm = float(np.linalg.norm(raw_embedding, axis=1)[0])
        return {
            "mode": "wespeaker_onnx",
            "model_path": str(model_path),
            "input_name": input_name,
            "probe_embedding_norm": output_norm,
            "model_has_internal_l2": abs(output_norm - 1.0) <= 0.01,
        }

    def embed_feature(self, feature_path, prefix="local"):
        feature_path = Path(feature_path)
        model_path = Path(self.config.get("wespeaker_onnx_model_path", ""))
        if not feature_path.is_file():
            raise FileNotFoundError(f"Feature not found: {feature_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"WeSpeaker ONNX model not found: {model_path}")

        feature = np.load(feature_path).astype(np.float32)
        feature = validate_wespeaker_feature(feature)

        session, input_name = self._create_session(model_path)
        outputs = session.run(None, {input_name: feature})
        if not outputs:
            raise RuntimeError("WeSpeaker ONNX inference returned no outputs.")

        raw_embedding = np.asarray(outputs[0], dtype=np.float32)
        raw_norm = float(np.linalg.norm(raw_embedding.reshape(1, -1), axis=1)[0])
        embedding = self._normalize_output_embedding(raw_embedding)

        local_output = Path(local_tmp_dir()) / f"{prefix}_{uuid4().hex}_embedding.npy"
        np.save(local_output, embedding.astype(np.float32))
        return {
            "embedding_path": str(local_output),
            "embedding": embedding,
            "stdout": "",
            "stderr": "",
            "raw_embedding_norm": raw_norm,
        }
