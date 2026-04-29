from pathlib import Path
from uuid import uuid4

import numpy as np

from gui.services.paths import local_tmp_dir
from utils.legacy_feature import normalize_legacy_feature_array


def _l2_normalize(x, axis=1, eps=1e-12):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


class PytorchLocalVerifier:
    def __init__(self, config):
        self.config = config

    def test_connection(self):
        checkpoint_path = Path(self.config["local_pytorch_checkpoint_path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"PyTorch checkpoint not found: {checkpoint_path}")
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch 未安装。请先安装项目依赖。") from exc
        return {
            "mode": "local_pytorch",
            "checkpoint_path": str(checkpoint_path),
        }

    def embed_feature(self, feature_path, prefix="local"):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch 未安装。请先安装项目依赖。") from exc

        from utils.speaker_verification import load_model

        feature_path = Path(feature_path)
        checkpoint_path = Path(self.config["local_pytorch_checkpoint_path"])
        if not feature_path.is_file():
            raise FileNotFoundError(f"Feature not found: {feature_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"PyTorch checkpoint not found: {checkpoint_path}")

        features = normalize_legacy_feature_array(np.load(feature_path))

        x = torch.from_numpy(features)
        model, _ = load_model(str(checkpoint_path), device="cpu", embedding_dim=None)
        model.eval()
        with torch.no_grad():
            embedding = model(x)
            embedding = F.normalize(embedding, dim=1)
            embedding = embedding.mean(dim=0, keepdim=True)
            embedding = F.normalize(embedding, dim=1)

        embedding = embedding.cpu().numpy().astype(np.float32)
        embedding = _l2_normalize(embedding, axis=1)
        local_output = Path(local_tmp_dir()) / f"{prefix}_{uuid4().hex}_embedding.npy"
        np.save(local_output, embedding.astype(np.float32))
        return {
            "embedding_path": str(local_output),
            "embedding": embedding,
            "stdout": "",
            "stderr": "",
        }
