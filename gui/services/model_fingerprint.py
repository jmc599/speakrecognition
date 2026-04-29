from __future__ import annotations

import hashlib
from pathlib import Path


def _file_signature(path_text: str) -> str:
    if not path_text:
        return "path=<empty>"

    path = Path(path_text)
    if not path.is_file():
        return f"path={path.resolve()}|missing=1"

    stat = path.stat()
    return (
        f"path={path.resolve()}|size={int(stat.st_size)}|"
        f"mtime_ns={int(getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1e9)))}"
    )


def resolve_backend_model_fingerprint(config: dict) -> str:
    backend = str(config.get("backend_mode", "local_onnx"))

    if backend == "local_onnx":
        signature = _file_signature(str(config.get("local_onnx_model_path", "")))
    elif backend == "local_pytorch":
        signature = _file_signature(str(config.get("local_pytorch_checkpoint_path", "")))
    elif backend == "wespeaker_onnx":
        signature = _file_signature(str(config.get("wespeaker_onnx_model_path", "")))
    elif backend == "remote_rknn":
        signature = (
            f"remote_model={config.get('remote_model_path', '')}|"
            f"remote_script={config.get('remote_runner_script') or config.get('remote_embed_script', '')}|"
            f"remote_python={config.get('remote_python', '')}"
        )
    else:
        signature = f"backend={backend}"

    digest = hashlib.sha256(f"{backend}|{signature}".encode("utf-8")).hexdigest()
    return f"{backend}:{digest}"
