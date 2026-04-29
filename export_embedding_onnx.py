import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from utils.legacy_feature import (
    legacy_feature_shape_text,
    normalize_legacy_feature_array,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export an ONNX embedding from a precomputed feature .npy file."
    )
    parser.add_argument(
        "feature",
        type=str,
        help=f"Input .npy path with shape {legacy_feature_shape_text()}.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/resnet_v7_vox2ft_s1_latest_fixed.onnx",
        help="Path to the ONNX model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="embedding_onnx.npy",
        help="Output embedding .npy path.",
    )
    return parser.parse_args()


def normalize_input(array):
    return normalize_legacy_feature_array(array)


def l2_normalize(x, axis=1, eps=1e-12):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def main():
    args = parse_args()
    feature_path = Path(args.feature)
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature not found: {feature_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    features = normalize_input(np.load(feature_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: features})
    emb = np.asarray(outputs[0], dtype=np.float32)
    emb = l2_normalize(emb, axis=1)
    emb = emb.mean(axis=0, keepdims=True)
    emb = l2_normalize(emb, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, emb.astype(np.float32))

    print(f"Feature: {feature_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Embedding shape: {emb.shape}")
    print(f"Embedding dtype: {emb.dtype}")


if __name__ == "__main__":
    main()
