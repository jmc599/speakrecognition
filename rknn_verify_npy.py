import argparse
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite

from utils.legacy_feature import normalize_legacy_feature_array


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two precomputed log-mel .npy features with an RKNN speaker model."
    )
    parser.add_argument("feature_a", type=str, help="First input .npy path.")
    parser.add_argument("feature_b", type=str, help="Second input .npy path.")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/resnet_v7_vox2ft_s1_latest_fixed_rt160_nonorm.rknn",
        help="Path to the RKNN model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional cosine threshold for same-speaker decision. Leave unset to print score only.",
    )
    return parser.parse_args()


def normalize_input(array):
    return normalize_legacy_feature_array(array)


def l2_normalize(x, axis=1, eps=1e-12):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def run_embedding(rknn_lite, feature_path):
    features = normalize_input(np.load(feature_path))
    outputs = rknn_lite.inference(inputs=[features])
    if not outputs:
        raise RuntimeError(f"RKNN inference returned no outputs for {feature_path}")
    embedding = np.asarray(outputs[0], dtype=np.float32)
    embedding = l2_normalize(embedding, axis=1)
    embedding = embedding.mean(axis=0, keepdims=True)
    embedding = l2_normalize(embedding, axis=1)
    return embedding


def cosine_score(emb_a, emb_b):
    return float(np.sum(emb_a * emb_b, axis=1)[0])


def main():
    args = parse_args()
    model_path = Path(args.model)
    feature_a = Path(args.feature_a)
    feature_b = Path(args.feature_b)

    if not model_path.is_file():
        raise FileNotFoundError(f"RKNN model not found: {model_path}")
    if not feature_a.is_file():
        raise FileNotFoundError(f"Feature A not found: {feature_a}")
    if not feature_b.is_file():
        raise FileNotFoundError(f"Feature B not found: {feature_b}")

    rknn_lite = RKNNLite(verbose=True)

    print("--> Load RKNN model")
    ret = rknn_lite.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")
    print("done")

    print("--> Init runtime")
    ret = rknn_lite.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")
    print("done")

    emb_a = run_embedding(rknn_lite, feature_a)
    emb_b = run_embedding(rknn_lite, feature_b)
    score = cosine_score(emb_a, emb_b)
    verdict = (
        "same speaker" if args.threshold is not None and score >= args.threshold
        else "different speaker" if args.threshold is not None
        else "threshold unset"
    )

    print(f"Feature A: {feature_a}")
    print(f"Feature B: {feature_b}")
    print(f"Embedding shape A: {emb_a.shape}")
    print(f"Embedding shape B: {emb_b.shape}")
    print(f"Cosine score: {score:.6f}")
    if args.threshold is None:
        print("Threshold: <unset>")
    else:
        print(f"Threshold: {args.threshold:.6f}")
    print(f"Decision: {verdict}")

    rknn_lite.release()


if __name__ == "__main__":
    main()
