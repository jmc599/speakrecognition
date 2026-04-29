import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch, ONNX, and RKNN embeddings for consistency."
    )
    parser.add_argument(
        "--pytorch",
        type=str,
        default="embedding_pytorch.npy",
        help="Path to the PyTorch embedding .npy file.",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="embedding_onnx.npy",
        help="Path to the ONNX embedding .npy file.",
    )
    parser.add_argument(
        "--rknn",
        type=str,
        default="embedding_rknn.npy",
        help="Path to the RKNN embedding .npy file.",
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=0.999,
        help="Minimum cosine similarity required to consider two embeddings aligned.",
    )
    parser.add_argument(
        "--max-l2",
        type=float,
        default=1e-3,
        help="Maximum L2 distance required to consider two embeddings aligned.",
    )
    return parser.parse_args()


def load_vector(path):
    array = np.load(path).astype(np.float32).reshape(-1)
    return array


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def l2(a, b):
    return float(np.linalg.norm(a - b))


def main():
    args = parse_args()
    paths = {
        "pytorch": Path(args.pytorch),
        "onnx": Path(args.onnx),
        "rknn": Path(args.rknn),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} embedding not found: {path}")

    pt = load_vector(paths["pytorch"])
    ox = load_vector(paths["onnx"])
    rk = load_vector(paths["rknn"])

    pairs = [
        ("pt", "onnx", pt, ox),
        ("pt", "rknn", pt, rk),
        ("onnx", "rknn", ox, rk),
    ]

    print(f"pt norm: {float(np.linalg.norm(pt)):.9f}")
    print(f"onnx norm: {float(np.linalg.norm(ox)):.9f}")
    print(f"rknn norm: {float(np.linalg.norm(rk)):.9f}")

    overall_ok = True
    for left_name, right_name, left, right in pairs:
        cos_value = cosine(left, right)
        l2_value = l2(left, right)
        pair_ok = cos_value >= args.min_cosine and l2_value <= args.max_l2
        overall_ok = overall_ok and pair_ok
        print(
            f"{left_name} vs {right_name}: "
            f"cos={cos_value:.9f} l2={l2_value:.9f} "
            f"status={'OK' if pair_ok else 'MISMATCH'}"
        )

    print(f"Consistency: {'PASS' if overall_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
