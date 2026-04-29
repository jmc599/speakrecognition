import argparse
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite

from utils.legacy_feature import (
    legacy_feature_shape_text,
    normalize_legacy_feature_array,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RKNN inference from a precomputed .npy feature tensor."
    )
    parser.add_argument(
        "feature",
        type=str,
        help=f"Path to an input .npy file with shape {legacy_feature_shape_text()} .",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/resnet_v7_vox2ft_s1_latest_fixed_rt160_nonorm.rknn",
        help="Path to the RKNN model.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="",
        help="Optional RKNN target platform. Leave empty when running on the board itself.",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default="",
        help="Optional path to save the output embedding as .npy.",
    )
    return parser.parse_args()


def normalize_input(array):
    return normalize_legacy_feature_array(array)


def main():
    args = parse_args()
    model_path = Path(args.model)
    feature_path = Path(args.feature)

    if not model_path.is_file():
        raise FileNotFoundError(f"RKNN model not found: {model_path}")
    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    features = normalize_input(np.load(feature_path))
    print(f"Feature shape: {features.shape}, dtype: {features.dtype}")

    rknn_lite = RKNNLite(verbose=True)

    print("--> Load RKNN model")
    ret = rknn_lite.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")
    print("done")

    print("--> Init runtime")
    if args.target:
        ret = rknn_lite.init_runtime(target=args.target)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")
    print("done")

    print("--> Run inference")
    outputs = rknn_lite.inference(inputs=[features])
    if not outputs:
        raise RuntimeError("RKNN inference returned no outputs.")
    output = np.asarray(outputs[0], dtype=np.float32)
    print("done")
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")

    if args.save_output:
        save_path = Path(args.save_output)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, output)
        print(f"Saved output to: {save_path}")

    rknn_lite.release()


if __name__ == "__main__":
    main()
