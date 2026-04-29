import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.legacy_feature import (
    legacy_feature_shape_text,
    normalize_legacy_feature_array,
)
from utils.speaker_verification import load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a PyTorch embedding from a precomputed feature .npy file."
    )
    parser.add_argument(
        "feature",
        type=str,
        help=f"Input .npy path with shape {legacy_feature_shape_text()}.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/resnet_v7_vox2ft_s1_latest.pth",
        help="Path to the PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="embedding_pytorch.npy",
        help="Output embedding .npy path.",
    )
    return parser.parse_args()


def normalize_input(array):
    return normalize_legacy_feature_array(array)


def main():
    args = parse_args()
    feature_path = Path(args.feature)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)

    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature not found: {feature_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    features = normalize_input(np.load(feature_path))
    x = torch.from_numpy(features)

    model, _ = load_model(str(checkpoint_path), device="cpu", embedding_dim=None)
    with torch.no_grad():
        emb = model(x)
        emb = F.normalize(emb, dim=1)
        emb = emb.mean(dim=0, keepdim=True)
        emb = F.normalize(emb, dim=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, emb.cpu().numpy().astype(np.float32))

    print(f"Feature: {feature_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Embedding shape: {tuple(emb.shape)}")
    print(f"Embedding dtype: {emb.dtype}")


if __name__ == "__main__":
    main()
