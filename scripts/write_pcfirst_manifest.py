import argparse
import hashlib
import json
from pathlib import Path


THRESHOLD_POLICY_CHOICES = ("eer", "far_1", "far_0p1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write a PC-First deployment manifest for the current baseline checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/resnet_v7_vox2ft_s1_latest.pth",
        help="Baseline checkpoint path.",
    )
    parser.add_argument(
        "--fixed-onnx",
        type=str,
        default="",
        help="Optional fixed-shape ONNX path. Defaults to <checkpoint_stem>_fixed.onnx.",
    )
    parser.add_argument(
        "--rknn",
        type=str,
        default="",
        help=(
            "Optional RKNN path. Defaults to <fixed_onnx_stem>_rt160_nonorm.rknn to "
            "match the RT160 fallback naming."
        ),
    )
    parser.add_argument(
        "--role",
        type=str,
        default="measured_reference",
        help="Current role label recorded in the manifest.",
    )
    parser.add_argument(
        "--threshold-policy",
        type=str,
        default="eer",
        choices=THRESHOLD_POLICY_CHOICES,
        help="Deployment threshold policy recorded in the manifest.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Fixed frame length expected for the PC reference ONNX export.",
    )
    parser.add_argument(
        "--preprocess-for-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record whether inference preprocessing is enabled in the PC-first baseline.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional manifest output path. Defaults to artifacts/manifests/<checkpoint_stem>_pcfirst_manifest.json.",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_artifact(path):
    resolved = path.resolve()
    info = {
        "path": str(path),
        "resolved_path": str(resolved),
        "exists": path.is_file(),
        "filename": path.name,
    }
    if path.is_file():
        stat = path.stat()
        info["size_bytes"] = int(stat.st_size)
        info["sha256"] = sha256_file(path)
    else:
        info["size_bytes"] = None
        info["sha256"] = None
    return info


def default_fixed_onnx_path(checkpoint_path):
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_fixed.onnx")


def default_rknn_path(fixed_onnx_path):
    return fixed_onnx_path.with_name(f"{fixed_onnx_path.stem}_rt160_nonorm.rknn")


def default_output_path(checkpoint_path):
    return Path("artifacts/manifests") / f"{checkpoint_path.stem}_pcfirst_manifest.json"


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    fixed_onnx_path = (
        Path(args.fixed_onnx) if args.fixed_onnx else default_fixed_onnx_path(checkpoint_path)
    )
    rknn_path = Path(args.rknn) if args.rknn else default_rknn_path(fixed_onnx_path)
    output_path = Path(args.output) if args.output else default_output_path(checkpoint_path)

    manifest = {
        "baseline_family": "resnet_v7_vox2ft_s1",
        "baseline_checkpoint": describe_artifact(checkpoint_path),
        "artifacts": {
            "fixed_onnx": describe_artifact(fixed_onnx_path),
            "rknn_rt160_nonorm": describe_artifact(rknn_path),
        },
        "role": str(args.role),
        "threshold_policy": str(args.threshold_policy),
        "max_frames": int(args.max_frames),
        "preprocess_for_inference": bool(args.preprocess_for_inference),
        "notes": [
            "Current PC-first baseline uses vox2ft_s1_latest directly.",
            "Final deployment freeze still depends on ONNX parity, GUI smoke, and later RKNN parity.",
            "remote_rknn_threshold must remain unset until RKNN consistency is verified.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Checkpoint exists: {manifest['baseline_checkpoint']['exists']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Fixed ONNX: {fixed_onnx_path}")
    print(f"RKNN: {rknn_path}")
    print(f"Manifest: {output_path}")


if __name__ == "__main__":
    main()
