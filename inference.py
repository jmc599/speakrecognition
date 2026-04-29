import argparse

import torch

from utils.legacy_feature import DEFAULT_MAX_FRAMES
from utils.speaker_verification import build_mel_transform, cosine_score, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Verify two real audio files.")
    parser.add_argument("audio_a", type=str, help="First audio path.")
    parser.add_argument("audio_b", type=str, help="Second audio path.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/resnet_v7_vox2ft_s1_latest.pth",
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="",
        help="Optional base path for relative audio paths.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=0,
        help="Embedding dimension. Default: infer from checkpoint.",
    )
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--num-eval", type=int, default=5)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional cosine threshold for same-speaker decision. Leave unset to print score only.",
    )
    parser.add_argument(
        "--disable-inference-preprocess",
        action="store_true",
        help="Disable inference-only preprocessing (silence trimming and loudness normalization).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_dim = args.embedding_dim or None

    model, _ = load_model(args.checkpoint, device=device, embedding_dim=embedding_dim)
    mel_transform = build_mel_transform()

    score, path_a, path_b = cosine_score(
        model,
        args.audio_a,
        args.audio_b,
        device=device,
        mel_transform=mel_transform,
        base_path=args.base_path,
        max_frames=args.max_frames,
        num_eval=args.num_eval,
        preprocess_for_inference=not args.disable_inference_preprocess,
    )

    verdict = (
        "same speaker"
        if args.threshold is not None and score >= args.threshold
        else "different speaker"
        if args.threshold is not None
        else "threshold unset"
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Audio A: {path_a}")
    print(f"Audio B: {path_b}")
    print(f"Cosine score: {score:.6f}")
    if args.threshold is None:
        print("Threshold: <unset>")
    else:
        print(f"Threshold: {args.threshold:.6f}")
    print(f"Inference preprocess: {not args.disable_inference_preprocess}")
    print(f"Decision: {verdict}")


if __name__ == "__main__":
    main()
