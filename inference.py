import argparse

import torch

from utils.speaker_verification import build_mel_transform, cosine_score, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Verify two real audio files.")
    parser.add_argument("audio_a", type=str, help="First audio path.")
    parser.add_argument("audio_b", type=str, help="Second audio path.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/resnet_v7_best.pth",
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
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--num-eval", type=int, default=5)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.22145,
        help="Cosine threshold for same-speaker decision. Default is the current full-eval threshold for resnet_v7.",
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
    )

    verdict = "same speaker" if score >= args.threshold else "different speaker"
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Audio A: {path_a}")
    print(f"Audio B: {path_b}")
    print(f"Cosine score: {score:.6f}")
    print(f"Threshold: {args.threshold:.4f}")
    print(f"Decision: {verdict}")


if __name__ == "__main__":
    main()
