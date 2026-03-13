import argparse

import torch

from utils.speaker_verification import build_mel_transform, load_model
from utils.verification_eval import (
    evaluate_trial_list,
    load_enroll_map,
    sample_trials,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate speaker verification trials.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/cnceleb_best.pth",
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=r"F:\cn_celeb\cn-celeb_v2\CN-Celeb_flac\eval",
        help="Base folder for trial audio paths.",
    )
    parser.add_argument(
        "--trials",
        type=str,
        default=r"F:\cn_celeb\cn-celeb_v2\CN-Celeb_flac\eval\lists\trials.lst",
        help="Trial list path.",
    )
    parser.add_argument(
        "--enroll-list",
        type=str,
        default=r"F:\cn_celeb\cn-celeb_v2\CN-Celeb_flac\eval\lists\enroll.lst",
        help="Optional mapping from enrollment key to audio path.",
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
        "--limit",
        type=int,
        default=0,
        help="Randomly sample N trials. 0 means evaluate all trials.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling trials.",
    )
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="balanced",
        choices=["balanced", "random"],
        help="Sampling mode used when --limit > 0.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_dim = args.embedding_dim or None

    model, _ = load_model(args.checkpoint, device=device, embedding_dim=embedding_dim)
    mel_transform = build_mel_transform()
    enroll_map = load_enroll_map(args.enroll_list)
    trials = sample_trials(
        args.trials,
        enroll_map=enroll_map,
        limit=args.limit,
        seed=args.seed,
        sample_mode=args.sample_mode,
    )

    if not trials:
        raise RuntimeError("No trials found.")

    metrics = evaluate_trial_list(
        model,
        device=device,
        mel_transform=mel_transform,
        base_path=args.base_path,
        trials=trials,
        max_frames=args.max_frames,
        num_eval=args.num_eval,
        show_progress=True,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Trials: {metrics['num_trials']}")
    print(f"Unique audio: {metrics['num_unique_audio']}")
    print(f"Target trials: {metrics['num_target_trials']}")
    print(f"Non-target trials: {metrics['num_non_target_trials']}")
    print(f"EER: {metrics['eer'] * 100:.4f}%")
    print(f"EER threshold: {metrics['eer_threshold']:.6f}")
    print(f"minDCF@0.01: {metrics['min_dcf']:.6f}")
    print(f"Target mean score: {metrics['target_mean']:.6f}")
    print(f"Non-target mean score: {metrics['non_target_mean']:.6f}")


if __name__ == "__main__":
    main()
