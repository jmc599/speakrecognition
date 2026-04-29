import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

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
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--num-eval", type=int, default=5)
    parser.add_argument(
        "--preprocess-for-inference",
        action="store_true",
        help="Apply trim silence + loudness normalization before feature extraction.",
    )
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
    parser.add_argument(
        "--plot-path",
        type=str,
        default="",
        help="Optional PNG path for score/threshold visualization.",
    )
    return parser.parse_args()


def save_threshold_plot(metrics, plot_path):
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    scores = metrics["scores"]
    labels = metrics["labels"]
    if scores is None or labels is None:
        raise RuntimeError("Raw scores are required to draw plots.")

    target_scores = scores[labels == 1]
    non_target_scores = scores[labels == 0]
    threshold = metrics["eer_threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    bins = 60
    target_weights = np.full(
        len(target_scores),
        100.0 / max(len(target_scores), 1),
        dtype=np.float32,
    )
    non_target_weights = np.full(
        len(non_target_scores),
        100.0 / max(len(non_target_scores), 1),
        dtype=np.float32,
    )
    axes[0].hist(
        target_scores,
        bins=bins,
        weights=target_weights,
        alpha=0.6,
        label="target",
        color="#2f6fed",
    )
    axes[0].hist(
        non_target_scores,
        bins=bins,
        weights=non_target_weights,
        alpha=0.6,
        label="non-target",
        color="#ef6c3b",
    )
    axes[0].axvline(threshold, color="black", linestyle="--", linewidth=2, label="EER threshold")
    axes[0].set_title("Score Distribution")
    axes[0].set_xlabel("Cosine score")
    axes[0].set_ylabel("Percentage of trials (%)")
    axes[0].yaxis.set_major_formatter(PercentFormatter())
    axes[0].legend()

    order = metrics["curve_scores"].argsort()
    curve_scores = metrics["curve_scores"][order]
    curve_far = metrics["curve_far"][order]
    curve_frr = metrics["curve_frr"][order]
    axes[1].plot(curve_scores, curve_far, label="FAR", color="#ef6c3b")
    axes[1].plot(curve_scores, curve_frr, label="FRR", color="#2f6fed")
    axes[1].axvline(threshold, color="black", linestyle="--", linewidth=2, label="EER threshold")
    axes[1].set_title("Threshold Curve")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Error rate")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return plot_path


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
        preprocess_for_inference=args.preprocess_for_inference,
        show_progress=True,
        return_raw=bool(args.plot_path),
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
    print(f"Accuracy@EER-threshold: {metrics['accuracy'] * 100:.4f}%")
    print(f"FAR@EER-threshold: {metrics['far'] * 100:.4f}%")
    print(f"FRR@EER-threshold: {metrics['frr'] * 100:.4f}%")
    print(f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")
    print(f"Target mean score: {metrics['target_mean']:.6f}")
    print(f"Non-target mean score: {metrics['non_target_mean']:.6f}")
    if args.plot_path:
        saved_path = save_threshold_plot(metrics, args.plot_path)
        print(f"Saved plot: {saved_path}")


if __name__ == "__main__":
    main()
