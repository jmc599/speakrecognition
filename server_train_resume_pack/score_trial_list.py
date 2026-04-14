import argparse
import csv
from pathlib import Path

import torch

from utils.legacy_feature import DEFAULT_MAX_FRAMES
from utils.speaker_verification import build_mel_transform, load_model
from utils.verification_eval import evaluate_trial_list, load_enroll_map, sample_trials


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score a raw-audio trial list with a PyTorch checkpoint and save per-trial scores."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base-path", type=str, required=True)
    parser.add_argument("--trials", type=str, required=True)
    parser.add_argument("--enroll-list", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--embedding-dim", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--num-eval", type=int, default=5)
    parser.add_argument("--preprocess-for-inference", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="balanced",
        choices=["balanced", "random"],
    )
    parser.add_argument(
        "--score-norm",
        type=str,
        default="none",
        choices=["none", "asnorm", "snorm"],
    )
    parser.add_argument("--cohort-top-k", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path(args.checkpoint)
    trials_path = Path(args.trials)
    enroll_list_path = Path(args.enroll_list)
    output_path = Path(args.output)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not trials_path.is_file():
        raise FileNotFoundError(f"Trials not found: {trials_path}")
    if not enroll_list_path.is_file():
        raise FileNotFoundError(f"Enroll list not found: {enroll_list_path}")

    enroll_map = load_enroll_map(str(enroll_list_path))
    trials = sample_trials(
        str(trials_path),
        enroll_map=enroll_map,
        limit=args.limit,
        seed=args.seed,
        sample_mode=args.sample_mode,
    )
    if not trials:
        raise RuntimeError("No trials found.")

    model, _ = load_model(
        str(checkpoint_path),
        device=device,
        embedding_dim=args.embedding_dim or None,
    )
    mel_transform = build_mel_transform()
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
        return_raw=True,
        score_norm=args.score_norm,
        cohort_top_k=args.cohort_top_k,
    )

    scores = metrics["scores"]
    labels = metrics["labels"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "label", "audio_a", "audio_b", "score"],
        )
        writer.writeheader()
        for idx, ((audio_a, audio_b, _), score, label) in enumerate(
            zip(trials, scores, labels), start=1
        ):
            writer.writerow(
                {
                    "index": idx,
                    "label": int(label),
                    "audio_a": str(audio_a),
                    "audio_b": str(audio_b),
                    "score": float(score),
                }
            )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Trials: {len(trials)}")
    print(f"Results: {output_path}")
    print(f"EER: {metrics['eer'] * 100:.4f}%")
    print(f"minDCF@0.01: {metrics['min_dcf']:.6f}")
    print(
        f"FRR@FAR<=1%: {metrics['frr_at_far_1'] * 100:.4f}% "
        f"(threshold={metrics['threshold_far_1']:.6f})"
    )


if __name__ == "__main__":
    main()
