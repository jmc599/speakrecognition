import argparse
import csv
import json
from pathlib import Path

import torch

from utils.checkpoint_io import load_torch_checkpoint
from utils.legacy_feature import DEFAULT_MAX_FRAMES
from utils.speaker_verification import build_mel_transform, load_model
from utils.verification_eval import evaluate_trial_list, load_enroll_map, sample_trials


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate saved checkpoints against full-trial official evaluation."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint path. Pass multiple times for best/latest comparisons.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="",
        help="Optional per-epoch metrics CSV used to verify monitor alignment.",
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
        help="Enrollment mapping list path.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=0,
        help="Embedding dimension override. Default: infer from checkpoint.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Frame length used for official evaluation.",
    )
    parser.add_argument("--num-eval", type=int, default=5)
    parser.add_argument(
        "--preprocess-for-inference",
        action="store_true",
        help="Apply inference preprocessing before embedding extraction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Randomly sample N trials instead of evaluating the full list.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="balanced",
        choices=["balanced", "random"],
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional JSON summary path.",
    )
    parser.add_argument(
        "--score-norm",
        type=str,
        default="none",
        choices=["none", "asnorm", "snorm"],
        help="Optional score normalization backend for full-trial scoring.",
    )
    parser.add_argument(
        "--cohort-top-k",
        type=int,
        default=300,
        help="Top-K cohort size used by AS-Norm. Ignored for none/S-Norm.",
    )
    return parser.parse_args()


def load_metrics_rows(metrics_csv):
    if not metrics_csv:
        return {}
    path = Path(metrics_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")

    rows_by_epoch = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = row.get("epoch")
            if epoch in (None, ""):
                continue
            rows_by_epoch[int(epoch)] = row
    return rows_by_epoch


def safe_float(value):
    if value in (None, ""):
        return None
    return float(value)


def summarize_checkpoint(
    checkpoint_path,
    metrics_row_by_epoch,
    *,
    device,
    mel_transform,
    trials,
    base_path,
    embedding_dim,
    max_frames,
    num_eval,
    preprocess_for_inference,
    score_norm,
    cohort_top_k,
):
    raw_checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    if not isinstance(raw_checkpoint, dict):
        raise RuntimeError(f"Expected full checkpoint dict: {checkpoint_path}")

    saved_epoch = int(raw_checkpoint.get("epoch", 0))
    saved_best_metric = safe_float(raw_checkpoint.get("best_metric"))
    checkpoint_monitor = raw_checkpoint.get("verification_metrics") or {}
    csv_row = metrics_row_by_epoch.get(saved_epoch)

    model, _ = load_model(
        checkpoint_path,
        device=device,
        embedding_dim=embedding_dim or None,
    )
    full_metrics = evaluate_trial_list(
        model,
        device=device,
        mel_transform=mel_transform,
        base_path=base_path,
        trials=trials,
        max_frames=max_frames,
        num_eval=num_eval,
        preprocess_for_inference=preprocess_for_inference,
        show_progress=True,
        return_raw=False,
        score_norm=score_norm,
        cohort_top_k=cohort_top_k,
    )

    checkpoint_monitor_eer = safe_float(checkpoint_monitor.get("eer"))
    csv_monitor_eer = safe_float(csv_row.get("eer")) if csv_row else None
    summary = {
        "checkpoint": str(checkpoint_path),
        "saved_epoch": saved_epoch,
        "saved_best_metric": saved_best_metric,
        "checkpoint_monitor_eer": checkpoint_monitor_eer,
        "checkpoint_monitor_min_dcf": safe_float(checkpoint_monitor.get("min_dcf")),
        "csv_monitor_eer": csv_monitor_eer,
        "csv_monitor_min_dcf": safe_float(csv_row.get("min_dcf")) if csv_row else None,
        "csv_selected_metric": safe_float(csv_row.get("selected_metric")) if csv_row else None,
        "full_trial_num_trials": int(full_metrics["num_trials"]),
        "full_trial_eer": float(full_metrics["eer"]),
        "full_trial_eer_threshold": float(full_metrics["eer_threshold"]),
        "full_trial_min_dcf": float(full_metrics["min_dcf"]),
        "full_trial_threshold_far_1": float(full_metrics["threshold_far_1"]),
        "full_trial_frr_at_far_1": float(full_metrics["frr_at_far_1"]),
        "full_trial_threshold_far_0p1": float(full_metrics["threshold_far_0p1"]),
        "full_trial_frr_at_far_0p1": float(full_metrics["frr_at_far_0p1"]),
        "full_trial_target_mean": float(full_metrics["target_mean"]),
        "full_trial_non_target_mean": float(full_metrics["non_target_mean"]),
        "score_norm": str(full_metrics.get("score_norm", score_norm)),
        "cohort_top_k": int(full_metrics.get("cohort_top_k", cohort_top_k)),
        "monitor_delta_from_checkpoint": (
            None
            if checkpoint_monitor_eer is None
            else float(full_metrics["eer"]) - checkpoint_monitor_eer
        ),
        "monitor_delta_from_csv": (
            None if csv_monitor_eer is None else float(full_metrics["eer"]) - csv_monitor_eer
        ),
    }
    return summary


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics_rows = load_metrics_rows(args.metrics_csv)
    enroll_map = load_enroll_map(args.enroll_list)
    trials = sample_trials(
        args.trials,
        enroll_map=enroll_map,
        limit=args.limit,
        seed=args.seed,
        sample_mode=args.sample_mode,
    )
    if not trials:
        raise RuntimeError("No trials found for evaluation.")

    mel_transform = build_mel_transform()
    summaries = []
    for checkpoint in args.checkpoint:
        summary = summarize_checkpoint(
            checkpoint,
            metrics_rows,
            device=device,
            mel_transform=mel_transform,
            trials=trials,
            base_path=args.base_path,
            embedding_dim=args.embedding_dim,
            max_frames=args.max_frames,
            num_eval=args.num_eval,
            preprocess_for_inference=args.preprocess_for_inference,
            score_norm=args.score_norm,
            cohort_top_k=args.cohort_top_k,
        )
        summaries.append(summary)
        print(f"Checkpoint: {summary['checkpoint']}")
        print(
            f"Score norm: {summary['score_norm']}"
            + (
                f" (top_k={summary['cohort_top_k']})"
                if summary["score_norm"] == "asnorm"
                else ""
            )
        )
        print(f"Saved epoch: {summary['saved_epoch']}")
        print(f"Full-trial EER: {summary['full_trial_eer'] * 100:.4f}%")
        print(f"Full-trial minDCF@0.01: {summary['full_trial_min_dcf']:.6f}")
        print(
            f"FRR@FAR<=1%: {summary['full_trial_frr_at_far_1'] * 100:.4f}% "
            f"(threshold={summary['full_trial_threshold_far_1']:.6f})"
        )
        print(
            f"FRR@FAR<=0.1%: {summary['full_trial_frr_at_far_0p1'] * 100:.4f}% "
            f"(threshold={summary['full_trial_threshold_far_0p1']:.6f})"
        )
        if summary["checkpoint_monitor_eer"] is not None:
            print(
                "Monitor delta vs checkpoint: "
                f"{summary['monitor_delta_from_checkpoint'] * 100:.4f} pp"
            )
        if summary["csv_monitor_eer"] is not None:
            print(
                "Monitor delta vs CSV: "
                f"{summary['monitor_delta_from_csv'] * 100:.4f} pp"
            )
        print("-" * 60)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"Saved summary: {output_path}")


if __name__ == "__main__":
    main()
