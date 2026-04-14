import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.detection_metrics import build_calibration_summary
from utils.legacy_feature import (
    DEFAULT_MAX_FRAMES,
    legacy_feature_shape_text,
    normalize_legacy_feature_array,
)
from utils.score_norm import apply_symmetric_score_norm, compute_cohort_stats
from utils.speaker_verification import load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a PyTorch checkpoint with profile-based verification."
    )
    parser.add_argument(
        "--feature-root",
        type=str,
        required=True,
        help="Root directory containing exported feature .npy files organized by speaker.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/resnet_v7_best.pth",
        help="Path to the PyTorch checkpoint.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=0,
        help="Embedding dimension override. Default: infer from checkpoint.",
    )
    parser.add_argument("--same-trials", type=int, default=1000)
    parser.add_argument("--diff-trials", type=int, default=1000)
    parser.add_argument("--enroll-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-results",
        type=str,
        default="artifacts/results/results_pytorch_profile.csv",
        help="CSV path for per-trial scores.",
    )
    parser.add_argument(
        "--save-calibration",
        type=str,
        default="",
        help="Optional JSON path for calibration output.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Frame length used to produce the features.",
    )
    parser.add_argument(
        "--preprocess-for-inference",
        action="store_true",
        help="Record that inference preprocessing was enabled when producing the features.",
    )
    parser.add_argument(
        "--write-gui-config",
        action="store_true",
        help="Write the calibrated FAR<=1%% threshold back to the local PyTorch GUI config.",
    )
    parser.add_argument(
        "--score-norm",
        type=str,
        default="none",
        choices=["none", "asnorm", "snorm"],
        help="Optional score normalization backend for profile scoring.",
    )
    parser.add_argument(
        "--cohort-top-k",
        type=int,
        default=300,
        help="Top-K cohort size used by AS-Norm. Ignored for none/S-Norm.",
    )
    return parser.parse_args()


def l2_normalize(x, axis=1, eps=1e-12):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def normalize_input(array):
    return normalize_legacy_feature_array(array)


def collect_features(feature_root):
    by_speaker = defaultdict(list)
    for path in Path(feature_root).rglob("*.npy"):
        by_speaker[path.parent.name].append(path)
    return {speaker_id: sorted(paths) for speaker_id, paths in by_speaker.items()}


def sample_same_trials(by_speaker, count, enroll_count, rng):
    speaker_ids = [
        speaker_id
        for speaker_id, paths in by_speaker.items()
        if len(paths) >= enroll_count + 1
    ]
    if not speaker_ids:
        raise RuntimeError("No speakers with enough feature files for same-speaker trials.")

    trials = []
    for _ in range(count):
        speaker_id = rng.choice(speaker_ids)
        paths = rng.sample(by_speaker[speaker_id], enroll_count + 1)
        trials.append(
            {
                "label": 1,
                "speaker_key": speaker_id,
                "enroll_paths": paths[:enroll_count],
                "verify_path": paths[-1],
            }
        )
    return trials


def sample_diff_trials(by_speaker, count, enroll_count, rng):
    speaker_ids = [
        speaker_id
        for speaker_id, paths in by_speaker.items()
        if len(paths) >= enroll_count
    ]
    if len(speaker_ids) < 2:
        raise RuntimeError("Need at least two speakers for different-speaker profile trials.")

    trials = []
    for _ in range(count):
        enroll_speaker, verify_speaker = rng.sample(speaker_ids, 2)
        trials.append(
            {
                "label": 0,
                "speaker_key": f"{enroll_speaker}|{verify_speaker}",
                "enroll_paths": rng.sample(by_speaker[enroll_speaker], enroll_count),
                "verify_path": rng.choice(by_speaker[verify_speaker]),
            }
        )
    return trials


def run_embedding(model, feature_path, cache, device):
    feature_key = str(feature_path)
    if feature_key in cache:
        return cache[feature_key]

    features = normalize_input(np.load(feature_path))
    x = torch.from_numpy(features).to(device)
    with torch.no_grad():
        embedding = model(x)
        embedding = F.normalize(embedding, dim=1)
        embedding = embedding.mean(dim=0, keepdim=True)
        embedding = F.normalize(embedding, dim=1)
    embedding = embedding.cpu().numpy().astype(np.float32)
    embedding = l2_normalize(embedding, axis=1)
    cache[feature_key] = embedding
    return embedding


def cosine_score(emb_a, emb_b):
    return float(np.sum(emb_a * emb_b, axis=1)[0])


def build_profile(embeddings):
    merged = np.concatenate(embeddings, axis=0)
    merged = l2_normalize(merged, axis=1)
    merged = merged.mean(axis=0, keepdims=True)
    merged = l2_normalize(merged, axis=1)
    return merged


def build_cohort_cache(by_speaker, model, cache, device):
    feature_paths = []
    for paths in by_speaker.values():
        feature_paths.extend(paths)
    feature_paths = sorted(feature_paths)

    embeddings = []
    path_to_index = {}
    for idx, feature_path in enumerate(feature_paths):
        embeddings.append(run_embedding(model, feature_path, cache, device))
        path_to_index[str(feature_path)] = idx
    cohort_embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
    return cohort_embeddings, path_to_index


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    feature_root = Path(args.feature_root)
    checkpoint_path = Path(args.checkpoint)
    results_path = Path(args.save_results)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not feature_root.is_dir():
        raise FileNotFoundError(f"Feature root not found: {feature_root}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    by_speaker = collect_features(feature_root)
    if not by_speaker:
        raise RuntimeError(
            f"No feature files found under: {feature_root}. "
            f"Expected shape {legacy_feature_shape_text()}."
        )

    model, _ = load_model(
        str(checkpoint_path),
        device=device,
        embedding_dim=args.embedding_dim or None,
    )
    cache = {}
    cohort_embeddings = None
    cohort_path_to_index = {}
    if args.score_norm != "none":
        cohort_embeddings, cohort_path_to_index = build_cohort_cache(
            by_speaker,
            model,
            cache,
            device,
        )

    same_trials = sample_same_trials(by_speaker, args.same_trials, args.enroll_count, rng)
    diff_trials = sample_diff_trials(by_speaker, args.diff_trials, args.enroll_count, rng)
    trials = same_trials + diff_trials
    rng.shuffle(trials)

    rows = []
    for idx, trial in enumerate(trials, start=1):
        enroll_embeddings = [
            run_embedding(model, enroll_path, cache, device)
            for enroll_path in trial["enroll_paths"]
        ]
        profile_embedding = build_profile(enroll_embeddings)
        verify_embedding = run_embedding(model, trial["verify_path"], cache, device)
        raw_score = cosine_score(profile_embedding, verify_embedding)
        score = raw_score
        if args.score_norm != "none":
            blocked = set()
            for enroll_path in trial["enroll_paths"]:
                blocked_idx = cohort_path_to_index.get(str(enroll_path))
                if blocked_idx is not None:
                    blocked.add(blocked_idx)
            verify_idx = cohort_path_to_index.get(str(trial["verify_path"]))
            if verify_idx is not None:
                blocked.add(verify_idx)

            profile_mean, profile_std = compute_cohort_stats(
                profile_embedding,
                cohort_embeddings,
                mode=args.score_norm,
                top_k=args.cohort_top_k,
                exclude_indices=[blocked],
            )
            verify_mean, verify_std = compute_cohort_stats(
                verify_embedding,
                cohort_embeddings,
                mode=args.score_norm,
                top_k=args.cohort_top_k,
                exclude_indices=[blocked],
            )
            score = apply_symmetric_score_norm(
                raw_score,
                profile_mean[0],
                profile_std[0],
                verify_mean[0],
                verify_std[0],
            )
        rows.append(
            {
                "index": idx,
                "label": trial["label"],
                "score": f"{score:.6f}",
                "raw_score": f"{raw_score:.6f}",
                "speaker_key": trial["speaker_key"],
                "enroll_paths": "|".join(str(path) for path in trial["enroll_paths"]),
                "verify_path": str(trial["verify_path"]),
            }
        )
        if idx % 100 == 0 or idx == len(trials):
            print(f"Processed {idx}/{len(trials)}")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "label",
                "score",
                "raw_score",
                "speaker_key",
                "enroll_paths",
                "verify_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float32)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    calibration = build_calibration_summary(
        scores,
        labels,
        checkpoint=str(checkpoint_path),
        max_frames=args.max_frames,
        preprocess_for_inference=args.preprocess_for_inference,
    )
    calibration["results"] = str(results_path)
    calibration["score_norm"] = args.score_norm
    calibration["cohort_top_k"] = int(args.cohort_top_k)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(
        f"Score norm: {args.score_norm}"
        + (f" (top_k={args.cohort_top_k})" if args.score_norm == "asnorm" else "")
    )
    print(f"Trials: {len(rows)}")
    print(f"Unique features embedded: {len(cache)}")
    print(f"Results: {results_path}")
    print(f"EER: {calibration['eer'] * 100:.4f}%")
    print(f"minDCF@0.01: {calibration['min_dcf_p01']:.6f}")
    print(
        f"FRR@FAR<=1%: {calibration['frr_at_far_1'] * 100:.4f}% "
        f"(threshold={calibration['threshold_far_1']:.6f})"
    )
    print(
        f"FRR@FAR<=0.1%: {calibration['frr_at_far_0p1'] * 100:.4f}% "
        f"(threshold={calibration['threshold_far_0p1']:.6f})"
    )

    if args.save_calibration:
        calibration_path = Path(args.save_calibration)
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        with calibration_path.open("w", encoding="utf-8") as f:
            json.dump(calibration, f, ensure_ascii=False, indent=2)
        print(f"Calibration: {calibration_path}")

    if args.write_gui_config:
        from gui.services.config_store import ConfigStore

        store = ConfigStore()
        config = store.load()
        config["backend_mode"] = "local_pytorch"
        config["local_pytorch_checkpoint_path"] = str(checkpoint_path)
        config["local_pytorch_threshold"] = float(calibration["threshold_far_1"])
        store.save(config)
        print(
            "GUI config updated: "
            "backend_mode=local_pytorch, "
            f"local_pytorch_checkpoint_path={checkpoint_path}, "
            f"local_pytorch_threshold={calibration['threshold_far_1']:.6f}"
        )


if __name__ == "__main__":
    main()
