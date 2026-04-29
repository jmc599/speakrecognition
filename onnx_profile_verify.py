import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort

from utils.detection_metrics import THRESHOLD_POLICY_CHOICES, build_calibration_summary
from utils.legacy_feature import DEFAULT_MAX_FRAMES, normalize_legacy_feature_array


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate local ONNX threshold for profile-based verification."
    )
    parser.add_argument(
        "--feature-root",
        type=str,
        required=True,
        help="Root directory containing exported feature .npy files organized by speaker.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/resnet_v7_vox2ft_s1_latest_fixed.onnx",
        help="Path to ONNX model.",
    )
    parser.add_argument(
        "--same-trials",
        type=int,
        default=1000,
        help="Number of same-speaker profile verification trials.",
    )
    parser.add_argument(
        "--diff-trials",
        type=int,
        default=1000,
        help="Number of different-speaker profile verification trials.",
    )
    parser.add_argument(
        "--enroll-count",
        type=int,
        default=3,
        help="Number of enrollment samples per profile.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default="artifacts/results/results_onnx_profile.csv",
        help="CSV path to save profile verification scores.",
    )
    parser.add_argument(
        "--save-calibration",
        type=str,
        default="",
        help="Optional JSON path for calibration output.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional checkpoint identifier stored in the calibration output.",
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
        help="Write the calibrated selected threshold back to the local ONNX GUI config.",
    )
    parser.add_argument(
        "--threshold-policy",
        type=str,
        default="eer",
        choices=THRESHOLD_POLICY_CHOICES,
        help="Operating point policy used for selected_threshold and GUI writes.",
    )
    return parser.parse_args()


def l2_normalize(x, axis=1, eps=1e-12):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def normalize_input(array):
    return normalize_legacy_feature_array(array)


def run_embedding(session, input_name, feature_path, cache):
    feature_key = str(feature_path)
    if feature_key in cache:
        return cache[feature_key]

    features = normalize_input(np.load(feature_path))
    outputs = session.run(None, {input_name: features})
    if not outputs:
        raise RuntimeError(f"ONNX inference returned no outputs for {feature_path}")

    embedding = np.asarray(outputs[0], dtype=np.float32)
    embedding = l2_normalize(embedding, axis=1)
    embedding = embedding.mean(axis=0, keepdims=True)
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


def collect_features(feature_root):
    by_speaker = defaultdict(list)
    for path in Path(feature_root).rglob("*.npy"):
        speaker_id = path.parent.name
        by_speaker[speaker_id].append(path)
    return {speaker_id: sorted(paths) for speaker_id, paths in by_speaker.items()}


def sample_same_trials(by_speaker, count, enroll_count, rng):
    speaker_ids = [speaker_id for speaker_id, paths in by_speaker.items() if len(paths) >= enroll_count + 1]
    if not speaker_ids:
        raise RuntimeError("No speakers with enough feature files for same-speaker trials.")

    trials = []
    for _ in range(count):
        speaker_id = rng.choice(speaker_ids)
        paths = rng.sample(by_speaker[speaker_id], enroll_count + 1)
        enroll_paths = paths[:enroll_count]
        verify_path = paths[-1]
        trials.append(
            {
                "label": 1,
                "speaker_key": speaker_id,
                "enroll_paths": enroll_paths,
                "verify_path": verify_path,
            }
        )
    return trials


def sample_diff_trials(by_speaker, count, enroll_count, rng):
    speaker_ids = [speaker_id for speaker_id, paths in by_speaker.items() if len(paths) >= enroll_count]
    if len(speaker_ids) < 2:
        raise RuntimeError("Need at least two speakers for different-speaker trials.")

    trials = []
    for _ in range(count):
        enroll_speaker, verify_speaker = rng.sample(speaker_ids, 2)
        enroll_paths = rng.sample(by_speaker[enroll_speaker], enroll_count)
        verify_path = rng.choice(by_speaker[verify_speaker])
        trials.append(
            {
                "label": 0,
                "speaker_key": f"{enroll_speaker}|{verify_speaker}",
                "enroll_paths": enroll_paths,
                "verify_path": verify_path,
            }
        )
    return trials


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    feature_root = Path(args.feature_root)
    model_path = Path(args.model)
    results_path = Path(args.save_results)

    if not feature_root.is_dir():
        raise FileNotFoundError(f"Feature root not found: {feature_root}")
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    by_speaker = collect_features(feature_root)
    if not by_speaker:
        raise RuntimeError(f"No feature files found under: {feature_root}")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    cache = {}

    same_trials = sample_same_trials(by_speaker, args.same_trials, args.enroll_count, rng)
    diff_trials = sample_diff_trials(by_speaker, args.diff_trials, args.enroll_count, rng)
    trials = same_trials + diff_trials
    rng.shuffle(trials)

    rows = []
    for idx, trial in enumerate(trials, start=1):
        enroll_embeddings = [
            run_embedding(session, input_name, enroll_path, cache)
            for enroll_path in trial["enroll_paths"]
        ]
        profile_embedding = build_profile(enroll_embeddings)
        verify_embedding = run_embedding(session, input_name, trial["verify_path"], cache)
        score = cosine_score(profile_embedding, verify_embedding)
        rows.append(
            {
                "index": idx,
                "label": trial["label"],
                "score": f"{score:.6f}",
                "speaker_key": trial["speaker_key"],
                "enroll_paths": "|".join(str(path) for path in trial["enroll_paths"]),
                "verify_path": str(trial["verify_path"]),
            }
        )

        if idx % 100 == 0 or idx == len(trials):
            print(f"Processed {idx}/{len(trials)}")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "label",
                "score",
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
        checkpoint=args.checkpoint or args.model,
        max_frames=args.max_frames,
        preprocess_for_inference=args.preprocess_for_inference,
        threshold_policy=args.threshold_policy,
    )
    calibration["results"] = str(results_path)

    print(f"Trials: {len(rows)}")
    print(f"Unique features embedded: {len(cache)}")
    print(f"Results: {results_path}")
    print(f"EER: {calibration['eer'] * 100:.4f}%")
    print(f"minDCF@0.01: {calibration['min_dcf_p01']:.6f}")
    print(f"Threshold policy: {calibration['threshold_policy']}")
    print(f"Selected threshold: {calibration['selected_threshold']:.6f}")
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
        from gui.services.config_store import (
            ConfigStore,
            THRESHOLD_SOURCE_CALIBRATION,
            set_backend_threshold_metadata,
        )

        store = ConfigStore()
        config = store.load()
        config["backend_mode"] = "local_onnx"
        config["local_onnx_model_path"] = str(model_path)
        set_backend_threshold_metadata(
            config,
            "local_onnx",
            threshold=float(calibration["selected_threshold"]),
            threshold_policy=calibration["threshold_policy"],
            source=THRESHOLD_SOURCE_CALIBRATION,
            calibrated=True,
        )
        store.save(config)
        print(
            "GUI config updated: "
            f"local_onnx_threshold={calibration['selected_threshold']:.6f}"
        )


if __name__ == "__main__":
    main()
