import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort

from utils.wespeaker_features import WESPEAKER_NUM_MEL_BINS, compute_wespeaker_fbank


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}
SPK_PATTERN = re.compile(r"^id\d+$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate profile threshold for WeSpeaker runtime ONNX backend."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Dataset root containing speaker folders or subfolders.",
    )
    parser.add_argument(
        "--include-subdir",
        type=str,
        default="",
        help="Optional subfolder under dataset root to scan (e.g. data).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/wespeaker/cnceleb_resnet34_LM.onnx",
        help="WeSpeaker runtime ONNX model path.",
    )
    parser.add_argument("--same-trials", type=int, default=1000)
    parser.add_argument("--diff-trials", type=int, default=1000)
    parser.add_argument("--enroll-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-results",
        type=str,
        default="artifacts/results/results_wespeaker_profile_2000.csv",
        help="CSV path for per-trial scores.",
    )
    parser.add_argument(
        "--write-gui-config",
        action="store_true",
        help="Write the calibrated threshold back to GUI config and mark calibrated=true.",
    )
    return parser.parse_args()


def l2_normalize(x, axis=1, eps=1e-12):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def infer_speaker_key(file_path: Path, dataset_root: Path) -> str:
    rel_parts = file_path.relative_to(dataset_root).parts
    for part in rel_parts:
        if SPK_PATTERN.match(part):
            return part.lower()
    return file_path.parent.name.lower()


def collect_audio(dataset_root: Path, include_subdir: str):
    root = dataset_root / include_subdir if include_subdir else dataset_root
    if not root.exists():
        raise FileNotFoundError(f"Audio root not found: {root}")

    by_speaker = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        speaker_key = infer_speaker_key(path, dataset_root)
        by_speaker[speaker_key].append(path)
    return {key: sorted(paths) for key, paths in by_speaker.items()}


def sample_same_trials(by_speaker, count, enroll_count, rng):
    speaker_ids = [speaker_id for speaker_id, paths in by_speaker.items() if len(paths) >= enroll_count + 1]
    if not speaker_ids:
        raise RuntimeError("No speakers with enough audio for same-speaker profile trials.")

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
        raise RuntimeError("Need at least two speakers for different-speaker profile trials.")

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


def validate_model_input(session):
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError("ONNX model has no inputs.")
    input_meta = inputs[0]
    shape = list(input_meta.shape)
    if len(shape) != 3:
        raise ValueError(f"Expected rank-3 input [1,T,80], got {shape}")
    if shape[2] not in (WESPEAKER_NUM_MEL_BINS, str(WESPEAKER_NUM_MEL_BINS)):
        raise ValueError(f"Expected last input dim=80, got {shape[2]}")
    if input_meta.type != "tensor(float)":
        raise ValueError(f"Expected float32 input tensor, got {input_meta.type}")
    return input_meta.name


def run_embedding(session, input_name, audio_path, cache):
    cache_key = str(audio_path)
    if cache_key in cache:
        return cache[cache_key]

    feature = compute_wespeaker_fbank(audio_path)
    outputs = session.run(None, {input_name: feature})
    if not outputs:
        raise RuntimeError(f"ONNX inference returned no outputs for {audio_path}")

    embedding = np.asarray(outputs[0], dtype=np.float32)
    if embedding.ndim == 1:
        embedding = embedding[None, :]
    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError(f"Expected embedding [1,D], got {tuple(embedding.shape)} for {audio_path}")
    if not np.isfinite(embedding).all():
        raise ValueError(f"Embedding contains NaN/Inf for {audio_path}")
    embedding = l2_normalize(embedding, axis=1)
    cache[cache_key] = embedding
    return embedding


def cosine_score(emb_a, emb_b):
    return float(np.sum(emb_a * emb_b, axis=1)[0])


def build_profile(embeddings):
    merged = np.concatenate(embeddings, axis=0)
    merged = l2_normalize(merged, axis=1)
    merged = merged.mean(axis=0, keepdims=True)
    merged = l2_normalize(merged, axis=1)
    return merged


def sweep_best_threshold(rows):
    if not rows:
        raise RuntimeError("No rows to sweep thresholds.")
    pairs = [(int(r["label"]), float(r["score"])) for r in rows]
    thresholds = sorted(set(score for _, score in pairs))
    best = None
    for threshold in thresholds:
        tp = tn = fp = fn = 0
        for label, score in pairs:
            pred = 1 if score >= threshold else 0
            if label == 1 and pred == 1:
                tp += 1
            elif label == 0 and pred == 0:
                tn += 1
            elif label == 0 and pred == 1:
                fp += 1
            else:
                fn += 1
        accuracy = (tp + tn) / len(pairs)
        if best is None or accuracy > best["accuracy"]:
            best = {
                "threshold": float(threshold),
                "accuracy": float(accuracy),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }
    return best


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    dataset_root = Path(args.dataset_root)
    model_path = Path(args.model)
    results_path = Path(args.save_results)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not model_path.is_file():
        raise FileNotFoundError(f"WeSpeaker ONNX model not found: {model_path}")

    by_speaker = collect_audio(dataset_root, args.include_subdir.strip())
    if not by_speaker:
        raise RuntimeError("No audio files found for calibration.")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = validate_model_input(session)

    same_trials = sample_same_trials(by_speaker, args.same_trials, args.enroll_count, rng)
    diff_trials = sample_diff_trials(by_speaker, args.diff_trials, args.enroll_count, rng)
    trials = same_trials + diff_trials
    rng.shuffle(trials)

    cache = {}
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

    best = sweep_best_threshold(rows)
    scores = np.asarray([float(item["score"]) for item in rows], dtype=np.float32)
    labels = np.asarray([int(item["label"]) for item in rows], dtype=np.int32)
    same_scores = scores[labels == 1]
    diff_scores = scores[labels == 0]
    print(f"Pairs: {len(rows)}")
    print(f"same min/max: {same_scores.min():.6f} / {same_scores.max():.6f}")
    print(f"diff min/max: {diff_scores.min():.6f} / {diff_scores.max():.6f}")
    print(f"Best threshold: {best['threshold']:.6f}")
    print(f"Best accuracy: {best['accuracy']:.6f}")
    print(f"TP={best['tp']} TN={best['tn']} FP={best['fp']} FN={best['fn']}")
    print(f"Results: {results_path}")
    print(f"Unique audio embedded: {len(cache)}")

    if args.write_gui_config:
        from gui.services.config_store import ConfigStore

        store = ConfigStore()
        config = store.load()
        config["wespeaker_onnx_threshold"] = float(best["threshold"])
        config["wespeaker_threshold_calibrated"] = True
        store.save(config)
        print(
            "GUI config updated: "
            f"wespeaker_onnx_threshold={best['threshold']:.6f}, "
            "wespeaker_threshold_calibrated=True"
        )


if __name__ == "__main__":
    main()
