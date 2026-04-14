from array import array
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.detection_metrics import compute_detection_metrics
from utils.legacy_feature import DEFAULT_MAX_FRAMES
from utils.score_norm import apply_symmetric_score_norm, compute_cohort_stats
from utils.speaker_verification import extract_embedding


def load_enroll_map(enroll_list_path):
    if not enroll_list_path:
        return {}

    path = Path(enroll_list_path)
    if not path.is_file():
        return {}

    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping


def iter_trials(trials_path, enroll_map, limit=0):
    with open(trials_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break

            parts = line.strip().split()
            if not parts:
                continue

            if len(parts) != 3:
                raise ValueError(f"Bad trial line: {line.strip()}")

            if parts[0] in {"0", "1"}:
                label = int(parts[0])
                audio_a = parts[1]
                audio_b = parts[2]
            else:
                audio_a = enroll_map.get(parts[0], parts[0])
                audio_b = parts[1]
                label = int(parts[2])

            yield audio_a, audio_b, label


def _reservoir_sample(stream, limit, rng):
    reservoir = []
    for idx, item in enumerate(stream, start=1):
        if len(reservoir) < limit:
            reservoir.append(item)
            continue
        swap_idx = int(rng.integers(0, idx))
        if swap_idx < limit:
            reservoir[swap_idx] = item
    return reservoir


def sample_trials(
    trials_path,
    enroll_map,
    limit=0,
    seed=42,
    sample_mode="balanced",
):
    if limit <= 0:
        return list(iter_trials(trials_path, enroll_map, limit=0))

    rng = np.random.default_rng(seed)
    if sample_mode == "random":
        return _reservoir_sample(
            iter_trials(trials_path, enroll_map, limit=0),
            limit=limit,
            rng=rng,
        )

    if sample_mode != "balanced":
        raise ValueError(f"Unsupported sample mode: {sample_mode}")

    if limit < 2:
        return _reservoir_sample(
            iter_trials(trials_path, enroll_map, limit=0),
            limit=limit,
            rng=rng,
        )

    target_limit = limit // 2
    non_target_limit = limit - target_limit
    target_trials = _reservoir_sample(
        (trial for trial in iter_trials(trials_path, enroll_map, limit=0) if trial[2] == 1),
        limit=target_limit,
        rng=rng,
    )
    non_target_trials = _reservoir_sample(
        (trial for trial in iter_trials(trials_path, enroll_map, limit=0) if trial[2] == 0),
        limit=non_target_limit,
        rng=rng,
    )

    sampled = target_trials + non_target_trials
    rng.shuffle(sampled)
    return sampled


@torch.no_grad()
def evaluate_trial_list(
    model,
    device,
    mel_transform,
    base_path,
    trials,
    max_frames=DEFAULT_MAX_FRAMES,
    num_eval=5,
    preprocess_for_inference=False,
    show_progress=True,
    return_raw=False,
    score_norm="none",
    cohort_top_k=300,
):
    unique_audio = set()
    target_count = 0
    for audio_a, audio_b, label in trials:
        unique_audio.add(audio_a)
        unique_audio.add(audio_b)
        target_count += label

    embedding_cache = {}
    audio_iter = sorted(unique_audio)
    if show_progress:
        audio_iter = tqdm(audio_iter, desc="Extract embeddings")
    for audio_path in audio_iter:
        emb, resolved = extract_embedding(
            model,
            audio_path,
            device=device,
            mel_transform=mel_transform,
            base_path=base_path,
            max_frames=max_frames,
            num_eval=num_eval,
            preprocess_for_inference=preprocess_for_inference,
        )
        embedding_cache[audio_path] = (emb, resolved)

    cohort_means = {}
    cohort_stds = {}
    if score_norm != "none":
        cohort_audio = sorted(unique_audio)
        cohort_embeddings = np.concatenate(
            [embedding_cache[audio_path][0].numpy() for audio_path in cohort_audio],
            axis=0,
        ).astype(np.float32)
        exclude_indices = [{idx} for idx in range(len(cohort_audio))]
        means, stds = compute_cohort_stats(
            cohort_embeddings,
            cohort_embeddings,
            mode=score_norm,
            top_k=cohort_top_k,
            exclude_indices=exclude_indices,
        )
        for idx, audio_path in enumerate(cohort_audio):
            cohort_means[audio_path] = float(means[idx])
            cohort_stds[audio_path] = float(stds[idx])

    scores = array("f")
    labels = array("b")
    trial_iter = trials
    if show_progress:
        trial_iter = tqdm(trials, desc="Score trials")
    for audio_a, audio_b, label in trial_iter:
        emb_a, _ = embedding_cache[audio_a]
        emb_b, _ = embedding_cache[audio_b]
        raw_score = torch.nn.functional.cosine_similarity(emb_a, emb_b).item()
        if score_norm == "none":
            score = raw_score
        else:
            score = apply_symmetric_score_norm(
                raw_score,
                cohort_means[audio_a],
                cohort_stds[audio_a],
                cohort_means[audio_b],
                cohort_stds[audio_b],
            )
        scores.append(score)
        labels.append(label)

    scores_np = np.asarray(scores, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int32)
    det = compute_detection_metrics(scores_np, labels_np)

    target_scores = scores_np[labels_np == 1]
    non_target_scores = scores_np[labels_np == 0]
    return {
        "num_trials": len(trials),
        "num_unique_audio": len(unique_audio),
        "num_target_trials": int(target_count),
        "num_non_target_trials": int(len(trials) - target_count),
        "eer": det["eer"],
        "eer_threshold": det["eer_threshold"],
        "min_dcf": det["min_dcf"],
        "accuracy": det["accuracy"],
        "far": det["far"],
        "frr": det["frr"],
        "tp": det["tp"],
        "tn": det["tn"],
        "fp": det["fp"],
        "fn": det["fn"],
        "target_mean": float(target_scores.mean()) if len(target_scores) else 0.0,
        "non_target_mean": float(non_target_scores.mean()) if len(non_target_scores) else 0.0,
        "curve_scores": det["curve_scores"],
        "curve_far": det["curve_far"],
        "curve_frr": det["curve_frr"],
        "scores": scores_np if return_raw else None,
        "labels": labels_np if return_raw else None,
        "threshold_far_1": det["threshold_far_1"],
        "frr_at_far_1": det["frr_at_far_1"],
        "threshold_far_0p1": det["threshold_far_0p1"],
        "frr_at_far_0p1": det["frr_at_far_0p1"],
        "score_norm": score_norm,
        "cohort_top_k": int(cohort_top_k),
    }
