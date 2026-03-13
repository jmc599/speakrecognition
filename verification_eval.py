from array import array
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

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


def compute_detection_metrics(scores, labels):
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise ValueError("Detection metrics require both target and non-target trials.")

    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]

    target_total = max(int(labels.sum()), 1)
    non_target_total = max(int((1 - labels).sum()), 1)

    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    fn = target_total - tp

    far = fp / non_target_total
    frr = fn / target_total
    diff = np.abs(far - frr)
    eer_idx = int(np.argmin(diff))
    eer = float((far[eer_idx] + frr[eer_idx]) / 2.0)
    eer_threshold = float(scores[eer_idx])

    p_target = 0.01
    c_miss = 1.0
    c_fa = 1.0
    dcf = c_miss * frr * p_target + c_fa * far * (1.0 - p_target)
    min_dcf = float(dcf.min() / min(c_miss * p_target, c_fa * (1.0 - p_target)))

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "min_dcf": min_dcf,
    }


@torch.no_grad()
def evaluate_trial_list(
    model,
    device,
    mel_transform,
    base_path,
    trials,
    max_frames=200,
    num_eval=5,
    show_progress=True,
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
        )
        embedding_cache[audio_path] = (emb, resolved)

    scores = array("f")
    labels = array("b")
    trial_iter = trials
    if show_progress:
        trial_iter = tqdm(trials, desc="Score trials")
    for audio_a, audio_b, label in trial_iter:
        emb_a, _ = embedding_cache[audio_a]
        emb_b, _ = embedding_cache[audio_b]
        score = torch.nn.functional.cosine_similarity(emb_a, emb_b).item()
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
        "target_mean": float(target_scores.mean()) if len(target_scores) else 0.0,
        "non_target_mean": float(non_target_scores.mean()) if len(non_target_scores) else 0.0,
    }
