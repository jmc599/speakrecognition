from pathlib import Path

import numpy as np


THRESHOLD_POLICY_CHOICES = ("eer", "far_1", "far_0p1")


def _validate_scores_labels(scores, labels):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if scores.size == 0:
        raise ValueError("Detection metrics require at least one score.")
    if scores.size != labels.size:
        raise ValueError("Scores and labels must have the same length.")
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise ValueError("Detection metrics require both target and non-target trials.")
    return scores, labels


def _sorted_detection_curves(scores, labels):
    scores, labels = _validate_scores_labels(scores, labels)

    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    target_total = max(int(sorted_labels.sum()), 1)
    non_target_total = max(int((1 - sorted_labels).sum()), 1)

    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    fn = target_total - tp
    tn = non_target_total - fp

    far = fp / non_target_total
    frr = fn / target_total
    return {
        "scores": sorted_scores,
        "labels": sorted_labels,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "far": far,
        "frr": frr,
        "target_total": target_total,
        "non_target_total": non_target_total,
    }


def _select_far_operating_point(curves, target_far):
    far = curves["far"]
    frr = curves["frr"]
    scores = curves["scores"]
    tp = curves["tp"]
    fp = curves["fp"]
    fn = curves["fn"]
    tn = curves["tn"]

    valid = np.where(far <= float(target_far))[0]
    if valid.size == 0:
        return {
            "threshold": float(np.nextafter(scores.max(), np.float32(np.inf))),
            "far": 0.0,
            "frr": 1.0,
            "tp": 0,
            "tn": int(curves["non_target_total"]),
            "fp": 0,
            "fn": int(curves["target_total"]),
        }

    idx = int(valid[np.argmin(frr[valid])])
    return {
        "threshold": float(scores[idx]),
        "far": float(far[idx]),
        "frr": float(frr[idx]),
        "tp": int(tp[idx]),
        "tn": int(tn[idx]),
        "fp": int(fp[idx]),
        "fn": int(fn[idx]),
    }


def evaluate_fixed_threshold(scores, labels, threshold, eps=1e-12):
    scores, labels = _validate_scores_labels(scores, labels)
    threshold = float(threshold)
    predictions = (scores >= threshold).astype(np.int32)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    target_total = max(int(labels.sum()), 1)
    non_target_total = max(int((1 - labels).sum()), 1)
    far = float(fp / max(non_target_total, eps))
    frr = float(fn / max(target_total, eps))
    accuracy = float((tp + tn) / max(len(labels), 1))
    return {
        "threshold": threshold,
        "far": far,
        "frr": frr,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _validate_threshold_policy(threshold_policy):
    threshold_policy = str(threshold_policy or "eer")
    if threshold_policy not in THRESHOLD_POLICY_CHOICES:
        raise ValueError(
            "Unsupported threshold policy "
            f"{threshold_policy!r}. Expected one of {THRESHOLD_POLICY_CHOICES}."
        )
    return threshold_policy


def _threshold_key_for_policy(threshold_policy):
    threshold_policy = _validate_threshold_policy(threshold_policy)
    return {
        "eer": "eer_threshold",
        "far_1": "threshold_far_1",
        "far_0p1": "threshold_far_0p1",
    }[threshold_policy]


def select_threshold_from_summary(summary, threshold_policy):
    threshold_key = _threshold_key_for_policy(threshold_policy)
    if threshold_key not in summary:
        raise KeyError(
            f"Threshold policy {threshold_policy!r} requires field {threshold_key!r}."
        )
    return float(summary[threshold_key])


def compute_detection_metrics(scores, labels, p_target=0.01, c_miss=1.0, c_fa=1.0):
    curves = _sorted_detection_curves(scores, labels)
    sorted_scores = curves["scores"]
    sorted_labels = curves["labels"]
    far = curves["far"]
    frr = curves["frr"]

    diff = np.abs(far - frr)
    eer_idx = int(np.argmin(diff))
    eer = float((far[eer_idx] + frr[eer_idx]) / 2.0)
    eer_threshold = float(sorted_scores[eer_idx])
    predictions = sorted_scores >= eer_threshold
    tp_eer = int(np.sum((predictions == 1) & (sorted_labels == 1)))
    tn_eer = int(np.sum((predictions == 0) & (sorted_labels == 0)))
    fp_eer = int(np.sum((predictions == 1) & (sorted_labels == 0)))
    fn_eer = int(np.sum((predictions == 0) & (sorted_labels == 1)))
    accuracy = float((tp_eer + tn_eer) / len(sorted_labels))

    dcf = c_miss * frr * p_target + c_fa * far * (1.0 - p_target)
    min_dcf_idx = int(np.argmin(dcf))
    min_dcf = float(
        dcf[min_dcf_idx] / min(c_miss * p_target, c_fa * (1.0 - p_target))
    )

    far_1 = _select_far_operating_point(curves, target_far=0.01)
    far_0p1 = _select_far_operating_point(curves, target_far=0.001)

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "min_dcf": min_dcf,
        "min_dcf_threshold": float(sorted_scores[min_dcf_idx]),
        "min_dcf_far": float(far[min_dcf_idx]),
        "min_dcf_frr": float(frr[min_dcf_idx]),
        "accuracy": accuracy,
        "far": float(far[eer_idx]),
        "frr": float(frr[eer_idx]),
        "tp": tp_eer,
        "tn": tn_eer,
        "fp": fp_eer,
        "fn": fn_eer,
        "curve_scores": sorted_scores.copy(),
        "curve_far": far.copy(),
        "curve_frr": frr.copy(),
        "threshold_far_1": far_1["threshold"],
        "far_at_far_1": far_1["far"],
        "frr_at_far_1": far_1["frr"],
        "threshold_far_0p1": far_0p1["threshold"],
        "far_at_far_0p1": far_0p1["far"],
        "frr_at_far_0p1": far_0p1["frr"],
    }


def load_score_label_pairs(results_path):
    pairs = np.genfromtxt(
        str(Path(results_path)),
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    if pairs.size == 0:
        raise RuntimeError(f"No rows found in results file: {results_path}")
    if pairs.ndim == 0:
        pairs = np.asarray([pairs], dtype=pairs.dtype)
    labels = np.asarray(pairs["label"], dtype=np.int32)
    scores = np.asarray(pairs["score"], dtype=np.float32)
    return scores, labels


def build_calibration_summary(
    scores,
    labels,
    *,
    checkpoint="",
    max_frames=None,
    preprocess_for_inference=None,
    threshold_policy="eer",
):
    scores, labels = _validate_scores_labels(scores, labels)
    threshold_policy = _validate_threshold_policy(threshold_policy)
    det = compute_detection_metrics(scores, labels)
    target_scores = scores[labels == 1]
    non_target_scores = scores[labels == 0]
    selected_threshold = select_threshold_from_summary(det, threshold_policy)
    selected_point = evaluate_fixed_threshold(scores, labels, selected_threshold)
    return {
        "checkpoint": str(checkpoint),
        "max_frames": None if max_frames is None else int(max_frames),
        "preprocess_for_inference": (
            None
            if preprocess_for_inference is None
            else bool(preprocess_for_inference)
        ),
        "num_trials": int(labels.size),
        "num_target_trials": int(labels.sum()),
        "num_non_target_trials": int(labels.size - labels.sum()),
        "eer": det["eer"],
        "eer_threshold": det["eer_threshold"],
        "min_dcf_p01": det["min_dcf"],
        "min_dcf_threshold": det["min_dcf_threshold"],
        "threshold_far_1": det["threshold_far_1"],
        "frr_at_far_1": det["frr_at_far_1"],
        "threshold_far_0p1": det["threshold_far_0p1"],
        "frr_at_far_0p1": det["frr_at_far_0p1"],
        "threshold_policy": threshold_policy,
        "selected_threshold_key": _threshold_key_for_policy(threshold_policy),
        "selected_threshold": selected_point["threshold"],
        "selected_accuracy": selected_point["accuracy"],
        "selected_far": selected_point["far"],
        "selected_frr": selected_point["frr"],
        "selected_tp": selected_point["tp"],
        "selected_tn": selected_point["tn"],
        "selected_fp": selected_point["fp"],
        "selected_fn": selected_point["fn"],
        "target_mean": float(target_scores.mean()) if target_scores.size else 0.0,
        "non_target_mean": (
            float(non_target_scores.mean()) if non_target_scores.size else 0.0
        ),
    }


def build_threshold_transfer_summary(
    scores,
    labels,
    reference_summary,
    *,
    checkpoint="",
    max_frames=None,
    preprocess_for_inference=None,
    threshold_policy="eer",
):
    scores, labels = _validate_scores_labels(scores, labels)
    threshold_policy = _validate_threshold_policy(threshold_policy)
    target_scores = scores[labels == 1]
    non_target_scores = scores[labels == 0]

    threshold_far_1 = float(reference_summary["threshold_far_1"])
    threshold_far_0p1 = float(reference_summary["threshold_far_0p1"])
    eer_threshold = (
        float(reference_summary["eer_threshold"])
        if "eer_threshold" in reference_summary
        else None
    )
    point_far_1 = evaluate_fixed_threshold(scores, labels, threshold_far_1)
    point_far_0p1 = evaluate_fixed_threshold(scores, labels, threshold_far_0p1)
    selected_threshold = select_threshold_from_summary(reference_summary, threshold_policy)
    selected_point = evaluate_fixed_threshold(scores, labels, selected_threshold)

    summary = {
        "checkpoint": str(checkpoint),
        "max_frames": None if max_frames is None else int(max_frames),
        "preprocess_for_inference": (
            None
            if preprocess_for_inference is None
            else bool(preprocess_for_inference)
        ),
        "num_trials": int(labels.size),
        "num_target_trials": int(labels.sum()),
        "num_non_target_trials": int(labels.size - labels.sum()),
        "reference_results": str(reference_summary.get("results", "")),
        "reference_threshold_far_1": threshold_far_1,
        "reference_threshold_far_0p1": threshold_far_0p1,
        "threshold_far_1": threshold_far_1,
        "frr_at_far_1": point_far_1["frr"],
        "far_at_far_1": point_far_1["far"],
        "threshold_far_0p1": threshold_far_0p1,
        "frr_at_far_0p1": point_far_0p1["frr"],
        "far_at_far_0p1": point_far_0p1["far"],
        "accuracy_at_far_1_threshold": point_far_1["accuracy"],
        "accuracy_at_far_0p1_threshold": point_far_0p1["accuracy"],
        "threshold_policy": threshold_policy,
        "selected_threshold_key": _threshold_key_for_policy(threshold_policy),
        "selected_threshold": selected_point["threshold"],
        "selected_accuracy": selected_point["accuracy"],
        "selected_far": selected_point["far"],
        "selected_frr": selected_point["frr"],
        "selected_tp": selected_point["tp"],
        "selected_tn": selected_point["tn"],
        "selected_fp": selected_point["fp"],
        "selected_fn": selected_point["fn"],
        "target_mean": float(target_scores.mean()) if target_scores.size else 0.0,
        "non_target_mean": (
            float(non_target_scores.mean()) if non_target_scores.size else 0.0
        ),
    }
    if eer_threshold is not None:
        point_eer = evaluate_fixed_threshold(scores, labels, eer_threshold)
        summary["reference_eer_threshold"] = eer_threshold
        summary["eer_threshold"] = eer_threshold
        summary["far_at_eer_threshold"] = point_eer["far"]
        summary["frr_at_eer_threshold"] = point_eer["frr"]
        summary["accuracy_at_eer_threshold"] = point_eer["accuracy"]
    return summary
