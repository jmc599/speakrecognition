import numpy as np


def _as_2d_array(embeddings):
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {array.shape}")
    return array


def compute_cohort_stats(
    query_embeddings,
    cohort_embeddings,
    *,
    mode="none",
    top_k=300,
    exclude_indices=None,
    eps=1e-6,
):
    query = _as_2d_array(query_embeddings)
    cohort = _as_2d_array(cohort_embeddings)

    if mode == "none":
        zeros = np.zeros(query.shape[0], dtype=np.float32)
        ones = np.ones(query.shape[0], dtype=np.float32)
        return zeros, ones

    similarities = np.matmul(query, cohort.T).astype(np.float32, copy=False)

    if exclude_indices is not None:
        for row_idx, blocked in enumerate(exclude_indices):
            if not blocked:
                continue
            similarities[row_idx, np.asarray(sorted(blocked), dtype=np.int64)] = np.nan

    means = np.zeros(query.shape[0], dtype=np.float32)
    stds = np.ones(query.shape[0], dtype=np.float32)

    for row_idx in range(similarities.shape[0]):
        row = similarities[row_idx]
        valid = row[np.isfinite(row)]
        if valid.size == 0:
            continue

        if mode == "asnorm":
            k = int(top_k)
            if k > 0 and valid.size > k:
                valid = np.sort(valid)[-k:]
        elif mode != "snorm":
            raise ValueError(f"Unsupported score normalization mode: {mode}")

        means[row_idx] = float(valid.mean())
        std = float(valid.std())
        stds[row_idx] = std if std > eps else eps

    return means, stds


def apply_symmetric_score_norm(
    raw_score,
    mean_a,
    std_a,
    mean_b,
    std_b,
):
    z_a = (float(raw_score) - float(mean_a)) / float(std_a)
    z_b = (float(raw_score) - float(mean_b)) / float(std_b)
    return 0.5 * (z_a + z_b)
