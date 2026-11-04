"""Evaluation metrics. Phase 2 provides the separability diagnostic used for embeddings;
retrieval and detection metrics (Recall@K, AP, F1) arrive in Phases 3 and 6."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata


def roc_auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    ``AUC = P(s_pos > s_neg) + 0.5 P(s_pos = s_neg)``: the probability that a randomly
    drawn positive pair scores higher than a randomly drawn negative pair. 0.5 is chance.
    """
    pos = np.asarray(positive_scores, dtype=np.float64).ravel()
    neg = np.asarray(negative_scores, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        raise ValueError("need at least one positive and one negative score")
    ranks = rankdata(np.concatenate([pos, neg]))  # average ranks handle ties
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2
    return float(u / (pos.size * neg.size))


def recall_at_k(
    candidate_ids: np.ndarray, positives: Sequence[np.ndarray], ks: Sequence[int] = (1, 5, 10)
) -> dict[int, float]:
    """Recall@K over *positive queries* (queries with at least one true revisit).

    ``Recall@K = |{q in Q+ : top-K(q) contains a true revisit}| / |Q+|``. Queries without
    an eligible positive are excluded from the denominator (they cannot be answered
    correctly by retrieval); they matter for false-positive analysis in Phase 6 instead.
    """
    candidate_ids = np.asarray(candidate_ids)
    hits = np.array([np.isin(candidate_ids[i], positives[i]) for i in range(len(positives))])
    has_pos = np.array([p.size > 0 for p in positives])
    if not has_pos.any():
        return {k: float("nan") for k in ks}
    out = {}
    for k in ks:
        if k > candidate_ids.shape[1]:
            raise ValueError(f"K={k} exceeds the {candidate_ids.shape[1]} retrieved candidates")
        out[k] = float(hits[has_pos, :k].any(axis=1).mean())
    return out


def first_hit_rank(candidate_ids: np.ndarray, positives: Sequence[np.ndarray]) -> np.ndarray:
    """1-based rank of the first true revisit per query (0 = none in the list)."""
    ranks = np.zeros(len(positives), dtype=np.int64)
    for i, pos in enumerate(positives):
        hit = np.flatnonzero(np.isin(candidate_ids[i], pos))
        ranks[i] = hit[0] + 1 if hit.size else 0
    return ranks
