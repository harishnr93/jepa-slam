"""Evaluation metrics. Phase 2 provides the separability diagnostic used for embeddings;
retrieval and detection metrics (Recall@K, AP, F1) arrive in Phases 3 and 6."""

from __future__ import annotations

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
