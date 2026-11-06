"""Phase 6 metrics: positive-query retrieval metrics and query-level detection metrics.

Two levels are kept apart:

* **Retrieval metrics** (Recall@K, AP, mAP) are computed over *positive queries* only:
  queries with at least one eligible true revisit. They measure ranking quality.
* **Detection metrics** (precision, recall, F1, false positives per 1,000 queries,
  confusion matrix, detection AP) are computed over *every* query that has at least one
  eligible reference, so queries without a revisit contribute false alarms.

A detector emits at most one loop closure per query: a candidate frame and a score
(cosine for retrieval-only methods, RANSAC inliers for geometrically verified ones). The
loop is *predicted* when ``score >= tau``. Ground truth is consulted only here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

# ----------------------------------------------------------------------------- retrieval


def average_precision_ranked(relevant: np.ndarray) -> float:
    """AP of one ranked list: ``(1/|P|) sum_k Precision@k * rel_k`` (``|P|`` = relevant items)."""
    relevant = np.asarray(relevant, dtype=bool)
    n_rel = int(relevant.sum())
    if n_rel == 0:
        return float("nan")
    hits = np.cumsum(relevant)
    ranks = np.arange(1, relevant.size + 1)
    return float((hits[relevant] / ranks[relevant]).sum() / n_rel)


def retrieval_average_precision(
    embeddings: np.ndarray,
    frame_ids: np.ndarray,
    eligible_counts: np.ndarray,
    positives: Sequence[np.ndarray],
) -> np.ndarray:
    """Per-query AP over the *full* ranking of each query's eligible references.

    References of query ``i`` are the eligible prefix ``[0, eligible_counts[i])``, ranked
    by cosine similarity (stable sort, so ties keep chronological order). Queries with no
    positive get ``nan``.
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    ids = np.asarray(frame_ids)
    out = np.full(len(positives), np.nan)
    for i, pos in enumerate(positives):
        n = int(eligible_counts[i])
        if pos.size == 0 or n == 0:
            continue
        sims = emb[:n] @ emb[i]
        order = np.argsort(-sims, kind="stable")
        out[i] = average_precision_ranked(np.isin(ids[order], pos))
    return out


# ----------------------------------------------------------------------------- detection


@dataclass
class Predictions:
    """One (optional) loop-closure hypothesis per query."""

    query_frame_ids: np.ndarray  # (N,)
    candidate_ids: np.ndarray  # (N,) -1 = no hypothesis
    scores: np.ndarray  # (N,) nan = no hypothesis
    correct: np.ndarray  # (N,) bool: candidate is a true revisit (labels only)
    has_positive: np.ndarray  # (N,) bool: query has an eligible true revisit
    evaluable: np.ndarray  # (N,) bool: query has at least one eligible reference
    candidate_distance_m: np.ndarray  # (N,) nan when no hypothesis

    def subset(self, mask: np.ndarray) -> Predictions:
        return Predictions(*(getattr(self, f)[mask] for f in self.__dataclass_fields__))

    @staticmethod
    def concat(parts: Sequence[Predictions]) -> Predictions:
        return Predictions(
            *(
                np.concatenate([getattr(p, f) for p in parts])
                for f in Predictions.__dataclass_fields__
            )
        )


def make_predictions(
    query_frame_ids: np.ndarray,
    candidate_ids: np.ndarray,
    scores: np.ndarray,
    positives: Sequence[np.ndarray],
    eligible_counts: np.ndarray,
    positions: np.ndarray,
) -> Predictions:
    """Attach ground-truth correctness and candidate distance to detector outputs."""
    ids = np.asarray(query_frame_ids)
    cand = np.asarray(candidate_ids, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64).copy()
    score[cand < 0] = np.nan
    index = {int(f): i for i, f in enumerate(ids)}
    xz = np.asarray(positions)[:, [0, 2]]
    correct = np.zeros(ids.size, dtype=bool)
    dist = np.full(ids.size, np.nan)
    for i, c in enumerate(cand):
        if c >= 0:
            correct[i] = c in positives[i]
            dist[i] = float(np.linalg.norm(xz[i] - xz[index[int(c)]]))
    return Predictions(
        ids,
        cand,
        score,
        correct,
        np.array([p.size > 0 for p in positives]),
        np.asarray(eligible_counts) > 0,
        dist,
    )


@dataclass
class DetectionMetrics:
    threshold: float
    queries: int  # evaluable queries (at least one eligible reference)
    positive_queries: int
    predicted: int
    tp: int
    fp: int
    fn: int  # positive queries without a correct loop closure
    tn: int  # queries without a revisit and without a prediction
    precision: float
    recall: float
    f1: float
    fp_per_1000: float
    # confusion matrix rows: has revisit / no revisit; cols: correct / wrong / no loop
    confusion: list[list[int]]
    fp_near_5_25m: int  # wrong loops whose candidate is 5-25 m away (overlapping views)

    def to_dict(self) -> dict:
        return asdict(self)


def detection_at(pred: Predictions, threshold: float) -> DetectionMetrics:
    """Query-level detection metrics when loops with ``score >= threshold`` are accepted.

    ``precision = TP / (TP + FP)``; ``recall = TP / |Q+|`` where ``Q+`` are the positive
    queries; ``F1`` is their harmonic mean; a positive query with a *wrong* loop counts
    as both an FP and an FN. ``FP per 1,000 = 1000 FP / N`` over all evaluable queries.
    """
    p = pred.subset(pred.evaluable)
    on = np.nan_to_num(p.scores, nan=-np.inf) >= threshold
    tp = int((on & p.correct).sum())
    fp = int((on & ~p.correct).sum())
    npos = int(p.has_positive.sum())
    fn = npos - tp
    tn = int((~on & ~p.has_positive).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / npos if npos else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    hp, nh = p.has_positive, ~p.has_positive
    confusion = [
        [tp, int((on & ~p.correct & hp).sum()), int((~on & hp).sum())],
        [0, int((on & nh).sum()), int((~on & nh).sum())],
    ]
    near = on & ~p.correct & (p.candidate_distance_m > 5) & (p.candidate_distance_m <= 25)
    return DetectionMetrics(
        float(threshold),
        int(p.query_frame_ids.size),
        npos,
        int(on.sum()),
        tp,
        fp,
        fn,
        tn,
        precision,
        recall,
        f1,
        1000.0 * fp / max(p.query_frame_ids.size, 1),
        confusion,
        int(near.sum()),
    )


@dataclass
class PRCurve:
    thresholds: np.ndarray  # descending distinct scores
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    fp: np.ndarray
    ap: float  # step-wise area: sum_i P_i (R_i - R_{i-1})
    positive_queries: int
    queries: int


def pr_curve(pred: Predictions) -> PRCurve:
    """Precision/recall of the detector at every distinct score threshold (descending)."""
    p = pred.subset(pred.evaluable)
    valid = np.isfinite(p.scores)
    s, c = p.scores[valid], p.correct[valid]
    order = np.argsort(-s, kind="stable")
    s, c = s[order], c[order]
    npos = int(p.has_positive.sum())
    tp_cum = np.cumsum(c)
    fp_cum = np.cumsum(~c)
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1] if s.size else np.array([], int)
    tp, fp = tp_cum[last], fp_cum[last]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / npos if npos else np.zeros_like(tp, dtype=float)
    f1 = np.where(
        2 * tp + fp + (npos - tp) > 0, 2 * tp / np.maximum(2 * tp + fp + (npos - tp), 1), 0.0
    )
    prev_r = np.r_[0.0, recall[:-1]] if recall.size else recall
    ap = float(np.sum(precision * (recall - prev_r))) if npos else float("nan")
    return PRCurve(
        s[last], precision, recall, f1, fp.astype(int), ap, npos, int(p.query_frame_ids.size)
    )


def select_threshold(curve: PRCurve) -> dict[str, float]:
    """Thresholds chosen on development data.

    ``f1``: the threshold maximising F1 (the highest such threshold on ties, i.e. the most
    conservative). ``p100``: the lowest threshold that still has precision 1.0
    (maximum recall at 100 % precision; ``inf`` if even the top score is wrong).
    """
    if curve.thresholds.size == 0:
        return {
            "f1": float("inf"),
            "p100": float("inf"),
            "f1_value": float("nan"),
            "recall_at_p100": 0.0,
        }
    i = int(np.argmax(curve.f1))  # first maximum = highest threshold
    perfect = np.flatnonzero(curve.precision >= 1.0)
    run = perfect[perfect == np.arange(perfect.size)] if perfect.size else perfect
    if run.size:
        j = int(run[-1])
        p100, r100 = float(curve.thresholds[j]), float(curve.recall[j])
    else:
        p100, r100 = float("inf"), 0.0
    return {
        "f1": float(curve.thresholds[i]),
        "f1_value": float(curve.f1[i]),
        "p100": p100,
        "recall_at_p100": r100,
    }
