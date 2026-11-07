"""Representation probes motivated by Balestriero & LeCun (2026), run on development data only.

* :func:`scene_explained_fraction`: how much of a descriptor's variation is *scene* rather
  than *nuisance*. "Good views" are two observations of the same place at different times
  (a true revisit); their squared distance is nuisance. With random frame pairs as the
  reference spread,

      rho(f) = 1 - E_same ||f(x) - f(x')||^2 / E_random ||f(x) - f(y)||^2

  ``rho = 1`` means revisits map to the same point; ``rho = 0`` means a revisit is no closer
  than a random frame. Computed in the full space or in the top-``k`` principal directions.
* :func:`pca_basis` / :func:`project`: PCA truncation, fitted on a *different* sequence.
* :func:`causal_average`: average each descriptor with those of the previous ``w - 1``
  frames (multi-view "frame averaging" that only uses the past, so it stays causal).

Ground-truth poses are used here only to *measure* (choose same-place pairs); no retrieval
decision depends on them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import binomtest


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def same_place_pairs(
    positives: Sequence[np.ndarray], frame_ids: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """One (query, closest true revisit) index pair per query that has a revisit."""
    ids = np.asarray(frame_ids)
    out = []
    for i, pos in enumerate(positives):
        if len(pos) == 0:
            continue
        j = np.searchsorted(ids, pos)
        d = np.linalg.norm(positions[j][:, [0, 2]] - positions[i, [0, 2]], axis=1)
        out.append((i, int(j[np.argmin(d)])))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def random_pairs(n: int, count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, count)
    b = (a + rng.integers(1, n, count)) % n  # never a == b
    return np.c_[a, b]


def scene_explained_fraction(
    emb: np.ndarray, same: np.ndarray, rand: np.ndarray, basis: np.ndarray | None = None
) -> float:
    """``rho = 1 - mean ||d_same||^2 / mean ||d_random||^2`` (optionally in ``basis``)."""
    emb = np.asarray(emb, dtype=np.float64)
    ds = emb[same[:, 0]] - emb[same[:, 1]]
    dr = emb[rand[:, 0]] - emb[rand[:, 1]]
    if basis is not None:
        ds, dr = ds @ basis, dr @ basis
    return float(1.0 - (ds**2).sum(1).mean() / (dr**2).sum(1).mean())


def pca_basis(emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and the top-``k`` principal directions (columns) of ``emb``."""
    x = np.asarray(emb, dtype=np.float64)
    mu = x.mean(0)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    return mu, vt[:k].T


def project(emb: np.ndarray, mu: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return l2n((np.asarray(emb, dtype=np.float64) - mu) @ basis)


def causal_average(emb: np.ndarray, window: int) -> np.ndarray:
    """``f~_t = normalise(sum_{s=max(0,t-w+1)}^{t} f_s)`` over the sampled sequence order."""
    if window < 1:
        raise ValueError("window must be >= 1")
    c = np.cumsum(np.asarray(emb, dtype=np.float64), axis=0)
    out = c.copy()
    out[window:] -= c[:-window]
    return l2n(out)


def concat_views(*parts: np.ndarray) -> np.ndarray:
    """Concatenate L2-normalised descriptors with equal weight, then re-normalise."""
    return l2n(np.concatenate([l2n(p) for p in parts], axis=1))


def mcnemar(base_hit: np.ndarray, new_hit: np.ndarray) -> dict[str, float]:
    """Exact two-sided McNemar test on paired per-query hits."""
    b = int((new_hit & ~base_hit).sum())  # new right, base wrong
    c = int((base_hit & ~new_hit).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"new_only": b, "base_only": c, "p": float(p)}
