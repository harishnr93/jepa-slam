"""Exact cosine-similarity retrieval with FAISS ``IndexFlatIP``.

For L2-normalised descriptors, the inner product is the cosine similarity,

    s(q, r) = q^T r = cos(angle(q, r))    with ||q|| = ||r|| = 1,

so an exhaustive inner-product index ranks references by cosine similarity. ``IndexFlatIP``
is exact (no quantisation or approximation): the returned top-K equal those of a
brute-force ``argsort`` of ``Q R^T`` up to ties and float32 rounding.

Descriptors are stored in insertion order, which is chronological, so position ``p`` in
the index corresponds to the ``p``-th observed frame. Temporal eligibility is a prefix
``[0, n_eligible)`` of the index and is enforced *inside* the search with
``faiss.IDSelectorRange`` (no over-fetching and post-filtering).
"""

from __future__ import annotations

import faiss
import numpy as np

NORM_TOLERANCE = 1e-3


class FlatIPIndex:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.index = faiss.IndexFlatIP(self.dim)
        self._frame_ids: list[int] = []
        self._times: list[float] = []

    def __len__(self) -> int:
        return self.index.ntotal

    @property
    def frame_ids(self) -> np.ndarray:
        return np.asarray(self._frame_ids, dtype=np.int64)

    @property
    def timestamps(self) -> np.ndarray:
        return np.asarray(self._times, dtype=np.float64)

    def add(
        self, frame_ids: np.ndarray, embeddings: np.ndarray, timestamps: np.ndarray | None = None
    ) -> None:
        """Append descriptors in chronological order (frame IDs must keep increasing)."""
        ids = np.atleast_1d(np.asarray(frame_ids, dtype=np.int64))
        emb = np.ascontiguousarray(np.atleast_2d(embeddings), dtype=np.float32)
        if emb.shape != (ids.size, self.dim):
            raise ValueError(
                f"expected embeddings of shape ({ids.size}, {self.dim}), got {emb.shape}"
            )
        norms = np.linalg.norm(emb, axis=1)
        if np.any(np.abs(norms - 1.0) > NORM_TOLERANCE):
            raise ValueError(
                "descriptors must be L2-normalised for cosine retrieval "
                f"(norms in [{norms.min():.4f}, {norms.max():.4f}])"
            )
        previous = self._frame_ids[-1] if self._frame_ids else None
        if np.any(np.diff(ids) <= 0) or (previous is not None and ids[0] <= previous):
            raise ValueError("frames must be added in chronological (increasing ID) order")
        times = (
            np.asarray(timestamps, dtype=np.float64)
            if timestamps is not None
            else ids.astype(np.float64)
        )
        self.index.add(emb)
        self._frame_ids.extend(int(i) for i in ids)
        self._times.extend(float(t) for t in times)

    def search(
        self, queries: np.ndarray, k: int, limits: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-``k`` references per query.

        ``limits[i]`` restricts query ``i`` to index positions ``[0, limits[i])``. Returns
        ``(scores, frame_ids)`` of shape ``(nq, k)``; unfilled slots have score ``nan``
        and frame ID ``-1`` (fewer than ``k`` eligible references).
        """
        q = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        nq = q.shape[0]
        scores = np.full((nq, k), np.nan, dtype=np.float32)
        ids = np.full((nq, k), -1, dtype=np.int64)
        if len(self) == 0 or k <= 0:
            return scores, ids
        if limits is None:
            limits = np.full(nq, len(self), dtype=np.int64)
        limits = np.minimum(np.asarray(limits, dtype=np.int64), len(self))
        all_ids = self.frame_ids
        for limit in np.unique(limits):  # one FAISS call per distinct eligible prefix
            rows = np.flatnonzero(limits == limit)
            if limit <= 0:
                continue
            kk = int(min(k, limit))
            params = faiss.SearchParameters(sel=faiss.IDSelectorRange(0, int(limit)))
            d, p = self.index.search(q[rows], kk, params=params)
            valid = p >= 0
            scores[rows, :kk] = np.where(valid, d, np.nan)
            ids[rows, :kk] = np.where(valid, all_ids[np.clip(p, 0, None)], -1)
        return scores, ids


def exact_search_numpy(
    queries: np.ndarray, references: np.ndarray, k: int, limits: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force reference implementation (tests): positions instead of frame IDs."""
    sims = np.atleast_2d(queries) @ np.asarray(references).T
    nq = sims.shape[0]
    scores = np.full((nq, k), np.nan, dtype=np.float32)
    pos = np.full((nq, k), -1, dtype=np.int64)
    for i in range(nq):
        n = int(limits[i])
        if n <= 0:
            continue
        order = np.argsort(-sims[i, :n], kind="stable")[:k]
        scores[i, : order.size] = sims[i, order]
        pos[i, : order.size] = order
    return scores, pos
