"""Loop-closure candidate retrieval: offline evaluation and causal replay.

Both modes return, for every query frame, the top-K *eligible* earlier frames ranked by
cosine similarity, after optional candidate filtering:

* **Offline evaluation** indexes all descriptors once, then searches every query with
  its own eligible prefix (:class:`TemporalExclusion`).
* **Causal replay** processes frames chronologically: query ``t`` is searched against an
  index holding only frames observed *before* ``t``, then frame ``t`` is added. Future
  frames can therefore never be retrieved, by construction.

With identical settings both modes return identical candidates (the eligible set is the
same prefix); this is asserted in the tests.

Candidate filtering, in order:

1. ``min_similarity``: drop candidates with ``s < tau``.
2. Temporal non-maximum suppression (``nms_frames > 0``): walking down the ranked list,
   drop a candidate whose frame ID lies within ``nms_frames`` of an already kept one, so
   the top-K are distinct places rather than K consecutive frames of one place.

Ground truth is never consulted here.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from latent_loop.retrieval.index import FlatIPIndex
from latent_loop.retrieval.temporal_filter import TemporalExclusion, validate_chronological

MODES = ("offline", "causal")


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 10
    temporal_exclusion_frames: int | None = 100
    temporal_exclusion_seconds: float | None = None
    min_similarity: float | None = None
    nms_frames: int = 0
    mode: str = "offline"

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.nms_frames < 0:
            raise ValueError("nms_frames must be >= 0")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> RetrievalConfig:
        values = dict(values or {})
        known = set(cls.__dataclass_fields__)
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown retrieval keys: {sorted(unknown)}")
        return cls(**values)

    @property
    def exclusion(self) -> TemporalExclusion:
        return TemporalExclusion(self.temporal_exclusion_frames, self.temporal_exclusion_seconds)


@dataclass
class RetrievalResult:
    """Ranked candidates per query. Unfilled slots: frame ID ``-1``, score ``nan``."""

    query_frame_ids: np.ndarray  # (N,)
    query_timestamps: np.ndarray  # (N,)
    candidate_ids: np.ndarray  # (N, K) original frame IDs
    scores: np.ndarray  # (N, K) cosine similarities, descending per row
    eligible_counts: np.ndarray  # (N,) references the query was allowed to see
    config: RetrievalConfig
    timing: dict[str, float] = field(default_factory=dict)

    @property
    def top1_ids(self) -> np.ndarray:
        return self.candidate_ids[:, 0]

    @property
    def top1_scores(self) -> np.ndarray:
        return self.scores[:, 0]

    def save(self, directory: str | Path, extra: Mapping[str, Any] | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "retrieval.npz",
            query_frame_ids=self.query_frame_ids,
            query_timestamps=self.query_timestamps,
            candidate_ids=self.candidate_ids,
            scores=self.scores,
            eligible_counts=self.eligible_counts,
        )
        meta = {"config": asdict(self.config), "timing": self.timing, **(extra or {})}
        (directory / "retrieval.json").write_text(json.dumps(meta, indent=2, default=str))
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> RetrievalResult:
        directory = Path(directory)
        data = np.load(directory / "retrieval.npz")
        meta = json.loads((directory / "retrieval.json").read_text())
        return cls(
            data["query_frame_ids"],
            data["query_timestamps"],
            data["candidate_ids"],
            data["scores"],
            data["eligible_counts"],
            RetrievalConfig(**meta["config"]),
            meta.get("timing", {}),
        )


def filter_candidates(
    scores: np.ndarray,
    frame_ids: np.ndarray,
    top_k: int,
    min_similarity: float | None,
    nms_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the similarity threshold and temporal NMS to one ranked candidate list."""
    keep_scores = np.full(top_k, np.nan, dtype=np.float32)
    keep_ids = np.full(top_k, -1, dtype=np.int64)
    kept: list[int] = []
    for score, fid in zip(scores, frame_ids, strict=True):
        if fid < 0 or len(kept) == top_k:
            break
        if min_similarity is not None and score < min_similarity:
            break  # the list is sorted, nothing below can pass
        if nms_frames and any(abs(int(fid) - k) <= nms_frames for k in kept):
            continue
        keep_scores[len(kept)] = score
        keep_ids[len(kept)] = fid
        kept.append(int(fid))
    return keep_scores, keep_ids


def _search_width(config: RetrievalConfig) -> int:
    # NMS discards neighbours of kept candidates, so search deeper to still fill K slots.
    return config.top_k if config.nms_frames == 0 else config.top_k * (2 * config.nms_frames + 1)


def _finalise(raw_scores, raw_ids, config) -> tuple[np.ndarray, np.ndarray]:
    if config.min_similarity is None and config.nms_frames == 0:
        return raw_scores[:, : config.top_k], raw_ids[:, : config.top_k]
    out = [
        filter_candidates(s, i, config.top_k, config.min_similarity, config.nms_frames)
        for s, i in zip(raw_scores, raw_ids, strict=True)
    ]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def retrieve(
    frame_ids: np.ndarray, timestamps: np.ndarray, embeddings: np.ndarray, config: RetrievalConfig
) -> RetrievalResult:
    """Run retrieval for every frame as a query, in ``config.mode``."""
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    validate_chronological(frame_ids, timestamps)
    counts = config.exclusion.eligible_counts(frame_ids, timestamps)
    width = _search_width(config)
    index = FlatIPIndex(embeddings.shape[1])
    start = time.perf_counter()
    if config.mode == "offline":
        index.add(frame_ids, embeddings, timestamps)
        built = time.perf_counter()
        raw_scores, raw_ids = index.search(embeddings, width, counts)
        searched = time.perf_counter()
        timing = {"index_build_s": built - start, "search_s": searched - built}
    else:
        raw_scores = np.full((frame_ids.size, width), np.nan, dtype=np.float32)
        raw_ids = np.full((frame_ids.size, width), -1, dtype=np.int64)
        search_s = add_s = 0.0
        for t in range(frame_ids.size):
            t0 = time.perf_counter()
            # Index holds frames 0..t-1 only; eligibility may shrink that prefix further.
            raw_scores[t], raw_ids[t] = (
                x[0]
                for x in index.search(embeddings[t], width, np.array([min(counts[t], len(index))]))
            )
            t1 = time.perf_counter()
            index.add(frame_ids[t : t + 1], embeddings[t : t + 1], timestamps[t : t + 1])
            search_s += t1 - t0
            add_s += time.perf_counter() - t1
        timing = {"index_build_s": add_s, "search_s": search_s}
    scores, ids = _finalise(raw_scores, raw_ids, config)
    timing["total_s"] = time.perf_counter() - start
    timing["search_ms_per_query"] = 1000 * timing["search_s"] / max(frame_ids.size, 1)
    return RetrievalResult(
        frame_ids,
        timestamps,
        ids,
        scores,
        counts,
        config,
        {k: round(v, 6) for k, v in timing.items()},
    )


def assert_causal(result: RetrievalResult) -> None:
    """Raise if any candidate is not strictly earlier and temporally eligible."""
    exclusion = result.config.exclusion
    time_of = dict(
        zip(result.query_frame_ids.tolist(), result.query_timestamps.tolist(), strict=True)
    )
    for q, t, cands in zip(
        result.query_frame_ids, result.query_timestamps, result.candidate_ids, strict=True
    ):
        valid = cands[cands >= 0]
        times = np.array([time_of[int(c)] for c in valid])
        if not np.all(exclusion.is_eligible(valid, times, int(q), float(t))):
            raise AssertionError(f"query {q} retrieved ineligible frames {valid}")
