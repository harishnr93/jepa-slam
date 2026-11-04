"""Temporal exclusion: which earlier frames a query is allowed to retrieve.

A reference ``r`` is *eligible* for query ``q`` iff all of the following hold:

* ``frame_id_r < frame_id_q``                       (never the query itself or the future)
* ``frame_id_r <= frame_id_q - exclusion_frames``   (if ``exclusion_frames`` is set)
* ``t_r <= t_q - exclusion_seconds``                (if ``exclusion_seconds`` is set)

Frame IDs are the **original** IDs of the source sequence, so the rule means the same
thing for any sampling stride; subsampled array indices are never used. Because a
sequence is stored in chronological order (frame IDs strictly increasing, timestamps
non-decreasing), the eligible references of a query always form a *prefix* of the
sequence. ``eligible_count`` returns the length of that prefix, which the FAISS index
turns into an exact ``IDSelectorRange`` search.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TemporalExclusion:
    frames: int | None = 100
    seconds: float | None = None

    def __post_init__(self) -> None:
        if self.frames is not None and self.frames < 1:
            raise ValueError("exclusion frames must be >= 1 (0 would allow the query itself)")
        if self.seconds is not None and self.seconds <= 0:
            raise ValueError("exclusion seconds must be > 0")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> TemporalExclusion:
        return cls(
            frames=cfg.get("temporal_exclusion_frames", 100),
            seconds=cfg.get("temporal_exclusion_seconds"),
        )

    def eligible_count(
        self,
        frame_ids: np.ndarray,
        timestamps: np.ndarray,
        query_frame_id: int,
        query_time: float,
    ) -> int:
        """Length of the eligible prefix of a chronologically ordered sequence."""
        count = int(np.searchsorted(frame_ids, query_frame_id, side="left"))  # strictly earlier
        if self.frames is not None:
            count = min(
                count, int(np.searchsorted(frame_ids, query_frame_id - self.frames, side="right"))
            )
        if self.seconds is not None:
            count = min(
                count, int(np.searchsorted(timestamps, query_time - self.seconds, side="right"))
            )
        return count

    def eligible_counts(self, frame_ids: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Eligible-prefix length for every frame of the sequence used as a query."""
        validate_chronological(frame_ids, timestamps)
        return np.array(
            [
                self.eligible_count(frame_ids, timestamps, f, t)
                for f, t in zip(frame_ids, timestamps, strict=True)
            ],
            dtype=np.int64,
        )

    def is_eligible(
        self,
        reference_frame_ids: np.ndarray,
        reference_times: np.ndarray,
        query_frame_id: int,
        query_time: float,
    ) -> np.ndarray:
        """Elementwise eligibility (used to *check* retrieved candidates)."""
        ref = np.asarray(reference_frame_ids)
        ok = ref < query_frame_id
        if self.frames is not None:
            ok &= ref <= query_frame_id - self.frames
        if self.seconds is not None:
            ok &= np.asarray(reference_times) <= query_time - self.seconds
        return ok


def validate_chronological(frame_ids: np.ndarray, timestamps: np.ndarray) -> None:
    frame_ids, timestamps = np.asarray(frame_ids), np.asarray(timestamps)
    if frame_ids.shape != timestamps.shape:
        raise ValueError("frame_ids and timestamps must have the same shape")
    if np.any(np.diff(frame_ids) <= 0):
        raise ValueError("frame_ids must be strictly increasing (chronological order)")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps must be non-decreasing (chronological order)")
