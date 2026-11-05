"""Verify saved retrieval candidates of a whole sequence.

For every query and each of its top ``verify_top_k`` retrieved candidates, ORB features of
both frames are matched and the pair is geometrically verified
(:func:`latent_loop.verification.geometry.verify_features`). The loop-closure decision of
a query is its highest-ranked *accepted* candidate (or none).

Only images and the camera calibration enter the decision. Ground-truth poses are used
afterwards, by :func:`label_pairs`, to annotate pairs for analysis and figures.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latent_loop.datasets.base import CameraCalibration
from latent_loop.datasets.ground_truth import IGNORED, NEGATIVE, POSITIVE
from latent_loop.retrieval import RetrievalResult
from latent_loop.utils.logging import get_logger
from latent_loop.verification.geometry import VerificationConfig, verify_features
from latent_loop.verification.orb import OrbConfig, OrbFeatures, extract_orb

logger = get_logger(__name__)


def run_dirname(verify_top_k: int, estimator: str) -> str:
    """Sub-directory of a verification run, so runs with other settings never overwrite it."""
    return f"top{int(verify_top_k)}_{estimator}"


class FeatureStore:
    """ORB features per original frame ID, computed on demand and kept in an LRU cache."""

    def __init__(
        self,
        load_image: Callable[[int], np.ndarray],  # original frame ID -> image
        config: OrbConfig,
        capacity: int = 4096,
    ) -> None:
        self._load = load_image
        self._config = config
        self._cache: OrderedDict[int, OrbFeatures] = OrderedDict()
        self._capacity = capacity
        self.extract_seconds = 0.0
        self.extracted = 0

    def __call__(self, frame_id: int) -> OrbFeatures:
        frame_id = int(frame_id)
        if frame_id in self._cache:
            self._cache.move_to_end(frame_id)
            return self._cache[frame_id]
        t0 = time.perf_counter()
        feats = extract_orb(self._load(frame_id), self._config)
        self.extract_seconds += time.perf_counter() - t0
        self.extracted += 1
        self._cache[frame_id] = feats
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return feats


@dataclass
class VerificationRun:
    pairs: pd.DataFrame  # one row per verified (query, candidate) pair
    config: VerificationConfig
    verify_top_k: int
    timing: dict[str, float] = field(default_factory=dict)

    def decisions(self) -> pd.DataFrame:
        """Per query: the highest-ranked accepted candidate (``-1`` if none)."""
        acc = self.pairs[self.pairs.accepted].sort_values(["query_frame_id", "rank"])
        first = acc.groupby("query_frame_id", sort=True).head(1).set_index("query_frame_id")
        queries = pd.Index(np.unique(self.pairs.query_frame_id), name="query_frame_id")
        out = pd.DataFrame(index=queries)
        out["loop_frame_id"] = first["candidate_frame_id"].reindex(queries).fillna(-1).astype(int)
        out["rank"] = first["rank"].reindex(queries).fillna(-1).astype(int)
        out["n_inliers"] = first["n_inliers"].reindex(queries).fillna(0).astype(int)
        return out.reset_index()

    def save(self, directory: str | Path, extra: Mapping[str, Any] | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.pairs.to_csv(directory / "verification.csv", index=False)
        meta = {
            "config": self.config.to_dict(),
            "verify_top_k": self.verify_top_k,
            "timing": self.timing,
            **(extra or {}),
        }
        (directory / "verification.json").write_text(json.dumps(meta, indent=2, default=str))
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> VerificationRun:
        directory = Path(directory)
        meta = json.loads((directory / "verification.json").read_text())
        pairs = pd.read_csv(
            directory / "verification.csv", dtype={"degenerate": str, "reason": str}
        )
        if "degenerate" in pairs:
            pairs["degenerate"] = pairs["degenerate"].fillna("")
        return cls(
            pairs,
            VerificationConfig.from_dict(meta["config"]),
            int(meta["verify_top_k"]),
            meta.get("timing", {}),
        )


def verify_candidates(
    retrieval: RetrievalResult,
    features: Callable[[int], OrbFeatures],
    calibration: CameraCalibration,
    config: VerificationConfig,
    verify_top_k: int = 1,
    query_indices: np.ndarray | None = None,
) -> VerificationRun:
    """Verify the top ``verify_top_k`` candidates of each query (or of ``query_indices``)."""
    if verify_top_k < 1:
        raise ValueError("verify_top_k must be >= 1")
    k = min(verify_top_k, retrieval.candidate_ids.shape[1])
    indices = np.arange(len(retrieval.query_frame_ids)) if query_indices is None else query_indices
    rows = []
    start = time.perf_counter()
    for n_done, i in enumerate(indices):
        if n_done and n_done % 100 == 0:
            logger.info(
                "Verification progress",
                extra={"queries": f"{n_done}/{len(indices)}", "pairs": len(rows)},
            )
        q = int(retrieval.query_frame_ids[i])
        for rank in range(k):
            c = int(retrieval.candidate_ids[i, rank])
            if c < 0:
                break
            res = verify_features(features(q), features(c), calibration, config)
            rows.append(
                {
                    "query_index": int(i),
                    "query_frame_id": q,
                    "candidate_frame_id": c,
                    "rank": rank + 1,
                    "cosine": float(retrieval.scores[i, rank]),
                    **res.record(),
                }
            )
    total = time.perf_counter() - start
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        pairs = pd.DataFrame(
            columns=["query_index", "query_frame_id", "candidate_frame_id", "rank", "accepted"]
        )
    timing = {
        "total_s": total,
        "pairs": len(pairs),
        "ms_per_pair_including_orb": 1000 * total / max(len(pairs), 1),
    }
    for stage in ("match", "ransac", "pose"):
        col = f"ms_{stage}"
        if col in pairs:
            timing[f"median_{col}"] = float(np.nanmedian(pairs[col]))
    return VerificationRun(pairs, config, k, timing)


def label_pairs(
    pairs: pd.DataFrame,
    frame_ids: np.ndarray,
    positions: np.ndarray,
    positive_radius_m: float = 5.0,
    negative_radius_m: float = 25.0,
) -> pd.DataFrame:
    """Add ground-truth distance and label columns (evaluation only, after decisions).

    ``gt_label`` is POSITIVE (1) within ``positive_radius_m`` (ground-plane x-z distance),
    NEGATIVE (0) beyond ``negative_radius_m`` and IGNORED (-1) in between.
    """
    out = pairs.copy()
    index = {int(f): i for i, f in enumerate(frame_ids)}
    xz = np.asarray(positions)[:, [0, 2]]
    qi = np.array([index[int(f)] for f in out.query_frame_id])
    ci = np.array([index[int(f)] for f in out.candidate_frame_id])
    dist = np.linalg.norm(xz[qi] - xz[ci], axis=1) if len(out) else np.zeros(0)
    label = np.full(len(out), IGNORED, dtype=np.int8)
    label[dist <= positive_radius_m] = POSITIVE
    label[dist > negative_radius_m] = NEGATIVE
    out["gt_distance_m"] = dist
    out["gt_label"] = label
    return out


def summarize(pairs: pd.DataFrame) -> dict[str, Any]:
    """Pair-level confusion counts and rejection-reason histograms per ground-truth label."""
    out: dict[str, Any] = {"pairs": int(len(pairs)), "accepted": int(pairs.accepted.sum())}
    if "gt_label" not in pairs:
        out["reasons"] = pairs.reason.value_counts().to_dict()
        return out
    pos, neg = pairs[pairs.gt_label == POSITIVE], pairs[pairs.gt_label == NEGATIVE]
    out.update(
        {
            "positive_pairs": int(len(pos)),
            "negative_pairs": int(len(neg)),
            "ignored_pairs": int((pairs.gt_label == IGNORED).sum()),
            "true_accepts": int(pos.accepted.sum()),
            "false_rejects": int((~pos.accepted).sum()),
            "false_accepts": int(neg.accepted.sum()),
            "true_rejects": int((~neg.accepted).sum()),
            "accept_rate_positive": float(pos.accepted.mean()) if len(pos) else float("nan"),
            "accept_rate_negative": float(neg.accepted.mean()) if len(neg) else float("nan"),
            "reasons_positive": pos.reason.value_counts().to_dict(),
            "reasons_negative": neg.reason.value_counts().to_dict(),
            "median_inliers_positive": float(pos.n_inliers.median()) if len(pos) else float("nan"),
            "median_inliers_negative": float(neg.n_inliers.median()) if len(neg) else float("nan"),
            "degenerate_flags": pairs[pairs.degenerate != ""].degenerate.value_counts().to_dict(),
        }
    )
    return out
