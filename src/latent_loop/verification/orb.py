"""ORB keypoints, binary descriptors and Hamming-distance matching.

ORB (Rublee et al., ICCV 2011) = oriented FAST corners on an image pyramid + rotated
BRIEF. Each descriptor is 256 bits (32 bytes): bit ``k`` is ``1`` iff the smoothed
intensity at the first point of a learned pixel-pair test is lower than at the second,
with the test pattern rotated by the keypoint's intensity-centroid orientation.

Two descriptors are compared with the Hamming distance (number of differing bits),

    d_H(a, b) = popcount(a XOR b),   0 <= d_H <= 256.

Matching (query image -> candidate image):

1. k-nearest neighbours (k = 2) by Hamming distance, brute force (exact).
2. Lowe's ratio test: keep ``i -> j1`` iff ``d1 < ratio * d2``. A feature whose best and
   second-best matches are almost equally good is *ambiguous* (typical of repeated
   structure such as windows or fence posts) and is discarded.
3. Optional absolute bound ``d1 <= max_hamming``.
4. Optional mutual (cross) check: ``j1``'s best match back in the query image must be
   ``i``, which also removes many-to-one matches.

Repeated structure. The ratio test removes individually ambiguous matches, but when a
scene is dominated by a repeating pattern (fences, tiled facades), the surviving matches
can be shifted by one period and still be geometrically consistent, so RANSAC cannot
catch the error. It is detected per image by self-similarity: the *twin fraction* is the
share of keypoints that have another keypoint in the same image, at least ``min_px``
away, whose descriptor is within ``hamming`` bits (see :func:`twin_fraction`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

DESCRIPTOR_BYTES = 32


@dataclass(frozen=True)
class OrbConfig:
    n_features: int = 2000
    scale_factor: float = 1.2
    n_levels: int = 8
    edge_threshold: int = 31
    fast_threshold: int = 20

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> OrbConfig:
        return cls(**dict(values or {}))

    def create(self) -> cv2.ORB:
        return cv2.ORB_create(
            nfeatures=self.n_features,
            scaleFactor=self.scale_factor,
            nlevels=self.n_levels,
            edgeThreshold=self.edge_threshold,
            patchSize=self.edge_threshold,
            fastThreshold=self.fast_threshold,
        )


@dataclass(frozen=True)
class MatchConfig:
    ratio: float = 0.8
    max_hamming: int | None = 64
    mutual: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> MatchConfig:
        return cls(**dict(values or {}))


@dataclass
class OrbFeatures:
    """Keypoints of one image. ``descriptors`` is ``(N, 32)`` uint8 (256-bit ORB)."""

    xy: np.ndarray  # (N, 2) float32 pixel coordinates
    size: np.ndarray  # (N,) keypoint diameter in pixels
    angle: np.ndarray  # (N,) orientation in degrees
    octave: np.ndarray  # (N,) pyramid level
    response: np.ndarray  # (N,) FAST/Harris score
    descriptors: np.ndarray  # (N, 32) uint8
    image_size: tuple[int, int]  # (width, height)
    _twin_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __len__(self) -> int:
        return int(self.xy.shape[0])

    @classmethod
    def empty(cls, image_size: tuple[int, int]) -> OrbFeatures:
        z = np.zeros(0, np.float32)
        return cls(
            np.zeros((0, 2), np.float32),
            z,
            z,
            np.zeros(0, np.int32),
            z,
            np.zeros((0, DESCRIPTOR_BYTES), np.uint8),
            image_size,
        )


@dataclass
class Matches:
    """Correspondences ``query_idx[m] <-> ref_idx[m]`` plus matching diagnostics."""

    query_idx: np.ndarray  # (M,) int
    ref_idx: np.ndarray  # (M,) int
    distance: np.ndarray  # (M,) Hamming distance of the kept match
    second_distance: np.ndarray  # (M,) distance of the second-best neighbour
    stats: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.query_idx.size)


def to_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got {image.dtype}")
    return np.ascontiguousarray(image)


def extract_orb(image: np.ndarray, config: OrbConfig | None = None) -> OrbFeatures:
    """Detect ORB keypoints and compute their binary descriptors (RGB or grey uint8)."""
    config = config or OrbConfig()
    gray = to_gray(image)
    size = (int(gray.shape[1]), int(gray.shape[0]))
    keypoints, descriptors = config.create().detectAndCompute(gray, None)
    if descriptors is None or not keypoints:
        return OrbFeatures.empty(size)
    return OrbFeatures(
        xy=np.array([k.pt for k in keypoints], dtype=np.float32),
        size=np.array([k.size for k in keypoints], dtype=np.float32),
        angle=np.array([k.angle for k in keypoints], dtype=np.float32),
        octave=np.array([k.octave for k in keypoints], dtype=np.int32),
        response=np.array([k.response for k in keypoints], dtype=np.float32),
        descriptors=np.ascontiguousarray(descriptors, dtype=np.uint8),
        image_size=size,
    )


def hamming_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between rows of ``a`` (N, B) and ``b`` (M, B) uint8."""
    a = np.unpackbits(np.asarray(a, np.uint8), axis=1)
    b = np.unpackbits(np.asarray(b, np.uint8), axis=1)
    # differing bits = |a| + |b| - 2 |a AND b|
    a_i, b_i = a.astype(np.int32), b.astype(np.int32)
    return a_i.sum(1)[:, None] + b_i.sum(1)[None, :] - 2 * (a_i @ b_i.T)


def _knn2(desc_a: np.ndarray, desc_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best index, best distance and second-best distance from each row of ``a`` into ``b``."""
    n = desc_a.shape[0]
    best_idx = np.full(n, -1, np.int64)
    d1 = np.full(n, np.inf)
    d2 = np.full(n, np.inf)
    if n == 0 or desc_b.shape[0] == 0:
        return best_idx, d1, d2
    knn = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(
        desc_a, desc_b, k=min(2, desc_b.shape[0])
    )
    for m in knn:
        if not m:
            continue
        i = m[0].queryIdx
        best_idx[i], d1[i] = m[0].trainIdx, m[0].distance
        if len(m) > 1:
            d2[i] = m[1].distance
    return best_idx, d1, d2


def match_features(
    query: OrbFeatures, ref: OrbFeatures, config: MatchConfig | None = None
) -> Matches:
    """Ratio-tested (and optionally mutual) Hamming matches from ``query`` into ``ref``.

    ``stats`` records the funnel: ``knn`` (query features with a nearest neighbour),
    ``ratio`` (survivors of the ratio test), ``max_hamming``, ``mutual`` and the
    ``ambiguous_fraction`` = share of query features with a *close* best match
    (``d1 <= max_hamming``) that were rejected by the ratio test. A high value signals
    repeated structure: many features have near-identical twins in the other image.
    """
    config = config or MatchConfig()
    best, d1, d2 = _knn2(query.descriptors, ref.descriptors)
    has_nn = best >= 0
    ratio_ok = has_nn & (d1 < config.ratio * d2)
    close = has_nn & (d1 <= (config.max_hamming if config.max_hamming is not None else 256))
    keep = ratio_ok & close
    stats = {
        "knn": int(has_nn.sum()),
        "ratio": int(ratio_ok.sum()),
        "max_hamming": int(keep.sum()),
        "close": int(close.sum()),
        "ambiguous_fraction": float((close & ~ratio_ok).sum() / close.sum())
        if close.any()
        else 0.0,
    }
    if config.mutual and keep.any():
        back, _, _ = _knn2(ref.descriptors, query.descriptors)
        idx = np.nonzero(keep)[0]
        keep[idx[back[best[idx]] != idx]] = False
    stats["mutual"] = int(keep.sum())
    q = np.nonzero(keep)[0]
    return Matches(
        query_idx=q,
        ref_idx=best[q],
        distance=d1[q].astype(np.float32),
        second_distance=d2[q].astype(np.float32),
        stats=stats,
    )


def twin_fraction(features: OrbFeatures, hamming: int = 24, min_px: float = 16.0) -> float:
    """Share of keypoints with a near-identical descriptor elsewhere in the same image.

    For each keypoint, its 8 nearest descriptors in the same image are searched; it is a
    *twin* if one of them (other than itself) is within ``hamming`` bits and at least
    ``min_px`` pixels away (closer ones are the same corner at another pyramid level).
    Cached per ``(hamming, min_px)`` on the feature object.
    """
    key = (int(hamming), float(min_px))
    if key in features._twin_cache:
        return features._twin_cache[key]
    n = len(features)
    if n < 2:
        value = 0.0
    else:
        knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
            features.descriptors, features.descriptors, k=min(8, n)
        )
        twins = 0
        for row in knn:
            for m in row:
                if m.distance > hamming:
                    break
                if m.trainIdx == m.queryIdx:
                    continue
                if np.hypot(*(features.xy[m.queryIdx] - features.xy[m.trainIdx])) >= min_px:
                    twins += 1
                    break
        value = twins / n
    features._twin_cache[key] = value
    return value


def raw_nearest_matches(query: OrbFeatures, ref: OrbFeatures) -> Matches:
    """Unfiltered nearest-neighbour matches (for visualising what the filters remove)."""
    best, d1, d2 = _knn2(query.descriptors, ref.descriptors)
    q = np.nonzero(best >= 0)[0]
    return Matches(
        q, best[q], d1[q].astype(np.float32), d2[q].astype(np.float32), {"knn": int(q.size)}
    )
