"""Two-view geometry for loop-closure verification: essential matrix, RANSAC, degeneracy.

Notation. ``K`` is the calibrated intrinsic matrix; a pixel ``u`` maps to the normalised
image point ``x = K^{-1} [u, 1]^T``. For the same 3D point seen in the query image
(``x_q``) and the candidate image (``x_r``) of a camera that moved rigidly,
``X_r = R X_q + t``, the epipolar constraint holds:

    x_r^T E x_q = 0,      E = [t]_x R,

where ``[t]_x`` is the cross-product matrix. ``E`` has two equal non-zero singular values
and one zero singular value, is defined only up to scale, and ``t`` is recovered only as a
unit **direction**: two-view geometry cannot observe metric translation scale.

RANSAC. Given ``M`` tentative matches, repeatedly fit ``E`` to a random minimal sample,
count matches whose Sampson distance (first-order geometric error, see ``sampson_sq``)
is below the threshold, and keep the model with the most inliers. With inlier fraction
``w`` and sample size ``s`` the number of iterations needed to draw at least one
all-inlier sample with probability ``p`` is ``N = log(1 - p) / log(1 - w^s)``.

Two estimators are provided:

* ``opencv`` (default): ``cv2.findEssentialMat`` (Nister's five-point solver, s = 5,
  inside OpenCV's RANSAC).
* ``numpy``: this module's own vectorised RANSAC around the normalised linear eight-point
  algorithm (s = 8) with adaptive stopping and least-squares local optimisation. It is a
  transparent reference implementation that is cross-checked against OpenCV in the tests.

Degeneracy. Two cases make ``E`` (and hence ``R``, ``t``) unreliable even when the
matches themselves are correct:

* **Pure rotation / low parallax**: when ``t`` is negligible relative to scene depth,
  every match also satisfies ``u_r ~ K R K^{-1} u_q``, a homography that is a conjugated
  rotation. Any ``t`` then fits, so ``E`` is not determined.
* **Planar scene**: all inliers explained by a single homography; the five-point solver
  still works, but the eight-point solver is degenerate and pose is less well conditioned.

They are detected with an optional homography diagnostic (``cv2.findHomography`` on the
same matches): if the homography explains almost as many matches as ``E``
(``h_ratio = n_H / n_E >= homography_dominance``), the configuration is degenerate; it is
rotation-type if ``K^{-1} H K`` is close to a scaled rotation, else planar. Independently,
the median rotation-compensated parallax angle ``angle(f_r, R f_q)`` of inlier bearings
``f`` is reported and flags low parallax below ``min_parallax_deg``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from latent_loop.datasets.base import CameraCalibration
from latent_loop.verification.orb import (
    MatchConfig,
    Matches,
    OrbConfig,
    OrbFeatures,
    extract_orb,
    match_features,
    twin_fraction,
)

ESTIMATORS = ("opencv", "numpy")
DEGENERATE_POLICIES = ("flag", "reject")
REASONS = (
    "accepted",
    "invalid_calibration",
    "missing_descriptors",
    "insufficient_matches",
    "repeated_structure",
    "ransac_failure",
    "too_few_inliers",
    "low_inlier_ratio",
    "concentrated_inliers",
    "pure_rotation",
    "low_parallax",
    "planar_scene",
)


# ----------------------------------------------------------------------------- config


@dataclass(frozen=True)
class VerificationConfig:
    """Acceptance thresholds. Defaults are common heuristic values, *not* tuned on KITTI."""

    orb: OrbConfig = field(default_factory=OrbConfig)
    matching: MatchConfig = field(default_factory=MatchConfig)
    min_keypoints: int = 50  # per image, else "missing_descriptors"
    min_matches: int = 20  # after ratio + mutual check
    twin_hamming: int = 24  # self-similarity: descriptor distance of a "twin" keypoint
    twin_min_px: float = 16.0  # ... at least this far away in the same image
    max_repeated_fraction: float = 0.5  # twin fraction (either image) -> "repeated_structure"
    estimator: str = "opencv"
    ransac_threshold_px: float = 1.0
    ransac_confidence: float = 0.999
    ransac_max_iters: int = 2000
    min_inliers: int = 30
    min_inlier_ratio: float = 0.3
    coverage_grid: tuple[int, int] = (8, 4)  # (columns, rows)
    min_coverage: float = 0.1  # fraction of grid cells holding an inlier, in both images
    homography_diagnostic: bool = True
    # One-sided 2D transfer error vs 1D Sampson distance: sqrt(5.99/3.84) * sqrt(2) ~ 1.8x.
    homography_threshold_px: float = 2.0
    homography_dominance: float = 0.9
    rotation_tolerance: float = 0.05  # singular-value spread of K^-1 H K for "rotation-like"
    pure_rotation_min_deg: float = 2.0
    min_parallax_deg: float = 1.0
    degenerate_policy: str = "flag"  # flag: accept, mark pose unreliable | reject
    seed: int = 0

    def __post_init__(self) -> None:
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}")
        if self.degenerate_policy not in DEGENERATE_POLICIES:
            raise ValueError(f"degenerate_policy must be one of {DEGENERATE_POLICIES}")
        if self.min_matches < 8:
            raise ValueError("min_matches must be >= 8 (minimal sample of the 8-point solver)")
        if self.ransac_threshold_px <= 0:
            raise ValueError("ransac_threshold_px must be > 0")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> VerificationConfig:
        values = dict(values or {})
        values.pop("verify_top_k", None)  # pipeline option, not a pair-level threshold
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown verification keys: {sorted(unknown)}")
        if "orb" in values:
            values["orb"] = OrbConfig.from_dict(values["orb"])
        if "matching" in values:
            values["matching"] = MatchConfig.from_dict(values["matching"])
        if "coverage_grid" in values:
            values["coverage_grid"] = tuple(values["coverage_grid"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------- geometry


def calibration_is_valid(K: np.ndarray, image_size: tuple[int, int]) -> bool:
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        return False
    width, height = image_size
    return bool(
        K[0, 0] > 0
        and K[1, 1] > 0
        and 0 <= K[0, 2] <= width
        and 0 <= K[1, 2] <= height
        and np.allclose(K[2], [0, 0, 1])
        and abs(K[1, 0]) < 1e-9
    )


def normalize_points(uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels ``(N, 2)`` -> normalised image coordinates ``(N, 2)`` via ``K^{-1}``."""
    uv = np.asarray(uv, dtype=np.float64)
    hom = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
    return hom[:, :2] / hom[:, 2:]


def bearings(x: np.ndarray) -> np.ndarray:
    f = np.c_[x, np.ones(len(x))]
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)


def project_to_essential(E: np.ndarray) -> np.ndarray:
    """Closest essential matrix (singular values ``(1, 1, 0)``); works on ``(..., 3, 3)``."""
    U, _, Vt = np.linalg.svd(E)
    return U @ (np.array([1.0, 1.0, 0.0])[..., :, None] * Vt)


def eight_point(x_q: np.ndarray, x_r: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Linear eight-point estimate(s) of E from normalised points.

    ``x_q``, ``x_r``: ``(..., n, 2)`` with ``n >= 8``. Each correspondence gives one row of
    ``A e = 0`` with ``e = vec(E)`` (row-major); the least-squares solution is the right
    singular vector of ``A`` with the smallest singular value, projected onto the
    essential manifold. Hartley normalisation (centre each point set, scale to mean
    distance sqrt(2)) is applied first; without it the linear system is badly
    conditioned and the fit is several times less accurate. Optional per-correspondence
    ``weights`` scale the rows of ``A`` (weighted least squares).
    """
    Tq, nq = _hartley(x_q)
    Tr, nr = _hartley(x_r)
    xq, yq = nq[..., 0], nq[..., 1]
    xr, yr = nr[..., 0], nr[..., 1]
    one = np.ones_like(xq)
    A = np.stack([xr * xq, xr * yq, xr, yr * xq, yr * yq, yr, xq, yq, one], axis=-1)
    if weights is not None:
        A = A * weights[..., None]
    _, _, Vt = np.linalg.svd(A)
    En = Vt[..., -1, :].reshape(*A.shape[:-2], 3, 3)
    E = np.swapaxes(Tr, -1, -2) @ En @ Tq
    E = project_to_essential(E)
    return E / np.linalg.norm(E, axis=(-2, -1), keepdims=True)


def _hartley(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Similarity ``T`` (``(..., 3, 3)``) and transformed points ``T x`` (``(..., n, 2)``)."""
    centre = x.mean(axis=-2, keepdims=True)
    d = np.linalg.norm(x - centre, axis=-1).mean(axis=-1)
    scale = np.sqrt(2.0) / np.maximum(d, 1e-12)
    T = np.zeros((*x.shape[:-2], 3, 3))
    T[..., 0, 0] = T[..., 1, 1] = scale
    T[..., 0, 2] = -scale * centre[..., 0, 0]
    T[..., 1, 2] = -scale * centre[..., 0, 1]
    T[..., 2, 2] = 1.0
    return T, (x - centre) * scale[..., None, None]


def sampson_sq(E: np.ndarray, x_q: np.ndarray, x_r: np.ndarray) -> np.ndarray:
    """Squared Sampson distance of every correspondence under every model.

    For ``x`` homogeneous and ``Ex_q = E x_q``, ``E^T x_r``:

        d^2 = (x_r^T E x_q)^2 / ((E x_q)_1^2 + (E x_q)_2^2 + (E^T x_r)_1^2 + (E^T x_r)_2^2)

    ``E``: ``(H, 3, 3)`` or ``(3, 3)``; points ``(N, 2)``. Returns ``(H, N)`` or ``(N,)``.
    """
    hq = np.c_[x_q, np.ones(len(x_q))]
    hr = np.c_[x_r, np.ones(len(x_r))]
    Exq = np.einsum("...ij,nj->...ni", E, hq)
    Etxr = np.einsum("...ji,nj->...ni", E, hr)
    num = np.einsum("ni,...ni->...n", hr, Exq) ** 2
    den = Exq[..., 0] ** 2 + Exq[..., 1] ** 2 + Etxr[..., 0] ** 2 + Etxr[..., 1] ** 2
    return num / np.maximum(den, 1e-300)


def sampson_den(E: np.ndarray, x_q: np.ndarray, x_r: np.ndarray) -> np.ndarray:
    """Squared gradient norm of the epipolar residual (denominator of ``sampson_sq``)."""
    hq = np.c_[x_q, np.ones(len(x_q))]
    hr = np.c_[x_r, np.ones(len(x_r))]
    Exq = hq @ E.T
    Etxr = hr @ E
    return Exq[:, 0] ** 2 + Exq[:, 1] ** 2 + Etxr[:, 0] ** 2 + Etxr[:, 1] ** 2


def ransac_iterations(inlier_fraction: float, sample_size: int, confidence: float) -> float:
    w = min(max(inlier_fraction, 0.0), 1.0)
    if w >= 1.0:
        return 1.0
    if w <= 0.0:
        return math.inf
    denom = math.log(1.0 - w**sample_size)
    return math.inf if denom == 0 else math.log(1.0 - confidence) / denom


@dataclass
class EssentialEstimate:
    E: np.ndarray | None
    inliers: np.ndarray  # (M,) bool
    iterations: int
    backend: str

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())


def ransac_essential_numpy(
    x_q: np.ndarray,
    x_r: np.ndarray,
    threshold: float,
    confidence: float = 0.999,
    max_iters: int = 2000,
    seed: int = 0,
    batch: int = 256,
    local_opt_rounds: int = 10,
) -> EssentialEstimate:
    """RANSAC + eight-point on normalised points; ``threshold`` is in normalised units."""
    m = len(x_q)
    s = 8
    empty = EssentialEstimate(None, np.zeros(m, bool), 0, "numpy")
    if m < s:
        return empty
    rng = np.random.default_rng(seed)
    thr2 = threshold**2
    best_E, best_mask, best_n = None, np.zeros(m, bool), 0
    best_raw = 0  # best support of a *minimal-sample* model (LO is triggered on it)
    needed, done = float(max_iters), 0
    while done < min(needed, max_iters):
        h = int(min(batch, max_iters - done))
        samples = np.argpartition(rng.random((h, m)), s - 1, axis=1)[:, :s]
        with np.errstate(all="ignore"):
            Es = eight_point(x_q[samples], x_r[samples])
            err = sampson_sq(Es, x_q, x_r)
        counts = (err < thr2).sum(1)
        counts[~np.all(np.isfinite(Es), axis=(1, 2))] = 0
        done += h
        for j in np.argsort(counts)[::-1][:4]:  # LO on the batch's best few new records
            if counts[j] <= best_raw:
                break
            E, mask = _local_optimisation(Es[j], err[j], x_q, x_r, threshold, local_opt_rounds)
            if mask.sum() > best_n:
                best_n, best_E, best_mask = int(mask.sum()), E, mask
        best_raw = max(best_raw, int(counts.max()))
        if best_n:
            # Stop on the inlier fraction of the best *minimal-sample* model: an
            # all-inlier 8-point sample is not guaranteed to give a good model under noise.
            needed = ransac_iterations(best_raw / m, s, confidence)
    if best_E is None or best_n < s:
        return EssentialEstimate(None, np.zeros(m, bool), done, "numpy")
    return EssentialEstimate(best_E, best_mask, done, "numpy")


def _local_optimisation(
    E: np.ndarray,
    err: np.ndarray,
    x_q: np.ndarray,
    x_r: np.ndarray,
    threshold: float,
    rounds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """LO-RANSAC step: iteratively reweighted least squares on the model's support.

    A model fitted to a minimal sample is noisy, so its support misses true inliers.
    The eight-point solver minimises the *algebraic* residual ``x_r^T E x_q``; dividing
    each row by the Sampson denominator ``sqrt(den)`` of the current model turns that
    residual into the Sampson distance used for scoring (Sampson-weighted IRLS). The
    support is taken at twice the threshold, the refit is scored at the threshold, and
    the loop repeats while the score improves. The result is never worse than the
    minimal-sample model.
    """
    thr2 = threshold**2
    best_E, best_mask = E, err < thr2
    current = E
    for _ in range(rounds):
        support = sampson_sq(current, x_q, x_r) < 4 * thr2
        if support.sum() < 8:
            break
        with np.errstate(all="ignore"):
            w = 1.0 / np.sqrt(np.maximum(sampson_den(current, x_q[support], x_r[support]), 1e-300))
            cand = eight_point(x_q[support], x_r[support], w / w.max())
            mask = sampson_sq(cand, x_q, x_r) < thr2
        if not np.all(np.isfinite(cand)) or mask.sum() <= best_mask.sum():
            break
        best_E, best_mask, current = cand, mask, cand
    return best_E, best_mask


def estimate_essential(
    uv_q: np.ndarray, uv_r: np.ndarray, K: np.ndarray, config: VerificationConfig
) -> EssentialEstimate:
    """Robustly estimate ``E`` (``x_r^T E x_q = 0``) from pixel correspondences."""
    m = len(uv_q)
    if config.estimator == "numpy":
        focal = 0.5 * (K[0, 0] + K[1, 1])
        return ransac_essential_numpy(
            normalize_points(uv_q, K),
            normalize_points(uv_r, K),
            config.ransac_threshold_px / focal,
            config.ransac_confidence,
            config.ransac_max_iters,
            config.seed,
        )
    if m < 5:
        return EssentialEstimate(None, np.zeros(m, bool), 0, "opencv")
    cv2.setRNGSeed(config.seed)
    try:
        E, mask = cv2.findEssentialMat(
            np.asarray(uv_q, np.float64),
            np.asarray(uv_r, np.float64),
            np.asarray(K, np.float64),
            method=cv2.RANSAC,
            prob=config.ransac_confidence,
            threshold=config.ransac_threshold_px,
            maxIters=config.ransac_max_iters,
        )
    except cv2.error:
        E, mask = None, None
    if E is None or mask is None or E.shape[0] < 3:
        return EssentialEstimate(None, np.zeros(m, bool), 0, "opencv")
    return EssentialEstimate(E[:3].astype(np.float64), mask.ravel().astype(bool), -1, "opencv")


def recover_pose(
    E: np.ndarray, uv_q: np.ndarray, uv_r: np.ndarray, K: np.ndarray, inliers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose ``E`` into ``R`` and a unit ``t`` (``X_r = R X_q + t``) by cheirality.

    Returns ``R``, ``t`` (unit norm; metric scale is unobservable) and the boolean mask of
    inliers that triangulate in front of both cameras.
    """
    mask = inliers.astype(np.uint8).reshape(-1, 1).copy()
    _, R, t, mask_out = cv2.recoverPose(
        E, np.asarray(uv_q, np.float64), np.asarray(uv_r, np.float64), K, mask=mask
    )
    t = t.ravel()
    return R, t / max(np.linalg.norm(t), 1e-12), mask_out.ravel() > 0


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def parallax_deg(x_q: np.ndarray, x_r: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotation-compensated parallax: angle between ``f_r`` and ``R f_q`` per match."""
    cos = np.einsum("ni,ni->n", bearings(x_r), bearings(x_q) @ R.T)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def homography_rotation_check(H: np.ndarray, K: np.ndarray) -> tuple[float, np.ndarray]:
    """Singular-value spread of ``K^{-1} H K`` (0 for a pure rotation) and nearest rotation.

    For a rotation-only camera motion ``H ~ K R K^{-1}``, so ``K^{-1} H K`` is a scaled
    rotation with three equal singular values. The nearest rotation (polar
    decomposition ``U V^T``) gives the rotation angle.
    """
    Hn = np.linalg.inv(K) @ H @ K
    U, S, Vt = np.linalg.svd(Hn)
    spread = float((S[0] - S[2]) / S[1])
    Rh = U @ Vt
    if np.linalg.det(Rh) < 0:
        Rh = -Rh
    return spread, Rh


def grid_coverage(uv: np.ndarray, image_size: tuple[int, int], grid: tuple[int, int]) -> float:
    """Fraction of ``grid`` cells (columns, rows) containing at least one point."""
    if len(uv) == 0:
        return 0.0
    cols, rows = grid
    w, h = image_size
    cx = np.clip((uv[:, 0] / w * cols).astype(int), 0, cols - 1)
    cy = np.clip((uv[:, 1] / h * rows).astype(int), 0, rows - 1)
    return float(np.unique(cy * cols + cx).size / (cols * rows))


# ----------------------------------------------------------------------------- decision


@dataclass
class VerificationResult:
    """Outcome of verifying one (query, candidate) pair. Poses are never an input."""

    accepted: bool
    reason: str
    n_keypoints_query: int = 0
    n_keypoints_ref: int = 0
    n_knn: int = 0
    n_ratio: int = 0
    n_matches: int = 0  # after ratio test, Hamming bound and mutual check
    ambiguous_fraction: float = 0.0
    repeated_fraction: float = 0.0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    coverage: float = 0.0
    n_cheirality: int = 0
    rotation_deg: float = float("nan")
    t_direction: tuple[float, float, float] = (float("nan"),) * 3
    parallax_deg: float = float("nan")
    h_inliers: int = -1  # -1: homography diagnostic not run
    h_ratio: float = float("nan")
    h_spread: float = float("nan")
    degenerate: str = ""  # "", "pure_rotation", "low_parallax" or "planar_scene"
    pose_reliable: bool = False
    timing_ms: dict[str, float] = field(default_factory=dict)
    # Optional arrays for visualisation (not serialised to CSV).
    matches: Matches | None = None
    inlier_mask: np.ndarray | None = None
    E: np.ndarray | None = None
    H: np.ndarray | None = None
    h_mask: np.ndarray | None = None

    def record(self) -> dict[str, Any]:
        """Flat scalar fields for tables (CSV/JSON)."""
        out = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("matches", "inlier_mask", "E", "H", "h_mask", "timing_ms", "t_direction")
        }
        out.update({f"t_{a}": float(v) for a, v in zip("xyz", self.t_direction, strict=True)})
        out.update({f"ms_{k}": round(v, 3) for k, v in self.timing_ms.items()})
        return out


def verify_features(
    query: OrbFeatures,
    ref: OrbFeatures,
    calibration: CameraCalibration,
    config: VerificationConfig | None = None,
    keep_details: bool = False,
) -> VerificationResult:
    """Verify a candidate pair from precomputed ORB features.

    Checks run in order and the first failing one is the rejection ``reason``:
    invalid calibration -> missing descriptors -> repeated structure -> insufficient
    matches -> RANSAC failure -> too few inliers -> low inlier ratio -> concentrated
    inliers -> degenerate geometry (only rejected with ``degenerate_policy="reject"``).
    """
    import time

    config = config or VerificationConfig()
    K = np.asarray(calibration.K, dtype=np.float64)
    res = VerificationResult(False, "", len(query), len(ref))
    timing: dict[str, float] = {}
    res.timing_ms = timing

    def done(reason: str, accepted: bool = False) -> VerificationResult:
        res.reason, res.accepted = reason, accepted
        return res

    if not calibration_is_valid(K, calibration.image_size):
        return done("invalid_calibration")
    if min(len(query), len(ref)) < config.min_keypoints:
        return done("missing_descriptors")
    res.repeated_fraction = max(
        twin_fraction(query, config.twin_hamming, config.twin_min_px),
        twin_fraction(ref, config.twin_hamming, config.twin_min_px),
    )
    if res.repeated_fraction >= config.max_repeated_fraction:
        return done("repeated_structure")

    t0 = time.perf_counter()
    matches = match_features(query, ref, config.matching)
    timing["match"] = 1000 * (time.perf_counter() - t0)
    res.n_knn, res.n_ratio = matches.stats["knn"], matches.stats["ratio"]
    res.n_matches, res.ambiguous_fraction = len(matches), matches.stats["ambiguous_fraction"]
    if keep_details:
        res.matches = matches
    if len(matches) < config.min_matches:
        return done("insufficient_matches")

    uv_q = query.xy[matches.query_idx].astype(np.float64)
    uv_r = ref.xy[matches.ref_idx].astype(np.float64)
    t0 = time.perf_counter()
    est = estimate_essential(uv_q, uv_r, K, config)
    timing["ransac"] = 1000 * (time.perf_counter() - t0)
    if keep_details:
        res.inlier_mask, res.E = est.inliers, est.E
    if est.E is None or not np.all(np.isfinite(est.E)) or est.n_inliers < 5:
        return done("ransac_failure")
    res.n_inliers = est.n_inliers
    res.inlier_ratio = est.n_inliers / len(matches)
    res.coverage = min(
        grid_coverage(uv_q[est.inliers], query.image_size, config.coverage_grid),
        grid_coverage(uv_r[est.inliers], ref.image_size, config.coverage_grid),
    )

    t0 = time.perf_counter()
    R, t, front = recover_pose(est.E, uv_q, uv_r, K, est.inliers)
    res.n_cheirality = int(front.sum())
    res.rotation_deg = rotation_angle_deg(R)
    res.t_direction = tuple(float(v) for v in t)
    use = front if front.sum() >= 5 else est.inliers
    res.parallax_deg = float(
        np.median(parallax_deg(normalize_points(uv_q[use], K), normalize_points(uv_r[use], K), R))
    )
    degenerate = ""
    if config.homography_diagnostic:
        H, hmask = cv2.findHomography(uv_q, uv_r, cv2.RANSAC, config.homography_threshold_px)
        if H is not None and np.all(np.isfinite(H)):
            hmask = hmask.ravel().astype(bool)
            res.h_inliers = int(hmask.sum())
            res.h_ratio = res.h_inliers / max(res.n_inliers, 1)
            res.h_spread, Rh = homography_rotation_check(H, K)
            if keep_details:
                res.H, res.h_mask = H, hmask
            if res.h_ratio >= config.homography_dominance:
                if res.h_spread <= config.rotation_tolerance:
                    # E (and its R) is undetermined here; report the homography's rotation.
                    res.rotation_deg = rotation_angle_deg(Rh)
                    sel = est.inliers
                    res.parallax_deg = float(
                        np.median(
                            parallax_deg(
                                normalize_points(uv_q[sel], K), normalize_points(uv_r[sel], K), Rh
                            )
                        )
                    )
                    big_turn = res.rotation_deg >= config.pure_rotation_min_deg
                    degenerate = "pure_rotation" if big_turn else "low_parallax"
                else:
                    degenerate = "planar_scene"
        else:
            res.h_inliers = 0
    if not degenerate and res.parallax_deg < config.min_parallax_deg:
        degenerate = "low_parallax"
    res.degenerate = degenerate
    res.pose_reliable = not degenerate
    timing["pose"] = 1000 * (time.perf_counter() - t0)

    if res.n_inliers < config.min_inliers:
        return done("too_few_inliers")
    if res.inlier_ratio < config.min_inlier_ratio:
        return done("low_inlier_ratio")
    if res.coverage < config.min_coverage:
        return done("concentrated_inliers")
    if degenerate and config.degenerate_policy == "reject":
        return done(degenerate)
    return done("accepted", accepted=True)


def verify_pair(
    image_q: np.ndarray,
    image_r: np.ndarray,
    calibration: CameraCalibration,
    config: VerificationConfig | None = None,
    keep_details: bool = False,
) -> tuple[VerificationResult, OrbFeatures, OrbFeatures]:
    """Extract ORB on both images and verify the pair."""
    config = config or VerificationConfig()
    fq = extract_orb(image_q, config.orb)
    fr = extract_orb(image_r, config.orb)
    return verify_features(fq, fr, calibration, config, keep_details), fq, fr
