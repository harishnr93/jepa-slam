"""Phase 4: ORB matching, essential-matrix RANSAC, degeneracy handling and the pipeline.

Image pairs are rendered synthetically with known geometry (plane-induced homographies
``H = K (R + t n^T / d) K^-1``), so every expected outcome follows from the construction.
"""

from __future__ import annotations

import inspect

import cv2
import numpy as np
import pandas as pd
import pytest

from latent_loop.datasets.base import CameraCalibration
from latent_loop.datasets.ground_truth import IGNORED, NEGATIVE, POSITIVE
from latent_loop.retrieval import RetrievalConfig, RetrievalResult
from latent_loop.verification import (
    REASONS,
    FeatureStore,
    MatchConfig,
    OrbFeatures,
    VerificationConfig,
    estimate_essential,
    extract_orb,
    hamming_distance,
    label_pairs,
    match_features,
    summarize,
    verify_candidates,
    verify_features,
    verify_pair,
)
from latent_loop.verification import geometry as geo
from latent_loop.verification.orb import twin_fraction

W, H, F = 640, 240, 500.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
CAL = CameraCalibration(K, (W, H))


# ----------------------------------------------------------------------------- fixtures


def texture(seed: int, w: int = W, h: int = H) -> np.ndarray:
    """Multi-scale smoothed noise: rich in corners, no repeated structure."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    for s in (2, 4, 8, 16):
        n = rng.random((h // s + 2, w // s + 2)).astype(np.float32)
        big = cv2.resize(n, (w + 2 * s, h + 2 * s), interpolation=cv2.INTER_CUBIC)
        img += big[s : s + h, s : s + w] * s
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)


def rotation_y(deg: float) -> np.ndarray:
    return cv2.Rodrigues(np.array([0.0, np.radians(deg), 0.0]))[0]


def plane_homography(R, t, d) -> np.ndarray:
    n = np.array([0.0, 0.0, 1.0])
    return K @ (R + np.outer(t, n) / d) @ np.linalg.inv(K)


def two_plane_pair(seed: int = 0):
    """Left half of the view at 4 m, the rest at 25 m; camera translates 0.63 m."""
    R, t = rotation_y(3.0), np.array([0.6, 0.0, 0.2])
    img1 = texture(seed)
    near_mask = np.zeros_like(img1)
    near_mask[:, : W // 2] = 255
    h_far, h_near = plane_homography(R, t, 25.0), plane_homography(R, t, 4.0)
    img2 = cv2.warpPerspective(img1, h_far, (W, H))
    near = cv2.warpPerspective(img1, h_near, (W, H))
    m = cv2.warpPerspective(near_mask, h_near, (W, H)) > 127
    img2[m] = near[m]
    return img1, img2


def rotation_pair(yaw_deg: float, seed: int = 0):
    img1 = texture(seed)
    return img1, cv2.warpPerspective(img1, K @ rotation_y(yaw_deg) @ np.linalg.inv(K), (W, H))


def periodic_pair():
    tile = texture(5, 32, 32)
    per = np.tile(tile, (H // 32 + 1, W // 32 + 2))[:H]
    return per[:, :W], per[:, 13 : 13 + W]


def point_scene(seed: int, outlier_fraction: float = 0.4, noise_px: float = 0.4):
    """Projected 3D points (KITTI-like intrinsics) with known R, t and outliers."""
    rng = np.random.default_rng(seed)
    k = np.array([[718.856, 0, 607.19], [0, 718.856, 185.2], [0, 0, 1]])
    R = rotation_y(rng.uniform(-10, 10))
    t = np.array([rng.uniform(-2, 2), 0.1, rng.uniform(0.5, 2)])
    X = np.c_[rng.uniform(-15, 15, 600), rng.uniform(-3, 2, 600), rng.uniform(4, 25, 600)]

    def proj(P):
        u = P @ k.T
        return u[:, :2] / u[:, 2:]

    uq, ur = proj(X), proj(X @ R.T + t)
    ok = np.all((uq > 0) & (uq < [1241, 376]) & (ur > 0) & (ur < [1241, 376]), axis=1)
    uq, ur = uq[ok], ur[ok]
    uq = uq + rng.normal(0, noise_px, uq.shape)
    ur = ur + rng.normal(0, noise_px, ur.shape)
    n_out = int(outlier_fraction * len(uq))
    ur[:n_out] = np.c_[rng.uniform(0, 1241, n_out), rng.uniform(0, 376, n_out)]
    return uq, ur, k, R, t, np.arange(len(uq)) >= n_out


# ----------------------------------------------------------------------------- ORB + matching


def test_hamming_distance_matches_opencv():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (20, 32), dtype=np.uint8)
    b = rng.integers(0, 256, (15, 32), dtype=np.uint8)
    d = hamming_distance(a, b)
    for i in range(0, 20, 7):
        for j in range(0, 15, 4):
            assert d[i, j] == cv2.norm(a[i], b[j], cv2.NORM_HAMMING)
    assert np.all(np.diag(hamming_distance(a, a)) == 0)


def _features(desc: np.ndarray, xy: np.ndarray | None = None) -> OrbFeatures:
    n = len(desc)
    xy = np.c_[np.arange(n) * 40.0, np.zeros(n)] if xy is None else xy
    z = np.zeros(n, np.float32)
    return OrbFeatures(xy.astype(np.float32), z, z, z.astype(np.int32), z, desc, (W, H))


def test_ratio_test_and_mutual_check():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, (3, 32), dtype=np.uint8)
    flip = np.zeros(32, np.uint8)
    flip[0] = 0b111  # 3 bits differ
    flip4 = np.zeros(32, np.uint8)
    flip4[1] = 0b1111  # 4 other bits differ
    ref = np.stack([base[0], base[1] ^ flip, base[1] ^ flip4, base[2] ^ flip])
    # base[0]: exact unique match (d1=0). base[1]: two candidates at d=3 and d=4, so
    # 3 < 0.7 * 4 fails the ratio test (ambiguous). base[2]: unique at d=3.
    m = match_features(_features(base), _features(ref), MatchConfig(ratio=0.7, mutual=False))
    assert set(m.query_idx) == {0, 2}
    assert m.ref_idx[m.query_idx == 0][0] == 0
    assert m.stats["ambiguous_fraction"] == pytest.approx(1 / 3)
    # Mutual check: two queries whose best match is the same reference keep only one.
    q = np.stack([base[2], base[2] ^ (flip << 3)])
    m = match_features(_features(q), _features(ref[3:]), MatchConfig(ratio=1.0, mutual=True))
    assert len(m) == 1


def test_orb_on_blank_and_textured_images():
    assert len(extract_orb(np.full((H, W), 128, np.uint8))) == 0
    f = extract_orb(np.dstack([texture(0)] * 3))  # RGB input is converted to grey
    assert len(f) > 1000 and f.descriptors.shape[1] == 32 and f.descriptors.dtype == np.uint8


def test_twin_fraction_detects_repetition():
    a, _ = periodic_pair()
    assert twin_fraction(extract_orb(a)) > 0.9
    assert twin_fraction(extract_orb(texture(0))) < 0.1


# ----------------------------------------------------------------------------- geometry


def test_eight_point_exact_without_noise_and_sampson_zero():
    uq, ur, k, R, t, _ = point_scene(0, outlier_fraction=0.0, noise_px=0.0)
    xq, xr = geo.normalize_points(uq, k), geo.normalize_points(ur, k)
    E_true = geo.skew(t) @ R
    E_true /= np.linalg.norm(E_true)
    E = geo.eight_point(xq[:8], xr[:8])
    assert min(np.linalg.norm(E - E_true), np.linalg.norm(E + E_true)) < 1e-8
    assert np.max(geo.sampson_sq(E_true, xq, xr)) < 1e-20
    s = np.linalg.svd(E, compute_uv=False)
    assert np.allclose(s, [s[0], s[0], 0], atol=1e-12)


def test_ransac_iterations_formula():
    assert geo.ransac_iterations(0.5, 8, 0.99) == pytest.approx(np.log(0.01) / np.log(1 - 0.5**8))
    assert geo.ransac_iterations(1.0, 5, 0.999) == 1.0
    assert geo.ransac_iterations(0.0, 5, 0.999) == np.inf


@pytest.mark.parametrize("estimator", ["opencv", "numpy"])
def test_ransac_recovers_inliers_and_pose(estimator):
    tps, fps, rot_err = [], [], []
    for seed in range(5):
        uq, ur, k, R, t, truth = point_scene(seed)
        est = estimate_essential(uq, ur, k, VerificationConfig(estimator=estimator, seed=seed))
        tps.append((est.inliers & truth).sum() / truth.sum())
        fps.append((est.inliers & ~truth).sum() / (~truth).sum())
        R_est, t_est, _ = geo.recover_pose(est.E, uq, ur, k, est.inliers)
        rot_err.append(geo.rotation_angle_deg(R_est.T @ R))
        assert np.linalg.norm(t_est) == pytest.approx(1.0)  # direction only, no scale
    assert np.mean(tps) > (0.85 if estimator == "opencv" else 0.7)
    assert np.mean(fps) < 0.02
    assert np.median(rot_err) < 1.5


def test_numpy_ransac_returns_none_with_too_few_points():
    est = geo.ransac_essential_numpy(np.zeros((5, 2)), np.zeros((5, 2)), 1e-3)
    assert est.E is None and est.n_inliers == 0


# ----------------------------------------------------------------------------- decisions


def test_accepts_translating_two_plane_scene():
    res, _, _ = verify_pair(*two_plane_pair(), CAL)
    assert res.accepted and res.reason == "accepted"
    assert res.degenerate == "" and res.pose_reliable
    assert res.n_inliers >= 300 and res.inlier_ratio > 0.8
    assert res.h_ratio < 0.8  # one homography cannot explain two planes


@pytest.mark.parametrize("estimator", ["opencv", "numpy"])
def test_pure_rotation_flagged_or_rejected(estimator):
    a, b = rotation_pair(6.0)
    res, _, _ = verify_pair(a, b, CAL, VerificationConfig(estimator=estimator))
    assert res.accepted and res.degenerate == "pure_rotation" and not res.pose_reliable
    assert res.rotation_deg == pytest.approx(6.0, abs=0.5)
    assert res.parallax_deg < 1.0
    res, _, _ = verify_pair(a, b, CAL, VerificationConfig(degenerate_policy="reject"))
    assert not res.accepted and res.reason == "pure_rotation"


def test_low_parallax_flagged():
    a = texture(0)
    res, _, _ = verify_pair(a, np.roll(a, 3, axis=1), CAL)
    assert res.degenerate == "low_parallax" and not res.pose_reliable
    res, _, _ = verify_pair(
        *rotation_pair(0.5), CAL, VerificationConfig(degenerate_policy="reject")
    )
    assert res.reason == "low_parallax"


def test_missing_descriptors():
    res, _, _ = verify_pair(np.full((H, W), 128, np.uint8), texture(1), CAL)
    assert res.reason == "missing_descriptors" and res.n_keypoints_query == 0


def test_unrelated_images_rejected():
    res, _, _ = verify_pair(texture(1), texture(2), CAL)
    assert not res.accepted
    assert res.reason in ("insufficient_matches", "too_few_inliers", "low_inlier_ratio")


def test_insufficient_matches():
    res, _, _ = verify_pair(texture(1), texture(2), CAL, VerificationConfig(min_matches=500))
    assert res.reason == "insufficient_matches"


def test_repeated_structure():
    res, _, _ = verify_pair(*periodic_pair(), CAL)
    assert res.reason == "repeated_structure" and res.repeated_fraction > 0.9


def test_invalid_calibration():
    bad = CameraCalibration(np.array([[0.0, 0, 320], [0, 500, 120], [0, 0, 1]]), (W, H))
    res, _, _ = verify_pair(*two_plane_pair(), bad)
    assert res.reason == "invalid_calibration"
    nan_k = CameraCalibration(np.full((3, 3), np.nan), (W, H))
    assert verify_pair(*two_plane_pair(), nan_k)[0].reason == "invalid_calibration"


def test_ransac_failure(monkeypatch):
    def failing(uv_q, uv_r, K, config):
        return geo.EssentialEstimate(None, np.zeros(len(uv_q), bool), 0, "test")

    monkeypatch.setattr(geo, "estimate_essential", failing)
    res, _, _ = verify_pair(*two_plane_pair(), CAL)
    assert res.reason == "ransac_failure" and not res.accepted


def test_concentrated_inliers():
    a = np.full((H, W), 128, np.uint8)
    b = a.copy()
    a[:120, :160] = texture(3)[:120, :160]
    b[:120, 4:164] = texture(3)[:120, :160]
    res, _, _ = verify_pair(a, b, CAL, VerificationConfig(min_coverage=0.2))
    assert res.reason == "concentrated_inliers" and res.coverage < 0.2


def test_thresholds_on_inliers_and_ratio():
    a, b = two_plane_pair()
    assert verify_pair(a, b, CAL, VerificationConfig(min_inliers=5000))[0].reason == (
        "too_few_inliers"
    )
    assert verify_pair(a, b, CAL, VerificationConfig(min_inlier_ratio=0.99))[0].reason == (
        "low_inlier_ratio"
    )


def test_details_and_record():
    res, fq, fr = verify_pair(*two_plane_pair(), CAL, keep_details=True)
    assert res.matches is not None and res.inlier_mask.shape == (len(res.matches),)
    assert res.inlier_mask.sum() == res.n_inliers and res.E.shape == (3, 3)
    assert res.h_mask is not None and res.h_mask.sum() == res.h_inliers
    rec = res.record()
    assert "matches" not in rec and rec["reason"] == "accepted" and "t_x" in rec
    assert set(REASONS) >= {res.reason}


def test_config_from_dict_and_unknown_keys():
    cfg = VerificationConfig.from_dict(
        {"verify_top_k": 3, "orb": {"n_features": 500}, "matching": {"ratio": 0.7}}
    )
    assert cfg.orb.n_features == 500 and cfg.matching.ratio == 0.7
    with pytest.raises(ValueError):
        VerificationConfig.from_dict({"nonsense": 1})
    with pytest.raises(ValueError):
        VerificationConfig(estimator="magic")


# ----------------------------------------------------------------------------- pipeline


def _fake_retrieval():
    ids = np.array([0, 10, 200, 210])
    cand = np.array([[-1, -1], [-1, -1], [0, 10], [10, 0]])
    scores = np.where(cand >= 0, 0.9, np.nan).astype(np.float32)
    return RetrievalResult(
        ids, ids / 10.0, cand, scores, np.array([0, 0, 2, 2]), RetrievalConfig(top_k=2)
    )


def test_verify_candidates_decisions_and_labels():
    a, b = two_plane_pair()
    images = {0: a, 10: texture(9), 200: b, 210: texture(8)}
    store = FeatureStore(lambda f: images[f], VerificationConfig().orb)
    run = verify_candidates(_fake_retrieval(), store, CAL, VerificationConfig(), verify_top_k=2)
    assert len(run.pairs) == 4 and store.extracted == 4  # features cached per frame
    acc = run.pairs.set_index(["query_frame_id", "candidate_frame_id"]).accepted
    assert acc[(200, 0)] and not acc[(200, 10)] and not acc[(210, 10)]
    dec = run.decisions().set_index("query_frame_id")
    assert dec.loc[200, "loop_frame_id"] == 0 and dec.loc[200, "rank"] == 1
    assert dec.loc[210, "loop_frame_id"] == -1

    positions = np.array([[0, 0, 0], [100, 0, 0], [2, 0, 0], [100, 0, 15.0]])
    labelled = label_pairs(run.pairs, np.array([0, 10, 200, 210]), positions, 5.0, 25.0)
    lab = labelled.set_index(["query_frame_id", "candidate_frame_id"]).gt_label
    assert lab[(200, 0)] == POSITIVE and lab[(200, 10)] == NEGATIVE
    assert lab[(210, 10)] == IGNORED  # 15 m: between the 5 m and 25 m radii
    assert lab[(210, 0)] == NEGATIVE  # 101 m
    s = summarize(labelled)
    assert s["true_accepts"] == 1 and s["positive_pairs"] == 1


def test_run_save_load_roundtrip(tmp_path):
    a, b = two_plane_pair()
    images = {0: a, 10: texture(9), 200: b, 210: texture(8)}
    store = FeatureStore(lambda f: images[f], VerificationConfig().orb)
    run = verify_candidates(_fake_retrieval(), store, CAL, VerificationConfig(), verify_top_k=1)
    run.save(tmp_path)
    back = type(run).load(tmp_path)
    assert back.verify_top_k == 1 and back.config == run.config
    pd.testing.assert_series_equal(back.pairs.accepted, run.pairs.accepted)


def test_decision_functions_never_take_poses():
    for fn in (verify_features, verify_pair, verify_candidates):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"poses", "pose", "positions", "ground_truth", "gt"}
