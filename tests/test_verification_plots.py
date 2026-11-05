"""Phase 4 figures and GIF frames on synthetic image pairs with known geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from latent_loop.verification import VerificationConfig, raw_nearest_matches, verify_pair
from latent_loop.visualization.matches import (
    GAP_PX,
    PairExample,
    epipolar_lines,
    inlier_canvas,
    plot_decision_gallery,
    plot_inliers_outliers,
    plot_orb_keypoints,
    plot_raw_vs_filtered,
    plot_verification_statistics,
    plot_verified_trajectory,
    render_verification_frames,
    stack_pair,
)
from latent_loop.visualization.plots import save_figure
from test_verification import CAL, K, texture, two_plane_pair


def _example(a, b, qid, cid, label):
    res, fq, fr = verify_pair(a, b, CAL, VerificationConfig(), keep_details=True)
    return PairExample(
        qid, cid, a, b, fq, fr, res, raw_nearest_matches(fq, fr), 0.9, 1, label, 1.0, "t"
    )


def test_epipolar_lines_pass_through_inliers():
    ex = _example(*two_plane_pair(), 10, 0, 1)
    m, mask = ex.result.matches, ex.result.inlier_mask
    uq = ex.features_q.xy[m.query_idx[mask]].astype(float)
    ur = ex.features_r.xy[m.ref_idx[mask]].astype(float)
    lines = epipolar_lines(ex.result.E, K, uq)
    dist = np.abs(np.sum(lines * np.c_[ur, np.ones(len(ur))], axis=1))
    assert np.median(dist) < 1.0  # inliers lie on their epipolar lines (pixels)


def test_canvas_shapes():
    a, b = two_plane_pair()
    assert stack_pair(a, b).shape == (2 * a.shape[0] + GAP_PX, a.shape[1], 3)
    ex = _example(a, b, 10, 0, 1)
    canvas = inlier_canvas(ex)
    assert canvas.shape == stack_pair(a, b).shape and not np.array_equal(canvas, stack_pair(a, b))


def test_phase4_figures_and_gif(tmp_path):
    good = _example(*two_plane_pair(), 10, 0, 1)
    bad = _example(texture(1), texture(2), 20, 5, 0)
    assert good.result.accepted and not bad.result.accepted
    pairs = pd.DataFrame(
        {
            "query_frame_id": [10, 20],
            "candidate_frame_id": [0, 5],
            "rank": [1, 1],
            "accepted": [True, False],
            "reason": [good.result.reason, bad.result.reason],
            "n_inliers": [good.result.n_inliers, bad.result.n_inliers],
            "inlier_ratio": [good.result.inlier_ratio, bad.result.inlier_ratio],
            "gt_label": [1, 0],
        }
    )
    positions = np.array([[0, 0, 0], [50, 0, 0], [1, 0, 0], [40, 0, 30.0]])
    frame_ids = np.array([0, 5, 10, 20])
    figs = {
        "kp": plot_orb_keypoints(good, "t"),
        "raw": plot_raw_vs_filtered(good, "t"),
        "inl": plot_inliers_outliers(good, K, "t"),
        "rej": plot_inliers_outliers(bad, K, "t"),
        "gal": plot_decision_gallery([good, bad], "t"),
        "stats": plot_verification_statistics(pairs, 30, 0.3, "t"),
        "traj": plot_verified_trajectory(pairs, positions, frame_ids, "t"),
    }
    for name, fig in figs.items():
        assert save_figure(fig, tmp_path / f"{name}.png").stat().st_size > 1000
    frames = render_verification_frames([good, bad], positions, frame_ids, "t", 600, (1, 1, 2))
    assert len(frames) == 8 and frames[0].shape[1] == 600
    assert not np.array_equal(frames[0], frames[1])  # matches -> inliers stage changes
