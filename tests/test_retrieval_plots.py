"""Phase 3 figures and GIF frames on oracle (position-derived) descriptors."""

from __future__ import annotations

import numpy as np

from conftest import SMALL_SYNTHETIC
from latent_loop.datasets import SamplingConfig, SyntheticLoopDataset
from latent_loop.datasets.ground_truth import revisit_ground_truth
from latent_loop.retrieval import RetrievalConfig, retrieve
from latent_loop.visualization.plots import save_figure
from latent_loop.visualization.retrieval import (
    choose_example_queries,
    plot_candidate_trajectory,
    plot_recall_curves,
    plot_retrieval_matrix,
    plot_topk_gallery,
    render_retrieval_frames,
)


def _setup():
    ds = SyntheticLoopDataset(SMALL_SYNTHETIC, SamplingConfig(stride=2))
    poses = ds.poses()
    xz = poses[:, [0, 2], 3]
    feat = np.concatenate([np.cos(xz / 3), np.sin(xz / 3), np.cos(xz / 40), np.sin(xz / 40)], 1)
    emb = (feat / np.linalg.norm(feat, axis=1, keepdims=True)).astype(np.float32)
    res = retrieve(
        ds.frame_ids, ds.timestamps(), emb, RetrievalConfig(top_k=5, temporal_exclusion_frames=50)
    )
    gt = revisit_ground_truth(poses, ds.frame_ids, 3.0, 50)
    return ds, emb, res, gt, poses[:, :3, 3]


def test_example_queries_cover_cases():
    _, _, res, gt, _ = _setup()
    picks = choose_example_queries(res, gt)
    cases = {c for _, c in picks}
    assert "true revisit at rank 1" in cases
    assert any("no revisit" in c for c in cases)
    for i, _ in picks:
        assert res.top1_ids[i] >= 0


def test_phase3_figures_render(tmp_path):
    ds, emb, res, gt, pos = _setup()
    figs = {
        "gallery": plot_topk_gallery(
            res, gt, pos, ds.get_image, choose_example_queries(res, gt), "t", k=3
        ),
        "traj": plot_candidate_trajectory(res, gt, pos, "t"),
        "matrix": plot_retrieval_matrix(emb @ emb.T, res, gt, "t"),
        "recall": plot_recall_curves([("a", res.candidate_ids, gt)], "t", 5),
    }
    for name, fig in figs.items():
        assert save_figure(fig, tmp_path / f"{name}.png").stat().st_size > 1000


def test_retrieval_gif_frames():
    ds, _, res, gt, pos = _setup()
    frames = render_retrieval_frames(
        res, gt, pos, ds.get_image, np.array([0, 60, len(ds) - 1]), "t", width_px=500
    )
    assert len(frames) == 3 and frames[0].shape[1] == 500
    assert not np.array_equal(frames[0], frames[-1])
