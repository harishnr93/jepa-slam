"""Phase 5 comparison figures and GIF frames on synthetic descriptors of known quality."""

from __future__ import annotations

import numpy as np

from conftest import SMALL_SYNTHETIC
from latent_loop.datasets import SamplingConfig, SyntheticLoopDataset
from latent_loop.datasets.ground_truth import pair_labels, revisit_ground_truth
from latent_loop.retrieval import RetrievalConfig, retrieve
from latent_loop.visualization.animations import save_gif
from latent_loop.visualization.baselines import (
    METHODS,
    MethodRun,
    choose_comparison_cases,
    correctness,
    plot_cost_comparison,
    plot_qualitative,
    plot_recall_comparison,
    plot_side_by_side,
    plot_similarity_comparison,
    render_comparison_frames,
)
from latent_loop.visualization.plots import save_figure


def _runs():
    ds = SyntheticLoopDataset(SMALL_SYNTHETIC, SamplingConfig(stride=2))
    poses = ds.poses()
    xz = poses[:, [0, 2], 3]
    rng = np.random.default_rng(0)
    runs = []
    for j, name in enumerate(METHODS):
        feat = np.concatenate([np.cos(xz / 3), np.sin(xz / 3), np.cos(xz / 40), np.sin(xz / 40)], 1)
        feat = feat + rng.normal(0, 0.15 * j, feat.shape)  # method j is noisier than j - 1
        emb = (feat / np.linalg.norm(feat, axis=1, keepdims=True)).astype(np.float32)
        res = retrieve(
            ds.frame_ids,
            ds.timestamps(),
            emb,
            RetrievalConfig(top_k=5, temporal_exclusion_frames=50),
        )
        card = {
            "ms_per_image": 10.0 ** (2 - j),
            "weights_file_bytes": [2**30, 2**28, 2**15, 0][j],
            "embedding_dim": 8,
            "runtime": {"device": "cpu"},
        }
        runs.append(MethodRun(name, emb, res, card))
    gt = revisit_ground_truth(poses, ds.frame_ids, 3.0, 50)
    return ds, runs, gt, poses[:, :3, 3]


def test_cases_and_correctness():
    _, runs, gt, _ = _runs()
    ok = correctness(runs[0], gt)
    assert ok.dtype == bool and ok.sum() > 0
    assert not np.any(ok & ~gt.has_positive)  # only positive queries can be correct
    cases = choose_comparison_cases(runs, gt)
    assert cases and len({i for i, _ in cases}) == len(cases)  # distinct queries
    assert any("no revisit" in c for _, c in cases)


def test_phase5_figures_and_gif(tmp_path):
    ds, runs, gt, pos = _runs()
    labels = pair_labels(pos, ds.frame_ids, 3.0, 10.0, 50)
    cases = choose_comparison_cases(runs, gt)
    figs = {
        "recall": plot_recall_comparison(runs, [("a", gt), ("b", gt)], "t", 5),
        "sim": plot_similarity_comparison(runs, gt, labels, "t"),
        "qual": plot_qualitative(runs, gt, pos, ds.get_image, cases[:3], "t"),
        "side": plot_side_by_side(runs, gt, pos, ds.get_image, cases[:2], "t", k=2),
        "cost": plot_cost_comparison(runs, {r.name: 0.5 for r in runs}, "t"),
    }
    for name, fig in figs.items():
        assert save_figure(fig, tmp_path / f"{name}.png").stat().st_size > 1000
    frames = render_comparison_frames(
        runs, gt, pos, ds.get_image, np.array([0, 80, len(ds) - 1]), "t", 600
    )
    assert len(frames) == 3 and frames[0].shape[1] == 600
    gif = save_gif(frames, tmp_path / "c.gif", fps=8, reserve_colors=["#0072B2"], colors=64)
    assert gif.stat().st_size > 1000
