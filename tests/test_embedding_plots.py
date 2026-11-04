"""Phase 2 plots on real synthetic poses with cheap deterministic descriptors."""

from __future__ import annotations

import numpy as np

from conftest import SMALL_SYNTHETIC
from latent_loop.datasets import SamplingConfig, SyntheticLoopDataset
from latent_loop.datasets.ground_truth import pair_labels
from latent_loop.visualization.embeddings import (
    pca,
    plot_descriptor_distribution,
    plot_patch_similarity,
    plot_preprocessing_comparison,
    plot_similarity_matrix,
    render_similarity_evolution_frames,
    separability,
)
from latent_loop.visualization.plots import save_figure


def _setup():
    ds = SyntheticLoopDataset(SMALL_SYNTHETIC, SamplingConfig(stride=2))
    poses = ds.poses()
    positions = poses[:, :3, 3]
    # Position-based descriptors: similarity decays with distance, so revisits score high.
    rng = np.random.default_rng(0)
    proj = rng.normal(size=(2, 16))
    feat = np.cos(positions[:, [0, 2]] @ proj / 8.0)
    emb = feat / np.linalg.norm(feat, axis=1, keepdims=True)
    labels = pair_labels(positions, ds.frame_ids, 5.0, 25.0, min_gap_frames=40)
    return ds, emb.astype(np.float32), labels, positions


def test_pca_explained_variance():
    x = np.random.default_rng(0).normal(size=(50, 6)) * [10, 1, 1, 1, 1, 1]
    scores, ratio = pca(x, 2)
    assert scores.shape == (50, 2) and ratio.shape == (6,)
    np.testing.assert_allclose(ratio.sum(), 1.0)
    assert ratio[0] > 0.8


def test_phase2_plots_render(tmp_path):
    ds, emb, labels, positions = _setup()
    sim = emb @ emb.T
    stats = separability(sim, labels)
    assert stats["positives"] > 0 and stats["auc"] > 0.9
    ids = ds.frame_ids
    figs = {
        "matrix": plot_similarity_matrix(sim, ids, labels, "t", 40, 5.0),
        "dist": plot_descriptor_distribution(
            emb, sim, labels, positions, ids, ds.timestamps(), "t", min_gap_frames=40
        ),
        "prep": plot_preprocessing_comparison(
            [{"resize_mode": m, "input": "8×8", **stats} for m in ("a", "b")], "t"
        ),
    }
    grid = (2, 3)
    rng = np.random.default_rng(1)
    pair = {
        "query_id": 0,
        "ref_id": 1,
        "label": "x",
        "distance_m": 1.0,
        "global_cos": 0.5,
        "query_image": ds.get_image(0),
        "ref_image": ds.get_image(1),
        "query_tokens": rng.normal(size=(6, 8)),
        "ref_tokens": rng.normal(size=(6, 8)),
    }
    figs["patch"] = plot_patch_similarity([pair, pair], grid, "t", display_size=(96, 32))
    for name, fig in figs.items():
        assert save_figure(fig, tmp_path / f"{name}.png").stat().st_size > 1000


def test_similarity_evolution_frames():
    ds, emb, labels, _ = _setup()
    frames = render_similarity_evolution_frames(
        ds.get_image,
        emb @ emb.T,
        labels,
        ds.frame_ids,
        ds.timestamps(),
        np.array([0, 30, len(ds) - 1]),
        40,
        "t",
        width_px=500,
    )
    assert len(frames) == 3 and frames[0].shape[1] == 500
    assert not np.array_equal(frames[0], frames[-1])
