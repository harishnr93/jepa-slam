from __future__ import annotations

import numpy as np
import pytest

from latent_loop.datasets.ground_truth import IGNORED, NEGATIVE, POSITIVE, pair_labels
from latent_loop.evaluation.metrics import roc_auc


def test_roc_auc_known_values():
    assert roc_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert roc_auc([0.1], [0.9]) == 0.0
    assert roc_auc([0.5], [0.5]) == 0.5  # ties count half
    # 3 of 4 (pos, neg) pairs are ordered correctly.
    assert roc_auc([0.4, 0.9], [0.5, 0.1]) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        roc_auc([], [0.1])


def test_roc_auc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 40), rng.normal(0, 1, 60)
    brute = np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :]))
    assert roc_auc(pos, neg) == pytest.approx(brute)


def test_pair_labels_use_original_frame_ids():
    # Out along z then back: frame k at z=k for k<50, then z decreases from 50 to 1.
    z = np.concatenate([np.arange(50.0), np.arange(50.0, 0.0, -1.0)])
    positions = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    ids = np.arange(z.size) * 3  # a stride-3 view: gaps are measured in original IDs
    labels = pair_labels(
        positions, ids, positive_radius_m=2.0, negative_radius_m=10.0, min_gap_frames=60
    )
    assert labels.dtype == np.int8 and labels.shape == (100, 100)
    assert np.all(labels[np.triu_indices(100)] == IGNORED)  # never the future or itself
    i = 99  # z = 1, original ID 297
    pos = np.flatnonzero(labels[i] == POSITIVE)
    np.testing.assert_array_equal(pos, [0, 1, 2, 3])  # z = 0..3 within 2 m
    assert labels[i, 20] == NEGATIVE  # z = 20, 19 m away
    assert labels[i, 8] == IGNORED  # z = 8: between the radii
    # Gap 60 original IDs = 20 view indices: frame 30 may match frames 0..10 only.
    assert np.all(labels[30, 11:] == IGNORED)
    with pytest.raises(ValueError):
        pair_labels(positions, ids, positive_radius_m=5, negative_radius_m=1)
