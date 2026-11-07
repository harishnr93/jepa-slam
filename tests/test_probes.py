"""Representation probes: rho, causal averaging, PCA, concatenation, McNemar."""

from __future__ import annotations

import numpy as np
import pytest

from latent_loop.evaluation.probes import (
    causal_average,
    concat_views,
    l2n,
    mcnemar,
    pca_basis,
    project,
    random_pairs,
    same_place_pairs,
    scene_explained_fraction,
)


def test_rho_limits():
    rng = np.random.default_rng(0)
    places = l2n(rng.normal(size=(50, 16)))
    emb = np.concatenate([places, places])  # every frame seen twice, identically
    same = np.c_[np.arange(50, 100), np.arange(50)]
    rand = random_pairs(100, 5000)
    assert scene_explained_fraction(emb, same, rand) == pytest.approx(1.0)
    noise = l2n(rng.normal(size=(100, 16)))  # revisits unrelated to each other
    assert abs(scene_explained_fraction(noise, same, rand)) < 0.1


def test_rho_in_subspace_ignores_nuisance_directions():
    rng = np.random.default_rng(1)
    scene = rng.normal(size=(40, 2)) * 3
    nuisance = rng.normal(size=(80, 6))
    emb = np.c_[np.concatenate([scene, scene]), nuisance]
    same = np.c_[np.arange(40, 80), np.arange(40)]
    rand = random_pairs(80, 4000)
    basis = np.eye(8)[:, :2]  # the two scene coordinates
    assert scene_explained_fraction(emb, same, rand, basis) == pytest.approx(1.0)
    assert scene_explained_fraction(emb, same, rand) < 0.9


def test_random_pairs_are_distinct():
    p = random_pairs(5, 1000, seed=3)
    assert (p[:, 0] != p[:, 1]).all() and p.max() < 5


def test_same_place_pairs_picks_closest_revisit():
    ids = np.array([0, 5, 10, 200])
    pos = np.zeros((4, 3))
    pos[:, 0] = [0.0, 3.0, 9.0, 2.5]
    positives = [np.array([]), np.array([]), np.array([]), np.array([0, 5])]
    assert same_place_pairs(positives, ids, pos).tolist() == [[3, 1]]


def test_causal_average_uses_only_the_past():
    x = l2n(np.eye(4))
    avg = causal_average(x, 2)
    np.testing.assert_allclose(avg[0], x[0])
    np.testing.assert_allclose(avg[2], l2n((x[1] + x[2])[None])[0])
    y = x.copy()
    y[3] = l2n(np.ones((1, 4)))[0]  # changing the future must not change the past
    np.testing.assert_allclose(causal_average(y, 2)[:3], avg[:3])
    with pytest.raises(ValueError):
        causal_average(x, 0)


def test_pca_projection_and_concat_are_unit_norm():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 10))
    mu, basis = pca_basis(x, 3)
    assert basis.shape == (10, 3)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(project(x, mu, basis), axis=1), 1, rtol=1e-6)
    c = concat_views(x[:, :4], x[:, 4:])
    np.testing.assert_allclose(np.linalg.norm(c, axis=1), 1, rtol=1e-6)


def test_mcnemar_counts_discordant_pairs():
    base = np.array([1, 1, 0, 0, 1], bool)
    new = np.array([1, 0, 1, 1, 1], bool)
    r = mcnemar(base, new)
    assert (r["new_only"], r["base_only"]) == (2, 1)
    assert mcnemar(base, base)["p"] == 1.0
