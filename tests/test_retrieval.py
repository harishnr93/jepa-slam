"""Phase 3: FAISS retrieval, temporal exclusion, causal replay and ground-truth labels."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import SMALL_SYNTHETIC
from latent_loop.datasets import SamplingConfig, SyntheticLoopDataset
from latent_loop.datasets.ground_truth import revisit_ground_truth
from latent_loop.evaluation.metrics import first_hit_rank, recall_at_k
from latent_loop.retrieval import (
    FlatIPIndex,
    RetrievalConfig,
    RetrievalResult,
    TemporalExclusion,
    assert_causal,
    filter_candidates,
    retrieve,
)
from latent_loop.retrieval.index import exact_search_numpy


def _unit(n: int, d: int = 16, seed: int = 0) -> np.ndarray:
    x = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# --------------------------------------------------------------------------- index
def test_flat_ip_matches_brute_force_with_prefix_limits():
    emb = _unit(200)
    ids = np.arange(200) * 3 + 7  # arbitrary increasing original IDs
    index = FlatIPIndex(16)
    index.add(ids, emb)
    limits = np.random.default_rng(1).integers(0, 201, size=200)
    scores, got = index.search(emb, 10, limits)
    ref_scores, ref_pos = exact_search_numpy(emb, emb, 10, limits)
    expected = np.where(ref_pos >= 0, ids[np.clip(ref_pos, 0, None)], -1)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_allclose(scores, ref_scores, atol=1e-5)
    for row, lim in zip(got, limits, strict=True):  # never beyond the eligible prefix
        assert set(row[row >= 0]) <= set(ids[:lim])
        assert (row >= 0).sum() == min(10, lim)


def test_index_rejects_unnormalised_and_out_of_order():
    index = FlatIPIndex(4)
    with pytest.raises(ValueError, match="L2-normalised"):
        index.add(np.array([0]), np.ones((1, 4)))
    index.add(np.array([5]), _unit(1, 4))
    with pytest.raises(ValueError, match="chronological"):
        index.add(np.array([5]), _unit(1, 4, seed=2))
    with pytest.raises(ValueError, match="shape"):
        index.add(np.array([6, 7]), _unit(1, 4))


def test_empty_and_zero_limit_searches_return_padding():
    index = FlatIPIndex(4)
    s, i = index.search(_unit(2, 4), 3)
    assert np.isnan(s).all() and (i == -1).all()
    index.add(np.arange(5), _unit(5, 4))
    s, i = index.search(_unit(1, 4), 3, np.array([0]))
    assert (i == -1).all()


# --------------------------------------------------------------------------- temporal rule
def test_exclusion_uses_original_frame_ids_not_indices():
    ids = np.arange(0, 1000, 10)  # stride-10 view: index gap 10 = 100 original frames
    times = ids / 10.0
    rule = TemporalExclusion(frames=100)
    counts = rule.eligible_counts(ids, times)
    # Query ID 500 may use IDs <= 400, i.e. the first 41 entries (0..400), not 500-100=400 rows.
    assert counts[50] == 41
    assert counts[:10].tolist() == [0] * 10 and counts[10] == 1
    # Seconds-based rule on the same data is equivalent at 10 Hz.
    np.testing.assert_array_equal(
        TemporalExclusion(frames=None, seconds=10.0).eligible_counts(ids, times), counts
    )


def test_exclusion_combines_rules_and_validates():
    ids = np.arange(20)
    times = np.concatenate([np.zeros(10), np.arange(1, 11.0)])  # a 1 s pause at the start
    both = TemporalExclusion(frames=5, seconds=6.0)
    # frames rule: IDs <= 14 (15 refs); seconds rule: t <= 4 -> IDs 0..13 (14 refs); both = min
    assert TemporalExclusion(frames=5).eligible_count(ids, times, 19, 10.0) == 15
    assert TemporalExclusion(frames=None, seconds=6.0).eligible_count(ids, times, 19, 10.0) == 14
    assert both.eligible_count(ids, times, 19, 10.0) == 14
    assert TemporalExclusion(frames=None).eligible_count(ids, times, 7, 0.0) == 7  # strictly <
    with pytest.raises(ValueError):
        TemporalExclusion(frames=0)
    with pytest.raises(ValueError, match="chronological"):
        both.eligible_counts(np.array([0, 2, 1]), np.zeros(3))
    ok = both.is_eligible(np.array([3, 13, 14, 19]), times[[3, 13, 14, 19]], 19, 10.0)
    assert ok.tolist() == [True, True, False, False]


# --------------------------------------------------------------------------- detector
@pytest.mark.parametrize(
    "cfg",
    [
        RetrievalConfig(top_k=5, temporal_exclusion_frames=20),
        RetrievalConfig(top_k=5, temporal_exclusion_frames=20, nms_frames=3),
        RetrievalConfig(
            top_k=3,
            temporal_exclusion_frames=None,
            temporal_exclusion_seconds=2.5,
            min_similarity=0.1,
        ),
    ],
)
def test_offline_and_causal_replay_are_identical_and_causal(cfg):
    ids = np.arange(0, 300, 2)
    times = ids * 0.1
    emb = _unit(ids.size, seed=4)
    offline = retrieve(ids, times, emb, cfg)
    causal = retrieve(ids, times, emb, RetrievalConfig(**{**cfg.__dict__, "mode": "causal"}))
    np.testing.assert_array_equal(offline.candidate_ids, causal.candidate_ids)
    np.testing.assert_allclose(offline.scores, causal.scores, atol=1e-6, equal_nan=True)
    assert_causal(offline)
    assert_causal(causal)
    for q, row in zip(ids, offline.candidate_ids, strict=True):  # never the future
        assert np.all(row[row >= 0] < q)


def test_assert_causal_detects_leakage():
    ids = np.arange(50)
    res = retrieve(ids, ids * 0.1, _unit(50), RetrievalConfig(top_k=2, temporal_exclusion_frames=5))
    res.candidate_ids[40, 0] = 45  # a future frame
    with pytest.raises(AssertionError, match="ineligible"):
        assert_causal(res)


def test_filter_candidates_threshold_and_nms():
    scores = np.array([0.9, 0.89, 0.88, 0.7, 0.5], dtype=np.float32)
    ids = np.array([100, 101, 140, 102, 300])
    s, i = filter_candidates(scores, ids, 3, None, 2)
    assert i.tolist() == [100, 140, 300]  # 101 and 102 are within 2 frames of 100
    s, i = filter_candidates(scores, ids, 3, 0.85, 0)
    assert i.tolist() == [100, 101, 140]
    s, i = filter_candidates(scores, ids, 4, 0.8, 0)
    assert i.tolist() == [100, 101, 140, -1] and np.isnan(s[3])


def test_retrieval_config_validation():
    with pytest.raises(ValueError):
        RetrievalConfig(top_k=0)
    with pytest.raises(ValueError):
        RetrievalConfig(mode="online")
    with pytest.raises(ValueError, match="Unknown"):
        RetrievalConfig.from_dict({"topk": 3})


def test_result_save_load_roundtrip(tmp_path):
    ids = np.arange(30)
    res = retrieve(ids, ids * 0.1, _unit(30), RetrievalConfig(top_k=4, temporal_exclusion_frames=3))
    res.save(tmp_path / "r", {"note": "x"})
    back = RetrievalResult.load(tmp_path / "r")
    np.testing.assert_array_equal(back.candidate_ids, res.candidate_ids)
    np.testing.assert_array_equal(back.scores, res.scores)
    assert back.config == res.config


# --------------------------------------------------------------------------- ground truth
def _straight_then_repeat():
    """Drive +z for 60 m, a 30 m detour, then repeat the first 60 m (same direction)."""
    poses = []
    for z in list(range(60)) + [None] * 30 + list(range(60)):
        p = np.eye(4)
        if z is None:  # detour 50 m to the side, facing the other way
            p[:3, :3] = np.diag([-1, 1, -1])
            p[0, 3] = 50.0
        else:
            p[2, 3] = float(z)
        poses.append(p)
    return np.stack(poses)


def test_revisit_ground_truth_distance_gap_and_heading():
    poses = _straight_then_repeat()
    ids = np.arange(len(poses))
    gt = revisit_ground_truth(poses, ids, spatial_threshold_m=2.0, exclusion_frames=50)
    q = 90 + 10  # second pass at z = 10
    np.testing.assert_array_equal(gt.positives[q], [8, 9, 10, 11, 12])
    assert not gt.has_positive[:90].any() and gt.has_positive[90:].all()
    gt_heading = revisit_ground_truth(poses, ids, 2.0, 50, max_heading_diff_deg=30.0)
    assert gt_heading.has_positive.sum() == gt.has_positive.sum()  # same direction here
    # Strided view: the gap is still measured in original IDs.
    view = np.arange(0, len(poses), 5)
    gt_view = revisit_ground_truth(poses[view], view, 2.0, 50)
    assert all(set(p) <= set(view) for p in gt_view.positives)


def test_recall_at_k_and_first_hit():
    cands = np.array([[5, 6, 7], [1, 2, 3], [9, 8, 4]])
    pos = (np.array([7]), np.array([], dtype=int), np.array([9]))
    r = recall_at_k(cands, pos, ks=(1, 3))
    assert r == {1: 0.5, 3: 1.0}  # the query without positives is excluded
    assert first_hit_rank(cands, pos).tolist() == [3, 0, 1]
    with pytest.raises(ValueError):
        recall_at_k(cands, pos, ks=(5,))


def test_known_synthetic_revisits_are_retrieved():
    """Position-derived descriptors (an oracle) must retrieve the constructed revisits;
    this checks the retrieval plumbing end to end, independent of any learned encoder."""
    ds = SyntheticLoopDataset(SMALL_SYNTHETIC, SamplingConfig(stride=1))
    poses = ds.poses()
    xz = poses[:, [0, 2], 3]
    # Two spatial frequencies: fine (3 m) for precision, coarse (40 m) to avoid aliasing.
    feat = np.concatenate(
        [np.cos(xz / 3.0), np.sin(xz / 3.0), np.cos(xz / 40.0), np.sin(xz / 40.0)], axis=1
    )
    emb = (feat / np.linalg.norm(feat, axis=1, keepdims=True)).astype(np.float32)
    res = retrieve(
        ds.frame_ids, ds.timestamps(), emb, RetrievalConfig(top_k=5, temporal_exclusion_frames=50)
    )
    gt = revisit_ground_truth(poses, ds.frame_ids, 2.0, 50)
    assert gt.has_positive.sum() > 20
    assert recall_at_k(res.candidate_ids, gt.positives, (1,))[1] > 0.95
    constructed = dict(map(tuple, ds.revisit_pairs))
    top1 = dict(zip(res.query_frame_ids.tolist(), res.top1_ids.tolist(), strict=True))
    near = [abs(top1[q] - r) <= 2 for q, r in constructed.items()]
    assert np.mean(near) > 0.95
