"""Phase 6 metrics: hand-computed cases and brute-force cross-checks."""

from __future__ import annotations

import numpy as np
import pytest

from latent_loop.evaluation.benchmark import (
    Predictions,
    average_precision_ranked,
    detection_at,
    make_predictions,
    pr_curve,
    retrieval_average_precision,
    select_threshold,
)


def test_average_precision_ranked():
    assert average_precision_ranked([1, 0, 1, 0]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision_ranked([0, 0, 1]) == pytest.approx(1 / 3)
    assert average_precision_ranked([1, 1]) == 1.0
    assert np.isnan(average_precision_ranked([0, 0]))


def test_retrieval_ap_uses_full_eligible_ranking():
    # 4 frames on a line; query 3 may see frames 0..1 only; its positive is frame 1.
    emb = np.eye(4, dtype=np.float32)
    emb[3] = [0.6, 0.8, 0, 0]  # closer to frame 1 than to frame 0
    ap = retrieval_average_precision(
        emb,
        np.arange(4),
        np.array([0, 0, 0, 2]),
        [np.array([]), np.array([]), np.array([]), np.array([1])],
    )
    assert np.isnan(ap[:3]).all() and ap[3] == 1.0
    emb[3] = [0.8, 0.6, 0, 0]  # now frame 0 ranks first: AP = 1/2
    ap = retrieval_average_precision(
        emb,
        np.arange(4),
        np.array([0, 0, 0, 2]),
        [np.array([]), np.array([]), np.array([]), np.array([1])],
    )
    assert ap[3] == pytest.approx(0.5)


def _toy() -> Predictions:
    # q0: no eligible refs; q1..q6 evaluable. Positives: q1, q2, q3, q4.
    return Predictions(
        query_frame_ids=np.arange(7),
        candidate_ids=np.array([-1, 10, 11, 12, -1, 14, 15]),
        scores=np.array([np.nan, 0.9, 0.8, 0.7, np.nan, 0.85, 0.6]),
        correct=np.array([False, True, False, True, False, False, False]),
        has_positive=np.array([False, True, True, True, True, False, False]),
        evaluable=np.array([False, True, True, True, True, True, True]),
        candidate_distance_m=np.array([np.nan, 1, 12, 2, np.nan, 300, 40]),
    )


def test_detection_at_hand_computed():
    m = detection_at(_toy(), 0.75)  # accepts q1 (correct), q2 (wrong, 12 m), q5 (wrong)
    assert (m.queries, m.positive_queries, m.predicted) == (6, 4, 3)
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 2, 3, 1)
    assert m.precision == pytest.approx(1 / 3) and m.recall == pytest.approx(1 / 4)
    assert m.f1 == pytest.approx(2 * 1 / (2 * 1 + 2 + 3))
    assert m.fp_per_1000 == pytest.approx(1000 * 2 / 6)
    assert m.confusion == [[1, 1, 2], [0, 1, 1]]
    assert sum(map(sum, m.confusion)) == m.queries
    assert m.fp_near_5_25m == 1


def test_pr_curve_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 300
    p = Predictions(
        np.arange(n),
        np.where(rng.random(n) < 0.9, 1, -1),
        np.round(rng.random(n), 2),  # ties on purpose
        rng.random(n) < 0.5,
        rng.random(n) < 0.6,
        rng.random(n) < 0.95,
        rng.random(n) * 50,
    )
    p.correct &= p.has_positive & (p.candidate_ids >= 0)
    curve = pr_curve(p)
    prev_r, ap = 0.0, 0.0
    for t, prec, rec in zip(curve.thresholds, curve.precision, curve.recall, strict=True):
        m = detection_at(p, t)
        assert m.precision == pytest.approx(prec) and m.recall == pytest.approx(rec)
        ap += prec * (rec - prev_r)
        prev_r = rec
    assert curve.ap == pytest.approx(ap)
    assert np.all(np.diff(curve.thresholds) < 0) and np.all(np.diff(curve.recall) >= 0)


def test_select_threshold():
    curve = pr_curve(_toy())
    sel = select_threshold(curve)
    # thresholds 0.9 (P=1, R=.25), .85 (P=.5), .8 (P=1/3), .7 (P=.5, R=.5), .6 (P=.4)
    assert sel["p100"] == 0.9 and sel["recall_at_p100"] == pytest.approx(0.25)
    f1s = {t: detection_at(_toy(), t).f1 for t in curve.thresholds}
    assert sel["f1_value"] == pytest.approx(max(f1s.values()))
    assert detection_at(_toy(), sel["f1"]).f1 == pytest.approx(sel["f1_value"])


def test_make_predictions_labels_and_distances():
    positions = np.array([[0, 0, 0], [0, 0, 100], [0, 0, 3.0]])
    pred = make_predictions(
        np.array([0, 5, 10]),
        np.array([-1, 0, 0]),
        np.array([0.1, 0.5, 0.9]),
        [np.array([]), np.array([]), np.array([0])],
        np.array([0, 1, 2]),
        positions,
    )
    assert np.isnan(pred.scores[0]) and list(pred.correct) == [False, False, True]
    assert pred.candidate_distance_m[1] == pytest.approx(100) and pred.candidate_distance_m[
        2
    ] == pytest.approx(3)
    assert list(pred.evaluable) == [False, True, True]


def test_parallel_verification_is_deterministic_and_cached(tmp_path):
    from dataclasses import asdict

    from conftest import SMALL_SYNTHETIC
    from latent_loop.verification import VerificationConfig
    from latent_loop.verification.parallel import config_hash, verify_pairs

    dcfg = {
        "dataset": {
            "name": "synthetic",
            "synthetic": asdict(SMALL_SYNTHETIC),
            "sampling": {"stride": 2},
        }
    }
    cfg = VerificationConfig(min_inliers=0, min_keypoints=10, min_matches=8)
    pairs = [(120, 0), (120, 2), (140, 20), (60, 0)]
    one, orb1 = verify_pairs(pairs, dcfg, cfg, tmp_path / "a", "toy", workers=1, chunk=2)
    two, _ = verify_pairs(pairs, dcfg, cfg, tmp_path / "b", "toy", workers=2, chunk=2)
    cols = ["query_frame_id", "candidate_frame_id", "accepted", "reason", "n_matches", "n_inliers"]
    assert one[cols].equals(two[cols])
    assert set(orb1.frame_id) == {0, 2, 20, 60, 120, 140}
    csv = tmp_path / "a" / "toy" / config_hash(cfg) / "pairs.csv"
    stamp = csv.stat().st_mtime_ns
    again, _ = verify_pairs(pairs[:2], dcfg, cfg, tmp_path / "a", "toy", workers=1)
    assert csv.stat().st_mtime_ns == stamp and len(again) == 2  # served from the cache


def test_benchmark_runner_protocol(tmp_path):
    import json

    from conftest import write_kitti_sequence
    from latent_loop.evaluation.runner import Benchmark
    from latent_loop.utils.config import load_config

    root = tmp_path / "kitti"
    for seq in ("00", "01"):
        write_kitti_sequence(root, seq, num_frames=12, missing=())
    cfg = load_config(
        "configs/benchmark.yaml",
        [
            f"benchmark.dataset_root={root}",
            f"benchmark.output_dir={tmp_path / 'out'}",
            "benchmark.stride=1",
            "benchmark.workers=1",
            "retrieval.temporal_exclusion_frames=2",
            f"extraction.cache_root={tmp_path / 'cache'}",
            "splits.dev=['00']",
            "splits.test=['01']",
        ],
    )
    cfg["benchmark"]["methods"] = {
        "hsv": {"encoder": "hsv", "verify": False},
        "hsv_gv": {"encoder": "hsv", "verify": True},
    }
    cfg["benchmark"]["variants"] = {}
    bench = Benchmark(cfg)
    with pytest.raises(RuntimeError, match="development stage first"):
        bench.run_test()
    dev = bench.run_dev()
    assert set(dev["thresholds"]) == {"hsv", "hsv_gv"} and "hsv" in dev["selection"]
    out = tmp_path / "out" / "kitti_stride5"
    assert (out / "selection.json").is_file() and (out / "thresholds.json").is_file()
    bench.run_test()
    rows = json.loads((out / "test_metrics.json").read_text())["rows"]
    assert {(r["method"], r["sequence"]) for r in rows} >= {
        ("hsv", "01"),
        ("hsv", "pooled"),
        ("hsv_gv", "pooled"),
    }
    assert (out / "predictions" / "test" / "hsv_01.csv").is_file()
    assert (out / "efficiency.csv").is_file()

    from latent_loop.visualization.phase6 import generate_phase6

    summary = generate_phase6(cfg, tmp_path / "assets", gif_frames=3, gif_width=500)
    for name in summary["outputs"].values():
        assert (tmp_path / "assets" / name).stat().st_size > 1000
    cfg["splits"]["test"] = ["00"]
    with pytest.raises(ValueError, match="overlap"):
        Benchmark(cfg).run_test()
