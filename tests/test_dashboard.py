"""Phase 7: dashboard data layer and every view through Streamlit's AppTest (no browser)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from latent_loop.dashboard.data import ResultStore, overview_numbers, query_table
from latent_loop.utils.config import REPO_ROOT

STORE = ResultStore()
HAVE_RESULTS = STORE.available() and bool(STORE.sequences().get("test"))
APP = str(REPO_ROOT / "app" / "streamlit_app.py")


def test_missing_results_are_reported(tmp_path):
    store = ResultStore(overrides=[f"benchmark.output_dir={tmp_path}"])
    assert not store.available()
    assert store.metrics("test") is None and store.efficiency() is None
    assert store.curves("test") == {} and store.sequences() == {}
    assert store.predictions("ijepa", "06") is None and store.retrieval("ijepa", "06") is None
    assert "run_benchmark.py" in store.missing_message().how


def test_overview_numbers_and_query_table():
    pred = pd.DataFrame(
        {
            "query_frame_ids": [0, 5, 10, 15],
            "candidate_ids": [-1, 0, 0, 5],
            "scores": [np.nan, 0.9, 0.5, 0.95],
            "correct": [False, True, False, False],
            "has_positive": [False, True, True, False],
            "evaluable": [False, True, True, True],
            "candidate_distance_m": [np.nan, 1.0, 2.0, 50.0],
        }
    )
    n = overview_numbers(pred, 0.8, None, None, "x")
    assert (n["candidate_loops"], n["accepted_loops"], n["correct_accepted"]) == (3, 2, 1)

    class R:  # minimal RetrievalResult stand-in
        query_frame_ids = pred.query_frame_ids.to_numpy()
        top1_ids = pred.candidate_ids.to_numpy()
        top1_scores = pred.scores.to_numpy()
        eligible_counts = np.array([0, 1, 2, 3])

    t = query_table(R, pred, 0.8)
    assert list(t.accepted) == [False, True, False, True]


@pytest.mark.skipif(not HAVE_RESULTS, reason="Phase 6 benchmark results not available")
def test_store_reads_benchmark_outputs():
    assert STORE.thresholds() and STORE.efficiency() is not None
    r = STORE.retrieval("ijepa_gv", "06")
    pred = STORE.predictions("ijepa_gv", "06")
    assert r is not None and len(pred) == len(r.query_frame_ids)
    row = STORE.row("ijepa_gv", "06")
    thr = STORE.thresholds()["ijepa_gv"]["f1"]
    n = overview_numbers(pred, thr, row, STORE.efficiency(), "ijepa_gv")
    assert n["correct_accepted"] == int(row.det_tp)  # the dashboard reproduces the benchmark


@pytest.mark.skipif(not HAVE_RESULTS, reason="Phase 6 benchmark results not available")
@pytest.mark.parametrize(
    "params",
    [
        {},
        {"view": "retrieval", "seq": "06", "method": "ijepa_gv", "q": "600"},
        {"view": "trajectory", "seq": "02", "method": "dinov2_gv", "q": "4570"},
        {"view": "verification", "seq": "06", "method": "ijepa_gv"},
        {"view": "benchmark", "split": "dev"},
    ],
)
def test_dashboard_views_render(params):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in params.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
