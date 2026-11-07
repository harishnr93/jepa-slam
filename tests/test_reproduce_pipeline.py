"""scripts/reproduce_pipeline.py: variant overrides and the comparison with committed results."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "reproduce_pipeline", ROOT / "scripts" / "reproduce_pipeline.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_variant_overrides_skip_window():
    mod = _module()
    sel = {
        "ijepa": {"variant": "aspect"},
        "dinov3": {"variant": "checkpoint=vitb16,resize_mode=squash,pooling=mean,window=2"},
        "hsv": {"variant": "window=2"},
    }
    assert mod.variant_overrides(sel) == [
        "ijepa.resize_mode=aspect",
        "dinov3.checkpoint=vitb16",
        "dinov3.resize_mode=squash",
        "dinov3.pooling=mean",
    ]


def _bench(d: Path, f1: float, pred: str) -> None:
    (d / "predictions" / "test").mkdir(parents=True)
    rows = [
        {"method": m, "sequence": "02", **{c: 0.5 for c in _module().METRICS}}
        for m in ("hsv", "dinov2")
    ]
    rows[1]["det_f1"] = f1
    pd.DataFrame(rows).to_csv(d / "test_metrics.csv", index=False)
    for f in ("selection.json", "thresholds.json"):
        (d / f).write_text(json.dumps({"hsv": 0.9}))
    (d / "predictions" / "test" / "hsv_02.csv").write_text(pred)


def test_compare_reports_metric_and_file_differences(tmp_path):
    mod = _module()
    old, same, diff = tmp_path / "old", tmp_path / "same", tmp_path / "diff"
    _bench(old, 0.6, "a,b\n1,2\n")
    _bench(same, 0.6, "a,b\n1,2\n")
    _bench(diff, 0.7, "a,b\n1,3\n")
    ok = mod.compare(same, old)
    assert ok["rows_compared"] == 2
    assert ok["metric_differences"] == []
    assert ok["identical_files"] == {"selection.json": True, "thresholds.json": True}
    assert ok["identical_test_prediction_files"] == "1/1"
    bad = mod.compare(diff, old)
    assert bad["metric_differences"] == [
        {"method": "dinov2", "sequence": "02", "metric": "det_f1", "rerun": 0.7, "committed": 0.6}
    ]
    assert bad["identical_test_prediction_files"] == "0/1"
