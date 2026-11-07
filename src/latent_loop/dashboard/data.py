"""Cached-result access for the Streamlit dashboard (no neural network is ever loaded).

Everything comes from ``results/benchmark/<name>/`` (Phase 6) plus the KITTI images and
poses on disk. Every loader returns ``None`` (or an empty frame) when its input is
missing, so the dashboard can show what is available and explain how to produce the rest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latent_loop.evaluation.runner import window_suffix
from latent_loop.retrieval import RetrievalResult
from latent_loop.utils.config import load_config, resolve_path

METHOD_ENCODER = {
    "ijepa": "ijepa",
    "ijepa_gv": "ijepa",
    "dinov2": "dinov2",
    "dinov2_gv": "dinov2",
    "dinov3": "dinov3",
    "dinov3_gv": "dinov3",
    "orb_bovw": "orb_bovw",
    "hsv": "hsv",
}


@dataclass
class Missing:
    what: str
    how: str


class ResultStore:
    """Read-only view of one benchmark run."""

    def __init__(
        self, config_path: str | Path = "configs/benchmark.yaml", overrides: list[str] | None = None
    ):
        self.config = load_config(config_path, overrides or [])
        b = self.config["benchmark"]
        self.root = resolve_path(b.get("output_dir", "results/benchmark")) / b.get(
            "name", "default"
        )
        self.methods: list[str] = list(b.get("methods", {}))

    # ------------------------------------------------------------------ tables
    def available(self) -> bool:
        return (self.root / "test_metrics.csv").is_file() or (
            self.root / "dev_metrics.csv"
        ).is_file()

    def missing_message(self) -> Missing:
        return Missing(
            f"No benchmark results in {self.root}",
            "python scripts/run_benchmark.py --stage dev; "
            "python scripts/run_benchmark.py --stage test",
        )

    def metrics(self, split: str) -> pd.DataFrame | None:
        p = self.root / f"{split}_metrics.csv"
        return pd.read_csv(p, dtype={"sequence": str}) if p.is_file() else None

    def json(self, name: str) -> Any | None:
        p = self.root / name
        return json.loads(p.read_text()) if p.is_file() else None

    def thresholds(self) -> dict | None:
        return self.json("thresholds.json")

    def efficiency(self) -> pd.DataFrame | None:
        p = self.root / "efficiency.csv"
        return pd.read_csv(p) if p.is_file() else None

    def curves(self, split: str) -> dict[str, dict]:
        d = self.root / "curves" / split
        return {p.stem: dict(np.load(p)) for p in sorted(d.glob("*.npz"))} if d.is_dir() else {}

    def sequences(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for split in ("dev", "test"):
            d = self.root / "predictions" / split
            if d.is_dir():
                out[split] = sorted({p.stem.rsplit("_", 1)[1] for p in d.glob("*.csv")})
        return out

    def split_of(self, seq: str) -> str | None:
        return next((s for s, seqs in self.sequences().items() if seq in seqs), None)

    def predictions(self, method: str, seq: str) -> pd.DataFrame | None:
        split = self.split_of(seq)
        p = self.root / "predictions" / str(split) / f"{method}_{seq}.csv"
        return pd.read_csv(p) if split and p.is_file() else None

    def row(self, method: str, seq: str) -> pd.Series | None:
        split = self.split_of(seq)
        df = self.metrics(split) if split else None
        if df is None:
            return None
        sel = df[(df.method == method) & (df.sequence == seq)]
        return sel.iloc[0] if len(sel) else None

    # ------------------------------------------------------------------ retrieval / verification
    def retrieval(self, method: str, seq: str) -> RetrievalResult | None:
        row = self.row(method, seq)
        if row is None:
            return None
        d = (
            self.root
            / "retrieval"
            / f"kitti_{seq}_cam{self.camera}"
            / f"{METHOD_ENCODER[method]}-{row.fingerprint}{window_suffix(self.window(row))}"
        )
        return RetrievalResult.load(d) if (d / "retrieval.npz").is_file() else None

    @staticmethod
    def window(row: pd.Series) -> int:
        """Frame-averaging window of a metrics row (1 when the run has no such column)."""
        w = row.get("window") if hasattr(row, "get") else None
        return int(w) if w is not None and pd.notna(w) else 1

    @property
    def camera(self) -> int:
        return int(self.config["benchmark"].get("camera", 2))

    def verification_config(self) -> dict | None:
        for split in ("test", "dev"):
            j = self.json(f"{split}_metrics.json")
            if j:
                return {"hash": j["verification_config_hash"], "config": j["verification_config"]}
        return None

    def pairs(self, seq: str) -> pd.DataFrame | None:
        vc = self.verification_config()
        if vc is None:
            return None
        p = self.root / "pair_cache" / f"kitti_{seq}_cam{self.camera}" / vc["hash"] / "pairs.csv"
        if not p.is_file():
            return None
        df = pd.read_csv(p, dtype={"degenerate": str, "reason": str}).fillna({"degenerate": ""})
        return df.set_index(["query_frame_id", "candidate_frame_id"])

    # ------------------------------------------------------------------ images and poses
    def dataset(self, seq: str):
        """KITTI sequence (lazy image loading); ``None`` when the raw data is missing."""
        from latent_loop.datasets import build_dataset  # noqa: PLC0415
        from latent_loop.evaluation.runner import sequence_config  # noqa: PLC0415

        try:
            return build_dataset(sequence_config(self.config, seq))
        except (FileNotFoundError, OSError, ValueError, RuntimeError):
            return None


def query_table(
    result: RetrievalResult, pred: pd.DataFrame | None, threshold: float | None
) -> pd.DataFrame:
    """One row per query: top-1 candidate/score and (if given) the detector's decision."""
    df = pd.DataFrame(
        {
            "query_frame_id": result.query_frame_ids,
            "top1_frame_id": result.top1_ids,
            "top1_cosine": result.top1_scores,
            "eligible_refs": result.eligible_counts,
        }
    )
    if pred is not None:
        df["loop_candidate"] = pred.candidate_ids.to_numpy()
        df["score"] = pred.scores.to_numpy()
        df["accepted"] = (
            pred.scores.fillna(-np.inf) >= (threshold if threshold is not None else np.inf)
        ).to_numpy()
        df["correct"] = pred.correct.to_numpy()
        df["has_revisit"] = pred.has_positive.to_numpy()
        df["candidate_distance_m"] = pred.candidate_distance_m.to_numpy()
    return df


def overview_numbers(
    pred: pd.DataFrame,
    threshold: float,
    row: pd.Series | None,
    eff: pd.DataFrame | None,
    method: str,
) -> dict[str, Any]:
    """Headline numbers of one (method, sequence) for the overview page."""
    on = pred.scores.fillna(-np.inf) >= threshold
    out = {
        "processed_frames": int(len(pred)),
        "evaluable_queries": int(pred.evaluable.sum()),
        "candidate_loops": int(pred.scores.notna().sum()),
        "accepted_loops": int(on.sum()),
        "correct_accepted": int((on & pred.correct).sum()),
        "revisit_queries": int(pred.has_positive.sum()),
    }
    if row is not None:
        out.update({"precision": row.det_precision, "recall": row.det_recall, "f1": row.det_f1})
    if eff is not None and method in set(eff.method):
        e = eff.set_index("method").loc[method]
        out["latency_ms"] = float(e.end_to_end_ms_per_query)
        out["device"] = str(e.device)
    return out
