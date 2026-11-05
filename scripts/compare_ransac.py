#!/usr/bin/env python
"""Compare the two essential-matrix RANSAC backends (OpenCV five-point vs own eight-point).

Usage:
    python scripts/compare_ransac.py --stride 5 \
        --set dataset.name=kitti dataset.root=data/raw/kitti/dataset "dataset.sequence='00'"

Re-verifies the rank-1 pairs of the saved Phase 4 result with both estimators (all other
settings identical) and writes results/metrics/ransac_comparison_<dataset_key>.json:
decision agreement, inlier counts and per-pair RANSAC time. CPU only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import numpy as np

from latent_loop.datasets import build_dataset
from latent_loop.encoders import build_encoder
from latent_loop.features.extraction import dataset_cache_key
from latent_loop.retrieval import RetrievalConfig
from latent_loop.utils.config import load_config, resolve_path
from latent_loop.utils.logging import setup_logging
from latent_loop.verification import (
    FeatureStore,
    VerificationConfig,
    VerificationRun,
    run_dirname,
    verify_features,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ijepa.yaml")
    parser.add_argument("--stride", type=int)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    overrides = list(args.set)
    if args.stride:
        overrides.append(f"dataset.sampling.stride={args.stride}")
    config = load_config(args.config, overrides)
    setup_logging(config["logging"]["level"])
    dataset = build_dataset(config)
    encoder = build_encoder(config.get("encoder", "ijepa"), config)
    rcfg = RetrievalConfig.from_dict(config["retrieval"])
    key = dataset_cache_key(dataset)
    sub = f"{key}/{encoder.name}-{encoder.fingerprint()}/{rcfg.mode}"
    base = VerificationConfig.from_dict(config.get("verification", {}))
    top_k = int(config.get("verification", {}).get("verify_top_k", 1))
    vdir = resolve_path(config["paths"]["results_dir"]) / "verification" / sub
    run = VerificationRun.load(vdir / run_dirname(top_k, base.estimator))
    pairs = run.pairs[run.pairs["rank"] == 1]
    configs = {e: dataclasses.replace(base, estimator=e) for e in ("opencv", "numpy")}
    store = FeatureStore(dataset.image_for_frame_id, base.orb)
    calib = dataset.get_calibration()
    rows = []
    for q, c in zip(pairs.query_frame_id, pairs.candidate_frame_id, strict=True):
        fq, fr = store(int(q)), store(int(c))
        out = {}
        for name, cfg in configs.items():
            r = verify_features(fq, fr, calib, cfg)
            out[name] = (r.accepted, r.n_inliers, r.timing_ms.get("ransac", np.nan), r.reason)
        rows.append(out)
    ran = [r for r in rows if not np.isnan(r["opencv"][2])]
    acc = np.array([[r["opencv"][0], r["numpy"][0]] for r in rows])
    inl = np.array([[r["opencv"][1], r["numpy"][1]] for r in ran], dtype=float)
    ms = np.array([[r["opencv"][2], r["numpy"][2]] for r in ran], dtype=float)
    both_acc = inl[(inl[:, 0] >= base.min_inliers)]
    summary = {
        "dataset_key": key,
        "pairs": len(rows),
        "pairs_reaching_ransac": len(ran),
        "device": "cpu",
        "decision_agreement": float(np.mean(acc[:, 0] == acc[:, 1])),
        "accepted_opencv": int(acc[:, 0].sum()),
        "accepted_numpy": int(acc[:, 1].sum()),
        "accepted_by_one_only": int((acc[:, 0] != acc[:, 1]).sum()),
        "median_inliers_ratio_numpy_over_opencv_on_opencv_accepted": float(
            np.median(both_acc[:, 1] / both_acc[:, 0])
        )
        if len(both_acc)
        else None,
        "median_ransac_ms": {
            "opencv": float(np.median(ms[:, 0])),
            "numpy": float(np.median(ms[:, 1])),
        },
        "disagreements": [
            {"query": int(q), "candidate": int(c), "opencv": r["opencv"], "numpy": r["numpy"]}
            for (q, c), r in zip(
                zip(pairs.query_frame_id, pairs.candidate_frame_id, strict=True), rows, strict=True
            )
            if r["opencv"][0] != r["numpy"][0]
        ],
    }
    out = resolve_path("results/metrics") / f"ransac_comparison_{key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "disagreements"}, indent=2))
    print(f"disagreements: {len(summary['disagreements'])} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
