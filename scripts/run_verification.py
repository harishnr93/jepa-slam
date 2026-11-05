#!/usr/bin/env python
"""Geometrically verify saved retrieval candidates (ORB + essential-matrix RANSAC).

Usage:
    python scripts/run_verification.py --config configs/ijepa.yaml --stride 2
    python scripts/run_verification.py --config configs/ijepa.yaml --stride 5 \
        --set dataset.name=kitti dataset.root=data/raw/kitti/dataset "dataset.sequence='00'"
    python scripts/run_verification.py --stride 2 --top-k 1 --estimator numpy

Reads the Phase 3 result results/retrieval/<dataset_key>/<encoder>-<fingerprint>/<mode>/
(run scripts/run_retrieval.py first with the same settings) and verifies the top
``verification.verify_top_k`` candidates of every query using only the two images and the
camera calibration. Outputs go to
results/verification/<same sub-path>/top<K>_<estimator>/:
verification.csv (one row per pair, with ground-truth distance/label columns added
*after* the decisions when poses exist), decisions.csv (highest-ranked accepted
candidate per query) and verification.json (config, timing, summary).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from latent_loop.datasets import build_dataset
from latent_loop.encoders import AVAILABLE_ENCODERS, build_encoder
from latent_loop.features.extraction import dataset_cache_key
from latent_loop.retrieval import RetrievalConfig, RetrievalResult
from latent_loop.utils.config import load_config, resolve_path
from latent_loop.utils.logging import get_logger, setup_logging
from latent_loop.verification import (
    FeatureStore,
    VerificationConfig,
    label_pairs,
    run_dirname,
    summarize,
    verify_candidates,
)

logger = get_logger("run_verification")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/ijepa.yaml")
    parser.add_argument("--encoder", choices=AVAILABLE_ENCODERS)
    parser.add_argument("--stride", type=int, help="override dataset.sampling.stride")
    parser.add_argument("--top-k", type=int, help="override verification.verify_top_k")
    parser.add_argument("--estimator", choices=["opencv", "numpy"])
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    overrides = list(args.set)
    for flag, key in (
        (args.stride, "dataset.sampling.stride"),
        (args.top_k, "verification.verify_top_k"),
        (args.estimator, "verification.estimator"),
        (args.encoder, "encoder"),
    ):
        if flag is not None:
            overrides.append(f"{key}={flag}")
    config = load_config(args.config, overrides)
    setup_logging(config["logging"]["level"], resolve_path(config["logging"].get("file")))

    dataset = build_dataset(config)
    encoder = build_encoder(config.get("encoder", "ijepa"), config)
    rcfg = RetrievalConfig.from_dict(config["retrieval"])
    key = dataset_cache_key(dataset)
    sub = f"{key}/{encoder.name}-{encoder.fingerprint()}/{rcfg.mode}"
    retrieval_dir = resolve_path(config["paths"]["results_dir"]) / "retrieval" / sub
    if not (retrieval_dir / "retrieval.npz").is_file():
        raise SystemExit(
            f"No retrieval result in {retrieval_dir}; run scripts/run_retrieval.py with the "
            "same --config/--stride/--set first."
        )
    retrieval = RetrievalResult.load(retrieval_dir)
    if not np.array_equal(retrieval.query_frame_ids, dataset.frame_ids):
        raise SystemExit("Saved retrieval result was computed on different frames; re-run it.")

    vcfg_dict = dict(config.get("verification", {}))
    verify_top_k = int(vcfg_dict.get("verify_top_k", 1))
    vcfg = VerificationConfig.from_dict(vcfg_dict)
    store = FeatureStore(dataset.image_for_frame_id, vcfg.orb)
    run = verify_candidates(retrieval, store, dataset.get_calibration(), vcfg, verify_top_k)
    run.timing["orb_frames"] = store.extracted
    run.timing["orb_ms_per_frame"] = 1000 * store.extract_seconds / max(store.extracted, 1)

    summary: dict = {"dataset_key": key, "queries": int(len(dataset)), "device": "cpu"}
    if dataset.has_poses and len(run.pairs):
        gt = config.get("ground_truth", {})
        run.pairs = label_pairs(
            run.pairs,
            dataset.frame_ids,
            dataset.poses()[:, :3, 3],
            float(gt.get("spatial_threshold_m", 5.0)),
            float(gt.get("negative_radius_m", 25.0)),
        )
    summary["pairs"] = summarize(run.pairs) if len(run.pairs) else {}
    for rank in sorted({1, verify_top_k}):
        subset = run.pairs[run.pairs["rank"] <= rank] if len(run.pairs) else run.pairs
        summary["pairs"][f"top{rank}"] = summarize(subset) if len(subset) else {}
    decisions = run.decisions() if len(run.pairs) else None
    if decisions is not None:
        summary["queries_with_accepted_loop"] = int((decisions.loop_frame_id >= 0).sum())

    out_dir = (
        resolve_path(config["paths"]["results_dir"])
        / "verification"
        / sub
        / run_dirname(verify_top_k, vcfg.estimator)
    )
    run.save(out_dir, {"summary": summary, "retrieval_dir": str(retrieval_dir)})
    if decisions is not None:
        decisions.to_csv(out_dir / "decisions.csv", index=False)
    summary["timing"] = run.timing
    summary["output_dir"] = str(out_dir)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
