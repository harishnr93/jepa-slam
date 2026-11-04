#!/usr/bin/env python
"""Retrieve loop-closure candidates from cached descriptors and save the results.

Usage:
    python scripts/run_retrieval.py --config configs/ijepa.yaml --stride 2
    python scripts/run_retrieval.py --config configs/ijepa.yaml --stride 5 \
        --set dataset.name=kitti dataset.root=data/raw/kitti/dataset "dataset.sequence='00'"
    python scripts/run_retrieval.py --config configs/ijepa.yaml --stride 2 --mode causal
    python scripts/run_retrieval.py --config configs/ijepa.yaml --stride 2 --check-causal

Descriptors come from the Phase 2 embedding cache (extracted first if missing). Results
are written to results/retrieval/<dataset_key>/<encoder>-<fingerprint>/<mode>/:
retrieval.npz (ranked candidates), retrieval.json (config, timing, preliminary
Recall@K under two ground-truth definitions) and top1.csv (one row per query).
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from latent_loop.datasets import build_dataset
from latent_loop.datasets.ground_truth import revisit_ground_truth
from latent_loop.encoders import AVAILABLE_ENCODERS, build_encoder
from latent_loop.evaluation.metrics import recall_at_k
from latent_loop.features.extraction import dataset_cache_key, extract_embeddings, make_cache
from latent_loop.retrieval import RetrievalConfig, assert_causal, retrieve
from latent_loop.utils.config import load_config, resolve_path
from latent_loop.utils.logging import get_logger, setup_logging

logger = get_logger("run_retrieval")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/ijepa.yaml")
    parser.add_argument("--encoder", choices=AVAILABLE_ENCODERS)
    parser.add_argument("--stride", type=int, help="override dataset.sampling.stride")
    parser.add_argument("--mode", choices=["offline", "causal"], help="override retrieval.mode")
    parser.add_argument("--top-k", type=int, help="override retrieval.top_k")
    parser.add_argument("--exclusion-frames", type=int, help="override exclusion (frames)")
    parser.add_argument(
        "--check-causal",
        action="store_true",
        help="also run causal replay and assert identical candidates",
    )
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    overrides = list(args.set)
    for flag, key in (
        (args.stride, "dataset.sampling.stride"),
        (args.mode, "retrieval.mode"),
        (args.top_k, "retrieval.top_k"),
        (args.exclusion_frames, "retrieval.temporal_exclusion_frames"),
        (args.encoder, "encoder"),
    ):
        if flag is not None:
            overrides.append(f"{key}={flag}")
    config = load_config(args.config, overrides)
    setup_logging(config["logging"]["level"], resolve_path(config["logging"].get("file")))
    ext = config.get("extraction", {})

    dataset = build_dataset(config)
    encoder = build_encoder(config.get("encoder", "ijepa"), config)
    cache = make_cache(
        resolve_path(ext.get("cache_root", "data/cache/embeddings")), dataset, encoder
    )
    extraction = extract_embeddings(
        dataset,
        encoder,
        cache,
        batch_size=int(ext.get("batch_size", 8)),
        cpu_fallback=bool(ext.get("cpu_fallback", True)),
    )
    encoder.unload()
    rcfg = RetrievalConfig.from_dict(config["retrieval"])
    result = retrieve(dataset.frame_ids, dataset.timestamps(), extraction.embeddings, rcfg)
    assert_causal(result)

    causal_check = None
    if args.check_causal:
        other_mode = "causal" if rcfg.mode == "offline" else "offline"
        other = retrieve(
            dataset.frame_ids,
            dataset.timestamps(),
            extraction.embeddings,
            RetrievalConfig(**{**rcfg.__dict__, "mode": other_mode}),
        )
        same_ids = bool(np.array_equal(result.candidate_ids, other.candidate_ids))
        max_score_diff = (
            float(np.nanmax(np.abs(result.scores - other.scores)))
            if np.isfinite(result.scores).any()
            else 0.0
        )
        causal_check = {
            "other_mode": other_mode,
            "identical_candidates": same_ids,
            "max_score_diff": max_score_diff,
            "other_timing": other.timing,
        }
        if not same_ids:
            raise SystemExit(f"offline and causal candidates differ: {causal_check}")

    gt_cfg = config.get("ground_truth", {})
    summary = {
        "dataset_key": dataset_cache_key(dataset),
        "queries": int(len(dataset)),
        "encoder": encoder.info(),
        "fingerprint": encoder.fingerprint(),
        "causal_check": causal_check,
        "ground_truth": {},
    }
    gts = {}
    if dataset.has_poses:
        poses = dataset.poses()
        for name, heading in (
            ("distance_only", None),
            ("same_direction", gt_cfg.get("max_heading_diff_deg", 30.0)),
        ):
            gt = revisit_ground_truth(
                poses,
                dataset.frame_ids,
                gt_cfg.get("spatial_threshold_m", 5.0),
                rcfg.temporal_exclusion_frames or 0,
                heading,
            )
            gts[name] = gt
            ks = [k for k in (1, 5, 10) if k <= rcfg.top_k]
            summary["ground_truth"][name] = {
                "spatial_threshold_m": gt.spatial_threshold_m,
                "max_heading_diff_deg": heading,
                "positive_queries": int(gt.has_positive.sum()),
                "recall_at_k_preliminary": recall_at_k(result.candidate_ids, gt.positives, ks),
            }
    out_dir = (
        resolve_path(config["paths"]["results_dir"])
        / "retrieval"
        / dataset_cache_key(dataset)
        / f"{encoder.name}-{encoder.fingerprint()}"
        / rcfg.mode
    )
    result.save(out_dir, summary)
    with open(out_dir / "top1.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "query_frame_id",
                "top1_frame_id",
                "top1_cosine",
                "eligible_refs",
                *(f"top1_is_revisit_{n}" for n in gts),
                *(f"query_has_revisit_{n}" for n in gts),
            ]
        )
        for i, q in enumerate(result.query_frame_ids):
            top = int(result.top1_ids[i])
            writer.writerow(
                [
                    int(q),
                    top,
                    f"{result.top1_scores[i]:.6f}",
                    int(result.eligible_counts[i]),
                    *(int(top in gt.positives[i]) for gt in gts.values()),
                    *(int(gt.has_positive[i]) for gt in gts.values()),
                ]
            )
    summary["timing"] = result.timing
    summary["output_dir"] = str(out_dir)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
