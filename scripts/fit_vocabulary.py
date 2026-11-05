#!/usr/bin/env python
"""Fit the ORB bag-of-visual-words vocabulary on development data only.

Usage:
    python scripts/fit_vocabulary.py                                   # for SYNTHETIC
    python scripts/fit_vocabulary.py \
        --set dataset.name=kitti dataset.root=data/raw/kitti/dataset "dataset.sequence='00'"

The fitting data is chosen from the *evaluated* dataset's config by
``latent_loop.encoders.classical.vocabulary_plan``: for KITTI, the sequences in
``orb_bovw.fit_kitti_sequences`` (default 05), which must be in ``splits.dev`` and not in
``splits.test``; for the synthetic city, a different generated city
(``orb_bovw.fit_synthetic_seed``). The evaluated frames are never used unless they belong
to a listed development sequence. Writes data/cache/vocabulary/<name>.npz + .json.
"""

from __future__ import annotations

import argparse
import json
import time

from latent_loop.datasets import build_dataset
from latent_loop.encoders.classical import fit_vocabulary, vocabulary_plan
from latent_loop.features.extraction import dataset_cache_key
from latent_loop.utils.config import load_config
from latent_loop.utils.logging import get_logger, setup_logging
from latent_loop.verification.orb import OrbConfig, extract_orb

logger = get_logger("fit_vocabulary")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--force", action="store_true", help="refit even if the file exists")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    setup_logging(config["logging"]["level"])
    plan = vocabulary_plan(config)
    if plan["path"].is_file() and not args.force:
        print(f"Vocabulary exists: {plan['path']} (use --force to refit)")
        return 0

    orb = OrbConfig(n_features=plan["n_features"])
    descriptor_sets, sources = [], []
    start = time.perf_counter()
    for src in plan["sources"]:
        cfg = json.loads(json.dumps(config))
        cfg["dataset"]["sampling"] = {"stride": plan["stride"]}
        if src["name"] == "kitti":
            cfg["dataset"].update(
                {"name": "kitti", "sequence": src["sequence"], "camera": src["camera"]}
            )
        else:
            cfg["dataset"]["synthetic"] = {
                **cfg["dataset"]["synthetic"],
                "seed": src["synthetic_seed"],
            }
        dataset = build_dataset(cfg)
        key = dataset_cache_key(dataset)
        n_before = len(descriptor_sets)
        for i in range(len(dataset)):
            descriptor_sets.append(extract_orb(dataset.get_image(i), orb).descriptors)
        sources.append({"dataset_key": key, "frames": len(dataset), "stride": plan["stride"]})
        logger.info(
            "ORB descriptors extracted",
            extra={"dataset": key, "frames": len(descriptor_sets) - n_before},
        )
    extract_s = time.perf_counter() - start
    vocab = fit_vocabulary(
        descriptor_sets,
        k=plan["k"],
        max_descriptors=plan["max_descriptors"],
        seed=plan["seed"],
        meta={
            "fit_sources": sources,
            "splits": config.get("splits"),
            "orb": orb.__dict__,
            "device": "cpu",
        },
    )
    vocab.meta["seconds"] = {
        "orb_extraction": round(extract_s, 1),
        "total": round(time.perf_counter() - start, 1),
    }
    path = vocab.save(plan["path"])
    print(
        json.dumps(
            {"path": str(path), **vocab.meta, "digest": vocab.digest()}, indent=2, default=str
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
