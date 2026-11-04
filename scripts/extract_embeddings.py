#!/usr/bin/env python
"""Extract (or load from cache) global descriptors for one dataset sequence.

Usage:
    python scripts/extract_embeddings.py --config configs/ijepa.yaml
    python scripts/extract_embeddings.py --config configs/ijepa.yaml --stride 2 --batch-size 4
    python scripts/extract_embeddings.py --config configs/ijepa.yaml --set ijepa.resize_mode=aspect
    python scripts/extract_embeddings.py --config configs/ijepa.yaml --device cpu --max-frames 16

Prints a JSON report (cache location, runtime, peak GPU/CPU memory) and writes it to
``results/metrics/``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from latent_loop.datasets import build_dataset
from latent_loop.encoders import AVAILABLE_ENCODERS, build_encoder
from latent_loop.features.extraction import dataset_cache_key, extract_embeddings, make_cache
from latent_loop.utils.config import load_config, resolve_path
from latent_loop.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/ijepa.yaml")
    parser.add_argument("--encoder", choices=AVAILABLE_ENCODERS, help="override 'encoder'")
    parser.add_argument("--stride", type=int, help="override dataset.sampling.stride")
    parser.add_argument("--max-frames", type=int, help="override dataset.sampling.max_frames")
    parser.add_argument("--batch-size", type=int, help="override extraction.batch_size")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], help="override device")
    parser.add_argument("--no-cache", action="store_true", help="encode everything, cache nothing")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="config overrides, e.g. ijepa.precision=fp32",
    )
    args = parser.parse_args()

    overrides = list(args.set)
    for flag, key in (
        (args.stride, "dataset.sampling.stride"),
        (args.max_frames, "dataset.sampling.max_frames"),
        (args.batch_size, "extraction.batch_size"),
        (args.device, "device"),
    ):
        if flag is not None:
            overrides.append(f"{key}={flag}")
    config = load_config(args.config, overrides)
    setup_logging(config["logging"]["level"], resolve_path(config["logging"].get("file")))
    ext = config.get("extraction", {})

    dataset = build_dataset(config)
    encoder = build_encoder(args.encoder or config.get("encoder", "ijepa"), config)
    cache = (
        None
        if args.no_cache
        else make_cache(
            resolve_path(ext.get("cache_root", "data/cache/embeddings")), dataset, encoder
        )
    )
    result = extract_embeddings(
        dataset,
        encoder,
        cache,
        batch_size=int(ext.get("batch_size", 8)),
        cpu_fallback=bool(ext.get("cpu_fallback", True)),
        flush_every=int(ext.get("flush_every", 256)),
    )
    norms = np.linalg.norm(result.embeddings, axis=1)
    report = {
        "dataset_key": dataset_cache_key(dataset),
        "dataset": dataset.summary(),
        "encoder": encoder.info(),
        "fingerprint": encoder.fingerprint(),
        "cache_dir": str(result.cache_dir) if result.cache_dir else None,
        "frames": int(result.frame_ids.size),
        "encoded_now": result.encoded_now,
        "final_batch_size": result.final_batch_size,
        "fallbacks": result.fallbacks,
        "embedding_shape": list(result.embeddings.shape),
        "embedding_bytes": int(result.embeddings.nbytes),
        "norm_min_max": [round(float(norms.min()), 6), round(float(norms.max()), 6)],
        "profile": result.profile,
    }
    report["dataset"].pop("config", None)
    print(json.dumps(report, indent=2, default=str))
    out = (
        resolve_path(config["paths"]["results_dir"])
        / "metrics"
        / (f"extraction_{dataset_cache_key(dataset)}_{encoder.name}-{encoder.fingerprint()}.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
