#!/usr/bin/env python
"""Phase 6 benchmark: select on development sequences, evaluate once on held-out ones.

Usage:
    python scripts/run_benchmark.py --stage dev     # selections + thresholds (splits.dev)
    python scripts/run_benchmark.py --stage test    # frozen evaluation (splits.test)
    python scripts/run_benchmark.py --stage all

Writes results/benchmark/<benchmark.name>/: selection.json, thresholds.json,
{dev,test}_metrics.{csv,json}, curves/<split>/<method>.npz,
predictions/<split>/<method>_<seq>.csv, test_timing.json and pair_cache/. Descriptors
are extracted on demand (GPU for the neural encoders) and cached in data/cache/embeddings.
"""

from __future__ import annotations

import argparse
import json

from latent_loop.evaluation.runner import Benchmark
from latent_loop.utils.config import load_config, resolve_path
from latent_loop.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--stage", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--workers", type=int, help="override benchmark.workers")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    overrides = list(args.set) + ([f"benchmark.workers={args.workers}"] if args.workers else [])
    config = load_config(args.config, overrides)
    setup_logging(config["logging"]["level"], resolve_path(config["logging"].get("file")))
    bench = Benchmark(config)
    if args.stage in ("dev", "all"):
        out = bench.run_dev()
        print(json.dumps(out, indent=2, default=str))
    if args.stage in ("test", "all"):
        bench.run_test()
        print(f"Test metrics: {bench.out / 'test_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
