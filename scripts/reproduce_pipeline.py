#!/usr/bin/env python3
"""Re-run the complete benchmark pipeline from scratch, fast; compare with committed results.

Usage:
    python scripts/reproduce_pipeline.py --fast-root ~/latent_loop_fast
    python scripts/reproduce_pipeline.py --fast-root ~/latent_loop_fast --runs v2 --skip-stage

Steps (each timed; nothing committed is overwritten):
1. **Stage** every ``benchmark.stride``-th KITTI frame of the development and test sequences
   (plus calib/times/poses) to ``--fast-root`` on a local Linux disk. The KITTI loader skips
   frame IDs without images, so the sampled frame IDs are identical to the full copy; image
   reads from ``/mnt/<drive>`` were the dominant cost of earlier runs.
2. **ORB vocabulary** fitted again on development sequence 05 into a fresh directory.
3. For each run (``v1`` = configs/benchmark.yaml, ``v2`` = configs/benchmark_v2.yaml):
   development stage -> encoder profile of the *selected* variants -> test stage, with fresh
   embedding caches on the fast disk and outputs in ``results/rerun/<name>``. v2 starts from
   v1's verified pairs (same verification config hash, deterministic decisions).
4. **Compare** every pooled and per-sequence metric with ``results/benchmark/<name>``.

Neural encoders run on the GPU (``device: auto``); verification runs on the CPU workers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from latent_loop.evaluation.runner import parse_variant
from latent_loop.utils.config import load_config, resolve_path

RUNS = {"v1": "configs/benchmark.yaml", "v2": "configs/benchmark_v2.yaml"}
METRICS = [
    "recall_at_1",
    "recall_at_5",
    "map",
    "detection_ap",
    "det_precision",
    "det_recall",
    "det_f1",
    "det_tp",
    "det_fp",
    "det_fn",
]


def step(log: list, name: str, cmd: list[str]) -> None:
    print(f"\n=== {name}\n$ {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    log.append({"step": name, "seconds": round(time.perf_counter() - t0, 1), "command": cmd})
    print(f"=== {name}: {log[-1]['seconds']} s", flush=True)


def stage(src_root: Path, dst_root: Path, seqs: list[str], camera: int, stride: int) -> dict:
    copied = skipped = 0
    for seq in seqs:
        s, d = src_root / "sequences" / seq, dst_root / "sequences" / seq
        (d / f"image_{camera}").mkdir(parents=True, exist_ok=True)
        for name in ("calib.txt", "times.txt"):
            shutil.copy2(s / name, d / name)
        pose = src_root / "poses" / f"{seq}.txt"
        if pose.is_file():
            (dst_root / "poses").mkdir(parents=True, exist_ok=True)
            shutil.copy2(pose, dst_root / "poses" / f"{seq}.txt")
        for img in sorted((s / f"image_{camera}").glob("*.png")):
            if int(img.stem) % stride:
                continue
            out = d / f"image_{camera}" / img.name
            if out.is_file() and out.stat().st_size == img.stat().st_size:
                skipped += 1
                continue
            shutil.copy2(img, out)
            copied += 1
    return {"copied": copied, "already_present": skipped}


def variant_overrides(selection: dict) -> list[str]:
    """``--set`` overrides that configure each encoder to its selected variant."""
    out = []
    for enc, sel in selection.items():
        for k, v in parse_variant(sel["variant"]).items():
            if k != "window":
                out.append(f"{enc}.{k}={v}")
    return out


def compare(new_dir: Path, old_dir: Path) -> dict:
    new = pd.read_csv(new_dir / "test_metrics.csv", dtype={"sequence": str})
    old = pd.read_csv(old_dir / "test_metrics.csv", dtype={"sequence": str})
    m = new.merge(old, on=["method", "sequence"], suffixes=("", "_committed"))
    diffs = []
    for c in METRICS:
        a, b = m[c].astype(float), m[f"{c}_committed"].astype(float)
        bad = ~((a - b).abs() < 1e-9) & ~(a.isna() & b.isna())
        for _, r in m[bad].iterrows():
            diffs.append(
                {
                    "method": r.method,
                    "sequence": r.sequence,
                    "metric": c,
                    "rerun": float(r[c]),
                    "committed": float(r[f"{c}_committed"]),
                }
            )
    same = {}
    for f in ("selection.json", "thresholds.json"):
        same[f] = json.loads((new_dir / f).read_text()) == json.loads((old_dir / f).read_text())
    preds = sorted((new_dir / "predictions" / "test").glob("*.csv"))
    identical = sum(
        (old_dir / "predictions" / "test" / p.name).read_bytes() == p.read_bytes() for p in preds
    )
    return {
        "rows_compared": int(len(m)),
        "metric_differences": diffs,
        "identical_files": same,
        "identical_test_prediction_files": f"{identical}/{len(preds)}",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fast-root", required=True, help="directory on a local (ext4) disk")
    ap.add_argument("--runs", nargs="*", default=["v1", "v2"], choices=list(RUNS))
    ap.add_argument("--out", default="results/rerun")
    ap.add_argument("--workers", type=int, default=8, help="CPU verification processes")
    ap.add_argument("--skip-stage", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    fast = Path(args.fast_root).expanduser().resolve()
    kitti, cache = fast / "kitti" / "dataset", fast / "cache"
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = load_config(RUNS["v1"], [])
    b = base["benchmark"]
    seqs = [f"{int(s):02d}" for s in base["splits"]["dev"] + base["splits"]["test"]]
    log: list[dict] = []
    t_all = time.perf_counter()

    if not args.skip_stage:
        t0 = time.perf_counter()
        info = stage(
            resolve_path(b["dataset_root"]),
            kitti,
            seqs,
            int(b.get("camera", 2)),
            int(b.get("stride", 5)),
        )
        log.append(
            {
                "step": "stage KITTI to fast disk",
                "seconds": round(time.perf_counter() - t0, 1),
                **info,
            }
        )
        print(log[-1], flush=True)

    common = [
        f"benchmark.dataset_root={kitti}",
        f"extraction.cache_root={cache / 'embeddings'}",
        f"orb_bovw.vocabulary_dir={cache / 'vocabulary'}",
        f"benchmark.output_dir={out}",
    ]
    kitti00 = ["dataset.name=kitti", f"dataset.root={kitti}", "dataset.sequence='00'"]
    step(
        log,
        "ORB vocabulary (dev 05)",
        [py, "scripts/fit_vocabulary.py", "--set", *kitti00, *common],
    )

    comparisons = {}
    for run in args.runs:
        cfg = RUNS[run]
        name = load_config(cfg, [])["benchmark"]["name"]
        if run == "v2" and "v1" in args.runs:
            src, dst = out / "kitti_stride5" / "pair_cache", out / name / "pair_cache"
            if src.is_dir() and not dst.exists():
                shutil.copytree(src, dst)
        prof = resolve_path(f"results/metrics/encoder_profile_rerun_{run}.json")
        sets = [*common, f"benchmark.encoder_profile={prof}"]
        step(
            log,
            f"{run} development stage",
            [
                py,
                "scripts/run_benchmark.py",
                "--config",
                cfg,
                "--stage",
                "dev",
                "--workers",
                str(args.workers),
                "--set",
                *sets,
            ],
        )
        selection = json.loads((out / name / "selection.json").read_text())
        step(
            log,
            f"{run} encoder profile (selected variants)",
            [
                py,
                "scripts/profile_encoders.py",
                "--config",
                cfg,
                "--tag",
                f"rerun_{run}",
                "--stride",
                str(b.get("stride", 5)),
                "--frames",
                "64",
                "--set",
                *kitti00,
                *common,
                *variant_overrides(selection),
            ],
        )
        step(
            log,
            f"{run} test stage",
            [
                py,
                "scripts/run_benchmark.py",
                "--config",
                cfg,
                "--stage",
                "test",
                "--workers",
                str(args.workers),
                "--set",
                *sets,
            ],
        )
        comparisons[run] = compare(out / name, resolve_path("results/benchmark") / name)
        print(
            json.dumps(
                {
                    run: {k: v for k, v in comparisons[run].items() if k != "metric_differences"},
                    "metric_differences": len(comparisons[run]["metric_differences"]),
                }
            ),
            flush=True,
        )

    summary = {
        "total_seconds": round(time.perf_counter() - t_all, 1),
        "fast_root": str(fast),
        "workers": args.workers,
        "steps": log,
        "comparison_with_committed": comparisons,
    }
    (out / "pipeline_run.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nTotal {summary['total_seconds']} s -> {out / 'pipeline_run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
