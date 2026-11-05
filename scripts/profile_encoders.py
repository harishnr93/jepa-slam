#!/usr/bin/env python
"""Profile every encoder under identical conditions (model size, dimension, runtime, memory).

Usage:
    python scripts/profile_encoders.py --stride 5 --frames 64 \
        --set dataset.name=kitti dataset.root=data/raw/kitti/dataset "dataset.sequence='00'"

The same ``--frames`` images (evenly spaced over the sequence) are loaded into RAM first,
so disk I/O is excluded. Each encoder is loaded alone (sequential model loading), one
warm-up batch is run, then all frames are encoded in batches of ``extraction.batch_size``.
Neural encoders run on their configured device/precision (``auto``); the classical ones
on the CPU. ``rss_increase_mb`` is the growth of this process's resident memory while the
encoder is loaded and running (host RAM; GPU memory is ``peak_gpu_mb``). Writes
results/metrics/encoder_profile_<dataset_key>.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np

from latent_loop.datasets import build_dataset
from latent_loop.encoders import AVAILABLE_ENCODERS, build_encoder
from latent_loop.features.extraction import dataset_cache_key
from latent_loop.utils.config import load_config, resolve_path
from latent_loop.utils.logging import setup_logging
from latent_loop.visualization.animations import select_animation_indices


def _model_card(name: str, enc, config) -> dict:
    card: dict = {"embedding_dim": enc.embedding_dim()}
    info = enc.info()
    if name in ("ijepa", "dinov2"):
        card["params"] = int(enc.num_params())
        card["weights_file_bytes"] = int(enc.weights_path.stat().st_size)
        card["input"] = f"{info['image_size']} px, {info['resize_mode']}"
        card["pooling"] = info["pooling"]
    elif name == "hsv":
        card["params"] = None
        card["weights_file_bytes"] = 0
        card["input"] = f"width {info['work_width']} px, HSV bins {info['bins_hsv']}"
    else:
        card["params"] = None  # no learned weights; the vocabulary is the model
        card["vocabulary_words"] = info["vocabulary_size"]
        card["vocabulary_fit_sources"] = info["vocabulary_fit_sources"]
        card["input"] = f"full resolution, {info['orb']['n_features']} ORB keypoints"
        card["weights_file_bytes"] = int(enc.vocabulary_path.stat().st_size)
    card["descriptor_bytes"] = 4 * card["embedding_dim"]
    card["runtime"] = info.get("runtime")
    return card


def _rss_mb() -> float:
    """Current resident set size of this process (Linux /proc)."""
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/dinov2.yaml")
    parser.add_argument("--stride", type=int)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--encoders", nargs="*", default=list(AVAILABLE_ENCODERS))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    overrides = list(args.set) + ([f"dataset.sampling.stride={args.stride}"] if args.stride else [])
    config = load_config(args.config, overrides)
    setup_logging(config["logging"]["level"])
    dataset = build_dataset(config)
    picks = select_animation_indices(len(dataset), args.frames)
    images = [dataset.get_image(int(i)) for i in picks]
    batch = int(config.get("extraction", {}).get("batch_size", 8))
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        torch = None

    report: dict = {
        "dataset_key": dataset_cache_key(dataset),
        "frames": len(images),
        "image_shape": list(images[0].shape),
        "batch_size": batch,
        "encoders": {},
    }
    for name in args.encoders:
        enc = build_encoder(name, config)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        rss0 = _rss_mb()
        t0 = time.perf_counter()
        enc.load_model()
        load_s = time.perf_counter() - t0
        enc.encode_batch(images[:batch])  # warm-up (kernels, allocator)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = [enc.encode_batch(images[i : i + batch]) for i in range(0, len(images), batch)]
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        desc = np.concatenate(out)
        card = _model_card(name, enc, config)
        card.update(
            {
                "load_seconds": round(load_s, 2),
                "ms_per_image": round(1000 * dt / len(images), 2),
                "images_per_second": round(len(images) / dt, 2),
                "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                if torch is not None and torch.cuda.is_available()
                else None,
                "rss_increase_mb": round(_rss_mb() - rss0, 1),
                "norm_min_max": [
                    float(np.linalg.norm(desc, axis=1).min()),
                    float(np.linalg.norm(desc, axis=1).max()),
                ],
                "fingerprint": enc.fingerprint(),
            }
        )
        report["encoders"][name] = card
        print(name, json.dumps(card, default=str))
        enc.unload()
        del enc
        gc.collect()
    out = resolve_path("results/metrics") / f"encoder_profile_{report['dataset_key']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
