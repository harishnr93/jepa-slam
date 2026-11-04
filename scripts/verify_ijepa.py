#!/usr/bin/env python
"""Verify the I-JEPA integration against Meta's reference implementation.

Usage:
    git clone --depth 1 https://github.com/facebookresearch/ijepa /tmp/ijepa_official
    python scripts/verify_ijepa.py --official-repo /tmp/ijepa_official

Checks, on the same preprocessed synthetic frames:

1. **Reference equivalence**: patch tokens from ``latent_loop.encoders.vit`` vs the
   official ``src/models/vision_transformer.vit_huge`` loaded with the same
   ``target_encoder`` weights (both FP32 on CPU). The official code is only imported
   from the clone you provide; it is not vendored (it is licensed CC BY-NC 4.0).
2. **Precision**: GPU FP16/BF16 descriptors vs CPU FP32 descriptors (cosine similarity).

The two models are run one after the other so that only one ViT-H (~2.5 GB) is in RAM
at a time. Results are written to ``results/metrics/ijepa_verification.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from latent_loop.datasets import SyntheticLoopDataset
from latent_loop.encoders.ijepa import IJEPAEncoder, load_mmap, target_encoder_path
from latent_loop.utils.config import resolve_path
from latent_loop.utils.logging import setup_logging


def official_tokens(repo: Path, weights: Path, batch: torch.Tensor) -> torch.Tensor:
    sys.path.insert(0, str(repo))
    try:
        from src.models.vision_transformer import vit_huge  # noqa: PLC0415
    finally:
        sys.path.pop(0)
    model = vit_huge(patch_size=14, img_size=[224])
    state = load_mmap(weights)["state_dict"]
    report = model.load_state_dict(state, strict=True)
    assert not report.missing_keys and not report.unexpected_keys
    model.eval()
    with torch.inference_mode():
        out = model(batch)
    del model, state
    gc.collect()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--official-repo", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--frames", type=int, nargs="*", default=[0, 150, 300, 450])
    parser.add_argument("--output", default="results/metrics/ijepa_verification.json")
    args = parser.parse_args()
    setup_logging()

    weights = target_encoder_path(resolve_path(args.checkpoint_dir), "vith14_in1k")
    dataset = SyntheticLoopDataset()
    images = [dataset.get_image(dataset.frame_ids.tolist().index(f)) for f in args.frames]

    ours_cpu = IJEPAEncoder(checkpoint_dir=resolve_path(args.checkpoint_dir), device="cpu")
    batch = torch.stack([ours_cpu.preprocess(im) for im in images])

    t0 = time.perf_counter()
    ref = official_tokens(args.official_repo, weights, batch)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    tokens, _ = ours_cpu.encode_tokens(images)
    desc_cpu = ours_cpu.encode_batch(images)
    t_ours = time.perf_counter() - t0
    ours_cpu.unload()
    gc.collect()

    diff = (tokens - ref).abs()
    token_cos = torch.nn.functional.cosine_similarity(tokens, ref, dim=-1)
    ref_desc = torch.nn.functional.normalize(ref.mean(1), dim=-1).numpy()
    result = {
        "checkpoint": str(weights),
        "frames": args.frames,
        "input_shape": list(batch.shape),
        "token_shape": list(tokens.shape),
        "reference_equivalence_fp32_cpu": {
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "token_abs_max": float(ref.abs().max()),
            "min_token_cosine": float(token_cos.min()),
            "min_descriptor_cosine": float((ref_desc * desc_cpu).sum(1).min()),
            "seconds_official": round(t_ref, 2),
            "seconds_ours": round(t_ours, 2),
        },
    }

    if torch.cuda.is_available():
        gpu = IJEPAEncoder(checkpoint_dir=resolve_path(args.checkpoint_dir), device="cuda")
        desc_gpu = gpu.encode_batch(images)
        result["gpu_vs_cpu"] = {
            "precision": gpu.precision,
            "device_name": torch.cuda.get_device_name(0),
            "min_descriptor_cosine": float((desc_gpu * desc_cpu).sum(1).min()),
            "max_abs_diff": float(np.abs(desc_gpu - desc_cpu).max()),
        }
        gpu.unload()
    else:
        result["gpu_vs_cpu"] = "skipped: CUDA unavailable"

    print(json.dumps(result, indent=2))
    out = resolve_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
