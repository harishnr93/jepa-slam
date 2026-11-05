#!/usr/bin/env python
"""Verify the DINOv2 ViT-B/14 integration against Meta's reference implementation.

Usage:
    git clone --depth 1 https://github.com/facebookresearch/dinov2 /tmp/dinov2_official
    python scripts/verify_dinov2.py --official-repo /tmp/dinov2_official

Checks, on the same preprocessed synthetic frames:

1. **Reference equivalence**: all output tokens of ``latent_loop.encoders.dinov2.DinoViT``
   vs the official ``dinov2.models.vision_transformer.vit_base`` (built with the hub's
   ViT-B/14 settings: 518 px, LayerScale, interpolate_offset 0.1, no antialias), both
   loaded with the same checkpoint, FP32 on CPU. Two input sizes are used: 224 x 224
   (the default) and 224 x 728 (the ``aspect`` mode on the 416 x 128 synthetic frames),
   so the position-embedding
   interpolation is exercised on square and non-square grids. The official code is only
   imported from the clone you provide; xFormers is disabled so both use plain PyTorch.
2. **Precision**: GPU descriptors (``auto`` precision) vs CPU FP32 descriptors.

Results are written to ``results/metrics/dinov2_verification.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from latent_loop.datasets import SyntheticLoopDataset
from latent_loop.encoders.dinov2 import DINOv2Encoder, checkpoint_path
from latent_loop.utils.config import resolve_path
from latent_loop.utils.logging import setup_logging


def official_model(repo: Path, weights: Path) -> torch.nn.Module:
    os.environ["XFORMERS_DISABLED"] = "1"
    sys.path.insert(0, str(repo))
    try:
        from dinov2.models.vision_transformer import vit_base  # noqa: PLC0415
    finally:
        sys.path.pop(0)
    model = vit_base(
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )
    report = model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True), strict=True
    )
    assert not report.missing_keys and not report.unexpected_keys
    return model.eval()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--official-repo", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--frames", type=int, nargs="*", default=[0, 150, 300, 450])
    parser.add_argument("--output", default="results/metrics/dinov2_verification.json")
    args = parser.parse_args()
    setup_logging()

    weights = checkpoint_path(resolve_path(args.checkpoint_dir), "vitb14")
    dataset = SyntheticLoopDataset()
    images = [dataset.get_image(dataset.frame_ids.tolist().index(f)) for f in args.frames]
    ref_model = official_model(args.official_repo, weights)
    result: dict = {"checkpoint": str(weights), "frames": args.frames, "reference_equivalence": {}}

    desc_cpu = None
    for mode in ("squash", "aspect"):
        ours = DINOv2Encoder(
            checkpoint_dir=resolve_path(args.checkpoint_dir), device="cpu", resize_mode=mode
        )
        batch = torch.stack([ours.preprocess(im) for im in images])
        t0 = time.perf_counter()
        with torch.inference_mode():
            feats = ref_model.forward_features(batch)
        ref = torch.cat([feats["x_norm_clstoken"][:, None], feats["x_norm_patchtokens"]], 1)
        t_ref = time.perf_counter() - t0
        t0 = time.perf_counter()
        tokens = ours.encode_tokens(images)
        t_ours = time.perf_counter() - t0
        desc = ours.encode_batch(images)
        ref_desc = torch.nn.functional.normalize(ref[:, 0], dim=-1).numpy()
        diff = (tokens - ref).abs()
        result["reference_equivalence"][mode] = {
            "input_shape": list(batch.shape),
            "token_shape": list(tokens.shape),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "token_abs_max": float(ref.abs().max()),
            "min_cls_descriptor_cosine": float((ref_desc * desc).sum(1).min()),
            "seconds_official": round(t_ref, 2),
            "seconds_ours": round(t_ours, 2),
        }
        if mode == "squash":
            desc_cpu = desc
            result["num_params"] = ours.num_params()
        ours.unload()

    if torch.cuda.is_available():
        gpu = DINOv2Encoder(checkpoint_dir=resolve_path(args.checkpoint_dir), device="cuda")
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
