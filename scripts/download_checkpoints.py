#!/usr/bin/env python
"""Download an official I-JEPA checkpoint and extract its target encoder.

Usage:
    python scripts/download_checkpoints.py list
    python scripts/download_checkpoints.py ijepa --name vith14_in1k
    python scripts/download_checkpoints.py ijepa --name vith14_in1k --skip-download

The official files are full training states (~10.4 GB for ViT-H). Only the target
encoder (~2.5 GB in FP32) is needed for inference; ``--delete-raw`` removes the raw file
after a successful extraction. An existing raw file in ``checkpoints/`` or
``checkpoints/ijepa/`` is reused. The weights are licensed CC BY-NC 4.0.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys

from latent_loop.encoders.ijepa import (
    CHECKPOINTS,
    extract_target_encoder,
    find_raw_checkpoint,
    target_encoder_path,
)
from latent_loop.utils.config import resolve_path
from latent_loop.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the official checkpoints")
    ij = sub.add_parser("ijepa", help="download and extract an I-JEPA checkpoint")
    ij.add_argument("--name", default="vith14_in1k", choices=sorted(CHECKPOINTS))
    ij.add_argument("--dir", default="checkpoints", help="checkpoint root directory")
    ij.add_argument("--skip-download", action="store_true", help="only extract an existing file")
    ij.add_argument("--delete-raw", action="store_true", help="delete the raw file afterwards")
    args = parser.parse_args()
    setup_logging()

    if args.command == "list":
        for spec in CHECKPOINTS.values():
            print(
                f"{spec.key:16s} {spec.arch:9s} patch {spec.patch_size:2d}  {spec.img_size} px  "
                f"{spec.pretraining:34s} {spec.url}"
            )
        return 0

    spec = CHECKPOINTS[args.name]
    root = resolve_path(args.dir)
    raw = find_raw_checkpoint(root, spec)
    if raw is None and not args.skip_download:
        if shutil.which("curl") is None:
            print("curl is required for downloading", file=sys.stderr)
            return 1
        raw = root / "ijepa" / spec.filename
        raw.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {spec.url}\n -> {raw} (resumable with -C -)")
        subprocess.run(
            ["curl", "-fL", "--retry", "5", "-C", "-", "-o", str(raw), spec.url], check=True
        )
    if raw is None or not raw.is_file():
        print(f"Raw checkpoint {spec.filename} not found under {root}", file=sys.stderr)
        return 1

    out = target_encoder_path(root, spec.key)
    print(f"Extracting target encoder from {raw}\n -> {out}")
    meta = extract_target_encoder(raw, out, spec)
    print(json.dumps(meta, indent=2))
    if args.delete_raw:
        raw.unlink()
        print(f"Deleted {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
