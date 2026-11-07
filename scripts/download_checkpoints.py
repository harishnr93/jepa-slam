#!/usr/bin/env python
"""Download official checkpoints: I-JEPA (target encoder extracted) and DINOv2.

Usage:
    python scripts/download_checkpoints.py list
    python scripts/download_checkpoints.py ijepa --name vith14_in1k
    python scripts/download_checkpoints.py ijepa --name vith14_in1k --skip-download
    python scripts/download_checkpoints.py dinov2 --name vitb14       # ~346 MB
    python scripts/download_checkpoints.py dinov3 --url "<URL from Meta's e-mail>"
    python scripts/download_checkpoints.py dinov3 --file PATH/TO/dinov3_vitb16_*.pth

DINOv3 weights are licence-gated (DINOv3 License): request them at
https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ and pass the personal
URL from Meta's e-mail (or a file you downloaded). The file is accepted only if its SHA-256
starts with the hash in the official file name (e.g. 73cec8be).

DINOv2 backbones are stored as checkpoints/dinov2/<file>.pth with a JSON sidecar holding
the SHA-256 (part of the embedding-cache fingerprint). An existing file of the expected
size is reused. The DINOv2 weights are licensed Apache-2.0.

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

from latent_loop.encoders.dinov2 import CHECKPOINTS as DINO_CHECKPOINTS
from latent_loop.encoders.dinov2 import checkpoint_path as dino_checkpoint_path
from latent_loop.encoders.dinov2 import write_sidecar
from latent_loop.encoders.dinov3 import CHECKPOINTS as DINO3_CHECKPOINTS
from latent_loop.encoders.dinov3 import checkpoint_path as dino3_checkpoint_path
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
    dn = sub.add_parser("dinov2", help="download a DINOv2 backbone checkpoint")
    dn.add_argument("--name", default="vitb14", choices=sorted(DINO_CHECKPOINTS))
    dn.add_argument("--dir", default="checkpoints", help="checkpoint root directory")
    d3 = sub.add_parser("dinov3", help="fetch a licence-gated DINOv3 backbone (URL or file)")
    d3.add_argument("--name", default="vitb16", choices=sorted(DINO3_CHECKPOINTS))
    d3.add_argument("--dir", default="checkpoints", help="checkpoint root directory")
    src = d3.add_mutually_exclusive_group()
    src.add_argument("--url", help="personal download URL from Meta's e-mail")
    src.add_argument("--file", help="an already downloaded checkpoint file")
    args = parser.parse_args()
    setup_logging()

    if args.command == "dinov3":
        spec3 = DINO3_CHECKPOINTS[args.name]
        path = dino3_checkpoint_path(resolve_path(args.dir), args.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.file:
            shutil.copyfile(args.file, path)
        elif args.url:
            if shutil.which("curl") is None:
                print("curl is required for downloading", file=sys.stderr)
                return 1
            print(f"Downloading {spec3.filename}\n -> {path} (resumable with -C -)")
            subprocess.run(
                ["curl", "-fL", "--retry", "5", "-C", "-", "-o", str(path), args.url], check=True
            )
        elif not path.is_file():
            print(
                f"{path} not found. DINOv3 weights are licence-gated; pass --url or --file "
                "(see the module docstring)",
                file=sys.stderr,
            )
            return 1
        meta = write_sidecar(path)
        if not meta["sha256"].startswith(spec3.sha256_prefix):
            print(
                f"SHA-256 {meta['sha256'][:8]}... does not match the official "
                f"{spec3.sha256_prefix}; refusing the file",
                file=sys.stderr,
            )
            path.with_suffix(".json").unlink()
            return 1
        print(json.dumps(meta, indent=2))
        return 0

    if args.command == "dinov2":
        spec = DINO_CHECKPOINTS[args.name]
        path = dino_checkpoint_path(resolve_path(args.dir), args.name)
        if path.is_file() and path.stat().st_size == spec.size_bytes:
            print(f"Reusing {path} ({spec.size_bytes} bytes)")
        else:
            if shutil.which("curl") is None:
                print("curl is required for downloading", file=sys.stderr)
                return 1
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {spec.url}\n -> {path} (resumable with -C -)")
            subprocess.run(
                ["curl", "-fL", "--retry", "5", "-C", "-", "-o", str(path), spec.url], check=True
            )
        if path.stat().st_size != spec.size_bytes:
            print(f"Size mismatch: {path.stat().st_size} != {spec.size_bytes}", file=sys.stderr)
            return 1
        print(json.dumps(write_sidecar(path), indent=2))
        return 0

    if args.command == "list":
        for spec in CHECKPOINTS.values():
            print(
                f"ijepa  {spec.key:16s} {spec.arch:9s} patch {spec.patch_size:2d}  "
                f"{spec.img_size} px  {spec.pretraining:34s} {spec.url}"
            )
        for d in DINO_CHECKPOINTS.values():
            print(
                f"dinov2 {d.key:16s} vit_base  patch {d.patch_size:2d}  {d.train_size} px  {d.url}"
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
