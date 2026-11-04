#!/usr/bin/env python
"""Prepare or validate datasets.

Usage:
    # Export the deterministic synthetic sequence in KITTI Odometry layout
    python scripts/prepare_dataset.py synthetic --out data/processed/synthetic_kitti

    # Validate a local KITTI Odometry copy and print a summary
    python scripts/prepare_dataset.py kitti --root data/raw/kitti/dataset --sequence 00
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from latent_loop.datasets import KittiOdometryDataset, SyntheticConfig, SyntheticLoopDataset
from latent_loop.datasets.ground_truth import positions_from_poses, trajectory_length
from latent_loop.datasets.synthetic import export_kitti_layout
from latent_loop.utils.config import load_config, resolve_path
from latent_loop.utils.logging import setup_logging

KITTI_HELP = """\
KITTI Odometry is not downloaded automatically (registration required, large archives).
1. Register and download from https://www.cvlibs.net/datasets/kitti/eval_odometry.php
     - data_odometry_color.zip  (~65 GB, cameras 2/3)  or  data_odometry_gray.zip (~22 GB)
     - data_odometry_calib.zip  (calib.txt + times.txt)
     - data_odometry_poses.zip  (ground truth for sequences 00-10)
2. Unzip all archives into data/raw/kitti/ so that you get
     data/raw/kitti/dataset/sequences/00/image_2/000000.png
     data/raw/kitti/dataset/sequences/00/calib.txt
     data/raw/kitti/dataset/sequences/00/times.txt
     data/raw/kitti/dataset/poses/00.txt
"""


def cmd_synthetic(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dataset = SyntheticLoopDataset(SyntheticConfig.from_dict(config["dataset"].get("synthetic")))
    out = resolve_path(args.out)
    seq_dir = export_kitti_layout(dataset, out, args.sequence)
    print(json.dumps(dataset.summary(), indent=2, default=str))
    print(f"Wrote {len(dataset.available_frame_ids)} frames to {seq_dir}")


def cmd_kitti(args: argparse.Namespace) -> None:
    root = resolve_path(args.root)
    try:
        dataset = KittiOdometryDataset(root, args.sequence, args.camera)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}\n\n{KITTI_HELP}", file=sys.stderr)
        sys.exit(1)
    info = dataset.summary()
    info["missing_frame_ids"] = dataset.missing_frame_ids[:20].tolist()
    info["timestamps_estimated"] = dataset.timestamps_estimated
    calib = dataset.get_calibration()
    info["K"] = np.round(calib.K, 4).tolist()
    info["T_cam0_cam_translation_m"] = np.round(calib.T_cam0_cam[:3, 3], 4).tolist()
    if dataset.has_poses:
        info["trajectory_length_m"] = round(
            trajectory_length(positions_from_poses(dataset.poses())), 1
        )
    print(json.dumps(info, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or validate datasets.")
    sub = parser.add_subparsers(dest="command", required=True)

    syn = sub.add_parser("synthetic", help="export the synthetic sequence in KITTI layout")
    syn.add_argument("--config", default="configs/default.yaml")
    syn.add_argument("--out", type=Path, default=Path("data/processed/synthetic_kitti"))
    syn.add_argument("--sequence", default="00")
    syn.set_defaults(func=cmd_synthetic)

    kitti = sub.add_parser("kitti", help="validate a local KITTI Odometry copy")
    kitti.add_argument("--root", type=Path, default=Path("data/raw/kitti/dataset"))
    kitti.add_argument("--sequence", default="00")
    kitti.add_argument("--camera", type=int, default=2)
    kitti.set_defaults(func=cmd_kitti)

    args = parser.parse_args()
    setup_logging("INFO")
    args.func(args)


if __name__ == "__main__":
    main()
