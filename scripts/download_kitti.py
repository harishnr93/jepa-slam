#!/usr/bin/env python
"""Download the parts of KITTI Odometry needed for loop-closure experiments.

Usage:
    python scripts/download_kitti.py                       # loop sequences, left colour camera
    python scripts/download_kitti.py --sequences 00 05 --cameras 2
    python scripts/download_kitti.py --list-only           # sizes only, nothing written
    python scripts/download_kitti.py --gray --cameras 0    # grayscale archive instead

Files are taken from the official archives at https://s3.eu-central-1.amazonaws.com/avg-kitti/
with HTTP range requests, so only the selected sequences/cameras are transferred. Every
file is checked against the CRC-32 in the zip directory. Re-running resumes.

Output layout (what configs/kitti.yaml expects)::

    data/raw/kitti/dataset/sequences/<SS>/{image_2/*.png, calib.txt, times.txt}
    data/raw/kitti/dataset/poses/<SS>.txt     (ground truth exists for 00-10)

The KITTI data is licensed CC BY-NC-SA 3.0 (non-commercial); see
https://www.cvlibs.net/datasets/kitti/.
"""

from __future__ import annotations

import argparse
import json
import time

from latent_loop.datasets.kitti_download import (
    ARCHIVES,
    KITTI_BASE_URL,
    LOOP_SEQUENCES,
    DownloadReport,
    HttpRangeSource,
    download_members,
    image_patterns,
    list_members,
    select_members,
)
from latent_loop.utils.config import resolve_path
from latent_loop.utils.logging import get_logger, setup_logging

logger = get_logger("download_kitti")


def _progress(report: DownloadReport) -> None:
    rate = report.bytes_downloaded / max(report.seconds, 1e-9) / 1e6
    logger.info(
        "Downloading",
        extra={
            "files": report.downloaded,
            "gb": round(report.bytes_downloaded / 1e9, 2),
            "mb_per_s": round(rate, 1),
        },
    )


def fetch(archive: str, patterns: list[str], out_root, list_only: bool, attempts: int) -> dict:
    source = HttpRangeSource(KITTI_BASE_URL + ARCHIVES[archive])
    members = select_members(list_members(source), patterns)
    total = sum(m.compress_size for m in members)
    summary = {
        "archive": ARCHIVES[archive],
        "archive_gb": round(source.size / 1e9, 2),
        "files": len(members),
        "selected_gb": round(total / 1e9, 3),
    }
    if list_only or not members:
        return summary
    for attempt in range(1, attempts + 1):
        try:
            report = download_members(
                source, members, out_root, strip_prefix="", progress=_progress
            )
            summary.update(
                {
                    "downloaded": report.downloaded,
                    "skipped_existing": report.skipped_existing,
                    "crc_verified": report.crc_verified,
                    "seconds": round(report.seconds, 1),
                }
            )
            return summary
        except (OSError, EOFError, ValueError) as exc:
            if attempt == attempts:
                raise
            logger.warning(
                "Download interrupted; resuming", extra={"attempt": attempt, "error": str(exc)}
            )
            time.sleep(5 * attempt)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sequences", nargs="+", default=list(LOOP_SEQUENCES))
    parser.add_argument("--cameras", nargs="+", type=int, default=[2])
    parser.add_argument("--gray", action="store_true", help="use the grayscale archive (cam 0/1)")
    parser.add_argument("--out", default="data/raw/kitti", help="root that receives dataset/")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--attempts", type=int, default=20, help="resume attempts per archive")
    args = parser.parse_args()
    setup_logging()
    sequences = [f"{int(s):02d}" for s in args.sequences]
    out_root = resolve_path(args.out)

    results = [
        fetch(
            "calib",
            ["dataset/sequences/*/calib.txt", "dataset/sequences/*/times.txt"],
            out_root,
            args.list_only,
            args.attempts,
        ),
        fetch("poses", ["dataset/poses/*.txt"], out_root, args.list_only, args.attempts),
        fetch(
            "gray" if args.gray else "color",
            image_patterns(sequences, args.cameras),
            out_root,
            args.list_only,
            args.attempts,
        ),
    ]
    print(
        json.dumps(
            {
                "out_root": str(out_root),
                "sequences": sequences,
                "cameras": args.cameras,
                "archives": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
