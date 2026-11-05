#!/usr/bin/env python
"""Generate the figures and GIFs of a project phase from real dataset/pipeline outputs.

Usage:
    python scripts/generate_visualizations.py --phase 1                       # synthetic data
    python scripts/generate_visualizations.py --phase 1 --config configs/kitti.yaml
    python scripts/generate_visualizations.py --phase 1 --stride 5 --max-frames 80 --gif-fps 8
    python scripts/generate_visualizations.py --phase 2                       # I-JEPA, synthetic
    python scripts/generate_visualizations.py --phase 2 --stride 1 --no-preprocessing-comparison
    python scripts/generate_visualizations.py --phase 3                       # retrieval, synthetic
    python scripts/generate_visualizations.py --phase 4                       # verification
    python scripts/generate_visualizations.py --phase 5                       # baselines
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from latent_loop.datasets import SyntheticLoopDataset, build_dataset
from latent_loop.utils.config import load_config, resolve_path
from latent_loop.utils.logging import get_logger, setup_logging
from latent_loop.visualization.animations import (
    render_trajectory_camera_frames,
    save_gif,
    select_animation_indices,
)
from latent_loop.visualization.plots import plot_frame_montage, save_figure
from latent_loop.visualization.trajectory import plot_trajectory_overview

logger = get_logger("generate_visualizations")
IMPLEMENTED_PHASES = (1, 2, 3, 4, 5)
DEFAULT_CONFIGS = {
    1: "configs/default.yaml",
    2: "configs/ijepa.yaml",
    3: "configs/ijepa.yaml",
    4: "configs/ijepa.yaml",
    5: "configs/dinov2.yaml",
}
DEFAULT_OUTPUT_DIRS = {
    2: "assets/phase_02_ijepa",
    3: "assets/phase_03_retrieval",
    4: "assets/phase_04_geometry",
    5: "assets/phase_05_baselines",
}


def _dataset_tag(config: dict[str, Any]) -> str:
    ds = config["dataset"]
    if ds["name"] == "kitti":
        return f"kitti_{int(ds['sequence']):02d}_cam{ds.get('camera', 2)}"
    return ds["name"]


def phase1(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    vis = config["visualization"]
    out_dir = resolve_path(args.output_dir or vis["output_dir"])
    tag = _dataset_tag(config)
    sampling = dict(config["dataset"].get("sampling") or {})

    # Full-rate view (same start/end window) for the trajectory, sampled view for markers.
    full_cfg = json.loads(json.dumps(config))
    full_cfg["dataset"]["sampling"] = {**sampling, "stride": 1, "max_frames": None}
    full = build_dataset(full_cfg)
    stride = args.stride or int(vis.get("sample_stride", sampling.get("stride", 1)))
    sampled_cfg = json.loads(json.dumps(config))
    sampled_cfg["dataset"]["sampling"] = {**sampling, "stride": stride}
    sampled = build_dataset(sampled_cfg)
    if not full.has_poses:
        raise RuntimeError("Phase 1 trajectory plots need ground-truth poses for this sequence")

    poses = full.poses()
    positions = poses[:, :3, 3]
    timestamps = full.timestamps()
    full_index = {int(f): i for i, f in enumerate(full.frame_ids)}

    revisits: dict[int, int] = {}
    revisit_pairs = None
    if isinstance(full, SyntheticLoopDataset):
        revisit_pairs = np.array(
            [
                (q, r)
                for q, r in full.revisit_pairs
                if int(q) in full_index and int(r) in full_index
            ],
            dtype=np.int64,
        ).reshape(-1, 2)
        revisits = {int(q): int(r) for q, r in revisit_pairs}

    outputs: dict[str, str] = {}

    # 1) Camera image sequence montage (evenly spaced sampled frames).
    montage_idx = select_animation_indices(len(sampled), 8)
    images, captions = [], []
    for i in montage_idx:
        frame = sampled.get_frame(int(i))
        x, z = frame.pose[0, 3], frame.pose[2, 3]
        caption = f"#{frame.frame_id:05d}  t={frame.timestamp:.1f} s  (x={x:.1f}, z={z:.1f}) m"
        if frame.frame_id in revisits:
            caption += f"  revisit of #{revisits[frame.frame_id]:05d}"
        images.append(frame.image)
        captions.append(caption)
    fig = plot_frame_montage(
        images, captions, ncols=2, title=f"Camera image sequence ({tag}, stride {stride})"
    )
    outputs["camera_sequence"] = str(
        save_figure(fig, out_dir / f"{tag}_camera_sequence.png", dpi=vis["dpi"])
    )

    # 2) Ground-truth trajectory + sampled frame positions.
    fig = plot_trajectory_overview(
        positions,
        timestamps,
        full.frame_ids,
        sampled_positions=sampled.poses()[:, :3, 3],
        sampled_frame_ids=sampled.frame_ids,
        stride=stride,
        revisit_pairs=revisit_pairs,
        title=f"Ground truth ({tag})",
    )
    outputs["trajectory"] = str(save_figure(fig, out_dir / f"{tag}_trajectory.png", dpi=vis["dpi"]))

    # 3) GIF: vehicle trajectory with synchronised camera frames.
    max_frames = args.max_frames or vis["gif_max_frames"]
    gif_idx = select_animation_indices(len(full), max_frames)
    start = time.perf_counter()
    frames = render_trajectory_camera_frames(
        full,
        gif_idx,
        width_px=args.gif_width or vis["gif_width_px"],
        revisits=revisits,
        title=f"Latent-Loop Phase 1: trajectory and camera ({tag})",
    )
    gif_path = save_gif(
        frames, out_dir / f"{tag}_trajectory_camera.gif", fps=args.gif_fps or vis["gif_fps"]
    )
    outputs["trajectory_gif"] = str(gif_path)

    summary = {
        "phase": 1,
        "dataset": full.summary(),
        "visualization_stride": stride,
        "gif_frames": len(frames),
        "gif_size_mb": round(gif_path.stat().st_size / 1e6, 2),
        "gif_render_seconds": round(time.perf_counter() - start, 1),
        "outputs": outputs,
    }
    (out_dir / f"{tag}_phase1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate phase visualizations.")
    parser.add_argument("--phase", type=int, required=True, choices=range(1, 9))
    parser.add_argument(
        "--config", help="default: configs/default.yaml (phase 1), configs/ijepa.yaml (phase 2)"
    )
    parser.add_argument("--sequence", help="override dataset.sequence (KITTI)")
    parser.add_argument("--output-dir", help="override visualization.output_dir")
    parser.add_argument("--stride", type=int, help="frame stride for sampled-frame plots")
    parser.add_argument("--max-frames", type=int, help="maximum GIF frames")
    parser.add_argument("--gif-pairs", type=int, help="phase 4: evenly spaced pairs in the GIF")
    parser.add_argument("--gif-fps", type=float, help="GIF frame rate")
    parser.add_argument("--gif-width", type=int, help="GIF width in pixels")
    parser.add_argument("--gif-colors", type=int, help="phase 5: GIF palette size (default 256)")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="config overrides, e.g. dataset.synthetic.seed=3",
    )
    parser.add_argument("--encoder", help="override config 'encoder' (phase 2+)")
    parser.add_argument(
        "--no-preprocessing-comparison",
        action="store_true",
        help="phase 2: skip extracting the center_crop/aspect variants",
    )
    args = parser.parse_args()

    overrides = list(args.set)
    if args.sequence:
        overrides.append(f"dataset.sequence='{args.sequence}'")
    if args.encoder:
        overrides.append(f"encoder={args.encoder}")
    config = load_config(args.config or DEFAULT_CONFIGS.get(args.phase), overrides)
    setup_logging(config["logging"]["level"], resolve_path(config["logging"].get("file")))

    if args.phase not in IMPLEMENTED_PHASES:
        parser.error(
            f"Phase {args.phase} visualizations are not implemented yet "
            f"(implemented: {IMPLEMENTED_PHASES})"
        )
    if args.phase == 1:
        summary = phase1(config, args)
    elif args.phase == 5:
        from latent_loop.visualization.phase5 import generate_phase5  # noqa: PLC0415

        vis = config["visualization"]
        summary = generate_phase5(
            config,
            resolve_path(args.output_dir or DEFAULT_OUTPUT_DIRS[5]),
            vis_stride=args.stride or 2,
            gif_frames=args.max_frames or 100,
            gif_fps=args.gif_fps or 8.0,
            gif_width=args.gif_width or vis["gif_width_px"],
            gif_colors=args.gif_colors or 256,
        )
    elif args.phase == 4:
        from latent_loop.visualization.phase4 import generate_phase4  # noqa: PLC0415

        vis = config["visualization"]
        summary = generate_phase4(
            config,
            resolve_path(args.output_dir or DEFAULT_OUTPUT_DIRS[4]),
            vis_stride=args.stride or 2,
            gif_pairs=args.gif_pairs or 12,
            gif_fps=args.gif_fps or 8.0,
            gif_width=args.gif_width or vis["gif_width_px"],
        )
    elif args.phase == 3:
        from latent_loop.visualization.phase3 import generate_phase3  # noqa: PLC0415

        vis = config["visualization"]
        summary = generate_phase3(
            config,
            resolve_path(args.output_dir or DEFAULT_OUTPUT_DIRS[3]),
            vis_stride=args.stride or 2,
            gif_frames=args.max_frames or vis["gif_max_frames"],
            gif_fps=args.gif_fps or 8.0,
            gif_width=args.gif_width or vis["gif_width_px"],
        )
    else:
        from latent_loop.visualization.phase2 import generate_phase2  # noqa: PLC0415

        vis = config["visualization"]
        summary = generate_phase2(
            config,
            resolve_path(args.output_dir or DEFAULT_OUTPUT_DIRS[2]),
            vis_stride=args.stride or 2,
            gif_frames=args.max_frames or vis["gif_max_frames"],
            gif_fps=args.gif_fps or vis["gif_fps"],
            gif_width=args.gif_width or vis["gif_width_px"],
            compare_preprocessing=not args.no_preprocessing_comparison,
        )
    logger.info(
        f"Phase {args.phase} visualizations written", extra={"gif_mb": summary["gif_size_mb"]}
    )
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
