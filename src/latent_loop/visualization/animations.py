"""Animated GIFs built from real dataset frames and poses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from latent_loop.datasets.base import VisualSequenceDataset
from latent_loop.datasets.ground_truth import yaw_from_poses
from latent_loop.visualization.plots import (
    COLOR_END,
    COLOR_REVISIT,
    COLOR_SAMPLED,
    COLOR_TRAJECTORY,
    apply_style,
    figure_to_array,
    plt,
)


def save_gif(frames: Sequence[np.ndarray], path: str | Path, fps: float = 10.0) -> Path:
    """Write RGB frames as a looping GIF.

    One 256-colour palette is fitted on a strip of up to 8 evenly spaced frames and
    applied to every frame without dithering; dithering noise defeats LZW compression.
    """
    if not frames:
        raise ValueError("No frames to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    picks = np.unique(np.linspace(0, len(frames) - 1, min(8, len(frames))).round().astype(int))
    strip = Image.fromarray(np.concatenate([frames[i] for i in picks], axis=0))
    palette = strip.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    images = [
        Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames
    ]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        optimize=True,
    )
    return path


def select_animation_indices(num_frames: int, max_frames: int) -> np.ndarray:
    """Evenly spaced view indices (first and last included), at most ``max_frames``."""
    if num_frames <= 0:
        return np.zeros(0, dtype=np.int64)
    count = min(num_frames, max_frames)
    return np.unique(np.linspace(0, num_frames - 1, count).round().astype(np.int64))


def render_trajectory_camera_frames(
    dataset: VisualSequenceDataset,
    indices: Sequence[int],
    width_px: int = 900,
    revisits: Mapping[int, int] | None = None,
    title: str = "",
) -> list[np.ndarray]:
    """Render synchronized (camera image, top-down trajectory) frames.

    ``revisits`` maps a query frame ID to the earlier frame ID of the same place; when
    the current frame is a revisit, a link to that earlier position is drawn.
    """
    apply_style()
    poses = dataset.poses()
    positions = poses[:, :3, 3]
    yaws = yaw_from_poses(poses)
    timestamps = dataset.timestamps()
    frame_ids = dataset.frame_ids
    id_to_index = {int(f): i for i, f in enumerate(frame_ids)}
    sample_image = dataset.get_image(int(indices[0]))
    img_h, img_w = sample_image.shape[:2]

    dpi = 100
    fig_w = width_px / dpi
    cam_h = fig_w * img_h / img_w
    traj_h = 3.6
    fig = plt.figure(figsize=(fig_w, cam_h + traj_h + 0.5), dpi=dpi)
    total_h = cam_h + traj_h + 0.5
    ax_cam = fig.add_axes((0.0, (traj_h + 0.15) / total_h, 1.0, cam_h / total_h))
    ax_traj = fig.add_axes((0.08, 0.09, 0.88, (traj_h - 0.45) / total_h))
    ax_cam.axis("off")
    im_artist = ax_cam.imshow(sample_image, cmap="gray" if sample_image.ndim == 2 else None)
    cam_text = ax_cam.text(
        0.01,
        0.95,
        "",
        transform=ax_cam.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="white",
        family="monospace",
        bbox={"facecolor": "black", "alpha": 0.6, "pad": 3, "edgecolor": "none"},
    )
    title_text = fig.text(
        0.5, 1 - 0.08 / total_h, title, ha="center", va="top", fontsize=10, fontweight="bold"
    )
    title_text.set_visible(bool(title))

    ax_traj.plot(
        positions[:, 0],
        positions[:, 2],
        color=COLOR_TRAJECTORY,
        lw=1,
        alpha=0.3,
        label="ground-truth trajectory",
    )
    (trail,) = ax_traj.plot([], [], color=COLOR_SAMPLED, lw=2.2, label="travelled")
    (revisit_trail,) = ax_traj.plot(
        [], [], color=COLOR_REVISIT, lw=3.5, alpha=0.8, label="travelled, revisiting (GT)"
    )
    revisit_flags = np.array([int(f) in (revisits or {}) for f in frame_ids])
    (link,) = ax_traj.plot([], [], color=COLOR_REVISIT, lw=2, ls="--")
    ref_marker = ax_traj.scatter(
        [], [], s=60, facecolors="none", edgecolors=COLOR_REVISIT, linewidths=2, zorder=6
    )
    current = ax_traj.scatter([], [], s=55, c=COLOR_END, zorder=7, label="current pose")
    span = float(np.ptp(positions[:, [0, 2]], axis=0).max()) or 1.0
    arrow_len = 0.06 * span
    heading = ax_traj.quiver(
        [0],
        [0],
        [0],
        [1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=COLOR_END,
        width=0.006,
        zorder=8,
    )
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("z [m]")
    ax_traj.set_aspect("equal", adjustable="box")
    pad = 0.05 * span
    ax_traj.set_xlim(positions[:, 0].min() - pad, positions[:, 0].max() + pad)
    ax_traj.set_ylim(positions[:, 2].min() - pad, positions[:, 2].max() + pad)
    ax_traj.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7, frameon=False)

    frames = []
    for index in indices:
        index = int(index)
        frame_id = int(frame_ids[index])
        image = dataset.get_image(index)
        im_artist.set_data(image)
        x, z = positions[index, 0], positions[index, 2]
        trail.set_data(positions[: index + 1, 0], positions[: index + 1, 2])
        seen_revisits = np.where(revisit_flags[: index + 1], 1.0, np.nan)  # NaN breaks the line
        revisit_trail.set_data(
            positions[: index + 1, 0] * seen_revisits, positions[: index + 1, 2] * seen_revisits
        )
        current.set_offsets([[x, z]])
        heading.set_offsets([[x, z]])
        heading.set_UVC([arrow_len * np.sin(yaws[index])], [arrow_len * np.cos(yaws[index])])
        text = f"frame #{frame_id:05d}  t={timestamps[index]:6.1f} s  x={x:7.1f} m  z={z:7.1f} m"

        ref_id = revisits.get(frame_id) if revisits else None
        if ref_id is not None and ref_id in id_to_index:
            ref = positions[id_to_index[ref_id]]
            link.set_data([x, ref[0]], [z, ref[2]])
            ref_marker.set_offsets([[ref[0], ref[2]]])
            dist = float(np.linalg.norm(positions[index] - ref))
            text += f"\nREVISIT of #{ref_id:05d} ({dist:.2f} m apart, constructed GT)"
        else:
            link.set_data([], [])
            ref_marker.set_offsets(np.empty((0, 2)))
        cam_text.set_text(text)
        frames.append(figure_to_array(fig))
    plt.close(fig)
    return frames
