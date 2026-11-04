"""Top-down trajectory plots in the KITTI x-z ground plane."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from latent_loop.visualization.plots import (
    COLOR_END,
    COLOR_REVISIT,
    COLOR_SAMPLED,
    COLOR_START,
    COLOR_TRAJECTORY,
    TIME_CMAP,
    apply_style,
)


def plot_trajectory_by_time(
    ax: plt.Axes,
    positions: np.ndarray,
    timestamps: np.ndarray,
    linewidth: float = 2.0,
    colorbar: bool = True,
) -> None:
    """Trajectory in the x-z plane coloured by time, with start/end markers."""
    xz = positions[:, [0, 2]]
    segments = np.stack([xz[:-1], xz[1:]], axis=1)
    lines = LineCollection(segments, cmap=TIME_CMAP, linewidths=linewidth, zorder=3)
    lines.set_array(timestamps[:-1])
    ax.add_collection(lines)
    ax.scatter(*xz[0], s=60, c=COLOR_START, marker="o", zorder=5, label="start")
    ax.scatter(*xz[-1], s=70, c=COLOR_END, marker="X", zorder=5, label="end")
    if colorbar:
        plt.colorbar(lines, ax=ax, fraction=0.046, pad=0.02, label="time [s]")
    format_xz_axes(ax)


def format_xz_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("x [m] (right of initial heading)")
    ax.set_ylabel("z [m] (initial forward direction)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()


def plot_trajectory_overview(
    positions: np.ndarray,
    timestamps: np.ndarray,
    frame_ids: np.ndarray,
    sampled_positions: np.ndarray,
    sampled_frame_ids: np.ndarray,
    stride: int,
    revisit_pairs: np.ndarray | None = None,
    title: str = "Ground-truth trajectory",
) -> plt.Figure:
    """(a) ground-truth trajectory, (b) sampled frame positions and, if given,
    (c) the revisit structure as query frame ID vs. reference frame ID.

    ``revisit_pairs`` holds ``(query_frame_id, reference_frame_id)`` rows whose IDs are
    contained in ``frame_ids``.
    """
    apply_style()
    has_revisits = revisit_pairs is not None and len(revisit_pairs) > 0
    ncols = 3 if has_revisits else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.4 * ncols, 5.2))
    ax_gt, ax_s = axes[0], axes[1]
    id_to_index = {int(f): i for i, f in enumerate(frame_ids)}

    plot_trajectory_by_time(ax_gt, positions, timestamps)
    if has_revisits:
        query_idx = np.array([id_to_index[int(q)] for q in revisit_pairs[:, 0]])
        xz = positions[query_idx][:, [0, 2]]
        ax_gt.scatter(
            xz[:, 0],
            xz[:, 1],
            s=26,
            facecolors="none",
            edgecolors=COLOR_REVISIT,
            linewidths=0.6,
            alpha=0.6,
            zorder=2,
            label=f"revisiting frames ({len(query_idx)})",
        )
    length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    ax_gt.set_title(
        f"(a) {title}\n{len(positions)} frames, {length:.0f} m, "
        f"{timestamps[-1] - timestamps[0]:.1f} s"
    )
    ax_gt.legend(loc="center", fontsize=8)

    ax_s.plot(
        positions[:, 0],
        positions[:, 2],
        color=COLOR_TRAJECTORY,
        lw=1,
        alpha=0.4,
        label="ground truth (all frames)",
    )
    ax_s.scatter(
        sampled_positions[:, 0],
        sampled_positions[:, 2],
        s=14,
        c=COLOR_SAMPLED,
        zorder=4,
        label=f"sampled (stride {stride}, n={len(sampled_positions)})",
    )
    step = max(1, len(sampled_frame_ids) // 10)
    for k, (pos, frame_id) in enumerate(
        zip(sampled_positions[::step], sampled_frame_ids[::step], strict=True)
    ):
        offset = (5, 5) if k % 2 == 0 else (5, -11)  # alternate to separate overlapping laps
        ax_s.annotate(
            f"#{frame_id}",
            (pos[0], pos[2]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            color=COLOR_SAMPLED,
        )
    format_xz_axes(ax_s)
    ax_s.set_title(f"(b) Frame sampling\nkeep frame f iff (f - start) mod {stride} = 0")
    ax_s.legend(loc="center", fontsize=8)

    if has_revisits:
        ax_r = axes[2]
        ref_idx = np.array([id_to_index[int(r)] for r in revisit_pairs[:, 1]])
        dist = np.linalg.norm(positions[query_idx] - positions[ref_idx], axis=1)
        sc = ax_r.scatter(
            revisit_pairs[:, 0],
            revisit_pairs[:, 1],
            c=dist,
            s=10,
            cmap="magma_r",
            vmin=0.0,
            vmax=max(1.0, float(dist.max())),
        )
        plt.colorbar(sc, ax=ax_r, fraction=0.046, pad=0.02, label="camera distance [m]")
        ax_r.set_xlabel("query frame ID (second lap)")
        ax_r.set_ylabel("reference frame ID (first lap)")
        gap = revisit_pairs[:, 0] - revisit_pairs[:, 1]
        ax_r.set_title(
            f"(c) Constructed revisit pairs: {len(revisit_pairs)}\n"
            f"frame gap {gap.min()}-{gap.max()}, distance <= {dist.max():.2f} m"
        )
    fig.tight_layout()
    return fig
