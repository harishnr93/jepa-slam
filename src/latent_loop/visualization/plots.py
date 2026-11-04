"""Shared plotting helpers and the camera image montage."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
}

# Fixed semantic colours reused across phases.
COLOR_TRAJECTORY = "#4a5568"
COLOR_SAMPLED = "#2b6cb0"
COLOR_REVISIT = "#dd6b20"
COLOR_START = "#2f855a"
COLOR_END = "#c53030"
TIME_CMAP = "viridis"


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 120) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_to_array(fig: plt.Figure) -> np.ndarray:
    """Rasterise a figure to an ``HxWx3`` uint8 RGB array."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def plot_frame_montage(
    images: Sequence[np.ndarray],
    captions: Sequence[str],
    ncols: int = 2,
    title: str | None = None,
    panel_width_in: float = 4.6,
) -> plt.Figure:
    """Grid of camera frames, each with a caption (e.g. frame ID, time, position)."""
    if len(images) != len(captions):
        raise ValueError("images and captions must have the same length")
    apply_style()
    nrows = int(np.ceil(len(images) / ncols))
    aspect = images[0].shape[0] / images[0].shape[1]
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * panel_width_in, nrows * (panel_width_in * aspect + 0.35)),
        squeeze=False,
    )
    for ax in axes.ravel():
        ax.axis("off")
    for ax, image, caption in zip(axes.ravel(), images, captions, strict=False):
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        ax.set_title(caption, fontsize=8, fontweight="normal", loc="left")
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig
