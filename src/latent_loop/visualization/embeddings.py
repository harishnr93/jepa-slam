"""Phase 2 visualisations of global and patch-level embeddings.

All functions take real embeddings (from the cache) and ground-truth labels from
:func:`latent_loop.datasets.ground_truth.pair_labels`; nothing is synthesised here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.colors import Normalize

from latent_loop.datasets.ground_truth import IGNORED, NEGATIVE, POSITIVE
from latent_loop.evaluation.metrics import roc_auc
from latent_loop.visualization.plots import (
    COLOR_REVISIT,
    COLOR_SAMPLED,
    COLOR_TRAJECTORY,
    TIME_CMAP,
    apply_style,
    figure_to_array,
    plt,
)

SIM_CMAP = "magma"


def pca(x: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Project rows of ``x`` onto the top-``k`` principal components.

    Returns ``(scores (N, k), explained_variance_ratio (min(N, D),))``.
    """
    centred = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    var = s**2
    return centred @ vt[:k].T, var / var.sum()


def separability(sim: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    pos, neg = sim[labels == POSITIVE], sim[labels == NEGATIVE]
    return {
        "positives": int(pos.size),
        "negatives": int(neg.size),
        "auc": roc_auc(pos, neg) if pos.size and neg.size else float("nan"),
        "pos_median": float(np.median(pos)) if pos.size else float("nan"),
        "neg_median": float(np.median(neg)) if neg.size else float("nan"),
        "neg_p99": float(np.percentile(neg, 99)) if neg.size else float("nan"),
    }


def plot_similarity_matrix(
    sim: np.ndarray,
    frame_ids: np.ndarray,
    labels: np.ndarray,
    title: str,
    min_gap_frames: int,
    positive_radius_m: float,
) -> plt.Figure:
    """Cosine similarity matrix (left) and the same matrix with GT revisits overlaid (right)."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), constrained_layout=True)
    extent = (frame_ids[0], frame_ids[-1], frame_ids[-1], frame_ids[0])
    vmin = float(np.percentile(sim, 1))
    for ax in axes:
        im = ax.imshow(
            sim, cmap=SIM_CMAP, vmin=vmin, vmax=1.0, extent=extent, interpolation="nearest"
        )
        ax.set_xlabel("reference frame ID")
        ax.set_ylabel("query frame ID")
        ax.grid(False)
    axes[0].set_title("Cosine similarity  $s_{ij} = d_i^\\top d_j$")
    qi, rj = np.nonzero(labels == POSITIVE)
    axes[1].scatter(
        frame_ids[rj],
        frame_ids[qi],
        s=2,
        c="#38bdf8",
        marker="s",
        linewidths=0,
        label=f"GT revisit pairs (≤ {positive_radius_m:g} m, ≥ {min_gap_frames} frames apart)",
    )
    gap = min_gap_frames
    axes[1].plot(
        [frame_ids[0], frame_ids[-1]],
        [frame_ids[0] + gap, frame_ids[-1] + gap],
        color="white",
        lw=1,
        ls="--",
        label=f"temporal exclusion boundary ({gap})",
    )
    axes[1].set_xlim(frame_ids[0], frame_ids[-1])
    axes[1].set_ylim(frame_ids[-1], frame_ids[0])
    axes[1].set_title("Lower triangle: earlier references; GT revisits highlighted")
    axes[1].legend(loc="upper right", fontsize=7.5, framealpha=0.85)
    fig.colorbar(im, ax=axes, shrink=0.85, label="cosine similarity")
    fig.suptitle(title, fontweight="bold")
    return fig


def plot_descriptor_distribution(
    emb: np.ndarray,
    sim: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    frame_ids: np.ndarray,
    timestamps: np.ndarray,
    title: str,
    positive_radius_m: float = 5.0,
    negative_radius_m: float = 25.0,
    min_gap_frames: int = 100,
) -> plt.Figure:
    """Similarity distributions (revisit vs different place), descriptor statistics, PCA."""
    apply_style()
    stats = separability(sim, labels)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), constrained_layout=True)

    ax = axes[0, 0]
    pos, neg = sim[labels == POSITIVE], sim[labels == NEGATIVE]
    bins = np.linspace(min(pos.min(), neg.min()), 1.0, 60)
    ax.hist(
        neg,
        bins=bins,
        density=True,
        alpha=0.65,
        color=COLOR_TRAJECTORY,
        label=f"different place (> {negative_radius_m:g} m), n={neg.size:,}",
    )
    ax.hist(
        pos,
        bins=bins,
        density=True,
        alpha=0.75,
        color=COLOR_REVISIT,
        label=f"GT revisit (≤ {positive_radius_m:g} m), n={pos.size:,}",
    )
    ax.axvline(stats["neg_p99"], color="black", lw=1, ls="--")
    ax.annotate(
        f"99th pct of negatives\n= {stats['neg_p99']:.3f}",
        (stats["neg_p99"], ax.get_ylim()[1] * 0.75),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        fontsize=8,
    )
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(
        f"Pair similarity (pairs ≥ {min_gap_frames} frames apart) — ROC AUC = {stats['auc']:.3f}"
    )
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[0, 1]
    ax.hist(emb.ravel(), bins=120, color=COLOR_SAMPLED, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("descriptor component value")
    ax.set_ylabel("count (log)")
    dims = emb.shape[1]
    ax.set_title(f"Descriptor values ({emb.shape[0]} frames × {dims} dims, L2-normalised)")
    norms = np.linalg.norm(emb, axis=1)
    ax.text(
        0.98,
        0.95,
        f"‖d‖₂ ∈ [{norms.min():.4f}, {norms.max():.4f}]\n"
        f"1/√D = {1 / np.sqrt(dims):.4f}\nmean |d_k| = {np.abs(emb).mean():.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    scores, ratio = pca(emb, 2)
    ax = axes[1, 0]
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=timestamps, cmap=TIME_CMAP, s=10)
    qi, rj = np.nonzero(labels == POSITIVE)
    if qi.size:
        first = {}
        for q, r in zip(qi, rj, strict=True):  # one link per query: its closest GT revisit
            d = np.linalg.norm(positions[q, [0, 2]] - positions[r, [0, 2]])
            if q not in first or d < first[q][1]:
                first[q] = (r, d)
        for q, (r, _) in list(first.items())[::4]:
            ax.plot(*np.stack([scores[q], scores[r]]).T, color=COLOR_REVISIT, lw=0.6, alpha=0.6)
        ax.plot([], [], color=COLOR_REVISIT, lw=1, label="query ↔ GT revisit (every 4th)")
        ax.legend(loc="best", fontsize=8)
    fig.colorbar(sc, ax=ax, label="time [s]")
    ax.set_xlabel(f"PC1 ({100 * ratio[0]:.1f} % var.)")
    ax.set_ylabel(f"PC2 ({100 * ratio[1]:.1f} % var.)")
    ax.set_title("PCA of global descriptors")

    ax = axes[1, 1]
    k = min(64, ratio.size)
    ax.plot(np.arange(1, k + 1), np.cumsum(ratio[:k]), marker="o", ms=3, color=COLOR_SAMPLED)
    for target in (0.5, 0.9):
        n = int(np.searchsorted(np.cumsum(ratio), target) + 1)
        ax.axhline(target, color="grey", lw=0.8, ls=":")
        ax.annotate(
            f"{target:.0%} at {n} PCs",
            (min(n, k), target),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.set_title("Effective dimensionality of the descriptors")
    fig.suptitle(title, fontweight="bold")
    return fig


def _overlay(
    image: np.ndarray, heat: np.ndarray, grid: tuple[int, int], norm: Normalize
) -> np.ndarray:
    import cv2  # noqa: PLC0415

    h, w = image.shape[:2]
    heat = cv2.resize(
        heat.reshape(grid).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
    )
    rgb = plt.get_cmap(SIM_CMAP)(norm(heat))[..., :3]
    base = image if image.ndim == 3 else np.dstack([image] * 3)
    return (0.35 * base / 255.0 + 0.65 * rgb).clip(0, 1)


def patch_heat(query_tokens: np.ndarray, ref_tokens: np.ndarray) -> np.ndarray:
    """``h_a = max_b cos(q_a, r_b)`` for every query patch ``a``."""
    q = query_tokens / np.linalg.norm(query_tokens, axis=1, keepdims=True)
    r = ref_tokens / np.linalg.norm(ref_tokens, axis=1, keepdims=True)
    return (q @ r.T).max(axis=1)


def plot_patch_similarity(
    pairs: Sequence[dict],
    grid: tuple[int, int],
    title: str,
    display_size: tuple[int, int],
) -> plt.Figure:
    """Per-patch feature similarity for revisit and non-revisit pairs.

    For L2-normalised query patch tokens ``q_a`` and reference tokens ``r_b`` the heat
    value is ``h_a = max_b cos(q_a, r_b)``: how well each query region is explained by
    *some* region of the reference image. The colour range is shared by all rows and
    set from the data (2nd percentile .. 1), because I-JEPA patch cosines are high.
    The last column shows the first three principal components of the patch tokens
    (fitted jointly on query and reference) as RGB: similar colours = similar features.
    """
    import cv2  # noqa: PLC0415

    apply_style()
    heats = [patch_heat(p["query_tokens"], p["ref_tokens"]) for p in pairs]
    vmin = float(np.floor(np.percentile(np.concatenate(heats), 2) * 100) / 100)
    norm = Normalize(vmin, 1.0)
    aspect = display_size[1] / display_size[0]
    fig, axes = plt.subplots(
        len(pairs),
        4,
        figsize=(18, 4.5 * aspect * len(pairs) + 1.2),
        constrained_layout=True,
        squeeze=False,
    )
    for row, pair, heat in zip(axes, pairs, heats, strict=True):
        q_img = cv2.resize(pair["query_image"], display_size, interpolation=cv2.INTER_AREA)
        r_img = cv2.resize(pair["ref_image"], display_size, interpolation=cv2.INTER_AREA)
        row[0].imshow(q_img)
        row[0].set_title(f"query #{pair['query_id']}", fontsize=9)
        row[1].imshow(r_img)
        row[1].set_title(
            f"ref #{pair['ref_id']}: {pair['label']}, {pair['distance_m']:.1f} m apart",
            fontsize=9,
        )
        row[2].imshow(_overlay(q_img, heat, grid, norm))
        row[2].set_title(
            f"max patch cos (mean {heat.mean():.3f}), global cos {pair['global_cos']:.3f}",
            fontsize=9,
        )
        scores, _ = pca(np.concatenate([pair["query_tokens"], pair["ref_tokens"]]), 3)
        lo, hi = np.percentile(scores, 2, axis=0), np.percentile(scores, 98, axis=0)
        colours = ((scores - lo) / (hi - lo + 1e-9)).clip(0, 1)
        n = pair["query_tokens"].shape[0]
        both = np.concatenate(
            [colours[:n].reshape(*grid, 3), colours[n:].reshape(*grid, 3)], axis=1
        )
        row[3].imshow(
            both, interpolation="nearest", aspect=aspect * (2 * grid[1]) / grid[0]
        )  # same shape as images
        row[3].axvline(grid[1] - 0.5, color="white", lw=2)
        row[3].set_title("patch PCA → RGB: query | reference", fontsize=9)
        for ax in row:
            ax.axis("off")
    fig.colorbar(
        plt.cm.ScalarMappable(norm, SIM_CMAP),
        ax=axes[:, 2],
        shrink=0.9,
        location="bottom",
        label="max patch cosine to the reference image (shared colour range)",
    )
    fig.suptitle(title, fontweight="bold")
    return fig


def plot_preprocessing_comparison(rows: Sequence[dict], title: str) -> plt.Figure:
    """Dot plots of measured separability for each resize mode (no truncated bars)."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), constrained_layout=True)
    x = np.arange(len(rows))
    labels = [f"{r['resize_mode']}\n({r['input']} px)" for r in rows]
    aucs = [r["auc"] for r in rows]
    axes[0].plot(x, aucs, "o", ms=9, color=COLOR_SAMPLED)
    for xi, auc in zip(x, aucs, strict=True):
        axes[0].annotate(
            f"{auc:.3f}", (xi, auc), xytext=(0, 8), textcoords="offset points", ha="center"
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_xlim(-0.5, len(rows) - 0.5)
    pad = 0.15 * (max(aucs) - min(aucs) + 1e-3)
    axes[0].set_ylim(min(aucs) - pad, max(aucs) + 2 * pad)
    axes[0].set_ylabel("ROC AUC (GT revisit vs different place)")
    axes[0].set_title("Separability (higher is better)")
    for xi, r in zip(x, rows, strict=True):
        axes[1].plot([xi, xi], [r["neg_median"], r["pos_median"]], color="#cbd5e0", lw=3)
    axes[1].plot(
        x,
        [r["pos_median"] for r in rows],
        "o",
        ms=8,
        color=COLOR_REVISIT,
        label="GT revisits: median cos",
    )
    axes[1].plot(
        x,
        [r["neg_p99"] for r in rows],
        "D",
        ms=6,
        color="black",
        label="different places: 99th pct cos",
    )
    axes[1].plot(
        x,
        [r["neg_median"] for r in rows],
        "s",
        ms=7,
        color=COLOR_TRAJECTORY,
        label="different places: median cos",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_xlim(-0.5, len(rows) - 0.5)
    axes[1].set_ylabel("cosine similarity")
    axes[1].set_title("Similarity statistics")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle(title, fontweight="bold")
    return fig


def render_similarity_evolution_frames(
    dataset_images,
    sim: np.ndarray,
    labels: np.ndarray,
    frame_ids: np.ndarray,
    timestamps: np.ndarray,
    steps: np.ndarray,
    min_gap_frames: int,
    title: str,
    width_px: int = 900,
    positive_radius_m: float = 5.0,
    negative_radius_m: float = 25.0,
) -> list[np.ndarray]:
    """GIF frames: camera frame, causal similarity matrix filling row by row, and the
    current query's similarity to every earlier frame with the temporal exclusion zone."""
    apply_style()
    dpi = 100
    first_image = dataset_images(0)
    img_aspect = first_image.shape[0] / first_image.shape[1]
    fig_w = width_px / dpi
    left_w = 0.52 * fig_w  # camera image and similarity profile share the left column
    img_h = left_w * img_aspect
    fig_h = img_h + 3.6
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    top = 1 - 0.45 / fig_h
    ax_img = fig.add_axes((0.04, top - img_h / fig_h, 0.52, img_h / fig_h))
    ax_line = fig.add_axes((0.09, 0.62 / fig_h, 0.47, (fig_h - img_h - 1.55) / fig_h))
    ax_mat = fig.add_axes((0.64, 0.62 / fig_h, 0.27, (fig_h - 1.07) / fig_h))
    cax = fig.add_axes((0.925, 0.62 / fig_h, 0.015, (fig_h - 1.07) / fig_h))
    head = fig.text(0.04, top + 0.12 / fig_h, "", fontweight="bold", fontsize=10.5)
    im = ax_img.imshow(first_image, cmap="gray" if first_image.ndim == 2 else None)
    ax_img.axis("off")

    n = sim.shape[0]
    vmin = float(np.percentile(sim, 1))
    masked = np.full_like(sim, np.nan)
    extent = (frame_ids[0], frame_ids[-1], frame_ids[-1], frame_ids[0])
    cmap = plt.get_cmap(SIM_CMAP).with_extremes(bad="#e2e8f0")
    mat = ax_mat.imshow(
        masked,
        cmap=cmap,
        vmin=vmin,
        vmax=1,
        extent=extent,
        interpolation="nearest",
        aspect="auto",
    )
    (cursor,) = ax_mat.plot([], [], color="#38bdf8", lw=1.2)
    ax_mat.set_xlabel("reference frame ID")
    ax_mat.set_ylabel("query frame ID")
    ax_mat.set_title("Causal similarity matrix", fontsize=9.5)
    ax_mat.grid(False)
    fig.colorbar(mat, cax=cax, format="%.2f").set_label("cosine similarity", fontsize=8)
    cax.tick_params(labelsize=7)

    (line,) = ax_line.plot([], [], color=COLOR_SAMPLED, lw=1.2, label="cos(query, earlier)")
    gt = ax_line.scatter(
        [],
        [],
        s=14,
        color=COLOR_REVISIT,
        zorder=3,
        label=f"GT revisit (≤ {positive_radius_m:g} m)",
    )
    (best,) = ax_line.plot([], [], "v", color="black", ms=7, label="best eligible")
    span = ax_line.axvspan(
        0, 0, color="#cbd5e0", alpha=0.6, label=f"excluded (< {min_gap_frames} frames)"
    )
    ax_line.set_xlim(frame_ids[0], frame_ids[-1])
    ax_line.set_ylim(vmin, 1.0 + 0.06 * (1.0 - vmin))
    ax_line.set_xlabel("earlier frame ID")
    ax_line.set_ylabel("cosine similarity")
    ax_line.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.85)
    note = ax_line.set_title("", fontsize=9, loc="left")

    frames = []
    for t in steps:
        image = dataset_images(int(t))
        im.set_data(image)
        masked[: t + 1, : t + 1] = sim[: t + 1, : t + 1]
        masked[np.triu_indices(n, 1)] = np.nan  # causal: only earlier references
        mat.set_data(masked)
        cursor.set_data([frame_ids[0], frame_ids[t]], [frame_ids[t], frame_ids[t]])
        ids_before = frame_ids[: t + 1]
        line.set_data(ids_before, sim[t, : t + 1])
        pos = np.flatnonzero(labels[t, : t + 1] == POSITIVE)
        gt.set_offsets(
            np.column_stack([frame_ids[pos], sim[t, pos]]) if pos.size else np.empty((0, 2))
        )
        lo = max(frame_ids[0], frame_ids[t] - min_gap_frames)
        span.set_x(lo)  # axvspan -> Rectangle in data-x / axes-y coordinates
        span.set_width(frame_ids[t] - lo)
        eligible = np.flatnonzero(frame_ids[: t + 1] <= frame_ids[t] - min_gap_frames)
        if eligible.size:
            j = eligible[np.argmax(sim[t, eligible])]
            best.set_data([frame_ids[j]], [sim[t, j]])
            verdict = {
                POSITIVE: "GT revisit",
                NEGATIVE: f"different place (> {negative_radius_m:g} m)",
                IGNORED: f"ambiguous ({positive_radius_m:g}-{negative_radius_m:g} m)",
            }[int(labels[t, j])]
            note.set_text(f"best eligible: #{frame_ids[j]}, cos {sim[t, j]:.3f} → {verdict}")
        else:
            best.set_data([], [])
            note.set_text(f"no eligible frame yet (needs ≥ {min_gap_frames} frames gap)")
        head.set_text(f"{title}: query #{frame_ids[t]}, t = {timestamps[t]:.1f} s")
        frames.append(figure_to_array(fig))
    plt.close(fig)
    return frames
