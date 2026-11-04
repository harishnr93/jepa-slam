"""Phase 3 figures and GIF frames built from saved retrieval results.

Correctness colours use the ground-truth revisit labels (labelling only): a candidate is
a *true revisit* if it is in the query's positive set, otherwise *not a revisit*; the
camera-centre distance to the query is printed next to each candidate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from latent_loop.datasets.ground_truth import RevisitGroundTruth
from latent_loop.evaluation.metrics import recall_at_k
from latent_loop.retrieval.detector import RetrievalResult
from latent_loop.visualization.plots import (
    COLOR_END,
    COLOR_TRAJECTORY,
    apply_style,
    figure_to_array,
    plt,
)
from latent_loop.visualization.trajectory import format_xz_axes

COLOR_TRUE = "#2f855a"
COLOR_FALSE = "#c53030"
SIM_CMAP = "magma"


def _verdict_colour(is_true: bool) -> str:
    return COLOR_TRUE if is_true else COLOR_FALSE


def choose_example_queries(
    result: RetrievalResult, gt: RevisitGroundTruth
) -> list[tuple[int, str]]:
    """Pick (row index, label) for: best true top-1, a missed revisit, a confident false
    candidate on a query that has no revisit. Deterministic."""
    top1 = result.top1_ids
    scores = np.nan_to_num(result.top1_scores, nan=-np.inf)
    correct = np.array([top1[i] in gt.positives[i] for i in range(top1.size)])
    picks: list[tuple[int, str]] = []
    hit = np.flatnonzero(gt.has_positive & correct)
    if hit.size:
        picks.append((int(hit[np.argmax(scores[hit])]), "true revisit at rank 1"))
    miss = np.flatnonzero(gt.has_positive & ~correct & (top1 >= 0))
    if miss.size:
        picks.append((int(miss[np.argmax(scores[miss])]), "revisit missed at rank 1"))
    none = np.flatnonzero(~gt.has_positive & (top1 >= 0))
    if none.size:
        picks.append((int(none[np.argmax(scores[none])]), "no revisit exists: false candidate"))
    return picks


def plot_topk_gallery(
    result: RetrievalResult,
    gt: RevisitGroundTruth,
    positions: np.ndarray,
    get_image: Callable[[int], np.ndarray],
    rows: Sequence[tuple[int, str]],
    title: str,
    k: int = 5,
) -> plt.Figure:
    """Query image followed by its top-K eligible references (score, distance, verdict)."""
    apply_style()
    index_of = {int(f): i for i, f in enumerate(result.query_frame_ids)}
    img0 = get_image(0)
    aspect = img0.shape[0] / img0.shape[1]
    fig, axes = plt.subplots(
        len(rows),
        k + 1,
        figsize=(2.9 * (k + 1), 2.9 * aspect * len(rows) + 0.4 * len(rows) + 0.8),
        squeeze=False,
        constrained_layout=True,
    )
    for row_axes, (qi, label) in zip(axes, rows, strict=True):
        qid = int(result.query_frame_ids[qi])
        row_axes[0].imshow(get_image(qi), cmap="gray" if img0.ndim == 2 else None)
        row_axes[0].set_title(f"QUERY #{qid}\n{label}", fontsize=8.5, color="black")
        for spine in row_axes[0].spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(3)
        for r, ax in enumerate(row_axes[1:]):
            cid = int(result.candidate_ids[qi, r])
            if cid < 0:
                ax.set_axis_off()
                continue
            ci = index_of[cid]
            is_true = cid in gt.positives[qi]
            dist = float(np.linalg.norm(positions[qi, [0, 2]] - positions[ci, [0, 2]]))
            ax.imshow(get_image(ci), cmap="gray" if img0.ndim == 2 else None)
            ax.set_title(
                f"#{r + 1}: frame {cid}  cos {result.scores[qi, r]:.3f}\n"
                f"{dist:.1f} m, {'TRUE revisit' if is_true else 'not a revisit'}",
                fontsize=8.5,
                color=_verdict_colour(is_true),
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(_verdict_colour(is_true))
                spine.set_linewidth(3)
        for ax in row_axes:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title, fontweight="bold")
    return fig


def plot_candidate_trajectory(
    result: RetrievalResult,
    gt: RevisitGroundTruth,
    positions: np.ndarray,
    title: str,
    strongest_fraction: float = 0.1,
) -> plt.Figure:
    """Top-1 loop candidates drawn on the ground-truth trajectory (top-down x-z).

    (a) every query that has a true revisit: link to its top-1 candidate;
    (b) the strongest ``strongest_fraction`` of all top-1 candidates by cosine, which is
        what a similarity threshold would accept, including queries without revisits.
    """
    apply_style()
    index_of = {int(f): i for i, f in enumerate(result.query_frame_ids)}
    top1 = result.top1_ids
    scores = result.top1_scores
    valid = top1 >= 0
    correct = np.array([top1[i] in gt.positives[i] for i in range(top1.size)])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), constrained_layout=True)
    thr = np.nanquantile(scores[valid], 1 - strongest_fraction) if valid.any() else np.inf
    panels = (
        (gt.has_positive & valid, "Queries with a true revisit: top-1 candidate"),
        (
            valid & (scores >= thr),
            f"Strongest {strongest_fraction:.0%} of top-1 candidates (cos ≥ {thr:.3f})",
        ),
    )
    for ax, (mask, subtitle) in zip(axes, panels, strict=True):
        ax.plot(positions[:, 0], positions[:, 2], color="#cbd5e0", lw=1.5, zorder=1)
        n_true = n_false = 0
        for i in np.flatnonzero(mask):
            j = index_of[int(top1[i])]
            colour = _verdict_colour(bool(correct[i]))
            n_true += int(correct[i])
            n_false += int(not correct[i])
            ax.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 2], positions[j, 2]],
                color=colour,
                lw=0.9,
                alpha=0.75,
                zorder=2,
            )
            ax.scatter(positions[i, 0], positions[i, 2], s=10, color=colour, zorder=3)
        ax.plot([], [], color=COLOR_TRUE, label=f"top-1 is a true revisit ({n_true})")
        ax.plot([], [], color=COLOR_FALSE, label=f"top-1 is not a revisit ({n_false})")
        ax.scatter(*positions[0, [0, 2]], marker="o", s=60, color="black", zorder=4, label="start")
        format_xz_axes(ax)
        ax.set_title(subtitle, fontsize=10)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(title, fontweight="bold")
    return fig


def plot_retrieval_matrix(
    sim: np.ndarray,
    result: RetrievalResult,
    gt: RevisitGroundTruth,
    title: str,
) -> plt.Figure:
    """Similarity restricted to *eligible* pairs, with GT revisits and top-1 choices."""
    apply_style()
    ids = result.query_frame_ids
    n = ids.size
    eligible = np.zeros((n, n), dtype=bool)
    for i, c in enumerate(result.eligible_counts):
        eligible[i, :c] = True
    masked = np.where(eligible, sim, np.nan)
    cmap = plt.get_cmap(SIM_CMAP).with_extremes(bad="#edf2f7")
    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    finite = masked[np.isfinite(masked)]
    vmin = float(np.percentile(finite, 1)) if finite.size else 0.0
    im = ax.imshow(
        masked,
        cmap=cmap,
        vmin=vmin,
        vmax=1.0,
        interpolation="nearest",
        extent=(ids[0], ids[-1], ids[-1], ids[0]),
    )
    q_pos, r_pos = [], []
    for i, pos in enumerate(gt.positives):
        q_pos.extend([ids[i]] * pos.size)
        r_pos.extend(pos.tolist())
    ax.scatter(
        r_pos, q_pos, s=2, c="#38bdf8", marker="s", linewidths=0, label="ground-truth revisit pairs"
    )
    top1 = result.top1_ids
    correct = np.array([top1[i] in gt.positives[i] for i in range(n)])
    has = top1 >= 0
    ax.scatter(
        top1[has & correct], ids[has & correct], s=7, color=COLOR_TRUE, label="top-1: true revisit"
    )
    ax.scatter(
        top1[has & ~correct],
        ids[has & ~correct],
        s=7,
        color=COLOR_FALSE,
        alpha=0.6,
        label="top-1: not a revisit",
    )
    ax.set_xlabel("reference frame ID")
    ax.set_ylabel("query frame ID")
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9, markerscale=2)
    fig.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity (eligible pairs only)")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    return fig


def plot_recall_curves(
    curves: Sequence[tuple[str, np.ndarray, RevisitGroundTruth]],
    title: str,
    max_k: int,
) -> plt.Figure:
    """Recall@K for K = 1..max_k (preliminary; Phase 6 adds the full protocol)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.3), constrained_layout=True)
    ks = list(range(1, max_k + 1))
    markers = ["o", "s", "D", "^"]
    for (label, candidates, gt), marker in zip(curves, markers, strict=False):
        r = recall_at_k(candidates, gt.positives, ks)
        values = [r[k] for k in ks]
        ax.plot(
            ks,
            values,
            marker=marker,
            ms=5,
            label=f"{label} ({int(gt.has_positive.sum())} positive queries)",
        )
        for k in (1, 5, 10):
            if k <= max_k:
                ax.annotate(
                    f"{r[k]:.3f}",
                    (k, r[k]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.5,
                )
    ax.set_xlabel("K (number of retrieved candidates)")
    ax.set_ylabel("Recall@K")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    return fig


def render_retrieval_frames(
    result: RetrievalResult,
    gt: RevisitGroundTruth,
    positions: np.ndarray,
    get_image: Callable[[int], np.ndarray],
    steps: np.ndarray,
    title: str,
    width_px: int = 900,
) -> list[np.ndarray]:
    """GIF frames of chronological (causal) retrieval: current frame, top-1 historical
    frame with score and verdict, trajectory with the candidate loop connection."""
    apply_style()
    index_of = {int(f): i for i, f in enumerate(result.query_frame_ids)}
    img0 = get_image(0)
    img_aspect = img0.shape[0] / img0.shape[1]
    dpi = 100
    fig_w = width_px / dpi
    img_w_frac = 0.5
    img_h = fig_w * img_w_frac * img_aspect
    title_h = 0.48  # room for a two-line caption above each image
    fig_h = max(2 * img_h + 2 * title_h + 0.75, 5.4)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    head = fig.text(0.02, 1 - 0.3 / fig_h, "", fontsize=10.5, fontweight="bold")
    q_top = 0.42 + title_h
    ax_q = fig.add_axes((0.02, 1 - (q_top + img_h) / fig_h, img_w_frac, img_h / fig_h))
    r_top = q_top + img_h + title_h + 0.1
    ax_r = fig.add_axes((0.02, 1 - (r_top + img_h) / fig_h, img_w_frac, img_h / fig_h))
    ax_map = fig.add_axes((0.62, 0.1, 0.36, 0.78))
    for ax in (ax_q, ax_r):
        ax.set_xticks([])
        ax.set_yticks([])
    im_q = ax_q.imshow(img0, cmap="gray" if img0.ndim == 2 else None)
    im_r = ax_r.imshow(np.full_like(img0, 230), cmap="gray" if img0.ndim == 2 else None)
    t_q = ax_q.set_title("", fontsize=9, loc="left")
    t_r = ax_r.set_title("", fontsize=9, loc="left")
    ax_map.plot(positions[:, 0], positions[:, 2], color="#e2e8f0", lw=1.5, zorder=1)
    (trail,) = ax_map.plot([], [], color=COLOR_TRAJECTORY, lw=1.6, zorder=2, label="driven so far")
    found_true = ax_map.scatter(
        [], [], s=9, color=COLOR_TRUE, zorder=3, label="query whose top-1 is a true revisit"
    )
    (link,) = ax_map.plot([], [], lw=2.2, zorder=4)
    (cur,) = ax_map.plot([], [], "o", color=COLOR_END, ms=8, zorder=5, label="current query")
    (ref,) = ax_map.plot([], [], "s", ms=7, zorder=5, color="black", label="top-1 candidate")
    format_xz_axes(ax_map)
    ax_map.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax_map.set_title("Ground-truth trajectory (labels only)", fontsize=9.5)
    correct = np.array([result.top1_ids[i] in gt.positives[i] for i in range(result.top1_ids.size)])
    frames = []
    for t in steps:
        t = int(t)
        qid = int(result.query_frame_ids[t])
        im_q.set_data(get_image(t))
        t_q.set_text(f"current frame #{qid}  (t = {result.query_timestamps[t]:.1f} s)")
        trail.set_data(positions[: t + 1, 0], positions[: t + 1, 2])
        cur.set_data([positions[t, 0]], [positions[t, 2]])
        done = np.flatnonzero(correct[: t + 1])
        found_true.set_offsets(positions[done][:, [0, 2]] if done.size else np.empty((0, 2)))
        cid = int(result.top1_ids[t])
        if cid >= 0:
            ci = index_of[cid]
            colour = _verdict_colour(bool(correct[t]))
            im_r.set_data(get_image(ci))
            dist = float(np.linalg.norm(positions[t, [0, 2]] - positions[ci, [0, 2]]))
            if correct[t]:
                verdict = "TRUE revisit"
            elif gt.has_positive[t]:
                verdict = "not a revisit (a true one exists)"
            else:
                verdict = "not a revisit (no revisit exists)"
            t_r.set_text(
                f"top-1 earlier frame #{cid}, cos {result.top1_scores[t]:.3f}\n"
                f"{dist:.1f} m from the query -> {verdict}"
            )
            t_r.set_color(colour)
            link.set_data([positions[t, 0], positions[ci, 0]], [positions[t, 2], positions[ci, 2]])
            link.set_color(colour)
            ref.set_data([positions[ci, 0]], [positions[ci, 2]])
            for spine in ax_r.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(3)
        else:
            im_r.set_data(np.full_like(img0, 230))
            t_r.set_text(
                f"no eligible earlier frame yet\n(exclusion: "
                f"{result.config.temporal_exclusion_frames} frames)"
            )
            t_r.set_color("black")
            link.set_data([], [])
            ref.set_data([], [])
            for spine in ax_r.spines.values():
                spine.set_edgecolor("#a0aec0")
        head.set_text(f"{title}: query {t + 1}/{result.query_frame_ids.size}")
        frames.append(figure_to_array(fig))
    plt.close(fig)
    return frames
