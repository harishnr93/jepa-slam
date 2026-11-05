"""Phase 4 figures and GIF frames: ORB keypoints, matches, RANSAC inliers, decisions.

Image pairs are drawn as a *stacked canvas* (query on top, candidate below) so match lines
stay short on wide KITTI frames. Lines are drawn with OpenCV (anti-aliased) and the
canvas is placed in Matplotlib for titles and legends.

Colours: tentative matches yellow, RANSAC inliers green, outliers red. Ground-truth
labels (true revisit / not a revisit) only annotate the figures; they never affect the
verification decisions, which come from the saved Phase 4 results.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from latent_loop.datasets.ground_truth import NEGATIVE, POSITIVE
from latent_loop.verification.geometry import VerificationResult
from latent_loop.verification.orb import Matches, OrbFeatures
from latent_loop.visualization.plots import (
    apply_style,
    figure_to_array,
    plt,
)
from latent_loop.visualization.trajectory import format_xz_axes

RGB_TENTATIVE = (255, 196, 0)
RGB_INLIER = (40, 220, 90)
RGB_OUTLIER = (235, 50, 50)
HEX_INLIER = "#28dc5a"
HEX_OUTLIER = "#eb3232"
COLOR_ACCEPT = "#2f855a"
COLOR_REJECT = "#c53030"
COLOR_MISSED = "#dd6b20"
GAP_PX = 6


@dataclass
class PairExample:
    """Everything needed to draw one verified pair (details recomputed from images)."""

    query_frame_id: int
    candidate_frame_id: int
    image_q: np.ndarray
    image_r: np.ndarray
    features_q: OrbFeatures
    features_r: OrbFeatures
    result: VerificationResult  # with keep_details=True
    raw: Matches  # unfiltered nearest-neighbour matches
    cosine: float
    rank: int
    gt_label: int | None = None
    gt_distance_m: float | None = None
    case: str = ""
    ratio: float = 0.8  # matching thresholds, for captions
    max_hamming: int | None = 64

    @property
    def verdict(self) -> str:
        r = self.result
        if r.accepted:
            extra = f", {r.degenerate}: pose unreliable" if r.degenerate else ""
            return f"ACCEPTED ({r.n_inliers} inliers, ratio {r.inlier_ratio:.2f}{extra})"
        return f"REJECTED: {r.reason.replace('_', ' ')}"

    @property
    def truth(self) -> str:
        if self.gt_label is None:
            return ""
        name = {POSITIVE: "true revisit", NEGATIVE: "not a revisit"}.get(
            self.gt_label, "ambiguous distance"
        )
        return f"ground truth: {name} ({self.gt_distance_m:.1f} m apart)"


# ----------------------------------------------------------------------------- canvas


def _rgb(image: np.ndarray) -> np.ndarray:
    return np.dstack([image] * 3) if image.ndim == 2 else image.copy()


def stack_pair(image_q: np.ndarray, image_r: np.ndarray, gap: int = GAP_PX) -> np.ndarray:
    a, b = _rgb(image_q), _rgb(image_r)
    w = max(a.shape[1], b.shape[1])
    pad = lambda im: np.pad(im, ((0, 0), (0, w - im.shape[1]), (0, 0)), constant_values=255)  # noqa: E731
    return np.vstack([pad(a), np.full((gap, w, 3), 255, np.uint8), pad(b)])


def draw_matches(
    canvas: np.ndarray,
    uv_q: np.ndarray,
    uv_r: np.ndarray,
    y_offset: int,
    colour: tuple[int, int, int],
    thickness: int = 1,
    alpha: float = 0.85,
    radius: int = 3,
) -> np.ndarray:
    """Draw match lines and endpoints (query top, candidate shifted by ``y_offset``)."""
    over = canvas.copy()
    for (x1, y1), (x2, y2) in zip(uv_q, uv_r, strict=True):
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2 + y_offset)))
        cv2.line(over, p1, p2, colour, thickness, cv2.LINE_AA)
        cv2.circle(over, p1, radius, colour, -1, cv2.LINE_AA)
        cv2.circle(over, p2, radius, colour, -1, cv2.LINE_AA)
    return cv2.addWeighted(over, alpha, canvas, 1 - alpha, 0)


def _subsample(n: int, k: int, seed: int = 0) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def tentative_canvas(ex: PairExample, max_lines: int = 300) -> np.ndarray:
    canvas = stack_pair(ex.image_q, ex.image_r)
    m = ex.result.matches
    if m is None or len(m) == 0:
        return canvas
    sel = _subsample(len(m), max_lines)
    off = ex.image_q.shape[0] + GAP_PX
    uq = ex.features_q.xy[m.query_idx[sel]]
    ur = ex.features_r.xy[m.ref_idx[sel]]
    return draw_matches(canvas, uq, ur, off, RGB_TENTATIVE)


def inlier_canvas(ex: PairExample, max_lines: int = 300) -> np.ndarray:
    canvas = stack_pair(ex.image_q, ex.image_r)
    m, mask = ex.result.matches, ex.result.inlier_mask
    if m is None or len(m) == 0 or mask is None:
        return canvas
    off = ex.image_q.shape[0] + GAP_PX
    uq, ur = ex.features_q.xy[m.query_idx], ex.features_r.xy[m.ref_idx]
    out = np.flatnonzero(~mask)
    inl = np.flatnonzero(mask)
    out = out[_subsample(out.size, max_lines // 2, 1)]
    inl = inl[_subsample(inl.size, max_lines, 2)]
    canvas = draw_matches(canvas, uq[out], ur[out], off, RGB_OUTLIER, 1, 0.7, 2)
    return draw_matches(canvas, uq[inl], ur[inl], off, RGB_INLIER, 1, 0.9, 3)


def _show(ax: plt.Axes, image: np.ndarray, title: str = "", colour: str = "black") -> None:
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None, interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=9, loc="left", color=colour)


def _frame_border(ax: plt.Axes, colour: str, width: float = 3.0) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(colour)
        spine.set_linewidth(width)


# ----------------------------------------------------------------------------- figures


def plot_orb_keypoints(ex: PairExample, title: str, n_circles: int = 120) -> plt.Figure:
    """ORB keypoints of both frames.

    Every keypoint is a dot coloured by pyramid level; the ``n_circles`` strongest also get
    a circle of the keypoint's support diameter (31 px x 1.2^level) and an orientation tick.
    """
    apply_style()
    h, w = ex.image_q.shape[:2]
    fig_w = 12.0
    img_h = (fig_w - 1.0) * h / w
    fig_h = 2 * img_h + 1.3
    fig = plt.figure(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap("plasma")
    axes = [
        fig.add_axes((0.005, (0.08 + img_h + 0.45) / fig_h, 0.9, img_h / fig_h)),
        fig.add_axes((0.005, 0.08 / fig_h, 0.9, img_h / fig_h)),
    ]
    n_levels = 8
    palette = [tuple(int(255 * c) for c in cmap(k / (n_levels - 1))[:3]) for k in range(n_levels)]
    for ax, img, f, name, fid in (
        (axes[0], ex.image_q, ex.features_q, "query", ex.query_frame_id),
        (axes[1], ex.image_r, ex.features_r, "candidate", ex.candidate_frame_id),
    ):
        canvas = _rgb(img)
        colours = [palette[min(int(o), n_levels - 1)] for o in f.octave]
        for i in range(len(f)):
            cv2.circle(canvas, (int(f.xy[i, 0]), int(f.xy[i, 1])), 2, colours[i], -1, cv2.LINE_AA)
        for i in np.argsort(-f.response)[:n_circles]:
            x, y = f.xy[i]
            r = max(int(round(f.size[i] / 2)), 2)
            cv2.circle(canvas, (int(x), int(y)), r, colours[i], 1, cv2.LINE_AA)
            a = np.radians(f.angle[i])
            tip = (int(x + r * np.cos(a)), int(y + r * np.sin(a)))
            cv2.line(canvas, (int(x), int(y)), tip, colours[i], 2, cv2.LINE_AA)
        _show(
            ax,
            canvas,
            f"{name} frame #{fid}: {len(f)} ORB keypoints (dots); circle = support region and "
            f"tick = orientation for the {min(n_circles, len(f))} strongest",
        )
    cax = fig.add_axes((0.93, 0.1, 0.014, 0.72))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_levels - 1))
    fig.colorbar(sm, cax=cax).set_label("pyramid level (image scaled by 1.2^-level)")
    fig.suptitle(title, fontsize=10.5, fontweight="bold", x=0.005, ha="left", y=0.995)
    return fig


def plot_raw_vs_filtered(ex: PairExample, title: str, max_lines: int = 300) -> plt.Figure:
    """Raw nearest-neighbour matches next to the ratio-tested, mutual matches."""
    apply_style()
    h, w = ex.image_q.shape[:2]
    fig_h = 7.4 * (2 * h + GAP_PX) / w + 1.1
    fig, axes = plt.subplots(1, 2, figsize=(15, fig_h))
    fig.subplots_adjust(
        left=0.005, right=0.995, wspace=0.02, top=1 - 0.75 / fig_h, bottom=0.4 / fig_h
    )
    off = h + GAP_PX
    raw = ex.raw
    sel = _subsample(len(raw), max_lines, 3)
    canvas = stack_pair(ex.image_q, ex.image_r)
    canvas = draw_matches(
        canvas,
        ex.features_q.xy[raw.query_idx[sel]],
        ex.features_r.xy[raw.ref_idx[sel]],
        off,
        RGB_TENTATIVE,
        1,
        0.75,
        2,
    )
    _show(
        axes[0],
        canvas,
        f"(a) raw nearest neighbours by Hamming distance: {len(raw)} ({len(sel)} drawn)",
    )
    st = ex.result.matches.stats if ex.result.matches is not None else {}
    _show(
        axes[1],
        tentative_canvas(ex, max_lines),
        f"(b) ratio test d1 < {ex.ratio:g} d2: {st.get('ratio', 0)}, then Hamming <= "
        f"{ex.max_hamming}: {st.get('max_hamming', 0)}, then mutual: {ex.result.n_matches}",
    )
    fig.suptitle(title, fontsize=10.5, fontweight="bold", x=0.01, ha="left")
    fig.text(
        0.01,
        0.01,
        "Query frame on top, candidate frame below. Each line joins a query keypoint to its "
        "matched candidate keypoint.",
        fontsize=8.5,
        color="#4a5568",
    )
    return fig


def epipolar_lines(E: np.ndarray, K: np.ndarray, uv_q: np.ndarray) -> np.ndarray:
    """Lines ``l = F u_q`` in the candidate image, ``F = K^{-T} E K^{-1}`` (rows a, b, c)."""
    Kinv = np.linalg.inv(K)
    Fm = Kinv.T @ E @ Kinv
    lines = np.c_[uv_q, np.ones(len(uv_q))] @ Fm.T
    return lines / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)


def plot_inliers_outliers(
    ex: PairExample, K: np.ndarray, title: str, n_epipolar: int = 10
) -> plt.Figure:
    """RANSAC inliers/outliers plus epipolar lines of a few inliers in the candidate image."""
    apply_style()
    r = ex.result
    h, w = ex.image_q.shape[:2]
    stacked_h = 13 * (2 * h + GAP_PX) / w
    single_h = 6.4 * h / w
    fig = plt.figure(figsize=(13, stacked_h + single_h + 1.6))
    fig_h = stacked_h + single_h + 1.6
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[stacked_h, single_h],
        hspace=0.5 / ((stacked_h + single_h) / 2),
        wspace=0.03,
        left=0.01,
        right=0.99,
        top=1 - 0.75 / fig_h,
        bottom=0.02,
    )
    ax_top = fig.add_subplot(gs[0, :])
    _show(
        ax_top,
        inlier_canvas(ex),
        f"(a) {r.n_inliers} RANSAC inliers (green) and {r.n_matches - r.n_inliers} outliers "
        f"(red) of {r.n_matches} matches; inlier ratio {r.inlier_ratio:.2f}",
    )
    ax_q, ax_r = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    if r.E is not None and r.inlier_mask is not None and r.inlier_mask.any():
        m = r.matches
        inl = np.flatnonzero(r.inlier_mask)
        pick = inl[np.linspace(0, inl.size - 1, min(n_epipolar, inl.size)).round().astype(int)]
        uq = ex.features_q.xy[m.query_idx[pick]].astype(float)
        ur = ex.features_r.xy[m.ref_idx[pick]].astype(float)
        lines = epipolar_lines(r.E, K, uq)
        cq, cr = _rgb(ex.image_q), _rgb(ex.image_r)
        cmap = plt.get_cmap("tab10")
        dists = []
        for j, ((a, b, c), p, q) in enumerate(zip(lines, uq, ur, strict=True)):
            colour = tuple(int(255 * v) for v in cmap(j % 10)[:3])
            cv2.circle(cq, (int(p[0]), int(p[1])), 6, colour, 2, cv2.LINE_AA)
            if abs(b) > 1e-9:
                y0, y1 = -c / b, -(a * w + c) / b
                cv2.line(cr, (0, int(y0)), (w, int(y1)), colour, 2, cv2.LINE_AA)
            cv2.circle(cr, (int(q[0]), int(q[1])), 6, colour, 2, cv2.LINE_AA)
            dists.append(abs(a * q[0] + b * q[1] + c))
        _show(ax_q, cq, f"(b) {len(pick)} inliers in the query frame #{ex.query_frame_id}")
        _show(
            ax_r,
            cr,
            f"(c) their epipolar lines l = F u_q in the candidate #{ex.candidate_frame_id}; "
            f"median point-line distance {np.median(dists):.2f} px",
        )
    else:
        _show(ax_q, ex.image_q, "(b) no essential matrix: no epipolar lines")
        _show(ax_r, ex.image_r, "")
    fig.suptitle(title, fontsize=10.5, fontweight="bold", x=0.01, ha="left")
    return fig


def plot_decision_gallery(examples: Sequence[PairExample], title: str) -> plt.Figure:
    """Grid of verified pairs with inliers drawn, decision, reason and ground truth."""
    apply_style()
    n = len(examples)
    cols = 2
    rows = int(np.ceil(n / cols))
    h, w = examples[0].image_q.shape[:2]
    cell_h = 7.2 * (2 * h + GAP_PX) / w + 0.85
    fig, axes = plt.subplots(rows, cols, figsize=(14.6, rows * cell_h + 0.6), squeeze=False)
    for ax, ex in zip(axes.ravel(), examples, strict=False):
        colour = COLOR_ACCEPT if ex.result.accepted else COLOR_REJECT
        _show(
            ax,
            inlier_canvas(ex, 200),
            f"{ex.case}\nquery #{ex.query_frame_id} vs candidate #{ex.candidate_frame_id} "
            f"(rank {ex.rank}, cos {ex.cosine:.3f})\n{ex.verdict}\n{ex.truth}",
            colour,
        )
        _frame_border(ax, colour)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_verification_statistics(
    pairs: pd.DataFrame, min_inliers: int, min_inlier_ratio: float, title: str
) -> plt.Figure:
    """Inlier statistics and rejection reasons, split by ground-truth label."""
    apply_style()
    pos = pairs[pairs.gt_label == POSITIVE]
    neg = pairs[pairs.gt_label == NEGATIVE]
    pre = (
        "invalid_calibration",
        "missing_descriptors",
        "repeated_structure",
        "insufficient_matches",
        "ransac_failure",
    )
    ran = ~pairs.reason.isin(pre)
    pos_r, neg_r = pos[ran[pos.index]], neg[ran[neg.index]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    ax = axes[0]
    top = max(int(pairs.n_inliers.max()), 10)
    bins = np.geomspace(4.5, top * 1.1, 32)
    for df, df_all, colour, name in (
        (neg_r, neg, COLOR_REJECT, "not a revisit (> 25 m)"),
        (pos_r, pos, COLOR_ACCEPT, "true revisit (<= 5 m)"),
    ):
        ax.hist(
            df.n_inliers,
            bins=bins,
            alpha=0.6,
            color=colour,
            label=f"{name}: {len(df)} (+{len(df_all) - len(df)} rejected before RANSAC)",
        )
    ax.axvline(min_inliers, color="black", ls="--", lw=1.2, label=f"min_inliers = {min_inliers}")
    ax.set_xscale("log")
    ax.set_xlabel("RANSAC inliers (log scale)")
    ax.set_ylabel("candidate pairs")
    ax.set_title("(a) Inlier count of pairs that reached RANSAC")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    for df, colour, name in (
        (neg_r, COLOR_REJECT, "not a revisit"),
        (pos_r, COLOR_ACCEPT, "true revisit"),
    ):
        ax.scatter(df.n_inliers, df.inlier_ratio, s=8, alpha=0.5, color=colour, label=name)
    ax.axvline(min_inliers, color="black", ls="--", lw=1, label=f"min_inliers = {min_inliers}")
    ax.axhline(
        min_inlier_ratio,
        color="black",
        ls=":",
        lw=1,
        label=f"min_inlier_ratio = {min_inlier_ratio:g}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("RANSAC inliers (log scale)")
    ax.set_ylabel("inlier ratio (inliers / matches)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(b) Accepted region: upper right of both lines")
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[2]
    reasons = [r for r in pairs.reason.value_counts().index]
    y = np.arange(len(reasons))
    cp = pos.reason.value_counts().reindex(reasons, fill_value=0)
    cn = neg.reason.value_counts().reindex(reasons, fill_value=0)
    ax.barh(y - 0.2, cp.values, 0.4, color=COLOR_ACCEPT, label="true revisit")
    ax.barh(y + 0.2, cn.values, 0.4, color=COLOR_REJECT, label="not a revisit")
    for yy, a, b in zip(y, cp.values, cn.values, strict=True):
        ax.text(a, yy - 0.2, f" {a}", va="center", fontsize=8)
        ax.text(b, yy + 0.2, f" {b}", va="center", fontsize=8)
    ax.set_yticks(y, [r.replace("_", " ") for r in reasons])
    ax.invert_yaxis()
    ax.set_xlabel("candidate pairs")
    ax.set_title("(c) Decision / rejection reason")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_verified_trajectory(
    pairs: pd.DataFrame, positions: np.ndarray, frame_ids: np.ndarray, title: str
) -> plt.Figure:
    """Top-1 candidates after verification on the trajectory.

    Links are drawn for accepted pairs (coloured by ground truth) and for rejected true
    revisits; queries whose top-1 candidate was rejected are grey dots, so the hundreds of
    correctly rejected long-range candidates do not hide the loop closures.
    """
    apply_style()
    index = {int(f): i for i, f in enumerate(frame_ids)}
    top1 = pairs[pairs["rank"] == 1]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.plot(positions[:, 0], positions[:, 2], color="#cbd5e0", lw=1.6, zorder=1)
    rejected = top1[~top1.accepted]
    rq = np.array([index[int(q)] for q in rejected.query_frame_id], dtype=int)
    if rq.size:
        ax.scatter(positions[rq, 0], positions[rq, 2], s=6, color="#a0aec0", zorder=2)
    n_neg = int(((~top1.accepted) & (top1.gt_label == NEGATIVE)).sum())
    ax.scatter(
        [],
        [],
        s=12,
        color="#a0aec0",
        label=(f"query whose top-1 was rejected: {len(rejected)} ({n_neg} of them not a revisit)"),
    )
    groups = (
        (
            "rejected true revisit (missed)",
            (~top1.accepted) & (top1.gt_label == POSITIVE),
            COLOR_MISSED,
            1.4,
            3,
        ),
        (
            "accepted, 5-25 m apart (unlabelled)",
            top1.accepted & (top1.gt_label == -1),
            "#2b6cb0",
            1.4,
            4,
        ),
        (
            "accepted true revisit (<= 5 m)",
            top1.accepted & (top1.gt_label == POSITIVE),
            COLOR_ACCEPT,
            1.8,
            5,
        ),
        (
            "accepted, not a revisit (> 25 m)",
            top1.accepted & (top1.gt_label == NEGATIVE),
            COLOR_REJECT,
            2.4,
            6,
        ),
    )
    for name, mask, colour, lw, z in groups:
        sub = top1[mask]
        for q, c in zip(sub.query_frame_id, sub.candidate_frame_id, strict=True):
            qi, ci = index[int(q)], index[int(c)]
            ax.plot(positions[[qi, ci], 0], positions[[qi, ci], 2], color=colour, lw=lw, zorder=z)
            ax.scatter(positions[qi, 0], positions[qi, 2], s=10, color=colour, zorder=z)
        ax.plot([], [], color=colour, lw=2.5, label=f"{name}: {len(sub)}")
    format_xz_axes(ax)
    ax.legend(loc="best", fontsize=8, framealpha=0.92)
    ax.set_title(title, fontsize=10.5)
    return fig


# ----------------------------------------------------------------------------- GIF


def render_verification_frames(
    examples: Sequence[PairExample],
    positions: np.ndarray,
    frame_ids: np.ndarray,
    title: str,
    width_px: int = 900,
    hold: tuple[int, int, int] = (3, 3, 5),
) -> list[np.ndarray]:
    """Per pair: tentative matches -> RANSAC inliers/outliers -> decision (+ trajectory).

    ``hold`` repeats each stage's frame so the GIF keeps a constant frame rate.
    """
    apply_style()
    index = {int(f): i for i, f in enumerate(frame_ids)}
    h, w = examples[0].image_q.shape[:2]
    dpi = 100
    fig_w = width_px / dpi
    img_frac = 0.66
    img_h = fig_w * img_frac * (2 * h + GAP_PX) / w
    fig_h = img_h + 1.72
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    head = fig.text(0.015, 1 - 0.28 / fig_h, "", fontsize=10.5, fontweight="bold")
    stage = fig.text(0.015, 1 - 0.58 / fig_h, "", fontsize=9.5)
    ax_img = fig.add_axes((0.01, 0.95 / fig_h, img_frac, img_h / fig_h))
    map_left = img_frac + 0.09
    ax_map = fig.add_axes((map_left, 0.95 / fig_h + 0.06, 0.985 - map_left, img_h / fig_h - 0.06))
    foot = fig.text(0.015, 0.5 / fig_h, "", fontsize=9.5, fontweight="bold")
    foot2 = fig.text(0.015, 0.16 / fig_h, "", fontsize=8.5, color="#4a5568")
    im = ax_img.imshow(stack_pair(examples[0].image_q, examples[0].image_r))
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.grid(False)
    ax_map.plot(positions[:, 0], positions[:, 2], color="#cbd5e0", lw=1.2)
    acc_sc = ax_map.scatter([], [], s=10, color=COLOR_ACCEPT, zorder=3, label="accepted so far")
    rej_sc = ax_map.scatter(
        [], [], s=10, color=COLOR_REJECT, marker="x", zorder=3, label="rejected so far"
    )
    (link,) = ax_map.plot([], [], lw=2, zorder=4)
    (qpt,) = ax_map.plot([], [], "o", color="black", ms=6, zorder=5, label="query")
    (cpt,) = ax_map.plot([], [], "s", color="#2b6cb0", ms=6, zorder=5, label="candidate")
    format_xz_axes(ax_map)
    ax_map.tick_params(labelsize=7)
    ax_map.set_xlabel("x [m]", fontsize=8)
    ax_map.set_ylabel("z [m]", fontsize=8)
    ax_map.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    ax_map.set_title("Trajectory (ground truth, labels only)", fontsize=8)
    acc_pts: list[np.ndarray] = []
    rej_pts: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    for n, ex in enumerate(examples):
        r = ex.result
        qi, ci = index[ex.query_frame_id], index[ex.candidate_frame_id]
        head.set_text(
            f"{title}: pair {n + 1}/{len(examples)}  query #{ex.query_frame_id} (top) vs "
            f"candidate #{ex.candidate_frame_id} (bottom), cos {ex.cosine:.3f}"
        )
        qpt.set_data([positions[qi, 0]], [positions[qi, 2]])
        cpt.set_data([positions[ci, 0]], [positions[ci, 2]])
        link.set_data(positions[[qi, ci], 0], positions[[qi, ci], 2])
        link.set_color("#718096")
        for s, count in enumerate(hold):
            if s == 0:
                im.set_data(tentative_canvas(ex))
                stage.set_text(
                    f"1/3 ORB matching: {r.n_keypoints_query} / {r.n_keypoints_ref} keypoints -> "
                    f"{r.n_matches} ratio-tested mutual matches (yellow)"
                )
                stage.set_color("black")
                foot.set_text("")
                foot2.set_text("")
                _frame_border(ax_img, "#a0aec0", 1)
            elif s == 1:
                im.set_data(inlier_canvas(ex))
                txt = (
                    f"2/3 essential-matrix RANSAC: {r.n_inliers} inliers (green), "
                    f"{max(r.n_matches - r.n_inliers, 0)} outliers (red)"
                    if r.n_matches and r.reason not in ("insufficient_matches", "ransac_failure")
                    else "2/3 essential-matrix RANSAC: not run (" + r.reason.replace("_", " ") + ")"
                )
                stage.set_text(txt)
            else:
                colour = COLOR_ACCEPT if r.accepted else COLOR_REJECT
                stage.set_text("3/3 decision")
                foot.set_text(ex.verdict)
                foot.set_color(colour)
                foot2.set_text(ex.truth)
                _frame_border(ax_img, colour, 4)
                link.set_color(colour)
                pt = positions[qi, [0, 2]]
                (acc_pts if r.accepted else rej_pts).append(pt)
                acc_sc.set_offsets(np.array(acc_pts) if acc_pts else np.empty((0, 2)))
                rej_sc.set_offsets(np.array(rej_pts) if rej_pts else np.empty((0, 2)))
            frame = figure_to_array(fig)
            frames.extend([frame] * count)
    plt.close(fig)
    return frames
