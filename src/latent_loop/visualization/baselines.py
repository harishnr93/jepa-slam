"""Phase 5 figures and GIF frames: I-JEPA vs DINOv2 vs ORB bag of words vs HSV histogram.

All methods are compared on the *same* frames, temporal exclusion, ground truth and
retrieval code; only the descriptor differs. Ground-truth labels annotate the figures.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from latent_loop.datasets.ground_truth import NEGATIVE, POSITIVE, RevisitGroundTruth
from latent_loop.evaluation.metrics import recall_at_k, roc_auc
from latent_loop.retrieval.detector import RetrievalResult
from latent_loop.visualization.plots import apply_style, figure_to_array, plt
from latent_loop.visualization.retrieval import COLOR_FALSE, COLOR_TRUE
from latent_loop.visualization.trajectory import format_xz_axes

METHODS = ("ijepa", "dinov2", "orb_bovw", "hsv")
LABELS = {
    "ijepa": "I-JEPA ViT-H/14",
    "dinov2": "DINOv2 ViT-B/14",
    "orb_bovw": "ORB bag of words",
    "hsv": "HSV histogram",
}
# Okabe-Ito colour-blind-safe palette (one fixed colour per method)
COLORS = {"ijepa": "#0072B2", "dinov2": "#E69F00", "orb_bovw": "#CC79A7", "hsv": "#009E73"}


@dataclass
class MethodRun:
    name: str
    embeddings: np.ndarray  # (N, D) L2-normalised, dataset order
    result: RetrievalResult
    card: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""

    @property
    def label(self) -> str:
        return LABELS.get(self.name, self.name)

    @property
    def color(self) -> str:
        return COLORS.get(self.name, "#4a5568")


def correctness(run: MethodRun, gt: RevisitGroundTruth) -> np.ndarray:
    top1 = run.result.top1_ids
    return np.array([top1[i] >= 0 and top1[i] in gt.positives[i] for i in range(top1.size)])


def choose_comparison_cases(
    runs: Sequence[MethodRun], gt: RevisitGroundTruth
) -> list[tuple[int, str]]:
    """Deterministic (query index, case) picks covering agreement and disagreement."""
    by = {r.name: r for r in runs}
    ok = {r.name: correctness(r, gt) for r in runs}
    pos = gt.has_positive
    ij = by.get("ijepa")
    score = ij.result.top1_scores if ij is not None else runs[0].result.top1_scores
    score = np.nan_to_num(score, nan=-np.inf)
    others = [n for n in ok if n != "ijepa"]
    all_ok = np.all([ok[n] for n in ok], axis=0)
    none_ok = ~np.any([ok[n] for n in ok], axis=0)
    picks: list[tuple[int, str]] = []

    def add(mask: np.ndarray, case: str) -> None:
        idx = np.flatnonzero(mask & ~np.isin(np.arange(mask.size), [p for p, _ in picks]))
        if idx.size:
            picks.append((int(idx[np.argmax(score[idx])]), case))

    add(pos & all_ok, "all four retrieve a true revisit")
    if "ijepa" in ok and others:
        add(
            pos & ok["ijepa"] & ~np.any([ok[n] for n in others], axis=0),
            "only I-JEPA retrieves the true revisit",
        )
        add(
            pos & ~ok["ijepa"] & np.all([ok[n] for n in others], axis=0),
            "every baseline is right, I-JEPA is not",
        )
        if "dinov2" in ok:
            add(pos & ok["ijepa"] & ~ok["dinov2"], "I-JEPA right, DINOv2 wrong")
            add(pos & ~ok["ijepa"] & ok["dinov2"], "DINOv2 right, I-JEPA wrong")
    add(pos & none_ok, "all four miss the true revisit")
    add(~pos & (runs[0].result.top1_ids >= 0), "no revisit exists (every candidate is false)")
    return picks


def _verdict(run: MethodRun, gt: RevisitGroundTruth, i: int, positions, index) -> tuple[str, str]:
    cid = int(run.result.top1_ids[i])
    if cid < 0:
        return "no eligible frame", "#a0aec0"
    d = float(np.linalg.norm(positions[i, [0, 2]] - positions[index[cid], [0, 2]]))
    ok = cid in gt.positives[i]
    return (
        f"#{cid}, cos {run.result.top1_scores[i]:.3f}, {d:.0f} m",
        COLOR_TRUE if ok else COLOR_FALSE,
    )


def _image_ax(ax, img, title, colour="black", border=None):
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=8.5, loc="left", color=colour)
    if border:
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_edgecolor(border)
            s.set_linewidth(3)


# ----------------------------------------------------------------------------- figures


def plot_recall_comparison(
    runs: Sequence[MethodRun],
    gts: Sequence[tuple[str, RevisitGroundTruth]],
    title: str,
    max_k: int = 10,
    y_min: float = 0.5,
) -> plt.Figure:
    """Recall@K curves per ground-truth definition (shared y axis from ``y_min``), plus
    Recall@1/5/10 bars from zero with the exact values printed."""
    apply_style()
    fig, axes = plt.subplots(1, len(gts) + 1, figsize=(5.4 * (len(gts) + 1), 4.8))
    ks = list(range(1, max_k + 1))
    for ax, (name, gt) in zip(axes, gts, strict=False):
        for r in runs:
            rec = recall_at_k(r.result.candidate_ids, gt.positives, ks)
            ax.plot(
                ks,
                [rec[k] for k in ks],
                "o-",
                ms=3.5,
                color=r.color,
                label=f"{r.label} (R@1 {rec[1]:.3f})",
            )
        ax.set_xlabel("K")
        ax.set_ylabel(f"Recall@K (positive queries; axis from {y_min:g})")
        ax.set_ylim(y_min, 1.01)
        ax.set_title(f"{name}\n{int(gt.has_positive.sum())} positive queries", fontsize=10)
        ax.set_xticks(ks)
        ax.legend(fontsize=7.5, loc="lower right")
    ax = axes[-1]
    sel = [k for k in (1, 5, 10) if k <= max_k]
    width = 0.8 / len(runs)
    gt0 = gts[0][1]
    for j, r in enumerate(runs):
        rec = recall_at_k(r.result.candidate_ids, gt0.positives, sel)
        x = np.arange(len(sel)) + (j - (len(runs) - 1) / 2) * width
        ax.bar(x, [rec[k] for k in sel], width, color=r.color, label=r.label)
        for xx, k in zip(x, sel, strict=True):
            ax.text(
                xx,
                rec[k] - 0.02,
                f"{rec[k]:.3f}",
                ha="center",
                va="top",
                fontsize=6.5,
                rotation=90,
                color="white",
                fontweight="bold",
            )
    ax.set_xticks(np.arange(len(sel)), [f"R@{k}" for k in sel])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall (axis from 0)")
    ax.set_title(f"{gts[0][0]}\nRecall@1 / 5 / 10", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def similarity_stats(run: MethodRun, labels: np.ndarray) -> dict[str, float]:
    sim = run.embeddings @ run.embeddings.T
    pos, neg = sim[labels == POSITIVE], sim[labels == NEGATIVE]
    return {
        "auc": roc_auc(pos, neg) if pos.size and neg.size else float("nan"),
        "pos_median": float(np.median(pos)) if pos.size else float("nan"),
        "neg_median": float(np.median(neg)) if neg.size else float("nan"),
        "neg_p99": float(np.percentile(neg, 99)) if neg.size else float("nan"),
    }


def plot_similarity_comparison(
    runs: Sequence[MethodRun], gt: RevisitGroundTruth, labels: np.ndarray, title: str
) -> plt.Figure:
    """Row 1: eligible cosine-similarity matrices. Row 2: revisit vs non-revisit cosines."""
    apply_style()
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(4.3 * n, 8.4), gridspec_kw={"height_ratios": [1.15, 1]})
    for j, r in enumerate(runs):
        ids = r.result.query_frame_ids
        sim = r.embeddings @ r.embeddings.T
        eligible = np.zeros_like(sim, dtype=bool)
        for i, c in enumerate(r.result.eligible_counts):
            eligible[i, :c] = True
        masked = np.where(eligible, sim, np.nan)
        finite = masked[np.isfinite(masked)]
        vmin, vmax = np.percentile(finite, [1, 99.9]) if finite.size else (0, 1)
        ax = axes[0, j]
        im = ax.imshow(
            masked,
            cmap=plt.get_cmap("magma").with_extremes(bad="#edf2f7"),
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=(ids[0], ids[-1], ids[-1], ids[0]),
        )
        ax.grid(False)
        ax.set_title(r.label, fontsize=10, color=r.color)
        ax.set_xlabel("reference frame ID")
        if j == 0:
            ax.set_ylabel("query frame ID")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="cosine (own scale: 1-99.9 pct)")
        st = similarity_stats(r, labels)
        sim_pos, sim_neg = sim[labels == POSITIVE], sim[labels == NEGATIVE]
        ax = axes[1, j]
        lo = float(np.percentile(np.r_[sim_pos, sim_neg], 0.5))
        bins = np.linspace(lo, 1.0, 50)
        ax.hist(
            sim_neg,
            bins=bins,
            color=COLOR_FALSE,
            alpha=0.55,
            density=True,
            label="not a revisit (> 25 m)",
        )
        ax.hist(
            sim_pos,
            bins=bins,
            color=COLOR_TRUE,
            alpha=0.6,
            density=True,
            label="true revisit (<= 5 m)",
        )
        ax.set_title(f"ROC AUC {st['auc']:.3f}", fontsize=10)
        ax.set_xlabel("cosine similarity")
        if j == 0:
            ax.set_ylabel("density (eligible pairs)")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_side_by_side(
    runs: Sequence[MethodRun],
    gt: RevisitGroundTruth,
    positions: np.ndarray,
    get_image: Callable[[int], np.ndarray],
    queries: Sequence[tuple[int, str]],
    title: str,
    k: int = 3,
) -> plt.Figure:
    """For each query: one row per method with the query and its top-k candidates."""
    apply_style()
    ids = runs[0].result.query_frame_ids
    index = {int(f): i for i, f in enumerate(ids)}
    h, w = get_image(0).shape[:2]
    cell_w = 4.0
    cell_h = cell_w * h / w + 0.42
    rows = len(queries) * len(runs)
    fig, axes = plt.subplots(
        rows,
        k + 1,
        figsize=((k + 1) * cell_w, rows * cell_h + 0.5 * len(queries) + 0.4),
        squeeze=False,
    )
    for qn, (i, case) in enumerate(queries):
        for mn, r in enumerate(runs):
            row = qn * len(runs) + mn
            ax = axes[row, 0]
            _image_ax(
                ax,
                get_image(i),
                f"Query {qn + 1} (#{int(ids[i])}): {case}\n{r.label}" if mn == 0 else r.label,
                r.color,
                border="black",
            )
            for rank in range(k):
                ax = axes[row, rank + 1]
                cid = int(r.result.candidate_ids[i, rank])
                if cid < 0:
                    ax.axis("off")
                    continue
                d = float(np.linalg.norm(positions[i, [0, 2]] - positions[index[cid], [0, 2]]))
                ok = cid in gt.positives[i]
                colour = COLOR_TRUE if ok else COLOR_FALSE
                _image_ax(
                    ax,
                    get_image(index[cid]),
                    f"rank {rank + 1}: #{cid}, cos {r.result.scores[i, rank]:.3f}, {d:.0f} m",
                    colour,
                    border=colour,
                )
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig_h = fig.get_size_inches()[1]
    fig.subplots_adjust(
        left=0.005, right=0.995, top=1 - 0.75 / fig_h, bottom=0.01, wspace=0.03, hspace=0.42
    )
    return fig


def plot_qualitative(
    runs: Sequence[MethodRun],
    gt: RevisitGroundTruth,
    positions: np.ndarray,
    get_image: Callable[[int], np.ndarray],
    cases: Sequence[tuple[int, str]],
    title: str,
) -> plt.Figure:
    """One row per case: the query and each method's top-1 (green = true revisit)."""
    apply_style()
    ids = runs[0].result.query_frame_ids
    index = {int(f): i for i, f in enumerate(ids)}
    h, w = get_image(0).shape[:2]
    cell_w = 3.4
    cell_h = cell_w * h / w + 0.75
    fig, axes = plt.subplots(
        len(cases),
        len(runs) + 1,
        figsize=((len(runs) + 1) * cell_w, len(cases) * cell_h + 0.5),
        squeeze=False,
    )
    for row, (i, case) in enumerate(cases):
        _image_ax(axes[row, 0], get_image(i), f"{case}\nquery #{int(ids[i])}", border="black")
        for j, r in enumerate(runs):
            text, colour = _verdict(r, gt, i, positions, index)
            cid = int(r.result.top1_ids[i])
            img = get_image(index[cid]) if cid >= 0 else np.full_like(get_image(i), 230)
            _image_ax(axes[row, j + 1], img, f"{r.label}\n{text}", colour, border=colour)
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_size_inches()[1]))
    return fig


def plot_cost_comparison(
    runs: Sequence[MethodRun], recall1: dict[str, float], title: str
) -> plt.Figure:
    """Computational trade-offs from the encoder profile, and Recall@1 vs encoding time."""
    apply_style()
    have = [r for r in runs if r.card]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    names = [r.label for r in have]
    colors = [r.color for r in have]

    def bars(ax, values, xlabel, log=False, fmt="{:.3g}"):
        y = np.arange(len(have))
        vals = [v if v > 0 else np.nan for v in values]  # 0 = no model: nothing to draw
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        if log:
            ax.set_xscale("log")
        for yy, v in zip(y, values, strict=True):
            if v > 0:
                ax.text(v, yy, " " + fmt.format(v), va="center", fontsize=8)
            else:
                ax.text(np.nanmin(vals), yy, " none", va="center", fontsize=8, color="#4a5568")
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)

    bars(
        axes[0], [r.card["ms_per_image"] for r in have], "encoding time per image [ms] (log)", True
    )
    axes[0].set_title(
        "(a) Runtime: "
        + ", ".join(f"{r.label.split()[0]} {r.card['runtime']['device'].upper()}" for r in have),
        fontsize=8.5,
    )
    model_mb = [r.card.get("weights_file_bytes", 0) / 2**20 for r in have]
    bars(axes[1], model_mb, "weights / vocabulary file [MB] (log)", True)
    axes[1].set_title("(b) Model / vocabulary size", fontsize=10)
    bars(
        axes[2],
        [r.card["embedding_dim"] for r in have],
        "descriptor dimension (float32)",
        fmt="{:.0f}",
    )
    axes[2].set_title("(c) Descriptor size", fontsize=10)
    ax = axes[3]
    for r in have:
        ax.scatter(
            r.card["ms_per_image"], recall1[r.name], s=70, color=r.color, label=r.label, zorder=3
        )
        ax.annotate(
            f"  {recall1[r.name]:.3f}", (r.card["ms_per_image"], recall1[r.name]), fontsize=8
        )
    ax.set_xscale("log")
    ax.set_xlabel("encoding time per image [ms] (log)")
    ax.set_ylabel("Recall@1 (distance-only)")
    ax.set_title("(d) Accuracy vs cost", fontsize=10)
    ax.legend(fontsize=7.5, loc="center right")
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ----------------------------------------------------------------------------- GIF


def render_comparison_frames(
    runs: Sequence[MethodRun],
    gt: RevisitGroundTruth,
    positions: np.ndarray,
    get_image: Callable[[int], np.ndarray],
    steps: np.ndarray,
    title: str,
    width_px: int = 900,
) -> list[np.ndarray]:
    """Chronological queries: the query, each method's top-1 (2 x 2) and a trajectory map."""
    apply_style()
    ids = runs[0].result.query_frame_ids
    index = {int(f): i for i, f in enumerate(ids)}
    img0 = get_image(0)
    h, w = img0.shape[:2]
    dpi = 100
    fig_w = width_px / dpi
    q_w = 0.58
    q_h = fig_w * q_w * h / w
    c_w = 0.485
    c_h = fig_w * c_w * h / w
    cap = 0.36
    fig_h = 0.45 + cap + q_h + 2 * (cap + c_h) + 0.15
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    head = fig.text(0.01, 1 - 0.28 / fig_h, "", fontsize=10.5, fontweight="bold")
    top = 0.45 + cap
    ax_q = fig.add_axes((0.01, 1 - (top + q_h) / fig_h, q_w, q_h / fig_h))
    ax_map = fig.add_axes(
        (q_w + 0.1, 1 - (top + q_h - 0.12) / fig_h, 0.97 - q_w - 0.1, (q_h - 0.12) / fig_h)
    )
    axes, ims, caps = [], [], []
    for j in range(min(4, len(runs))):
        row, col = divmod(j, 2)
        y = top + q_h + cap + row * (cap + c_h)
        ax = fig.add_axes((0.01 + col * (c_w + 0.01), 1 - (y + c_h) / fig_h, c_w, c_h / fig_h))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ims.append(ax.imshow(np.full_like(img0, 230), cmap="gray" if img0.ndim == 2 else None))
        caps.append(ax.set_title("", fontsize=8.5, loc="left"))
        axes.append(ax)
    im_q = ax_q.imshow(img0, cmap="gray" if img0.ndim == 2 else None)
    ax_q.set_xticks([])
    ax_q.set_yticks([])
    ax_q.grid(False)
    t_q = ax_q.set_title("", fontsize=9, loc="left")
    ax_map.plot(positions[:, 0], positions[:, 2], color="#e2e8f0", lw=1.2)
    (trail,) = ax_map.plot([], [], color="#718096", lw=1.2)
    (qpt,) = ax_map.plot([], [], "o", color="black", ms=6, zorder=5)
    cpts = [
        ax_map.plot([], [], "s", ms=6, color=r.color, zorder=4, label=r.label.split()[0])[0]
        for r in runs[:4]
    ]
    format_xz_axes(ax_map)
    ax_map.tick_params(labelsize=6)
    ax_map.set_xlabel("x [m]", fontsize=7)
    ax_map.set_ylabel("z [m]", fontsize=7)
    ax_map.legend(fontsize=6, loc="upper left", framealpha=0.85, handletextpad=0.2)
    ax_map.set_title("top-1 positions (labels only)", fontsize=7.5)
    hits = {r.name: 0 for r in runs}
    seen_pos = 0
    ok = {r.name: correctness(r, gt) for r in runs}
    frames = []
    last = -1
    for t in steps:
        t = int(t)
        for tt in range(last + 1, t + 1):  # running Recall@1 over all positive queries so far
            if gt.has_positive[tt]:
                seen_pos += 1
                for r in runs:
                    hits[r.name] += int(ok[r.name][tt])
        last = t
        im_q.set_data(get_image(t))
        rev = "has a true revisit" if gt.has_positive[t] else "no revisit exists"
        t_q.set_text(f"query #{int(ids[t])} ({rev})")
        trail.set_data(positions[: t + 1, 0], positions[: t + 1, 2])
        qpt.set_data([positions[t, 0]], [positions[t, 2]])
        for j, r in enumerate(runs[:4]):
            text, colour = _verdict(r, gt, t, positions, index)
            cid = int(r.result.top1_ids[t])
            ims[j].set_data(get_image(index[cid]) if cid >= 0 else np.full_like(img0, 230))
            r1 = f" | R@1 so far {hits[r.name] / seen_pos:.2f}" if seen_pos else ""
            caps[j].set_text(f"{r.label.split()[0]}: {text}{r1}")
            caps[j].set_color(colour)
            for s in axes[j].spines.values():
                s.set_visible(True)
                s.set_edgecolor(colour)
                s.set_linewidth(3)
            if cid >= 0:
                cpts[j].set_data([positions[index[cid], 0]], [positions[index[cid], 2]])
            else:
                cpts[j].set_data([], [])
        head.set_text(f"{title}: query {t + 1}/{ids.size}")
        frames.append(figure_to_array(fig))
    plt.close(fig)
    return frames
