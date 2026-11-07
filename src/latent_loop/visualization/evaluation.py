"""Phase 6 figures and GIF frames from the saved benchmark outputs.

Every figure reads ``results/benchmark/<name>/``; detection results are shown at the
threshold selected on the *development* sequences (``thresholds.json``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from latent_loop.visualization.plots import apply_style, figure_to_array, plt
from latent_loop.visualization.trajectory import format_xz_axes

LABELS = {
    "ijepa": "I-JEPA",
    "ijepa_gv": "I-JEPA + GV",
    "dinov2": "DINOv2",
    "dinov2_gv": "DINOv2 + GV",
    "dinov3": "DINOv3",
    "dinov3_gv": "DINOv3 + GV",
    "orb_bovw": "ORB bag of words",
    "hsv": "HSV histogram",
}
COLORS = {  # Okabe-Ito; "+ GV" uses a related darker/lighter hue
    "ijepa": "#56B4E9",
    "ijepa_gv": "#0072B2",
    "dinov2": "#E69F00",
    "dinov2_gv": "#D55E00",
    "dinov3": "#999999",
    "dinov3_gv": "#000000",
    "orb_bovw": "#CC79A7",
    "hsv": "#009E73",
}
C_TP, C_FP, C_FN = "#2f855a", "#c53030", "#dd6b20"


def label(m: str) -> str:
    return LABELS.get(m, m)


def color(m: str) -> str:
    return COLORS.get(m, "#4a5568")


def _title(fig, text):
    fig.suptitle(text, fontsize=11, fontweight="bold", x=0.01, ha="left")


# ----------------------------------------------------------------------------- accuracy


def plot_pr_curves(
    curves: dict[str, dict[str, dict]],
    operating: dict[str, dict],
    methods: Sequence[str],
    title: str,
) -> plt.Figure:
    """Pooled detection PR curves on development and test data, with the test operating
    point at the development-selected threshold."""
    apply_style()
    splits = [s for s in ("dev", "test") if s in curves]
    fig, axes = plt.subplots(1, len(splits), figsize=(6.6 * len(splits), 5.6), squeeze=False)
    for ax, split in zip(axes[0], splits, strict=True):
        for m in methods:
            c = curves[split].get(m)
            if c is None:
                continue
            r, p = np.r_[0.0, c["recall"]], np.r_[1.0, c["precision"]]
            ax.step(
                r,
                p,
                where="post",
                color=color(m),
                lw=1.8,
                ls="--" if m.endswith("_gv") else "-",
                label=f"{label(m)} (AP {float(c['ap']):.3f})",
            )
            if split == "test" and m in operating:
                o = operating[m]
                ax.plot(
                    o["recall"], o["precision"], "o", color=color(m), ms=8, mec="black", mew=0.8
                )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("recall (correct loops / positive queries)")
        ax.set_ylabel("precision")
        name = "development (00, 05)" if split == "dev" else "held-out test (02, 06, 08, 09)"
        npos = next(iter(curves[split].values()))["positive_queries"]
        nq = next(iter(curves[split].values()))["queries"]
        ax.set_title(f"{name}: {int(nq)} queries, {int(npos)} with a revisit", fontsize=10)
        ax.legend(fontsize=7.5, loc="lower left")
    if "test" in splits:
        axes[0][-1].annotate(
            "dots: operating point at the threshold chosen on development data",
            (0.02, 0.02),
            xycoords="axes fraction",
            fontsize=7.5,
            color="#4a5568",
            va="bottom",
            ha="left",
            xytext=(0, 115),
            textcoords="offset points",
        )
    _title(fig, title)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_recall_bars(pooled: pd.DataFrame, methods: Sequence[str], title: str) -> plt.Figure:
    """Positive-query retrieval metrics (Recall@1/5/10, mAP) on the pooled test set."""
    apply_style()
    metrics = [
        ("recall_at_1", "R@1"),
        ("recall_at_5", "R@5"),
        ("recall_at_10", "R@10"),
        ("map", "mAP"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.8 / len(methods)
    for j, m in enumerate(methods):
        row = pooled.loc[m]
        x = np.arange(len(metrics)) + (j - (len(methods) - 1) / 2) * width
        vals = [float(row[k]) for k, _ in metrics]
        ax.bar(x, vals, width, color=color(m), label=label(m))
        for xx, v in zip(x, vals, strict=True):
            ax.text(
                xx,
                v - 0.02,
                f"{v:.3f}",
                ha="center",
                va="top",
                rotation=90,
                fontsize=7,
                color="white",
                fontweight="bold",
            )
    ax.set_xticks(np.arange(len(metrics)), [n for _, n in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("value (axis from 0)")
    npos = int(pooled.iloc[0]["positive_queries"])
    ax.set_title(
        f"Retrieval metrics over the {npos} positive test queries (geometric verification "
        "does not change retrieval)",
        fontsize=9.5,
    )
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    _title(fig, title)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_f1_comparison(test: pd.DataFrame, methods: Sequence[str], title: str) -> plt.Figure:
    """F1 per test sequence and pooled at the dev threshold; pooled precision/recall; FP/1000."""
    apply_style()
    seqs = [s for s in test.sequence.unique() if s != "pooled"] + ["pooled"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    ax = axes[0]
    width = 0.8 / len(methods)
    for j, m in enumerate(methods):
        rows = test[test.method == m].set_index("sequence")
        x = np.arange(len(seqs)) + (j - (len(methods) - 1) / 2) * width
        ax.bar(
            x,
            [float(rows.loc[s, "det_f1"]) for s in seqs],
            width,
            color=color(m),
            label=label(m),
            hatch="//" if m.endswith("_gv") else None,
            edgecolor="white",
        )
    for k, sq in enumerate(seqs):
        vals = [
            float(test[(test.method == m) & (test.sequence == sq)].det_f1.iloc[0]) for m in methods
        ]
        if max(vals) == 0:
            ax.text(k, 0.03, "F1 = 0 for\nevery method", ha="center", fontsize=8, color="#4a5568")
    ax.set_xticks(np.arange(len(seqs)), [f"KITTI {s}" if s != "pooled" else "pooled" for s in seqs])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 at the development threshold")
    ax.set_title(
        "(a) Detection F1 (axis from 0; hatched = with geometric verification)", fontsize=10
    )
    ax.legend(fontsize=7.5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    pooled = test[test.sequence == "pooled"].set_index("method")
    ax = axes[1]
    for m in methods:
        r = pooled.loc[m]
        ax.scatter(
            r.det_recall,
            r.det_precision,
            s=80,
            color=color(m),
            edgecolor="black",
            lw=0.6,
            marker="s" if m.endswith("_gv") else "o",
            label=label(m),
            zorder=3,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("(b) Pooled precision vs recall", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[2]
    y = np.arange(len(methods))
    vals = [float(pooled.loc[m, "det_fp_per_1000"]) for m in methods]
    ax.barh(y, vals, color=[color(m) for m in methods])
    ax.set_yticks(y, [label(m) for m in methods])
    ax.invert_yaxis()
    for yy, v, m in zip(y, vals, methods, strict=True):
        ax.text(v, yy, f" {v:.1f} ({int(pooled.loc[m, 'det_fp'])} FP)", va="center", fontsize=8)
    ax.set_xlabel("false positives per 1,000 queries")
    ax.set_title("(c) False alarms (pooled)", fontsize=10)
    ax.grid(axis="y", visible=False)
    _title(fig, title)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_confusion(pooled: pd.DataFrame, methods: Sequence[str], title: str) -> plt.Figure:
    apply_style()
    n = len(methods)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.0 * rows), squeeze=False)
    for ax, m in zip(axes.ravel(), methods, strict=False):
        cm = np.array(json.loads(pooled.loc[m, "det_confusion"]))
        ax.imshow(cm / cm.sum(axis=1, keepdims=True).clip(min=1), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if cm[i, j] / max(cm[i].sum(), 1) > 0.6 else "black",
                )
        ax.set_xticks(range(3), ["correct loop", "wrong loop", "no loop"], fontsize=8)
        ax.set_yticks(range(2), ["has revisit", "no revisit"], fontsize=8)
        ax.grid(False)
        ax.set_title(label(m), fontsize=10, color=color(m))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    _title(fig, title)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# ----------------------------------------------------------------------------- efficiency


def plot_runtime(eff: pd.DataFrame, methods: Sequence[str], title: str) -> plt.Figure:
    apply_style()
    e = eff.set_index("method").loc[list(methods)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    y = np.arange(len(methods))
    parts = [
        ("encoder_ms_per_frame", "encoder", "#4a5568"),
        ("retrieval_ms_per_query", "retrieval", "#a0aec0"),
        ("verification_ms_per_query", "geometric verification (CPU)", "#dd6b20"),
    ]
    left = np.zeros(len(methods))
    for col, name, c in parts:
        v = e[col].to_numpy(float)
        ax.barh(y, v, left=left, color=c, label=name)
        left += v
    for yy, tot, m in zip(y, left, methods, strict=True):
        dev = f"{e.loc[m, 'device']}".upper()
        ax.text(tot, yy, f" {tot:.0f} ms (encoder on {dev})", va="center", fontsize=8)
    ax.set_yticks(y, [label(m) for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel("end-to-end latency per query [ms]")
    ax.set_xlim(0, left.max() * 1.35)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("(a) End-to-end latency = encoder + retrieval + verification", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    v = e["retrieval_ms_per_query"].to_numpy(float)
    ax.barh(y, v, color=[color(m) for m in methods])
    for yy, vv in zip(y, v, strict=True):
        ax.text(vv, yy, f" {vv:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, [label(m) for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel("retrieval latency per query [ms] (CPU, exact FAISS, causal)")
    ax.set_title("(b) Retrieval alone", fontsize=10)
    ax.grid(axis="y", visible=False)
    _title(fig, title)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_memory(eff: pd.DataFrame, methods: Sequence[str], title: str) -> plt.Figure:
    apply_style()
    e = eff.set_index("method").loc[list(methods)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    specs = [
        ("peak_gpu_mb", "peak GPU memory [MB] (encoder)", "(a) GPU memory"),
        ("cpu_rss_increase_mb", "host RSS increase [MB] (encoder loaded)", "(b) CPU memory"),
        (
            "embedding_storage_mb_test",
            "descriptor storage for the test frames [MB]",
            "(c) Embedding storage",
        ),
    ]
    y = np.arange(len(methods))
    for ax, (col, xlabel, t) in zip(axes, specs, strict=True):
        v = pd.to_numeric(e[col], errors="coerce").to_numpy(float)
        ax.barh(y, np.nan_to_num(v), color=[color(m) for m in methods])
        for yy, vv, m in zip(y, v, methods, strict=True):
            txt = "n/a" if not np.isfinite(vv) else f"{vv:.1f}"
            if col == "embedding_storage_mb_test":
                txt += f" ({int(e.loc[m, 'embedding_dim'])}-D)"
            ax.text(np.nan_to_num(vv), yy, " " + txt, va="center", fontsize=8)
        ax.set_yticks(y, [label(m) for m in methods])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(t, fontsize=10)
        ax.grid(axis="y", visible=False)
    _title(fig, title)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


# ----------------------------------------------------------------------------- trajectories


def _links(ax, pred: pd.DataFrame, positions: np.ndarray, index: dict, threshold: float) -> dict:
    on = pred.scores.fillna(-np.inf) >= threshold
    counts = {"tp": 0, "fp": 0, "fn": 0}
    for _, r in pred[on].iterrows():
        qi, ci = index[int(r.query_frame_ids)], index[int(r.candidate_ids)]
        c = C_TP if r.correct else C_FP
        ax.plot(
            positions[[qi, ci], 0],
            positions[[qi, ci], 2],
            color=c,
            lw=1.6 if r.correct else 1.2,
            zorder=4 if not r.correct else 3,
        )
        counts["tp" if r.correct else "fp"] += 1
    fn = pred[pred.has_positive & ~(on & pred.correct)]
    qi = [index[int(q)] for q in fn.query_frame_ids]
    if qi:
        ax.scatter(positions[qi, 0], positions[qi, 2], s=10, color=C_FN, marker="x", zorder=5)
    counts["fn"] = len(qi)
    return counts


def plot_sequence_trajectories(
    preds: dict[str, pd.DataFrame],
    positions: np.ndarray,
    frame_ids: np.ndarray,
    thresholds: dict[str, float],
    methods: Sequence[str],
    title: str,
) -> plt.Figure:
    apply_style()
    index = {int(f): i for i, f in enumerate(frame_ids)}
    cols = 3
    rows = int(np.ceil(len(methods) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 5.0 * rows), squeeze=False)
    for ax, m in zip(axes.ravel(), methods, strict=False):
        ax.plot(positions[:, 0], positions[:, 2], color="#cbd5e0", lw=1.4, zorder=1)
        c = _links(ax, preds[m], positions, index, thresholds[m])
        format_xz_axes(ax)
        ax.set_title(
            f"{label(m)}: {c['tp']} correct, {c['fp']} wrong, {c['fn']} missed",
            fontsize=9.5,
            color=color(m),
        )
    for ax in axes.ravel()[len(methods) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([], [], color=C_TP, lw=2, label="predicted loop, correct (<= 5 m)"),
        plt.Line2D([], [], color=C_FP, lw=2, label="predicted loop, wrong"),
        plt.Line2D(
            [], [], color=C_FN, marker="x", ls="", label="revisit query without a correct loop"
        ),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8.5, ncol=3)
    _title(fig, title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_examples(
    examples: Sequence[dict], get_image: Callable[[str, int], np.ndarray], title: str
) -> plt.Figure:
    """Rows: query | predicted loop | closest true revisit (if any)."""
    apply_style()
    h, w = get_image(examples[0]["sequence"], examples[0]["query"]).shape[:2]
    cell = 4.2
    fig, axes = plt.subplots(
        len(examples),
        3,
        figsize=(3 * cell, len(examples) * (cell * h / w + 0.62) + 0.5),
        squeeze=False,
    )
    for row, ex in zip(axes, examples, strict=True):
        seq = ex["sequence"]
        border = C_FP if ex["kind"] == "FP" else C_FN

        def show(ax, fid, text, colr, seq=seq):
            if fid is None or fid < 0:
                ax.axis("off")
                ax.text(0.02, 0.5, text, transform=ax.transAxes, fontsize=8.5, color=colr)
                return
            ax.imshow(get_image(seq, int(fid)))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(text, fontsize=8.5, loc="left", color=colr)
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_edgecolor(colr)
                s.set_linewidth(3)

        show(
            row[0],
            ex["query"],
            f"{ex['kind']} ({label(ex['method'])}), KITTI {seq}\nquery #{ex['query']}",
            border,
        )
        if ex["candidate"] >= 0:
            show(
                row[1],
                ex["candidate"],
                f"predicted loop #{ex['candidate']}, score {ex['score']:.3g}\n"
                f"{ex['distance']:.0f} m from the query"
                + (" (accepted)" if ex["accepted"] else " (below threshold)"),
                C_FP if ex["kind"] == "FP" or ex["accepted"] else "#4a5568",
            )
        else:
            show(row[1], None, "no candidate passed verification", "#4a5568")
        if ex.get("true_ref") is not None:
            show(
                row[2],
                ex["true_ref"],
                f"closest true revisit #{ex['true_ref']}\n{ex['true_distance']:.1f} m away",
                C_TP,
            )
        else:
            show(row[2], None, "no true revisit exists for this query", "#4a5568")
    _title(fig, title)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_size_inches()[1]))
    return fig


def render_playback_frames(
    preds: dict[str, pd.DataFrame],
    positions: np.ndarray,
    frame_ids: np.ndarray,
    thresholds: dict[str, float],
    methods: Sequence[str],
    get_image: Callable[[int], np.ndarray],
    steps: np.ndarray,
    title: str,
    width_px: int = 900,
) -> list[np.ndarray]:
    """Trajectory playback: camera frame + one map per method with predicted loops so far."""
    apply_style()
    index = {int(f): i for i, f in enumerate(frame_ids)}
    img0 = get_image(0)
    h, w = img0.shape[:2]
    dpi = 100
    fig_w = width_px / dpi
    img_h = fig_w * 0.6 * h / w
    map_h = fig_w / len(methods) * 0.95
    fig_h = 0.7 + img_h + 0.35 + map_h + 0.55
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    head = fig.text(0.01, 1 - 0.28 / fig_h, "", fontsize=10.5, fontweight="bold")
    ax_img = fig.add_axes((0.2, 1 - (0.7 + img_h) / fig_h, 0.6, img_h / fig_h))
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.grid(False)
    im = ax_img.imshow(img0)
    cap = ax_img.set_title("", fontsize=9, loc="left")
    maps, arts = [], []
    for j, m in enumerate(methods):
        ax = fig.add_axes(
            (0.04 + j / len(methods), 0.55 / fig_h, 1 / len(methods) - 0.07, map_h / fig_h)
        )
        ax.plot(positions[:, 0], positions[:, 2], color="#edf2f7", lw=1.2)
        (trail,) = ax.plot([], [], color="#718096", lw=1.2)
        (cur,) = ax.plot([], [], "o", color="black", ms=5, zorder=6)
        format_xz_axes(ax)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("")
        ax.set_ylabel("")
        t = ax.set_title(label(m), fontsize=8.5, color=color(m))
        maps.append(ax)
        arts.append((trail, cur, t, []))
    fig.text(
        0.01,
        0.2 / fig_h,
        "green: predicted loop, correct   red: predicted loop, wrong   orange x: "
        "revisit without a correct loop (threshold from development data)",
        fontsize=8,
        color="#4a5568",
    )
    on = {m: (preds[m].scores.fillna(-np.inf) >= thresholds[m]).to_numpy() for m in methods}
    frames = []
    for t in steps:
        t = int(t)
        im.set_data(get_image(t))
        cap.set_text(f"frame #{int(frame_ids[t])}")
        for ax, m, (trail, cur, ttl, drawn) in zip(maps, methods, arts, strict=True):
            for a in drawn:
                a.remove()
            drawn.clear()
            p = preds[m].iloc[: t + 1]
            o = on[m][: t + 1]
            tp = fp = 0
            for k in np.flatnonzero(o):
                qi, ci = k, index[int(p.candidate_ids.iloc[k])]
                ok = bool(p.correct.iloc[k])
                (ln,) = ax.plot(
                    positions[[qi, ci], 0],
                    positions[[qi, ci], 2],
                    color=C_TP if ok else C_FP,
                    lw=1.5,
                    zorder=4,
                )
                drawn.append(ln)
                tp, fp = tp + ok, fp + (not ok)
            miss = np.flatnonzero(p.has_positive.to_numpy() & ~(o & p.correct.to_numpy()))
            if miss.size:
                drawn.append(
                    ax.scatter(
                        positions[miss, 0],
                        positions[miss, 2],
                        s=8,
                        color=C_FN,
                        marker="x",
                        zorder=5,
                    )
                )
            trail.set_data(positions[: t + 1, 0], positions[: t + 1, 2])
            cur.set_data([positions[t, 0]], [positions[t, 2]])
            ttl.set_text(f"{label(m)}: {tp} correct, {fp} wrong, {miss.size} missed")
        head.set_text(f"{title}: frame {t + 1}/{len(frame_ids)}")
        frames.append(figure_to_array(fig))
    plt.close(fig)
    return frames
