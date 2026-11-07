"""Phase 8: one end-to-end GIF of the whole pipeline, rendered from saved outputs.

Every number and image comes from disk: cached I-JEPA descriptors, the benchmark's saved
retrieval result, the pair cache (ORB + RANSAC, recomputed here for the drawing and
checked against the cache), the saved predictions, thresholds and test metrics. Ground
truth poses are used only to draw the trajectory and to label the outcome.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from latent_loop.dashboard.data import METHOD_ENCODER, ResultStore
from latent_loop.features.cache import EmbeddingCache
from latent_loop.utils.config import resolve_path
from latent_loop.verification import VerificationConfig, raw_nearest_matches, verify_features
from latent_loop.verification.orb import extract_orb
from latent_loop.visualization.animations import save_gif
from latent_loop.visualization.evaluation import C_FN, C_FP, C_TP, color, label
from latent_loop.visualization.matches import PairExample, inlier_canvas
from latent_loop.visualization.plots import figure_to_array

STEPS = ("frames", "embedding", "retrieval", "verification", "decision", "trajectory", "benchmark")
ACTIVE = "#22543d"  # dark green, white text (the README diagram colour)
GREY = "#cbd5e0"
EXCLUDED = "#ecc94b"


def _canvas(active: str | tuple[str, ...], caption: str, size=(9.6, 5.4)):
    """Figure with the pipeline ribbon on top and a caption line at the bottom."""
    active = (active,) if isinstance(active, str) else active
    fig = plt.figure(figsize=size, dpi=100, facecolor="white")
    n = len(STEPS)
    for k, name in enumerate(STEPS):
        on = name in active
        x = 0.01 + k * 0.98 / n
        fig.patches.append(
            FancyBboxPatch(
                (x + 0.004, 0.925),
                0.98 / n - 0.008,
                0.055,
                boxstyle="round,pad=0,rounding_size=0.008",
                transform=fig.transFigure,
                facecolor=ACTIVE if on else "#f7fafc",
                edgecolor=ACTIVE if on else GREY,
                linewidth=1.0,
            )
        )
        fig.text(
            x + 0.49 / n,
            0.952,
            f"{k + 1} {name}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if on else "#4a5568",
            weight="bold" if on else "normal",
        )
    fig.text(0.5, 0.018, caption, ha="center", va="bottom", fontsize=8.5, color="#2d3748")
    return fig


def _traj(ax, xz, upto=None, title="trajectory (ground truth, display only)"):
    """Ground-truth path; very elongated paths (KITTI 06) are stretched across, and say so."""
    ax.set_facecolor("white")
    ax.plot(xz[:, 0], xz[:, 1], color=GREY, lw=1.2, zorder=1)
    if upto is not None:
        ax.plot(xz[: upto + 1, 0], xz[: upto + 1, 1], color="#4a5568", lw=1.6, zorder=2)
    span = np.ptp(xz, axis=0)
    if span[1] > 3 * span[0]:
        title += ", x stretched"
    else:
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GREY)
    ax.set_title(title, fontsize=8.5)


def _img(ax, img, title="", border=None):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(border or GREY)
        s.set_linewidth(3 if border else 1)
    if title:
        ax.set_title(title, fontsize=8.5, color=border or "black")


def generate_phase8(
    config_path: str | Path,
    out_dir: Path,
    sequence: str = "06",
    method: str = "ijepa_gv",
    query: int | None = 1010,
    gif_fps: float = 8.0,
    gif_width: int = 960,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Render ``pipeline_end_to_end.gif`` for one held-out query and the whole sequence."""
    t_start = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ResultStore(config_path, overrides)
    enc = METHOD_ENCODER[method]
    row = store.row(method, sequence)
    result = store.retrieval(method, sequence)
    pred = store.predictions(method, sequence)
    pairs = store.pairs(sequence)
    vc = store.verification_config()
    ds = store.dataset(sequence)
    if any(x is None for x in (row, result, pred, pairs, vc, ds)):
        raise FileNotFoundError(
            "Phase 8 needs the Phase 6 benchmark outputs and the KITTI images/poses. "
            + store.missing_message().how
        )
    thr = float(store.thresholds()[method]["f1"])
    cache = EmbeddingCache(
        resolve_path(store.config["extraction"]["cache_root"]),
        f"kitti_{sequence}_cam{store.camera}",
        enc,
        str(row.fingerprint),
    ).load()
    if cache is None:
        raise FileNotFoundError(f"No cached {enc} descriptors for KITTI {sequence}")

    ids = result.query_frame_ids
    xz_all = ds.poses()[:, [0, 2], 3]
    pos = {int(f): xz_all[k] for k, f in enumerate(ds.frame_ids)}
    xz = np.array([pos[int(f)] for f in ids])
    if query is None:  # median correct accepted loop
        ok = pred[pred.correct & (pred.scores >= thr)].sort_values("scores")
        query = int(ok.query_frame_ids.iloc[len(ok) // 2])
    qi = int(np.searchsorted(ids, query))
    cands = [int(c) for c in result.candidate_ids[qi, :5] if c >= 0]
    cos = [float(s) for s in result.scores[qi, : len(cands)]]
    n_elig = int(result.eligible_counts[qi])
    img_q = ds.image_for_frame_id(query)
    thumbs = {c: ds.image_for_frame_id(c) for c in cands}
    frames: list[np.ndarray] = []
    seq_name = f"KITTI {sequence} (held-out test)"

    def push(fig):
        frames.append(figure_to_array(fig))
        plt.close(fig)

    # ---------------------------------------------------------------- 1-2 frames -> embedding
    emb_ids = np.unique(np.linspace(0, qi, 18).round().astype(int))
    vecs = cache.select(ids[emb_ids])
    vmax = float(np.abs(vecs).max())
    for n, k in enumerate(emb_ids):
        fig = _canvas(
            ("frames", "embedding"),
            f"{seq_name}: each frame (every 5th) -> frozen I-JEPA ViT-H/14 -> mean-pooled, "
            f"L2-normalised {vecs.shape[1]}-d descriptor (read from the embedding cache)",
        )
        gs = fig.add_gridspec(
            2,
            2,
            left=0.02,
            right=0.98,
            top=0.88,
            bottom=0.08,
            width_ratios=[1.75, 1],
            hspace=0.25,
            wspace=0.05,
        )
        _img(
            fig.add_subplot(gs[0, 0]),
            ds.image_for_frame_id(int(ids[k])),
            f"camera frame {int(ids[k])}",
        )
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(vecs[n].reshape(16, -1), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"descriptor of frame {int(ids[k])} (16 x {vecs.shape[1] // 16} view)", fontsize=8.5
        )
        a = fig.add_subplot(gs[:, 1])
        _traj(a, xz, upto=k)
        a.scatter(*xz[k], s=40, color=C_FN, zorder=3)
        push(fig)
    for _ in range(3):
        frames.append(frames[-1])

    # ---------------------------------------------------------------- 3 retrieval
    gt_d = {c: float(np.linalg.norm(pos[c] - pos[query])) for c in cands}
    for shown in list(range(0, len(cands) + 1)) + [len(cands)] * 4:
        fig = _canvas(
            "retrieval",
            f"FAISS inner product over the {n_elig} references older than the temporal "
            f"exclusion window (yellow); top-5 by cosine. Distances: ground truth, display only.",
        )
        gs = fig.add_gridspec(
            2,
            5,
            left=0.02,
            right=0.98,
            top=0.88,
            bottom=0.08,
            height_ratios=[1.5, 1],
            hspace=0.12,
            wspace=0.06,
        )
        _img(fig.add_subplot(gs[0, :3]), img_q, f"query frame {query}")
        a = fig.add_subplot(gs[0, 3:])
        _traj(a, xz, upto=qi)
        a.scatter(*xz[n_elig : qi + 1].T, s=6, color=EXCLUDED, zorder=3, label="excluded")
        a.scatter(*xz[qi], s=50, color="black", zorder=5, label="query")
        for r, c in enumerate(cands[:shown]):
            p = pos[c]
            a.scatter(*p, s=40, marker="s", color=color(method), zorder=4)
            a.annotate(str(r + 1), p, xytext=(4, 4), textcoords="offset points", fontsize=7)
        a.legend(fontsize=7, loc="lower right", frameon=False)
        for r in range(len(cands)):
            ax = fig.add_subplot(gs[1, r])
            if r < shown:
                c = cands[r]
                _img(ax, thumbs[c], f"#{r + 1} frame {c}\ncos {cos[r]:.3f}, {gt_d[c]:.0f} m")
            else:
                ax.axis("off")
        push(fig)

    # ---------------------------------------------------------------- 4 verification
    cfg = VerificationConfig.from_dict(vc["config"])
    K = ds.get_calibration()
    fq = extract_orb(img_q, cfg.orb)
    verified: list[dict] = []
    canv = {}
    for r, c in enumerate(cands):
        fr = extract_orb(thumbs[c], cfg.orb)
        res = verify_features(fq, fr, K, cfg, keep_details=True)
        cached = pairs.loc[(query, c)]
        verified.append(
            {
                "candidate": c,
                "rank": r + 1,
                "inliers": int(res.n_inliers),
                "reason": res.reason,
                "accepted_checks": bool(res.accepted),
                "matches_cache": bool(
                    res.reason == cached.reason and res.n_inliers == int(cached.n_inliers)
                ),
            }
        )
        ex = PairExample(
            query, c, img_q, thumbs[c], fq, fr, res, raw_nearest_matches(fq, fr), cos[r], r + 1
        )
        canv[c] = inlier_canvas(ex, max_lines=200)
    for r, c in enumerate(cands):
        for _ in range(3):
            fig = _canvas(
                "verification",
                "ORB (ratio + mutual test) -> essential-matrix RANSAC (OpenCV five-point) -> "
                "geometric checks. Green: inliers, red: rejected matches.",
            )
            gs = fig.add_gridspec(
                1,
                2,
                left=0.02,
                right=0.98,
                top=0.88,
                bottom=0.1,
                width_ratios=[1.6, 1],
                wspace=0.2,
            )
            v = verified[r]
            _img(
                fig.add_subplot(gs[0]),
                canv[c],
                f"query {query} (top) vs candidate #{r + 1} frame {c} (bottom): "
                f"{v['inliers']} inliers, {v['reason'].replace('_', ' ')}",
            )
            ax = fig.add_subplot(gs[1])
            vals = [verified[j]["inliers"] if j <= r else 0 for j in range(len(cands))]
            bars = ax.bar(
                range(1, len(cands) + 1),
                vals,
                color=[
                    color(method) if verified[j]["accepted_checks"] else GREY
                    for j in range(len(cands))
                ],
            )
            for b, val, j in zip(bars, vals, range(len(cands)), strict=True):
                if j <= r:
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        val,
                        str(val),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            ax.axhline(thr, color=C_FP, ls="--", lw=1)
            ax.text(
                len(cands) + 0.45,
                thr,
                f"threshold {thr:.0f} (dev data)",
                va="bottom",
                ha="right",
                fontsize=7.5,
                color=C_FP,
            )
            ax.set_xticks(range(1, len(cands) + 1), [f"#{j + 1}" for j in range(len(cands))])
            ax.set_ylim(0, max(thr, max(v["inliers"] for v in verified)) * 1.15)
            ax.set_ylabel("RANSAC inliers")
            ax.set_title("inliers per candidate (grey: failed a check)", fontsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)
            push(fig)

    # ---------------------------------------------------------------- 5 decision
    valid = [v for v in verified if v["accepted_checks"]]
    best = max(valid, key=lambda v: v["inliers"]) if valid else None
    p = pred.set_index("query_frame_ids").loc[query]
    accepted = best is not None and best["inliers"] >= thr
    truth = "true revisit" if bool(p.correct) else "not a revisit"
    outcome = ("TP" if p.correct else "FP") if accepted else ("FN" if p.has_positive else "TN")
    for _ in range(14):
        fig = _canvas(
            "decision",
            "Score = max inliers over the valid top-5 candidates; one "
            "hypothesis per query. Ground truth is consulted only to grade it.",
        )
        gs = fig.add_gridspec(
            2,
            2,
            left=0.03,
            right=0.97,
            top=0.86,
            bottom=0.1,
            height_ratios=[1, 0.62],
            wspace=0.05,
            hspace=0.1,
        )
        bc = best["candidate"] if best else cands[0]
        ok = C_TP if outcome == "TP" else C_FP
        _img(fig.add_subplot(gs[0, 0]), img_q, f"query {query}")
        _img(fig.add_subplot(gs[0, 1]), thumbs[bc], f"loop candidate {bc}", border=ok)
        ax = fig.add_subplot(gs[1, :])
        ax.axis("off")
        verdict = "ACCEPTED" if accepted else "REJECTED"
        ax.text(
            0.5,
            0.72,
            f"{verdict}: {best['inliers'] if best else 0} inliers "
            f"{'>=' if accepted else '<'} {thr:.0f} (threshold chosen on KITTI 00 + 05)",
            ha="center",
            fontsize=13,
            weight="bold",
            color=ok,
        )
        ax.text(
            0.5,
            0.25,
            f"graded against ground truth: {truth}, "
            f"{float(p.candidate_distance_m):.1f} m apart  ->  {outcome}",
            ha="center",
            fontsize=11,
            color="#2d3748",
        )
        push(fig)

    # ---------------------------------------------------------------- 6 trajectory playback
    on = (pred.scores.fillna(-np.inf) >= thr).to_numpy()
    steps = np.unique(np.linspace(0, len(ids) - 1, 30).round().astype(int))
    for k in list(steps) + [steps[-1]] * 6:
        fig = _canvas(
            "trajectory",
            f"Causal replay of {seq_name} with {label(method)}: every "
            "accepted loop drawn as an edge. Thresholds fixed beforehand on dev data.",
        )
        ax = fig.add_axes([0.2, 0.08, 0.6, 0.8])
        _traj(ax, xz, upto=k, title=f"frame {int(ids[k])}: ground truth path, display only")
        tp = fp = 0
        for j in range(k + 1):
            if on[j]:
                c = int(pred.candidate_ids.iloc[j])
                good = bool(pred.correct.iloc[j])
                tp, fp = tp + good, fp + (not good)
                ax.plot(
                    *np.c_[xz[j], pos[c]], color=C_TP if good else C_FP, lw=1.4, alpha=0.9, zorder=3
                )
        ax.scatter(*xz[k], s=45, color="black", zorder=5)
        ax.text(
            1.02,
            0.6,
            f"accepted loops\n correct: {tp}\n false: {fp}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
        )
        ax.plot([], [], color=C_TP, label="correct loop (TP)")
        ax.plot([], [], color=C_FP, label="false loop (FP)")
        ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(1.0, 0.0), frameon=False)
        push(fig)

    # ---------------------------------------------------------------- 7 benchmark
    test = store.metrics("test")
    pooled = test[test.sequence == "pooled"].set_index("method")
    methods = [m for m in store.methods if m in pooled.index]
    for _ in range(18):
        fig = _canvas(
            "benchmark",
            "Pooled over test sequences 02, 06, 08, 09 (2,208 queries, "
            "190 revisits). KITTI 08 (opposite-direction revisits): F1 = 0 for all.",
        )
        ax = fig.add_axes([0.08, 0.14, 0.88, 0.7])
        x = np.arange(len(methods))
        for off, (col, name, alpha) in zip(
            (-0.2, 0.2),
            (("recall_at_1", "Recall@1", 0.45), ("det_f1", "detection F1", 1.0)),
            strict=True,
        ):
            vals = pooled.loc[methods, col].to_numpy(float)
            ax.bar(
                x + off,
                vals,
                0.38,
                color=[color(m) for m in methods],
                alpha=alpha,
                label=name,
                edgecolor="white",
            )
            for xi, v in zip(x + off, vals, strict=True):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x, [label(m) for m in methods], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("pooled test value (axis from 0)")
        ax.legend(
            fontsize=8,
            frameon=False,
            loc="upper right",
            title="light: Recall@1, solid: detection F1",
            title_fontsize=8,
        )
        ax.spines[["top", "right"]].set_visible(False)
        push(fig)

    from PIL import Image  # noqa: PLC0415

    h = int(round(frames[0].shape[0] * gif_width / frames[0].shape[1]))
    frames = [np.asarray(Image.fromarray(f).resize((gif_width, h), Image.LANCZOS)) for f in frames]
    gif = save_gif(
        frames,
        out_dir / "pipeline_end_to_end.gif",
        fps=gif_fps,
        reserve_colors=[color(m) for m in methods] + [C_TP, C_FP, C_FN, ACTIVE, EXCLUDED],
    )
    summary = {
        "phase": 8,
        "sequence": sequence,
        "method": method,
        "query": query,
        "threshold": thr,
        "top5": [
            {**v, "cosine": cos[k], "gt_distance_m": gt_d[v["candidate"]]}
            for k, v in enumerate(verified)
        ],
        "decision": {
            "accepted": accepted,
            "outcome": outcome,
            "saved_prediction_score": float(p.scores),
        },
        "outputs": {"pipeline_gif": gif.name},
        "gif_frames": len(frames),
        "gif_fps": gif_fps,
        "gif_size_mb": round(gif.stat().st_size / 1e6, 2),
        "render_seconds": round(time.perf_counter() - t_start, 1),
    }
    (out_dir / "phase8_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


__all__ = ["generate_phase8"]
