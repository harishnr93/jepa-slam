"""Phase 6 artefacts from ``results/benchmark/<name>/`` (nothing is recomputed here)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latent_loop.datasets import build_dataset
from latent_loop.datasets.ground_truth import revisit_ground_truth
from latent_loop.evaluation.runner import sequence_config
from latent_loop.utils.config import resolve_path
from latent_loop.visualization.animations import save_gif, select_animation_indices
from latent_loop.visualization.evaluation import (
    C_FN,
    C_FP,
    C_TP,
    color,
    plot_confusion,
    plot_examples,
    plot_f1_comparison,
    plot_memory,
    plot_pr_curves,
    plot_recall_bars,
    plot_runtime,
    plot_sequence_trajectories,
    render_playback_frames,
)
from latent_loop.visualization.plots import save_figure

RETRIEVAL_METHODS = ("ijepa", "dinov2", "orb_bovw", "hsv")


def _load_curves(out: Path, split: str) -> dict[str, dict]:
    d = out / "curves" / split
    return {p.stem: dict(np.load(p)) for p in sorted(d.glob("*.npz"))} if d.is_dir() else {}


def _examples(out, test, seqs, thresholds, methods, per_kind=2):
    """Highest-scoring false positives and highest-ranked missed revisits per method."""
    exs = []
    for m in methods:
        preds = []
        for seq in seqs:
            p = pd.read_csv(out / "predictions" / "test" / f"{m}_{seq}.csv")
            p["sequence"] = seq
            preds.append(p)
        p = pd.concat(preds, ignore_index=True)
        on = p.scores.fillna(-np.inf) >= thresholds[m]["f1"]
        fp = p[on & ~p.correct].sort_values("scores", ascending=False).head(per_kind)
        # Missed revisits with *no* accepted loop (a wrong loop is already shown as an FP):
        # the median such query of each of the sequences with the most misses.
        fn_all = p[p.has_positive & ~on & p.evaluable]
        order = fn_all.sequence.value_counts().index[:per_kind]
        fn = (
            pd.concat(
                [
                    fn_all[fn_all.sequence == sq].iloc[[len(fn_all[fn_all.sequence == sq]) // 2]]
                    for sq in order
                ]
            )
            if len(fn_all)
            else fn_all
        )
        for kind, rows in (("FP", fp), ("FN", fn)):
            for _, r in rows.iterrows():
                exs.append(
                    {
                        "method": m,
                        "kind": kind,
                        "sequence": r.sequence,
                        "query": int(r.query_frame_ids),
                        "candidate": int(r.candidate_ids),
                        "score": float(r.scores) if np.isfinite(r.scores) else float("nan"),
                        "distance": float(r.candidate_distance_m)
                        if np.isfinite(r.candidate_distance_m)
                        else float("nan"),
                        "accepted": bool(np.isfinite(r.scores) and r.scores >= thresholds[m]["f1"]),
                    }
                )
    return exs


def generate_phase6(
    config: dict[str, Any],
    out_dir: Path,
    gif_frames: int = 110,
    gif_fps: float = 8.0,
    gif_width: int = 900,
    gif_sequence: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    b = config["benchmark"]
    bench = resolve_path(b.get("output_dir", "results/benchmark")) / b.get("name", "default")
    test = pd.read_csv(bench / "test_metrics.csv", dtype={"sequence": str})
    dev = pd.read_csv(bench / "dev_metrics.csv", dtype={"sequence": str})
    thresholds = json.loads((bench / "thresholds.json").read_text())
    eff = pd.read_csv(bench / "efficiency.csv")
    methods = [m for m in b["methods"] if m in set(test.method)]
    pooled = test[test.sequence == "pooled"].set_index("method")
    seqs = [s for s in test.sequence.unique() if s != "pooled"]
    outputs: dict[str, str] = {}
    stem = out_dir / b.get("name", "benchmark")

    curves = {"dev": _load_curves(bench, "dev"), "test": _load_curves(bench, "test")}
    operating = {
        m: {"precision": pooled.loc[m, "det_precision"], "recall": pooled.loc[m, "det_recall"]}
        for m in methods
    }
    fig = plot_pr_curves(
        curves,
        operating,
        methods,
        "Loop-closure detection precision-recall (pooled; one hypothesis per query)",
    )
    outputs["pr_curves"] = str(save_figure(fig, Path(f"{stem}_pr_curves.png")))
    rm = [m for m in RETRIEVAL_METHODS if m in methods]
    fig = plot_recall_bars(
        pooled, rm, "Recall@K and mAP on the held-out test sequences (02, 06, 08, 09)"
    )
    outputs["recall_bars"] = str(save_figure(fig, Path(f"{stem}_recall_at_k.png")))
    fig = plot_f1_comparison(
        test, methods, "Detection at the threshold selected on development data (test sequences)"
    )
    outputs["f1"] = str(save_figure(fig, Path(f"{stem}_f1_comparison.png")))
    fig = plot_confusion(
        pooled, methods, "Query-level confusion matrices (pooled test, development thresholds)"
    )
    outputs["confusion"] = str(save_figure(fig, Path(f"{stem}_confusion.png")))
    fig = plot_runtime(
        eff,
        methods,
        "Runtime per query (encoder: profile on 64 KITTI frames; rest: measured on test)",
    )
    outputs["runtime"] = str(save_figure(fig, Path(f"{stem}_runtime.png")))
    fig = plot_memory(eff, methods, "Memory and storage")
    outputs["memory"] = str(save_figure(fig, Path(f"{stem}_memory.png")))

    datasets, gts = {}, {}
    for seq in seqs:
        ds = build_dataset(sequence_config(config, seq))
        datasets[seq] = ds
        excl = int(config["retrieval"].get("temporal_exclusion_frames") or 0)
        gts[seq] = revisit_ground_truth(
            ds.poses(), ds.frame_ids, float(config["ground_truth"]["spatial_threshold_m"]), excl
        )
    thr = {m: thresholds[m]["f1"] for m in methods}
    for seq in seqs:
        ds = datasets[seq]
        preds = {m: pd.read_csv(bench / "predictions" / "test" / f"{m}_{seq}.csv") for m in methods}
        fig = plot_sequence_trajectories(
            preds,
            ds.poses()[:, :3, 3],
            ds.frame_ids,
            thr,
            methods,
            f"Predicted loop closures on KITTI {seq} (held-out; development thresholds)",
        )
        outputs[f"trajectory_{seq}"] = str(save_figure(fig, Path(f"{stem}_trajectory_{seq}.png")))

    gv = [m for m in ("ijepa_gv", "dinov2_gv") if m in methods] or methods[:2]
    exs = _examples(bench, test, seqs, thresholds, gv)
    for ex in exs:  # closest true revisit (labels only)
        ds, gt = datasets[ex["sequence"]], gts[ex["sequence"]]
        i = int(np.searchsorted(ds.frame_ids, ex["query"]))
        pos = gt.positives[i]
        if pos.size:
            xz = ds.poses()[:, [0, 2], 3]
            idx = np.searchsorted(ds.frame_ids, pos)
            d = np.linalg.norm(xz[idx] - xz[i], axis=1)
            ex["true_ref"], ex["true_distance"] = int(pos[np.argmin(d)]), float(d.min())
        else:
            ex["true_ref"] = None
    fig = plot_examples(
        exs,
        lambda s, f: datasets[s].image_for_frame_id(f),
        "False positives (highest scores) and false negatives (missed revisits), test sequences",
    )
    outputs["fp_fn_examples"] = str(save_figure(fig, Path(f"{stem}_fp_fn_examples.png")))

    if gif_sequence is None:
        gif_sequence = max(seqs, key=lambda s: int(gts[s].has_positive.sum()))
    ds = datasets[gif_sequence]
    preds = {m: pd.read_csv(bench / "predictions" / "test" / f"{m}_{gif_sequence}.csv") for m in gv}
    start = time.perf_counter()
    steps = select_animation_indices(len(ds), gif_frames)
    frames = render_playback_frames(
        preds,
        ds.poses()[:, :3, 3],
        ds.frame_ids,
        thr,
        gv,
        ds.get_image,
        steps,
        f"Loop closures on held-out KITTI {gif_sequence}",
        gif_width,
    )
    gif = save_gif(
        frames,
        Path(f"{stem}_playback_{gif_sequence}.gif"),
        fps=gif_fps,
        reserve_colors=[color(m) for m in gv] + [C_TP, C_FP, C_FN],
    )
    outputs["playback_gif"] = str(gif)
    summary = {
        "phase": 6,
        "benchmark_dir": str(bench),
        "methods": methods,
        "test_sequences": seqs,
        "gif_sequence": gif_sequence,
        "gif_frames": len(frames),
        "gif_size_mb": round(gif.stat().st_size / 1e6, 2),
        "gif_render_seconds": round(time.perf_counter() - start, 1),
        "examples": exs,
        "outputs": {k: Path(v).name for k, v in outputs.items()},
        "dev_rows": int(len(dev)),
    }
    (out_dir / "phase6_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
