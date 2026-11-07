#!/usr/bin/env python3
"""Development-only probes of the descriptors, motivated by Balestriero & LeCun (2026).

Usage:
    python scripts/probe_representations.py                 # KITTI 00 + 05, all encoders
    python scripts/probe_representations.py --no-dinov2-mean   # skip the extra extraction

For each encoder (preprocessing variant chosen in Phase 6):
1. the scene-explained fraction rho, in the full space and in the top-k principal
   directions (PCA fitted on the *other* development sequence);
2. retrieval Recall@1 and mAP for causal frame averaging (window w) x PCA truncation (k),
   pooled over the development sequences, with paired McNemar tests against the baseline.

Only ``splits.dev`` sequences are accepted; the held-out test sequences are never read.
Writes ``results/probes/probes.json`` and ``probes.csv`` and figures in
``assets/improvements/``.
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from latent_loop.evaluation.benchmark import retrieval_average_precision
from latent_loop.evaluation.probes import (
    causal_average,
    concat_views,
    mcnemar,
    pca_basis,
    project,
    random_pairs,
    same_place_pairs,
    scene_explained_fraction,
)
from latent_loop.evaluation.runner import Benchmark
from latent_loop.retrieval import RetrievalConfig, retrieve
from latent_loop.utils.config import load_config, resolve_path
from latent_loop.utils.logging import setup_logging
from latent_loop.visualization.plots import apply_style, save_figure

LABEL = {
    "ijepa": "I-JEPA (mean patch)",
    "dinov2": "DINOv2 (CLS)",
    "dinov2_mean": "DINOv2 (mean patch)",
    "dinov2_cls_mean": "DINOv2 (CLS + mean)",
    "orb_bovw": "ORB bag of words",
    "hsv": "HSV histogram",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--sequences", nargs="*", help="default: splits.dev")
    ap.add_argument("--windows", type=int, nargs="*", default=[1, 2, 3, 5])
    ap.add_argument("--pca", type=int, nargs="*", default=[0, 64, 256], help="0 = no PCA")
    ap.add_argument("--rho-k", type=int, nargs="*", default=[8, 32, 128])
    ap.add_argument("--no-dinov2-mean", action="store_true")
    ap.add_argument("--out", default="results/probes")
    ap.add_argument("--assets", default="assets/improvements")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--figures-only", action="store_true", help="redraw from saved tables")
    args = ap.parse_args()
    if args.figures_only:
        out = resolve_path(args.out)
        rho = pd.read_csv(out / "rho.csv", dtype={"sequence": str})
        figures(
            pd.read_csv(out / "probes.csv"),
            rho,
            sorted(rho.sequence.unique()),
            args.rho_k,
            resolve_path(args.assets),
        )
        return 0

    over = [*args.set, f"benchmark.output_dir='{args.out}'"]
    config = load_config(args.config, over)
    setup_logging(config["logging"]["level"])
    dev = [str(s) for s in config["splits"]["dev"]]
    seqs = args.sequences or dev
    if not set(seqs) <= set(dev) or len(seqs) < 2:
        ap.error(f"probes run on >= 2 development sequences only ({dev}); got {seqs}")
    sel = json.loads(
        (
            resolve_path(load_config(args.config, args.set)["benchmark"]["output_dir"])
            / config["benchmark"]["name"]
            / "selection.json"
        ).read_text()
    )
    variant = {e: sel[e]["variant"] for e in sel}
    bench = Benchmark(config)
    rcfg = RetrievalConfig.from_dict({**config["retrieval"], "mode": "offline"})
    t0 = time.perf_counter()

    emb: dict[str, dict[str, np.ndarray]] = {}
    for e in ("ijepa", "dinov2", "orb_bovw", "hsv"):
        emb[e] = {s: bench.descriptors(s, e, variant[e])["embeddings"] for s in seqs}
    if not args.no_dinov2_mean:
        mean_bench = Benchmark(load_config(args.config, [*over, "dinov2.pooling=mean"]))
        emb["dinov2_mean"] = {
            s: mean_bench.descriptors(s, "dinov2", variant["dinov2"])["embeddings"] for s in seqs
        }
        emb["dinov2_cls_mean"] = {
            s: concat_views(emb["dinov2"][s], emb["dinov2_mean"][s]) for s in seqs
        }

    rows, rho_rows = [], []
    per_query: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for e, per_seq in emb.items():
        # ---- rho: scene-explained fraction (PCA fitted on the other sequence)
        for s in seqs:
            other = next(o for o in seqs if o != s)
            S = bench.sequence(s)
            same = same_place_pairs(S["gt"].positives, S["dataset"].frame_ids, S["positions"])
            rand = random_pairs(len(per_seq[s]), 20000, seed=0)
            r = {"encoder": e, "sequence": s, "pairs": len(same), "dim": per_seq[s].shape[1]}
            r["rho_full"] = scene_explained_fraction(per_seq[s], same, rand)
            for k in args.rho_k:
                if k < per_seq[s].shape[1]:
                    _, basis = pca_basis(per_seq[other], k)
                    r[f"rho_top{k}"] = scene_explained_fraction(per_seq[s], same, rand, basis)
            rho_rows.append(r)
        # ---- retrieval grid: causal frame averaging x PCA truncation
        base_hits = None
        for w in args.windows:
            for k in args.pca:
                if k and k >= per_seq[seqs[0]].shape[1]:
                    continue
                hits, aps = [], []
                for s in seqs:
                    S = bench.sequence(s)
                    x = causal_average(per_seq[s], w)
                    if k:
                        other = next(o for o in seqs if o != s)
                        mu, basis = pca_basis(causal_average(emb[e][other], w), k)
                        x = project(x, mu, basis)
                    res = retrieve(S["dataset"].frame_ids, S["dataset"].timestamps(), x, rcfg)
                    pos = S["gt"].positives
                    has = S["gt"].has_positive
                    hits.append(
                        np.array(
                            [np.isin(res.candidate_ids[i, 0], pos[i]) for i in range(len(pos))]
                        )[has]
                    )
                    a = retrieval_average_precision(
                        x, S["dataset"].frame_ids, res.eligible_counts, pos
                    )
                    aps.append(a[np.isfinite(a)])
                h, a = np.concatenate(hits), np.concatenate(aps)
                per_query[f"{e}|w{w}|k{k}"] = (h, a)
                row = {
                    "encoder": e,
                    "window": w,
                    "pca": k,
                    "queries": int(h.size),
                    "recall_at_1": float(h.mean()),
                    "map": float(a.mean()),
                }
                if w == 1 and k == 0:
                    base_hits = h
                elif base_hits is not None:
                    row.update({f"vs_base_{k2}": v for k2, v in mcnemar(base_hits, h).items()})
                rows.append(row)
                print(
                    f"{e:16s} w={w} pca={k:3d}  R@1 {row['recall_at_1']:.3f}  mAP {row['map']:.3f}"
                )

    def compare(new: str, base: str) -> dict:
        (hn, an), (hb, ab) = per_query[new], per_query[base]
        return {
            "new": new,
            "base": base,
            "recall_at_1": [float(hb.mean()), float(hn.mean())],
            "map": [float(ab.mean()), float(an.mean())],
            "r1_mcnemar": mcnemar(hb, hn),
            "ap_wilcoxon_p": float(wilcoxon(an, ab).pvalue) if np.any(an != ab) else 1.0,
        }

    comparisons = [
        compare(key, key.split("|")[0] + "|w1|k0")
        for key in per_query
        if not key.endswith("|w1|k0")
    ]
    for new in ("dinov2_mean", "dinov2_cls_mean", "ijepa"):
        if f"{new}|w1|k0" in per_query and new != "dinov2":
            comparisons.append(compare(f"{new}|w1|k0", "dinov2|w1|k0"))
    for c in comparisons:
        print(
            f"{c['new']:24s} vs {c['base']:18s} R@1 {c['recall_at_1'][0]:.3f}->"
            f"{c['recall_at_1'][1]:.3f} (+{c['r1_mcnemar']['new_only']}/-"
            f"{c['r1_mcnemar']['base_only']}, p={c['r1_mcnemar']['p']:.3g})  mAP "
            f"{c['map'][0]:.3f}->{c['map'][1]:.3f} (Wilcoxon p={c['ap_wilcoxon_p']:.3g})"
        )
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "per_query.npz",
        **{f"{k}|hits": v[0] for k, v in per_query.items()},
        **{f"{k}|ap": v[1] for k, v in per_query.items()},
    )
    df, rho = pd.DataFrame(rows), pd.DataFrame(rho_rows)
    df.to_csv(out / "probes.csv", index=False)
    rho.to_csv(out / "rho.csv", index=False)
    summary = {
        "sequences": seqs,
        "variants": variant,
        "seconds": round(time.perf_counter() - t0, 1),
        "device_note": "descriptors from the Phase 6 caches (GPU FP32 for the neural encoders; "
        "DINOv2 mean-patch extracted here on the GPU); probes on the CPU",
        "rho": rho_rows,
        "retrieval": rows,
        "comparisons": comparisons,
    }
    (out / "probes.json").write_text(json.dumps(summary, indent=2, default=float))

    figures(df, rho, seqs, args.rho_k, resolve_path(args.assets))
    print(json.dumps({"seconds": summary["seconds"], "out": str(out)}))
    return 0


def figures(df: pd.DataFrame, rho: pd.DataFrame, seqs, rho_k, assets) -> None:
    """``rho_by_encoder.png`` and ``frame_averaging.png`` from the saved tables."""
    apply_style()
    pooled = rho.groupby("encoder", sort=False).mean(numeric_only=True)
    cols = ["rho_full"] + [f"rho_top{k}" for k in rho_k]
    fig, ax = plt.subplots(figsize=(9.5, 4))
    xs = np.arange(len(pooled))
    width = 0.8 / len(cols)
    for n, c in enumerate(cols):
        name = "full space" if c == "rho_full" else c.replace("rho_top", "top-") + " PCs"
        ax.bar(xs + (n - (len(cols) - 1) / 2) * width, pooled[c], width, label=name)
    ax.set_xticks(xs, [LABEL.get(e, e).replace(" (", "\n(") for e in pooled.index], fontsize=8)
    ax.set_ylabel("scene-explained fraction rho (axis from 0)")
    ax.set_ylim(0, 1)
    ax.legend(title="PCA fitted on the other dev sequence", fontsize=8, ncols=4, loc="upper left")
    ax.set_title(
        "rho = 1 - mean ||f(x)-f(x')||^2 over revisits / mean over random pairs  "
        f"(KITTI {' + '.join(seqs)}, development)",
        fontsize=9,
    )
    save_figure(fig, assets / "rho_by_encoder.png")

    encs = [e for e in ("ijepa", "dinov2", "dinov2_mean") if e in set(df.encoder)]
    fig, axs = plt.subplots(1, len(encs), figsize=(4 * len(encs), 3.8), sharey=True)
    for ax, e in zip(np.atleast_1d(axs), encs, strict=True):
        d = df[(df.encoder == e) & (df.pca == 0)]
        ax.plot(d.window, d.recall_at_1, marker="o", label="Recall@1")
        ax.plot(d.window, d["map"], marker="s", label="mAP")
        for _, r in d.iterrows():
            ax.annotate(
                f"{r['map']:.2f}",
                (r.window, r["map"]),
                xytext=(0, -12),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        ax.set_title(LABEL[e], fontsize=9)
        ax.set_xlabel("causal window w (frames averaged, stride 5)")
        ax.set_xticks(sorted(d.window.unique()))
        ax.set_ylim(0.55, 0.95)
    np.atleast_1d(axs)[0].set_ylabel("pooled development value (axis starts at 0.55)")
    np.atleast_1d(axs)[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        f"Causal frame averaging, no PCA (KITTI {' + '.join(seqs)}, development)", fontsize=10
    )
    save_figure(fig, assets / "frame_averaging.png")


if __name__ == "__main__":
    raise SystemExit(main())
