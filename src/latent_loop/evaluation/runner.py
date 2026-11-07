"""Phase 6 benchmark runner: development selection, then one held-out evaluation.

Protocol (enforced in code):

1. **Development stage** (``splits.dev``):
   * Descriptors and retrieval for every method and every preprocessing *variant* of the
     neural encoders (``benchmark.variants``).
   * Variant selection per encoder by pooled development retrieval mAP -> ``selection.json``.
   * Geometric verification of the selected methods' top-K candidates.
   * Threshold selection per method on the pooled development PR curve
     (maximum F1; maximum recall at 100 % precision is also recorded) -> ``thresholds.json``.
2. **Test stage** (``splits.test``): reads ``selection.json`` and ``thresholds.json``
   (refuses to run without them), computes everything once with the frozen choices and
   reports metrics at the development thresholds.

Ground truth is used only for labels, metrics and the development selections.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latent_loop.datasets import build_dataset
from latent_loop.datasets.ground_truth import revisit_ground_truth
from latent_loop.encoders import build_encoder
from latent_loop.evaluation.benchmark import (
    Predictions,
    detection_at,
    make_predictions,
    pr_curve,
    retrieval_average_precision,
    select_threshold,
)
from latent_loop.evaluation.metrics import recall_at_k
from latent_loop.evaluation.probes import causal_average
from latent_loop.features.extraction import dataset_cache_key, extract_embeddings, make_cache
from latent_loop.retrieval import RetrievalConfig, RetrievalResult, retrieve
from latent_loop.utils.config import resolve_path
from latent_loop.utils.logging import get_logger
from latent_loop.verification import VerificationConfig
from latent_loop.verification.parallel import config_hash, pairs_from_candidates, verify_pairs

logger = get_logger(__name__)


def _copy(cfg: dict) -> dict:
    return json.loads(json.dumps(cfg))


def sequence_config(config: dict, sequence: str) -> dict:
    b = config["benchmark"]
    cfg = _copy(config)
    cfg["dataset"].update(
        {
            "name": "kitti",
            "root": b.get("dataset_root", "data/raw/kitti/dataset"),
            "sequence": sequence,
            "camera": int(b.get("camera", 2)),
            "sampling": {"stride": int(b.get("stride", 5))},
        }
    )
    return cfg


def window_suffix(window: int) -> str:
    """Retrieval-folder suffix of a frame-averaging window (none for 1)."""
    return f"-w{window}" if window > 1 else ""


def parse_variant(variant: str | None) -> dict[str, str]:
    """``"aspect"`` (Phase 6: a resize mode) or ``"pooling=mean,resize_mode=aspect,window=2"``."""
    if variant is None:
        return {}
    if "=" not in variant:
        return {"resize_mode": variant}
    return dict(part.split("=", 1) for part in variant.split(","))


def variant_list(b: dict, encoder: str) -> list[str | None]:
    """Candidate variants of one encoder, in config order (the first wins ties).

    ``benchmark.variant_grid.<encoder>`` (a mapping of key -> values) is expanded as a
    product; otherwise ``benchmark.variants.<encoder>`` (resize modes) is used.
    Keys: ``checkpoint``, ``resize_mode`` and ``pooling`` (encoder settings) and
    ``window`` (causal frame averaging of the descriptors, a post-processing step; 1 = off).
    """
    grid = (b.get("variant_grid") or {}).get(encoder)
    if not grid:
        return list((b.get("variants") or {}).get(encoder, [None]))
    unknown = set(grid) - {"checkpoint", "resize_mode", "pooling", "window"}
    if unknown:
        raise ValueError(f"unknown variant keys for {encoder}: {sorted(unknown)}")
    keys = list(grid)
    return [
        ",".join(f"{k}={v}" for k, v in zip(keys, combo, strict=True))
        for combo in itertools.product(*(grid[k] for k in keys))
    ]


def with_variant(cfg: dict, encoder: str, variant: str | None) -> dict:
    cfg = _copy(cfg)
    for k, v in parse_variant(variant).items():
        if k != "window":
            cfg.setdefault(encoder, {})[k] = v
    return cfg


class Benchmark:
    def __init__(self, config: dict) -> None:
        self.config = config
        b = config["benchmark"]
        self.b = b
        self.out = resolve_path(b.get("output_dir", "results/benchmark")) / b.get("name", "default")
        self.out.mkdir(parents=True, exist_ok=True)
        self.methods: dict[str, dict] = b["methods"]
        self.variants: dict[str, list[str]] = b.get("variants", {})
        self.verify_top_k = int(b.get("verify_top_k", 5))
        rc = dict(config["retrieval"])
        rc["mode"] = "causal"  # identical candidates to offline; gives per-query timing
        self.rcfg = RetrievalConfig.from_dict(rc)
        vdict = {**config.get("verification", {}), **b.get("verification_overrides", {})}
        self.vcfg = VerificationConfig.from_dict(vdict)
        self.workers = int(b.get("workers", 4))
        gt = config.get("ground_truth", {})
        self.radius = float(gt.get("spatial_threshold_m", 5.0))
        self.heading = float(gt.get("max_heading_diff_deg", 30.0))
        self._seq: dict[str, Any] = {}
        self._desc: dict[tuple, dict] = {}

    # ------------------------------------------------------------------ per sequence
    def sequence(self, seq: str) -> dict[str, Any]:
        if seq not in self._seq:
            cfg = sequence_config(self.config, seq)
            ds = build_dataset(cfg)
            poses = ds.poses()
            excl = self.rcfg.temporal_exclusion_frames or 0
            self._seq[seq] = {
                "config": cfg,
                "dataset": ds,
                "key": dataset_cache_key(ds),
                "positions": poses[:, :3, 3],
                "gt": revisit_ground_truth(poses, ds.frame_ids, self.radius, excl),
                "gt_dir": revisit_ground_truth(
                    poses, ds.frame_ids, self.radius, excl, self.heading
                ),
            }
        return self._seq[seq]

    def descriptors(self, seq: str, encoder: str, variant: str | None) -> dict[str, Any]:
        key = (seq, encoder, variant)
        if key not in self._desc:
            self._desc[key] = self._descriptors(seq, encoder, variant)
        return self._desc[key]

    def _descriptors(self, seq: str, encoder: str, variant: str | None) -> dict[str, Any]:
        s = self.sequence(seq)
        cfg = with_variant(s["config"], encoder, variant)
        enc = build_encoder(encoder, cfg)
        ext = cfg.get("extraction", {})
        cache = make_cache(
            resolve_path(ext.get("cache_root", "data/cache/embeddings")), s["dataset"], enc
        )
        res = extract_embeddings(s["dataset"], enc, cache, batch_size=int(ext.get("batch_size", 8)))
        if res.encoded_now:
            report = {
                "dataset_key": s["key"],
                "encoder": enc.info(),
                "fingerprint": enc.fingerprint(),
                "frames": int(res.frame_ids.size),
                "encoded_now": res.encoded_now,
                "profile": res.profile,
            }
            path = (
                resolve_path("results/metrics")
                / f"extraction_{s['key']}_{enc.name}-{enc.fingerprint()}.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, default=str))
        fp = enc.fingerprint()
        info = enc.info()
        enc.unload()
        window = int(parse_variant(variant).get("window", 1))
        emb = res.embeddings if window == 1 else causal_average(res.embeddings, window)
        rdir = self.out / "retrieval" / s["key"] / f"{encoder}-{fp}{window_suffix(window)}"
        result = retrieve(s["dataset"].frame_ids, s["dataset"].timestamps(), emb, self.rcfg)
        result.save(rdir)
        return {
            "embeddings": emb,
            "result": result,
            "fingerprint": fp,
            "info": info,
            "variant": variant,
            "window": window if "window" in parse_variant(variant) else None,
        }

    def retrieval_metrics(self, seq: str, d: dict) -> dict[str, Any]:
        s, r = self.sequence(seq), d["result"]
        ap = retrieval_average_precision(
            d["embeddings"], s["dataset"].frame_ids, r.eligible_counts, s["gt"].positives
        )
        ks = (1, 5, 10)
        return {
            "recall": recall_at_k(r.candidate_ids, s["gt"].positives, ks),
            "recall_same_direction": recall_at_k(r.candidate_ids, s["gt_dir"].positives, ks),
            "ap_per_query": ap,
            "map": float(np.nanmean(ap)) if np.isfinite(ap).any() else float("nan"),
            "positive_queries": int(s["gt"].has_positive.sum()),
            "positive_queries_same_direction": int(s["gt_dir"].has_positive.sum()),
        }

    def verified(self, seq: str, result: RetrievalResult) -> tuple[pd.DataFrame, pd.DataFrame]:
        s = self.sequence(seq)
        pairs = pairs_from_candidates(
            result.query_frame_ids, result.candidate_ids, self.verify_top_k
        )
        return verify_pairs(
            pairs, s["config"], self.vcfg, self.out / "pair_cache", s["key"], workers=self.workers
        )

    def predictions(self, seq: str, method: str, d: dict) -> tuple[Predictions, dict]:
        s, r = self.sequence(seq), d["result"]
        extra: dict[str, Any] = {}
        if not self.methods[method].get("verify"):
            cand, score = r.top1_ids, r.top1_scores
        else:
            pairs, orb = self.verified(seq, r)
            pairs["rank"] = pairs.groupby("query_frame_id").cumcount() + 1
            ok = pairs[pairs.accepted].sort_values(
                ["query_frame_id", "n_inliers", "rank"], ascending=[True, False, True]
            )
            best = ok.groupby("query_frame_id").head(1).set_index("query_frame_id")
            ids = pd.Index(r.query_frame_ids)
            cand = best.candidate_frame_id.reindex(ids).fillna(-1).astype(int).to_numpy()
            score = best.n_inliers.reindex(ids).astype(float).to_numpy()
            # verification latency per query: ORB of the query frame + all its pair checks
            pair_ms = (
                pairs.reindex(columns=["ms_match", "ms_ransac", "ms_pose"]).fillna(0).sum(axis=1)
            )  # stages a pair never reached cost nothing
            per_q = pair_ms.groupby(pairs.query_frame_id).sum()
            orb_ms = orb.set_index("frame_id").orb_ms
            q_ms = per_q.add(orb_ms.reindex(per_q.index).fillna(0))
            extra = {"verify_ms_per_query": q_ms, "pairs": pairs}
        pred = make_predictions(
            r.query_frame_ids, cand, score, s["gt"].positives, r.eligible_counts, s["positions"]
        )
        return pred, extra

    # ------------------------------------------------------------------ stages
    def encoder_of(self, method: str) -> str:
        return self.methods[method]["encoder"]

    def run_dev(self) -> dict[str, Any]:
        dev = [f"{int(x):02d}" for x in self.config["splits"]["dev"]]
        t0 = time.perf_counter()
        # 1. variant selection by pooled development mAP
        selection: dict[str, Any] = {}
        for enc in sorted({self.encoder_of(m) for m in self.methods}):
            scores = {}
            for v in variant_list(self.b, enc):
                aps = []
                for seq in dev:
                    rm = self.retrieval_metrics(seq, self.descriptors(seq, enc, v))
                    aps.append(rm["ap_per_query"][np.isfinite(rm["ap_per_query"])])
                scores[str(v)] = float(np.concatenate(aps).mean())
            best = max(scores, key=scores.get)  # first listed wins ties
            selection[enc] = {"variant": None if best == "None" else best, "dev_map": scores}
            logger.info(
                "Variant selected on development data",
                extra={"encoder": enc, **scores, "chosen": best},
            )
        (self.out / "selection.json").write_text(json.dumps(selection, indent=2))
        # 2. thresholds on pooled development predictions
        rows, thresholds, curves, pooled_preds = [], {}, {}, {}
        for m in self.methods:
            preds = []
            for seq in dev:
                d = self.descriptors(
                    seq, self.encoder_of(m), selection[self.encoder_of(m)]["variant"]
                )
                p, _ = self.predictions(seq, m, d)
                self._save_predictions("dev", m, seq, p)
                preds.append(p)
                rows.append(self._row(m, seq, d, p, None))
            pooled_preds[m] = Predictions.concat(preds)
            curves[m] = pr_curve(pooled_preds[m])
            thresholds[m] = select_threshold(curves[m])
        for row in rows:  # dev metrics at the selected (in-sample) thresholds, for reference
            t = thresholds[row["method"]]["f1"]
            row.update(self._det_fields(row.pop("_pred"), t))
        for m in self.methods:
            row = {"method": m, "sequence": "pooled", **self._pooled_retrieval(m, rows)}
            row["detection_ap"] = curves[m].ap
            row.update(self._det_fields(pooled_preds[m], thresholds[m]["f1"]))
            rows.append(row)
        self._write_split("dev", rows, curves, thresholds)
        (self.out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
        logger.info("Development stage done", extra={"seconds": round(time.perf_counter() - t0, 1)})
        return {"selection": selection, "thresholds": thresholds}

    def run_test(self) -> dict[str, Any]:
        sel_path, thr_path = self.out / "selection.json", self.out / "thresholds.json"
        if not sel_path.is_file() or not thr_path.is_file():
            raise RuntimeError(
                "Run the development stage first: selections and thresholds come from it"
            )
        selection = json.loads(sel_path.read_text())
        thresholds = json.loads(thr_path.read_text())
        test = [f"{int(x):02d}" for x in self.config["splits"]["test"]]
        overlap = set(test) & {f"{int(x):02d}" for x in self.config["splits"]["dev"]}
        if overlap:
            raise ValueError(f"test and development splits overlap: {sorted(overlap)}")
        rows, curves, timing = [], {}, {}
        for m in self.methods:
            preds, ms_v, search, add, frames = [], [], [], [], 0
            for seq in test:
                d = self.descriptors(
                    seq, self.encoder_of(m), selection[self.encoder_of(m)]["variant"]
                )
                p, extra = self.predictions(seq, m, d)
                self._save_predictions("test", m, seq, p)
                preds.append(p)
                row = self._row(m, seq, d, p, thresholds[m]["f1"])
                row.update(self._det_fields(row.pop("_pred"), thresholds[m]["f1"]))
                rows.append(row)
                n = d["result"].query_frame_ids.size
                frames += n
                search.append(d["result"].timing["search_s"] * 1000)
                add.append(d["result"].timing["index_build_s"] * 1000)
                if "verify_ms_per_query" in extra:
                    ms_v.append(extra["verify_ms_per_query"].to_numpy())
                timing.setdefault(m, {"fingerprint": d["fingerprint"], "info": d["info"]})
            pooled = Predictions.concat(preds)
            curves[m] = pr_curve(pooled)
            row = {"method": m, "sequence": "pooled", **self._pooled_retrieval(m, rows)}
            row["detection_ap"] = curves[m].ap
            row.update(self._det_fields(pooled, thresholds[m]["f1"]))
            p100 = detection_at(pooled, thresholds[m]["p100"])
            row["recall_at_dev_p100_threshold"], row["precision_at_dev_p100_threshold"] = (
                p100.recall,
                p100.precision,
            )
            rows.append(row)
            timing[m].update(
                {
                    "frames": frames,
                    "search_ms_per_query": float(sum(search) / frames),
                    "index_add_ms_per_frame": float(sum(add) / frames),
                    "verify_ms_per_query_mean": float(np.concatenate(ms_v).mean()) if ms_v else 0.0,
                    "verify_ms_per_query_median": float(np.median(np.concatenate(ms_v)))
                    if ms_v
                    else 0.0,
                }
            )
        self._write_split("test", rows, curves, thresholds)
        (self.out / "test_timing.json").write_text(json.dumps(timing, indent=2, default=str))
        prof = self.b.get("encoder_profile")
        eff = efficiency_table(self.out, resolve_path(prof) if prof else None)
        return {"rows": rows, "timing": timing, "efficiency": eff}

    # ------------------------------------------------------------------ helpers
    def _row(
        self, method: str, seq: str, d: dict, pred: Predictions, threshold: float | None
    ) -> dict:
        rm = self.retrieval_metrics(seq, d)
        return {
            "method": method,
            "sequence": seq,
            "variant": d["variant"] if d.get("variant") else d["info"].get("resize_mode"),
            "fingerprint": d["fingerprint"],
            **({"window": d["window"]} if d.get("window") is not None else {}),
            "queries": int(pred.evaluable.sum()),
            "positive_queries": rm["positive_queries"],
            "recall_at_1": rm["recall"][1],
            "recall_at_5": rm["recall"][5],
            "recall_at_10": rm["recall"][10],
            "recall_at_1_same_direction": rm["recall_same_direction"][1],
            "positive_queries_same_direction": rm["positive_queries_same_direction"],
            "map": rm["map"],
            "_ap": rm["ap_per_query"],
            "_recall_hits": {k: rm["recall"][k] * rm["positive_queries"] for k in (1, 5, 10)},
            "detection_ap": pr_curve(pred).ap,
            "_pred": pred,
        }

    def _save_predictions(self, split: str, method: str, seq: str, p: Predictions) -> None:
        pdir = self.out / "predictions" / split
        pdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({f: getattr(p, f) for f in p.__dataclass_fields__}).to_csv(
            pdir / f"{method}_{seq}.csv", index=False
        )

    @staticmethod
    def _det_fields(pred: Predictions, threshold: float) -> dict:
        m = detection_at(pred, threshold)
        out = m.to_dict()
        out["confusion"] = json.dumps(out["confusion"])
        return {f"det_{k}" if k not in ("threshold",) else k: v for k, v in out.items()}

    @staticmethod
    def _pooled_retrieval(method: str, rows: list[dict]) -> dict:
        mine = [r for r in rows if r["method"] == method and r["sequence"] != "pooled"]
        npos = sum(r["positive_queries"] for r in mine)
        aps = np.concatenate([r["_ap"][np.isfinite(r["_ap"])] for r in mine])
        out = {
            "queries": sum(r["queries"] for r in mine),
            "positive_queries": npos,
            "variant": mine[0]["variant"],
            **({"window": mine[0]["window"]} if "window" in mine[0] else {}),
            "fingerprint": ",".join(sorted({r["fingerprint"] for r in mine})),
            "map": float(aps.mean()),
        }
        for k in (1, 5, 10):
            out[f"recall_at_{k}"] = sum(r["_recall_hits"][k] for r in mine) / npos
        nsd = sum(r["positive_queries_same_direction"] for r in mine)
        hits = sum(
            r["recall_at_1_same_direction"] * r["positive_queries_same_direction"]
            for r in mine
            if r["positive_queries_same_direction"]
        )
        out["positive_queries_same_direction"] = nsd
        out["recall_at_1_same_direction"] = hits / nsd if nsd else float("nan")
        return out

    def _write_split(self, split: str, rows: list[dict], curves: dict, thresholds: dict) -> None:
        clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        df = pd.DataFrame(clean)
        df.to_csv(self.out / f"{split}_metrics.csv", index=False)
        (self.out / f"{split}_metrics.json").write_text(
            json.dumps(
                {
                    "rows": clean,
                    "thresholds": thresholds,
                    "verification_config_hash": config_hash(self.vcfg),
                    "verification_config": self.vcfg.to_dict(),
                    "retrieval_config": asdict(self.rcfg),
                },
                indent=2,
                default=str,
            )
        )
        cdir = self.out / "curves" / split
        cdir.mkdir(parents=True, exist_ok=True)
        for m, c in curves.items():
            np.savez(
                cdir / f"{m}.npz",
                thresholds=c.thresholds,
                precision=c.precision,
                recall=c.recall,
                f1=c.f1,
                fp=c.fp,
                ap=c.ap,
                positive_queries=c.positive_queries,
                queries=c.queries,
            )


def efficiency_table(out_dir: str | Path, profile_path: str | Path | None) -> pd.DataFrame:
    """Per-method computational measurements on the test sequences.

    Encoder latency and memory come from ``scripts/profile_encoders.py`` (same frames in RAM,
    warm-up, batch 8) matched by descriptor fingerprint; when no matching profile exists the
    extraction-time measurement (``results/metrics/extraction_*``) is used and marked.
    Retrieval latency is the causal per-query search plus index insertion; verification
    latency is ORB of the query frame plus the checks of its top-K pairs (CPU).
    """
    out_dir = Path(out_dir)
    timing = json.loads((out_dir / "test_timing.json").read_text())
    profile = {}
    if profile_path and Path(profile_path).is_file():
        for name, card in json.loads(Path(profile_path).read_text())["encoders"].items():
            profile[card.get("fingerprint")] = {"encoder": name, **card}
    rows = []
    for method, t in timing.items():
        info, fp = t["info"], t["fingerprint"]
        card = profile.get(fp)
        if card is not None:
            enc_ms, source = card["ms_per_image"], "profile"
            gpu, rss = card.get("peak_gpu_mb"), card.get("rss_increase_mb")
        else:
            reports = list(
                resolve_path("results/metrics").glob(f"extraction_*_{info['name']}-{fp}.json")
            )
            ms = [json.loads(p.read_text())["profile"]["encode"]["ms_per_item"] for p in reports]
            enc_ms, source = (float(np.mean(ms)) if ms else float("nan")), "extraction"
            gpu, rss = None, None
        dim = int(info.get("embedding_dim", 0))
        retrieval_ms = t["search_ms_per_query"] + t["index_add_ms_per_frame"]
        rows.append(
            {
                "method": method,
                "device": info.get("runtime", {}).get("device"),
                "precision": info.get("runtime", {}).get("precision"),
                "encoder_ms_per_frame": enc_ms,
                "encoder_latency_source": source,
                "retrieval_ms_per_query": retrieval_ms,
                "verification_ms_per_query": t["verify_ms_per_query_mean"],
                "verification_ms_per_query_median": t["verify_ms_per_query_median"],
                "end_to_end_ms_per_query": enc_ms + retrieval_ms + t["verify_ms_per_query_mean"],
                "peak_gpu_mb": gpu,
                "cpu_rss_increase_mb": rss,
                "embedding_dim": dim,
                "embedding_bytes_per_frame": 4 * dim,
                "embedding_storage_mb_test": 4 * dim * t["frames"] / 2**20,
                "test_frames": t["frames"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "efficiency.csv", index=False)
    (out_dir / "efficiency.json").write_text(df.to_json(orient="records", indent=2))
    return df
