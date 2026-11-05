"""Phase 4 artefacts: ORB keypoints, raw vs filtered matches, RANSAC inliers with epipolar
lines, accepted/rejected pair gallery, statistics, trajectory and the verification GIF.

Decisions come from the *saved* verification result
(``results/verification/.../top<K>_<estimator>/``,
written by ``scripts/run_verification.py``); it is computed first if missing. Pairs
shown in figures are re-verified with ``keep_details=True`` to obtain the matches and
inlier masks, and the re-run must reproduce the saved decision and inlier count exactly
(checked; recorded in the summary).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latent_loop.datasets import build_dataset
from latent_loop.datasets.ground_truth import NEGATIVE, POSITIVE
from latent_loop.encoders import build_encoder
from latent_loop.features.extraction import dataset_cache_key
from latent_loop.retrieval import RetrievalConfig, RetrievalResult
from latent_loop.utils.config import resolve_path
from latent_loop.utils.logging import get_logger
from latent_loop.verification import (
    FeatureStore,
    VerificationConfig,
    VerificationRun,
    label_pairs,
    raw_nearest_matches,
    run_dirname,
    summarize,
    verify_candidates,
    verify_features,
)
from latent_loop.visualization.animations import save_gif
from latent_loop.visualization.matches import (
    PairExample,
    plot_decision_gallery,
    plot_inliers_outliers,
    plot_orb_keypoints,
    plot_raw_vs_filtered,
    plot_verification_statistics,
    plot_verified_trajectory,
    render_verification_frames,
)
from latent_loop.visualization.plots import save_figure

logger = get_logger(__name__)


def choose_examples(pairs: pd.DataFrame) -> list[tuple[int, str]]:
    """(row label, case) for the gallery. Rank-1 pairs; deterministic."""
    top = pairs[pairs["rank"] == 1]
    pos, neg = top.gt_label == POSITIVE, top.gt_label == NEGATIVE
    picks: list[tuple[int, str]] = []

    def add(mask: pd.Series, case: str, by: str, which: str = "max") -> None:
        sub = top[mask & ~top.index.isin([p for p, _ in picks])]
        if sub.empty:
            return
        if which == "median":
            row = sub.iloc[(sub[by] - sub[by].median()).abs().argsort().iloc[0]]
            picks.append((int(row.name), case))
        else:
            picks.append((int(sub[by].idxmax()), case))

    add(pos & top.accepted, "True revisit, accepted (median inlier count)", "n_inliers", "median")
    add(neg & ~top.accepted, "Confident false candidate, rejected by geometry", "cosine")
    add(pos & ~top.accepted, "True revisit, rejected (missed)", "cosine")
    add(neg & top.accepted, "False candidate, accepted (false positive)", "n_inliers")
    add(top.accepted & (top.degenerate != ""), "Accepted, degenerate geometry flagged", "n_inliers")
    add(pos & top.accepted, "True revisit, accepted (most inliers)", "n_inliers")
    return picks


def _gif_rows(pairs: pd.DataFrame, picks: list[int], n: int) -> list[int]:
    top = pairs[(pairs["rank"] == 1) & (pairs.gt_label != -1)].sort_values("query_frame_id")
    even = top.index[np.linspace(0, len(top) - 1, min(n, len(top))).round().astype(int)]
    rows = sorted(set(even) | set(picks), key=lambda r: pairs.loc[r, "query_frame_id"])
    return rows[: n + len(picks)]


def generate_phase4(
    config: dict[str, Any],
    out_dir: Path,
    vis_stride: int = 2,
    gif_pairs: int = 12,
    gif_fps: float = 8.0,
    gif_width: int = 900,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    config = json.loads(json.dumps(config))
    sampling = dict(config["dataset"].get("sampling") or {})
    config["dataset"]["sampling"] = {**sampling, "stride": vis_stride * sampling.get("stride", 1)}
    dataset = build_dataset(config)
    if not dataset.has_poses:
        raise RuntimeError("Phase 4 figures need ground-truth poses for labelling")
    encoder = build_encoder(config.get("encoder", "ijepa"), config)
    rcfg = RetrievalConfig.from_dict(config["retrieval"])
    key = dataset_cache_key(dataset)
    label = "SYNTHETIC" if dataset.name == "synthetic" else f"KITTI {dataset.sequence}"
    sub = f"{key}/{encoder.name}-{encoder.fingerprint()}/{rcfg.mode}"
    results = resolve_path(config["paths"]["results_dir"])
    vdict = dict(config.get("verification", {}))
    verify_top_k = int(vdict.get("verify_top_k", 1))
    vcfg = VerificationConfig.from_dict(vdict)
    vdir = results / "verification" / sub / run_dirname(verify_top_k, vcfg.estimator)
    store = FeatureStore(dataset.image_for_frame_id, vcfg.orb)
    calib = dataset.get_calibration()

    loaded = (vdir / "verification.csv").is_file()
    if loaded:
        run = VerificationRun.load(vdir)
        if run.config != vcfg or run.verify_top_k != verify_top_k:
            loaded = False
    if not loaded:
        rdir = results / "retrieval" / sub
        if not (rdir / "retrieval.npz").is_file():
            raise RuntimeError(f"No retrieval result in {rdir}; run scripts/run_retrieval.py")
        logger.info("No matching saved verification result; computing it", extra={"dir": str(vdir)})
        run = verify_candidates(RetrievalResult.load(rdir), store, calib, vcfg, verify_top_k)
    positions = dataset.poses()[:, :3, 3]
    gt = config.get("ground_truth", {})
    pairs = label_pairs(
        run.pairs,
        dataset.frame_ids,
        positions,
        float(gt.get("spatial_threshold_m", 5.0)),
        float(gt.get("negative_radius_m", 25.0)),
    )
    if not loaded:
        run.pairs = pairs
        run.save(vdir, {"summary": summarize(pairs)})

    def example(row: int, case: str = "") -> PairExample:
        p = pairs.loc[row]
        q, c = int(p.query_frame_id), int(p.candidate_frame_id)
        fq, fr = store(q), store(c)
        res = verify_features(fq, fr, calib, vcfg, keep_details=True)
        if res.reason != p.reason or res.n_inliers != int(p.n_inliers):
            raise RuntimeError(
                f"Re-verification of ({q}, {c}) gave {res.reason}/{res.n_inliers}, "
                f"saved {p.reason}/{p.n_inliers}"
            )
        return PairExample(
            q,
            c,
            dataset.image_for_frame_id(q),
            dataset.image_for_frame_id(c),
            fq,
            fr,
            res,
            raw_nearest_matches(fq, fr),
            float(p.cosine),
            int(p["rank"]),
            int(p.gt_label),
            float(p.gt_distance_m),
            case,
            vcfg.matching.ratio,
            vcfg.matching.max_hamming,
        )

    picks = choose_examples(pairs)
    gallery = [example(r, c) for r, c in picks]
    outputs: dict[str, str] = {}
    main = gallery[0]
    stem = out_dir / key
    fig = plot_orb_keypoints(main, f"ORB keypoints of a verified pair ({label})")
    outputs["orb_keypoints"] = str(save_figure(fig, Path(f"{stem}_orb_keypoints.png")))
    fig = plot_raw_vs_filtered(main, f"Raw vs filtered ORB matches ({label}, {main.case.lower()})")
    outputs["raw_matches"] = str(save_figure(fig, Path(f"{stem}_raw_matches.png")))
    fig = plot_inliers_outliers(
        main, calib.K, f"Essential-matrix RANSAC on a true revisit ({label}): {main.verdict}"
    )
    outputs["ransac_inliers"] = str(save_figure(fig, Path(f"{stem}_ransac_inliers.png")))
    rejected = [g for g in gallery if not g.result.accepted and g.gt_label == NEGATIVE]
    if rejected:
        ex = rejected[0]
        fig = plot_inliers_outliers(
            ex, calib.K, f"RANSAC on a confident false candidate ({label}): {ex.verdict}"
        )
        outputs["ransac_rejected"] = str(save_figure(fig, Path(f"{stem}_ransac_rejected.png")))
    fig = plot_decision_gallery(
        gallery,
        f"Accepted and rejected candidate pairs ({label}; green border = accepted, "
        "red = rejected; ground truth only annotates)",
    )
    outputs["decision_gallery"] = str(save_figure(fig, Path(f"{stem}_decision_gallery.png")))
    params = f"min_inliers {vcfg.min_inliers}, min ratio {vcfg.min_inlier_ratio}"
    fig = plot_verification_statistics(
        pairs,
        vcfg.min_inliers,
        vcfg.min_inlier_ratio,
        f"Geometric verification of the top-{run.verify_top_k} I-JEPA candidates "
        f"({label}; {len(pairs)} pairs; {params})",
    )
    outputs["statistics"] = str(save_figure(fig, Path(f"{stem}_verification_stats.png")))
    fig = plot_verified_trajectory(
        pairs,
        positions,
        dataset.frame_ids,
        f"Top-1 candidates after geometric verification ({label})",
    )
    outputs["trajectory"] = str(save_figure(fig, Path(f"{stem}_verified_trajectory.png")))

    start = time.perf_counter()
    rows = _gif_rows(pairs, [r for r, _ in picks], gif_pairs)
    case_of = dict(picks)
    gif_examples = [example(r, case_of.get(r, "")) for r in rows]
    frames = render_verification_frames(
        gif_examples, positions, dataset.frame_ids, f"Geometric verification ({label})", gif_width
    )
    gif = save_gif(frames, Path(f"{stem}_verification.gif"), fps=gif_fps)
    outputs["verification_gif"] = str(gif)

    summary = {
        "phase": 4,
        "dataset_key": key,
        "queries": int(len(dataset)),
        "frame_stride": int(config["dataset"]["sampling"]["stride"]),
        "verify_top_k": run.verify_top_k,
        "verification_config": vcfg.to_dict(),
        "verification_dir": str(vdir),
        "loaded_saved_result": loaded,
        "timing": run.timing,
        "pairs": summarize(pairs),
        "top1": summarize(pairs[pairs["rank"] == 1]),
        "examples": [
            {
                "case": e.case,
                "query": e.query_frame_id,
                "candidate": e.candidate_frame_id,
                "reason": e.result.reason,
                "inliers": e.result.n_inliers,
                "reverified_identically": True,
            }
            for e in gallery
        ],
        "gif_pairs": len(gif_examples),
        "gif_frames": len(frames),
        "gif_size_mb": round(gif.stat().st_size / 1e6, 2),
        "gif_render_seconds": round(time.perf_counter() - start, 1),
        "outputs": {k: Path(v).name for k, v in outputs.items()},
    }
    (out_dir / f"{key}_phase4_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
