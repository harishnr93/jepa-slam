"""Phase 5 artefacts: baseline comparison figures and the side-by-side retrieval GIF.

Reads, for every method, the cached descriptors and the *saved* retrieval result written
by ``scripts/run_retrieval.py --encoder <name>``; nothing is re-extracted here, so a
missing method raises with the command to run. Encoder costs come from
``results/metrics/encoder_profile_<dataset>.json`` (``scripts/profile_encoders.py``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from latent_loop.datasets import build_dataset
from latent_loop.datasets.ground_truth import pair_labels, revisit_ground_truth
from latent_loop.encoders import build_encoder
from latent_loop.evaluation.metrics import recall_at_k
from latent_loop.features.extraction import dataset_cache_key, make_cache
from latent_loop.retrieval import RetrievalConfig, RetrievalResult, assert_causal
from latent_loop.utils.config import resolve_path
from latent_loop.utils.logging import get_logger
from latent_loop.visualization.animations import save_gif, select_animation_indices
from latent_loop.visualization.baselines import (
    METHODS,
    MethodRun,
    choose_comparison_cases,
    plot_cost_comparison,
    plot_qualitative,
    plot_recall_comparison,
    plot_side_by_side,
    plot_similarity_comparison,
    render_comparison_frames,
    similarity_stats,
)
from latent_loop.visualization.plots import save_figure
from latent_loop.visualization.retrieval import COLOR_FALSE, COLOR_TRUE

logger = get_logger(__name__)


def load_method_runs(config: dict[str, Any], dataset, methods=METHODS) -> list[MethodRun]:
    """Cached descriptors + saved retrieval result per method (common interface)."""
    rcfg = RetrievalConfig.from_dict(config["retrieval"])
    key = dataset_cache_key(dataset)
    cache_root = resolve_path(
        config.get("extraction", {}).get("cache_root", "data/cache/embeddings")
    )
    results = resolve_path(config["paths"]["results_dir"])
    profile_path = results / "metrics" / f"encoder_profile_{key}.json"
    profile = json.loads(profile_path.read_text())["encoders"] if profile_path.is_file() else {}
    runs = []
    for name in methods:
        enc = build_encoder(name, config)
        cached = make_cache(cache_root, dataset, enc).load()
        rdir = results / "retrieval" / key / f"{enc.name}-{enc.fingerprint()}" / rcfg.mode
        hint = (
            f"python scripts/run_retrieval.py --encoder {name} --stride "
            f"{dataset.sampling.stride} (with the same --set)"
        )
        if cached is None or not (rdir / "retrieval.npz").is_file():
            raise RuntimeError(f"No cached descriptors / retrieval result for {name}: run {hint}")
        result = RetrievalResult.load(rdir)
        if not np.array_equal(result.query_frame_ids, dataset.frame_ids) or result.config != rcfg:
            raise RuntimeError(
                f"Saved {name} retrieval does not match these frames/settings: {hint}"
            )
        assert_causal(result)
        run = MethodRun(name, cached.select(dataset.frame_ids), result, profile.get(name, {}))
        run.fingerprint = enc.fingerprint()
        runs.append(run)
        enc.unload()
    return runs


def generate_phase5(
    config: dict[str, Any],
    out_dir: Path,
    vis_stride: int = 2,
    gif_frames: int = 100,
    gif_fps: float = 8.0,
    gif_width: int = 900,
    gif_colors: int = 256,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    config = json.loads(json.dumps(config))
    sampling = dict(config["dataset"].get("sampling") or {})
    config["dataset"]["sampling"] = {**sampling, "stride": vis_stride * sampling.get("stride", 1)}
    dataset = build_dataset(config)
    if not dataset.has_poses:
        raise RuntimeError("Phase 5 figures need ground-truth poses for labelling")
    key = dataset_cache_key(dataset)
    label = "SYNTHETIC" if dataset.name == "synthetic" else f"KITTI {dataset.sequence}"
    runs = load_method_runs(config, dataset)
    poses = dataset.poses()
    positions = poses[:, :3, 3]
    gtc = config.get("ground_truth", {})
    radius = float(gtc.get("spatial_threshold_m", 5.0))
    heading = float(gtc.get("max_heading_diff_deg", 30.0))
    excl = runs[0].result.config.temporal_exclusion_frames or 0
    gt = revisit_ground_truth(poses, dataset.frame_ids, radius, excl)
    gt_dir = revisit_ground_truth(poses, dataset.frame_ids, radius, excl, heading)
    labels = pair_labels(positions, dataset.frame_ids, radius, 25.0, excl)
    stem = out_dir / key
    outputs: dict[str, str] = {}
    get_image = dataset.get_image

    fig = plot_recall_comparison(
        runs,
        [("distance-only (<= 5 m)", gt), (f"same direction (<= 5 m, <= {heading:g} deg)", gt_dir)],
        f"Preliminary Recall@K, identical protocol for every method ({label}, top-10, "
        f"exclusion {excl} frames)",
    )
    outputs["recall_comparison"] = str(save_figure(fig, Path(f"{stem}_recall_comparison.png")))
    fig = plot_similarity_comparison(
        runs,
        gt,
        labels,
        f"Embedding similarity per method ({label}; eligible pairs, labels <= 5 m / > 25 m)",
    )
    outputs["similarity_comparison"] = str(
        save_figure(fig, Path(f"{stem}_similarity_comparison.png"))
    )
    cases = choose_comparison_cases(runs, gt)
    fig = plot_qualitative(
        runs,
        gt,
        positions,
        get_image,
        cases,
        f"Qualitative successes and failures ({label}; top-1 per method, green = true revisit)",
    )
    outputs["qualitative"] = str(save_figure(fig, Path(f"{stem}_qualitative.png")))
    split = [c for c in cases if "I-JEPA" in c[1]][:1]  # I-JEPA and the baselines disagree
    side = ([cases[0]] if cases else []) + [c for c in split if c not in cases[:1]]
    fig = plot_side_by_side(
        runs,
        gt,
        positions,
        get_image,
        side,
        f"Side-by-side top-3 retrieval ({label}; green = true revisit, red = not)",
    )
    outputs["side_by_side"] = str(save_figure(fig, Path(f"{stem}_side_by_side.png")))
    ks = (1, 5, 10)
    recall = {
        r.name: {
            "distance_only": recall_at_k(r.result.candidate_ids, gt.positives, ks),
            "same_direction": recall_at_k(r.result.candidate_ids, gt_dir.positives, ks),
        }
        for r in runs
    }
    if all(r.card for r in runs):
        fig = plot_cost_comparison(
            runs,
            {n: v["distance_only"][1] for n, v in recall.items()},
            f"Computational trade-offs (encoder profile on {label} frames; I/O excluded)",
        )
        outputs["cost_comparison"] = str(save_figure(fig, Path(f"{stem}_cost_comparison.png")))

    start = time.perf_counter()
    steps = select_animation_indices(len(dataset), gif_frames)
    frames = render_comparison_frames(
        runs, gt, positions, get_image, steps, f"Side-by-side retrieval ({label})", gif_width
    )
    gif = save_gif(
        frames,
        Path(f"{stem}_baseline_retrieval.gif"),
        fps=gif_fps,
        reserve_colors=[r.color for r in runs] + [COLOR_TRUE, COLOR_FALSE],
        colors=gif_colors,
    )
    outputs["comparison_gif"] = str(gif)

    summary = {
        "phase": 5,
        "dataset_key": key,
        "queries": int(len(dataset)),
        "frame_stride": int(config["dataset"]["sampling"]["stride"]),
        "positive_queries": {
            "distance_only": int(gt.has_positive.sum()),
            "same_direction": int(gt_dir.has_positive.sum()),
        },
        "recall_at_k_preliminary": recall,
        "similarity": {r.name: similarity_stats(r, labels) for r in runs},
        "retrieval_config": runs[0].result.config.__dict__,
        "descriptor_fingerprints": {r.name: r.fingerprint for r in runs},
        "encoder_profile": {r.name: r.card for r in runs},
        "cases": [{"query": int(dataset.frame_ids[i]), "case": c} for i, c in cases],
        "gif_frames": len(frames),
        "gif_width_px": gif_width,
        "gif_palette_colors": gif_colors,
        "gif_size_mb": round(gif.stat().st_size / 1e6, 2),
        "gif_render_seconds": round(time.perf_counter() - start, 1),
        "outputs": {k: Path(v).name for k, v in outputs.items()},
    }
    (out_dir / f"{key}_phase5_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
