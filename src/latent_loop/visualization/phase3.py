"""Phase 3 artefacts: top-K galleries, candidate trajectory, eligible similarity matrix,
Recall@K curves and the chronological retrieval GIF.

Everything is drawn from *saved* retrieval results (``results/retrieval/...``) and the
Phase 2 embedding cache. If the retrieval result is missing it is computed with the same
deterministic code path (``latent_loop.retrieval.retrieve``) and saved first.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from latent_loop.datasets import build_dataset
from latent_loop.datasets.ground_truth import revisit_ground_truth
from latent_loop.encoders import build_encoder
from latent_loop.evaluation.metrics import recall_at_k
from latent_loop.features.extraction import dataset_cache_key, extract_embeddings, make_cache
from latent_loop.retrieval import RetrievalConfig, RetrievalResult, assert_causal, retrieve
from latent_loop.utils.config import resolve_path
from latent_loop.utils.logging import get_logger
from latent_loop.visualization.animations import save_gif, select_animation_indices
from latent_loop.visualization.plots import save_figure
from latent_loop.visualization.retrieval import (
    choose_example_queries,
    plot_candidate_trajectory,
    plot_recall_curves,
    plot_retrieval_matrix,
    plot_topk_gallery,
    render_retrieval_frames,
)

logger = get_logger(__name__)


def generate_phase3(
    config: dict[str, Any],
    out_dir: Path,
    vis_stride: int = 2,
    gif_frames: int = 120,
    gif_fps: float = 8.0,
    gif_width: int = 900,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    config = json.loads(json.dumps(config))
    sampling = dict(config["dataset"].get("sampling") or {})
    config["dataset"]["sampling"] = {**sampling, "stride": vis_stride * sampling.get("stride", 1)}
    dataset = build_dataset(config)
    if not dataset.has_poses:
        raise RuntimeError("Phase 3 figures need ground-truth poses for labelling")
    encoder = build_encoder(config.get("encoder", "ijepa"), config)
    rcfg = RetrievalConfig.from_dict(config["retrieval"])
    key = dataset_cache_key(dataset)
    label = "SYNTHETIC" if dataset.name == "synthetic" else f"KITTI {dataset.sequence}"
    result_dir = (
        resolve_path(config["paths"]["results_dir"])
        / "retrieval"
        / key
        / f"{encoder.name}-{encoder.fingerprint()}"
        / rcfg.mode
    )

    ext = config.get("extraction", {})
    cache = make_cache(
        resolve_path(ext.get("cache_root", "data/cache/embeddings")), dataset, encoder
    )
    emb = extract_embeddings(
        dataset, encoder, cache, batch_size=int(ext.get("batch_size", 8))
    ).embeddings
    encoder.unload()
    loaded = (result_dir / "retrieval.npz").is_file()
    if loaded:
        result = RetrievalResult.load(result_dir)
        stale = (
            not np.array_equal(result.query_frame_ids, dataset.frame_ids) or result.config != rcfg
        )
        if stale:
            loaded = False
    if not loaded:
        logger.info(
            "No matching saved retrieval result; computing it", extra={"dir": str(result_dir)}
        )
        result = retrieve(dataset.frame_ids, dataset.timestamps(), emb, rcfg)
        result.save(result_dir)
    assert_causal(result)

    poses = dataset.poses()
    positions = poses[:, :3, 3]
    gt_cfg = config.get("ground_truth", {})
    radius = float(gt_cfg.get("spatial_threshold_m", 5.0))
    heading = gt_cfg.get("max_heading_diff_deg", 30.0)
    excl = rcfg.temporal_exclusion_frames or 0
    gt = revisit_ground_truth(poses, dataset.frame_ids, radius, excl)
    gt_dir = revisit_ground_truth(poses, dataset.frame_ids, radius, excl, heading)
    params = (
        f"top-{rcfg.top_k}, exclusion {rcfg.temporal_exclusion_frames} frames, "
        f"revisit ≤ {radius:g} m"
    )
    outputs: dict[str, str] = {}

    picks = choose_example_queries(result, gt)
    fig = plot_topk_gallery(
        result,
        gt,
        positions,
        dataset.get_image,
        picks,
        f"I-JEPA top-5 retrieval ({label}; {params})",
    )
    outputs["topk_gallery"] = str(save_figure(fig, out_dir / f"{key}_topk_gallery.png"))

    fig = plot_candidate_trajectory(
        result, gt, positions, f"Loop-closure candidates on the trajectory ({label}; {params})"
    )
    outputs["candidate_trajectory"] = str(
        save_figure(fig, out_dir / f"{key}_candidate_trajectory.png")
    )

    fig = plot_retrieval_matrix(
        emb @ emb.T, result, gt, f"Eligible similarity matrix and top-1 choices ({label})"
    )
    outputs["retrieval_matrix"] = str(save_figure(fig, out_dir / f"{key}_retrieval_matrix.png"))

    fig = plot_recall_curves(
        [
            ("revisit ≤ 5 m", result.candidate_ids, gt),
            (f"revisit ≤ 5 m and heading ≤ {heading:g}°", result.candidate_ids, gt_dir),
        ],
        f"Recall@K, preliminary ({label}, I-JEPA, {rcfg.mode} retrieval)",
        rcfg.top_k,
    )
    outputs["recall_curve"] = str(save_figure(fig, out_dir / f"{key}_recall_at_k.png"))

    steps = select_animation_indices(len(dataset), gif_frames)
    start = time.perf_counter()
    frames = render_retrieval_frames(
        result,
        gt,
        positions,
        dataset.get_image,
        steps,
        f"I-JEPA causal retrieval ({label})",
        width_px=gif_width,
    )
    gif = save_gif(frames, out_dir / f"{key}_retrieval.gif", fps=gif_fps)
    outputs["retrieval_gif"] = str(gif)

    ks = [k for k in (1, 5, 10) if k <= rcfg.top_k]
    summary = {
        "phase": 3,
        "dataset_key": key,
        "queries": int(len(dataset)),
        "frame_stride": int(config["dataset"]["sampling"]["stride"]),
        "retrieval_config": rcfg.__dict__,
        "retrieval_result_dir": str(result_dir),
        "loaded_saved_result": loaded,
        "timing": result.timing,
        "recall_at_k_preliminary": {
            "distance_only": recall_at_k(result.candidate_ids, gt.positives, ks),
            "same_direction": recall_at_k(result.candidate_ids, gt_dir.positives, ks),
        },
        "positive_queries": {
            "distance_only": int(gt.has_positive.sum()),
            "same_direction": int(gt_dir.has_positive.sum()),
        },
        "example_queries": [
            {"frame_id": int(result.query_frame_ids[i]), "case": c} for i, c in picks
        ],
        "gif_frames": len(frames),
        "gif_size_mb": round(gif.stat().st_size / 1e6, 2),
        "gif_render_seconds": round(time.perf_counter() - start, 1),
        "outputs": {k: Path(v).name for k, v in outputs.items()},
    }
    (out_dir / f"{key}_phase3_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
