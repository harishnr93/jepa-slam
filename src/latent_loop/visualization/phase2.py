"""Phase 2 artefacts: I-JEPA similarity matrix, descriptor statistics, patch similarity,
preprocessing comparison and the similarity-evolution GIF.

Descriptors come from the embedding cache (they are extracted first if missing), so
every figure reflects saved pipeline outputs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from latent_loop.datasets import build_dataset
from latent_loop.datasets.ground_truth import NEGATIVE, POSITIVE, pair_labels
from latent_loop.encoders import build_encoder
from latent_loop.features.extraction import dataset_cache_key, extract_embeddings, make_cache
from latent_loop.utils.config import resolve_path
from latent_loop.utils.logging import get_logger
from latent_loop.visualization.animations import save_gif, select_animation_indices
from latent_loop.visualization.embeddings import (
    plot_descriptor_distribution,
    plot_patch_similarity,
    plot_preprocessing_comparison,
    plot_similarity_matrix,
    render_similarity_evolution_frames,
    separability,
)
from latent_loop.visualization.plots import save_figure

logger = get_logger(__name__)

GT = {"positive_radius_m": 5.0, "negative_radius_m": 25.0, "min_gap_frames": 100}


def _extract(dataset, encoder, config) -> tuple[np.ndarray, dict[str, Any]]:
    ext = config.get("extraction", {})
    cache = make_cache(
        resolve_path(ext.get("cache_root", "data/cache/embeddings")), dataset, encoder
    )
    result = extract_embeddings(
        dataset,
        encoder,
        cache,
        batch_size=int(ext.get("batch_size", 8)),
        cpu_fallback=bool(ext.get("cpu_fallback", True)),
    )
    return result.embeddings, {
        "cache_dir": str(result.cache_dir),
        "encoded_now": result.encoded_now,
        "profile": result.profile,
    }


def generate_phase2(
    config: dict[str, Any],
    out_dir: Path,
    vis_stride: int = 2,
    gif_frames: int = 120,
    gif_fps: float = 10.0,
    gif_width: int = 900,
    compare_preprocessing: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    sampling = dict(config["dataset"].get("sampling") or {})
    config = json.loads(json.dumps(config))
    config["dataset"]["sampling"] = {**sampling, "stride": vis_stride * sampling.get("stride", 1)}
    dataset = build_dataset(config)
    if not dataset.has_poses:
        raise RuntimeError("Phase 2 diagnostics need ground-truth poses for labelling")
    encoder = build_encoder(config.get("encoder", "ijepa"), config)
    tag = dataset_cache_key(dataset)
    label = "SYNTHETIC" if dataset.name == "synthetic" else f"KITTI {dataset.sequence}"
    enc_label = (
        f"I-JEPA {encoder.info().get('arch', '')} ({encoder.info().get('checkpoint')}, "
        f"{encoder.info().get('resize_mode')} {encoder.info().get('image_size')} px)"
    )

    emb, extraction = _extract(dataset, encoder, config)
    frame_ids = dataset.frame_ids
    poses = dataset.poses()
    positions = poses[:, :3, 3]
    timestamps = dataset.timestamps()
    labels = pair_labels(positions, frame_ids, **GT)
    sim = emb @ emb.T
    stats = separability(sim, labels)
    outputs: dict[str, str] = {}
    title = f"{label} — {enc_label}"

    fig = plot_similarity_matrix(
        sim,
        frame_ids,
        labels,
        f"Embedding similarity matrix: {title}",
        GT["min_gap_frames"],
        GT["positive_radius_m"],
    )
    outputs["similarity_matrix"] = str(save_figure(fig, out_dir / f"{tag}_similarity_matrix.png"))

    fig = plot_descriptor_distribution(
        emb,
        sim,
        labels,
        positions,
        frame_ids,
        timestamps,
        f"Descriptor distribution: {title}",
        **GT,
    )
    outputs["descriptor_distribution"] = str(
        save_figure(fig, out_dir / f"{tag}_descriptor_distribution.png")
    )

    # Patch-level similarity for: a GT revisit, the hardest negative, a typical negative.
    rng = np.random.default_rng(0)
    qi, rj = np.nonzero(labels == POSITIVE)
    k = int(np.argmin(np.linalg.norm(positions[qi] - positions[rj], axis=1)))
    ni, nj = np.nonzero(labels == NEGATIVE)
    hard = int(np.argmax(sim[ni, nj]))
    typical = int(rng.integers(ni.size))
    chosen = [
        (qi[k], rj[k], "GT revisit"),
        (ni[hard], nj[hard], "hardest negative"),
        (ni[typical], nj[typical], "random negative"),
    ]
    images = {i: dataset.get_image(int(i)) for pair in chosen for i in pair[:2]}
    tokens, grid = encoder.encode_tokens([images[i] for pair in chosen for i in pair[:2]])
    pairs = []
    for n, (q, r, kind) in enumerate(chosen):
        pairs.append(
            {
                "query_id": int(frame_ids[q]),
                "ref_id": int(frame_ids[r]),
                "label": kind,
                "query_image": images[q],
                "ref_image": images[r],
                "query_tokens": tokens[2 * n].numpy(),
                "ref_tokens": tokens[2 * n + 1].numpy(),
                "distance_m": float(np.linalg.norm(positions[q, [0, 2]] - positions[r, [0, 2]])),
                "global_cos": float(sim[q, r]),
            }
        )
    calib_w, calib_h = dataset.get_calibration().image_size
    fig = plot_patch_similarity(
        pairs, grid, f"Feature similarity heatmaps: {title}", display_size=(calib_w, calib_h)
    )
    outputs["patch_similarity"] = str(save_figure(fig, out_dir / f"{tag}_patch_similarity.png"))

    # Preprocessing comparison (same frames, same weights; only the resize mode changes).
    comparison = []
    if compare_preprocessing and hasattr(encoder, "resize_mode"):
        original = encoder.resize_mode
        for mode in ("squash", "center_crop", "aspect"):
            encoder.resize_mode = mode
            mode_emb = emb if mode == original else _extract(dataset, encoder, config)[0]
            shape = tuple(encoder.preprocess(dataset.get_image(0)).shape[-2:])
            comparison.append(
                {
                    "resize_mode": mode,
                    "input": f"{shape[1]}×{shape[0]}",
                    **separability(mode_emb @ mode_emb.T, labels),
                }
            )
        encoder.resize_mode = original
        fig = plot_preprocessing_comparison(
            comparison, f"Preprocessing comparison ({label}, diagnostic only — not a benchmark)"
        )
        outputs["preprocessing"] = str(
            save_figure(fig, out_dir / f"{tag}_preprocessing_comparison.png")
        )
    encoder.unload()

    steps = select_animation_indices(len(dataset), gif_frames)
    start = time.perf_counter()
    frames = render_similarity_evolution_frames(
        dataset.get_image,
        sim,
        labels,
        frame_ids,
        timestamps,
        steps,
        GT["min_gap_frames"],
        f"I-JEPA similarity evolution ({label})",
        width_px=gif_width,
        positive_radius_m=GT["positive_radius_m"],
        negative_radius_m=GT["negative_radius_m"],
    )
    gif = save_gif(frames, out_dir / f"{tag}_similarity_evolution.gif", fps=gif_fps)
    outputs["similarity_gif"] = str(gif)

    summary = {
        "phase": 2,
        "dataset_key": tag,
        "frames": int(len(dataset)),
        "frame_stride": int(config["dataset"]["sampling"]["stride"]),
        "encoder": encoder.info(),
        "fingerprint": encoder.fingerprint(),
        "ground_truth": GT,
        "separability": stats,
        "preprocessing_comparison": comparison,
        "extraction": extraction,
        "patch_pairs": [
            {k: v for k, v in p.items() if not k.endswith(("image", "tokens"))} for p in pairs
        ],
        "gif_frames": len(frames),
        "gif_size_mb": round(gif.stat().st_size / 1e6, 2),
        "gif_render_seconds": round(time.perf_counter() - start, 1),
        "outputs": {k: Path(v).name for k, v in outputs.items()},
    }
    (out_dir / f"{tag}_phase2_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
