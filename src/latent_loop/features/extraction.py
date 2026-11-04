"""Batched, cached, OOM-tolerant descriptor extraction over a dataset.

Algorithm:

1. Ask the cache which of the dataset's original frame IDs are not encoded yet.
2. Load the next batch of images in a background thread (lazy disk/render I/O overlaps
   with GPU compute) and encode the current batch under ``torch.inference_mode``.
3. On CUDA out-of-memory: release cached blocks, halve the batch size and retry the
   same frames; at batch size 1, optionally move the encoder to the CPU in FP32.
4. Persist new rows every ``flush_every`` frames, so an interrupted run resumes.
5. Return the descriptors of exactly the dataset's frames, in dataset order.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from latent_loop.datasets.base import VisualSequenceDataset
from latent_loop.encoders.base import VisualEncoder
from latent_loop.evaluation.profiling import Profiler
from latent_loop.features.cache import EmbeddingCache
from latent_loop.utils.logging import get_logger

logger = get_logger(__name__)


def dataset_cache_key(dataset: VisualSequenceDataset) -> str:
    """Stable identifier of the *pixels* a dataset produces (not of its sampling)."""
    if dataset.name == "kitti":
        return f"kitti_{dataset.sequence}_cam{dataset.camera}"
    config = getattr(dataset, "config", None)
    if config is not None:
        digest = hashlib.sha1(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
        return f"{dataset.name}_{digest[:8]}"
    return dataset.name


def make_cache(
    root: str | Path, dataset: VisualSequenceDataset, encoder: VisualEncoder
) -> EmbeddingCache:
    return EmbeddingCache(root, dataset_cache_key(dataset), encoder.name, encoder.fingerprint())


def is_oom(exc: BaseException) -> bool:
    try:
        import torch  # noqa: PLC0415

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _empty_cuda_cache() -> None:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@dataclass
class ExtractionResult:
    frame_ids: np.ndarray
    embeddings: np.ndarray  # (N, D) float32, L2-normalised, dataset order
    profile: dict[str, Any] = field(default_factory=dict)
    encoded_now: int = 0
    final_batch_size: int = 0
    fallbacks: list[str] = field(default_factory=list)
    cache_dir: Path | None = None


def extract_embeddings(
    dataset: VisualSequenceDataset,
    encoder: VisualEncoder,
    cache: EmbeddingCache | None = None,
    batch_size: int = 8,
    cpu_fallback: bool = True,
    flush_every: int = 256,
    log_every: int = 100,
) -> ExtractionResult:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    ids = dataset.frame_ids
    todo = cache.missing(ids) if cache is not None else ids
    view_index = {int(f): i for i, f in enumerate(ids)}
    profiler = Profiler()
    results: dict[int, np.ndarray] = {}
    pending_ids: list[np.ndarray] = []
    pending_emb: list[np.ndarray] = []
    fallbacks: list[str] = []
    done = 0

    def load(batch_ids: np.ndarray) -> list[np.ndarray]:
        return [dataset.get_image(view_index[int(f)]) for f in batch_ids]

    def flush() -> None:
        if cache is not None and pending_ids:
            info = encoder.info()
            cache.add(
                np.concatenate(pending_ids),
                np.concatenate(pending_emb),
                info,
                {"batch_size": batch_size, **info.get("runtime", {}), "fallbacks": list(fallbacks)},
            )
            pending_ids.clear()
            pending_emb.clear()

    if todo.size:
        encoder.load_model()
        logger.info(
            "Extracting embeddings",
            extra={
                "encoder": encoder.name,
                "to_encode": int(todo.size),
                "cached": int(ids.size - todo.size),
                "batch_size": batch_size,
            },
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        start = 0
        future = pool.submit(load, todo[:batch_size]) if todo.size else None
        while start < todo.size:
            batch_ids = todo[start : start + batch_size]
            with profiler.stage("image_load_wait", batch_ids.size, sync_cuda=False):
                images = future.result()[: batch_ids.size]  # shorter after an OOM retry
            try:
                with profiler.stage("encode", batch_ids.size):
                    emb = encoder.encode_batch(images)
            except Exception as exc:  # only CUDA OOM is recovered, everything else re-raises
                if not is_oom(exc):
                    raise
                _empty_cuda_cache()
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    fallbacks.append(f"oom->batch_size={batch_size}")
                    logger.warning("CUDA OOM; halving batch size", extra={"batch_size": batch_size})
                elif cpu_fallback:
                    fallbacks.append("oom->cpu_fp32")
                    logger.warning("CUDA OOM at batch size 1; falling back to CPU fp32")
                    encoder.to_device("cpu", "fp32")
                else:
                    raise
                future = pool.submit(lambda imgs=images: imgs)  # retry the same frames
                continue
            next_start = start + batch_ids.size
            if next_start < todo.size:
                future = pool.submit(load, todo[next_start : next_start + batch_size])
            results.update({int(f): e for f, e in zip(batch_ids, emb, strict=True)})
            pending_ids.append(batch_ids)
            pending_emb.append(emb)
            if sum(p.size for p in pending_ids) >= flush_every:
                flush()
            if log_every and done // log_every != (done + batch_ids.size) // log_every:
                logger.info(
                    "Extraction progress",
                    extra={"done": done + batch_ids.size, "total": int(todo.size)},
                )
            done += batch_ids.size
            start = next_start
    flush()

    if cache is not None:
        cached = cache.load()
        embeddings = (
            cached.select(ids)
            if cached is not None
            else np.zeros((0, encoder.embedding_dim()), np.float32)
        )
    else:
        embeddings = np.stack([results[int(f)] for f in ids])
    profile = profiler.summary()
    return ExtractionResult(
        ids,
        embeddings,
        profile,
        encoded_now=done,
        final_batch_size=batch_size,
        fallbacks=fallbacks,
        cache_dir=cache.dir if cache is not None else None,
    )
