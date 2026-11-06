"""Parallel, cached verification of arbitrary (query, candidate) frame pairs.

Used by the Phase 6 benchmark, where the top-K candidates of several methods on several
sequences must be verified. Each worker process builds the dataset from its config and
keeps its own ORB :class:`FeatureStore`; pairs are grouped by query so a worker reuses the
query's features. Results are cached per dataset and verification-config hash in a CSV,
so pairs shared by several methods (or re-runs) are verified once.

Verification is deterministic (seeded RANSAC), so the worker count does not change results.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latent_loop.utils.logging import get_logger
from latent_loop.verification.geometry import VerificationConfig, verify_features

logger = get_logger(__name__)

_STATE: dict[str, Any] = {}


def config_hash(config: VerificationConfig) -> str:
    return hashlib.sha1(
        json.dumps(config.to_dict(), sort_keys=True, default=str).encode()
    ).hexdigest()[:10]


def _init(dataset_config: dict, vcfg: dict) -> None:
    from latent_loop.datasets import build_dataset  # noqa: PLC0415
    from latent_loop.verification.pipeline import FeatureStore  # noqa: PLC0415

    dataset = build_dataset(dataset_config)
    config = VerificationConfig.from_dict(vcfg)
    _STATE.update(
        dataset=dataset,
        config=config,
        calibration=dataset.get_calibration(),
        store=FeatureStore(dataset.image_for_frame_id, config.orb, capacity=512),
    )


def _verify_chunk(pairs: Sequence[tuple[int, int]]) -> tuple[list[dict], dict[int, float]]:
    store, config, calib = _STATE["store"], _STATE["config"], _STATE["calibration"]
    rows: list[dict] = []
    orb_ms: dict[int, float] = {}
    for q, c in pairs:
        for f in (q, c):
            if f not in orb_ms:
                before = store.extract_seconds
                store(f)
                orb_ms[f] = 1000 * (store.extract_seconds - before)
        res = verify_features(store(q), store(c), calib, config)
        rows.append({"query_frame_id": int(q), "candidate_frame_id": int(c), **res.record()})
    return rows, orb_ms


def verify_pairs(
    pairs: Sequence[tuple[int, int]],
    dataset_config: dict,
    config: VerificationConfig,
    cache_dir: str | Path,
    dataset_key: str,
    workers: int = 4,
    chunk: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify ``pairs`` (cached). Returns (pair results, per-frame ORB times in ms).

    ORB time is measured when a frame's features are first computed in a worker
    (image read + ORB + twin fraction). A frame computed in several workers keeps its
    first measurement.
    """
    cache_dir = Path(cache_dir) / dataset_key / config_hash(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pair_csv, orb_csv = cache_dir / "pairs.csv", cache_dir / "orb_ms.csv"
    (cache_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2, default=str))
    have = (
        pd.read_csv(pair_csv, dtype={"degenerate": str}).fillna({"degenerate": ""})
        if pair_csv.is_file()
        else pd.DataFrame(columns=["query_frame_id", "candidate_frame_id"])
    )
    orb = (
        pd.read_csv(orb_csv) if orb_csv.is_file() else pd.DataFrame(columns=["frame_id", "orb_ms"])
    )
    done = set(
        zip(have.query_frame_id.astype(int), have.candidate_frame_id.astype(int), strict=True)
    )
    todo = sorted({(int(q), int(c)) for q, c in pairs} - done)
    if todo:
        start = time.perf_counter()
        chunks = [todo[i : i + chunk] for i in range(0, len(todo), chunk)]
        logger.info(
            "Verifying pairs",
            extra={"dataset": dataset_key, "pairs": len(todo), "workers": workers},
        )
        rows, orb_ms = [], {}
        if workers <= 1:
            _init(dataset_config, config.to_dict())
            results = map(_verify_chunk, chunks)
        else:
            pool = ProcessPoolExecutor(
                workers,
                mp.get_context("spawn"),
                initializer=_init,
                initargs=(dataset_config, config.to_dict()),
            )
            results = pool.map(_verify_chunk, chunks)
        for n, (r, o) in enumerate(results):
            rows.extend(r)
            for f, ms in o.items():
                orb_ms.setdefault(f, ms)
            if (n + 1) % 25 == 0:
                logger.info("Verification progress", extra={"pairs": len(rows), "of": len(todo)})
        if workers > 1:
            pool.shutdown()
        new = pd.DataFrame(rows)
        have = pd.concat([have, new], ignore_index=True) if len(have) else new
        new_orb = pd.DataFrame({"frame_id": list(orb_ms), "orb_ms": list(orb_ms.values())})
        orb = (
            pd.concat([orb, new_orb], ignore_index=True).drop_duplicates("frame_id")
            if len(orb)
            else new_orb
        )
        have.to_csv(pair_csv, index=False)
        orb.to_csv(orb_csv, index=False)
        logger.info(
            "Pairs verified",
            extra={
                "dataset": dataset_key,
                "pairs": len(todo),
                "seconds": round(time.perf_counter() - start, 1),
            },
        )
    keys = pd.MultiIndex.from_tuples(
        [(int(q), int(c)) for q, c in pairs], names=["query_frame_id", "candidate_frame_id"]
    )
    out = have.set_index(["query_frame_id", "candidate_frame_id"]).reindex(keys).reset_index()
    if out.accepted.isna().any():
        raise RuntimeError("some pairs are missing from the verification cache")
    out["accepted"] = out.accepted.astype(bool)
    return out, orb


def pairs_from_candidates(
    query_ids: np.ndarray, candidate_ids: np.ndarray, top_k: int
) -> list[tuple[int, int]]:
    out = []
    for q, row in zip(query_ids, candidate_ids[:, :top_k], strict=True):
        out.extend((int(q), int(c)) for c in row if c >= 0)
    return out
