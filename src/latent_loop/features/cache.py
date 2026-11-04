"""Disk cache for global descriptors, keyed by dataset, sequence and encoder fingerprint.

Layout::

    <root>/<dataset_key>/<encoder>-<fingerprint>/
        embeddings.npy   # (N, D) float32, rows sorted by original frame ID
        frame_ids.npy    # (N,) int64 original frame IDs
        meta.json        # encoder info, embedding dimension, extraction runs

Rows are keyed by *original frame ID*, so caches built with different sampling share
entries and extraction is incremental: only frames that are not cached yet are
encoded. Every write is atomic (temporary file + ``os.replace``).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CachedEmbeddings:
    frame_ids: np.ndarray
    embeddings: np.ndarray
    meta: dict[str, Any]

    def select(self, frame_ids: np.ndarray) -> np.ndarray:
        """Rows for ``frame_ids`` in the given order; ``KeyError`` if any is missing."""
        frame_ids = np.asarray(frame_ids, dtype=np.int64)
        missing = np.setdiff1d(frame_ids, self.frame_ids)
        if missing.size:
            raise KeyError(f"{missing.size} frame IDs are not cached (first: {missing[:5]})")
        return self.embeddings[np.searchsorted(self.frame_ids, frame_ids)]


class EmbeddingCache:
    def __init__(self, root: str | Path, dataset_key: str, encoder_name: str, fingerprint: str):
        self.dir = Path(root) / dataset_key / f"{encoder_name}-{fingerprint}"
        self.fingerprint = fingerprint

    @property
    def exists(self) -> bool:
        return (self.dir / "meta.json").is_file()

    def load(self) -> CachedEmbeddings | None:
        if not self.exists:
            return None
        meta = json.loads((self.dir / "meta.json").read_text())
        if meta.get("fingerprint") != self.fingerprint:
            raise ValueError(
                f"Cache {self.dir} has fingerprint {meta.get('fingerprint')}, "
                f"expected {self.fingerprint}"
            )
        embeddings = np.load(self.dir / "embeddings.npy")
        frame_ids = np.load(self.dir / "frame_ids.npy")
        if embeddings.shape != (frame_ids.size, meta["embedding_dim"]):
            raise ValueError(
                f"Corrupt cache {self.dir}: embeddings {embeddings.shape}, "
                f"frame_ids {frame_ids.shape}"
            )
        return CachedEmbeddings(frame_ids, embeddings, meta)

    def missing(self, frame_ids: np.ndarray) -> np.ndarray:
        cached = self.load()
        frame_ids = np.asarray(frame_ids, dtype=np.int64)
        return frame_ids if cached is None else np.setdiff1d(frame_ids, cached.frame_ids)

    def add(
        self,
        frame_ids: np.ndarray,
        embeddings: np.ndarray,
        encoder_info: dict[str, Any],
        run_info: dict[str, Any] | None = None,
    ) -> CachedEmbeddings:
        """Merge rows into the cache; on duplicate frame IDs the new rows win."""
        frame_ids = np.asarray(frame_ids, dtype=np.int64)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != frame_ids.size:
            raise ValueError("embeddings must be (N, D) with one frame ID per row")
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("refusing to cache non-finite embeddings")
        old = self.load()
        added = frame_ids.size
        if old is None:
            meta = {
                "fingerprint": self.fingerprint,
                "encoder": encoder_info,
                "embedding_dim": int(embeddings.shape[1]),
                "runs": [],
            }
        else:
            if embeddings.shape[1] != old.meta["embedding_dim"]:
                raise ValueError("embedding dimension differs from the cached one")
            keep = ~np.isin(old.frame_ids, frame_ids)
            added = int(np.setdiff1d(frame_ids, old.frame_ids).size)
            frame_ids = np.concatenate([old.frame_ids[keep], frame_ids])
            embeddings = np.concatenate([old.embeddings[keep], embeddings])
            meta = old.meta
        order = np.argsort(frame_ids, kind="stable")
        frame_ids, embeddings = frame_ids[order], embeddings[order]
        meta["num_frames"] = int(frame_ids.size)
        meta["runs"].append(
            {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "added_frames": int(added),
                **(run_info or {}),
            }
        )
        self.dir.mkdir(parents=True, exist_ok=True)
        _atomic_save(self.dir / "embeddings.npy", embeddings)
        _atomic_save(self.dir / "frame_ids.npy", frame_ids)
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=2, default=str))
        os.replace(tmp, self.dir / "meta.json")
        return CachedEmbeddings(frame_ids, embeddings, meta)


def _atomic_save(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, array)
    os.replace(tmp, path)
