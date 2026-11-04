"""Embedding cache and batched extraction (incremental, OOM recovery, CPU fallback)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest
import torch

from conftest import SMALL_SYNTHETIC
from latent_loop.datasets import SamplingConfig, SyntheticLoopDataset
from latent_loop.encoders.base import VisualEncoder
from latent_loop.evaluation.profiling import Profiler
from latent_loop.features.cache import EmbeddingCache
from latent_loop.features.extraction import (
    dataset_cache_key,
    extract_embeddings,
    is_oom,
    make_cache,
)


class MeanColourEncoder(VisualEncoder):
    """Cheap deterministic encoder: normalised mean RGB plus a constant channel."""

    name = "meancolour"

    def __init__(self, oom_above: int | None = None, oom_always: bool = False) -> None:
        self.calls: list[int] = []
        self.oom_above = oom_above
        self.oom_always = oom_always
        self.device = "cuda"

    def load_model(self) -> None: ...

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return np.atleast_3d(image).reshape(-1, np.atleast_3d(image).shape[-1]).mean(0)

    def encode_batch(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if self.device == "cuda" and (
            self.oom_always or (self.oom_above is not None and len(images) > self.oom_above)
        ):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        self.calls.append(len(images))
        x = np.stack([np.append(self.preprocess(im), 50.0) for im in images]).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def embedding_dim(self) -> int:
        return 4

    def to_device(self, device: str, precision: str | None = None) -> None:
        self.device = device

    def info(self):
        return {"name": self.name, "runtime": {"device": self.device}}


@pytest.fixture
def dataset() -> SyntheticLoopDataset:
    return SyntheticLoopDataset(SMALL_SYNTHETIC, SamplingConfig(stride=4, max_frames=10))


def test_cache_roundtrip_and_incremental_merge(tmp_path):
    cache = EmbeddingCache(tmp_path, "ds", "enc", "abc")
    assert cache.load() is None
    np.testing.assert_array_equal(cache.missing(np.array([3, 1])), [3, 1])
    cache.add(np.array([5, 1]), np.eye(2, dtype=np.float32), {"name": "enc"})
    cached = cache.add(np.array([3, 5]), np.array([[0.6, 0.8], [0.8, 0.6]]), {"name": "enc"})
    np.testing.assert_array_equal(cached.frame_ids, [1, 3, 5])
    np.testing.assert_allclose(cached.select(np.array([5, 1])), [[0.8, 0.6], [0, 1]])
    reloaded = cache.load()
    np.testing.assert_array_equal(reloaded.frame_ids, [1, 3, 5])
    assert [r["added_frames"] for r in reloaded.meta["runs"]] == [2, 1]
    np.testing.assert_array_equal(cache.missing(np.array([1, 2, 3])), [2])
    with pytest.raises(KeyError):
        reloaded.select(np.array([2]))
    with pytest.raises(ValueError, match="dimension"):
        cache.add(np.array([9]), np.ones((1, 3)), {})
    with pytest.raises(ValueError, match="non-finite"):
        cache.add(np.array([9]), np.full((1, 2), np.nan), {})


def test_cache_rejects_foreign_fingerprint(tmp_path):
    EmbeddingCache(tmp_path, "ds", "enc", "abc").add(np.array([0]), np.ones((1, 2)), {})
    other = EmbeddingCache(tmp_path, "ds", "enc", "abc")
    other.fingerprint = "zzz"
    with pytest.raises(ValueError, match="fingerprint"):
        other.load()


def test_extraction_is_incremental_and_ordered(tmp_path, dataset):
    enc = MeanColourEncoder()
    cache = make_cache(tmp_path, dataset, enc)
    first = extract_embeddings(dataset, enc, cache, batch_size=4)
    assert first.encoded_now == 10 and enc.calls == [4, 4, 2]
    np.testing.assert_array_equal(first.frame_ids, dataset.frame_ids)
    expected = enc.encode_batch([dataset.get_image(i) for i in range(len(dataset))])
    np.testing.assert_allclose(first.embeddings, expected, atol=1e-6)

    again = extract_embeddings(dataset, enc, cache, batch_size=4)
    assert again.encoded_now == 0  # everything served from the cache
    np.testing.assert_array_equal(again.embeddings, first.embeddings)

    denser = SyntheticLoopDataset(SMALL_SYNTHETIC, SamplingConfig(stride=2, max_frames=20))
    more = extract_embeddings(denser, enc, cache, batch_size=8)
    assert more.encoded_now == 10  # only the new odd multiples of 2 are encoded
    assert cache.load().meta["num_frames"] == 20


def test_extraction_without_cache(dataset):
    result = extract_embeddings(dataset, MeanColourEncoder(), None, batch_size=3)
    assert result.embeddings.shape == (10, 4) and result.cache_dir is None


def test_oom_halves_batch_size(dataset):
    enc = MeanColourEncoder(oom_above=2)
    reference = extract_embeddings(dataset, MeanColourEncoder(), None, batch_size=8)
    result = extract_embeddings(dataset, enc, None, batch_size=8)
    assert result.final_batch_size == 2
    assert result.fallbacks == ["oom->batch_size=4", "oom->batch_size=2"]
    assert max(enc.calls) == 2 and sum(enc.calls) == 10
    np.testing.assert_allclose(result.embeddings, reference.embeddings)


def test_oom_at_batch_one_falls_back_to_cpu(dataset):
    enc = MeanColourEncoder(oom_always=True)
    result = extract_embeddings(dataset, enc, None, batch_size=2)
    assert result.fallbacks == ["oom->batch_size=1", "oom->cpu_fp32"]
    assert enc.device == "cpu" and result.embeddings.shape == (10, 4)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        extract_embeddings(dataset, MeanColourEncoder(oom_always=True), None, 1, cpu_fallback=False)


def test_non_oom_errors_propagate(dataset):
    class Broken(MeanColourEncoder):
        def encode_batch(self, images):
            raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        extract_embeddings(dataset, Broken(), None)
    assert is_oom(RuntimeError("CUDA error: out of memory"))
    assert not is_oom(ValueError("out of memory"))


def test_dataset_cache_key_tracks_synthetic_config(dataset, small_synthetic):
    assert dataset_cache_key(dataset) == dataset_cache_key(small_synthetic)  # sampling ignored
    other = SyntheticLoopDataset(replace(SMALL_SYNTHETIC, seed=99))
    assert dataset_cache_key(other) != dataset_cache_key(small_synthetic)


def test_profiler_summary():
    prof = Profiler()
    for _ in range(3):
        with prof.stage("work", items=2, sync_cuda=False):
            sum(range(1000))
    summary = prof.summary()
    assert summary["work"]["calls"] == 3 and summary["work"]["items"] == 6
    assert summary["peak_cpu_rss_mb"] > 0
