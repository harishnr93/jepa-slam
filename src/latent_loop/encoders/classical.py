"""Classical global descriptors: HSV colour histograms and ORB bag of visual words.

Both implement :class:`VisualEncoder`, so they share the embedding cache, FAISS
retrieval and evaluation code with the neural encoders.

**HSV histogram.** The image is resized to a fixed working width, converted to HSV
(OpenCV: H in [0, 180), S and V in [0, 256)) and a joint 3-D histogram with
``bins = (b_H, b_S, b_V)`` is computed per cell of a ``grid`` (default 1 x 1, i.e. a
global histogram). Each cell histogram ``p`` is normalised to sum 1 and mapped to
``sqrt(p)`` (Hellinger embedding), then the cells are concatenated and scaled by
``1 / sqrt(cells)``. The result has unit L2 norm, and the cosine similarity of two such
vectors is the Bhattacharyya coefficient ``sum_i sqrt(p_i q_i)`` averaged over cells.

**ORB bag of visual words.** ORB descriptors (256-bit) are quantised to the nearest of
``K`` binary *visual words* (Hamming distance). A visual vocabulary is fitted once, on
**development data only** (:func:`fit_vocabulary`): k-means on the descriptors' bits
(as 0/1 floats, where squared Euclidean distance equals Hamming distance), then each
centroid is binarised by majority vote (k-majority). Each image becomes a word histogram
weighted by tf-idf,

    v_w = (n_w / n) * log(N / df_w),

with ``n_w`` the count of word ``w`` in the image, ``n`` its number of descriptors, ``N``
the number of vocabulary-fitting images and ``df_w`` how many of them contain ``w``.
``v`` is L2-normalised, so retrieval uses cosine similarity. An image without
descriptors gets the uniform vector ``1 / sqrt(K)``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from latent_loop.encoders.base import VisualEncoder
from latent_loop.encoders.ijepa import to_rgb
from latent_loop.utils.logging import get_logger
from latent_loop.verification.orb import OrbConfig, extract_orb

logger = get_logger(__name__)


class HSVHistogramEncoder(VisualEncoder):
    name = "hsv"

    def __init__(
        self,
        bins: Sequence[int] = (16, 4, 4),
        grid: Sequence[int] = (1, 1),
        work_width: int = 320,
    ) -> None:
        self.bins = tuple(int(b) for b in bins)
        self.grid = tuple(int(g) for g in grid)
        if len(self.bins) != 3 or min(self.bins) < 1 or len(self.grid) != 2 or min(self.grid) < 1:
            raise ValueError("bins must be 3 positive ints and grid 2 positive ints")
        self.work_width = int(work_width)

    def load_model(self) -> None:
        return None

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        rgb = to_rgb(image)
        h, w = rgb.shape[:2]
        if w != self.work_width:
            size = (self.work_width, max(1, round(h * self.work_width / w)))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    def _encode_one(self, image: np.ndarray) -> np.ndarray:
        hsv = self.preprocess(image)
        gx, gy = self.grid
        h, w = hsv.shape[:2]
        cells = []
        for r in range(gy):
            for c in range(gx):
                cell = hsv[r * h // gy : (r + 1) * h // gy, c * w // gx : (c + 1) * w // gx]
                hist = cv2.calcHist(
                    [cell], [0, 1, 2], None, list(self.bins), [0, 180, 0, 256, 0, 256]
                )
                hist = hist.ravel().astype(np.float64)
                cells.append(np.sqrt(hist / max(hist.sum(), 1.0)))
        return (np.concatenate(cells) / np.sqrt(gx * gy)).astype(np.float32)

    def encode_batch(self, images: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack([self._encode_one(im) for im in images])

    def embedding_dim(self) -> int:
        return int(np.prod(self.bins) * self.grid[0] * self.grid[1])

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bins_hsv": list(self.bins),
            "grid": list(self.grid),
            "work_width": self.work_width,
            "transform": "per-cell normalised histogram -> sqrt (Hellinger)",
            "l2_normalized": True,
            "embedding_dim": self.embedding_dim(),
            "runtime": {"device": "cpu", "precision": "fp32"},
        }


# ----------------------------------------------------------------------------- vocabulary


@dataclass
class Vocabulary:
    centroids: np.ndarray  # (K, 32) uint8 binary visual words
    idf: np.ndarray  # (K,) float32
    meta: dict[str, Any] = field(default_factory=dict)
    _index: Any = field(default=None, repr=False, compare=False)

    @property
    def size(self) -> int:
        return int(self.centroids.shape[0])

    def digest(self) -> str:
        h = hashlib.sha1(self.centroids.tobytes())
        h.update(self.idf.astype(np.float32).tobytes())
        return h.hexdigest()[:12]

    def quantize(self, descriptors: np.ndarray) -> np.ndarray:
        """Nearest visual word (Hamming) of each ``(n, 32)`` uint8 descriptor."""
        if descriptors.shape[0] == 0:
            return np.zeros(0, np.int64)
        if self._index is None:
            import faiss  # noqa: PLC0415

            self._index = faiss.IndexBinaryFlat(8 * self.centroids.shape[1])
            self._index.add(np.ascontiguousarray(self.centroids))
        _, idx = self._index.search(np.ascontiguousarray(descriptors, dtype=np.uint8), 1)
        return idx[:, 0].astype(np.int64)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, centroids=self.centroids, idf=self.idf)
        tmp.replace(path)
        meta = {**self.meta, "digest": self.digest(), "size": self.size}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        path = Path(path)
        data = np.load(path)
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        return cls(data["centroids"], data["idf"].astype(np.float32), meta)


def fit_vocabulary(
    descriptor_sets: Sequence[np.ndarray],
    k: int = 1000,
    max_descriptors: int = 200_000,
    niter: int = 20,
    seed: int = 0,
    meta: dict[str, Any] | None = None,
) -> Vocabulary:
    """k-majority vocabulary + idf from per-image ORB descriptor sets (development data)."""
    import faiss  # noqa: PLC0415

    sets = [np.asarray(d, np.uint8).reshape(-1, 32) for d in descriptor_sets]
    allx = np.concatenate([d for d in sets if len(d)]) if any(len(d) for d in sets) else None
    if allx is None or len(allx) < k:
        raise ValueError(f"need at least k={k} descriptors to fit a vocabulary")
    rng = np.random.default_rng(seed)
    sample = allx[rng.choice(len(allx), min(max_descriptors, len(allx)), replace=False)]
    bits = np.unpackbits(sample, axis=1).astype(np.float32)
    km = faiss.Kmeans(bits.shape[1], k, niter=niter, seed=seed, verbose=False)
    km.train(bits)
    centroids = np.packbits((km.centroids >= 0.5).astype(np.uint8), axis=1)
    vocab = Vocabulary(centroids, np.zeros(k, np.float32))
    df = np.zeros(k, np.int64)
    for d in sets:
        if len(d):
            df[np.unique(vocab.quantize(d))] += 1
    n_images = len(sets)
    vocab.idf = np.log(n_images / np.maximum(df, 1)).astype(np.float32)
    vocab.meta = {
        **(meta or {}),
        "k": k,
        "fit_images": n_images,
        "fit_descriptors_total": int(len(allx)),
        "fit_descriptors_sampled": int(len(sample)),
        "kmeans_niter": niter,
        "seed": seed,
        "unused_words": int((df == 0).sum()),
    }
    return vocab


class OrbBoVWEncoder(VisualEncoder):
    name = "orb_bovw"

    def __init__(
        self,
        vocabulary_path: str | Path,
        orb: OrbConfig | None = None,
        tfidf: bool = True,
        fit_command: str = "",
    ) -> None:
        self.vocabulary_path = Path(vocabulary_path)
        self.orb = orb or OrbConfig(n_features=1000)
        self.tfidf = tfidf
        self.fit_command = fit_command
        self.vocab: Vocabulary | None = None

    def load_model(self) -> None:
        if self.vocab is not None:
            return
        if not self.vocabulary_path.is_file():
            raise FileNotFoundError(
                f"ORB visual vocabulary not found: {self.vocabulary_path}\n"
                f"Fit it on development data first: {self.fit_command}"
            )
        self.vocab = Vocabulary.load(self.vocabulary_path)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return extract_orb(image, self.orb).descriptors

    def _encode_one(self, image: np.ndarray) -> np.ndarray:
        assert self.vocab is not None
        k = self.vocab.size
        words = self.vocab.quantize(self.preprocess(image))
        if words.size == 0:
            return np.full(k, 1 / np.sqrt(k), np.float32)
        v = np.bincount(words, minlength=k).astype(np.float64) / words.size
        if self.tfidf:
            v *= self.vocab.idf
        n = np.linalg.norm(v)
        return (v / n if n > 0 else np.full(k, 1 / np.sqrt(k))).astype(np.float32)

    def encode_batch(self, images: Sequence[np.ndarray]) -> np.ndarray:
        self.load_model()
        return np.stack([self._encode_one(im) for im in images])

    def embedding_dim(self) -> int:
        self.load_model()
        assert self.vocab is not None
        return self.vocab.size

    def info(self) -> dict[str, Any]:
        self.load_model()
        assert self.vocab is not None
        return {
            "name": self.name,
            "vocabulary_digest": self.vocab.digest(),
            "vocabulary_size": self.vocab.size,
            "vocabulary_fit_sources": self.vocab.meta.get("fit_sources"),
            "orb": self.orb.__dict__,
            "tfidf": self.tfidf,
            "l2_normalized": True,
            "embedding_dim": self.vocab.size,
            "runtime": {"device": "cpu", "precision": "fp32"},
        }


# ----------------------------------------------------------------------------- config


def vocabulary_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Which development data fits the vocabulary for the configured dataset, and where.

    KITTI: the sequences in ``orb_bovw.fit_kitti_sequences``, each of which must be listed
    in ``splits.dev`` and must not be in ``splits.test`` (raises otherwise). Synthetic: the
    same generator with ``orb_bovw.fit_synthetic_seed``, which must differ from the
    evaluated seed. The evaluated dataset is never used for fitting unless it is itself a
    development sequence named in the fit list (reported as ``in_sample``).
    """
    from latent_loop.utils.config import resolve_path  # noqa: PLC0415

    ob = config.get("orb_bovw", {})
    k, nfeat, stride = (
        int(ob.get("vocab_size", 1000)),
        int(ob.get("n_features", 1000)),
        int(ob.get("fit_stride", 5)),
    )
    ds = config["dataset"]
    splits = config.get("splits", {})
    if ds["name"] == "kitti":
        seqs = [f"{int(s):02d}" for s in ob.get("fit_kitti_sequences", ["05"])]
        dev = {f"{int(s):02d}" for s in splits.get("dev", [])}
        test = {f"{int(s):02d}" for s in splits.get("test", [])}
        bad = [s for s in seqs if s in test or s not in dev]
        if bad:
            raise ValueError(
                f"Vocabulary fit sequences {bad} are not development-only "
                f"(splits.dev={sorted(dev)}, splits.test={sorted(test)})"
            )
        camera = int(ds.get("camera", 2))
        tag = f"kitti_{'-'.join(seqs)}_cam{camera}_s{stride}"
        sources = [
            {"name": "kitti", "root": ds.get("root"), "sequence": s, "camera": camera} for s in seqs
        ]
        in_sample = f"{int(ds['sequence']):02d}" in seqs
    else:
        seed = int(ob.get("fit_synthetic_seed", 11))
        stride = int(ob.get("fit_synthetic_stride", 1))  # few ORB features per synthetic frame
        if seed == int(ds.get("synthetic", {}).get("seed", 7)):
            raise ValueError("fit_synthetic_seed must differ from the evaluated synthetic seed")
        tag = f"synthetic_seed{seed}_s{stride}"
        sources = [{"name": "synthetic", "synthetic_seed": seed}]
        in_sample = False
    path = (
        resolve_path(ob.get("vocabulary_dir", "data/cache/vocabulary"))
        / f"orb_bovw_k{k}_n{nfeat}_{tag}.npz"
    )
    return {
        "path": path,
        "k": k,
        "n_features": nfeat,
        "stride": stride,
        "sources": sources,
        "tag": tag,
        "in_sample": in_sample,
        "max_descriptors": int(ob.get("max_fit_descriptors", 200_000)),
        "seed": int(ob.get("seed", 0)),
    }


def hsv_from_config(config: dict[str, Any]) -> HSVHistogramEncoder:
    h = config.get("hsv", {})
    return HSVHistogramEncoder(
        h.get("bins", (16, 4, 4)), h.get("grid", (1, 1)), h.get("work_width", 320)
    )


def orb_bovw_from_config(config: dict[str, Any]) -> OrbBoVWEncoder:
    plan = vocabulary_plan(config)
    ob = config.get("orb_bovw", {})
    if plan["in_sample"]:
        logger.warning(
            "Evaluated KITTI sequence is one of the vocabulary-fitting sequences (in-sample)",
            extra={"sequence": config["dataset"].get("sequence")},
        )
    return OrbBoVWEncoder(
        plan["path"],
        OrbConfig(n_features=plan["n_features"]),
        bool(ob.get("tfidf", True)),
        fit_command="python scripts/fit_vocabulary.py with the same --config/--set",
    )
