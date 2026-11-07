"""Phase 5 baselines: DINOv2 ViT-B/14, HSV histograms and ORB bag of visual words."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from latent_loop.encoders import AVAILABLE_ENCODERS, build_encoder
from latent_loop.encoders.classical import (
    HSVHistogramEncoder,
    OrbBoVWEncoder,
    Vocabulary,
    fit_vocabulary,
    vocabulary_plan,
)
from latent_loop.encoders.dinov2 import DinoSpec, DINOv2Encoder, DinoViT, checkpoint_path
from latent_loop.features.extraction import extract_embeddings, make_cache
from latent_loop.retrieval import RetrievalConfig, retrieve
from latent_loop.utils.config import REPO_ROOT, load_config
from test_verification import texture

TINY = DinoSpec("tiny", 32, 2, 2, 14, 56, "tiny.pth", "", 0)  # 4 x 4 stored grid
REAL_DINO = checkpoint_path(REPO_ROOT / "checkpoints", "vitb14")


@pytest.fixture
def tiny_dino(tmp_path, monkeypatch) -> Path:
    import latent_loop.encoders.dinov2 as d

    torch.manual_seed(0)
    model = DinoViT(TINY)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    path = tmp_path / "tiny.pth"
    torch.save(model.state_dict(), path)
    monkeypatch.setitem(d.CHECKPOINTS, "tiny", TINY)
    return path


# ----------------------------------------------------------------------------- DINOv2


def test_dino_pos_embed_interpolation(tiny_dino):
    model = DinoViT(TINY)
    model.load_state_dict(torch.load(tiny_dino, weights_only=True), strict=True)
    assert model.interpolate_pos_encoding(4, 4) is model.pos_embed  # stored grid: unchanged
    pe = model.interpolate_pos_encoding(3, 7)
    assert pe.shape == (1, 1 + 21, 32)
    assert torch.equal(pe[:, 0], model.pos_embed[:, 0])  # [CLS] position is never resized
    out = model(torch.randn(2, 3, 42, 98))
    assert out.shape == (2, 1 + 3 * 7, 32)


def test_dino_strict_load_rejects_other_architecture(tiny_dino):
    other = DinoViT(DinoSpec("t", 32, 3, 2, 14, 56, "", "", 0))
    with pytest.raises(RuntimeError):
        other.load_state_dict(torch.load(tiny_dino, weights_only=True), strict=True)


def test_dino_encoder_pooling_and_normalisation(tiny_dino):
    img = texture(0, 96, 64)
    rgb = np.dstack([img] * 3)
    cls = DINOv2Encoder("tiny", weights_path=tiny_dino, device="cpu", image_size=56)
    mean = DINOv2Encoder(
        "tiny", weights_path=tiny_dino, device="cpu", image_size=56, pooling="mean"
    )
    a, b = cls.encode_batch([rgb, img]), mean.encode_batch([rgb])
    assert a.shape == (2, 32) and np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-5)
    assert np.allclose(a[0], a[1], atol=1e-6)  # grey input is replicated to RGB
    tokens = cls.encode_tokens([rgb])
    ref = torch.nn.functional.normalize(tokens[:, 1:].mean(1), dim=-1).numpy()
    assert np.allclose(b, ref, atol=1e-6)
    assert cls.fingerprint() != mean.fingerprint()
    with pytest.raises(ValueError):
        DINOv2Encoder("tiny", weights_path=tiny_dino, image_size=50)


def test_dino_missing_weights_message(tmp_path):
    enc = DINOv2Encoder(checkpoint_dir=tmp_path, device="cpu")
    with pytest.raises(FileNotFoundError, match="download_checkpoints.py dinov2"):
        enc.load_model()


@pytest.mark.skipif(not REAL_DINO.is_file(), reason="DINOv2 ViT-B/14 weights not downloaded")
def test_real_dinov2_checkpoint(small_synthetic):
    enc = DINOv2Encoder(checkpoint_dir=REPO_ROOT / "checkpoints", device="cpu")
    assert enc.num_params() == 86_580_480
    images = [small_synthetic.get_image(i) for i in (0, 1, 60)]
    d = enc.encode_batch(images)
    assert d.shape == (3, 768) and np.allclose(np.linalg.norm(d, axis=1), 1, atol=1e-5)
    assert d[0] @ d[1] > d[0] @ d[2]  # consecutive frames are more similar than distant ones
    enc.unload()


# ----------------------------------------------------------------------------- HSV


def test_hsv_histogram_properties():
    enc = HSVHistogramEncoder(bins=(8, 2, 2))
    red = np.zeros((40, 80, 3), np.uint8)
    red[..., 0] = 220
    blue = np.zeros_like(red)
    blue[..., 2] = 220
    mix = np.concatenate([red[:, :40], blue[:, :40]], axis=1)
    d = enc.encode_batch([red, blue, mix, red])
    assert d.shape == (4, 32) and np.allclose(np.linalg.norm(d, axis=1), 1)
    assert d[0] @ d[3] == pytest.approx(1.0)
    assert d[0] @ d[1] == pytest.approx(0.0, abs=1e-6)  # disjoint colours
    assert d[0] @ d[2] == pytest.approx(np.sqrt(0.5), abs=1e-3)  # Bhattacharyya of 1 vs 1/2
    grey = enc.encode_batch([np.full((40, 80), 100, np.uint8)])
    assert np.isclose(np.linalg.norm(grey), 1)
    grid = HSVHistogramEncoder(bins=(8, 2, 2), grid=(2, 1))
    g = grid.encode_batch([mix])
    assert g.shape == (1, 64) and np.isclose(np.linalg.norm(g), 1)
    assert grid.fingerprint() != enc.fingerprint()


# ----------------------------------------------------------------------------- BoVW


def _clustered_descriptors(rng, centres, per=30, flips=6):
    sets = []
    for _ in range(20):
        rows = []
        for c in centres[rng.choice(len(centres), 5, replace=False)]:
            bits = np.repeat(np.unpackbits(c)[None], per, 0)
            for r in bits:
                r[rng.choice(256, flips, replace=False)] ^= 1
            rows.append(np.packbits(bits, axis=1))
        sets.append(np.concatenate(rows))
    return sets


def test_fit_vocabulary_recovers_binary_clusters_and_idf(tmp_path):
    rng = np.random.default_rng(0)
    centres = rng.integers(0, 256, (8, 32), dtype=np.uint8)
    sets = _clustered_descriptors(rng, centres)
    # Over-complete k (as in real use, k=1000): with k equal to the number of true clusters,
    # randomly initialised k-means can merge two of them (a known local minimum).
    vocab = fit_vocabulary(sets, k=16, seed=0, meta={"fit_sources": ["test"]})
    # every true centre is quantised to a word whose binary centroid is (almost) the centre
    for c in centres:
        w = vocab.quantize(c[None])[0]
        dist = np.unpackbits(np.bitwise_xor(vocab.centroids[w], c)).sum()
        assert dist <= 2
    assert vocab.centroids.dtype == np.uint8 and vocab.idf.shape == (16,)
    assert np.all(vocab.idf >= 0) and vocab.meta["fit_images"] == 20
    path = vocab.save(tmp_path / "v.npz")
    back = Vocabulary.load(path)
    assert back.digest() == vocab.digest() and back.meta["fit_sources"] == ["test"]
    with pytest.raises(ValueError):
        fit_vocabulary([centres[:3]], k=16)


def test_orb_bovw_encoder(tmp_path):
    imgs = [texture(s, 320, 160) for s in range(6)]
    from latent_loop.verification.orb import OrbConfig, extract_orb

    orb = OrbConfig(n_features=300)
    vocab = fit_vocabulary([extract_orb(i, orb).descriptors for i in imgs], k=64, seed=0)
    path = vocab.save(tmp_path / "v.npz")
    enc = OrbBoVWEncoder(path, orb)
    shifted = np.roll(imgs[0], 5, axis=1)
    d = enc.encode_batch([imgs[0], shifted, imgs[5], np.full((160, 320), 90, np.uint8)])
    assert d.shape == (4, 64) and np.allclose(np.linalg.norm(d, axis=1), 1, atol=1e-5)
    assert d[0] @ d[1] > d[0] @ d[2]  # same scene beats a different texture
    assert np.allclose(d[3], 1 / 8)  # no descriptors -> uniform vector
    assert enc.info()["vocabulary_digest"] == vocab.digest()
    missing = OrbBoVWEncoder(tmp_path / "none.npz", fit_command="fit it")
    with pytest.raises(FileNotFoundError, match="fit it"):
        missing.load_model()


def test_vocabulary_plan_enforces_development_only():
    cfg = load_config(
        "configs/default.yaml",
        ["dataset.name=kitti", "dataset.root=data/raw/kitti/dataset", "dataset.sequence='00'"],
    )
    plan = vocabulary_plan(cfg)
    assert [s["sequence"] for s in plan["sources"]] == ["05"] and not plan["in_sample"]
    assert "kitti_05_cam2_s5" in plan["path"].name
    for bad in (["08"], ["01"], ["05", "02"]):  # test sequence; not in dev; mixed
        cfg["orb_bovw"]["fit_kitti_sequences"] = bad
        with pytest.raises(ValueError, match="development-only"):
            vocabulary_plan(cfg)
    cfg["orb_bovw"]["fit_kitti_sequences"] = ["00"]
    assert vocabulary_plan(cfg)["in_sample"]  # flagged, not silently allowed
    syn = load_config("configs/default.yaml")
    assert (
        vocabulary_plan(syn)["sources"][0]["synthetic_seed"] != syn["dataset"]["synthetic"]["seed"]
    )
    syn["orb_bovw"]["fit_synthetic_seed"] = syn["dataset"]["synthetic"]["seed"]
    with pytest.raises(ValueError):
        vocabulary_plan(syn)


# ----------------------------------------------------------------------------- common interface


def test_registry_and_common_pipeline(small_synthetic, tmp_path):
    assert set(AVAILABLE_ENCODERS) == {"ijepa", "dinov2", "dinov3", "hsv", "orb_bovw"}
    cfg = load_config("configs/default.yaml")
    enc = build_encoder("hsv", cfg)
    cache = make_cache(tmp_path, small_synthetic, enc)
    res = extract_embeddings(small_synthetic, enc, cache, batch_size=16)
    again = extract_embeddings(small_synthetic, enc, cache, batch_size=16)
    assert again.encoded_now == 0 and np.array_equal(res.embeddings, again.embeddings)
    ids, ts = small_synthetic.frame_ids, small_synthetic.timestamps()
    off = retrieve(ids, ts, res.embeddings, RetrievalConfig(top_k=3, temporal_exclusion_frames=20))
    cau = retrieve(
        ids,
        ts,
        res.embeddings,
        RetrievalConfig(top_k=3, temporal_exclusion_frames=20, mode="causal"),
    )
    assert np.array_equal(off.candidate_ids, cau.candidate_ids)
    with pytest.raises(ValueError):
        build_encoder("nope", cfg)


def test_ijepa_fingerprint_unchanged_by_phase5_config():
    # Moving shared keys into default.yaml must not invalidate the Phase 2 caches.
    pytest.importorskip("torch")
    enc = build_encoder("ijepa", load_config("configs/ijepa.yaml"))
    assert enc.fingerprint() == "d49bc21384" or not enc.weights_meta  # no sidecar on CI


def test_cv2_hsv_ranges_assumption():
    hsv = cv2.cvtColor(np.array([[[0, 0, 255]]], np.uint8), cv2.COLOR_RGB2HSV)
    assert hsv[0, 0, 0] == 120  # OpenCV hue is in [0, 180), as the histogram ranges assume
