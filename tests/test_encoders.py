"""I-JEPA encoder tests on a tiny ViT saved in the official checkpoint format.

The real ViT-H/14 checkpoint is exercised by ``test_real_ijepa_checkpoint`` when its
extracted target encoder exists locally; otherwise that test is skipped.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from latent_loop.encoders import build_encoder
from latent_loop.encoders.ijepa import (
    CHECKPOINTS,
    IJEPAEncoder,
    extract_target_encoder,
    find_raw_checkpoint,
    resize_image,
    target_encoder_path,
)
from latent_loop.encoders.vit import build_vit, sincos_pos_embed_2d
from latent_loop.features.pooling import l2_normalize, pool_tokens
from latent_loop.utils.config import load_config, resolve_path
from latent_loop.utils.device import resolve_device, resolve_precision

TINY = {"embed_dim": 32, "depth": 2, "num_heads": 4}
TINY_ARCH = {**TINY, "patch_size": 8, "img_size": 32}


def _official_style_checkpoint(path, seed: int = 0) -> dict[str, torch.Tensor]:
    """Write a file shaped like the official I-JEPA full training state."""
    torch.manual_seed(seed)
    encoder = build_vit(TINY, patch_size=8, img_size=32).state_dict()
    target = build_vit(TINY, patch_size=8, img_size=32).state_dict()
    torch.save(
        {
            "encoder": {f"module.{k}": v for k, v in encoder.items()},
            "target_encoder": {f"module.{k}": v for k, v in target.items()},
            "predictor": {"module.mask_token": torch.zeros(1, 1, 16)},
            "opt": {"state": {}, "param_groups": []},
            "scaler": None,
            "epoch": 300,
            "loss": 0.1,
            "batch_size": 2048,
            "world_size": 16,
            "lr": 0.001,
        },
        path,
    )
    return target


@pytest.fixture
def tiny_weights(tmp_path):
    raw = tmp_path / "raw.pth.tar"
    target = _official_style_checkpoint(raw)
    out = tmp_path / "ijepa" / "tiny_target_encoder.pt"
    meta = extract_target_encoder(raw, out)
    return out, target, meta


def _encoder(weights, **kwargs) -> IJEPAEncoder:
    return IJEPAEncoder(
        checkpoint="tiny", weights_path=weights, arch_override=TINY_ARCH, device="cpu", **kwargs
    )


def _images(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (40, 120, 3), dtype=np.uint8) for _ in range(n)]


# --------------------------------------------------------------------------- positions
def test_sincos_matches_definition():
    d, gh, gw = 16, 3, 5
    pe = sincos_pos_embed_2d(d, gh, gw)
    assert pe.shape == (gh * gw, d)
    omega = 1.0 / 10000 ** (np.arange(d // 4) / (d / 4))
    row, col = 2, 4
    token = pe[row * gw + col]
    np.testing.assert_allclose(token[: d // 4], np.sin(col * omega))  # first half: column
    np.testing.assert_allclose(token[d // 4 : d // 2], np.cos(col * omega))
    np.testing.assert_allclose(token[d // 2 : 3 * d // 4], np.sin(row * omega))  # then row
    np.testing.assert_allclose(token[3 * d // 4 :], np.cos(row * omega))
    # Square grids are a prefix-consistent special case of rectangular ones.
    np.testing.assert_allclose(
        sincos_pos_embed_2d(d, 3, 3).reshape(3, 3, d), pe.reshape(3, 5, d)[:, :3]
    )


# --------------------------------------------------------------------------- checkpoint
def test_extract_target_encoder_strips_prefix_and_keeps_target(tiny_weights):
    out, target, meta = tiny_weights
    blob = torch.load(out, weights_only=True)
    assert set(blob["state_dict"]) == set(target)
    for key, value in target.items():
        torch.testing.assert_close(blob["state_dict"][key], value.float())
    assert meta["epoch"] == 300
    assert "target_encoder" in meta["source_keys"] and len(meta["source_sha256"]) == 64
    assert out.with_suffix(".json").is_file()


def test_extract_rejects_files_without_target_encoder(tmp_path):
    raw = tmp_path / "bad.pth"
    torch.save({"encoder": {}}, raw)
    with pytest.raises(KeyError, match="target_encoder"):
        extract_target_encoder(raw, tmp_path / "out.pt")


def test_strict_loading_rejects_wrong_architecture(tiny_weights):
    out, _, _ = tiny_weights
    wrong = IJEPAEncoder(
        checkpoint="tiny", weights_path=out, device="cpu", arch_override={**TINY_ARCH, "depth": 3}
    )
    with pytest.raises(RuntimeError, match="Missing key"):
        wrong.load_model()


def test_pos_embed_check_tolerates_ema_drift_but_rejects_wrong_layout(tmp_path):
    torch.manual_seed(0)
    state = build_vit(TINY, patch_size=8, img_size=32).state_dict()
    drifted = {**state, "pos_embed": state["pos_embed"] + 1e-4}  # like the EMA target encoder
    d = state["pos_embed"].shape[-1] // 2
    swapped = {
        **state,
        "pos_embed": torch.cat([state["pos_embed"][..., d:], state["pos_embed"][..., :d]], -1),
    }
    for name, sd in (("drifted", drifted), ("swapped", swapped)):
        torch.save({"state_dict": sd, "meta": {}}, tmp_path / f"{name}.pt")
    enc = _encoder(tmp_path / "drifted.pt")
    enc.load_model()
    assert enc.pos_embed_drift == pytest.approx(1e-4, rel=1e-2)
    with pytest.raises(ValueError, match="sincos"):
        _encoder(tmp_path / "swapped.pt").load_model()


def test_missing_weights_explain_how_to_download(tmp_path):
    enc = IJEPAEncoder(checkpoint_dir=tmp_path, device="cpu")
    with pytest.raises(FileNotFoundError, match="download_checkpoints.py"):
        enc.load_model()


def test_checkpoint_registry_and_raw_lookup(tmp_path):
    spec = CHECKPOINTS["vith14_in1k"]
    assert (spec.arch, spec.patch_size, spec.img_size) == ("vit_huge", 14, 224)
    assert spec.url == "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar"
    assert find_raw_checkpoint(tmp_path, spec) is None
    (tmp_path / spec.filename).write_bytes(b"x")
    assert find_raw_checkpoint(tmp_path, spec) == tmp_path / spec.filename
    assert target_encoder_path(tmp_path, "vith14_in1k").parent == tmp_path / "ijepa"


# --------------------------------------------------------------------------- preprocessing
@pytest.mark.parametrize(
    ("mode", "shape"), [("squash", (32, 32)), ("center_crop", (32, 32)), ("aspect", (32, 96))]
)
def test_resize_modes(mode, shape):
    image = np.zeros((40, 120, 3), np.uint8)
    assert resize_image(image, mode, 32, 8).shape[:2] == shape


def test_center_crop_keeps_the_centre():
    image = np.zeros((40, 120, 3), np.uint8)
    image[:, 40:80] = 255  # bright centre third
    crop = resize_image(image, "center_crop", 32, 8)
    assert crop.mean() > 200


def test_preprocess_normalisation_and_grayscale(tiny_weights):
    enc = _encoder(tiny_weights[0])
    mean_colour = (np.array([0.485, 0.456, 0.406]) * 255).round().astype(np.uint8)
    x = enc.preprocess(np.full((40, 120, 3), mean_colour, np.uint8))
    assert x.shape == (3, 32, 32) and x.dtype == torch.float32
    assert x.abs().max() < 0.02  # the ImageNet mean maps to ~0
    gray = enc.preprocess(np.full((40, 120), 128, np.uint8))
    torch.testing.assert_close(gray[0] * 0.229 + 0.485, gray[1] * 0.224 + 0.456)
    with pytest.raises(ValueError):
        enc.preprocess(np.zeros((4, 4, 4), np.uint8))


# --------------------------------------------------------------------------- inference
def test_encode_batch_is_normalised_deterministic_and_batch_invariant(tiny_weights):
    enc = _encoder(tiny_weights[0])
    images = _images(5)
    batch = enc.encode_batch(images)
    assert batch.shape == (5, 32) and batch.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1.0, atol=1e-5)
    singles = np.concatenate([enc.encode_batch([im]) for im in images])
    np.testing.assert_allclose(batch, singles, atol=1e-5)
    np.testing.assert_array_equal(batch, enc.encode_batch(images))


def test_descriptor_is_mean_of_patch_tokens(tiny_weights):
    enc = _encoder(tiny_weights[0])
    images = _images(2)
    tokens, grid = enc.encode_tokens(images)
    assert grid == (4, 4) and tokens.shape == (2, 16, 32)
    expected = tokens.mean(dim=1)
    expected = (expected / expected.norm(dim=1, keepdim=True)).numpy()
    np.testing.assert_allclose(enc.encode_batch(images), expected, atol=1e-5)


def test_aspect_mode_uses_wider_grid(tiny_weights):
    enc = _encoder(tiny_weights[0], resize_mode="aspect")
    tokens, grid = enc.encode_tokens(_images(1))
    assert grid == (4, 12) and tokens.shape == (1, 48, 32)


def test_fingerprint_ignores_runtime_but_tracks_preprocessing(tiny_weights):
    a = _encoder(tiny_weights[0])
    b = _encoder(tiny_weights[0])
    b.to_device("cpu", "fp32")
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != _encoder(tiny_weights[0], resize_mode="aspect").fingerprint()
    assert a.info()["source_sha256"] == tiny_weights[2]["source_sha256"]  # known before load


def test_pooling_helpers():
    tokens = torch.arange(24, dtype=torch.float16).reshape(1, 4, 6)
    pooled = pool_tokens(tokens)
    assert pooled.dtype == torch.float32
    torch.testing.assert_close(pooled, tokens.float().mean(1))
    with pytest.raises(ValueError):
        pool_tokens(tokens, "max")
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_device_and_precision_resolution():
    assert resolve_device("cpu") == "cpu"
    assert resolve_precision("auto", "cpu") == "fp32"
    assert resolve_precision("fp16", "cpu") == "fp32"
    with pytest.raises(ValueError):
        resolve_precision("int8", "cpu")
    with pytest.raises(ValueError):
        resolve_device("tpu")
    if torch.cuda.is_available():
        from latent_loop.utils import device as device_mod

        choice = resolve_precision("auto", "cuda")
        if torch.cuda.is_bf16_supported(including_emulation=False):
            assert choice == "bf16"
        else:  # decided by measurement, not by architecture name
            assert choice == ("fp16" if device_mod.fp16_faster_than_fp32() else "fp32")
        assert resolve_precision("fp16", "cuda") == "fp16"  # explicit requests honoured


def test_build_encoder_from_config():
    config = load_config("configs/ijepa.yaml", ["device=cpu"])
    enc = build_encoder("ijepa", config)
    assert isinstance(enc, IJEPAEncoder) and enc.device == "cpu"
    assert enc.image_size == 224 and enc.embedding_dim() == 1280
    with pytest.raises(ValueError):
        build_encoder("nope", config)


# --------------------------------------------------------------------------- real weights
REAL_WEIGHTS = target_encoder_path(resolve_path("checkpoints"), "vith14_in1k")


@pytest.mark.skipif(not REAL_WEIGHTS.is_file(), reason="I-JEPA ViT-H/14 weights not extracted")
def test_real_ijepa_checkpoint(small_synthetic):
    enc = IJEPAEncoder(checkpoint_dir=resolve_path("checkpoints"), device="auto")
    enc.load_model()  # strict state-dict load + sincos pos_embed check
    assert enc.num_params() == 630_434_560  # ViT-H/14 without fixed position embeddings
    images = [small_synthetic.get_image(i) for i in (0, 1, 30)]
    emb = enc.encode_batch(images)
    assert emb.shape == (3, 1280) and np.all(np.isfinite(emb))
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)
    # Consecutive frames (1 m apart) are more similar than frames 30 m apart.
    assert emb[0] @ emb[1] > emb[0] @ emb[2]
    enc.unload()
