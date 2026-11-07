"""DINOv3 wrapper around Meta's official code (random weights; the real ones are gated)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from latent_loop.encoders.dinov3 import DINOv3Encoder
from latent_loop.utils.config import resolve_path

REPO = resolve_path("external/dinov3")
pytestmark = pytest.mark.skipif(
    not (REPO / "dinov3" / "hub" / "backbones.py").is_file(),
    reason="official DINOv3 code not cloned to external/dinov3",
)


@pytest.fixture(scope="module")
def random_weights(tmp_path_factory):
    torch.manual_seed(0)
    enc = DINOv3Encoder(repo_dir=REPO, device="cpu")
    path = tmp_path_factory.mktemp("w") / "random.pth"
    torch.save(enc.build(pretrained=False).state_dict(), path)
    return path


def test_descriptors_from_official_network(random_weights):
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, (370, 1226, 3), dtype=np.uint8) for _ in range(2)]
    out = {}
    for pooling in ("cls", "mean"):
        enc = DINOv3Encoder(
            repo_dir=REPO, device="cpu", pooling=pooling, weights_path=random_weights
        )
        out[pooling] = enc.encode_batch(images)
        assert out[pooling].shape == (2, 768) and out[pooling].dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(out[pooling], axis=1), 1, rtol=1e-5)
    assert not np.allclose(out["cls"], out["mean"])
    a = DINOv3Encoder(repo_dir=REPO, device="cpu", weights_path=random_weights)
    b = DINOv3Encoder(repo_dir=REPO, device="cpu", pooling="mean", weights_path=random_weights)
    assert a.fingerprint() != b.fingerprint()


def test_strict_load_and_missing_weights(random_weights, tmp_path):
    state = torch.load(random_weights, weights_only=True)
    state.pop(next(iter(state)))
    broken = tmp_path / "broken.pth"
    torch.save(state, broken)
    with pytest.raises(RuntimeError, match="Missing key"):
        DINOv3Encoder(repo_dir=REPO, device="cpu", weights_path=broken).build()
    with pytest.raises(FileNotFoundError, match="licence-gated"):
        DINOv3Encoder(repo_dir=REPO, device="cpu", weights_path=tmp_path / "none.pth").build()
