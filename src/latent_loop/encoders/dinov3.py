"""Frozen DINOv3 encoder (Siméoni et al., 2025), using Meta's own model code.

Unlike I-JEPA and DINOv2, the backbone is not re-implemented here: the official repository
(``https://github.com/facebookresearch/dinov3``) is cloned to ``dinov3.repo_dir``
(default ``external/dinov3``, git-ignored) and its ``dinov3_vitb16`` builder constructs
the network. The official checkpoint is loaded with ``strict=True``. Its file name carries
the first 8 hex digits of its SHA-256 (``...-73cec8be.pth``), which ``download_checkpoints.py
dinov3`` checks.

The weights are licence-gated (DINOv3 License): they are obtained by the user through
Meta's request form or the gated Hugging Face repository and placed in ``checkpoints/``.

Descriptor: the final-norm CLS token (``pooling="cls"``) or the mean of the patch tokens
(``pooling="mean"``, storage/register tokens excluded), then L2 normalisation.
Preprocessing as for DINOv2 (ImageNet mean/std), with patch size 16.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from latent_loop.encoders.base import VisualEncoder
from latent_loop.encoders.ijepa import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESIZE_MODES,
    resize_image,
    to_rgb,
)
from latent_loop.features.pooling import l2_normalize
from latent_loop.utils.device import resolve_device, resolve_precision, torch_dtype
from latent_loop.utils.logging import get_logger

logger = get_logger(__name__)

POOLINGS = ("cls", "mean")


@dataclass(frozen=True)
class Dinov3Spec:
    builder: str
    filename: str
    sha256_prefix: str
    embed_dim: int
    patch_size: int = 16


CHECKPOINTS = {
    # ViT-B/16 distilled from the 7B teacher on LVD-1689M (86M parameters)
    "vitb16": Dinov3Spec(
        "dinov3_vitb16", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", "73cec8be", 768
    ),
    # ViT-L/16 distilled on LVD-1689M (0.3B parameters): the largest that fits 4 GB in FP32
    "vitl16": Dinov3Spec(
        "dinov3_vitl16", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "8aa4cbdd", 1024
    ),
}


def checkpoint_path(checkpoint_dir: str | Path, checkpoint: str) -> Path:
    return Path(checkpoint_dir) / "dinov3" / CHECKPOINTS[checkpoint].filename


def official_builder(repo_dir: str | Path, builder: str):
    """Import a backbone builder from the cloned official repository."""
    repo_dir = Path(repo_dir)
    if not (repo_dir / "dinov3" / "hub" / "backbones.py").is_file():
        raise FileNotFoundError(
            f"Official DINOv3 code not found in {repo_dir}\n"
            f"Run: git clone https://github.com/facebookresearch/dinov3 {repo_dir}"
        )
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from dinov3.hub import backbones  # noqa: PLC0415

    return getattr(backbones, builder)


class DINOv3Encoder(VisualEncoder):
    """Frozen official DINOv3 backbone + CLS (or patch-mean) pooling + L2 normalisation."""

    name = "dinov3"

    def __init__(
        self,
        checkpoint: str = "vitb16",
        checkpoint_dir: str | Path = "checkpoints",
        repo_dir: str | Path = "external/dinov3",
        device: str = "auto",
        precision: str = "auto",
        resize_mode: str = "squash",
        image_size: int = 224,
        pooling: str = "cls",
        weights_path: str | Path | None = None,
    ) -> None:
        if checkpoint not in CHECKPOINTS:
            raise ValueError(
                f"Unknown DINOv3 checkpoint {checkpoint!r}; known: {sorted(CHECKPOINTS)}"
            )
        if resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {RESIZE_MODES}")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}")
        self.spec = CHECKPOINTS[checkpoint]
        if image_size % self.spec.patch_size:
            raise ValueError(f"image_size must be a multiple of {self.spec.patch_size}")
        self.checkpoint = checkpoint
        self.repo_dir = Path(repo_dir)
        self.weights_path = (
            Path(weights_path) if weights_path else checkpoint_path(checkpoint_dir, checkpoint)
        )
        self.resize_mode = resize_mode
        self.image_size = int(image_size)
        self.pooling = pooling
        self.device = resolve_device(device)
        self.precision = resolve_precision(precision, self.device)
        self.model: torch.nn.Module | None = None
        sidecar = self.weights_path.with_suffix(".json")
        self.weights_meta: dict[str, Any] = (
            json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        )
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def build(self, pretrained: bool = True) -> torch.nn.Module:
        """The official network; with ``pretrained`` the local checkpoint is loaded strictly."""
        model = official_builder(self.repo_dir, self.spec.builder)(pretrained=False)
        if pretrained:
            if not self.weights_path.is_file():
                raise FileNotFoundError(
                    f"DINOv3 weights not found: {self.weights_path}\n"
                    "They are licence-gated: request them from Meta "
                    "(https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) "
                    "and run: python scripts/download_checkpoints.py dinov3 --url <URL>"
                )
            state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
        return model.requires_grad_(False).eval()

    def load_model(self) -> None:
        if self.model is not None:
            return
        self.model = self.build(pretrained=True).to(self.device, dtype=torch_dtype(self.precision))
        logger.info(
            "Loaded DINOv3 backbone",
            extra={
                "checkpoint": self.checkpoint,
                "device": self.device,
                "precision": self.precision,
            },
        )

    def num_params(self) -> int:
        self.load_model()
        assert self.model is not None
        return sum(p.numel() for p in self.model.parameters())

    def to_device(self, device: str, precision: str | None = None) -> None:
        self.device = resolve_device(device)
        self.precision = resolve_precision(precision or "auto", self.device)
        if self.model is not None:
            self.model = self.model.to(self.device, dtype=torch_dtype(self.precision))

    def unload(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        rgb = resize_image(to_rgb(image), self.resize_mode, self.image_size, self.spec.patch_size)
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        return (x - self._mean) / self._std

    @staticmethod
    def pool(features: dict[str, torch.Tensor], pooling: str) -> torch.Tensor:
        if pooling == "cls":
            return features["x_norm_clstoken"]
        return features["x_norm_patchtokens"].mean(1)  # storage tokens excluded

    @torch.inference_mode()
    def encode_batch(self, images: Sequence[np.ndarray]) -> np.ndarray:
        self.load_model()
        assert self.model is not None
        batch = torch.stack([self.preprocess(im) for im in images])
        batch = batch.to(self.device, dtype=torch_dtype(self.precision), non_blocking=True)
        desc = self.pool(self.model.forward_features(batch), self.pooling).float().cpu()
        return l2_normalize(desc).numpy().astype(np.float32)

    def embedding_dim(self) -> int:
        return self.spec.embed_dim

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "checkpoint": self.checkpoint,
            "source_sha256": self.weights_meta.get("sha256"),
            "arch": f"{self.spec.builder} (official dinov3 code)",
            "patch_size": self.spec.patch_size,
            "image_size": self.image_size,
            "resize_mode": self.resize_mode,
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "pooling": self.pooling,
            "l2_normalized": True,
            "embedding_dim": self.embedding_dim(),
            "runtime": {"device": self.device, "precision": self.precision},
        }


def from_config(config: dict[str, Any]) -> DINOv3Encoder:
    from latent_loop.utils.config import resolve_path  # noqa: PLC0415

    enc = config.get("dinov3", {})
    return DINOv3Encoder(
        checkpoint=enc.get("checkpoint", "vitb16"),
        checkpoint_dir=resolve_path(config["paths"].get("checkpoints_dir", "checkpoints")),
        repo_dir=resolve_path(enc.get("repo_dir", "external/dinov3")),
        device=config.get("device", "auto"),
        precision=enc.get("precision", "auto"),
        resize_mode=enc.get("resize_mode", "squash"),
        image_size=int(enc.get("image_size") or 224),
        pooling=enc.get("pooling", "cls"),
    )
