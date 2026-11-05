"""Frozen pretrained DINOv2 ViT-B/14 (Oquab et al., TMLR 2024) as a global place descriptor.

Weights are the official backbone checkpoint ``dinov2_vitb14_pretrain.pth`` from
https://github.com/facebookresearch/dinov2 (Apache-2.0). The network is re-implemented
here with parameter names that match the checkpoint, so it loads with ``strict=True``;
``scripts/verify_dinov2.py`` checks the outputs against Meta's reference code.

Architecture (ViT-B/14, no registers): patch 14, width 768, 12 blocks, 12 heads, MLP
ratio 4, LayerNorm eps 1e-6, LayerScale on both residual branches, a learnable [CLS]
token and *learned* absolute position embeddings for a 37 x 37 grid (518 px). Other grids
use the official bicubic interpolation with a +0.1 scale offset and no antialiasing.

Descriptor: the final-LayerNorm [CLS] token (``pooling="cls"``, default; the token
DINOv2's image-level objective is trained on) or the mean of the patch tokens
(``pooling="mean"``), followed by L2 normalisation. Preprocessing matches I-JEPA for a
like-for-like comparison: ImageNet normalisation and the same resize modes at 224 px
(16 x 16 patches).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from latent_loop.encoders.base import VisualEncoder
from latent_loop.encoders.ijepa import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESIZE_MODES,
    resize_image,
    sha256_file,
    to_rgb,
)
from latent_loop.encoders.vit import MLP, Attention, PatchEmbed
from latent_loop.features.pooling import l2_normalize
from latent_loop.utils.device import resolve_device, resolve_precision, torch_dtype
from latent_loop.utils.logging import get_logger

logger = get_logger(__name__)

POOLINGS = ("cls", "mean")


@dataclass(frozen=True)
class DinoSpec:
    key: str
    embed_dim: int
    depth: int
    num_heads: int
    patch_size: int
    train_size: int  # resolution of the stored position-embedding grid
    filename: str
    url: str
    size_bytes: int


CHECKPOINTS: dict[str, DinoSpec] = {
    "vitb14": DinoSpec(
        "vitb14",
        768,
        12,
        12,
        14,
        518,
        "dinov2_vitb14_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        346_378_731,
    ),
}


def checkpoint_path(checkpoint_dir: str | Path, key: str) -> Path:
    return Path(checkpoint_dir) / "dinov2" / CHECKPOINTS[key].filename


class LayerScale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class DinoBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads, qkv_bias=True)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, 4 * dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoViT(nn.Module):
    """DINOv2 backbone returning final-LayerNorm tokens ``(B, 1 + N, D)`` ([CLS] first)."""

    def __init__(self, spec: DinoSpec) -> None:
        super().__init__()
        self.spec = spec
        d = spec.embed_dim
        grid = spec.train_size // spec.patch_size
        self.patch_embed = PatchEmbed(spec.patch_size, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid * grid, d))
        self.mask_token = nn.Parameter(
            torch.zeros(1, d)
        )  # used only in training; kept for strict load
        self.blocks = nn.ModuleList(DinoBlock(d, spec.num_heads) for _ in range(spec.depth))
        self.norm = nn.LayerNorm(d, eps=1e-6)

    def interpolate_pos_encoding(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Official scheme: bicubic resize of the 37 x 37 grid with scale (g + 0.1) / 37."""
        n = self.pos_embed.shape[1] - 1
        m = int(math.sqrt(n))
        if grid_h == m and grid_w == m:
            return self.pos_embed
        pos = self.pos_embed.float()
        cls_pos, patch_pos = pos[:, :1], pos[:, 1:]
        d = pos.shape[-1]
        patch_pos = F.interpolate(
            patch_pos.reshape(1, m, m, d).permute(0, 3, 1, 2),
            scale_factor=((grid_h + 0.1) / m, (grid_w + 0.1) / m),
            mode="bicubic",
            antialias=False,
        )
        if patch_pos.shape[-2:] != (grid_h, grid_w):
            raise RuntimeError(f"pos-embed interpolation gave {tuple(patch_pos.shape[-2:])}")
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, d)
        return torch.cat([cls_pos, patch_pos], dim=1).to(self.pos_embed.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.spec.patch_size
        if x.shape[-2] % p or x.shape[-1] % p:
            raise ValueError(f"input size {tuple(x.shape[-2:])} is not divisible by patch {p}")
        tokens = self.patch_embed(x)
        tokens = torch.cat([self.cls_token.expand(x.shape[0], -1, -1).to(tokens.dtype), tokens], 1)
        tokens = tokens + self.interpolate_pos_encoding(x.shape[-2] // p, x.shape[-1] // p)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


class DINOv2Encoder(VisualEncoder):
    """Frozen DINOv2 backbone + [CLS] (or patch-mean) pooling + L2 normalisation."""

    name = "dinov2"

    def __init__(
        self,
        checkpoint: str = "vitb14",
        checkpoint_dir: str | Path = "checkpoints",
        device: str = "auto",
        precision: str = "auto",
        resize_mode: str = "squash",
        image_size: int = 224,
        pooling: str = "cls",
        weights_path: str | Path | None = None,
    ) -> None:
        if checkpoint not in CHECKPOINTS:
            raise ValueError(
                f"Unknown DINOv2 checkpoint {checkpoint!r}; known: {sorted(CHECKPOINTS)}"
            )
        if resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {RESIZE_MODES}")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}")
        self.spec = CHECKPOINTS[checkpoint]
        if image_size % self.spec.patch_size:
            raise ValueError(f"image_size must be a multiple of {self.spec.patch_size}")
        self.checkpoint = checkpoint
        self.weights_path = (
            Path(weights_path) if weights_path else checkpoint_path(checkpoint_dir, checkpoint)
        )
        self.resize_mode = resize_mode
        self.image_size = int(image_size)
        self.pooling = pooling
        self.device = resolve_device(device)
        self.precision = resolve_precision(precision, self.device)
        self.model: DinoViT | None = None
        sidecar = self.weights_path.with_suffix(".json")
        self.weights_meta: dict[str, Any] = (
            json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        )
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def load_model(self) -> None:
        if self.model is not None:
            return
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"DINOv2 weights not found: {self.weights_path}\n"
                f"Run: python scripts/download_checkpoints.py dinov2 --name {self.checkpoint}"
            )
        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        model = DinoViT(self.spec)
        model.load_state_dict(state, strict=True)  # every tensor must match
        model.requires_grad_(False).eval()
        self.model = model.to(self.device, dtype=torch_dtype(self.precision))
        logger.info(
            "Loaded DINOv2 backbone",
            extra={
                "checkpoint": self.checkpoint,
                "params_M": round(self.num_params() / 1e6, 1),
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

    @torch.inference_mode()
    def encode_tokens(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        """Final-LayerNorm tokens ``(B, 1 + N, D)`` as float32 on the CPU."""
        self.load_model()
        assert self.model is not None
        batch = torch.stack([self.preprocess(im) for im in images])
        batch = batch.to(self.device, dtype=torch_dtype(self.precision), non_blocking=True)
        return self.model(batch).float().cpu()

    @torch.inference_mode()
    def encode_batch(self, images: Sequence[np.ndarray]) -> np.ndarray:
        tokens = self.encode_tokens(images)
        desc = tokens[:, 0] if self.pooling == "cls" else tokens[:, 1:].mean(1)
        return l2_normalize(desc).numpy().astype(np.float32)

    def embedding_dim(self) -> int:
        return self.spec.embed_dim

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "checkpoint": self.checkpoint,
            "source_sha256": self.weights_meta.get("sha256"),
            "arch": f"vit_base patch{self.spec.patch_size}",
            "patch_size": self.spec.patch_size,
            "image_size": self.image_size,
            "resize_mode": self.resize_mode,
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "pooling": self.pooling,
            "l2_normalized": True,
            "embedding_dim": self.embedding_dim(),
            "runtime": {"device": self.device, "precision": self.precision},
        }


def write_sidecar(weights_path: str | Path) -> dict[str, Any]:
    """Record the checkpoint's SHA-256 and size next to it (part of the cache fingerprint)."""
    weights_path = Path(weights_path)
    meta = {
        "file": weights_path.name,
        "bytes": weights_path.stat().st_size,
        "sha256": sha256_file(weights_path),
    }
    weights_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def from_config(config: dict[str, Any]) -> DINOv2Encoder:
    from latent_loop.utils.config import resolve_path  # noqa: PLC0415

    enc = config.get("dinov2", {})
    return DINOv2Encoder(
        checkpoint=enc.get("checkpoint", "vitb14"),
        checkpoint_dir=resolve_path(config["paths"].get("checkpoints_dir", "checkpoints")),
        device=config.get("device", "auto"),
        precision=enc.get("precision", "auto"),
        resize_mode=enc.get("resize_mode", "squash"),
        image_size=int(enc.get("image_size") or 224),
        pooling=enc.get("pooling", "cls"),
    )
