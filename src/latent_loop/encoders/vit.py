"""Minimal Vision Transformer compatible with the official I-JEPA encoder checkpoints.

An independent implementation of the standard ViT encoder used by I-JEPA
(Assran et al., CVPR 2023): patch embedding, *fixed* 2-D sine-cosine position
embeddings, pre-norm transformer blocks, a final LayerNorm and **no class token**.
Parameter names match the official checkpoints, so their ``target_encoder`` state
dict loads with ``strict=True``.

Differences from the reference code that do not change the function:

* attention uses ``F.scaled_dot_product_attention`` (fused, memory-efficient kernels);
* position embeddings are generated for the actual patch grid, so non-square inputs
  work (the official ``interpolate_pos_encoding`` assumes a class token).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ViTConfig:
    embed_dim: int
    depth: int
    num_heads: int
    patch_size: int
    img_size: int = 224  # pre-training resolution: defines the stored pos_embed grid
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    norm_eps: float = 1e-6

    @property
    def train_grid(self) -> int:
        return self.img_size // self.patch_size


ARCHITECTURES: dict[str, dict[str, Any]] = {
    "vit_huge": {"embed_dim": 1280, "depth": 32, "num_heads": 16},
    "vit_giant": {"embed_dim": 1408, "depth": 40, "num_heads": 16, "mlp_ratio": 48 / 11},
}


def sincos_pos_embed_2d(embed_dim: int, grid_h: int, grid_w: int) -> np.ndarray:
    """Fixed 2-D sine-cosine embedding of shape ``(grid_h * grid_w, embed_dim)``.

    MAE / I-JEPA convention: the first half of the channels encodes the **column**
    index, the second half the **row** index. Each half is ``[sin(p w_k), cos(p w_k)]``
    with ``w_k = 10000^(-k / (D/4))``, ``k = 0 .. D/4 - 1``. Tokens are row-major.
    """
    if embed_dim % 4:
        raise ValueError("embed_dim must be divisible by 4")
    cols, rows = np.meshgrid(
        np.arange(grid_w, dtype=np.float64), np.arange(grid_h, dtype=np.float64)
    )

    def encode_1d(pos: np.ndarray, dim: int) -> np.ndarray:
        omega = 1.0 / 10000 ** (np.arange(dim // 2, dtype=np.float64) / (dim / 2.0))
        out = np.outer(pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    half = embed_dim // 2
    return np.concatenate([encode_1d(cols, half), encode_1d(rows, half)], axis=1)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int, in_chans: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)  # softmax(Q K^T / sqrt(d_head)) V
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim, eps=cfg.norm_eps)
        self.attn = Attention(cfg.embed_dim, cfg.num_heads, cfg.qkv_bias)
        self.norm2 = nn.LayerNorm(cfg.embed_dim, eps=cfg.norm_eps)
        self.mlp = MLP(cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    """ViT encoder returning final-LayerNorm patch tokens ``(B, N, D)``."""

    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.patch_size, cfg.embed_dim)
        g = cfg.train_grid
        # Stored in the checkpoint but fixed (never trained); verified against sincos on load.
        self.pos_embed = nn.Parameter(
            torch.from_numpy(sincos_pos_embed_2d(cfg.embed_dim, g, g)).float().unsqueeze(0),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.norm_eps)

    def pos_embed_for(self, grid_h: int, grid_w: int) -> torch.Tensor:
        g = self.cfg.train_grid
        if (grid_h, grid_w) == (g, g):
            return self.pos_embed
        pe = torch.from_numpy(sincos_pos_embed_2d(self.cfg.embed_dim, grid_h, grid_w))
        return pe.unsqueeze(0).to(self.pos_embed.device, self.pos_embed.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.cfg.patch_size
        if x.shape[-2] % p or x.shape[-1] % p:
            raise ValueError(f"input size {tuple(x.shape[-2:])} is not divisible by patch {p}")
        x = self.patch_embed(x) + self.pos_embed_for(x.shape[-2] // p, x.shape[-1] // p)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def build_vit(arch: str | dict[str, Any], patch_size: int, img_size: int) -> VisionTransformer:
    params = dict(ARCHITECTURES[arch]) if isinstance(arch, str) else dict(arch)
    return VisionTransformer(ViTConfig(patch_size=patch_size, img_size=img_size, **params))
