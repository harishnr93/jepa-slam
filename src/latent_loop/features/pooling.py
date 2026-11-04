"""Global descriptor pooling and L2 normalisation.

I-JEPA has no class token, so the global image descriptor is the mean of the
last-layer patch tokens (after the final LayerNorm), then L2-normalised:

    g = (1/N) sum_i z_i,        d = g / max(||g||_2, eps)

For unit vectors the inner product equals cosine similarity, <d_a, d_b> = cos(theta),
which is what FAISS ``IndexFlatIP`` will exploit in Phase 3.
"""

from __future__ import annotations

import numpy as np
import torch

POOLINGS = ("mean",)


def pool_tokens(tokens: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Pool ``(B, N, D)`` patch tokens into ``(B, D)`` descriptors, computed in float32."""
    if method not in POOLINGS:
        raise ValueError(f"Unknown pooling {method!r}; available: {POOLINGS}")
    return tokens.float().mean(dim=1)


def l2_normalize(x: np.ndarray | torch.Tensor, eps: float = 1e-12) -> np.ndarray | torch.Tensor:
    """Row-wise L2 normalisation. All-zero rows stay zero instead of becoming NaN."""
    if isinstance(x, torch.Tensor):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)
