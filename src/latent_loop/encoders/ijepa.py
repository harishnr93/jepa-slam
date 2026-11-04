"""Frozen pretrained I-JEPA encoder (Assran et al., CVPR 2023) as a global place descriptor.

Checkpoints are the official *full training states* from
https://github.com/facebookresearch/ijepa (weights licensed CC BY-NC 4.0). Each file
holds ``encoder`` (context encoder), ``predictor``, ``target_encoder`` (the EMA copy of
the context encoder) and optimiser state. The paper evaluates the **target encoder**,
so :func:`extract_target_encoder` keeps only that network.

Preprocessing. The official repository only defines *training* transforms
(``RandomResizedCrop(224)`` followed by ImageNet mean/std normalisation). For
deterministic inference we keep the pre-training resolution and normalisation and offer
three geometric modes:

* ``squash`` (default): resize the full frame to S x S. Keeps the entire field of view
  of wide driving images (KITTI is 3.3:1) at the cost of anisotropic scaling.
* ``center_crop``: ImageNet evaluation style, short side to S then an S x S centre crop.
  Discards ~70 % of a KITTI frame.
* ``aspect``: height S, width rounded to a multiple of the patch size; position
  embeddings are generated for the wider grid (columns beyond the pre-training grid are
  extrapolated).
"""

from __future__ import annotations

import hashlib
import json
import mmap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from latent_loop.encoders.base import VisualEncoder
from latent_loop.encoders.vit import (
    ARCHITECTURES,
    VisionTransformer,
    build_vit,
    sincos_pos_embed_2d,
)
from latent_loop.features.pooling import l2_normalize, pool_tokens
from latent_loop.utils.device import resolve_device, resolve_precision, torch_dtype
from latent_loop.utils.logging import get_logger

logger = get_logger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_MODES = ("squash", "center_crop", "aspect")
POS_EMBED_ATOL = 1e-3


@dataclass(frozen=True)
class CheckpointSpec:
    key: str
    filename: str
    arch: str
    patch_size: int
    img_size: int
    pretraining: str

    @property
    def url(self) -> str:
        return f"https://dl.fbaipublicfiles.com/ijepa/{self.filename}"


CHECKPOINTS: dict[str, CheckpointSpec] = {
    s.key: s
    for s in (
        CheckpointSpec(
            "vith14_in1k",
            "IN1K-vit.h.14-300e.pth.tar",
            "vit_huge",
            14,
            224,
            "ImageNet-1K, 300 epochs",
        ),
        CheckpointSpec(
            "vith16_448_in1k",
            "IN1K-vit.h.16-448px-300e.pth.tar",
            "vit_huge",
            16,
            448,
            "ImageNet-1K, 300 epochs, 448 px",
        ),
        CheckpointSpec(
            "vith14_in22k",
            "IN22K-vit.h.14-900e.pth.tar",
            "vit_huge",
            14,
            224,
            "ImageNet-22K, 900 epochs",
        ),
        CheckpointSpec(
            "vitg16_in22k",
            "IN22K-vit.g.16-600e.pth.tar",
            "vit_giant",
            16,
            224,
            "ImageNet-22K, 600 epochs",
        ),
    )
}


def target_encoder_path(checkpoint_dir: str | Path, key: str) -> Path:
    return Path(checkpoint_dir) / "ijepa" / f"{key}_target_encoder.pt"


def find_raw_checkpoint(checkpoint_dir: str | Path, spec: CheckpointSpec) -> Path | None:
    for candidate in (
        Path(checkpoint_dir) / spec.filename,
        Path(checkpoint_dir) / "ijepa" / spec.filename,
    ):
        if candidate.is_file():
            return candidate
    return None


def sha256_file(path: str | Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_mmap(path: str | Path) -> dict[str, Any]:
    """``torch.load`` with a *shared, read-only-in-practice* memory map.

    PyTorch's default ``mmap=True`` uses a private copy-on-write mapping, which Linux
    charges against RAM + swap under the default heuristic overcommit. A 10.4 GB
    checkpoint therefore fails with ``ENOMEM`` on a 7.4 GiB machine. A shared mapping is
    backed by the file itself and is not charged. Nothing writes to the loaded tensors
    (they are copied before use), so the file is never modified.
    """
    with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def extract_target_encoder(
    raw_path: str | Path, out_path: str | Path, spec: CheckpointSpec | None = None
) -> dict[str, Any]:
    """Extract the ``target_encoder`` weights of an official full checkpoint.

    The raw file (~10.4 GB for ViT-H) is memory-mapped (shared), so peak RAM stays near the
    encoder size (~2.5 GB in FP32). The ``module.`` prefix that DistributedDataParallel
    adds is stripped and weights are stored in FP32 next to a JSON sidecar.
    """
    raw_path, out_path = Path(raw_path), Path(out_path)
    source_sha256 = sha256_file(raw_path)  # hash the file before it is mapped
    ckpt = load_mmap(raw_path)
    if "target_encoder" not in ckpt:
        raise KeyError(f"{raw_path} has no 'target_encoder' entry (keys: {sorted(ckpt)})")
    state = {
        k.removeprefix("module."): v.float().clone() for k, v in ckpt["target_encoder"].items()
    }
    meta: dict[str, Any] = {
        "source_file": raw_path.name,
        "source_bytes": raw_path.stat().st_size,
        "source_sha256": source_sha256,
        "source_keys": sorted(ckpt),
        "epoch": int(ckpt["epoch"]) if "epoch" in ckpt else None,
        "num_tensors": len(state),
        "num_params": int(sum(v.numel() for k, v in state.items() if k != "pos_embed")),
    }
    if spec is not None:
        meta.update(
            {
                "key": spec.key,
                "url": spec.url,
                "arch": spec.arch,
                "patch_size": spec.patch_size,
                "img_size": spec.img_size,
                "pretraining": spec.pretraining,
            }
        )
    del ckpt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    torch.save({"state_dict": state, "meta": meta}, tmp)
    tmp.replace(out_path)
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def to_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"expected HxW or HxWx3 image, got shape {image.shape}")


def resize_image(image: np.ndarray, mode: str, size: int, patch: int) -> np.ndarray:
    """Geometric preprocessing (see module docstring); returns ``uint8`` RGB."""
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if min(h, w) >= size else cv2.INTER_CUBIC
    if mode == "squash":
        return cv2.resize(image, (size, size), interpolation=interp)
    if mode == "center_crop":
        scale = size / min(h, w)
        nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=interp)
        y0, x0 = (nh - size) // 2, (nw - size) // 2
        return resized[y0 : y0 + size, x0 : x0 + size]
    if mode == "aspect":
        nw = max(patch, round(w * size / h / patch) * patch)
        return cv2.resize(image, (nw, size), interpolation=interp)
    raise ValueError(f"resize_mode must be one of {RESIZE_MODES}, got {mode!r}")


class IJEPAEncoder(VisualEncoder):
    """Frozen I-JEPA target encoder + patch-token mean pooling + L2 normalisation."""

    name = "ijepa"

    def __init__(
        self,
        checkpoint: str = "vith14_in1k",
        checkpoint_dir: str | Path = "checkpoints",
        device: str = "auto",
        precision: str = "auto",
        resize_mode: str = "squash",
        image_size: int | None = None,
        pooling: str = "mean",
        weights_path: str | Path | None = None,
        arch_override: dict[str, Any] | None = None,
    ) -> None:
        if checkpoint not in CHECKPOINTS and arch_override is None:
            raise ValueError(
                f"Unknown I-JEPA checkpoint {checkpoint!r}; known: {sorted(CHECKPOINTS)}"
            )
        if resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {RESIZE_MODES}, got {resize_mode!r}")
        self.spec = CHECKPOINTS.get(checkpoint)
        self.checkpoint = checkpoint
        self.weights_path = (
            Path(weights_path) if weights_path else target_encoder_path(checkpoint_dir, checkpoint)
        )
        arch = dict(arch_override or ARCHITECTURES[self.spec.arch])
        self.patch_size = int(arch.pop("patch_size", self.spec.patch_size if self.spec else 14))
        self.train_size = int(arch.pop("img_size", self.spec.img_size if self.spec else 224))
        self.arch = arch
        self.image_size = int(image_size or self.train_size)
        self.resize_mode = resize_mode
        self.pooling = pooling
        self.device = resolve_device(device)
        self.precision = resolve_precision(precision, self.device)
        self.model: VisionTransformer | None = None
        sidecar = self.weights_path.with_suffix(".json")
        self.weights_meta: dict[str, Any] = (
            json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        )
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    # --- model lifecycle ------------------------------------------------------------
    def load_model(self) -> None:
        if self.model is not None:
            return
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"I-JEPA target-encoder weights not found: {self.weights_path}\n"
                f"Run: python scripts/download_checkpoints.py ijepa --name {self.checkpoint}"
            )
        blob = load_mmap(self.weights_path)
        self.weights_meta = blob.get("meta", self.weights_meta)
        model = build_vit(self.arch, self.patch_size, self.train_size)
        model.load_state_dict(blob["state_dict"], strict=True)  # every tensor must match
        g = self.train_size // self.patch_size
        expected = torch.from_numpy(sincos_pos_embed_2d(model.cfg.embed_dim, g, g)).float()
        # The *target* encoder is an EMA, p <- m p + (1 - m) q, applied to every parameter
        # including the fixed pos_embed; FP32 rounding over ~187k updates drifts it by up to
        # 1.4e-4 (the context encoder's copy matches sincos to 3e-8). A wrong layout differs
        # by ~2, so 1e-3 separates the two cases. The stored values are used unchanged.
        drift = float((model.pos_embed[0] - expected).abs().max())
        if drift > POS_EMBED_ATOL:
            raise ValueError(
                f"Checkpoint pos_embed differs from the fixed 2-D sincos embedding by {drift:.3g}"
                f" (> {POS_EMBED_ATOL}); this is not the expected I-JEPA architecture."
            )
        self.pos_embed_drift = drift
        model.requires_grad_(False).eval()
        self.model = model.to(self.device, dtype=torch_dtype(self.precision))
        del blob
        logger.info(
            "Loaded I-JEPA target encoder",
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
        return sum(p.numel() for n, p in self.model.named_parameters() if n != "pos_embed")

    def to_device(self, device: str, precision: str | None = None) -> None:
        self.device = resolve_device(device)
        self.precision = resolve_precision(precision or "auto", self.device)
        if self.model is not None:
            self.model = self.model.to(self.device, dtype=torch_dtype(self.precision))

    def unload(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- inference ------------------------------------------------------------------
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """``uint8`` image -> normalised ``(3, H', W')`` float32 tensor."""
        rgb = resize_image(to_rgb(image), self.resize_mode, self.image_size, self.patch_size)
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        return (x - self._mean) / self._std

    def _forward(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        self.load_model()
        assert self.model is not None
        batch = torch.stack([self.preprocess(im) for im in images])
        batch = batch.to(self.device, dtype=torch_dtype(self.precision), non_blocking=True)
        return self.model(batch)

    @torch.inference_mode()
    def encode_tokens(self, images: Sequence[np.ndarray]) -> tuple[torch.Tensor, tuple[int, int]]:
        """Last-layer patch tokens ``(B, N, D)`` (float32, CPU) and the patch grid ``(h, w)``."""
        tokens = self._forward(images)
        h, w = self.preprocess(images[0]).shape[-2:]
        return tokens.float().cpu(), (h // self.patch_size, w // self.patch_size)

    @torch.inference_mode()
    def encode_batch(self, images: Sequence[np.ndarray]) -> np.ndarray:
        desc = l2_normalize(pool_tokens(self._forward(images), self.pooling))
        return desc.cpu().numpy().astype(np.float32)

    def embedding_dim(self) -> int:
        return int(self.arch["embed_dim"])

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "checkpoint": self.checkpoint,
            "source_sha256": self.weights_meta.get("source_sha256"),
            "arch": self.spec.arch if self.spec else self.arch,
            "patch_size": self.patch_size,
            "image_size": self.image_size,
            "resize_mode": self.resize_mode,
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "pooling": self.pooling,
            "l2_normalized": True,
            "embedding_dim": self.embedding_dim(),
            "runtime": {"device": self.device, "precision": self.precision},
        }


def from_config(config: dict[str, Any]) -> IJEPAEncoder:
    from latent_loop.utils.config import resolve_path  # noqa: PLC0415

    enc = config.get("ijepa", {})
    return IJEPAEncoder(
        checkpoint=enc.get("checkpoint", "vith14_in1k"),
        checkpoint_dir=resolve_path(config["paths"].get("checkpoints_dir", "checkpoints")),
        device=config.get("device", "auto"),
        precision=enc.get("precision", "auto"),
        resize_mode=enc.get("resize_mode", "squash"),
        image_size=enc.get("image_size"),
        pooling=enc.get("pooling", "mean"),
    )
