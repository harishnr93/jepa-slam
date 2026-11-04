"""Common interface for global visual descriptors (neural and classical)."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np


class VisualEncoder(ABC):
    """Maps ``uint8`` images (``HxWx3`` RGB or ``HxW`` gray) to L2-normalised
    ``(N, D)`` float32 descriptors."""

    name: str = "encoder"

    @abstractmethod
    def load_model(self) -> None:
        """Load weights / vocabularies. Must be idempotent."""

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Any:
        """Convert one image into the encoder's input format."""

    @abstractmethod
    def encode_batch(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Encode a batch of images into L2-normalised float32 descriptors."""

    @abstractmethod
    def embedding_dim(self) -> int: ...

    def info(self) -> dict[str, Any]:
        """Everything that determines descriptor values, plus a ``runtime`` entry."""
        return {"name": self.name}

    def fingerprint(self) -> str:
        """Short hash of ``info()`` without ``runtime`` (device / precision).

        Weights, resolution, resize mode or pooling changes alter the fingerprint and
        therefore the cache directory; switching GPU <-> CPU does not.
        """
        stable = {k: v for k, v in self.info().items() if k != "runtime"}
        return hashlib.sha1(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:10]

    def unload(self) -> None:  # noqa: B027 — optional hook, no-op for stateless encoders
        """Release model memory (sequential model loading)."""

    def to_device(self, device: str, precision: str | None = None) -> None:
        """Move the encoder to another device (used for the CPU fallback)."""
        raise NotImplementedError(f"{self.name} does not support device changes")
