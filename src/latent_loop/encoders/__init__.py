"""Visual encoders producing L2-normalised global place descriptors."""

from __future__ import annotations

from typing import Any

from latent_loop.encoders.base import VisualEncoder

AVAILABLE_ENCODERS = ("ijepa",)


def build_encoder(name: str, config: dict[str, Any]) -> VisualEncoder:
    """Instantiate an encoder by name; heavy dependencies are imported lazily."""
    if name == "ijepa":
        from latent_loop.encoders import ijepa  # noqa: PLC0415

        return ijepa.from_config(config)
    raise ValueError(f"Unknown encoder {name!r}; available: {AVAILABLE_ENCODERS}")


__all__ = ["AVAILABLE_ENCODERS", "VisualEncoder", "build_encoder"]
