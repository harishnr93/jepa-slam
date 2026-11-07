"""Visual encoders producing L2-normalised global place descriptors."""

from __future__ import annotations

from typing import Any

from latent_loop.encoders.base import VisualEncoder

AVAILABLE_ENCODERS = ("ijepa", "dinov2", "dinov3", "hsv", "orb_bovw")


def build_encoder(name: str, config: dict[str, Any]) -> VisualEncoder:
    """Instantiate an encoder by name; heavy dependencies are imported lazily."""
    if name == "ijepa":
        from latent_loop.encoders import ijepa  # noqa: PLC0415

        return ijepa.from_config(config)
    if name == "dinov2":
        from latent_loop.encoders import dinov2  # noqa: PLC0415

        return dinov2.from_config(config)
    if name == "dinov3":
        from latent_loop.encoders import dinov3  # noqa: PLC0415

        return dinov3.from_config(config)
    if name == "hsv":
        from latent_loop.encoders import classical  # noqa: PLC0415

        return classical.hsv_from_config(config)
    if name == "orb_bovw":
        from latent_loop.encoders import classical  # noqa: PLC0415

        return classical.orb_bovw_from_config(config)
    raise ValueError(f"Unknown encoder {name!r}; available: {AVAILABLE_ENCODERS}")


__all__ = ["AVAILABLE_ENCODERS", "VisualEncoder", "build_encoder"]
