"""Dataset abstractions, KITTI Odometry loading and synthetic sequences."""

from __future__ import annotations

from typing import Any

from latent_loop.datasets.base import (
    CameraCalibration,
    Frame,
    SamplingConfig,
    VisualSequenceDataset,
    sample_frame_ids,
)
from latent_loop.datasets.kitti import KittiOdometryDataset
from latent_loop.datasets.synthetic import SyntheticConfig, SyntheticLoopDataset
from latent_loop.utils.config import resolve_path

__all__ = [
    "CameraCalibration",
    "Frame",
    "KittiOdometryDataset",
    "SamplingConfig",
    "SyntheticConfig",
    "SyntheticLoopDataset",
    "VisualSequenceDataset",
    "build_dataset",
    "sample_frame_ids",
]


def build_dataset(config: dict[str, Any]) -> VisualSequenceDataset:
    """Instantiate the dataset described by ``config["dataset"]``."""
    ds_cfg = config["dataset"]
    sampling = SamplingConfig.from_dict(ds_cfg.get("sampling"))
    name = ds_cfg["name"]
    if name == "synthetic":
        return SyntheticLoopDataset(SyntheticConfig.from_dict(ds_cfg.get("synthetic")), sampling)
    if name == "kitti":
        return KittiOdometryDataset(
            root=resolve_path(ds_cfg["root"]),
            sequence=ds_cfg.get("sequence", "00"),
            camera=int(ds_cfg.get("camera", 2)),
            sampling=sampling,
        )
    raise ValueError(f"Unknown dataset name: {name!r}")
