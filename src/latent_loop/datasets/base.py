"""Common dataset interface and frame sampling.

Every dataset exposes a *sampled view* of a camera sequence. Indices passed to
``get_frame``/``get_pose``/``get_timestamp`` address that view (``0 .. len-1``), while
``frame_id(index)`` returns the original frame number in the source sequence. Later
phases must use original frame IDs or timestamps (never view indices) for temporal
reasoning, because the view may be strided and may skip missing frames.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SamplingConfig:
    """Frame sampling parameters, all expressed in *original* frame IDs.

    A frame with ID ``f`` is kept iff ``start <= f < end`` and ``(f - start) % stride == 0``,
    and it is available (image present). The first ``max_frames`` kept frames are used.
    Selecting on the ID grid rather than on positions in the list keeps sampled frames
    evenly spaced in time even when some source frames are missing.
    """

    start: int = 0
    end: int | None = None
    stride: int = 1
    max_frames: int | None = None

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end is not None and self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {self.max_frames}")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> SamplingConfig:
        values = dict(values or {})
        unknown = set(values) - {"start", "end", "stride", "max_frames"}
        if unknown:
            raise ValueError(f"Unknown sampling keys: {sorted(unknown)}")
        return cls(**values)


def sample_frame_ids(available_ids: np.ndarray, sampling: SamplingConfig) -> np.ndarray:
    """Apply ``sampling`` to sorted, unique original frame IDs."""
    ids = np.asarray(available_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("available_ids must be one-dimensional")
    if ids.size and np.any(np.diff(ids) <= 0):
        raise ValueError("available_ids must be strictly increasing")
    mask = ids >= sampling.start
    if sampling.end is not None:
        mask &= ids < sampling.end
    mask &= (ids - sampling.start) % sampling.stride == 0
    selected = ids[mask]
    if sampling.max_frames is not None:
        selected = selected[: sampling.max_frames]
    return selected


@dataclass(frozen=True)
class CameraCalibration:
    """Rectified pinhole calibration of the camera that produced the images.

    ``K`` is the 3x3 intrinsic matrix, ``P`` the 3x4 projection matrix and
    ``T_cam0_cam`` the rigid transform from this camera into the reference camera 0
    (identity for camera 0 and for the synthetic dataset).
    """

    K: np.ndarray
    image_size: tuple[int, int]  # (width, height) in pixels
    P: np.ndarray | None = None
    T_cam0_cam: np.ndarray = field(default_factory=lambda: np.eye(4))

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    def is_valid(self) -> bool:
        width, height = self.image_size
        return bool(
            self.K.shape == (3, 3)
            and np.all(np.isfinite(self.K))
            and self.fx > 0
            and self.fy > 0
            and 0 <= self.cx <= width
            and 0 <= self.cy <= height
        )


@dataclass
class Frame:
    """One sampled observation. ``image`` is HxW (grayscale) or HxWx3 RGB ``uint8``."""

    index: int
    frame_id: int
    timestamp: float
    image: np.ndarray
    pose: np.ndarray | None  # 4x4 T_world_camera, None when ground truth is unavailable


class VisualSequenceDataset(ABC):
    """Abstract sampled view over one camera sequence with optional ground-truth poses."""

    name: str = "base"

    def __init__(self, available_frame_ids: np.ndarray, sampling: SamplingConfig | None = None):
        self.sampling = sampling or SamplingConfig()
        self.available_frame_ids = np.asarray(available_frame_ids, dtype=np.int64)
        self._frame_ids = sample_frame_ids(self.available_frame_ids, self.sampling)

    # --- subclass hooks -------------------------------------------------------------
    @abstractmethod
    def _load_image(self, frame_id: int) -> np.ndarray: ...

    @abstractmethod
    def _pose_for(self, frame_id: int) -> np.ndarray | None: ...

    @abstractmethod
    def _timestamp_for(self, frame_id: int) -> float: ...

    @abstractmethod
    def get_calibration(self) -> CameraCalibration: ...

    @property
    @abstractmethod
    def has_poses(self) -> bool: ...

    # --- public interface -----------------------------------------------------------
    def __len__(self) -> int:
        return int(self._frame_ids.size)

    def __iter__(self) -> Iterator[Frame]:
        for index in range(len(self)):
            yield self.get_frame(index)

    @property
    def frame_ids(self) -> np.ndarray:
        """Original frame IDs of the sampled view (read-only copy)."""
        return self._frame_ids.copy()

    def _check_index(self, index: int) -> int:
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} out of range for dataset of length {len(self)}")
        return index

    def frame_id(self, index: int) -> int:
        return int(self._frame_ids[self._check_index(index)])

    def get_image(self, index: int) -> np.ndarray:
        return self._load_image(self.frame_id(index))

    def get_pose(self, index: int) -> np.ndarray | None:
        pose = self._pose_for(self.frame_id(index))
        return None if pose is None else pose.copy()

    def get_timestamp(self, index: int) -> float:
        return float(self._timestamp_for(self.frame_id(index)))

    def get_frame(self, index: int) -> Frame:
        index = self._check_index(index)
        return Frame(
            index=index,
            frame_id=self.frame_id(index),
            timestamp=self.get_timestamp(index),
            image=self.get_image(index),
            pose=self.get_pose(index),
        )

    def timestamps(self) -> np.ndarray:
        return np.array([self._timestamp_for(int(f)) for f in self._frame_ids], dtype=np.float64)

    def poses(self) -> np.ndarray:
        """``(N, 4, 4)`` poses of the sampled view; raises if ground truth is missing."""
        if not self.has_poses:
            raise RuntimeError(f"{self.name} dataset has no ground-truth poses")
        return np.stack([self._pose_for(int(f)) for f in self._frame_ids])

    def summary(self) -> dict[str, Any]:
        calib = self.get_calibration()
        return {
            "dataset": self.name,
            "available_frames": int(self.available_frame_ids.size),
            "sampled_frames": len(self),
            "first_frame_id": int(self._frame_ids[0]) if len(self) else None,
            "last_frame_id": int(self._frame_ids[-1]) if len(self) else None,
            "sampling": vars(self.sampling),
            "has_poses": self.has_poses,
            "image_size": calib.image_size,
            "fx": calib.fx,
            "fy": calib.fy,
        }
