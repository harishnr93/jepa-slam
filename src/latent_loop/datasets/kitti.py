"""KITTI Odometry loader.

Layout (``root`` is the ``dataset`` directory of the official archives)::

    root/sequences/XX/image_{0,1,2,3}/NNNNNN.png   images (0/1 grayscale, 2/3 colour)
    root/sequences/XX/calib.txt                    P0..P3 (3x4) and Tr (velodyne->cam0)
    root/sequences/XX/times.txt                    one timestamp [s] per frame
    root/poses/XX.txt                              ground truth, sequences 00-10 only

Frame ``k`` is line ``k`` of ``times.txt`` and of ``poses/XX.txt`` and the image
``NNNNNN.png`` with ``NNNNNN = k``. That shared index is the original frame ID kept by
the sampled view, so image/pose alignment never depends on the sampling.

Ground-truth poses describe camera 0. For camera ``i`` the pose is
``T_w_cami = T_w_cam0 @ T_cam0_cami``, where the rectified projection matrices give
``P_i = K [I | b_i]`` and therefore ``T_cam0_cami = [I | -K^-1 P_i[:, 3]]``.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from latent_loop.datasets.base import CameraCalibration, SamplingConfig, VisualSequenceDataset
from latent_loop.datasets.ground_truth import load_kitti_poses
from latent_loop.utils.logging import get_logger

logger = get_logger(__name__)

KITTI_FRAME_RATE_HZ = 10.0
_GRAYSCALE_CAMERAS = {0, 1}


def parse_calibration_file(path: str | Path) -> dict[str, np.ndarray]:
    """Parse ``calib.txt`` into ``{"P0": 3x4, ..., "Tr": 4x4}``."""
    path = Path(path)
    matrices: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            key, sep, rest = line.partition(":")
            if not sep:
                raise ValueError(f"{path}:{line_no}: expected 'KEY: values'")
            values = np.asarray(rest.split(), dtype=np.float64)
            if values.size != 12:
                raise ValueError(f"{path}:{line_no}: {key} has {values.size} values, expected 12")
            matrix = values.reshape(3, 4)
            if key.strip() == "Tr":
                matrix = np.vstack([matrix, [0.0, 0.0, 0.0, 1.0]])
            matrices[key.strip()] = matrix
    return matrices


def calibration_from_projection(
    projection: np.ndarray, image_size: tuple[int, int]
) -> CameraCalibration:
    """Build a calibration (K and camera->cam0 transform) from a rectified 3x4 ``P``."""
    K = projection[:, :3].copy()
    if abs(np.linalg.det(K)) < 1e-9:
        raise ValueError("Projection matrix has a singular intrinsic block")
    offset = np.linalg.solve(K, projection[:, 3])  # b_i: cam0 point -> cam_i point offset
    T_cam0_cam = np.eye(4)
    T_cam0_cam[:3, 3] = -offset
    return CameraCalibration(K=K, image_size=image_size, P=projection.copy(), T_cam0_cam=T_cam0_cam)


def _list_image_ids(image_dir: Path) -> np.ndarray:
    ids = []
    with os.scandir(image_dir) as entries:
        for entry in entries:
            stem, ext = os.path.splitext(entry.name)
            if ext.lower() == ".png" and stem.isdigit():
                ids.append(int(stem))
    return np.array(sorted(ids), dtype=np.int64)


class KittiOdometryDataset(VisualSequenceDataset):
    """Lazy-loading view of one KITTI Odometry sequence and camera."""

    name = "kitti"

    def __init__(
        self,
        root: str | Path,
        sequence: str | int = "00",
        camera: int = 2,
        sampling: SamplingConfig | None = None,
        poses_path: str | Path | None = None,
    ):
        if camera not in (0, 1, 2, 3):
            raise ValueError(f"camera must be 0-3, got {camera}")
        self.root = Path(root)
        self.sequence = f"{int(sequence):02d}"
        self.camera = camera
        self.sequence_dir = self.root / "sequences" / self.sequence
        self.image_dir = self.sequence_dir / f"image_{camera}"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        image_ids = _list_image_ids(self.image_dir)
        if image_ids.size == 0:
            raise FileNotFoundError(f"No NNNNNN.png images in {self.image_dir}")

        self._timestamps, self.timestamps_estimated = self._load_timestamps(image_ids)
        num_frames = self._timestamps.size
        self.missing_frame_ids = np.setdiff1d(np.arange(num_frames), image_ids)
        if self.missing_frame_ids.size:
            logger.warning(
                "Frames without images are skipped",
                extra={"sequence": self.sequence, "missing": int(self.missing_frame_ids.size)},
            )
        available = image_ids[image_ids < num_frames]
        if available.size < image_ids.size:
            logger.warning(
                "Images beyond the end of times.txt are ignored",
                extra={"ignored": int(image_ids.size - available.size)},
            )

        self._poses = self._load_poses(poses_path, num_frames)
        self._calibration = self._load_calibration(int(available[0]))
        super().__init__(available, sampling)
        logger.info(
            "Loaded KITTI sequence",
            extra={
                "sequence": self.sequence,
                "camera": camera,
                "frames": len(self),
                "has_poses": self.has_poses,
            },
        )

    # --- loading helpers ------------------------------------------------------------
    def _load_timestamps(self, image_ids: np.ndarray) -> tuple[np.ndarray, bool]:
        times_path = self.sequence_dir / "times.txt"
        if times_path.exists():
            times = np.loadtxt(times_path, dtype=np.float64, ndmin=1)
            if np.any(np.diff(times) <= 0):
                raise ValueError(f"Timestamps in {times_path} are not strictly increasing")
            return times, False
        logger.warning(
            "times.txt missing; timestamps estimated from the nominal frame rate",
            extra={"frame_rate_hz": KITTI_FRAME_RATE_HZ},
        )
        return np.arange(int(image_ids[-1]) + 1) / KITTI_FRAME_RATE_HZ, True

    def _load_poses(self, poses_path: str | Path | None, num_frames: int) -> np.ndarray | None:
        path = Path(poses_path) if poses_path else self.root / "poses" / f"{self.sequence}.txt"
        if not path.exists():
            logger.warning("No ground-truth poses for this sequence", extra={"path": str(path)})
            return None
        poses = load_kitti_poses(path)
        if len(poses) != num_frames:
            raise ValueError(
                f"Pose/frame misalignment: {path} has {len(poses)} poses "
                f"but the sequence has {num_frames} frames"
            )
        return poses

    def _load_calibration(self, sample_frame_id: int) -> CameraCalibration:
        calib_path = self.sequence_dir / "calib.txt"
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")
        matrices = parse_calibration_file(calib_path)
        key = f"P{self.camera}"
        if key not in matrices:
            raise ValueError(f"{calib_path} does not define {key}")
        height, width = self._load_image(sample_frame_id).shape[:2]
        return calibration_from_projection(matrices[key], (width, height))

    # --- VisualSequenceDataset hooks ------------------------------------------------
    def _load_image(self, frame_id: int) -> np.ndarray:
        path = self.image_dir / f"{frame_id:06d}.png"
        if self.camera in _GRAYSCALE_CAMERAS:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            raise FileNotFoundError(f"Could not read image {path}")
        return image

    def _pose_for(self, frame_id: int) -> np.ndarray | None:
        if self._poses is None:
            return None
        return self._poses[frame_id] @ self._calibration.T_cam0_cam

    def _timestamp_for(self, frame_id: int) -> float:
        return float(self._timestamps[frame_id])

    def get_calibration(self) -> CameraCalibration:
        return self._calibration

    @property
    def has_poses(self) -> bool:
        return self._poses is not None
