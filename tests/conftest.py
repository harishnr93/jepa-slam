"""Shared fixtures: a miniature KITTI Odometry sequence and a small synthetic dataset."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from latent_loop.datasets import SyntheticConfig, SyntheticLoopDataset

# Calibration in KITTI calib.txt format with realistic values (fx ~ 718 px, 0.54 m baseline,
# camera 2 offset ~6 cm from camera 0). Not a verified copy of an official file.
KITTI_CALIB = """\
P0: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 0.000000000000e+00 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.861448000000e+02 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 4.538225000000e+01 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 -1.130887000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 3.779761000000e-03
P3: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.372877000000e+02 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 2.369057000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 4.915215000000e-03
Tr: 4.276802385584e-04 -9.999672484946e-01 -8.084491683471e-03 -1.198459927713e-02 -7.210626507497e-03 8.081198471645e-03 -9.999413164504e-01 -5.403984729748e-02 9.999738645903e-01 4.859485810390e-04 -7.206933692422e-03 -2.921968648686e-01
"""

KITTI_NUM_FRAMES = 12
KITTI_IMAGE_SIZE = (64, 20)  # (width, height)
KITTI_MISSING_FRAME = 5


def kitti_pose(frame_id: int) -> np.ndarray:
    """Straight drive along +z at 1 m/frame with a slow yaw, so every pose is unique."""
    yaw = 0.01 * frame_id
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    pose[:3, 3] = [0.1 * frame_id, 0.0, float(frame_id)]
    return pose


def write_kitti_sequence(
    root: Path,
    sequence: str = "00",
    num_frames: int = KITTI_NUM_FRAMES,
    missing: tuple[int, ...] = (KITTI_MISSING_FRAME,),
    with_poses: bool = True,
    with_times: bool = True,
    pose_count: int | None = None,
) -> Path:
    """Write a tiny KITTI-layout sequence. Image pixels encode the frame ID (value = 10*id)."""
    seq_dir = root / "sequences" / sequence
    width, height = KITTI_IMAGE_SIZE
    for camera in (0, 2):
        image_dir = seq_dir / f"image_{camera}"
        image_dir.mkdir(parents=True, exist_ok=True)
        for frame_id in range(num_frames):
            if frame_id in missing:
                continue
            value = 10 * frame_id
            if camera == 0:
                image = np.full((height, width), value, np.uint8)
            else:  # BGR on disk: R channel = value, so RGB after loading = (value, 0, 255-value)
                image = np.dstack(
                    [np.full((height, width), c, np.uint8) for c in (255 - value, 0, value)]
                )
            cv2.imwrite(str(image_dir / f"{frame_id:06d}.png"), image)
    (seq_dir / "calib.txt").write_text(KITTI_CALIB)
    if with_times:
        np.savetxt(seq_dir / "times.txt", np.arange(num_frames) * 0.1036, fmt="%.6e")
    if with_poses:
        count = num_frames if pose_count is None else pose_count
        poses = np.stack([kitti_pose(i) for i in range(count)])
        (root / "poses").mkdir(parents=True, exist_ok=True)
        np.savetxt(
            root / "poses" / f"{sequence}.txt", poses[:, :3, :4].reshape(count, 12), fmt="%.9e"
        )
    return seq_dir


@pytest.fixture
def kitti_root(tmp_path: Path) -> Path:
    write_kitti_sequence(tmp_path)
    return tmp_path


SMALL_SYNTHETIC = SyntheticConfig(
    seed=3,
    image_width=96,
    image_height=32,
    block_length_m=60.0,
    block_width_m=40.0,
    corner_radius_m=8.0,
    laps=1.3,
    frame_spacing_m=1.0,
)


@pytest.fixture(scope="session")
def small_synthetic() -> SyntheticLoopDataset:
    return SyntheticLoopDataset(SMALL_SYNTHETIC)
