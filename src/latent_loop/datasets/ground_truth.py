"""Ground-truth pose parsing and trajectory geometry.

Pose convention (KITTI Odometry): each line of ``poses/XX.txt`` holds 12 floats, the
row-major 3x4 matrix ``[R | t]`` of the homogeneous transform ``T_w_c`` that maps a point
from the camera-0 frame at time *k* into the world frame, where the world frame is the
camera-0 frame at the first image:

    p_w = R_k @ p_c + t_k

Camera axes follow the KITTI/OpenCV convention: x right, y down, z forward. The vehicle
therefore moves mostly in the x-z plane, and top-down plots use (x, z).

Ground truth is used only for labelling, evaluation and visualisation, never for
selecting retrieval candidates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

POSE_VALUES_PER_LINE = 12


def parse_pose_line(line: str) -> np.ndarray:
    """Parse one KITTI pose line into a 4x4 homogeneous transform."""
    values = line.split()
    if len(values) != POSE_VALUES_PER_LINE:
        raise ValueError(f"Expected {POSE_VALUES_PER_LINE} values per pose line, got {len(values)}")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :4] = np.asarray(values, dtype=np.float64).reshape(3, 4)
    if not np.all(np.isfinite(pose)):
        raise ValueError("Pose contains non-finite values")
    return pose


def load_kitti_poses(path: str | Path) -> np.ndarray:
    """Load a KITTI pose file as an ``(N, 4, 4)`` array. Blank lines are ignored."""
    path = Path(path)
    poses = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                poses.append(parse_pose_line(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not poses:
        raise ValueError(f"No poses found in {path}")
    return np.stack(poses)


def save_kitti_poses(path: str | Path, poses: np.ndarray) -> None:
    """Write ``(N, 4, 4)`` poses in KITTI format (row-major 3x4 per line)."""
    poses = np.asarray(poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses of shape (N, 4, 4), got {poses.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, poses[:, :3, :4].reshape(len(poses), 12), fmt="%.9e")


def is_valid_rotation(rotation: np.ndarray, atol: float = 1e-4) -> bool:
    """True if ``rotation`` is orthonormal with determinant +1."""
    rotation = np.asarray(rotation, dtype=np.float64)
    return bool(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=atol)
    )


def invert_pose(pose: np.ndarray) -> np.ndarray:
    """Closed-form inverse of an SE(3) transform: ``[R^T | -R^T t]``."""
    inverse = np.eye(4, dtype=np.float64)
    rotation = pose[:3, :3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ pose[:3, 3]
    return inverse


def relative_pose(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """Transform from frame b into frame a: ``T_a_b = T_w_a^-1 @ T_w_b``."""
    return invert_pose(pose_a) @ pose_b


def positions_from_poses(poses: np.ndarray) -> np.ndarray:
    """Camera centres in world coordinates, shape ``(N, 3)``."""
    return np.asarray(poses)[:, :3, 3].copy()


def yaw_from_poses(poses: np.ndarray) -> np.ndarray:
    """Heading of the camera's forward (+z) axis in the world x-z plane, in radians.

    ``yaw = atan2(f_x, f_z)`` where ``f = R @ [0, 0, 1]``; yaw 0 points along world +z.
    """
    forward = np.asarray(poses)[:, :3, 2]
    return np.arctan2(forward[:, 0], forward[:, 2])


def cumulative_distance(positions: np.ndarray) -> np.ndarray:
    """Arc length travelled up to each pose, ``s_0 = 0``, in metres."""
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def trajectory_length(positions: np.ndarray) -> float:
    return float(cumulative_distance(positions)[-1]) if len(positions) > 1 else 0.0


POSITIVE, NEGATIVE, IGNORED = 1, 0, -1


def pair_labels(
    positions: np.ndarray,
    frame_ids: np.ndarray,
    positive_radius_m: float = 5.0,
    negative_radius_m: float = 25.0,
    min_gap_frames: int = 100,
) -> np.ndarray:
    """Label every (query i, earlier frame j) pair for evaluation.

    With ``d_ij = ||p_i - p_j||_2`` (ground-plane x-z distance) and the temporal gap
    ``g_ij = frame_id_i - frame_id_j`` measured in *original* frame IDs:

    * ``POSITIVE`` (1):  ``g_ij >= min_gap_frames`` and ``d_ij <= positive_radius_m``
    * ``NEGATIVE`` (0):  ``g_ij >= min_gap_frames`` and ``d_ij >  negative_radius_m``
    * ``IGNORED``  (-1): too recent (``g_ij < min_gap_frames``), not earlier (``j >= i``)
      or in the ambiguous band between the two radii.

    Returns an ``(N, N)`` int8 matrix. For labelling and evaluation only.
    """
    if negative_radius_m < positive_radius_m:
        raise ValueError("negative_radius_m must be >= positive_radius_m")
    xz = np.asarray(positions)[:, [0, 2]]
    ids = np.asarray(frame_ids, dtype=np.int64)
    dist = np.linalg.norm(xz[:, None, :] - xz[None, :, :], axis=-1)
    eligible = (ids[:, None] - ids[None, :]) >= min_gap_frames
    labels = np.full(dist.shape, IGNORED, dtype=np.int8)
    labels[eligible & (dist <= positive_radius_m)] = POSITIVE
    labels[eligible & (dist > negative_radius_m)] = NEGATIVE
    return labels
