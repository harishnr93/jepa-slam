import numpy as np
import pytest

from latent_loop.datasets.ground_truth import (
    cumulative_distance,
    invert_pose,
    is_valid_rotation,
    load_kitti_poses,
    parse_pose_line,
    positions_from_poses,
    relative_pose,
    save_kitti_poses,
    trajectory_length,
    yaw_from_poses,
)

# A pose line in KITTI format with realistic magnitudes (small rotation, ~0.86 m forward).
KITTI_LINE = (
    "9.999978e-01 5.272628e-04 -2.066935e-03 -4.690294e-02 -5.296506e-04 9.999992e-01 "
    "-1.154865e-03 -2.839928e-02 2.066324e-03 1.155958e-03 9.999971e-01 8.586941e-01"
)


def _pose(yaw: float, t: tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    pose[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    pose[:3, 3] = t
    return pose


def test_parse_pose_line_row_major():
    pose = parse_pose_line(KITTI_LINE)
    assert pose.shape == (4, 4)
    np.testing.assert_allclose(pose[3], [0, 0, 0, 1])
    np.testing.assert_allclose(pose[:3, 3], [-4.690294e-02, -2.839928e-02, 8.586941e-01])
    assert pose[0, 1] == pytest.approx(5.272628e-04)  # row-major: second value is R[0, 1]
    assert is_valid_rotation(pose[:3, :3], atol=1e-5)


@pytest.mark.parametrize("line", ["1 2 3", " ".join(["1"] * 13), " ".join(["nan"] * 12)])
def test_parse_pose_line_rejects_malformed(line):
    with pytest.raises(ValueError):
        parse_pose_line(line)


def test_load_save_roundtrip_and_blank_lines(tmp_path):
    poses = np.stack([_pose(0.1 * i, (i, 0.0, 2.0 * i)) for i in range(5)])
    path = tmp_path / "poses.txt"
    save_kitti_poses(path, poses)
    path.write_text(path.read_text() + "\n\n")  # trailing blank lines are ignored
    np.testing.assert_allclose(load_kitti_poses(path), poses, atol=1e-8)


def test_load_reports_line_number(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text(KITTI_LINE + "\n1 2 3\n")
    with pytest.raises(ValueError, match="bad.txt:2"):
        load_kitti_poses(path)


def test_load_empty_file_raises(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("\n")
    with pytest.raises(ValueError, match="No poses"):
        load_kitti_poses(path)


def test_invert_and_relative_pose():
    a = _pose(0.3, (1.0, 2.0, 3.0))
    b = _pose(-0.2, (4.0, 0.5, -1.0))
    np.testing.assert_allclose(invert_pose(a) @ a, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(invert_pose(a), np.linalg.inv(a), atol=1e-12)
    np.testing.assert_allclose(a @ relative_pose(a, b), b, atol=1e-12)


def test_is_valid_rotation_rejects_reflection_and_scale():
    assert is_valid_rotation(np.eye(3))
    assert not is_valid_rotation(np.diag([1.0, 1.0, -1.0]))
    assert not is_valid_rotation(2 * np.eye(3))


def test_yaw_follows_kitti_convention():
    # yaw 0 looks along +z; yaw +pi/2 looks along +x (a right turn).
    poses = np.stack([_pose(0.0, (0, 0, 0)), _pose(np.pi / 2, (0, 0, 0))])
    np.testing.assert_allclose(yaw_from_poses(poses), [0.0, np.pi / 2], atol=1e-12)
    np.testing.assert_allclose(poses[1][:3, :3] @ [0, 0, 1], [1, 0, 0], atol=1e-12)


def test_cumulative_distance_and_length():
    positions = np.array([[0, 0, 0], [3, 0, 4], [3, 0, 10]], dtype=float)
    np.testing.assert_allclose(cumulative_distance(positions), [0, 5, 11])
    assert trajectory_length(positions) == pytest.approx(11.0)
    assert trajectory_length(positions[:1]) == 0.0
    poses = np.stack([_pose(0, tuple(p)) for p in positions])
    np.testing.assert_allclose(positions_from_poses(poses), positions)
