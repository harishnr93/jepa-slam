import numpy as np
import pytest

from conftest import (
    KITTI_IMAGE_SIZE,
    KITTI_MISSING_FRAME,
    KITTI_NUM_FRAMES,
    SMALL_SYNTHETIC,
    kitti_pose,
    write_kitti_sequence,
)
from latent_loop.datasets import (
    KittiOdometryDataset,
    SamplingConfig,
    SyntheticConfig,
    SyntheticLoopDataset,
    build_dataset,
    sample_frame_ids,
)
from latent_loop.datasets.ground_truth import is_valid_rotation
from latent_loop.datasets.kitti import calibration_from_projection, parse_calibration_file
from latent_loop.datasets.synthetic import export_kitti_layout

# --- frame sampling -------------------------------------------------------------------


def test_sampling_uses_frame_id_grid_not_list_positions():
    available = np.array([0, 1, 2, 4, 5, 6, 7, 8])  # frame 3 missing
    ids = sample_frame_ids(available, SamplingConfig(stride=2))
    # Grid 0, 2, 4, 6, 8 is kept even though frame 3 is missing (list slicing would give 0,2,5,7).
    np.testing.assert_array_equal(ids, [0, 2, 4, 6, 8])


def test_sampling_start_end_max_frames():
    ids = sample_frame_ids(np.arange(100), SamplingConfig(start=10, end=50, stride=5, max_frames=4))
    np.testing.assert_array_equal(ids, [10, 15, 20, 25])
    ids = sample_frame_ids(np.arange(100), SamplingConfig(start=95))
    np.testing.assert_array_equal(ids, [95, 96, 97, 98, 99])


@pytest.mark.parametrize(
    "kwargs", [{"stride": 0}, {"start": -1}, {"start": 5, "end": 5}, {"max_frames": 0}]
)
def test_sampling_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        SamplingConfig(**kwargs)


def test_sampling_from_dict_rejects_unknown_keys():
    assert SamplingConfig.from_dict(None) == SamplingConfig()
    with pytest.raises(ValueError, match="Unknown"):
        SamplingConfig.from_dict({"strid": 2})


def test_sampling_requires_sorted_ids():
    with pytest.raises(ValueError):
        sample_frame_ids(np.array([0, 2, 1]), SamplingConfig())


# --- KITTI loader ---------------------------------------------------------------------


def test_kitti_loads_and_skips_missing_frames(kitti_root):
    ds = KittiOdometryDataset(kitti_root, "00", camera=2)
    assert len(ds) == KITTI_NUM_FRAMES - 1
    assert KITTI_MISSING_FRAME not in ds.frame_ids
    np.testing.assert_array_equal(ds.missing_frame_ids, [KITTI_MISSING_FRAME])
    assert ds.has_poses and not ds.timestamps_estimated


def test_kitti_image_pose_timestamp_alignment_after_missing_frame(kitti_root):
    ds = KittiOdometryDataset(kitti_root, "00", camera=2)
    calib = ds.get_calibration()
    for index in range(len(ds)):
        frame = ds.get_frame(index)
        # Pixel values encode the frame ID, so this checks image <-> frame-ID alignment.
        assert frame.image.shape == (KITTI_IMAGE_SIZE[1], KITTI_IMAGE_SIZE[0], 3)
        assert tuple(frame.image[0, 0]) == (10 * frame.frame_id, 0, 255 - 10 * frame.frame_id)
        np.testing.assert_allclose(frame.pose, kitti_pose(frame.frame_id) @ calib.T_cam0_cam)
        assert frame.timestamp == pytest.approx(frame.frame_id * 0.1036, rel=1e-6)
    assert ds.frame_id(KITTI_MISSING_FRAME) == KITTI_MISSING_FRAME + 1


def test_kitti_grayscale_camera_and_identity_offset(kitti_root):
    ds = KittiOdometryDataset(kitti_root, "00", camera=0)
    image = ds.get_image(3)
    assert image.ndim == 2 and image[0, 0] == 30
    np.testing.assert_allclose(ds.get_calibration().T_cam0_cam, np.eye(4))
    np.testing.assert_allclose(ds.get_pose(3), kitti_pose(3))


def test_kitti_calibration_parsing(kitti_root):
    matrices = parse_calibration_file(kitti_root / "sequences/00/calib.txt")
    assert set(matrices) == {"P0", "P1", "P2", "P3", "Tr"}
    assert matrices["Tr"].shape == (4, 4)
    calib = KittiOdometryDataset(kitti_root, "00", camera=2).get_calibration()
    assert calib.image_size == KITTI_IMAGE_SIZE
    assert calib.fx == pytest.approx(718.856) and calib.cy == pytest.approx(185.2157)
    # T_cam0_cam2 = [I | -K^-1 P2[:, 3]]: camera 2 sits ~6 cm to the left (-x) of camera 0.
    expected = -np.linalg.solve(matrices["P2"][:, :3], matrices["P2"][:, 3])
    np.testing.assert_allclose(calib.T_cam0_cam[:3, 3], expected)
    assert -0.07 < calib.T_cam0_cam[0, 3] < -0.05


def test_calibration_validity_checks():
    P = np.array([[700.0, 0, 300, 0], [0, 700, 100, 0], [0, 0, 1, 0]])
    assert calibration_from_projection(P, (600, 200)).is_valid()
    assert not calibration_from_projection(P, (200, 50)).is_valid()  # principal point outside
    with pytest.raises(ValueError):
        calibration_from_projection(np.zeros((3, 4)), (600, 200))


def test_kitti_sampling_preserves_original_ids(kitti_root):
    ds = KittiOdometryDataset(
        kitti_root, "00", camera=2, sampling=SamplingConfig(start=1, stride=2, max_frames=4)
    )
    np.testing.assert_array_equal(ds.frame_ids, [1, 3, 7, 9])  # 5 is missing
    frame = ds.get_frame(2)
    assert frame.index == 2 and frame.frame_id == 7
    assert frame.image[0, 0, 0] == 70
    with pytest.raises(IndexError):
        ds.get_frame(4)


def test_kitti_without_poses(tmp_path):
    write_kitti_sequence(tmp_path, with_poses=False)
    ds = KittiOdometryDataset(tmp_path, "00")
    assert not ds.has_poses
    assert ds.get_pose(0) is None and ds.get_frame(0).pose is None
    with pytest.raises(RuntimeError):
        ds.poses()


def test_kitti_without_times_estimates_timestamps(tmp_path):
    write_kitti_sequence(tmp_path, with_times=False)
    ds = KittiOdometryDataset(tmp_path, "00")
    assert ds.timestamps_estimated
    assert ds.get_timestamp(4) == pytest.approx(0.4)


def test_kitti_pose_count_mismatch_raises(tmp_path):
    write_kitti_sequence(tmp_path, pose_count=KITTI_NUM_FRAMES - 2)
    with pytest.raises(ValueError, match="misalignment"):
        KittiOdometryDataset(tmp_path, "00")


def test_kitti_missing_directory_or_calibration(tmp_path):
    with pytest.raises(FileNotFoundError):
        KittiOdometryDataset(tmp_path, "00")
    seq_dir = write_kitti_sequence(tmp_path)
    (seq_dir / "calib.txt").unlink()
    with pytest.raises(FileNotFoundError, match="calib"):
        KittiOdometryDataset(tmp_path, "00")


def test_kitti_malformed_calibration(tmp_path):
    seq_dir = write_kitti_sequence(tmp_path)
    (seq_dir / "calib.txt").write_text("P2: 1 2 3\n")
    with pytest.raises(ValueError):
        KittiOdometryDataset(tmp_path, "00")


# --- synthetic dataset ----------------------------------------------------------------


def test_synthetic_is_deterministic():
    a = SyntheticLoopDataset(SMALL_SYNTHETIC)
    b = SyntheticLoopDataset(SMALL_SYNTHETIC)
    np.testing.assert_array_equal(a.get_image(17), b.get_image(17))
    np.testing.assert_array_equal(a.poses(), b.poses())
    other = SyntheticLoopDataset(SyntheticConfig(**{**vars(SMALL_SYNTHETIC), "seed": 4}))
    assert not np.array_equal(a.get_image(17), other.get_image(17))


def test_synthetic_poses_and_images(small_synthetic):
    ds = small_synthetic
    poses = ds.poses()
    assert all(is_valid_rotation(p[:3, :3]) for p in poses)
    np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-6)  # world = first camera frame
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    assert np.all(np.abs(steps - 1.0) < 0.1)
    np.testing.assert_allclose(poses[:, 1, 3], 0.0)  # camera stays at constant height
    image = ds.get_image(0)
    assert image.dtype == np.uint8 and image.shape == (32, 96, 3)
    assert ds.get_calibration().is_valid()
    np.testing.assert_allclose(ds.timestamps()[:3], [0.0, 0.1, 0.2])


def test_synthetic_revisit_pairs_are_close_and_temporally_separated(small_synthetic):
    ds = small_synthetic
    pairs = ds.revisit_pairs
    assert len(pairs) > 10
    positions = ds.poses()[:, :3, 3]
    dist = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    assert dist.max() < 1.0
    gap = pairs[:, 0] - pairs[:, 1]
    assert gap.min() >= 0.9 * ds.perimeter_m / ds.config.frame_spacing_m
    # Every frame of the second lap is a revisit, and no first-lap frame is.
    assert set(pairs[:, 0]) == set(np.flatnonzero(ds.arc_length_m >= ds.perimeter_m))


def test_synthetic_revisits_look_more_similar_than_other_places(small_synthetic):
    ds = small_synthetic
    pairs = ds.revisit_pairs[::5]
    revisit_diff, other_diff = [], []
    for query, ref in pairs:
        q = ds.render(int(query)).astype(float)
        revisit_diff.append(np.abs(q - ds.render(int(ref))).mean())
        far = (int(ref) + len(ds.available_frame_ids) // 3) % int(pairs[0, 0])
        other_diff.append(np.abs(q - ds.render(far)).mean())
    assert np.mean(revisit_diff) < 0.5 * np.mean(other_diff)


def test_synthetic_sampling_view(small_synthetic):
    ds = SyntheticLoopDataset(SMALL_SYNTHETIC, SamplingConfig(start=3, stride=4, max_frames=5))
    np.testing.assert_array_equal(ds.frame_ids, [3, 7, 11, 15, 19])
    np.testing.assert_array_equal(ds.get_image(1), small_synthetic.render(7))
    assert ds.get_timestamp(1) == pytest.approx(0.7)


def test_synthetic_config_validation():
    with pytest.raises(ValueError):
        SyntheticConfig(laps=0)
    with pytest.raises(ValueError):
        SyntheticConfig(corner_radius_m=50.0)
    with pytest.raises(ValueError, match="Unknown"):
        SyntheticConfig.from_dict({"sed": 1})


def test_synthetic_export_roundtrips_through_kitti_loader(tmp_path):
    source = SyntheticLoopDataset(SyntheticConfig(**{**vars(SMALL_SYNTHETIC), "laps": 0.3}))
    export_kitti_layout(source, tmp_path, "07")
    loaded = KittiOdometryDataset(tmp_path, "07", camera=2)
    assert len(loaded) == len(source)
    np.testing.assert_allclose(loaded.poses(), source.poses(), atol=1e-6)
    np.testing.assert_array_equal(loaded.get_image(5), source.get_image(5))
    np.testing.assert_allclose(loaded.get_calibration().K, source.get_calibration().K)
    np.testing.assert_allclose(loaded.timestamps(), source.timestamps(), atol=1e-6)
    assert source.revisit_pairs.shape == (0, 2)  # a 0.3-lap drive never revisits a place
    assert (tmp_path / "sequences/07/revisit_pairs.txt").exists()


def test_build_dataset_from_config(kitti_root):
    synthetic = build_dataset(
        {
            "dataset": {
                "name": "synthetic",
                "synthetic": vars(SMALL_SYNTHETIC),
                "sampling": {"stride": 10},
            }
        }
    )
    assert isinstance(synthetic, SyntheticLoopDataset) and synthetic.frame_ids[1] == 10
    kitti = build_dataset(
        {"dataset": {"name": "kitti", "root": str(kitti_root), "sequence": "00", "camera": 0}}
    )
    assert isinstance(kitti, KittiOdometryDataset) and kitti.camera == 0
    with pytest.raises(ValueError):
        build_dataset({"dataset": {"name": "nuscenes"}})
