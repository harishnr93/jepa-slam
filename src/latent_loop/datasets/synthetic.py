"""Deterministic synthetic driving sequence with known revisits.

A vehicle drives around a rounded rectangular city block for ``laps`` (> 1) laps at a
constant frame spacing. Buildings with seeded sizes, colours and window grids line both
sides of the road and are rendered with a pinhole camera. Because the second lap passes
the same buildings, every frame with arc length ``s >= perimeter`` revisits the place
of the first-lap frame at arc length ``s - perimeter``. Those pairs are exposed as
``revisit_pairs`` (constructed, not thresholded), giving exact ground truth for tests.

Poses use the KITTI camera convention (x right, y down, z forward) with the world frame
equal to the first camera frame, so the ground plane is ``y = camera_height_m``.

The images are intentionally simple: they contain place-specific structure (building
layout, colour and windows) but are not photorealistic. Results on this data are
sanity checks, not evidence about real-world place recognition.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from latent_loop.datasets.base import CameraCalibration, SamplingConfig, VisualSequenceDataset
from latent_loop.datasets.ground_truth import save_kitti_poses

_NEAR_M = 1.0
_FAR_M = 90.0
_SKY_TOP = np.array([110, 160, 215], dtype=np.float32)
_SKY_HORIZON = np.array([200, 220, 235], dtype=np.float32)
_GROUND_HORIZON = np.array([150, 150, 145], dtype=np.float32)
_GROUND_NEAR = np.array([80, 80, 82], dtype=np.float32)


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 7
    image_width: int = 416
    image_height: int = 128
    frame_spacing_m: float = 1.0
    frame_rate_hz: float = 10.0
    block_length_m: float = 120.0  # extent along world z
    block_width_m: float = 80.0  # extent along world x
    corner_radius_m: float = 15.0
    laps: float = 1.6
    revisit_lateral_offset_m: float = 0.5
    camera_height_m: float = 1.65
    noise_std: float = 2.0

    def __post_init__(self) -> None:
        if self.laps <= 0:
            raise ValueError("laps must be positive")
        if self.frame_spacing_m <= 0 or self.frame_rate_hz <= 0:
            raise ValueError("frame_spacing_m and frame_rate_hz must be positive")
        if 2 * self.corner_radius_m >= min(self.block_length_m, self.block_width_m):
            raise ValueError("corner_radius_m is too large for the block size")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> SyntheticConfig:
        values = dict(values or {})
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown synthetic dataset keys: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class Building:
    x: float
    z: float
    width: float
    height: float
    color: tuple[int, int, int]
    window_rows: int
    window_cols: int


def rounded_rectangle_path(
    length: float, width: float, radius: float, step: float = 0.05
) -> np.ndarray:
    """Dense closed polyline (x, z) of a rounded rectangle, starting at the origin heading +z.

    The loop turns right (towards +x) at every corner, i.e. it is clockwise in a top-down
    (x right, z up) plot.
    """

    def arc(cx: float, cz: float, start: float, end: float) -> np.ndarray:
        n = max(int(abs(end - start) * radius / step), 2)
        angles = np.linspace(start, end, n, endpoint=False)
        return np.stack([cx + radius * np.cos(angles), cz + radius * np.sin(angles)], axis=1)

    def line(p0: tuple[float, float], p1: tuple[float, float]) -> np.ndarray:
        n = max(int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / step), 2)
        t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        return (1 - t) * np.asarray(p0) + t * np.asarray(p1)

    # Start halfway along the left straight so the initial heading is exactly +z.
    r, w, L = radius, width, length
    pieces = [
        line((0, L / 2), (0, L - r)),
        arc(r, L - r, np.pi, np.pi / 2),
        line((r, L), (w - r, L)),
        arc(w - r, L - r, np.pi / 2, 0),
        line((w, L - r), (w, r)),
        arc(w - r, r, 0, -np.pi / 2),
        line((w - r, 0), (r, 0)),
        arc(r, r, -np.pi / 2, -np.pi),
        line((0, r), (0, L / 2)),
    ]
    path = np.concatenate(pieces)
    path[:, 1] -= L / 2
    return np.vstack([path, path[:1]])  # close the loop


def _interpolate_path(path: np.ndarray, s_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions and yaw at arc lengths ``s_query`` (wrapping around the closed path)."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    perimeter = s[-1]
    s_mod = np.mod(s_query, perimeter)
    x = np.interp(s_mod, s, path[:, 0])
    z = np.interp(s_mod, s, path[:, 1])
    delta = 0.5
    x_f = np.interp(np.mod(s_query + delta, perimeter), s, path[:, 0])
    z_f = np.interp(np.mod(s_query + delta, perimeter), s, path[:, 1])
    x_b = np.interp(np.mod(s_query - delta, perimeter), s, path[:, 0])
    z_b = np.interp(np.mod(s_query - delta, perimeter), s, path[:, 1])
    yaw = np.arctan2(x_f - x_b, z_f - z_b)  # 0 = +z, pi/2 = +x
    return np.stack([x, z], axis=1), yaw


def pose_from_xz_yaw(x: float, z: float, yaw: float) -> np.ndarray:
    """``T_w_c`` for a level camera at (x, 0, z) whose forward axis has heading ``yaw``."""
    c, s = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4)
    pose[:3, 0] = [c, 0.0, -s]  # camera x (right)
    pose[:3, 1] = [0.0, 1.0, 0.0]  # camera y (down)
    pose[:3, 2] = [s, 0.0, c]  # camera z (forward)
    pose[:3, 3] = [x, 0.0, z]
    return pose


class SyntheticLoopDataset(VisualSequenceDataset):
    """Rendered loop-route sequence with exact, constructed revisit pairs."""

    name = "synthetic"

    def __init__(
        self, config: SyntheticConfig | None = None, sampling: SamplingConfig | None = None
    ):
        self.config = config or SyntheticConfig()
        cfg = self.config
        self._path = rounded_rectangle_path(
            cfg.block_length_m, cfg.block_width_m, cfg.corner_radius_m
        )
        seg = np.linalg.norm(np.diff(self._path, axis=0), axis=1)
        self.perimeter_m = float(seg.sum())

        num_frames = int(np.floor(cfg.laps * self.perimeter_m / cfg.frame_spacing_m)) + 1
        self.arc_length_m = np.arange(num_frames) * cfg.frame_spacing_m
        self._poses = self._build_poses()
        self.buildings = self._place_buildings()

        width, height = cfg.image_width, cfg.image_height
        focal = 0.55 * width
        K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
        self._calibration = CameraCalibration(
            K=K, image_size=(width, height), P=np.hstack([K, np.zeros((3, 1))])
        )
        self._background = self._render_background()
        super().__init__(np.arange(num_frames), sampling)

    # --- construction ---------------------------------------------------------------
    def _build_poses(self) -> np.ndarray:
        cfg = self.config
        xz, yaw = _interpolate_path(self._path, self.arc_length_m)
        # Second-lap frames drift smoothly to a small lateral offset (right of travel).
        lap_progress = (self.arc_length_m - self.perimeter_m) / 20.0
        blend = np.clip(lap_progress, 0.0, 1.0)
        blend = blend * blend * (3 - 2 * blend)  # smoothstep
        offset = cfg.revisit_lateral_offset_m * blend
        right = np.stack([np.cos(yaw), -np.sin(yaw)], axis=1)
        xz = xz + offset[:, None] * right
        return np.stack([pose_from_xz_yaw(x, z, a) for (x, z), a in zip(xz, yaw, strict=True)])

    def _place_buildings(self) -> list[Building]:
        rng = np.random.default_rng(self.config.seed)
        dense_path = self._path
        buildings: list[Building] = []
        s = 0.0
        while s < self.perimeter_m:
            (pos,), (yaw,) = _interpolate_path(self._path, np.array([s]))
            right = np.array([np.cos(yaw), -np.sin(yaw)])
            for side in (-1.0, 1.0):
                if rng.random() < 0.15:  # occasional gap between buildings
                    continue
                lateral = rng.uniform(9.0, 15.0)
                centre = pos + side * lateral * right + rng.normal(0.0, 1.0, size=2)
                if np.min(np.linalg.norm(dense_path - centre, axis=1)) < 7.0:
                    continue  # never block the road (e.g. inside tight corners)
                hsv = np.uint8(
                    [[[rng.integers(0, 180), rng.integers(70, 220), rng.integers(110, 230)]]]
                )
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
                height = float(rng.uniform(4.0, 20.0))
                width = float(rng.uniform(4.0, 11.0))
                buildings.append(
                    Building(
                        x=float(centre[0]),
                        z=float(centre[1]),
                        width=width,
                        height=height,
                        color=(int(rgb[0]), int(rgb[1]), int(rgb[2])),
                        window_rows=max(1, int(height / 3.0)),
                        window_cols=max(1, int(width / 2.5)),
                    )
                )
            s += rng.uniform(7.0, 12.0)
        return buildings

    def _render_background(self) -> np.ndarray:
        cfg = self.config
        h, w = cfg.image_height, cfg.image_width
        horizon = int(round(self._calibration.cy))
        image = np.empty((h, w, 3), dtype=np.float32)
        t_sky = np.linspace(0.0, 1.0, horizon)[:, None]
        image[:horizon] = ((1 - t_sky) * _SKY_TOP + t_sky * _SKY_HORIZON)[:, None, :]
        t_ground = np.linspace(0.0, 1.0, h - horizon)[:, None]
        image[horizon:] = ((1 - t_ground) * _GROUND_HORIZON + t_ground * _GROUND_NEAR)[:, None, :]
        return image

    # --- rendering ------------------------------------------------------------------
    def render(self, frame_id: int) -> np.ndarray:
        """Render the RGB image of original frame ``frame_id`` (deterministic)."""
        cfg = self.config
        calib = self._calibration
        image = self._background.copy()
        pose = self._poses[frame_id]
        rotation_cw = pose[:3, :3].T
        centre_w = pose[:3, 3]

        positions = np.array([[b.x, cfg.camera_height_m, b.z] for b in self.buildings])
        cam = (positions - centre_w) @ rotation_cw.T  # building base centres in camera frame
        order = np.argsort(-cam[:, 2])  # painter's algorithm: far to near
        for idx in order:
            x_c, y_c, z_c = cam[idx]
            if not _NEAR_M < z_c < _FAR_M:
                continue
            b = self.buildings[idx]
            u0 = calib.fx * (x_c - b.width / 2) / z_c + calib.cx
            u1 = calib.fx * (x_c + b.width / 2) / z_c + calib.cx
            if u1 < 0 or u0 >= cfg.image_width:
                continue
            v_bottom = calib.fy * y_c / z_c + calib.cy
            v_top = calib.fy * (y_c - b.height) / z_c + calib.cy
            haze = 1.0 - np.exp(-z_c / 70.0)  # aerial perspective: fade towards the horizon
            colour = (1 - haze) * np.asarray(b.color, np.float32) + haze * _SKY_HORIZON
            _fill(image, u0, v_top, u1, v_bottom, colour)
            self._draw_windows(image, b, u0, u1, v_top, v_bottom, colour)

        if cfg.noise_std > 0:
            rng = np.random.default_rng((cfg.seed, frame_id))
            image += rng.normal(0.0, cfg.noise_std, size=image.shape).astype(np.float32)
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def _draw_windows(
        image: np.ndarray,
        b: Building,
        u0: float,
        u1: float,
        v_top: float,
        v_bottom: float,
        colour: np.ndarray,
    ) -> None:
        cell_w = (u1 - u0) / b.window_cols
        cell_h = (v_bottom - v_top) / b.window_rows
        if cell_w < 3 or cell_h < 3:
            return
        window_colour = colour * 0.45
        for row in range(b.window_rows):
            for col in range(b.window_cols):
                wu0 = u0 + (col + 0.25) * cell_w
                wv0 = v_top + (row + 0.25) * cell_h
                _fill(image, wu0, wv0, wu0 + 0.5 * cell_w, wv0 + 0.45 * cell_h, window_colour)

    # --- VisualSequenceDataset hooks ------------------------------------------------
    def _load_image(self, frame_id: int) -> np.ndarray:
        return self.render(frame_id)

    def _pose_for(self, frame_id: int) -> np.ndarray | None:
        return self._poses[frame_id]

    def _timestamp_for(self, frame_id: int) -> float:
        return frame_id / self.config.frame_rate_hz

    def get_calibration(self) -> CameraCalibration:
        return self._calibration

    @property
    def has_poses(self) -> bool:
        return True

    # --- ground truth by construction -----------------------------------------------
    @property
    def revisit_pairs(self) -> np.ndarray:
        """``(M, 2)`` array of ``(query_frame_id, reference_frame_id)`` original IDs.

        Every second-lap frame is paired with the first-lap frame at the same arc
        position (nearest frame on the spacing grid). Pairs refer to the full sequence,
        independent of sampling.
        """
        spacing = self.config.frame_spacing_m
        query = np.flatnonzero(self.arc_length_m >= self.perimeter_m)
        reference = np.rint((self.arc_length_m[query] - self.perimeter_m) / spacing).astype(
            np.int64
        )
        return np.stack([query, reference], axis=1)

    def summary(self) -> dict[str, Any]:
        info = super().summary()
        info.update(
            {
                "perimeter_m": round(self.perimeter_m, 2),
                "buildings": len(self.buildings),
                "revisit_pairs": int(len(self.revisit_pairs)),
                "config": asdict(self.config),
            }
        )
        return info


def _fill(
    image: np.ndarray, u0: float, v0: float, u1: float, v1: float, colour: np.ndarray
) -> None:
    h, w = image.shape[:2]
    c0, c1 = max(int(np.floor(u0)), 0), min(int(np.ceil(u1)), w)
    r0, r1 = max(int(np.floor(v0)), 0), min(int(np.ceil(v1)), h)
    if c0 < c1 and r0 < r1:
        image[r0:r1, c0:c1] = colour


def export_kitti_layout(
    dataset: SyntheticLoopDataset, root: str | Path, sequence: str = "00"
) -> Path:
    """Write the full synthetic sequence in KITTI Odometry layout (camera 2 only).

    ``P0 = P2 = K [I | 0]`` and ``P1 = P3`` carry a 0.54 m stereo baseline, so camera-2
    poses equal the exported camera-0 poses exactly. Returns the sequence directory.
    """
    root = Path(root)
    seq_dir = root / "sequences" / sequence
    image_dir = seq_dir / "image_2"
    image_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = dataset.available_frame_ids
    for frame_id in frame_ids:
        rgb = dataset.render(int(frame_id))
        cv2.imwrite(str(image_dir / f"{frame_id:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    K = dataset.get_calibration().K
    p_left = np.hstack([K, np.zeros((3, 1))])
    p_right = np.hstack([K, (K @ np.array([-0.54, 0.0, 0.0]))[:, None]])
    with (seq_dir / "calib.txt").open("w", encoding="utf-8") as handle:
        for key, matrix in (
            ("P0", p_left),
            ("P1", p_right),
            ("P2", p_left),
            ("P3", p_right),
            ("Tr", np.hstack([np.eye(3), np.zeros((3, 1))])),
        ):
            handle.write(f"{key}: " + " ".join(f"{v:.12e}" for v in matrix.ravel()) + "\n")
    np.savetxt(seq_dir / "times.txt", frame_ids / dataset.config.frame_rate_hz, fmt="%.6e")
    save_kitti_poses(root / "poses" / f"{sequence}.txt", dataset._poses[frame_ids])
    np.savetxt(
        seq_dir / "revisit_pairs.txt",
        dataset.revisit_pairs,
        fmt="%d",
        header="query_frame_id reference_frame_id (constructed synthetic revisits)",
    )
    return seq_dir
