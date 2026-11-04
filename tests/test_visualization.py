import numpy as np
from PIL import Image

from latent_loop.visualization.animations import (
    render_trajectory_camera_frames,
    save_gif,
    select_animation_indices,
)
from latent_loop.visualization.plots import figure_to_array, plot_frame_montage, plt
from latent_loop.visualization.trajectory import plot_trajectory_overview


def test_select_animation_indices():
    np.testing.assert_array_equal(select_animation_indices(5, 10), [0, 1, 2, 3, 4])
    idx = select_animation_indices(1000, 50)
    assert len(idx) == 50 and idx[0] == 0 and idx[-1] == 999
    assert select_animation_indices(0, 10).size == 0


def test_frame_montage(small_synthetic):
    images = [small_synthetic.get_image(i) for i in (0, 10, 20)]
    fig = plot_frame_montage(images, ["a", "b", "c"], ncols=2, title="t")
    image = figure_to_array(fig)
    assert image.ndim == 3 and image.shape[2] == 3
    plt.close(fig)


def test_trajectory_overview_with_revisits(small_synthetic):
    ds = small_synthetic
    positions = ds.poses()[:, :3, 3]
    fig = plot_trajectory_overview(
        positions,
        ds.timestamps(),
        ds.frame_ids,
        sampled_positions=positions[::5],
        sampled_frame_ids=ds.frame_ids[::5],
        stride=5,
        revisit_pairs=ds.revisit_pairs,
    )
    titles = [ax.get_title() for ax in fig.axes]
    assert any("Constructed revisit pairs" in t for t in titles)
    assert any("mod 5" in t for t in titles)
    plt.close(fig)


def test_trajectory_overview_without_revisits(small_synthetic):
    positions = small_synthetic.poses()[:, :3, 3]
    fig = plot_trajectory_overview(
        positions,
        small_synthetic.timestamps(),
        small_synthetic.frame_ids,
        positions[::5],
        small_synthetic.frame_ids[::5],
        stride=5,
    )
    assert not any("revisit" in ax.get_title() for ax in fig.axes)
    plt.close(fig)


def test_gif_frames_and_writer(small_synthetic, tmp_path):
    ds = small_synthetic
    revisits = {int(q): int(r) for q, r in ds.revisit_pairs}
    indices = select_animation_indices(len(ds), 6)
    frames = render_trajectory_camera_frames(ds, indices, width_px=300, revisits=revisits)
    assert len(frames) == len(indices)
    assert all(f.shape == frames[0].shape for f in frames)
    assert frames[0].shape[1] == 300
    assert not np.array_equal(frames[0], frames[-1])  # the animation actually changes

    path = save_gif(frames, tmp_path / "anim.gif", fps=8)
    with Image.open(path) as gif:
        assert gif.n_frames == len(frames)
        assert abs(gif.info["duration"] - 125) <= 10  # GIF delays have 10 ms resolution
        assert gif.size == (frames[0].shape[1], frames[0].shape[0])
