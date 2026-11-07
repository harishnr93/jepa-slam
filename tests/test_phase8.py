"""Phase 8 end-to-end GIF: fails loudly without results; frame helpers have a fixed size."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from latent_loop.visualization.phase8 import STEPS, _canvas, _traj, generate_phase8
from latent_loop.visualization.plots import figure_to_array


def test_missing_benchmark_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_benchmark.py"):
        generate_phase8(
            "configs/benchmark.yaml",
            tmp_path / "out",
            overrides=[f"benchmark.output_dir='{tmp_path}'"],
        )
    assert not (tmp_path / "out" / "pipeline_end_to_end.gif").exists()


def test_canvas_size_and_stretch_note():
    fig = _canvas(STEPS[2], "caption")
    ax = fig.add_subplot(111)
    long_path = np.c_[np.zeros(50), np.linspace(0, 400, 50)]
    _traj(ax, long_path, upto=10)
    assert "x stretched" in ax.get_title()
    assert figure_to_array(fig).shape == (540, 960, 3)
    square = np.c_[np.linspace(0, 100, 50), np.linspace(0, 100, 50)]
    _traj(ax, square)
    assert ax.get_aspect() == 1.0
