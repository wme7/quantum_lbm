"""Tests for visualization helpers."""

from pathlib import Path

import numpy as np

from lbm.core.viz import save_velocity_plot, vorticity


def test_vorticity_finite():
    ux = np.full((8, 8), 0.1)
    uy = np.zeros((8, 8))
    omega = vorticity(ux, uy)
    assert omega.shape == ux.shape
    assert np.all(np.isfinite(omega))


def test_save_velocity_plot_with_vorticity(tmp_path: Path):
    ny, nx = 12, 12
    ux = np.full((ny, nx), 0.05)
    uy = np.zeros((ny, nx))
    obstacle = np.zeros((ny, nx), dtype=bool)
    obstacle[4:8, 4:8] = True
    path = tmp_path / "vel_vort.png"
    save_velocity_plot(
        ux, uy, obstacle, path, title="test", include_vorticity=True
    )
    assert path.is_file()
    assert path.stat().st_size > 0
