"""Tests for far-field geometry and cylinder mask."""

import numpy as np

from lbm.classical import SimulationConfig
from lbm.core.geometry import FlowGeometry, make_cylinder_mask, make_obstacle_mask


def test_cylinder_area_approximately_pi_r2():
    geom = FlowGeometry(nx=80, ny=80, diameter=20.0, cx=40.0, cy=40.0)
    mask = make_cylinder_mask(geom)
    area = int(mask.sum())
    expected = np.pi * geom.radius**2
    assert abs(area - expected) / expected < 0.08


def test_obstacle_is_cylinder_only_no_walls():
    geom = FlowGeometry(nx=40, ny=40, diameter=8.0, cx=20.0, cy=20.0)
    obs = make_obstacle_mask(geom)
    assert not obs[0, :].all()
    assert not obs[-1, :].all()
    assert not obs[:, 0].all()
    assert not obs[:, -1].all()
    assert obs[20, 20]
    assert not obs[20, 35]


def test_default_center_is_mid_domain():
    config = SimulationConfig(nx=201, ny=201)
    assert config.resolved_cx() == 100.5
    assert config.resolved_cy() == 100.5
