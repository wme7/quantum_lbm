"""Tests for classical far-field and bounce-back boundaries."""

import numpy as np

from lbm.classical.boundaries import (
    apply_boundaries,
    apply_far_field,
    bounce_back,
    fluid_state,
)
from lbm.classical.collide_stream import collide
from lbm.classical.lattice import equilibrium, moments
from lbm.core.constants import CS2
from lbm.core.stream import stream


def test_bounce_back_zeros_solid_velocity():
    ny, nx = 12, 12
    obstacle = np.zeros((ny, nx), dtype=bool)
    obstacle[4:8, 4:8] = True
    rho = np.ones((ny, nx))
    ux = np.full((ny, nx), 0.05)
    uy = np.full((ny, nx), -0.02)
    T = np.full((ny, nx), CS2)
    f = equilibrium(rho, ux, uy, T)
    f = stream(collide(f, nu=0.1))
    bounce_back(f, obstacle)
    _, ux_m, uy_m, _ = fluid_state(f, obstacle)
    assert np.allclose(ux_m[obstacle], 0.0)
    assert np.allclose(uy_m[obstacle], 0.0)


def test_far_field_recovers_free_stream_on_all_edges():
    ny, nx = 16, 16
    rho_inf, u_inf, T_inf = 1.1, 0.07, 0.4
    rho = np.full((ny, nx), 0.9)
    ux = np.full((ny, nx), -0.02)
    uy = np.full((ny, nx), 0.03)
    T = np.full((ny, nx), CS2)
    f = equilibrium(rho, ux, uy, T)
    apply_far_field(f, rho_inf=rho_inf, u_inf=u_inf, T_inf=T_inf)
    rho_b, ux_b, uy_b, T_b = moments(f)

    edges = (
        (slice(None), 0),
        (slice(None), -1),
        (0, slice(None)),
        (-1, slice(None)),
    )
    for ys, xs in edges:
        assert np.allclose(rho_b[ys, xs], rho_inf)
        assert np.allclose(ux_b[ys, xs], u_inf)
        assert np.allclose(uy_b[ys, xs], 0.0)
        assert np.allclose(T_b[ys, xs], T_inf)


def test_apply_boundaries_far_field_then_bounce_back():
    ny, nx = 20, 20
    obstacle = np.zeros((ny, nx), dtype=bool)
    obstacle[8:12, 8:12] = True
    rho_inf, u_inf, T_inf = 1.0, 0.05, 0.45
    rho = np.ones((ny, nx))
    ux = np.full((ny, nx), u_inf)
    uy = np.zeros((ny, nx))
    T = np.full((ny, nx), T_inf)
    f = equilibrium(rho, ux, uy, T)
    f = stream(collide(f, nu=0.08))
    apply_boundaries(f, obstacle, rho_inf=rho_inf, u_inf=u_inf, T_inf=T_inf)
    rho_b, ux_b, uy_b, T_b = moments(f)
    assert np.allclose(rho_b[:, 0], rho_inf)
    assert np.allclose(ux_b[:, 0], u_inf)
    assert np.allclose(T_b[:, 0], T_inf)
    _, ux_f, uy_f, _ = fluid_state(f, obstacle)
    assert np.allclose(ux_f[obstacle], 0.0)
    assert np.allclose(uy_f[obstacle], 0.0)
