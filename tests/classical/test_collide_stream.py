"""Tests for classical BGK collision and shared streaming."""

import numpy as np

from lbm.classical.collide_stream import collide
from lbm.classical.lattice import equilibrium, moments
from lbm.core.constants import C2, CS2
from lbm.core.stream import stream


def test_streaming_is_pure_shift():
    ny, nx = 6, 7
    f = np.zeros((9, ny, nx))
    f[1, 2, 3] = 1.0
    f[2, 2, 3] = 2.0
    out = stream(f)
    assert out[1, 2, 4] == 1.0
    assert out[2, 3, 3] == 2.0


def test_streaming_periodic_wrap():
    ny, nx = 5, 5
    f = np.zeros((9, ny, nx))
    f[1, 0, nx - 1] = 3.0
    out = stream(f)
    assert out[1, 0, 0] == 3.0


def test_mass_conserved_periodic_bgk():
    rng = np.random.default_rng(0)
    ny, nx = 16, 16
    rho = 1.0 + 0.01 * rng.standard_normal((ny, nx))
    ux = 0.02 * rng.standard_normal((ny, nx))
    uy = 0.02 * rng.standard_normal((ny, nx))
    T = CS2 + 0.01 * rng.standard_normal((ny, nx))
    f = equilibrium(rho, ux, uy, T)
    mass0 = float(f.sum())
    nu = 0.1
    for _ in range(50):
        f = stream(collide(f, nu))
    assert np.isclose(f.sum(), mass0, rtol=1e-12)


def test_energy_conserved_periodic_bgk():
    rng = np.random.default_rng(1)
    ny, nx = 12, 12
    rho = 1.0 + 0.01 * rng.standard_normal((ny, nx))
    ux = 0.02 * rng.standard_normal((ny, nx))
    uy = 0.02 * rng.standard_normal((ny, nx))
    T = 0.3 + 0.02 * rng.standard_normal((ny, nx))
    f = equilibrium(rho, ux, uy, T)
    energy0 = float((C2[:, None, None] * f).sum())
    nu = 0.08
    for _ in range(40):
        f = collide(f, nu)
        f = stream(f)
    energy1 = float((C2[:, None, None] * f).sum())
    assert np.isclose(energy1, energy0, rtol=1e-12)


def test_collide_relaxes_toward_equilibrium():
    ny, nx = 4, 4
    rho = np.ones((ny, nx))
    ux = np.full((ny, nx), 0.05)
    uy = np.zeros((ny, nx))
    T = np.full((ny, nx), CS2)
    feq = equilibrium(rho, ux, uy, T)
    f = feq.copy()
    f[1] *= 1.2
    f[3] *= 0.8
    res0 = np.linalg.norm(f - feq)
    f1 = collide(f, nu=0.1)
    res1 = np.linalg.norm(f1 - equilibrium(*moments(f1)))
    assert res1 < res0
