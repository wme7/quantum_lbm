"""Tests for D2Q9 constants and classical Shan-He N=2 equilibrium."""

import numpy as np

from lbm.classical.lattice import equilibrium, moments
from lbm.core.constants import C2, CS2, CX, CY, D, OPPOSITE, Q, WEIGHTS


def test_weights_sum_to_one():
    assert np.isclose(WEIGHTS.sum(), 1.0)


def test_opposite_is_involution():
    for i in range(Q):
        assert OPPOSITE[OPPOSITE[i]] == i


def test_sound_speed_from_weights():
    cxx = float(np.sum(WEIGHTS * CX.astype(float) ** 2))
    cyy = float(np.sum(WEIGHTS * CY.astype(float) ** 2))
    cxy = float(np.sum(WEIGHTS * CX.astype(float) * CY.astype(float)))
    assert np.isclose(cxx, CS2)
    assert np.isclose(cyy, CS2)
    assert np.isclose(cxy, 0.0)


def _isothermal_polynomial(rho, ux, uy):
    u_sq = ux * ux + uy * uy
    feq = np.empty((Q, *np.shape(rho)), dtype=np.float64)
    for i in range(Q):
        cu = CX[i] * ux + CY[i] * uy
        feq[i] = rho * WEIGHTS[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq)
    return feq


def test_feq_at_cs2_matches_isothermal():
    rho0 = 1.05
    ux0, uy0 = 0.04, -0.02
    rho = np.full((8, 10), rho0)
    ux = np.full((8, 10), ux0)
    uy = np.full((8, 10), uy0)
    T = np.full((8, 10), CS2)
    feq = equilibrium(rho, ux, uy, T)
    assert np.allclose(feq, _isothermal_polynomial(rho, ux, uy))


def test_feq_recovers_macroscopics():
    rho0 = 1.05
    ux0, uy0 = 0.04, -0.02
    T0 = 0.28
    rho = np.full((8, 10), rho0)
    ux = np.full((8, 10), ux0)
    uy = np.full((8, 10), uy0)
    T = np.full((8, 10), T0)
    feq = equilibrium(rho, ux, uy, T)
    rho_m, ux_m, uy_m, T_m = moments(feq)
    assert np.allclose(rho_m, rho0)
    assert np.allclose(ux_m, ux0)
    assert np.allclose(uy_m, uy0)
    assert np.allclose(T_m, T0)


def test_feq_pressure_tensor():
    rho0 = 0.9
    ux0, uy0 = 0.03, 0.02
    T0 = 0.4
    rho = np.full((5, 6), rho0)
    ux = np.full((5, 6), ux0)
    uy = np.full((5, 6), uy0)
    T = np.full((5, 6), T0)
    feq = equilibrium(rho, ux, uy, T)
    pxx = (CX.astype(float)[:, None, None] ** 2 * feq).sum(axis=0)
    pyy = (CY.astype(float)[:, None, None] ** 2 * feq).sum(axis=0)
    pxy = (
        CX.astype(float)[:, None, None] * CY.astype(float)[:, None, None] * feq
    ).sum(axis=0)
    assert np.allclose(pxx, rho0 * ux0 * ux0 + rho0 * T0)
    assert np.allclose(pyy, rho0 * uy0 * uy0 + rho0 * T0)
    assert np.allclose(pxy, rho0 * ux0 * uy0)
    energy = (C2[:, None, None] * feq).sum(axis=0)
    assert np.allclose(energy, rho0 * (ux0**2 + uy0**2 + D * T0))
