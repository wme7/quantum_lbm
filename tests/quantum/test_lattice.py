"""Tests for quantum Yang-Hung equilibrium."""

import numpy as np

from lbm.classical.lattice import equilibrium as classical_equilibrium
from lbm.core.constants import C2, CX, CY, D
from lbm.quantum.eos import chi, fugacity
from lbm.quantum.lattice import equilibrium, moments


def test_mb_feq_matches_classical():
    rho0, ux0, uy0, T0 = 1.1, 0.04, -0.02, 0.35
    h, eta = 1.0, 0
    z0 = float(fugacity(rho0, T0, h, eta))
    rho = np.full((6, 7), rho0)
    ux = np.full((6, 7), ux0)
    uy = np.full((6, 7), uy0)
    T = np.full((6, 7), T0)
    z = np.full((6, 7), z0)
    feq_q = equilibrium(rho, ux, uy, T, z, eta)
    feq_c = classical_equilibrium(rho, ux, uy, T)
    assert np.allclose(feq_q, feq_c)


def test_feq_pressure_tensor_with_chi():
    rho0, ux0, uy0, T0 = 0.95, 0.03, 0.02, 0.4
    h, eta = 1.0, -1  # FD
    z0 = float(fugacity(rho0, T0, h, eta))
    chi0 = float(chi(z0, eta))
    rho = np.full((5, 5), rho0)
    ux = np.full((5, 5), ux0)
    uy = np.full((5, 5), uy0)
    T = np.full((5, 5), T0)
    z = np.full((5, 5), z0)
    feq = equilibrium(rho, ux, uy, T, z, eta)
    pxx = (CX.astype(float)[:, None, None] ** 2 * feq).sum(axis=0)
    pyy = (CY.astype(float)[:, None, None] ** 2 * feq).sum(axis=0)
    pxy = (
        CX.astype(float)[:, None, None] * CY.astype(float)[:, None, None] * feq
    ).sum(axis=0)
    assert np.allclose(pxx, rho0 * ux0 * ux0 + rho0 * T0 * chi0)
    assert np.allclose(pyy, rho0 * uy0 * uy0 + rho0 * T0 * chi0)
    assert np.allclose(pxy, rho0 * ux0 * uy0)
    energy = (C2[:, None, None] * feq).sum(axis=0)
    assert np.allclose(energy, rho0 * (ux0**2 + uy0**2 + D * T0 * chi0))
    rho_m, ux_m, uy_m, T_m, z_m = moments(feq, h=h, eta=eta)
    assert np.allclose(rho_m, rho0)
    assert np.allclose(T_m, T0, rtol=1e-5)
    assert np.allclose(z_m, z0, rtol=1e-4)
