"""Tests for quantum EOS helpers."""

import numpy as np

from lbm.core.constants import D
from lbm.quantum.eos import chi, eta_from_statistic, fugacity, recover_TZ
from lbm.quantum.lattice import equilibrium, moments


def test_eta_from_statistic():
    assert eta_from_statistic("FD") == -1
    assert eta_from_statistic("BE") == 1
    assert eta_from_statistic("MB") == 0


def test_chi_mb_is_one():
    z = np.array([0.1, 0.5, 2.0])
    assert np.allclose(chi(z, eta=0), 1.0)


def test_chi_fd_gt_mb_gt_be_at_same_z():
    z = 0.2
    chi_fd = float(chi(z, eta=-1))
    chi_mb = float(chi(z, eta=0))
    chi_be = float(chi(z, eta=1))
    assert chi_fd > chi_mb > chi_be


def test_recover_TZ_roundtrip_mb():
    rho0, T0, ux0, uy0 = 1.05, 0.4, 0.03, -0.02
    h, eta = 1.0, 0
    z0 = float(fugacity(rho0, T0, h, eta))
    rho = np.full((4, 5), rho0)
    ux = np.full((4, 5), ux0)
    uy = np.full((4, 5), uy0)
    T = np.full((4, 5), T0)
    z = np.full((4, 5), z0)
    feq = equilibrium(rho, ux, uy, T, z, eta)
    rho_m, ux_m, uy_m, T_m, z_m = moments(feq, h=h, eta=eta)
    assert np.allclose(rho_m, rho0)
    assert np.allclose(ux_m, ux0)
    assert np.allclose(uy_m, uy0)
    assert np.allclose(T_m, T0)
    assert np.allclose(z_m, z0)


def test_recover_TZ_roundtrip_fd():
    rho0, T0 = 1.0, 0.5
    h, eta = 1.0, -1
    z0 = float(fugacity(rho0, T0, h, eta))
    chi0 = float(chi(z0, eta))
    u_sq = 0.01
    energy = rho0 * (u_sq + D * T0 * chi0)
    T, z, chi_f = recover_TZ(
        np.array([rho0]),
        np.array([energy]),
        np.array([u_sq]),
        h,
        eta,
    )
    assert np.isclose(T[0], T0, rtol=1e-6)
    assert np.isclose(z[0], z0, rtol=1e-5)
    assert np.isclose(chi_f[0], chi0, rtol=1e-5)
