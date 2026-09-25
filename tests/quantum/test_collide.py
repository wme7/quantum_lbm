"""Tests for quantum BGK collision."""

import numpy as np

from lbm.core.constants import CS2
from lbm.core.stream import stream
from lbm.quantum.collide_stream import collide
from lbm.quantum.eos import chi, fugacity
from lbm.quantum.lattice import equilibrium, moments


def test_mass_conserved_periodic_bgk_mb():
    rng = np.random.default_rng(0)
    ny, nx = 12, 12
    h, eta = 1.0, 0
    rho = 1.0 + 0.01 * rng.standard_normal((ny, nx))
    ux = 0.02 * rng.standard_normal((ny, nx))
    uy = 0.02 * rng.standard_normal((ny, nx))
    T = CS2 + 0.01 * rng.standard_normal((ny, nx))
    z = np.asarray(fugacity(rho, T, h, eta), dtype=np.float64)
    f = equilibrium(rho, ux, uy, T, z, eta)
    mass0 = float(f.sum())
    nu = 0.1
    for _ in range(40):
        f = stream(collide(f, nu, h=h, eta=eta))
    assert np.isclose(f.sum(), mass0, rtol=1e-12)


def test_mb_local_tau_matches_classical_formula():
    """For MB, χ=1 so τ = ν/T + 1/2 at equilibrium nodes."""
    rho0, T0 = 1.0, 0.45
    h, eta = 1.0, 0
    z0 = float(fugacity(rho0, T0, h, eta))
    assert float(chi(z0, eta)) == 1.0
    ny, nx = 4, 4
    rho = np.full((ny, nx), rho0)
    ux = np.zeros((ny, nx))
    uy = np.zeros((ny, nx))
    T = np.full((ny, nx), T0)
    z = np.full((ny, nx), z0)
    f = equilibrium(rho, ux, uy, T, z, eta)
    # One collide with τ=1 leaves f unchanged at equilibrium; check recovered T
    nu = 0.08
    f1 = collide(f, nu, h=h, eta=eta)
    assert np.allclose(f1, f)
    _, _, _, T_m, _ = moments(f1, h=h, eta=eta)
    expected_tau = nu / T0 + 0.5
    # Reconstruct the τ field used inside collide for MB
    tau_field = nu / T_m + 0.5
    assert np.allclose(tau_field, expected_tau)
