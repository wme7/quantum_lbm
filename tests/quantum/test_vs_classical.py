"""Cross-check: quantum MB must match classical Shan-He (source of truth)."""

import numpy as np

from lbm.classical import SimulationConfig as ClassicalConfig
from lbm.classical import run_simulation as run_classical
from lbm.classical.collide_stream import collide as classical_collide
from lbm.classical.lattice import equilibrium as classical_eq
from lbm.core.stream import stream
from lbm.quantum import SimulationConfig as QuantumConfig
from lbm.quantum import run_simulation as run_quantum
from lbm.quantum.collide_stream import collide as quantum_collide
from lbm.quantum.eos import fugacity
from lbm.quantum.lattice import equilibrium as quantum_eq


def test_periodic_mb_matches_classical_fields():
    rng = np.random.default_rng(42)
    ny, nx = 16, 16
    h, eta = 1.0, 0
    rho = 1.0 + 0.01 * rng.standard_normal((ny, nx))
    ux = 0.02 * rng.standard_normal((ny, nx))
    uy = 0.02 * rng.standard_normal((ny, nx))
    T = 0.35 + 0.01 * rng.standard_normal((ny, nx))
    z = np.asarray(fugacity(rho, T, h, eta), dtype=np.float64)

    f_c = classical_eq(rho, ux, uy, T)
    f_q = quantum_eq(rho, ux, uy, T, z, eta)
    assert np.allclose(f_c, f_q)

    nu = 0.1
    for _ in range(30):
        f_c = stream(classical_collide(f_c, nu))
        f_q = stream(quantum_collide(f_q, nu, h=h, eta=eta))
    assert np.allclose(f_c, f_q, rtol=1e-10, atol=1e-12)


def test_tiny_cylinder_mb_matches_classical():
    shared = dict(
        nx=32,
        ny=32,
        re=20.0,
        u_inf=0.05,
        T_inf=0.5,
        diameter=6.0,
        steps=80,
        rho_inf=1.0,
        save_every=0,
        out_dir=None,
    )
    classical = run_classical(ClassicalConfig(**shared))
    quantum = run_quantum(
        QuantumConfig(**shared, statistic="MB", h=1.0)
    )
    fluid = ~classical.obstacle
    assert np.allclose(classical.rho[fluid], quantum.rho[fluid], rtol=1e-8, atol=1e-10)
    assert np.allclose(classical.ux[fluid], quantum.ux[fluid], rtol=1e-8, atol=1e-10)
    assert np.allclose(classical.uy[fluid], quantum.uy[fluid], rtol=1e-8, atol=1e-10)
    assert np.allclose(classical.T[fluid], quantum.T[fluid], rtol=1e-7, atol=1e-9)
    assert np.isclose(classical.tau, quantum.tau)
