"""Quantum simulation smoke tests."""

import numpy as np

from lbm.quantum import SimulationConfig, run_simulation


def test_quantum_fd_smoke():
    config = SimulationConfig(
        nx=32,
        ny=32,
        re=20.0,
        u_inf=0.05,
        T_inf=0.5,
        diameter=6.0,
        steps=50,
        statistic="FD",
        h=1.0,
        save_every=0,
        out_dir=None,
    )
    result = run_simulation(config)
    fluid = ~result.obstacle
    assert np.all(np.isfinite(result.rho[fluid]))
    assert np.all(np.isfinite(result.T[fluid]))
    assert np.all(np.isfinite(result.z[fluid]))
    assert result.z_inf > 0.0
    assert result.statistic == "FD"
    assert result.tau > 0.5


def test_quantum_be_rejects_degenerate_h():
    config = SimulationConfig(
        nx=32,
        ny=32,
        statistic="BE",
        h=3.0,  # drives z → 1 at T=0.5, rho=1
        T_inf=0.5,
        rho_inf=1.0,
    )
    try:
        config.validate()
    except ValueError as exc:
        assert "fugacity" in str(exc).lower() or "bose" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for degenerate Bose free stream")
