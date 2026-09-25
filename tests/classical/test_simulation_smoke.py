"""End-to-end smoke test for classical far-field cylinder run."""

import numpy as np

from lbm.classical import SimulationConfig, run_simulation


def test_simulation_smoke():
    config = SimulationConfig(
        nx=40,
        ny=40,
        re=20.0,
        u_inf=0.05,
        T_inf=0.5,
        diameter=6.0,
        steps=200,
        save_every=0,
        out_dir=None,
    )
    result = run_simulation(config)
    assert np.all(np.isfinite(result.rho))
    assert np.all(np.isfinite(result.ux))
    assert np.all(np.isfinite(result.uy))
    assert np.all(np.isfinite(result.T))
    assert result.kinetic_energy.shape == (200,)
    assert np.all(np.isfinite(result.kinetic_energy))
    speed = np.sqrt(result.ux**2 + result.uy**2)
    assert float(np.max(speed)) < 0.5
    assert result.tau > 0.5
    assert result.wall_time_s >= 0.0
    fluid = ~result.obstacle
    assert np.all(result.T[fluid] > 0.0)
    assert not result.obstacle[0, :].all()
    assert not result.obstacle[-1, :].all()


def test_config_tau_from_re():
    config = SimulationConfig(u_inf=0.1, diameter=20.0, re=20.0, T_inf=0.5)
    nu = config.viscosity()
    assert np.isclose(nu, 0.1 * 20.0 / 20.0)
    assert np.isclose(config.tau(), nu / 0.5 + 0.5)
    assert config.tau() > 0.5


def test_config_tau_uses_T_inf():
    config = SimulationConfig(u_inf=0.1, diameter=20.0, re=20.0, T_inf=0.4)
    nu = config.viscosity()
    assert np.isclose(config.tau(), nu / 0.4 + 0.5)


def test_config_requires_square_domain():
    config = SimulationConfig(nx=40, ny=41)
    try:
        config.validate()
    except ValueError as exc:
        assert "square" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for non-square domain")
