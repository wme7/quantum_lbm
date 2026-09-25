"""Quantum Yang-Hung simulation configuration and time-stepping loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from lbm.core.geometry import FlowGeometry, make_obstacle_mask
from lbm.core.stream import stream
from lbm.quantum.boundaries import (
    apply_boundaries,
    fluid_state,
    free_stream_fugacity,
)
from lbm.quantum.collide_stream import collide
from lbm.quantum.eos import chi, eta_from_statistic
from lbm.quantum.lattice import equilibrium

StatisticName = Literal["FD", "BE", "MB"]


@dataclass(frozen=True)
class SimulationConfig:
    nx: int = 201
    ny: int = 201
    re: float = 20.0
    u_inf: float = 0.1
    T_inf: float = 0.5
    diameter: float = 20.0
    cx: float | None = None
    cy: float | None = None
    steps: int = 5000
    rho_inf: float = 1.0
    save_every: int = 0
    include_vorticity: bool = False
    out_dir: Path | None = None
    statistic: StatisticName = "MB"
    h: float = 1.0

    def resolved_cx(self) -> float:
        return float(self.nx / 2 if self.cx is None else self.cx)

    def resolved_cy(self) -> float:
        return float(self.ny / 2 if self.cy is None else self.cy)

    def eta(self) -> int:
        return eta_from_statistic(self.statistic)  # type: ignore[arg-type]

    def viscosity(self) -> float:
        return self.u_inf * self.diameter / self.re

    def z_inf(self) -> float:
        return free_stream_fugacity(self.rho_inf, self.T_inf, self.h, self.eta())

    def chi_inf(self) -> float:
        return float(chi(self.z_inf(), self.eta()))

    def tau(self) -> float:
        return self.viscosity() / (self.T_inf * self.chi_inf()) + 0.5

    def geometry(self) -> FlowGeometry:
        return FlowGeometry(
            nx=self.nx,
            ny=self.ny,
            diameter=self.diameter,
            cx=self.resolved_cx(),
            cy=self.resolved_cy(),
        )

    def validate(self) -> None:
        if self.statistic not in ("FD", "BE", "MB"):
            raise ValueError(f"statistic must be FD, BE, or MB, got {self.statistic!r}")
        if self.h <= 0.0:
            raise ValueError("h must be positive")
        if self.nx < 8 or self.ny < 8:
            raise ValueError("nx and ny must be at least 8")
        if self.nx != self.ny:
            raise ValueError(f"square domain required: nx ({self.nx}) != ny ({self.ny})")
        if self.u_inf <= 0.0:
            raise ValueError("u_inf must be positive")
        if self.T_inf <= 0.0:
            raise ValueError("T_inf must be positive")
        if self.rho_inf <= 0.0:
            raise ValueError("rho_inf must be positive")
        if self.re <= 0.0:
            raise ValueError("re must be positive")
        if self.diameter <= 0.0:
            raise ValueError("diameter must be positive")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        z_inf = self.z_inf()
        if self.statistic == "BE" and z_inf >= 1.0 - 1e-6:
            raise ValueError(
                f"Bose free-stream fugacity z_inf={z_inf:.6f} too close to 1; "
                "reduce h or density"
            )
        tau = self.tau()
        if tau <= 0.5:
            raise ValueError(f"tau must be > 0.5, got {tau}")
        mach = self.u_inf / np.sqrt(self.T_inf * self.chi_inf())
        if mach > 0.3:
            raise ValueError(
                f"free-stream Mach number too high ({mach:.3f}); reduce u_inf"
            )


@dataclass
class SimulationResult:
    rho: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    T: np.ndarray
    z: np.ndarray
    obstacle: np.ndarray
    kinetic_energy: np.ndarray
    tau: float
    re: float
    u_inf: float
    T_inf: float
    z_inf: float
    statistic: str
    h: float
    wall_time_s: float
    config: SimulationConfig


def kinetic_energy(ux: np.ndarray, uy: np.ndarray, obstacle: np.ndarray) -> float:
    fluid = ~obstacle
    return 0.5 * float(np.sum(ux[fluid] ** 2 + uy[fluid] ** 2))


def initialize_populations(
    config: SimulationConfig, obstacle: np.ndarray
) -> np.ndarray:
    ny, nx = config.ny, config.nx
    eta = config.eta()
    z_inf = config.z_inf()
    rho = np.full((ny, nx), config.rho_inf, dtype=np.float64)
    ux = np.full((ny, nx), config.u_inf, dtype=np.float64)
    uy = np.zeros((ny, nx), dtype=np.float64)
    T = np.full((ny, nx), config.T_inf, dtype=np.float64)
    z = np.full((ny, nx), z_inf, dtype=np.float64)
    ux[obstacle] = 0.0
    uy[obstacle] = 0.0
    return equilibrium(rho, ux, uy, T, z, eta)


def _save_fields(
    ux: np.ndarray,
    uy: np.ndarray,
    T: np.ndarray,
    z: np.ndarray,
    obstacle: np.ndarray,
    velocity_path: Path,
    temperature_path: Path,
    fugacity_path: Path,
    title: str,
    *,
    include_vorticity: bool = False,
) -> None:
    from lbm.core.viz import (
        save_fugacity_plot,
        save_temperature_plot,
        save_velocity_plot,
    )

    save_velocity_plot(
        ux,
        uy,
        obstacle,
        velocity_path,
        title=title,
        include_vorticity=include_vorticity,
    )
    save_temperature_plot(T, obstacle, temperature_path, title=title)
    save_fugacity_plot(z, obstacle, fugacity_path, title=title)


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Run quantum D2Q9 Yang-Hung flow past a cylinder in a square far-field box."""
    config.validate()
    geom = config.geometry()
    obstacle = make_obstacle_mask(geom)
    nu = config.viscosity()
    tau = config.tau()
    eta = config.eta()
    z_inf = config.z_inf()
    f = initialize_populations(config, obstacle)

    ke_series = np.empty(config.steps, dtype=np.float64)
    out_dir = config.out_dir
    if out_dir is not None and config.save_every > 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    t0 = perf_counter()
    for step in range(config.steps):
        f = collide(f, nu, h=config.h, eta=eta)
        f = stream(f)
        apply_boundaries(
            f,
            obstacle,
            rho_inf=config.rho_inf,
            u_inf=config.u_inf,
            T_inf=config.T_inf,
            z_inf=z_inf,
            eta=eta,
        )
        rho, ux, uy, T, z = fluid_state(f, obstacle, h=config.h, eta=eta)
        ke_series[step] = kinetic_energy(ux, uy, obstacle)

        if (
            out_dir is not None
            and config.save_every > 0
            and (step + 1) % config.save_every == 0
        ):
            _save_fields(
                ux,
                uy,
                T,
                z,
                obstacle,
                out_dir / f"velocity_{step + 1:06d}.png",
                out_dir / f"temperature_{step + 1:06d}.png",
                out_dir / f"fugacity_{step + 1:06d}.png",
                title=f"step {step + 1}",
                include_vorticity=config.include_vorticity,
            )

    wall_time = perf_counter() - t0
    rho, ux, uy, T, z = fluid_state(f, obstacle, h=config.h, eta=eta)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_fields(
            ux,
            uy,
            T,
            z,
            obstacle,
            out_dir / "velocity_final.png",
            out_dir / "temperature_final.png",
            out_dir / "fugacity_final.png",
            title=f"final (Re={config.re:g}, {config.statistic})",
            include_vorticity=config.include_vorticity,
        )
        np.save(out_dir / "kinetic_energy.npy", ke_series)

    return SimulationResult(
        rho=rho,
        ux=ux,
        uy=uy,
        T=T,
        z=z,
        obstacle=obstacle,
        kinetic_energy=ke_series,
        tau=tau,
        re=config.re,
        u_inf=config.u_inf,
        T_inf=config.T_inf,
        z_inf=z_inf,
        statistic=config.statistic,
        h=config.h,
        wall_time_s=wall_time,
        config=config,
    )
