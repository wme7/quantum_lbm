"""Classical Shan-He simulation configuration and time-stepping loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from lbm.classical.boundaries import apply_boundaries, fluid_state
from lbm.classical.collide_stream import collide
from lbm.classical.lattice import equilibrium
from lbm.core.constants import CX, CY, OPPOSITE, Q
from lbm.core.geometry import FlowGeometry, make_obstacle_mask
from lbm.core.stream import stream


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

    def resolved_cx(self) -> float:
        return float(self.nx / 2 if self.cx is None else self.cx)

    def resolved_cy(self) -> float:
        return float(self.ny / 2 if self.cy is None else self.cy)

    def viscosity(self) -> float:
        return self.u_inf * self.diameter / self.re

    def tau(self) -> float:
        return self.viscosity() / self.T_inf + 0.5

    def geometry(self) -> FlowGeometry:
        return FlowGeometry(
            nx=self.nx,
            ny=self.ny,
            diameter=self.diameter,
            cx=self.resolved_cx(),
            cy=self.resolved_cy(),
        )

    def validate(self) -> None:
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
        tau = self.tau()
        if tau <= 0.5:
            raise ValueError(f"tau must be > 0.5, got {tau}")
        mach = self.u_inf / np.sqrt(self.T_inf)
        if mach > 0.3:
            raise ValueError(f"free-stream Mach number too high ({mach:.3f}); reduce u_inf")


@dataclass
class SimulationResult:
    rho: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    T: np.ndarray
    obstacle: np.ndarray
    kinetic_energy: np.ndarray
    tau: float
    re: float
    u_inf: float
    T_inf: float
    wall_time_s: float
    config: SimulationConfig


def kinetic_energy(ux: np.ndarray, uy: np.ndarray, obstacle: np.ndarray) -> float:
    fluid = ~obstacle
    return 0.5 * float(np.sum(ux[fluid] ** 2 + uy[fluid] ** 2))


def momentum_exchange_force(
    f_pre_bb: np.ndarray,
    f_post_bb: np.ndarray,
    obstacle: np.ndarray,
) -> tuple[float, float]:
    """Drag/lift via momentum exchange on bounce-back links (cylinder)."""
    fx = 0.0
    fy = 0.0
    if not np.any(obstacle):
        return 0.0, 0.0

    for i in range(Q):
        opp = int(OPPOSITE[i])
        delta = f_pre_bb[i, obstacle] + f_post_bb[opp, obstacle]
        fx += float(CX[i]) * float(np.sum(delta))
        fy += float(CY[i]) * float(np.sum(delta))
    return fx, fy


def initialize_populations(
    config: SimulationConfig, obstacle: np.ndarray
) -> np.ndarray:
    ny, nx = config.ny, config.nx
    rho = np.full((ny, nx), config.rho_inf, dtype=np.float64)
    ux = np.full((ny, nx), config.u_inf, dtype=np.float64)
    uy = np.zeros((ny, nx), dtype=np.float64)
    T = np.full((ny, nx), config.T_inf, dtype=np.float64)
    ux[obstacle] = 0.0
    uy[obstacle] = 0.0
    return equilibrium(rho, ux, uy, T)


def _save_fields(
    ux: np.ndarray,
    uy: np.ndarray,
    T: np.ndarray,
    obstacle: np.ndarray,
    velocity_path: Path,
    temperature_path: Path,
    title: str,
    *,
    include_vorticity: bool = False,
) -> None:
    from lbm.core.viz import save_temperature_plot, save_velocity_plot

    save_velocity_plot(
        ux,
        uy,
        obstacle,
        velocity_path,
        title=title,
        include_vorticity=include_vorticity,
    )
    save_temperature_plot(T, obstacle, temperature_path, title=title)


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Run classical D2Q9 Shan-He flow past a cylinder in a square far-field box."""
    config.validate()
    geom = config.geometry()
    obstacle = make_obstacle_mask(geom)
    nu = config.viscosity()
    tau = config.tau()
    f = initialize_populations(config, obstacle)

    ke_series = np.empty(config.steps, dtype=np.float64)
    out_dir = config.out_dir
    if out_dir is not None and config.save_every > 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    t0 = perf_counter()
    for step in range(config.steps):
        f = collide(f, nu)
        f = stream(f)
        apply_boundaries(
            f,
            obstacle,
            rho_inf=config.rho_inf,
            u_inf=config.u_inf,
            T_inf=config.T_inf,
        )
        rho, ux, uy, T = fluid_state(f, obstacle)
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
                obstacle,
                out_dir / f"velocity_{step + 1:06d}.png",
                out_dir / f"temperature_{step + 1:06d}.png",
                title=f"step {step + 1}",
                include_vorticity=config.include_vorticity,
            )

    wall_time = perf_counter() - t0
    rho, ux, uy, T = fluid_state(f, obstacle)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_fields(
            ux,
            uy,
            T,
            obstacle,
            out_dir / "velocity_final.png",
            out_dir / "temperature_final.png",
            title=f"final (Re={config.re:g})",
            include_vorticity=config.include_vorticity,
        )
        np.save(out_dir / "kinetic_energy.npy", ke_series)

    return SimulationResult(
        rho=rho,
        ux=ux,
        uy=uy,
        T=T,
        obstacle=obstacle,
        kinetic_energy=ke_series,
        tau=tau,
        re=config.re,
        u_inf=config.u_inf,
        T_inf=config.T_inf,
        wall_time_s=wall_time,
        config=config,
    )
