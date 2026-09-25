"""Command-line interface for classical and quantum D2Q9 LBM solvers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer

from lbm import __version__

app = typer.Typer(
    name="lbm",
    help="2D D2Q9 LBM: classical Shan-He and quantum Yang-Hung cylinder flow.",
    no_args_is_help=True,
)

run_app = typer.Typer(help="Run a cylinder simulation.", no_args_is_help=True)
app.add_typer(run_app, name="run")


@app.callback()
def main() -> None:
    """Quantum-LBM: classical Shan-He and quantum Yang-Hung solvers."""


def _shared_echo(
    *,
    nx: int,
    ny: int,
    re: float,
    u_inf: float,
    t_inf: float,
    tau: float,
    nu: float,
    steps: int,
    wall_time_s: float,
    ke_final: float,
    out: Path | None,
    extra_lines: list[str] | None = None,
) -> None:
    typer.echo(f"quantum-lbm {__version__}")
    typer.echo(f"  grid:     {nx} x {ny}")
    typer.echo(f"  Re:       {re:g}")
    typer.echo(f"  u_inf:    {u_inf:g}")
    typer.echo(f"  T_inf:    {t_inf:g}")
    if extra_lines:
        for line in extra_lines:
            typer.echo(line)
    typer.echo(f"  tau:      {tau:.6f}")
    typer.echo(f"  nu:       {nu:.6e}")
    typer.echo(f"  steps:    {steps}")
    typer.echo(f"  wall:     {wall_time_s:.3f} s")
    typer.echo(f"  KE final: {ke_final:.6e}")
    if out is not None:
        typer.echo(f"  wrote:    {out}")


@run_app.command("classical")
def run_classical(
    nx: int = typer.Option(201, help="Grid points in x (square domain: nx == ny)."),
    ny: int = typer.Option(201, help="Grid points in y (square domain: nx == ny)."),
    re: float = typer.Option(20.0, help="Reynolds number based on diameter."),
    u_inf: float = typer.Option(
        0.1, "--u-inf", help="Free-stream velocity (lattice units)."
    ),
    t_inf: float = typer.Option(
        0.5, "--t-inf", help="Free-stream temperature (lattice units)."
    ),
    diameter: float = typer.Option(20.0, help="Cylinder diameter (lattice units)."),
    cx: float | None = typer.Option(None, help="Cylinder center x (default: nx/2)."),
    cy: float | None = typer.Option(None, help="Cylinder center y (default: ny/2)."),
    steps: int = typer.Option(5000, help="Number of time steps."),
    out: Path | None = typer.Option(None, "--out", help="Output directory for plots."),
    save_every: int = typer.Option(
        0,
        "--save-every",
        help="Save velocity and temperature PNGs every N steps (0 = final only if --out).",
    ),
    vorticity: bool = typer.Option(
        False,
        "--vorticity/--no-vorticity",
        help="Include vorticity panel in velocity PNGs (requires --out).",
    ),
    rho_inf: float = typer.Option(
        1.0, "--rho-inf", help="Free-stream density (lattice units)."
    ),
) -> None:
    """Run classical Shan-He N=2 thermal D2Q9 (verification baseline)."""
    from lbm.classical import SimulationConfig, run_simulation

    config = SimulationConfig(
        nx=nx,
        ny=ny,
        re=re,
        u_inf=u_inf,
        T_inf=t_inf,
        diameter=diameter,
        cx=cx,
        cy=cy,
        steps=steps,
        rho_inf=rho_inf,
        save_every=save_every,
        include_vorticity=vorticity,
        out_dir=out,
    )
    result = run_simulation(config)
    _shared_echo(
        nx=nx,
        ny=ny,
        re=result.re,
        u_inf=result.u_inf,
        t_inf=result.T_inf,
        tau=result.tau,
        nu=config.viscosity(),
        steps=steps,
        wall_time_s=result.wall_time_s,
        ke_final=float(result.kinetic_energy[-1]),
        out=out,
        extra_lines=["  solver:   classical Shan-He"],
    )


@run_app.command("quantum")
def run_quantum(
    statistic: Literal["FD", "BE", "MB"] = typer.Option(
        ...,
        "--statistic",
        help="Quantum statistics: FD (Fermi–Dirac), BE (Bose–Einstein), MB (Maxwell–Boltzmann).",
    ),
    h: float = typer.Option(
        ...,
        "--h",
        help="Planck/degeneracy parameter for find_fugacity.",
    ),
    nx: int = typer.Option(201, help="Grid points in x (square domain: nx == ny)."),
    ny: int = typer.Option(201, help="Grid points in y (square domain: nx == ny)."),
    re: float = typer.Option(20.0, help="Reynolds number based on diameter."),
    u_inf: float = typer.Option(
        0.1, "--u-inf", help="Free-stream velocity (lattice units)."
    ),
    t_inf: float = typer.Option(
        0.5, "--t-inf", help="Free-stream temperature (lattice units)."
    ),
    diameter: float = typer.Option(20.0, help="Cylinder diameter (lattice units)."),
    cx: float | None = typer.Option(None, help="Cylinder center x (default: nx/2)."),
    cy: float | None = typer.Option(None, help="Cylinder center y (default: ny/2)."),
    steps: int = typer.Option(5000, help="Number of time steps."),
    out: Path | None = typer.Option(None, "--out", help="Output directory for plots."),
    save_every: int = typer.Option(
        0,
        "--save-every",
        help="Save field PNGs every N steps (0 = final only if --out).",
    ),
    vorticity: bool = typer.Option(
        False,
        "--vorticity/--no-vorticity",
        help="Include vorticity panel in velocity PNGs (requires --out).",
    ),
    rho_inf: float = typer.Option(
        1.0, "--rho-inf", help="Free-stream density (lattice units)."
    ),
) -> None:
    """Run quantum Yang-Hung UUB-BGK D2Q9 (N=2)."""
    from lbm.quantum import SimulationConfig, run_simulation

    config = SimulationConfig(
        nx=nx,
        ny=ny,
        re=re,
        u_inf=u_inf,
        T_inf=t_inf,
        diameter=diameter,
        cx=cx,
        cy=cy,
        steps=steps,
        rho_inf=rho_inf,
        save_every=save_every,
        include_vorticity=vorticity,
        out_dir=out,
        statistic=statistic,
        h=h,
    )
    result = run_simulation(config)
    _shared_echo(
        nx=nx,
        ny=ny,
        re=result.re,
        u_inf=result.u_inf,
        t_inf=result.T_inf,
        tau=result.tau,
        nu=config.viscosity(),
        steps=steps,
        wall_time_s=result.wall_time_s,
        ke_final=float(result.kinetic_energy[-1]),
        out=out,
        extra_lines=[
            "  solver:   quantum Yang-Hung",
            f"  statistic:{result.statistic}",
            f"  h:        {result.h:g}",
            f"  z_inf:    {result.z_inf:.6f}",
        ],
    )


if __name__ == "__main__":
    app()
