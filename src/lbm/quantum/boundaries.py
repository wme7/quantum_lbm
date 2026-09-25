"""Quantum boundary conditions: free-stream far field and bounce-back."""

from __future__ import annotations

import numpy as np

from lbm.core.constants import OPPOSITE, Q
from lbm.quantum.eos import fugacity
from lbm.quantum.lattice import equilibrium, moments


def bounce_back(f: np.ndarray, obstacle: np.ndarray) -> None:
    """Fullway bounce-back on solid nodes (in-place)."""
    f_solid = f[:, obstacle].copy()
    for i in range(Q):
        f[i, obstacle] = f_solid[OPPOSITE[i]]


def apply_far_field(
    f: np.ndarray,
    rho_inf: float,
    u_inf: float,
    T_inf: float,
    z_inf: float,
    eta: int,
) -> None:
    """Set all four outer edges to free-stream quantum equilibrium (in-place)."""
    ny, nx = f.shape[1], f.shape[2]
    rho = np.full((ny, nx), rho_inf, dtype=np.float64)
    ux = np.full((ny, nx), u_inf, dtype=np.float64)
    uy = np.zeros((ny, nx), dtype=np.float64)
    T = np.full((ny, nx), T_inf, dtype=np.float64)
    z = np.full((ny, nx), z_inf, dtype=np.float64)
    feq = equilibrium(rho, ux, uy, T, z, eta)
    f[:, :, 0] = feq[:, :, 0]
    f[:, :, -1] = feq[:, :, -1]
    f[:, 0, :] = feq[:, 0, :]
    f[:, -1, :] = feq[:, -1, :]


def apply_boundaries(
    f: np.ndarray,
    obstacle: np.ndarray,
    rho_inf: float,
    u_inf: float,
    T_inf: float,
    z_inf: float,
    eta: int,
) -> None:
    """Apply free-stream far-field equilibrium, then bounce-back on the cylinder."""
    apply_far_field(
        f,
        rho_inf=rho_inf,
        u_inf=u_inf,
        T_inf=T_inf,
        z_inf=z_inf,
        eta=eta,
    )
    bounce_back(f, obstacle)


def fluid_state(
    f: np.ndarray,
    obstacle: np.ndarray,
    h: float,
    eta: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Moments with solid nodes zeroed for diagnostics."""
    rho, ux, uy, T, z = moments(f, h=h, eta=eta)
    fluid = ~obstacle
    ux = np.where(fluid, ux, 0.0)
    uy = np.where(fluid, uy, 0.0)
    rho = np.where(fluid, rho, 1.0)
    T = np.where(fluid, T, 0.0)
    z = np.where(fluid, z, 0.0)
    return rho, ux, uy, T, z


def free_stream_fugacity(
    rho_inf: float, T_inf: float, h: float, eta: int
) -> float:
    """Scalar free-stream fugacity from ``(ρ∞, T∞, h, η)``."""
    return float(fugacity(rho_inf, T_inf, h, eta))
