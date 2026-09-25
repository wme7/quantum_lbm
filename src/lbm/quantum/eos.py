"""Quantum EOS helpers wrapping ``ideal_gases`` (Yang-Hung)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from ideal_gases import G, find_fugacity

from lbm.core.constants import D

Statistic = Literal["FD", "BE", "MB"]

_ETA = {"FD": -1, "BE": 1, "MB": 0}
_T_MIN = 1e-12
_DEFAULT_ITERS = 12
_DEFAULT_TOL = 1e-10


def eta_from_statistic(statistic: Statistic) -> int:
    try:
        return _ETA[statistic]
    except KeyError as exc:
        raise ValueError(f"unknown statistic {statistic!r}; expected FD, BE, or MB") from exc


def chi(z: np.ndarray | float, eta: int) -> np.ndarray | float:
    """Pressure factor ``g_{5/2}(z) / g_{3/2}(z)`` (MB → 1)."""
    if eta == 0:
        if np.isscalar(z):
            return 1.0
        return np.ones(np.shape(z), dtype=np.float64)
    g32 = G(1.5, z, eta)
    g52 = G(2.5, z, eta)
    return g52 / g32


def fugacity(
    rho: np.ndarray | float,
    T: np.ndarray | float,
    h: float,
    eta: int,
) -> np.ndarray | float:
    """Invert ``(ρ, T) → z`` with 3D quantum density law (Yang-Hung)."""
    return find_fugacity(rho, T, dim=3, h=h, eta=eta)


def recover_TZ(
    rho: np.ndarray,
    energy: np.ndarray,
    u_sq: np.ndarray,
    h: float,
    eta: int,
    *,
    n_iter: int = _DEFAULT_ITERS,
    tol: float = _DEFAULT_TOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover ``(T, z, χ)`` from density and second moment.

    Lattice energy uses spatial ``D=2``:
    ``energy = ρ (u² + D T χ(z))``.
    Fugacity uses the 3D density relation via ``find_fugacity``.
    """
    rho_safe = np.maximum(rho, _T_MIN)
    thermal = np.maximum((energy / rho_safe - u_sq) / D, _T_MIN)
    T = thermal.copy()
    z = np.empty_like(T)
    chi_field = np.ones_like(T)

    for _ in range(n_iter):
        z = np.asarray(fugacity(rho_safe, T, h, eta), dtype=np.float64)
        if eta == 1:
            z = np.minimum(z, 1.0 - 1e-12)
        chi_field = np.asarray(chi(z, eta), dtype=np.float64)
        T_new = thermal / np.maximum(chi_field, _T_MIN)
        if float(np.max(np.abs(T_new - T))) < tol:
            T = T_new
            break
        T = T_new

    if eta == 1 and np.any(z >= 1.0):
        raise ValueError("Bose fugacity reached z >= 1 during recover_TZ")

    return T, z, chi_field
