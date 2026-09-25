"""Shan-He N=2 thermal equilibrium (classical Maxwell–Boltzmann)."""

from __future__ import annotations

import numpy as np

from lbm.core.constants import C2, CS2, CS4, CX, CY, D, Q, WEIGHTS


def equilibrium(
    rho: np.ndarray, ux: np.ndarray, uy: np.ndarray, T: np.ndarray
) -> np.ndarray:
    """Shan-He N=2 D2Q9 equilibrium (thermal when ``T`` varies).

    Parameters
    ----------
    rho, ux, uy, T :
        Macroscopic fields with shape ``(ny, nx)`` (or broadcastable).
        ``T == CS^2`` recovers the isothermal polynomial.

    Returns
    -------
    feq :
        Array of shape ``(9, ny, nx)``.
    """
    u_sq = ux * ux + uy * uy
    inv_cs2 = 1.0 / CS2
    inv_2cs4 = 0.5 / CS4
    thermal = (T - CS2) * inv_2cs4
    shape = np.broadcast_shapes(np.shape(rho), np.shape(ux), np.shape(uy), np.shape(T))
    feq = np.empty((Q, *shape), dtype=np.float64)
    for i in range(Q):
        cu = CX[i] * ux + CY[i] * uy
        feq[i] = rho * WEIGHTS[i] * (
            1.0
            + inv_cs2 * cu
            + 0.5 * inv_cs2 * inv_cs2 * cu * cu
            - 0.5 * inv_cs2 * u_sq
            + thermal * (C2[i] - D * CS2)
        )
    return feq


def moments(
    f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Density, velocity, and temperature from populations ``f`` of shape ``(9, ny, nx)``."""
    rho = np.sum(f, axis=0)
    ux = (CX[:, None, None] * f).sum(axis=0) / rho
    uy = (CY[:, None, None] * f).sum(axis=0) / rho
    energy = (C2[:, None, None] * f).sum(axis=0)
    T = (energy / rho - ux * ux - uy * uy) / D
    return rho, ux, uy, T
