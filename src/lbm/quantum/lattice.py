"""Yang-Hung N=2 quantum equilibrium (UUB-BGK / Hermite)."""

from __future__ import annotations

import numpy as np

from lbm.core.constants import C2, CS2, CS4, CX, CY, D, Q, WEIGHTS
from lbm.quantum.eos import chi, recover_TZ


def equilibrium(
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    T: np.ndarray,
    z: np.ndarray,
    eta: int,
) -> np.ndarray:
    """Yang-Hung N=2 D2Q9 equilibrium with quantum factor ``χ(z)``.

    Replaces the classical ``(T - cs²)`` term by ``(T χ(z) - cs²)``.
    """
    u_sq = ux * ux + uy * uy
    inv_cs2 = 1.0 / CS2
    inv_2cs4 = 0.5 / CS4
    chi_field = np.asarray(chi(z, eta), dtype=np.float64)
    thermal = (T * chi_field - CS2) * inv_2cs4
    shape = np.broadcast_shapes(
        np.shape(rho), np.shape(ux), np.shape(uy), np.shape(T), np.shape(chi_field)
    )
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
    h: float,
    eta: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Density, velocity, temperature, and fugacity from populations."""
    rho = np.sum(f, axis=0)
    ux = (CX[:, None, None] * f).sum(axis=0) / rho
    uy = (CY[:, None, None] * f).sum(axis=0) / rho
    energy = (C2[:, None, None] * f).sum(axis=0)
    u_sq = ux * ux + uy * uy
    T, z, _chi = recover_TZ(rho, energy, u_sq, h, eta)
    return rho, ux, uy, T, z
