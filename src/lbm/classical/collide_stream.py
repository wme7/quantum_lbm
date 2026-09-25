"""Classical Shan-He BGK collision (stream lives in ``lbm.core``)."""

from __future__ import annotations

import numpy as np

from lbm.classical.lattice import equilibrium, moments
from lbm.core.stream import stream

_T_MIN = 1e-12


def collide(f: np.ndarray, nu: float) -> np.ndarray:
    """BGK collision with local ``tau = nu / T + 0.5``."""
    rho, ux, uy, T = moments(f)
    feq = equilibrium(rho, ux, uy, T)
    tau = nu / np.maximum(T, _T_MIN) + 0.5
    return f - (f - feq) / tau


def collide_stream(f: np.ndarray, nu: float) -> np.ndarray:
    """Convenience: collide then stream."""
    return stream(collide(f, nu))
