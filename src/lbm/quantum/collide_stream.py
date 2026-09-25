"""Quantum Yang-Hung BGK collision (stream lives in ``lbm.core``)."""

from __future__ import annotations

import numpy as np

from lbm.core.stream import stream
from lbm.quantum.eos import chi
from lbm.quantum.lattice import equilibrium, moments

_T_MIN = 1e-12


def collide(f: np.ndarray, nu: float, h: float, eta: int) -> np.ndarray:
    """BGK collision with local ``tau = nu / (T χ) + 0.5``."""
    rho, ux, uy, T, z = moments(f, h=h, eta=eta)
    feq = equilibrium(rho, ux, uy, T, z, eta)
    chi_field = np.asarray(chi(z, eta), dtype=np.float64)
    tau = nu / np.maximum(T * chi_field, _T_MIN) + 0.5
    return f - (f - feq) / tau


def collide_stream(f: np.ndarray, nu: float, h: float, eta: int) -> np.ndarray:
    """Convenience: collide then stream."""
    return stream(collide(f, nu, h=h, eta=eta))
