"""Lattice streaming operator."""

from __future__ import annotations

import numpy as np

from lbm.core.constants import CX, CY, Q


def stream(f: np.ndarray) -> np.ndarray:
    """Stream populations along lattice velocities (periodic ``np.roll``)."""
    out = np.empty_like(f)
    for i in range(Q):
        out[i] = np.roll(f[i], shift=(int(CY[i]), int(CX[i])), axis=(0, 1))
    return out
