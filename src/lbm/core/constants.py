"""D2Q9 lattice constants shared by classical and quantum solvers."""

from __future__ import annotations

import numpy as np

# Direction indexing (x right, y up in array row-up sense with axis 1 = x, axis 0 = y):
#  6  2  5
#  3  0  1
#  7  4  8
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int8)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int8)
WEIGHTS = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=np.float64,
)
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int8)

CS2 = 1.0 / 3.0
CS4 = CS2 * CS2
D = 2
Q = 9
C2 = CX.astype(np.float64) ** 2 + CY.astype(np.float64) ** 2
