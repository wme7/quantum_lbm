"""Square far-field geometry and solid masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlowGeometry:
    """Square far-field box with a circular cylinder obstacle."""

    nx: int
    ny: int
    diameter: float
    cx: float
    cy: float

    @property
    def radius(self) -> float:
        return self.diameter / 2.0


def make_cylinder_mask(geom: FlowGeometry) -> np.ndarray:
    """Boolean obstacle mask of shape ``(ny, nx)``; ``True`` = solid."""
    y, x = np.ogrid[: geom.ny, : geom.nx]
    dist2 = (x - geom.cx) ** 2 + (y - geom.cy) ** 2
    return dist2 <= geom.radius**2


def make_obstacle_mask(geom: FlowGeometry) -> np.ndarray:
    """Solid mask: cylinder only (no channel walls)."""
    return make_cylinder_mask(geom)
