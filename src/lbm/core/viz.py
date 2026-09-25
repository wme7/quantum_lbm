"""Visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def velocity_magnitude(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    return np.sqrt(ux * ux + uy * uy)


def vorticity(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Discrete vorticity ``duy/dx - dux/dy`` via central differences."""
    duy_dx = 0.5 * (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
    dux_dy = 0.5 * (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0))
    return duy_dx - dux_dy


def save_velocity_plot(
    ux: np.ndarray,
    uy: np.ndarray,
    obstacle: np.ndarray,
    path: Path | str,
    title: str = "",
    *,
    include_vorticity: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    speed = velocity_magnitude(ux, uy)
    speed = np.ma.array(speed, mask=obstacle)

    if include_vorticity:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
        im0 = axes[0].imshow(speed, origin="lower", cmap="viridis", aspect="equal")
        axes[0].set_title(title or "|u|")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)
        vort = np.ma.array(vorticity(ux, uy), mask=obstacle)
        vmax = float(np.nanpercentile(np.abs(vort.compressed()), 99)) or 1.0
        im1 = axes[1].imshow(
            vort, origin="lower", cmap="RdBu_r", aspect="equal", vmin=-vmax, vmax=vmax
        )
        axes[1].set_title("vorticity")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
    else:
        fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
        im = ax.imshow(speed, origin="lower", cmap="viridis", aspect="equal")
        ax.set_title(title or "|u|")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_temperature_plot(
    T: np.ndarray,
    obstacle: np.ndarray,
    path: Path | str,
    title: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    field = np.ma.array(T, mask=obstacle)
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    im = ax.imshow(field, origin="lower", cmap="magma", aspect="equal")
    ax.set_title(title or "T")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_fugacity_plot(
    z: np.ndarray,
    obstacle: np.ndarray,
    path: Path | str,
    title: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    field = np.ma.array(z, mask=obstacle)
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    im = ax.imshow(field, origin="lower", cmap="cividis", aspect="equal")
    ax.set_title(title or "z")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=120)
    plt.close(fig)
