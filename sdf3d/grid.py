"""Grid sampling utilities for 3D signed distance functions."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import numpy.typing as npt

from .geometry import Geometry3D

_Array = npt.NDArray[np.floating]
_Bounds3D = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
_Resolution3D = Tuple[int, int, int]


def sample_levelset_3d(
    geom: Geometry3D,
    bounds: _Bounds3D,
    resolution: _Resolution3D,
) -> _Array:
    """Sample *geom* on a uniform 3-D cell-centred grid.

    Parameters
    ----------
    geom:
        A 3-D geometry whose ``sdf()`` method accepts ``(..., 3)`` arrays.
    bounds:
        ``((x0, x1), (y0, y1), (z0, z1))`` physical extents of the domain.
    resolution:
        ``(nx, ny, nz)`` number of cells along each axis.

    Returns
    -------
    numpy.ndarray
        Shape ``(nz, ny, nx)`` array of signed distances, z-first indexing.
    """
    (x0, x1), (y0, y1), (z0, z1) = bounds
    nx, ny, nz = resolution

    xs = np.linspace(x0, x1, nx, endpoint=False) + (x1 - x0) / (2.0 * nx)
    ys = np.linspace(y0, y1, ny, endpoint=False) + (y1 - y0) / (2.0 * ny)
    zs = np.linspace(z0, z1, nz, endpoint=False) + (z1 - z0) / (2.0 * nz)

    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    p = np.stack([X, Y, Z], axis=-1)
    return geom.sdf(p)


def save_npy(path: str, phi: _Array) -> None:
    """Save *phi* array to *path* (creates parent directories if needed)."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(path, phi)
