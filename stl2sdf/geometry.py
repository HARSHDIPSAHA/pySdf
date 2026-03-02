"""Public API for stl2sdf: a single function, stl_to_geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from sdf3d.geometry import Geometry3D as _Geometry3D
from ._math import _stl_to_triangles, _triangles_to_sdf


def stl_to_geometry(
    path: Union[str, Path],
    *,
    ray_dir: Optional[np.ndarray] = None,
) -> _Geometry3D:
    """Load an STL file and return a :class:`sdf3d.geometry.Geometry3D`.

    The returned object has the same interface as analytic primitives
    (``Sphere3D``, ``Box3D``, etc.) and can be combined with them using
    ``.union()``, ``.subtract()``, ``.translate()``, etc.

    Sign convention: phi < 0 inside, phi = 0 on surface, phi > 0 outside.
    Requires a **watertight** mesh — sign via ray casting is undefined near
    gaps.  Complexity is O(F × N) per ``.sdf()`` call.

    Parameters
    ----------
    path:
        Path to the ``.stl`` file (binary or ASCII).
    ray_dir:
        Optional ``(3,)`` unit ray direction for sign determination.
        Defaults to an irrational direction that avoids axis-aligned
        degeneracies.

    Examples
    --------
    >>> from stl2sdf import stl_to_geometry
    >>> from sdf3d import Sphere3D
    >>> from sdf3d.grid import sample_levelset_3d
    >>>
    >>> wheel = stl_to_geometry("mars_wheel.stl")
    >>> combined = wheel.union(Sphere3D(0.3).translate(0.5, 0, 0))
    >>> phi = sample_levelset_3d(combined, bounds=((-1,1),(-1,1),(-1,1)), resolution=(32,32,32))
    """
    triangles = _stl_to_triangles(path)

    def _sdf(p: np.ndarray) -> np.ndarray:
        shape = p.shape[:-1]
        return _triangles_to_sdf(p.reshape(-1, 3), triangles, ray_dir=ray_dir).reshape(shape)

    return _Geometry3D(_sdf)
