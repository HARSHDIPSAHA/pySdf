"""Public API for stl2sdf: a single function, stl_to_geometry."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

from sdf3d.geometry import Geometry3D as _Geometry3D
from ._math import _stl_to_triangles


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
    Requires a **watertight** mesh.  SDF evaluation is parallelised across
    CPU cores by `pysdf <https://github.com/sxyu/sdf>`_.

    Parameters
    ----------
    path:
        Path to the ``.stl`` file (binary or ASCII).
    ray_dir:
        Deprecated and ignored.  ``pysdf`` handles sign determination
        internally; pass ``None`` (the default).

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
    from pysdf import SDF  # lazy import: clear ImportError if pysdf is missing

    if ray_dir is not None:
        warnings.warn(
            "ray_dir is ignored; pysdf handles sign determination internally.",
            UserWarning,
            stacklevel=2,
        )

    triangles = _stl_to_triangles(path)  # (F, 3, 3) float64
    # Deduplicate vertices so pysdf receives a proper indexed mesh.
    # np.unique on float32 STL coordinates is exact (no arithmetic, just sorting).
    verts_raw = triangles.reshape(-1, 3).astype(np.float32)
    verts, inverse = np.unique(verts_raw, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.uint32)
    verts = np.ascontiguousarray(verts)
    faces = np.ascontiguousarray(faces)

    _sdf_obj = SDF(verts, faces)

    def _sdf(p: np.ndarray) -> np.ndarray:
        shape = p.shape[:-1]
        pts = np.ascontiguousarray(p.reshape(-1, 3), dtype=np.float32)
        # pysdf: positive inside, (N,1) float32 — negate and cast to match our convention
        return -_sdf_obj(pts).reshape(shape).astype(np.float64)

    return _Geometry3D(_sdf)
