"""3D geometry primitives and boolean operations for signed distance functions."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np
import numpy.typing as npt

import sdf_lib as sdf

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
_Array = npt.NDArray[np.floating]
_SDFFunc = Callable[[_Array], _Array]


# ===========================================================================
# Base class
# ===========================================================================

class Geometry3D:
    """Base class for 3D signed-distance-function geometries.

    A ``Geometry3D`` wraps a callable ``func(p) -> distances`` where *p* is
    a ``(..., 3)`` array of 3D points and the return value is a ``(...)``
    array of signed distances.

    Implements:
    - Boolean operations: :meth:`union`, :meth:`subtract`, :meth:`intersect`
    - Modifiers:          :meth:`round`, :meth:`onion`
    - Transforms:         :meth:`translate`, :meth:`scale`, :meth:`elongate`
    - Rotations:          :meth:`rotate_x`, :meth:`rotate_y`, :meth:`rotate_z`
    """

    def __init__(self, func: _SDFFunc) -> None:
        self._func = func

    def sdf(self, p: _Array) -> _Array:
        """Evaluate signed distance at *p* (shape ``(..., 3)``)."""
        return self._func(p)

    def __call__(self, p: _Array) -> _Array:
        return self._func(p)

    # ------------------------------------------------------------------
    # Boolean operations
    # ------------------------------------------------------------------

    def union(self, other: Geometry3D) -> Geometry3D:
        """Return the union (min) of this shape and *other*."""
        return Geometry3D(lambda p: sdf.opUnion(self.sdf(p), other.sdf(p)))

    def subtract(self, other: Geometry3D) -> Geometry3D:
        """Subtract *other* from this shape."""
        return Geometry3D(lambda p: sdf.opSubtraction(other.sdf(p), self.sdf(p)))

    def intersect(self, other: Geometry3D) -> Geometry3D:
        """Return the intersection (max) of this shape and *other*."""
        return Geometry3D(lambda p: sdf.opIntersection(self.sdf(p), other.sdf(p)))

    # ------------------------------------------------------------------
    # Modifiers
    # ------------------------------------------------------------------

    def round(self, rad: float) -> Geometry3D:
        """Round the surface outward by *rad*."""
        return Geometry3D(lambda p: sdf.opRound(p, self.sdf, rad))

    def onion(self, thickness: float) -> Geometry3D:
        """Turn the solid into a hollow shell of *thickness*."""
        return Geometry3D(lambda p: sdf.opOnion(self.sdf(p), thickness))

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def translate(self, tx: float, ty: float, tz: float) -> Geometry3D:
        """Translate by ``(tx, ty, tz)``."""
        t = np.array([tx, ty, tz])
        return Geometry3D(lambda p: self.sdf(p - t))

    def scale(self, s: float) -> Geometry3D:
        """Uniformly scale by factor *s*."""
        return Geometry3D(lambda p: sdf.opScale(p, s, self.sdf))

    def elongate(self, hx: float, hy: float, hz: float) -> Geometry3D:
        """Elongate along each axis by ``(hx, hy, hz)``."""
        h = np.array([hx, hy, hz])
        return Geometry3D(lambda p: sdf.opElongate2(p, self.sdf, h))

    def rotate_x(self, angle_rad: float) -> Geometry3D:
        """Rotate around the X axis by *angle_rad* radians."""
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rot = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        return Geometry3D(lambda p: sdf.opTx(p, rot, np.zeros(3), self.sdf))

    def rotate_y(self, angle_rad: float) -> Geometry3D:
        """Rotate around the Y axis by *angle_rad* radians."""
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        return Geometry3D(lambda p: sdf.opTx(p, rot, np.zeros(3), self.sdf))

    def rotate_z(self, angle_rad: float) -> Geometry3D:
        """Rotate around the Z axis by *angle_rad* radians."""
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return Geometry3D(lambda p: sdf.opTx(p, rot, np.zeros(3), self.sdf))


# ===========================================================================
# Primitive shapes
# ===========================================================================

class Sphere3D(Geometry3D):
    """Sphere centred at origin with given *radius*."""

    def __init__(self, radius: float) -> None:
        super().__init__(lambda p: sdf.sdSphere(p, radius))


class Box3D(Geometry3D):
    """Axis-aligned box with *half_size* ``(hx, hy, hz)`` centred at origin."""

    def __init__(self, half_size: Sequence[float]) -> None:
        b = np.array(half_size, dtype=float)
        super().__init__(lambda p: sdf.sdBox(p, b))


class RoundBox3D(Geometry3D):
    """Axis-aligned box with corner *radius* and *half_size* ``(hx, hy, hz)``."""

    def __init__(self, half_size: Sequence[float], radius: float) -> None:
        b = np.array(half_size, dtype=float)
        super().__init__(lambda p: sdf.sdRoundBox(p, b, radius))


class Cylinder3D(Geometry3D):
    """Infinite cylinder.

    Parameters
    ----------
    axis_offset:
        ``(cx, cz)`` — offset of the axis in XZ-plane.
    radius:
        Cylinder radius.
    """

    def __init__(self, axis_offset: Sequence[float], radius: float) -> None:
        c = np.array([axis_offset[0], axis_offset[1], radius], dtype=float)
        super().__init__(lambda p: sdf.sdCylinder(p, c))


class ConeExact3D(Geometry3D):
    """Exact (signed) cone.

    Parameters
    ----------
    sincos:
        ``(sin(half_angle), cos(half_angle))`` of the cone's apex angle.
    height:
        Height of the cone along Y.
    """

    def __init__(self, sincos: Sequence[float], height: float) -> None:
        c = np.array(sincos, dtype=float)
        super().__init__(lambda p: sdf.sdConeExact(p, c, height))


class Torus3D(Geometry3D):
    """Torus in the XZ plane.

    Parameters
    ----------
    major_minor:
        ``(R, r)`` where *R* is the major (tube-centre) radius and
        *r* is the minor (tube-cross-section) radius.
    """

    def __init__(self, major_minor: Sequence[float]) -> None:
        t = np.array(major_minor, dtype=float)
        super().__init__(lambda p: sdf.sdTorus(p, t))


# ===========================================================================
# Boolean operation classes
# ===========================================================================

class Union3D(Geometry3D):
    """Union of two or more 3-D geometries (minimum SDF)."""

    def __init__(self, *geoms: Geometry3D) -> None:
        def _sdf(p: _Array) -> _Array:
            d = geoms[0].sdf(p)
            for g in geoms[1:]:
                d = sdf.opUnion(d, g.sdf(p))
            return d

        super().__init__(_sdf)


class Intersection3D(Geometry3D):
    """Intersection of two or more 3-D geometries (maximum SDF)."""

    def __init__(self, *geoms: Geometry3D) -> None:
        def _sdf(p: _Array) -> _Array:
            d = geoms[0].sdf(p)
            for g in geoms[1:]:
                d = sdf.opIntersection(d, g.sdf(p))
            return d

        super().__init__(_sdf)


class Subtraction3D(Geometry3D):
    """Subtract *cutter* from *base*."""

    def __init__(self, base: Geometry3D, cutter: Geometry3D) -> None:
        super().__init__(
            lambda p: sdf.opSubtraction(cutter.sdf(p), base.sdf(p))
        )
