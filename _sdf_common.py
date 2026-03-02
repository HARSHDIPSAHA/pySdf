"""Shared SDF helpers used by both sdf2d and sdf3d.

This module provides:

* **Type alias**: :data:`_F`
* **Vector constructors**: :func:`vec2`, :func:`vec3`
* **Math helpers**: :func:`length`, :func:`dot`, :func:`dot2`, :func:`clamp`,
  :func:`safe_div`
* **Shared boolean/domain operators** (used by both 2D and 3D geometry):
  :func:`opUnion`, :func:`opSubtraction`, :func:`opIntersection`,
  :func:`opRound`, :func:`opOnion`, :func:`opScale`

Not meant to be imported directly by end users — import from
``sdf2d.primitives`` or ``sdf3d.primitives`` instead.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
_F = npt.NDArray[np.floating]

__all__ = [
    "_F",
    "vec2", "vec3",
    "length", "dot", "dot2", "clamp", "safe_div",
    "opUnion", "opSubtraction", "opIntersection",
    "opRound", "opOnion", "opScale",
]


# ===========================================================================
# Vector constructors
# ===========================================================================

def vec2(x: _F, y: _F) -> _F:
    """Stack *x* and *y* into a ``(..., 2)`` array."""
    x, y = np.broadcast_arrays(x, y)
    return np.stack([x, y], axis=-1)


def vec3(x: _F, y: _F, z: _F) -> _F:
    """Stack *x*, *y*, *z* into a ``(..., 3)`` array."""
    x, y, z = np.broadcast_arrays(x, y, z)
    return np.stack([x, y, z], axis=-1)


# ===========================================================================
# Math helpers
# ===========================================================================

def length(v: _F) -> _F:
    """Euclidean length along the last axis."""
    return np.linalg.norm(v, axis=-1)


def dot(a: _F, b: _F) -> _F:
    """Dot product along the last axis."""
    return np.sum(a * b, axis=-1)


def dot2(a: _F) -> _F:
    """Squared length: ``dot(a, a)``."""
    return dot(a, a)


def clamp(x: _F, lo: float | _F, hi: float | _F) -> _F:
    """Clamp *x* element-wise to ``[lo, hi]``."""
    return np.minimum(np.maximum(x, lo), hi)


def safe_div(n: _F, d: _F, eps: float = 1e-12) -> _F:
    """Division that avoids exact zero in the denominator."""
    return n / np.where(np.abs(d) < eps, np.sign(d) * eps + eps, d)


# ===========================================================================
# Shared boolean / domain operators (used by both sdf2d and sdf3d)
# ===========================================================================

def opUnion(d1: _F, d2: _F) -> _F:
    """Union of two SDFs: ``min(d1, d2)``."""
    return np.minimum(d1, d2)


def opSubtraction(d1: _F, d2: _F) -> _F:
    """Subtract *d1* from *d2*: ``max(-d1, d2)``."""
    return np.maximum(-d1, d2)


def opIntersection(d1: _F, d2: _F) -> _F:
    """Intersection of two SDFs: ``max(d1, d2)``."""
    return np.maximum(d1, d2)


def opRound(p: _F, primitive: "_SDFFunc", rad: float) -> _F:  # type: ignore[name-defined]
    """Round a primitive outward by *rad*."""
    return primitive(p) - rad


def opOnion(sdf_val: _F, thickness: float) -> _F:
    """Turn a solid into a shell of *thickness*."""
    return np.abs(sdf_val) - thickness


def opScale(p: _F, s: float, primitive: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Uniformly scale a primitive by factor *s*."""
    return primitive(p / s) * s
