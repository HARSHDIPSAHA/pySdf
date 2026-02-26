"""Core SDF math primitives and operators (numpy, no AMReX dependency).

This module provides:

* **Vector constructors**: :func:`vec2`, :func:`vec3`
* **Math helpers**: :func:`length`, :func:`dot`, :func:`dot2`, :func:`clamp`,
  :func:`safe_div`
* **3-D primitive SDFs**: ``sdSphere``, ``sdBox``, ``sdTorus``, …
* **3-D unsigned-distance helpers**: ``udTriangle``, ``udQuad``
* **2-D primitive SDFs**: ``sdCircle``, ``sdBox2D``, ``sdHexagon2D``, …
* **Boolean / domain operators**: ``opUnion``, ``opSubtraction``,
  ``opIntersection``, ``opSmoothUnion``, …
* **Space-warp operators**: ``opTx``, ``opElongate2``, ``opRevolution``,
  ``opTwist``, …

All functions accept and return ``numpy.ndarray`` objects and support
broadcasting over arbitrary leading batch dimensions.  A "point array" *p*
has shape ``(..., 2)`` for 2-D functions and ``(..., 3)`` for 3-D functions;
scalar SDF results have shape ``(...,)``.

Formulas are adapted from Inigo Quilez's distance function reference:
https://iquilezles.org/articles/distfunctions/
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Type alias used throughout this module
# ---------------------------------------------------------------------------
_F = npt.NDArray[np.floating]


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
# 3-D primitive SDFs
# ===========================================================================

def sdSphere(p: _F, s: float) -> _F:
    """Sphere of radius *s* centred at the origin."""
    return length(p) - s


def sdBox(p: _F, b: _F) -> _F:
    """Axis-aligned box with half-extents *b* ``(bx, by, bz)``."""
    q = np.abs(p) - b
    return length(np.maximum(q, 0.0)) + np.minimum(np.max(q, axis=-1), 0.0)


def sdRoundBox(p: _F, b: _F, r: float) -> _F:
    """Axis-aligned box with half-extents *b* and corner radius *r*."""
    q = np.abs(p) - b + r
    return length(np.maximum(q, 0.0)) + np.minimum(np.max(q, axis=-1), 0.0) - r


def sdBoxFrame(p: _F, b: _F, e: float) -> _F:
    """Wireframe box with half-extents *b* and wire thickness *e*."""
    p = np.abs(p) - b
    q = np.abs(p + e) - e
    a = length(np.maximum(vec3(p[..., 0], q[..., 1], q[..., 2]), 0.0)) + np.minimum(
        np.max(vec3(p[..., 0], q[..., 1], q[..., 2]), axis=-1), 0.0
    )
    b_ = length(np.maximum(vec3(q[..., 0], p[..., 1], q[..., 2]), 0.0)) + np.minimum(
        np.max(vec3(q[..., 0], p[..., 1], q[..., 2]), axis=-1), 0.0
    )
    c = length(np.maximum(vec3(q[..., 0], q[..., 1], p[..., 2]), 0.0)) + np.minimum(
        np.max(vec3(q[..., 0], q[..., 1], p[..., 2]), axis=-1), 0.0
    )
    return np.minimum(np.minimum(a, b_), c)


def sdTorus(p: _F, t: _F) -> _F:
    """Torus in the XZ plane; *t* = ``(R, r)`` (major, minor radii)."""
    q = vec2(length(p[..., [0, 2]]) - t[0], p[..., 1])
    return length(q) - t[1]


def sdCappedTorus(p: _F, sc: _F, ra: float, rb: float) -> _F:
    """Capped torus; *sc* = ``(sin, cos)`` of the cap half-angle."""
    px = np.abs(p[..., 0])
    py = p[..., 1]
    k = np.where(sc[1] * px > sc[0] * py, px * sc[0] + py * sc[1], length(vec2(px, py)))
    return np.sqrt(dot2(p) + ra * ra - 2.0 * ra * k) - rb


def sdLink(p: _F, le: float, r1: float, r2: float) -> _F:
    """Chain link; *le* is half-length, *r1*/*r2* are inner/wire radii."""
    q = vec3(p[..., 0], np.maximum(np.abs(p[..., 1]) - le, 0.0), p[..., 2])
    return length(vec2(length(q[..., [0, 1]]) - r1, q[..., 2])) - r2


def sdCylinder(p: _F, c: _F) -> _F:
    """Infinite cylinder; *c* = ``(cx, cz, radius)``."""
    return length(vec2(p[..., 0] - c[0], p[..., 2] - c[1])) - c[2]


def sdConeExact(p: _F, c: _F, h: float) -> _F:
    """Exact signed cone; *c* = ``(sin, cos)`` of half-angle, *h* is height."""
    q = h * vec2(safe_div(c[0], c[1]), -1.0)
    w = vec2(length(p[..., [0, 2]]), p[..., 1])
    a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0)[..., None]
    b = w - q * vec2(clamp(safe_div(w[..., 0], q[0]), 0.0, 1.0), 1.0)
    k = np.sign(q[1])
    d = np.minimum(dot2(a), dot2(b))
    s = np.maximum(k * (w[..., 0] * q[1] - w[..., 1] * q[0]), k * (w[..., 1] - q[1]))
    return np.sqrt(d) * np.sign(s)


def sdConeBound(p: _F, c: _F, h: float) -> _F:
    """Bound (over-estimate) cone; *c* = ``(sin, cos)``, *h* is height."""
    q = length(p[..., [0, 2]])
    return np.maximum(c[0] * q + c[1] * p[..., 1], -h - p[..., 1])


def sdConeInfinite(p: _F, c: _F) -> _F:
    """Infinite cone; *c* = ``(sin, cos)`` of half-angle."""
    q = vec2(length(p[..., [0, 2]]), -p[..., 1])
    d = length(q - c * np.maximum(dot(q, c), 0.0)[..., None])
    return d * np.where(q[..., 0] * c[1] - q[..., 1] * c[0] < 0.0, -1.0, 1.0)


def sdPlane(p: _F, n: _F, h: float) -> _F:
    """Half-space plane; *n* is the normal (need not be unit), *h* the offset."""
    n = n / np.linalg.norm(n)
    return dot(p, n) + h


def sdHexPrism(p: _F, h: _F) -> _F:
    """Hexagonal prism; *h* = ``(half_hex_radius, half_height)``."""
    k = np.array([-0.8660254, 0.5, 0.57735])
    p = np.abs(p)
    p_xy = p[..., :2]
    p_xy = p_xy - 2.0 * np.minimum(dot(p_xy, k[:2]), 0.0)[..., None] * k[:2]
    d = vec2(
        length(p_xy - vec2(clamp(p_xy[..., 0], -k[2] * h[0], k[2] * h[0]), h[0]))
        * np.sign(p_xy[..., 1] - h[0]),
        p[..., 2] - h[1],
    )
    return np.minimum(np.maximum(d[..., 0], d[..., 1]), 0.0) + length(np.maximum(d, 0.0))


def sdTriPrism(p: _F, h: _F) -> _F:
    """Triangular prism; *h* = ``(half_base, half_height)``."""
    q = np.abs(p)
    return np.maximum(
        q[..., 2] - h[1],
        np.maximum(q[..., 0] * 0.866025 + p[..., 1] * 0.5, -p[..., 1]) - h[0] * 0.5,
    )


def sdCapsule(p: _F, a: _F, b: _F, r: float) -> _F:
    """Capsule from *a* to *b* with radius *r*."""
    pa = p - a
    ba = b - a
    h = clamp(dot(pa, ba) / dot2(ba), 0.0, 1.0)
    return length(pa - ba * h[..., None]) - r


def sdVerticalCapsule(p: _F, h: float, r: float) -> _F:
    """Vertical capsule along Y with half-height *h* and radius *r*."""
    py = p[..., 1] - clamp(p[..., 1], 0.0, h)
    return length(vec3(p[..., 0], py, p[..., 2])) - r


def sdCappedCylinder(p: _F, r: float, h: float) -> _F:
    """Capped cylinder of radius *r* and half-height *h* along Y."""
    d = np.abs(vec2(length(p[..., [0, 2]]), p[..., 1])) - vec2(r, h)
    return np.minimum(np.maximum(d[..., 0], d[..., 1]), 0.0) + length(np.maximum(d, 0.0))


def sdCappedCylinderSegment(p: _F, a: _F, b: _F, r: float) -> _F:
    """Arbitrarily oriented capped cylinder from *a* to *b* with radius *r*."""
    ba   = b - a
    pa   = p - a
    baba = dot2(ba)
    paba = dot(pa, ba)
    x    = length(pa * baba - ba * paba[..., None]) - r * baba
    y    = np.abs(paba - baba * 0.5) - baba * 0.5
    x2   = x * x
    y2   = y * y * baba
    d = np.where(
        np.maximum(x, y) < 0.0,
        -np.minimum(x2, y2),
        (np.where(x > 0.0, x2, 0.0) + np.where(y > 0.0, y2, 0.0)),
    )
    return np.sign(d) * np.sqrt(np.abs(d)) / baba


def sdRoundedCylinder(p: _F, ra: float, rb: float, h: float) -> _F:
    """Rounded cylinder of outer radius *ra*, edge radius *rb*, half-height *h*."""
    d = vec2(length(p[..., [0, 2]]) - ra + rb, np.abs(p[..., 1]) - h + rb)
    return (
        np.minimum(np.maximum(d[..., 0], d[..., 1]), 0.0)
        + length(np.maximum(d, 0.0))
        - rb
    )


def sdCappedCone(p: _F, h: float, r1: float, r2: float) -> _F:
    """Capped cone along Y with half-height *h*, base radius *r1*, tip radius *r2*."""
    q  = vec2(length(p[..., [0, 2]]), p[..., 1])
    k1 = vec2(r2, h)
    k2 = vec2(r2 - r1, 2.0 * h)
    ca = vec2(
        q[..., 0] - np.minimum(q[..., 0], np.where(q[..., 1] < 0.0, r1, r2)),
        np.abs(q[..., 1]) - h,
    )
    cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot2(k2), 0.0, 1.0)[..., None]
    s  = np.where((cb[..., 0] < 0.0) & (ca[..., 1] < 0.0), -1.0, 1.0)
    return s * np.sqrt(np.minimum(dot2(ca), dot2(cb)))


def sdCappedConeSegment(p: _F, a: _F, b: _F, ra: float, rb: float) -> _F:
    """Capped cone from *a* to *b* with radii *ra* (at *a*) and *rb* (at *b*)."""
    rba  = rb - ra
    baba = dot2(b - a)
    papa = dot2(p - a)
    paba = dot(p - a, b - a) / baba
    x    = np.sqrt(papa - paba * paba * baba)
    cax  = np.maximum(0.0, x - np.where(paba < 0.5, ra, rb))
    cay  = np.abs(paba - 0.5) - 0.5
    k    = rba * rba + baba
    f    = clamp((rba * (x - ra) + paba * baba) / k, 0.0, 1.0)
    cbx  = x - ra - f * rba
    cby  = paba - f
    s    = np.where((cbx < 0.0) & (cay < 0.0), -1.0, 1.0)
    return s * np.sqrt(
        np.minimum(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba)
    )


def sdSolidAngle(p: _F, c: _F, ra: float) -> _F:
    """Solid angle (partial sphere); *c* = ``(sin, cos)`` of the cap half-angle."""
    q = vec2(length(p[..., [0, 2]]), p[..., 1])
    l = length(q) - ra
    m = length(q - c * clamp(dot(q, c), 0.0, ra)[..., None])
    return np.maximum(l, m * np.sign(c[1] * q[..., 0] - c[0] * q[..., 1]))


def sdCutSphere(p: _F, r: float, h: float) -> _F:
    """Sphere of radius *r* with a planar cut at height *h*."""
    w = np.sqrt(r * r - h * h)
    q = vec2(length(p[..., [0, 2]]), p[..., 1])
    s = np.maximum(
        (h - r) * q[..., 0] * q[..., 0] + w * w * (h + r - 2.0 * q[..., 1]),
        h * q[..., 0] - w * q[..., 1],
    )
    return np.where(
        s < 0.0,
        length(q) - r,
        np.where(q[..., 0] < w, h - q[..., 1], length(q - vec2(w, h))),
    )


def sdCutHollowSphere(p: _F, r: float, h: float, t: float) -> _F:
    """Cut hollow sphere of radius *r*, cut at *h*, shell thickness *t*."""
    w = np.sqrt(r * r - h * h)
    q = vec2(length(p[..., [0, 2]]), p[..., 1])
    return np.where(
        h * q[..., 0] < w * q[..., 1],
        length(q - vec2(w, h)),
        np.abs(length(q) - r),
    ) - t


def sdDeathStar(p2: _F, ra: float, rb: float, d: float) -> _F:
    """Death Star: large sphere *ra* with spherical bite *rb* at distance *d*."""
    a    = (ra * ra - rb * rb + d * d) / (2.0 * d)
    b    = np.sqrt(np.maximum(ra * ra - a * a, 0.0))
    p    = vec2(p2[..., 0], length(p2[..., [1, 2]]))
    cond = p[..., 0] * b - p[..., 1] * a > d * np.maximum(b - p[..., 1], 0.0)
    return np.where(
        cond,
        length(p - vec2(a, b)),
        np.maximum(length(p) - ra, -(length(p - vec2(d, 0.0)) - rb)),
    )


def sdRoundCone(p: _F, r1: float, r2: float, h: float) -> _F:
    """Round cone along Y; *r1* is base radius, *r2* is tip, *h* is height."""
    b = (r1 - r2) / h
    a = np.sqrt(1.0 - b * b)
    q = vec2(length(p[..., [0, 2]]), p[..., 1])
    k = dot(q, vec2(-b, a))
    return np.where(
        k < 0.0,
        length(q) - r1,
        np.where(k > a * h, length(q - vec2(0.0, h)) - r2, dot(q, vec2(a, b)) - r1),
    )


def sdRoundConeSegment(p: _F, a: _F, b: _F, r1: float, r2: float) -> _F:
    """Round cone from *a* to *b* with radii *r1* and *r2*."""
    ba  = b - a
    l2  = dot2(ba)
    rr  = r1 - r2
    a2  = l2 - rr * rr
    il2 = 1.0 / l2
    pa  = p - a
    y   = dot(pa, ba)
    z   = y - l2
    x2  = dot2(pa * l2 - ba * y[..., None])
    y2  = y * y * l2
    z2  = z * z * l2
    k   = np.sign(rr) * rr * rr * x2
    c1  = np.sign(z) * a2 * z2 > k
    c2  = np.sign(y) * a2 * y2 < k
    o1  = np.sqrt(x2 + z2) * il2 - r2
    o2  = np.sqrt(x2 + y2) * il2 - r1
    o3  = (np.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1
    return np.where(c1, o1, np.where(c2, o2, o3))


def sdEllipsoid(p: _F, r: _F) -> _F:
    """Ellipsoid with semi-axes *r* ``(rx, ry, rz)``."""
    k0 = length(p / r)
    k1 = length(p / (r * r))
    return k0 * (k0 - 1.0) / np.where(k1 == 0.0, 1e-12, k1)


def sdVesicaSegment(p: _F, a: _F, b: _F, w: float) -> _F:
    """Vesica piscis along segment *a*→*b* with half-width *w*."""
    c   = (a + b) * 0.5
    l   = length(b - a)
    v   = (b - a) / l
    y   = dot(p - c, v)
    q   = vec2(length(p - c - v * y[..., None]), np.abs(y))
    r   = 0.5 * l
    d   = 0.5 * (r * r - w * w) / w
    cond = r * q[..., 0] < d * (q[..., 1] - r)
    h   = np.where(cond[..., None], vec3(0.0, r, 0.0), vec3(-d, 0.0, d + w))
    return length(q - h[..., :2]) - h[..., 2]


def sdRhombus(p: _F, la: float, lb: float, h: float, ra: float) -> _F:
    """Rhombus with half-extents *la*/*lb*, height *h*, and edge radius *ra*."""
    p  = np.abs(p)
    f  = clamp((la * p[..., 0] - lb * p[..., 2] + lb * lb) / (la * la + lb * lb), 0.0, 1.0)
    w  = p[..., [0, 2]] - vec2(la, lb) * vec2(f, 1.0 - f)
    q  = vec2(length(w) * np.sign(w[..., 0]) - ra, p[..., 1] - h)
    return np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0) + length(np.maximum(q, 0.0))


def sdOctahedronExact(p: _F, s: float) -> _F:
    """Exact signed octahedron with inradius *s*."""
    p = np.abs(p)
    m = p[..., 0] + p[..., 1] + p[..., 2] - s
    res = m * 0.57735027
    mask1 = 3.0 * p[..., 0] < m
    mask2 = (~mask1) & (3.0 * p[..., 1] < m)
    mask3 = (~mask1) & (~mask2) & (3.0 * p[..., 2] < m)
    q = np.zeros_like(p)
    q = np.where(mask1[..., None], p, q)
    q = np.where(mask2[..., None], p[..., [1, 2, 0]], q)
    q = np.where(mask3[..., None], p[..., [2, 0, 1]], q)
    k    = clamp(0.5 * (q[..., 2] - q[..., 1] + s), 0.0, s)
    dist = length(vec3(q[..., 0], q[..., 1] - s + k, q[..., 2] - k))
    return np.where(mask1 | mask2 | mask3, dist, res)


def sdOctahedronBound(p: _F, s: float) -> _F:
    """Bounding (over-estimate) octahedron with inradius *s*."""
    p = np.abs(p)
    return (p[..., 0] + p[..., 1] + p[..., 2] - s) * 0.57735027


def sdPyramid(p: _F, h: float) -> _F:
    """Square-base pyramid of half-base 0.5 and height *h*."""
    m2  = h * h + 0.25
    pxz = np.abs(p[..., [0, 2]])
    swap = pxz[..., 1] > pxz[..., 0]
    px  = np.where(swap, pxz[..., 1], pxz[..., 0])
    pz  = np.where(swap, pxz[..., 0], pxz[..., 1])
    pxz = vec2(px, pz) - 0.5
    q   = vec3(pxz[..., 1], h * p[..., 1] - 0.5 * pxz[..., 0], h * pxz[..., 0] + 0.5 * p[..., 1])
    s   = np.maximum(-q[..., 0], 0.0)
    t   = clamp((q[..., 1] - 0.5 * pxz[..., 1]) / (m2 + 0.25), 0.0, 1.0)
    a   = m2 * (q[..., 0] + s) * (q[..., 0] + s) + q[..., 1] * q[..., 1]
    b   = m2 * (q[..., 0] + 0.5 * t) * (q[..., 0] + 0.5 * t) + (q[..., 1] - m2 * t) * (q[..., 1] - m2 * t)
    d2  = np.where(
        np.minimum(q[..., 1], -q[..., 0] * m2 - q[..., 1] * 0.5) > 0.0,
        0.0, np.minimum(a, b),
    )
    return np.sqrt((d2 + q[..., 2] * q[..., 2]) / m2) * np.sign(np.maximum(q[..., 2], -p[..., 1]))


# ===========================================================================
# 3-D unsigned-distance helpers
# ===========================================================================

def udTriangle(p: _F, a: _F, b: _F, c: _F) -> _F:
    """Unsigned distance to a 3-D triangle *a*-*b*-*c*."""
    ba  = b - a;  pa = p - a
    cb  = c - b;  pb = p - b
    ac  = a - c;  pc = p - c
    nor = np.cross(ba, ac)
    cond = (
        np.sign(dot(np.cross(ba, nor), pa))
        + np.sign(dot(np.cross(cb, nor), pb))
        + np.sign(dot(np.cross(ac, nor), pc))
        < 2.0
    )
    d1 = dot2(ba * clamp(dot(ba, pa) / dot2(ba), 0.0, 1.0)[..., None] - pa)
    d2 = dot2(cb * clamp(dot(cb, pb) / dot2(cb), 0.0, 1.0)[..., None] - pb)
    d3 = dot2(ac * clamp(dot(ac, pc) / dot2(ac), 0.0, 1.0)[..., None] - pc)
    d       = np.minimum(np.minimum(d1, d2), d3)
    d_plane = dot(nor, pa) * dot(nor, pa) / dot2(nor)
    return np.sqrt(np.where(cond, d, d_plane))


def udQuad(p: _F, a: _F, b: _F, c: _F, d: _F) -> _F:
    """Unsigned distance to a 3-D quad *a*-*b*-*c*-*d*."""
    ba = b - a;  pa = p - a
    cb = c - b;  pb = p - b
    dc = d - c;  pc = p - c
    ad = a - d;  pd = p - d
    nor = np.cross(ba, ad)
    cond = (
        np.sign(dot(np.cross(ba, nor), pa))
        + np.sign(dot(np.cross(cb, nor), pb))
        + np.sign(dot(np.cross(dc, nor), pc))
        + np.sign(dot(np.cross(ad, nor), pd))
        < 3.0
    )
    d1 = dot2(ba * clamp(dot(ba, pa) / dot2(ba), 0.0, 1.0)[..., None] - pa)
    d2 = dot2(cb * clamp(dot(cb, pb) / dot2(cb), 0.0, 1.0)[..., None] - pb)
    d3 = dot2(dc * clamp(dot(dc, pc) / dot2(dc), 0.0, 1.0)[..., None] - pc)
    d4 = dot2(ad * clamp(dot(ad, pd) / dot2(ad), 0.0, 1.0)[..., None] - pd)
    d_min   = np.minimum(np.minimum(np.minimum(d1, d2), d3), d4)
    d_plane = dot(nor, pa) * dot(nor, pa) / dot2(nor)
    return np.sqrt(np.where(cond, d_min, d_plane))


# ===========================================================================
# 2-D primitive SDFs
# ===========================================================================

def sdCircle(p: _F, r: float) -> _F:
    """2-D circle of radius *r* centred at origin."""
    return length(p) - r


def sdBox2D(p: _F, b: _F) -> _F:
    """2-D axis-aligned box with half-extents *b* ``(bx, by)``."""
    d = np.abs(p) - b
    return length(np.maximum(d, 0.0)) + np.minimum(np.maximum(d[..., 0], d[..., 1]), 0.0)


def sdRoundedBox2D(p: _F, b: _F, r: float) -> _F:
    """2-D rounded box with half-extents *b* and corner radius *r*."""
    d = np.abs(p) - b + r
    return length(np.maximum(d, 0.0)) + np.minimum(np.maximum(d[..., 0], d[..., 1]), 0.0) - r


def sdOrientedBox2D(p: _F, a: _F, b: _F, th: float) -> _F:
    """2-D oriented box from *a* to *b* with half-thickness *th*."""
    l = length(b - a)
    d = (b - a) / l
    q = p - (a + b) * 0.5
    q = vec2(dot(q, d), np.abs(dot(q, vec2(-d[1], d[0]))))
    q = np.abs(q) - vec2(l * 0.5, th)
    return length(np.maximum(q, 0.0)) + np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0)


def sdSegment2D(p: _F, a: _F, b: _F) -> _F:
    """2-D line segment from *a* to *b* (zero-width)."""
    pa = p - a
    ba = b - a
    h  = clamp(dot(pa, ba) / dot2(ba), 0.0, 1.0)
    return length(pa - ba * h[..., None])


def sdRhombus2D(p: _F, b: _F) -> _F:
    """2-D rhombus with half-extents *b*."""
    p = np.abs(p)
    h = clamp((-2.0 * dot2(p) + dot2(b)) / dot2(b), -1.0, 1.0)
    d = length(p - 0.5 * b * vec2(1.0 - h, 1.0 + h))
    return d * np.sign(p[..., 0] * b[1] + p[..., 1] * b[0] - b[0] * b[1])


def sdTrapezoid2D(p: _F, r1: float, r2: float, he: float) -> _F:
    """2-D isosceles trapezoid with base radii *r1*/*r2* and height *he*."""
    k1 = vec2(r2, he)
    k2 = vec2(r2 - r1, 2.0 * he)
    px = np.abs(p[..., 0])
    py = p[..., 1]
    ca = vec2(np.maximum(0.0, px - np.where(py < 0.0, r1, r2)), np.abs(py) - he)
    cb = p - k1 + k2 * clamp(dot(k1 - vec2(px, py), k2) / dot2(k2), 0.0, 1.0)[..., None]
    s  = np.where((cb[..., 0] < 0.0) & (ca[..., 1] < 0.0), -1.0, 1.0)
    return s * np.sqrt(np.minimum(dot2(ca), dot2(cb)))


def sdParallelogram2D(p: _F, wi: float, he: float, sk: float) -> _F:
    """2-D parallelogram with half-width *wi*, half-height *he*, x-skew *sk*.

    Vertices: ``(-wi,-he)``, ``(wi,-he)``, ``(wi+sk,he)``, ``(-wi+sk,he)``.
    The skew shifts the top edge rightward by *sk* relative to the bottom edge.
    """
    v = np.array([[-wi, -he], [wi, -he], [wi + sk, he], [-wi + sk, he]], dtype=float)
    return sdPolygon2D(p, v)


def sdEquilateralTriangle2D(p: _F, r: float) -> _F:
    """2-D equilateral triangle with circumradius *r*."""
    k  = np.array([np.sqrt(3.0), -1.0])
    px = np.abs(p[..., 0]) - r
    py = p[..., 1] + r / np.sqrt(3.0)
    px = px - 2.0 * np.minimum(0.0, k[0] * px + k[1] * py) * k[0]
    py = py - 2.0 * np.minimum(0.0, k[0] * px + k[1] * py) * k[1]
    px = px - clamp(px, -2.0 * r, 0.0)
    return -length(vec2(px, py)) * np.sign(py)


def sdTriangleIsosceles2D(p: _F, q: _F) -> _F:
    """2-D isosceles triangle; *q* = ``(half_base, height)``."""
    px = np.abs(p[..., 0])
    py = p[..., 1]
    a  = px - q[0] * clamp(safe_div(px, q[0]), 0.0, 1.0)
    b  = vec2(px - q[0], np.abs(py) - q[1])
    d  = np.minimum(dot2(vec2(a, py + q[1])), dot2(b))
    return -np.sqrt(d) * np.sign(px * q[1] + py * q[0] - q[0] * q[1])


def sdTriangle2D(p: _F, p0: _F, p1: _F, p2: _F) -> _F:
    """2-D triangle from three vertices *p0*, *p1*, *p2*."""
    e0  = p1 - p0;  v0 = p - p0
    e1  = p2 - p1;  v1 = p - p1
    e2  = p0 - p2;  v2 = p - p2
    pq0 = v0 - e0 * clamp(dot(v0, e0) / dot2(e0), 0.0, 1.0)[..., None]
    pq1 = v1 - e1 * clamp(dot(v1, e1) / dot2(e1), 0.0, 1.0)[..., None]
    pq2 = v2 - e2 * clamp(dot(v2, e2) / dot2(e2), 0.0, 1.0)[..., None]
    s   = np.sign(e0[0] * e2[1] - e0[1] * e2[0])
    d   = np.minimum(np.minimum(
        vec2(dot2(pq0), s * (v0[..., 0] * e0[1] - v0[..., 1] * e0[0])),
        vec2(dot2(pq1), s * (v1[..., 0] * e1[1] - v1[..., 1] * e1[0]))),
        vec2(dot2(pq2), s * (v2[..., 0] * e2[1] - v2[..., 1] * e2[0])))
    return -np.sqrt(d[..., 0]) * np.sign(d[..., 1])


def sdUnevenCapsule2D(p: _F, r1: float, r2: float, h: float) -> _F:
    """2-D capsule with radii *r1* (bottom) and *r2* (top), height *h*."""
    px   = np.abs(p[..., 0])
    py   = p[..., 1]
    b    = (r1 - r2) / h
    a    = np.sqrt(1.0 - b * b)
    k    = dot(vec2(px, py), vec2(-b, a))
    c1   = k < 0.0
    c2   = k > a * h
    return np.where(c1, length(vec2(px, py)) - r1,
           np.where(c2, length(vec2(px, py - h)) - r2,
                    dot(vec2(px, py), vec2(a, b)) - r1))


def sdPentagon2D(p: _F, r: float) -> _F:
    """2-D regular pentagon with circumradius *r*."""
    k  = np.array([0.809016994, 0.587785252, 0.726542528])
    px = np.abs(p[..., 0])
    py = p[..., 1]
    # Each fold step updates both components simultaneously (mirrors GLSL compound assignment)
    d1 = 2.0 * np.minimum(dot(vec2(px, py), vec2(-k[0], k[1])), 0.0)
    px = px - d1 * (-k[0]);  py = py - d1 * k[1]
    d2 = 2.0 * np.minimum(dot(vec2(px, py), vec2(k[0], k[1])), 0.0)
    px = px - d2 * k[0];     py = py - d2 * k[1]
    px = px - clamp(px, -r * k[2], r * k[2])
    py = py - r
    return length(vec2(px, py)) * np.sign(py)


def sdHexagon2D(p: _F, r: float) -> _F:
    """2-D regular hexagon with inradius *r* (distance from centre to a flat face)."""
    k  = np.array([-0.866025404, 0.5, 0.577350269])
    px = np.abs(p[..., 0])
    py = np.abs(p[..., 1])
    # Fold step: compute dot product once, apply to both components simultaneously
    d  = 2.0 * np.minimum(dot(vec2(px, py), k[:2]), 0.0)
    px = px - d * k[0];  py = py - d * k[1]
    px = px - clamp(px, -k[2] * r, k[2] * r)
    py = py - r
    return length(vec2(px, py)) * np.sign(py)


def sdOctagon2D(p: _F, r: float) -> _F:
    """2-D regular octagon with inradius *r*."""
    k  = np.array([-0.9238795325, 0.3826834323, 0.4142135623])
    px = np.abs(p[..., 0])
    py = np.abs(p[..., 1])
    d  = 2.0 * np.minimum(dot(vec2(px, py), k[:2]), 0.0)
    px = px - d * k[0];  py = py - d * k[1]
    px = px - clamp(px, -k[2] * r, k[2] * r)
    py = py - r
    return length(vec2(px, py)) * np.sign(py)


def sdHexagram2D(p: _F, r: float) -> _F:
    """2-D hexagram (6-pointed star) with circumradius *r*."""
    k  = np.array([-0.5, 0.8660254038, 0.5773502692, 1.7320508076])
    px = np.abs(p[..., 0])
    py = np.abs(p[..., 1])
    d  = 2.0 * np.minimum(dot(vec2(px, py), k[:2]), 0.0)
    px = px - d * k[0];  py = py - d * k[1]
    px = px - clamp(px, r * k[2], r * k[3])
    py = py - r
    return length(vec2(px, py)) * np.sign(py)


def sdStar5(p: _F, r: float, rf: float) -> _F:
    """2-D 5-pointed star; *r* outer radius, *rf* inner factor (0–1)."""
    k1 = np.array([0.809016994375, -0.587785252292])
    k2 = np.array([-k1[0], k1[1]])
    px = p[..., 0];  py = p[..., 1]
    # Both fold steps update components simultaneously
    d1 = 2.0 * np.maximum(dot(vec2(px, py), k1), 0.0)
    px = px - d1 * k1[0];  py = py - d1 * k1[1]
    d2 = 2.0 * np.maximum(dot(vec2(px, py), k2), 0.0)
    px = px - d2 * k2[0];  py = py - d2 * k2[1]
    # Rotation (simultaneous)
    new_px = px * k1[0] + py * k1[1]
    py     = py * k1[0] - px * k1[1]
    px     = np.abs(new_px);  py = py - r
    ba = rf * vec2(-k1[1], k1[0]) - vec2(0.0, 1.0)
    h  = clamp(dot(vec2(px, py), ba) / dot2(ba), 0.0, r)
    return length(vec2(px - ba[0] * h, py - ba[1] * h)) * np.sign(py * ba[0] - px * ba[1])


def sdStar(p: _F, r: float, n: int, m: float) -> _F:
    """2-D N-pointed star; *r* radius, *n* points, *m* inner factor."""
    an  = np.pi / n
    en  = np.pi / m
    acs = vec2(np.cos(an), np.sin(an))
    ecs = vec2(np.cos(en), np.sin(en))
    bn  = np.arctan2(np.abs(p[..., 0]), p[..., 1]) % (2.0 * an) - an
    px  = length(p) * np.cos(bn)
    py  = length(p) * np.abs(np.sin(bn))
    px  = px - r * acs[0];  py = py - r * acs[1]
    px  = px + ecs[1] * clamp(-dot(vec2(px, py), ecs), 0.0, r * acs[1] / ecs[1])
    py  = py + ecs[0] * clamp(-dot(vec2(px, py), ecs), 0.0, r * acs[1] / ecs[1])
    return length(vec2(px, py)) * np.sign(px)


def sdPie2D(p: _F, c: _F, r: float) -> _F:
    """2-D pie sector; *c* = ``(sin, cos)`` of half-angle, *r* radius."""
    px = np.abs(p[..., 0]);  py = p[..., 1]
    l  = length(p) - r
    d_proj = clamp(dot(vec2(px, py), c), 0.0, r)
    m  = length(vec2(px - c[0] * d_proj, py - c[1] * d_proj))
    return np.maximum(l, m * np.sign(c[1] * px - c[0] * py))


def sdCutDisk2D(p: _F, r: float, h: float) -> _F:
    """2-D circle of radius *r* with planar cut at height *h*."""
    w    = np.sqrt(r * r - h * h)
    px   = np.abs(p[..., 0]);  py = p[..., 1]
    s    = np.maximum((h - r) * px * px + w * w * (h + r - 2.0 * py), h * px - w * py)
    c1   = s < 0.0;  c2 = px < w
    return np.where(c1, length(p) - r,
           np.where(c2, h - py, length(vec2(px, py) - vec2(w, h))))


def sdArc2D(p: _F, sc: _F, ra: float, rb: float) -> _F:
    """2-D arc; *sc* = ``(sin, cos)`` of half-angle, *ra* radius, *rb* thickness."""
    px   = np.abs(p[..., 0]);  py = p[..., 1]
    cond = sc[1] * px > sc[0] * py
    return np.where(cond,
                    length(vec2(px, py) - sc * ra) - rb,
                    np.abs(length(p) - ra) - rb)


def sdRing2D(p: _F, r1: float, r2: float) -> _F:
    """2-D ring (annulus) with inner radius *r1* and outer radius *r2*."""
    l = length(p)
    return np.maximum(r1 - l, l - r2)


def sdHorseshoe2D(p: _F, c: _F, r: float, w: _F) -> _F:
    """2-D horseshoe; *c* = ``(sin, cos)`` of gap half-angle, *r* radius, *w* arm widths."""
    px = np.abs(p[..., 0]);  py = p[..., 1]
    l  = length(p)
    px = np.where(py > 0.0, px, l) * np.sign(-c[0])
    py = np.where(py > 0.0, py, 0.0)
    px = px - c[0] * r;  py = py - c[1] * r
    q  = vec2(length(vec2(np.maximum(px, 0.0), py)),
              np.where(px < 0.0, py, length(vec2(px, py))))
    d  = vec2(q[..., 0] - w[0], q[..., 1] - w[1])
    return np.minimum(np.maximum(d[..., 0], d[..., 1]), 0.0) + length(np.maximum(d, 0.0))


def sdVesica2D(p: _F, r: float, d: float) -> _F:
    """2-D vesica piscis; *r* radius, *d* half-distance between circle centres."""
    px = np.abs(p[..., 0]);  py = p[..., 1]
    b  = np.sqrt(r * r - d * d)
    c  = (py - b) * d > px * b
    return np.where(c,
                    length(vec2(px, py) - vec2(0.0, b)) * np.sign(d),
                    length(vec2(px, py) - vec2(-d, 0.0)) - r)


def sdMoon2D(p: _F, d: float, ra: float, rb: float) -> _F:
    """2-D crescent moon; *d* offset, *ra* outer radius, *rb* inner radius."""
    py = np.abs(p[..., 1])
    a  = (ra * ra - rb * rb + d * d) / (2.0 * d)
    b  = np.sqrt(np.maximum(ra * ra - a * a, 0.0))
    c  = d * (p[..., 0] * b - py * a) > d * d * np.maximum(b - py, 0.0)
    return np.where(c,
                    length(vec2(p[..., 0], py) - vec2(a, b)),
                    np.maximum(length(p) - ra, -(length(vec2(p[..., 0] - d, py)) - rb)))


def sdRoundedCross2D(p: _F, h: float) -> _F:
    """2-D rounded cross of size *h*."""
    k  = 0.5 * (h + 1.0 / h)
    px = np.abs(p[..., 0]);  py = np.abs(p[..., 1])
    c  = px < 1.0
    qx = np.where(c, 1.0, px)
    qy = np.where(c, k - py, py - k)
    r1 = length(vec2(qx - 1.0, qy)) * np.sign(qy)
    r2 = length(vec2(px - k, py)) * np.sign(px - k)
    return np.minimum(r1, r2)


def sdEgg2D(p: _F, ra: float, rb: float) -> _F:
    """2-D egg; *ra* large radius, *rb* small radius."""
    k  = np.sqrt(3.0)
    px = np.abs(p[..., 0]);  py = p[..., 1]
    r  = ra - rb
    c  = py < 0.0
    return np.where(c, length(vec2(px, py)) - r,
           np.where(k * (px + r) < py, length(vec2(px, py - k * r)),
                    length(vec2(px + r, py)) - 2.0 * r) - rb)


def sdHeart2D(p: _F) -> _F:
    """2-D heart shape (unit-scale)."""
    px = np.abs(p[..., 0]);  py = p[..., 1]
    c  = px + py > 1.0
    qx = np.where(c, px - 0.25, px)
    qy = np.where(c, py - 0.75, py)
    return length(vec2(qx, qy)) - np.where(c, np.sqrt(2.0) / 4.0, 1.0)


def sdCross2D(p: _F, b: _F, r: float) -> _F:
    """2-D plus-sign cross; *b* = ``(half_arm_len, half_arm_width)``, *r* rounding."""
    px = np.abs(p[..., 0]);  py = np.abs(p[..., 1])
    c  = px > py
    px_n = np.where(c, px, py);  py_n = np.where(c, py, px)
    q  = vec2(px_n - b[0], py_n - b[1])
    k  = np.maximum(q[..., 1], q[..., 0])
    w  = np.where((k > 0.0)[..., np.newaxis], q, vec2(b[1] - px_n, -k))
    return np.sign(k) * length(np.maximum(w, 0.0)) + r


def sdRoundedX2D(p: _F, w: float, r: float) -> _F:
    """2-D rounded X (cross at 45°); *w* width, *r* rounding."""
    px = np.abs(p[..., 0]);  py = np.abs(p[..., 1])
    q  = (px + py - w) * 0.5
    return length(vec2(px - q, py - q)) - r


def sdPolygon2D(p: _F, v: _F) -> _F:
    """2-D polygon from *N* vertices *v* (shape ``(N, 2)``)."""
    N = v.shape[0]
    d = dot2(p - v[0])
    s = 1.0
    for i in range(N):
        j = (i + 1) % N
        e = v[j] - v[i]
        w = p - v[i]
        b = w - e * clamp(dot(w, e) / dot2(e), 0.0, 1.0)[..., None]
        d = np.minimum(d, dot2(b))
        cond = np.array([
            p[..., 1] >= v[i][1],
            p[..., 1] < v[j][1],
            e[0] * w[..., 1] > e[1] * w[..., 0],
        ])
        s = np.where(np.all(cond, axis=0) | np.all(~cond, axis=0), -s, s)
    return s * np.sqrt(d)


def sdEllipse2D(p: _F, ab: _F) -> _F:
    """2-D ellipse with semi-axes *ab* = ``(a, b)``."""
    px   = np.abs(p[..., 0]);  py = np.abs(p[..., 1])
    c    = px > py
    px_n = np.where(c, px, py);  py_n = np.where(c, py, px)
    # Swap semi-axes per point: a0 follows the dominant axis, a1 the minor
    a0   = np.where(c, ab[0], ab[1]);  a1 = np.where(c, ab[1], ab[0])
    l    = a1 * a1 - a0 * a0
    m    = a0 * px_n / l;  n_ = a1 * py_n / l
    m2   = m * m;  n2 = n_ * n_
    c3   = (m2 + n2 - 1.0) / 3.0
    c3c  = c3 * c3 * c3
    d    = c3c + m2 * n2
    q    = d + m2 * n2
    g    = m + m * n2
    # Guard branches against invalid inputs to avoid NaN in the unused branch
    co   = np.where(
        d < 0.0,
        (1.0 / 3.0) * np.arccos(np.clip(safe_div(q, np.power(np.maximum(np.abs(c3c), 1e-30), 0.5)), -1.0, 1.0)) - np.pi / 3.0,
        (1.0 / 3.0) * np.log(safe_div(np.sqrt(np.maximum(q, 0.0)) + np.sqrt(np.maximum(d, 0.0)), np.maximum(g, 1e-30))),
    )
    rx   = np.where(c, a0, a1) * np.cos(co)
    ry   = np.where(c, a1, a0) * np.sin(co)
    return length(vec2(px_n - rx, py_n - ry)) * np.sign(py_n - ry)


def sdParabola2D(p: _F, k: float) -> _F:
    """2-D parabola ``y = k·x²``; *k* is the curvature."""
    px = np.abs(p[..., 0]);  py = p[..., 1]
    ik = 1.0 / k
    p2 = ik * (py - 0.5 * ik) / 3.0
    q  = px / (k * k)
    h  = q * q - p2 * p2 * p2
    r  = np.sqrt(np.abs(h))
    x  = np.where(h > 0.0,
                  np.power(q + r, 1.0 / 3.0) - np.power(np.abs(q - r), 1.0 / 3.0) * np.sign(r - q),
                  2.0 * np.cos(np.arctan2(r, q) / 3.0) * np.sqrt(p2))
    return length(vec2(px - x * x, py - x)) * np.sign(py - x)


def sdParabolaSegment2D(p: _F, wi: float, he: float) -> _F:
    """2-D bounded parabola segment; *wi* half-width, *he* height."""
    px = np.abs(p[..., 0]);  py = p[..., 1]
    ik = wi * wi / he
    p2 = ik * (he - py - 0.5 * ik) / 3.0
    q  = px * ik * ik * 0.25
    h  = q * q - p2 * p2 * p2
    r  = np.sqrt(np.abs(h))
    x  = np.where(h > 0.0,
                  np.power(q + r, 1.0 / 3.0) - np.power(np.abs(q - r), 1.0 / 3.0) * np.sign(r - q),
                  2.0 * np.cos(np.arctan2(r, q) / 3.0) * np.sqrt(p2))
    x  = np.minimum(x, wi)
    return length(vec2(px - x, py - he + x * x * he / (wi * wi))) * np.sign(ik * (py - he) + px * px)


def sdBezier2D(p: _F, A: _F, B: _F, C: _F) -> _F:
    """2-D quadratic Bézier curve with control points *A*, *B* (ctrl), *C*."""
    a   = B - A;  b = A - 2.0 * B + C;  c = a * 2.0;  d = A - p
    kk  = 1.0 / dot2(b)
    kx  = kk * dot(a, b)
    ky  = kk * (2.0 * dot2(a) + dot(d, b)) / 3.0
    kz  = kk * dot(d, a)
    p1  = ky - kx * kx
    p3  = p1 * p1 * p1
    q2  = kx * (2.0 * kx * kx - 3.0 * ky) + kz
    h   = q2 * q2 + 4.0 * p3
    h_p = h >= 0.0
    z   = np.where(h_p[..., None], np.sqrt(h[..., None]), np.array([0.0, 0.0]))
    v   = np.sign(q2 + h_p * z[..., 0]) * np.power(np.abs(q2 + h_p * z[..., 0]), 1.0 / 3.0)
    u   = np.sign(q2 - h_p * z[..., 0]) * np.power(np.abs(q2 - h_p * z[..., 0]), 1.0 / 3.0)
    t   = clamp(np.where(h_p, (v + u) - kx,
                         2.0 * np.cos(np.arctan2(np.sqrt(-h), q2) / 3.0) * np.sqrt(-p1) - kx),
                0.0, 1.0)
    q3  = d + (c + b * t[..., None]) * t[..., None]
    return length(q3)


def sdBlobbyCross2D(p: _F, he: float) -> _F:
    """2-D blobby cross of size *he*."""
    px = np.abs(p[..., 0]);  py = np.abs(p[..., 1])
    px = np.where(py > px, py, px);  py = np.where(py > px, px, py)
    a  = px - py
    b  = px + py - 2.0 * he
    c1 = a * a;  c2 = b * b + 4.0 * he * he
    d  = np.where(a > 0.0, c1, c2)
    return 0.5 * (np.sqrt(d) - he)


def sdTunnel2D(p: _F, wh: _F) -> _F:
    """2-D tunnel/arch; *wh* = ``(half_width, height)``."""
    px = np.abs(p[..., 0]);  py = p[..., 1]
    px = px - wh[0]
    q  = vec2(px, np.maximum(np.abs(py) - wh[1], 0.0))
    return length(q) - 0.5 * wh[0]


def sdStairs2D(p: _F, wh: _F, n: int) -> _F:
    """2-D staircase; *wh* = ``(step_width, step_height)``, *n* steps."""
    ba  = wh * n
    d   = np.minimum(
        dot2(p - vec2(clamp(p[..., 0], 0.0, ba[0]), clamp(p[..., 1], 0.0, ba[1]))),
        dot2(p - vec2(np.minimum(p[..., 0], ba[0]), np.minimum(p[..., 1], ba[1]))),
    )
    s    = np.sign(np.maximum(-p[..., 1], p[..., 0] - ba[0]))
    dia  = length(wh)
    px_m = p[..., 0] - wh[0] * clamp(np.round(p[..., 0] / wh[0]), 0.0, n)
    py_m = p[..., 1] - wh[1] * clamp(np.round(p[..., 1] / wh[1]), 0.0, n)
    d    = np.minimum(d, dot2(vec2(px_m, py_m) - 0.5 * wh) - 0.25 * dia * dia)
    return np.sqrt(np.maximum(d, 0.0)) * s


def sdQuadraticCircle2D(p: _F) -> _F:
    """2-D quadratic-circle approximation (unit-scale)."""
    px = np.abs(p[..., 0]);  py = np.abs(p[..., 1])
    c  = py > px
    px_n = np.where(c, py, px);  py_n = np.where(c, px, py)
    a  = px_n - py_n;  b = px_n + py_n
    c3 = (2.0 * b - 1.0) / 3.0
    h  = a * a + c3 * c3 * c3
    c3_safe = np.maximum(c3, 1e-30)
    t  = np.where(h >= 0.0,
                  a + np.sign(a) * np.power(np.maximum(h, 0.0), 1.0 / 3.0),
                  a + 2.0 * c3 * np.cos(np.arccos(np.clip(a / (c3_safe * np.sqrt(c3_safe)), -1.0, 1.0)) / 3.0))
    t  = np.minimum(t, 1.0)
    d  = length(vec2(px_n - t, py_n - t * t))
    return d * np.sign(b - 1.0 - t * t)


def sdHyperbola2D(p: _F, k: float, he: float) -> _F:
    """2-D hyperbola; *k* curvature, *he* half-height."""
    px = np.abs(p[..., 0]);  py = np.abs(p[..., 1])
    kv = vec2(k, 1.0)
    kd = dot2(kv)
    px = px - 2.0 * np.minimum(dot(vec2(px, py), kv), 0.0) * k / kd
    py = py - 2.0 * np.minimum(dot(vec2(px, py), kv), 0.0) / kd
    x2 = px * px / 16.0;  y2 = py * py / 16.0
    r  = dot2(vec2(px * py, he * he * (1.0 - 2.0 * k) * px - k * py * py))
    q  = (x2 - y2) * (x2 - y2)
    q  = np.where(r != 0.0, (3.0 * y2 - x2) * x2 * x2 + r, q)
    return (
        length(vec2(px, py - he)) * np.sign(py - he)
        + np.sqrt(np.abs(q)) * np.sign(r) * 0.0625
    )


def sdNGon2D(p: _F, r: float, n: int) -> _F:
    """2-D regular N-gon; *r* circumradius, *n* sides."""
    an  = np.pi / n
    acs = vec2(np.cos(an), np.sin(an))
    bn  = np.arctan2(np.abs(p[..., 0]), p[..., 1]) % (2.0 * an) - an
    px  = length(p) * np.cos(bn);  py = length(p) * np.abs(np.sin(bn))
    px  = px - r * acs[0];  py = py - r * acs[1]
    return length(np.maximum(vec2(px, py), 0.0)) + np.minimum(np.maximum(px, py), 0.0)


# ===========================================================================
# Boolean / domain operators
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


def opXor(d1: _F, d2: _F) -> _F:
    """Exclusive-or of two SDFs."""
    return np.maximum(np.minimum(d1, d2), -np.maximum(d1, d2))


def opSmoothUnion(d1: _F, d2: _F, k: float) -> _F:
    """Smooth union with smoothing factor *k*."""
    k = k * 4.0
    h = np.maximum(k - np.abs(d1 - d2), 0.0)
    return np.minimum(d1, d2) - h * h * 0.25 / k


def opSmoothSubtraction(d1: _F, d2: _F, k: float) -> _F:
    """Smooth subtraction with smoothing factor *k*."""
    return -opSmoothUnion(d1, -d2, k)


def opSmoothIntersection(d1: _F, d2: _F, k: float) -> _F:
    """Smooth intersection with smoothing factor *k*."""
    return -opSmoothUnion(-d1, -d2, k)


# ===========================================================================
# Space-warp operators
# ===========================================================================

def opRevolution(p: _F, primitive2d: "_SDFFunc", o: float) -> _F:  # type: ignore[name-defined]
    """Revolve a 2-D primitive around the Y axis with offset *o*."""
    q = vec2(length(p[..., [0, 2]]) - o, p[..., 1])
    return primitive2d(q)


def opExtrusion(p: _F, primitive2d: "_SDFFunc", h: float) -> _F:  # type: ignore[name-defined]
    """Extrude a 2-D primitive along Z to half-height *h*."""
    d = primitive2d(p[..., :2])
    w = vec2(d, np.abs(p[..., 2]) - h)
    return np.minimum(np.maximum(w[..., 0], w[..., 1]), 0.0) + length(np.maximum(w, 0.0))


def opElongate1(p: _F, primitive3d: "_SDFFunc", h: _F) -> _F:  # type: ignore[name-defined]
    """Elongate by clamping *p* to ``[-h, h]`` (type 1)."""
    q = p - clamp(p, -h, h)
    return primitive3d(q)


def opElongate2(p: _F, primitive3d: "_SDFFunc", h: _F) -> _F:  # type: ignore[name-defined]
    """Elongate by folding *p* beyond ``[-h, h]`` (type 2, exact)."""
    q = np.abs(p) - h
    return primitive3d(np.maximum(q, 0.0)) + np.minimum(np.max(q, axis=-1), 0.0)


def opRound(p: _F, primitive3d: "_SDFFunc", rad: float) -> _F:  # type: ignore[name-defined]
    """Round a primitive outward by *rad*."""
    return primitive3d(p) - rad


def opOnion(sdf_val: _F, thickness: float) -> _F:
    """Turn a solid into a shell of *thickness*."""
    return np.abs(sdf_val) - thickness


def opScale(p: _F, s: float, primitive3d: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Uniformly scale a primitive by factor *s*."""
    return primitive3d(p / s) * s


def opSymX(p: _F, primitive3d: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Mirror the primitive in the YZ plane."""
    p = p.copy()
    p[..., 0] = np.abs(p[..., 0])
    return primitive3d(p)


def opSymXZ(p: _F, primitive3d: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Mirror the primitive in both YZ and XY planes."""
    p = p.copy()
    p[..., 0] = np.abs(p[..., 0])
    p[..., 2] = np.abs(p[..., 2])
    return primitive3d(p)


def opRepetition(p: _F, s: _F, primitive3d: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Tile the primitive infinitely with cell size *s*."""
    q = p - s * np.round(p / s)
    return primitive3d(q)


def opLimitedRepetition(p: _F, s: _F, l: _F, primitive3d: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Tile the primitive with cell size *s*, limited to ``[-l, l]`` repetitions."""
    q = p - s * clamp(np.round(p / s), -l, l)
    return primitive3d(q)


def opDisplace(p: _F, primitive3d: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Add sinusoidal displacement noise to the primitive."""
    d1 = primitive3d(p)
    d2 = np.sin(20.0 * p[..., 0]) * np.sin(20.0 * p[..., 1]) * np.sin(20.0 * p[..., 2])
    return d1 + d2


def opTwist(p: _F, primitive3d: "_SDFFunc", k: float) -> _F:  # type: ignore[name-defined]
    """Twist the primitive around Y with frequency *k*."""
    c = np.cos(k * p[..., 1]);  s = np.sin(k * p[..., 1])
    x = c * p[..., 0] - s * p[..., 2]
    z = s * p[..., 0] + c * p[..., 2]
    return primitive3d(vec3(x, p[..., 1], z))


def opCheapBend(p: _F, primitive3d: "_SDFFunc", k: float) -> _F:  # type: ignore[name-defined]
    """Bend the primitive around the Y axis with frequency *k*."""
    c = np.cos(k * p[..., 0]);  s = np.sin(k * p[..., 0])
    x = c * p[..., 0] - s * p[..., 1]
    y = s * p[..., 0] + c * p[..., 1]
    return primitive3d(vec3(x, y, p[..., 2]))


def opTx(p: _F, rot: _F, trans: _F, primitive3d: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Apply rotation *rot* and translation *trans* to the primitive.

    *rot* is a ``(3, 3)`` rotation matrix; the inverse (transpose) is applied
    to transform the query point into the primitive's local frame.
    """
    inv_rot = rot.T
    q = (p - trans) @ inv_rot
    return primitive3d(q)


def opTx2D(p: _F, mat: _F, trans: _F, sdf_func: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Apply 2-D rotation *mat* and translation *trans* to *sdf_func*."""
    p_transformed = np.dot(p, mat.T) - trans
    return sdf_func(p_transformed)
