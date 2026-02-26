"""2-D SDF math primitives for the sdf2d package.

Re-exports all shared helpers from :mod:`_sdf_common`, then adds every
2-D primitive SDF and the 2-D transform operator.

All functions accept and return ``numpy.ndarray`` objects and support
broadcasting over arbitrary leading batch dimensions.  A "point array" *p*
has shape ``(..., 2)``; scalar SDF results have shape ``(...,)``.

Formulas are adapted from Inigo Quilez's distance function reference:
https://iquilezles.org/articles/distfunctions2d/
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from _sdf_common import *  # noqa: F401, F403  — re-export shared helpers
from _sdf_common import _F  # explicit import so _F is available for annotations

# ---------------------------------------------------------------------------
# Type alias (re-declared so type-checkers see it in this module's namespace)
# ---------------------------------------------------------------------------
_F = npt.NDArray[np.floating]


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


# Backward-compat alias for old misspelling
sdOctogon2D = sdOctagon2D


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
# 2-D transform operator
# ===========================================================================

def opTx2D(p: _F, mat: _F, trans: _F, sdf_func: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Apply 2-D rotation *mat* and translation *trans* to *sdf_func*."""
    p_transformed = np.dot(p, mat.T) - trans
    return sdf_func(p_transformed)
