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

from _sdf_common import *  # noqa: F401, F403  — re-export shared helpers


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
    px = np.abs(p[..., 0]);  py = np.abs(p[..., 1])
    # IQ: negate b.y, then h = clamp((dot(b,p) + b.y²) / dot(b,b), 0, 1)
    h  = clamp((b[0] * px - b[1] * py + b[1] * b[1]) / (b[0] * b[0] + b[1] * b[1]), 0.0, 1.0)
    px = px - b[0] * h
    py = py + b[1] * (h - 1.0)
    return length(vec2(px, py)) * np.sign(px)


def sdTrapezoid2D(p: _F, r1: float, r2: float, he: float) -> _F:
    """2-D isosceles trapezoid with base radii *r1*/*r2* and height *he*."""
    k1 = vec2(r2, he)
    k2 = vec2(r2 - r1, 2.0 * he)
    px = np.abs(p[..., 0])
    py = p[..., 1]
    pv = vec2(px, py)
    ca = vec2(np.maximum(0.0, px - np.where(py < 0.0, r1, r2)), np.abs(py) - he)
    cb = pv - k1 + k2 * clamp(dot(k1 - pv, k2) / dot2(k2), 0.0, 1.0)[..., None]
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
    k  = np.sqrt(3.0)
    px = np.abs(p[..., 0]) - r
    py = p[..., 1] + r / k
    # IQ: if p.x + k*p.y > 0: p = vec2(p.x-k*p.y, -k*p.x-p.y) / 2
    cond   = px + k * py > 0.0
    new_px = np.where(cond, (px - k * py) / 2.0, px)
    new_py = np.where(cond, (-k * px - py) / 2.0, py)
    new_px = new_px - clamp(new_px, -2.0 * r, 0.0)
    return -length(vec2(new_px, new_py)) * np.sign(new_py)


def sdTriangleIsosceles2D(p: _F, q: _F) -> _F:
    """2-D isosceles triangle; *q* = ``(half_base, height)``."""
    px = np.abs(p[..., 0]);  py = p[..., 1]
    pv = vec2(px, py)
    # a = p - q*clamp(dot(p,q)/dot(q,q), 0, 1)
    a  = pv - q * clamp(dot(pv, q) / dot2(q), 0.0, 1.0)[..., None]
    # b = p - q*vec2(clamp(p.x/q.x, 0, 1), 1)
    b  = pv - q * vec2(clamp(px / q[0], 0.0, 1.0), np.ones_like(px))
    s  = -np.sign(q[1])
    d  = np.minimum(
        vec2(dot2(a), s * (px * q[1] - py * q[0])),
        vec2(dot2(b), s * (py - q[1])),
    )
    return -np.sqrt(d[..., 0]) * np.sign(d[..., 1])


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
    # Two fold steps: k.xy then k.yx
    d1 = 2.0 * np.minimum(dot(vec2(px, py), k[:2]), 0.0)
    px = px - d1 * k[0];  py = py - d1 * k[1]
    d2 = 2.0 * np.minimum(dot(vec2(px, py), vec2(k[1], k[0])), 0.0)
    px = px - d2 * k[1];  py = py - d2 * k[0]
    px = px - clamp(px, -k[2] * r, k[2] * r)
    py = py - r
    return length(vec2(px, py)) * np.sign(py)


def sdHexagram2D(p: _F, r: float) -> _F:
    """2-D hexagram (6-pointed star) with circumradius *r*."""
    k  = np.array([-0.5, 0.8660254038, 0.5773502692, 1.7320508076])
    px = np.abs(p[..., 0])
    py = np.abs(p[..., 1])
    # Two fold steps: k.xy then k.yx
    d1 = 2.0 * np.minimum(dot(vec2(px, py), k[:2]), 0.0)
    px = px - d1 * k[0];  py = py - d1 * k[1]
    d2 = 2.0 * np.minimum(dot(vec2(px, py), vec2(k[1], k[0])), 0.0)
    px = px - d2 * k[1];  py = py - d2 * k[0]
    px = px - clamp(px, r * k[2], r * k[3])
    py = py - r
    return length(vec2(px, py)) * np.sign(py)


def sdStar5(p: _F, r: float, rf: float) -> _F:
    """2-D 5-pointed star; *r* outer radius, *rf* inner factor (0–1)."""
    return sdStar(p, r, 5, rf)


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
    # Simultaneous update: d = clamp(-dot(p,ecs), 0, limit); p += ecs*d
    d   = clamp(-dot(vec2(px, py), ecs), 0.0, r * acs[1] / ecs[1])
    px  = px + ecs[0] * d
    py  = py + ecs[1] * d
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
    # Apply mat2(-c.x, c.y, c.y, c.x) — GLSL column-major: col0=(-cx,cy), col1=(cy,cx)
    new_px = -c[0] * px + c[1] * py
    new_py =  c[1] * px + c[0] * py
    px, py = new_px, new_py
    # p = vec2((py>0||px>0)?px:l*sign(-cx), (px>0)?py:l)  — both read pre-update values
    px_out = np.where((py > 0.0) | (px > 0.0), px, l * np.sign(-c[0]))
    py_out = np.where(px > 0.0, py, l)
    px, py = px_out, py_out
    # p = vec2(px, abs(py-r)) - w  then standard 2D box SDF
    qx = px - w[0]
    qy = np.abs(py - r) - w[1]
    return length(np.maximum(vec2(qx, qy), 0.0)) + np.minimum(0.0, np.maximum(qx, qy))


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
    k   = 0.5 * (h + 1.0 / h)
    px  = np.abs(p[..., 0]);  py = np.abs(p[..., 1])
    # IQ: (px<1 && py<px*(k-h)+h) ? k-sqrt(dot2(p-vec2(1,k))) : sqrt(min(dot2(p-vec2(0,h)), dot2(p-vec2(1,0))))
    cond    = (px < 1.0) & (py < px * (k - h) + h)
    inside  = k - np.sqrt(dot2(vec2(px - 1.0, py - k)))
    outside = np.sqrt(np.minimum(dot2(vec2(px, py - h)), dot2(vec2(px - 1.0, py))))
    return np.where(cond, inside, outside)


def sdEgg2D(p: _F, ra: float, rb: float) -> _F:
    """2-D egg; *ra* large radius, *rb* small radius."""
    k  = np.sqrt(3.0)
    px = np.abs(p[..., 0]);  py = p[..., 1]
    r  = ra - rb
    # All three branches share the outer -rb; it must apply to every case
    return np.where(py < 0.0,
                    length(vec2(px, py)) - r,
                    np.where(k * (px + r) < py,
                             length(vec2(px, py - k * r)),
                             length(vec2(px + r, py)) - 2.0 * r)) - rb


def sdHeart2D(p: _F) -> _F:
    """2-D heart shape (unit-scale)."""
    px = np.abs(p[..., 0]);  py = p[..., 1]
    # IQ: (px+py>1) ? sqrt(dot2(p-vec2(0.25,0.75)))-sqrt(2)/4
    #               : sqrt(min(dot2(p-vec2(0,1)), dot2(p-0.5*max(px+py,0))))*sign(px-py)
    cond    = px + py > 1.0
    inside  = np.sqrt(dot2(vec2(px - 0.25, py - 0.75))) - np.sqrt(2.0) / 4.0
    s       = 0.5 * np.maximum(px + py, 0.0)
    outside = np.sqrt(np.minimum(
        dot2(vec2(px, py - 1.0)),
        dot2(vec2(px - s, py - s)),
    )) * np.sign(px - py)
    return np.where(cond, inside, outside)


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
    px = np.abs(p[..., 0]);  py = np.abs(p[..., 1])
    # IQ swaps if p.x > p.y so that after swap p.x <= p.y; ab swaps with p
    swap = px > py
    px_s = np.where(swap, py, px);  py_s = np.where(swap, px, py)
    a    = np.where(swap, ab[1], ab[0]);  b = np.where(swap, ab[0], ab[1])
    l    = b * b - a * a
    m    = a * px_s / l;   m2 = m * m
    n_   = b * py_s / l;   n2 = n_ * n_
    c    = (m2 + n2 - 1.0) / 3.0;   c3 = c * c * c
    q    = c3 + m2 * n2 * 2.0
    d    = c3 + m2 * n2
    g    = m + m * n2
    # d < 0 branch (3 real roots via acos)
    h_n  = np.arccos(np.clip(safe_div(q, c3), -1.0, 1.0)) / 3.0
    s_n  = np.cos(h_n);   t_n = np.sin(h_n) * np.sqrt(3.0)
    rx_n = np.sqrt(np.maximum(-c * (s_n + t_n + 2.0) + m2, 0.0))
    ry_n = np.sqrt(np.maximum(-c * (s_n - t_n + 2.0) + m2, 0.0))
    co_n = (ry_n + np.sign(l) * rx_n + np.abs(g) / np.maximum(rx_n * ry_n, 1e-30) - m) / 2.0
    # d >= 0 branch (1 real root via cube roots)
    h_p  = 2.0 * m * n_ * np.sqrt(np.maximum(d, 0.0))
    s_p  = np.sign(q + h_p) * np.power(np.abs(q + h_p), 1.0 / 3.0)
    u_p  = np.sign(q - h_p) * np.power(np.abs(q - h_p), 1.0 / 3.0)
    rx_p = -s_p - u_p - c * 4.0 + 2.0 * m2
    ry_p = (s_p - u_p) * np.sqrt(3.0)
    rm   = np.sqrt(np.maximum(rx_p * rx_p + ry_p * ry_p, 0.0))
    co_p = (safe_div(ry_p, np.sqrt(np.maximum(rm - rx_p, 0.0))) + 2.0 * g / np.maximum(rm, 1e-30) - m) / 2.0
    co   = np.where(d < 0.0, co_n, co_p)
    r_x  = a * co
    r_y  = b * np.sqrt(np.maximum(1.0 - co * co, 0.0))
    return length(vec2(r_x - px_s, r_y - py_s)) * np.sign(py_s - r_y)


def sdParabola2D(p: _F, k: float) -> _F:
    """2-D parabola ``y = k·x²``; *k* is the curvature."""
    px = np.abs(p[..., 0]);  py = p[..., 1]
    ik = 1.0 / k
    p2 = ik * (py - 0.5 * ik) / 3.0
    q  = 0.25 * ik * ik * px          # IQ: 0.25/k², not 1/k²
    h  = q * q - p2 * p2 * p2
    r  = np.sqrt(np.abs(h))
    # h > 0: IQ computes r = cbrt(q+sqrt(h)), x = r + p2/r
    cbrt = np.power(np.maximum(q + r, 0.0), 1.0 / 3.0)
    x_p  = cbrt + p2 / np.maximum(cbrt, 1e-30)
    # h <= 0: trigonometric method
    x_n  = 2.0 * np.cos(np.arctan2(r, q) / 3.0) * np.sqrt(np.maximum(p2, 0.0))
    x    = np.where(h > 0.0, x_p, x_n)
    return length(vec2(px - x, py - k * x * x)) * np.sign(px - x)


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
    # IQ: pos = vec2(|px-py|, 1-px-py) / sqrt(2)
    pos_x = np.abs(px - py) / np.sqrt(2.0)
    pos_y = (1.0 - px - py) / np.sqrt(2.0)
    p_val = (he - pos_y - 0.25 / he) / (6.0 * he)
    q_val = pos_x / (he * he * 16.0)
    h     = q_val * q_val - p_val * p_val * p_val
    r     = np.sqrt(np.abs(h))
    cbrt_p = np.power(np.maximum(q_val + r, 0.0), 1.0 / 3.0)
    cbrt_m = np.power(np.maximum(np.abs(q_val - r), 0.0), 1.0 / 3.0) * np.sign(r - q_val)
    r_sqrt = np.sqrt(np.maximum(p_val, 0.0))
    x = np.minimum(
        np.where(h > 0.0,
                 cbrt_p - cbrt_m,
                 2.0 * r_sqrt * np.cos(np.arccos(np.clip(
                     safe_div(q_val, np.maximum(p_val * r_sqrt, 1e-30)), -1.0, 1.0)) / 3.0)),
        np.sqrt(2.0) / 2.0,
    )
    zx = x - pos_x
    zy = he * (1.0 - 2.0 * x * x) - pos_y
    return length(vec2(zx, zy)) * np.sign(zy)


def sdTunnel2D(p: _F, wh: _F) -> _F:
    """2-D tunnel/arch; *wh* = ``(half_width, height)``."""
    px = np.abs(p[..., 0]);  py = -p[..., 1]   # IQ negates p.y
    qx = px - wh[0];  qy = py - wh[1]
    d1  = dot2(vec2(np.maximum(qx, 0.0), qy))
    # IQ: q.x = (py>0) ? q.x : length(px,py) - wh.x
    qx2 = np.where(py > 0.0, qx, length(vec2(px, py)) - wh[0])
    d2  = dot2(vec2(qx2, np.maximum(qy, 0.0)))
    d   = np.sqrt(np.minimum(d1, d2))
    return np.where(np.maximum(qx2, qy) < 0.0, -d, d)


def sdStairs2D(p: _F, wh: _F, n: int) -> _F:
    """2-D staircase; *wh* = ``(step_width, step_height)``, *n* steps."""
    ba  = wh * n
    px  = p[..., 0];  py = p[..., 1]
    # IQ: d = min(dot2(p - vec2(clamp(p.x,0,ba.x), 0)),
    #              dot2(p - vec2(ba.x, clamp(p.y,0,ba.y))))
    d   = np.minimum(
        dot2(p - vec2(clamp(px, 0.0, ba[0]), np.zeros_like(py))),
        dot2(p - vec2(np.full_like(px, ba[0]), clamp(py, 0.0, ba[1]))),
    )
    s   = np.sign(np.maximum(-py, px - ba[0]))
    dia = length(wh)
    # Rotate into stair-aligned frame: mat2(wh.x,-wh.y, wh.y,wh.x)/dia
    rx  = (wh[0] * px - wh[1] * py) / dia
    ry  = (wh[1] * px + wh[0] * py) / dia
    id_ = clamp(np.round(rx / dia), 0.0, n - 1.0)
    rx  = rx - id_ * dia
    # Rotate back: mat2(wh.x,wh.y,-wh.y,wh.x)/dia
    bx  = (wh[0] * rx + wh[1] * ry) / dia
    by  = (-wh[1] * rx + wh[0] * ry) / dia
    hh  = wh[1] / 2.0
    by  = by - hh
    s   = np.where(by > hh * np.sign(bx), 1.0, s)
    # IQ: p = (id<0.5 || p.x>0) ? p : -p
    flip = ~((id_ < 0.5) | (bx > 0.0))
    bx   = np.where(flip, -bx, bx);  by = np.where(flip, -by, by)
    d    = np.minimum(d, dot2(vec2(bx, by - clamp(by, -hh, hh))))
    d    = np.minimum(d, dot2(vec2(clamp(bx, 0.0, wh[0]), by - hh)))
    return np.sqrt(np.maximum(d, 0.0)) * s


def sdQuadraticCircle2D(p: _F) -> _F:
    """2-D quadratic-circle approximation (unit-scale)."""
    px = np.abs(p[..., 0]);  py = np.abs(p[..., 1])
    # Swap so px >= py
    swap = py > px
    px_s = np.where(swap, py, px);  py_s = np.where(swap, px, py)
    a = px_s - py_s;  b = px_s + py_s
    c = (2.0 * b - 1.0) / 3.0
    H = a * a + c * c * c
    # h >= 0 branch: h=sqrt(H), t = sign(h-a)*|h-a|^(1/3) - (h+a)^(1/3)
    h_s  = np.sqrt(np.maximum(H, 0.0))
    t_p  = (np.sign(h_s - a) * np.power(np.maximum(np.abs(h_s - a), 0.0), 1.0 / 3.0)
            - np.power(np.maximum(h_s + a, 0.0), 1.0 / 3.0))
    # h < 0 branch: z=sqrt(-c), v=acos(a/(c*z))/3, t=-z*(cos(v)+sin(v)*sqrt(3))
    z    = np.sqrt(np.maximum(-c, 0.0))
    cz   = c * z   # negative when c<0: c*sqrt(-c)
    v    = np.arccos(np.clip(safe_div(a, cz), -1.0, 1.0)) / 3.0
    t_n  = -z * (np.cos(v) + np.sin(v) * np.sqrt(3.0))
    t    = np.where(H >= 0.0, t_p, t_n) * 0.5
    # w = vec2(-t,t) + 0.75 - t² - p
    wx   = -t + 0.75 - t * t - px_s
    wy   =  t + 0.75 - t * t - py_s
    return length(vec2(wx, wy)) * np.sign(a * a * 0.5 + b - 1.5)


def sdHyperbola2D(p: _F, k: float, he: float) -> _F:
    """2-D hyperbola; *k* curvature, *he* half-height."""
    px = np.abs(p[..., 0]);  py = np.abs(p[..., 1])
    # IQ: rotate 45°  →  p = vec2(p.x-p.y, p.x+p.y)/sqrt(2)
    px_r = (px - py) / np.sqrt(2.0)
    py_r = (px + py) / np.sqrt(2.0)
    x2  = px_r * px_r / 16.0;  y2 = py_r * py_r / 16.0
    r   = k * (4.0 * k - px_r * py_r) / 12.0
    q   = (x2 - y2) * k * k
    H   = q * q + r * r * r
    # H < 0 branch
    m_n = np.sqrt(np.maximum(-r, 0.0))
    rm  = r * m_n                                       # r * sqrt(-r), negative when r<0
    u_n = m_n * np.cos(np.arccos(np.clip(safe_div(q, rm), -1.0, 1.0)) / 3.0)
    # H >= 0 branch
    m_p = np.power(np.maximum(np.sqrt(np.maximum(H, 0.0)) - q, 0.0), 1.0 / 3.0)
    u_p = (m_p - safe_div(r, m_p)) / 2.0
    u   = np.where(H < 0.0, u_n, u_p)
    w   = np.sqrt(np.maximum(u + x2, 0.0))
    b_v = k * py_r - x2 * px_r * 2.0
    t   = px_r / 4.0 - w + np.sqrt(np.maximum(2.0 * x2 - u + safe_div(b_v, w) / 4.0, 0.0))
    t   = np.maximum(t, np.sqrt(he * he * 0.5 + k) - he / np.sqrt(2.0))
    d   = length(vec2(px_r - t, py_r - k / np.maximum(t, 1e-30)))
    return np.where(px_r * py_r < k, d, -d)


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
