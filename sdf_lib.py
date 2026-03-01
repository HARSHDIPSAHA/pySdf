import numpy as np


def vec2(x, y):
    x, y = np.broadcast_arrays(x, y)
    return np.stack([x, y], axis=-1)


def vec3(x, y, z):
    x, y, z = np.broadcast_arrays(x, y, z)
    return np.stack([x, y, z], axis=-1)


def length(v):
    return np.linalg.norm(v, axis=-1)


def dot(a, b):
    return np.sum(a * b, axis=-1)


def dot2(a):
    return dot(a, a)


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def safe_div(n, d, eps=1e-12):
    return n / np.where(np.abs(d) < eps, np.sign(d) * eps + eps, d)

def cbrt(x):
    """Computes real cube root (handles negatives correctly)."""
    return np.sign(x) * np.power(np.abs(x), 1.0/3.0)

def safe_sqrt(x):
    """Clips negative noise before sqrt."""
    return np.sqrt(np.maximum(x, 0.0))

def safe_acos(x):
    """Clips input to [-1, 1] before arccos."""
    return np.arccos(np.clip(x, -1.0, 1.0))


def sdSphere(p, s):
    return length(p) - s


def sdBox(p, b):
    q = np.abs(p) - b
    return length(np.maximum(q, 0.0)) + np.minimum(np.max(q, axis=-1), 0.0)


def sdRoundBox(p, b, r):
    q = np.abs(p) - b + r
    return length(np.maximum(q, 0.0)) + np.minimum(np.max(q, axis=-1), 0.0) - r


def sdBoxFrame(p, b, e):
    p = np.abs(p) - b
    q = np.abs(p + e) - e
    a = length(np.maximum(vec3(p[..., 0], q[..., 1], q[..., 2]), 0.0)) + np.minimum(
        np.max(vec3(p[..., 0], q[..., 1], q[..., 2]), axis=-1), 0.0
    )
    b = length(np.maximum(vec3(q[..., 0], p[..., 1], q[..., 2]), 0.0)) + np.minimum(
        np.max(vec3(q[..., 0], p[..., 1], q[..., 2]), axis=-1), 0.0
    )
    c = length(np.maximum(vec3(q[..., 0], q[..., 1], p[..., 2]), 0.0)) + np.minimum(
        np.max(vec3(q[..., 0], q[..., 1], p[..., 2]), axis=-1), 0.0
    )
    return np.minimum(np.minimum(a, b), c)


def sdTorus(p, t):
    q = vec2(length(p[..., [0, 2]]) - t[0], p[..., 1])
    return length(q) - t[1]


def sdCappedTorus(p, sc, ra, rb):
    px = np.abs(p[..., 0])
    py = p[..., 1]
    k = np.where(sc[1] * px > sc[0] * py, px * sc[0] + py * sc[1], length(vec2(px, py)))
    return np.sqrt(dot2(p) + ra * ra - 2.0 * ra * k) - rb


def sdLink(p, le, r1, r2):
    q = vec3(p[..., 0], np.maximum(np.abs(p[..., 1]) - le, 0.0), p[..., 2])
    return length(vec2(length(q[..., [0, 1]]) - r1, q[..., 2])) - r2


def sdCylinder(p, c):
    return length(vec2(p[..., 0] - c[0], p[..., 2] - c[1])) - c[2]


def sdConeExact(p, c, h):
    q = h * vec2(safe_div(c[0], c[1]), -1.0)
    w = vec2(length(p[..., [0, 2]]), p[..., 1])
    a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0)[..., None]
    b = w - q * vec2(clamp(safe_div(w[..., 0], q[0]), 0.0, 1.0), 1.0)
    k = np.sign(q[1])
    d = np.minimum(dot2(a), dot2(b))
    s = np.maximum(k * (w[..., 0] * q[1] - w[..., 1] * q[0]), k * (w[..., 1] - q[1]))
    return np.sqrt(d) * np.sign(s)


def sdConeBound(p, c, h):
    q = length(p[..., [0, 2]])
    return np.maximum(c[0] * q + c[1] * p[..., 1], -h - p[..., 1])


def sdConeInfinite(p, c):
    q = vec2(length(p[..., [0, 2]]), -p[..., 1])
    d = length(q - c * np.maximum(dot(q, c), 0.0)[..., None])
    return d * np.where(q[..., 0] * c[1] - q[..., 1] * c[0] < 0.0, -1.0, 1.0)


def sdPlane(p, n, h):
    n = n / np.linalg.norm(n)
    return dot(p, n) + h


def sdHexPrism(p, h):
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


def sdTriPrism(p, h):
    q = np.abs(p)
    return np.maximum(
        q[..., 2] - h[1],
        np.maximum(q[..., 0] * 0.866025 + p[..., 1] * 0.5, -p[..., 1]) - h[0] * 0.5,
    )


def sdCapsule(p, a, b, r):
    pa = p - a
    ba = b - a
    h = clamp(dot(pa, ba) / dot2(ba), 0.0, 1.0)
    return length(pa - ba * h[..., None]) - r


def sdVerticalCapsule(p, h, r):
    py = p[..., 1] - clamp(p[..., 1], 0.0, h)
    return length(vec3(p[..., 0], py, p[..., 2])) - r


def sdCappedCylinder(p, r, h):
    d = np.abs(vec2(length(p[..., [0, 2]]), p[..., 1])) - vec2(r, h)
    return np.minimum(np.maximum(d[..., 0], d[..., 1]), 0.0) + length(np.maximum(d, 0.0))


def sdCappedCylinderSegment(p, a, b, r):
    ba = b - a
    pa = p - a
    baba = dot2(ba)
    paba = dot(pa, ba)
    x = length(pa * baba - ba * paba[..., None]) - r * baba
    y = np.abs(paba - baba * 0.5) - baba * 0.5
    x2 = x * x
    y2 = y * y * baba
    d = np.where(
        np.maximum(x, y) < 0.0,
        -np.minimum(x2, y2),
        (np.where(x > 0.0, x2, 0.0) + np.where(y > 0.0, y2, 0.0)),
    )
    return np.sign(d) * np.sqrt(np.abs(d)) / baba


def sdRoundedCylinder(p, ra, rb, h):
    d = vec2(length(p[..., [0, 2]]) - ra + rb, np.abs(p[..., 1]) - h + rb)
    return (
        np.minimum(np.maximum(d[..., 0], d[..., 1]), 0.0)
        + length(np.maximum(d, 0.0))
        - rb
    )


def sdCappedCone(p, h, r1, r2):
    q = vec2(length(p[..., [0, 2]]), p[..., 1])
    k1 = vec2(r2, h)
    k2 = vec2(r2 - r1, 2.0 * h)
    ca = vec2(
        q[..., 0] - np.minimum(q[..., 0], np.where(q[..., 1] < 0.0, r1, r2)),
        np.abs(q[..., 1]) - h,
    )
    cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot2(k2), 0.0, 1.0)[..., None]
    s = np.where((cb[..., 0] < 0.0) & (ca[..., 1] < 0.0), -1.0, 1.0)
    return s * np.sqrt(np.minimum(dot2(ca), dot2(cb)))


def sdCappedConeSegment(p, a, b, ra, rb):
    rba = rb - ra
    baba = dot2(b - a)
    papa = dot2(p - a)
    paba = dot(p - a, b - a) / baba
    x = np.sqrt(papa - paba * paba * baba)
    cax = np.maximum(0.0, x - np.where(paba < 0.5, ra, rb))
    cay = np.abs(paba - 0.5) - 0.5
    k = rba * rba + baba
    f = clamp((rba * (x - ra) + paba * baba) / k, 0.0, 1.0)
    cbx = x - ra - f * rba
    cby = paba - f
    s = np.where((cbx < 0.0) & (cay < 0.0), -1.0, 1.0)
    return s * np.sqrt(
        np.minimum(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba)
    )


def sdSolidAngle(p, c, ra):
    q = vec2(length(p[..., [0, 2]]), p[..., 1])
    l = length(q) - ra
    m = length(q - c * clamp(dot(q, c), 0.0, ra)[..., None])
    return np.maximum(l, m * np.sign(c[1] * q[..., 0] - c[0] * q[..., 1]))


def sdCutSphere(p, r, h):
    w = np.sqrt(r * r - h * h)
    q = vec2(length(p[..., [0, 2]]), p[..., 1])
    s = np.maximum((h - r) * q[..., 0] * q[..., 0] + w * w * (h + r - 2.0 * q[..., 1]),
                   h * q[..., 0] - w * q[..., 1])
    return np.where(
        s < 0.0,
        length(q) - r,
        np.where(q[..., 0] < w, h - q[..., 1], length(q - vec2(w, h))),
    )


def sdCutHollowSphere(p, r, h, t):
    w = np.sqrt(r * r - h * h)
    q = vec2(length(p[..., [0, 2]]), p[..., 1])
    return np.where(h * q[..., 0] < w * q[..., 1], length(q - vec2(w, h)), np.abs(length(q) - r)) - t


def sdDeathStar(p2, ra, rb, d):
    a = (ra * ra - rb * rb + d * d) / (2.0 * d)
    b = np.sqrt(np.maximum(ra * ra - a * a, 0.0))
    p = vec2(p2[..., 0], length(p2[..., [1, 2]]))
    cond = p[..., 0] * b - p[..., 1] * a > d * np.maximum(b - p[..., 1], 0.0)
    return np.where(
        cond,
        length(p - vec2(a, b)),
        np.maximum(length(p) - ra, -(length(p - vec2(d, 0.0)) - rb)),
    )


def sdRoundCone(p, r1, r2, h):
    b = (r1 - r2) / h
    a = np.sqrt(1.0 - b * b)
    q = vec2(length(p[..., [0, 2]]), p[..., 1])
    k = dot(q, vec2(-b, a))
    return np.where(
        k < 0.0,
        length(q) - r1,
        np.where(k > a * h, length(q - vec2(0.0, h)) - r2, dot(q, vec2(a, b)) - r1),
    )


def sdRoundConeSegment(p, a, b, r1, r2):
    ba = b - a
    l2 = dot2(ba)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = p - a
    y = dot(pa, ba)
    z = y - l2
    x2 = dot2(pa * l2 - ba * y[..., None])
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    cond1 = np.sign(z) * a2 * z2 > k
    cond2 = np.sign(y) * a2 * y2 < k
    out1 = np.sqrt(x2 + z2) * il2 - r2
    out2 = np.sqrt(x2 + y2) * il2 - r1
    out3 = (np.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1
    return np.where(cond1, out1, np.where(cond2, out2, out3))


def sdEllipsoid(p, r):
    k0 = length(p / r)
    k1 = length(p / (r * r))
    return k0 * (k0 - 1.0) / np.where(k1 == 0.0, 1e-12, k1)


def sdVesicaSegment(p, a, b, w):
    c = (a + b) * 0.5
    l = length(b - a)
    v = (b - a) / l
    y = dot(p - c, v)
    q = vec2(length(p - c - v * y[..., None]), np.abs(y))
    r = 0.5 * l
    d = 0.5 * (r * r - w * w) / w
    cond = r * q[..., 0] < d * (q[..., 1] - r)
    h = np.where(cond[..., None], vec3(0.0, r, 0.0), vec3(-d, 0.0, d + w))
    return length(q - h[..., :2]) - h[..., 2]


def sdRhombus(p, la, lb, h, ra):
    p = np.abs(p)
    f = clamp((la * p[..., 0] - lb * p[..., 2] + lb * lb) / (la * la + lb * lb), 0.0, 1.0)
    w = p[..., [0, 2]] - vec2(la, lb) * vec2(f, 1.0 - f)
    q = vec2(length(w) * np.sign(w[..., 0]) - ra, p[..., 1] - h)
    return np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0) + length(np.maximum(q, 0.0))


def sdOctahedronExact(p, s):
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
    k = clamp(0.5 * (q[..., 2] - q[..., 1] + s), 0.0, s)
    dist = length(vec3(q[..., 0], q[..., 1] - s + k, q[..., 2] - k))
    return np.where(mask1 | mask2 | mask3, dist, res)


def sdOctahedronBound(p, s):
    p = np.abs(p)
    return (p[..., 0] + p[..., 1] + p[..., 2] - s) * 0.57735027


def sdPyramid(p, h):
    m2 = h * h + 0.25
    pxz = np.abs(p[..., [0, 2]])
    swap = pxz[..., 1] > pxz[..., 0]
    px = np.where(swap, pxz[..., 1], pxz[..., 0])
    pz = np.where(swap, pxz[..., 0], pxz[..., 1])
    pxz = vec2(px, pz) - 0.5
    q = vec3(pxz[..., 1], h * p[..., 1] - 0.5 * pxz[..., 0], h * pxz[..., 0] + 0.5 * p[..., 1])
    s = np.maximum(-q[..., 0], 0.0)
    t = clamp((q[..., 1] - 0.5 * pxz[..., 1]) / (m2 + 0.25), 0.0, 1.0)
    a = m2 * (q[..., 0] + s) * (q[..., 0] + s) + q[..., 1] * q[..., 1]
    b = m2 * (q[..., 0] + 0.5 * t) * (q[..., 0] + 0.5 * t) + (q[..., 1] - m2 * t) * (
        q[..., 1] - m2 * t
    )
    d2 = np.where(np.minimum(q[..., 1], -q[..., 0] * m2 - q[..., 1] * 0.5) > 0.0, 0.0, np.minimum(a, b))
    return np.sqrt((d2 + q[..., 2] * q[..., 2]) / m2) * np.sign(np.maximum(q[..., 2], -p[..., 1]))


def udTriangle(p, a, b, c):
    ba = b - a
    pa = p - a
    cb = c - b
    pb = p - b
    ac = a - c
    pc = p - c
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
    d = np.minimum(np.minimum(d1, d2), d3)
    d_plane = dot(nor, pa) * dot(nor, pa) / dot2(nor)
    return np.sqrt(np.where(cond, d, d_plane))


def udQuad(p, a, b, c, d):
    ba = b - a
    pa = p - a
    cb = c - b
    pb = p - b
    dc = d - c
    pc = p - c
    ad = a - d
    pd = p - d
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
    d = np.minimum(np.minimum(np.minimum(d1, d2), d3), d4)
    d_plane = dot(nor, pa) * dot(nor, pa) / dot2(nor)
    return np.sqrt(np.where(cond, d, d_plane))


def sdCircle2d(p, r):
    return length(p) - r


def sdBox2d(p, b):
    q = np.abs(p) - b
    return length(np.maximum(q, 0.0)) + np.minimum(np.max(q, axis=-1), 0.0)


def opUnion(d1, d2):
    return np.minimum(d1, d2)


def opSubtraction(d1, d2):
    return np.maximum(-d1, d2)


def opIntersection(d1, d2):
    return np.maximum(d1, d2)


def opXor(d1, d2):
    return np.maximum(np.minimum(d1, d2), -np.maximum(d1, d2))


def opSmoothUnion(d1, d2, k):
    k = k * 4.0
    h = np.maximum(k - np.abs(d1 - d2), 0.0)
    return np.minimum(d1, d2) - h * h * 0.25 / k


def opSmoothSubtraction(d1, d2, k):
    return -opSmoothUnion(d1, -d2, k)


def opSmoothIntersection(d1, d2, k):
    return -opSmoothUnion(-d1, -d2, k)


def opRevolution(p, primitive2d, o):
    q = vec2(length(p[..., [0, 2]]) - o, p[..., 1])
    return primitive2d(q)


def opExtrusion(p, primitive2d, h):
    d = primitive2d(p[..., :2])
    w = vec2(d, np.abs(p[..., 2]) - h)
    return np.minimum(np.maximum(w[..., 0], w[..., 1]), 0.0) + length(np.maximum(w, 0.0))


def opElongate1(p, primitive3d, h):
    q = p - clamp(p, -h, h)
    return primitive3d(q)


def opElongate2(p, primitive3d, h):
    q = np.abs(p) - h
    return primitive3d(np.maximum(q, 0.0)) + np.minimum(np.max(q, axis=-1), 0.0)


def opRound(p, primitive3d, rad):
    return primitive3d(p) - rad


def opOnion(sdf, thickness):
    return np.abs(sdf) - thickness


def opScale(p, s, primitive3d):
    return primitive3d(p / s) * s


def opSymX(p, primitive3d):
    p = p.copy()
    p[..., 0] = np.abs(p[..., 0])
    return primitive3d(p)


def opSymXZ(p, primitive3d):
    p = p.copy()
    p[..., 0] = np.abs(p[..., 0])
    p[..., 2] = np.abs(p[..., 2])
    return primitive3d(p)


def opRepetition(p, s, primitive3d):
    q = p - s * np.round(p / s)
    return primitive3d(q)


def opLimitedRepetition(p, s, l, primitive3d):
    q = p - s * clamp(np.round(p / s), -l, l)
    return primitive3d(q)


def opDisplace(p, primitive3d):
    d1 = primitive3d(p)
    d2 = np.sin(20.0 * p[..., 0]) * np.sin(20.0 * p[..., 1]) * np.sin(20.0 * p[..., 2])
    return d1 + d2


def opTwist(p, primitive3d, k):
    c = np.cos(k * p[..., 1])
    s = np.sin(k * p[..., 1])
    x = c * p[..., 0] - s * p[..., 2]
    z = s * p[..., 0] + c * p[..., 2]
    q = vec3(x, p[..., 1], z)
    return primitive3d(q)


def opCheapBend(p, primitive3d, k):
    c = np.cos(k * p[..., 0])
    s = np.sin(k * p[..., 0])
    x = c * p[..., 0] - s * p[..., 1]
    y = s * p[..., 0] + c * p[..., 1]
    q = vec3(x, y, p[..., 2])
    return primitive3d(q)


def opTx(p, rot, trans, primitive3d):
    inv_rot = rot.T
    q = (p - trans) @ inv_rot
    return primitive3d(q)

# ============================================================================
# 2D SDF FUNCTIONS (from Inigo Quilez)
# ============================================================================

# 2D Circle
def sdCircle(p, r):
    """2D Circle SDF. p is vec2, r is radius."""
    return length(p) - r

# 2D Box (Rectangle)
def sdBox2D(p, b):
    """2D Box SDF. p is vec2, b is vec2 half-size."""
    d = np.abs(p) - b
    return length(np.maximum(d, 0.0)) + np.minimum(np.maximum(d[..., 0], d[..., 1]), 0.0)

# 2D Rounded Box
def sdRoundedBox2D(p, b, r):
    """2D Rounded Box. p is vec2, b is vec2 half-size, r is corner radius."""
    d = np.abs(p) - b + r
    return length(np.maximum(d, 0.0)) + np.minimum(np.maximum(d[..., 0], d[..., 1]), 0.0) - r

# 2D Oriented Box
def sdOrientedBox2D(p, a, b, th):
    """2D Oriented Box. p is vec2, a/b are vec2 corners, th is thickness."""
    l = length(b - a)
    d = (b - a) / l
    q = p - (a + b) * 0.5
    q = vec2(dot(q, d), np.abs(dot(q, vec2(-d[1], d[0])))) 
    q = np.abs(q) - vec2(l * 0.5, th)
    return length(np.maximum(q, 0.0)) + np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0)

# 2D Segment
def sdSegment2D(p, a, b):
    """2D Line Segment. p is vec2, a/b are vec2 endpoints."""
    pa = p - a
    ba = b - a
    h = clamp(dot(pa, ba) / dot2(ba), 0.0, 1.0)
    return length(pa - ba * h[..., None])

# 2D Rhombus
def sdRhombus2D(p, b):
    """2D Rhombus. p is vec2, b is vec2 half-size."""
    p = np.abs(p)
    h = clamp((-2.0 * dot2(p) + dot2(b)) / dot2(b), -1.0, 1.0)
    d = length(p - 0.5 * b * vec2(1.0 - h, 1.0 + h))
    return d * np.sign(p[..., 0] * b[1] + p[..., 1] * b[0] - b[0] * b[1])

# 2D Trapezoid
def sdTrapezoid2D(p, r1, r2, he):
    """2D Isosceles Trapezoid. p is vec2, r1/r2 are base widths, he is height."""
    k1 = vec2(r2, he)
    k2 = vec2(r2 - r1, 2.0 * he)
    px = np.abs(p[..., 0])
    py = p[..., 1]
    ca = vec2(np.maximum(0.0, px - np.where(py < 0.0, r1, r2)), np.abs(py) - he)
    p_sym = vec2(px, py)
    cb = p_sym - k1 + k2 * clamp(dot(k1 - p_sym, k2) / dot2(k2), 0.0, 1.0)[..., None]
    s = np.where((cb[..., 0] < 0.0) & (ca[..., 1] < 0.0), -1.0, 1.0)
    return s * np.sqrt(np.minimum(dot2(ca), dot2(cb)))

# 2D Parallelogram
def sdParallelogram2D(p, wi, he, sk):
    """2D Parallelogram. p is vec2, wi is width, he is height, sk is skew."""
    e = vec2(sk, he)
    p = np.where((p[..., 1] < 0.0)[..., None], -p, p)
    #horizontal edge
    w = p - e
    w_x = w[..., 0] - clamp(w[...,0],-wi,wi)
    w = vec2(w_x,w[...,1])
    d = vec2(dot2(w), -w[...,1])
    #vertical edge
    s = p[...,0]*e[1] - p[...,1] *e[0]
    p = np.where((s<0.0)[...,None],-p,p)
    v = p - vec2(wi,0.0)
    factor = clamp(dot(v, e) / dot2(e), -1.0, 1.0)
    v -= e * factor[..., None]
    d_new = vec2(dot2(v), wi*he - np.abs(s))
    #combine
    d = np.minimum(d,d_new)
    return np.sqrt(d[...,0]) * np.sign(-d[...,1])

# 2D Equilateral Triangle
def sdEquilateralTriangle2D(p, r):
    """2D Equilateral Triangle. p is vec2, r is circumradius."""
    k = np.sqrt(3.0)
    px = np.abs(p[..., 0]) - r
    py = p[..., 1] + r / k
    cond = (px + k * py > 0.0)
    px_new = np.where(cond, (px - k * py) * 0.5, px)
    py_new = np.where(cond, (-k * px - py) * 0.5, py)
    px_new = px_new - clamp(px_new, -2.0 * r, 0.0)
    return -length(vec2(px_new, py_new)) * np.sign(py_new)

# 2D Isosceles Triangle
def sdTriangleIsosceles2D(p, q):
    """2D Isosceles Triangle. p is vec2, q is vec2(width, height)."""
    px = np.abs(p[..., 0])
    py = p[..., 1]
    p_vec = vec2(px, py)
    a = p_vec - q * clamp(dot(p_vec, q)/ dot2(q), 0.0, 1.0)[..., None]
    t_b = clamp(px / q[0], 0.0, 1.0)
    b = p_vec - q * vec2(t_b, np.ones_like(t_b))
    k = np.sign(q[1])
    d = np.minimum(dot2(a), dot2(b))
    s = np.maximum(k * (px * q[1] - py * q[0]), k * (py - q[1]))
    return np.sqrt(d) * np.sign(s)

# 2D Triangle (from 3 vertices)
def sdTriangle2D(p, p0, p1, p2):
    """2D Triangle from 3 vertices. p, p0, p1, p2 are vec2."""
    e0 = p1 - p0
    e1 = p2 - p1
    e2 = p0 - p2
    v0 = p - p0
    v1 = p - p1
    v2 = p - p2
    pq0 = v0 - e0 * clamp(dot(v0, e0) / dot2(e0), 0.0, 1.0)[..., None]
    pq1 = v1 - e1 * clamp(dot(v1, e1) / dot2(e1), 0.0, 1.0)[..., None]
    pq2 = v2 - e2 * clamp(dot(v2, e2) / dot2(e2), 0.0, 1.0)[..., None]
    s = np.sign(e0[0] * e2[1] - e0[1] * e2[0])
    d = np.minimum(np.minimum(
        vec2(dot2(pq0), s * (v0[..., 0] * e0[1] - v0[..., 1] * e0[0])),
        vec2(dot2(pq1), s * (v1[..., 0] * e1[1] - v1[..., 1] * e1[0]))),
        vec2(dot2(pq2), s * (v2[..., 0] * e2[1] - v2[..., 1] * e2[0])))
    return -np.sqrt(d[..., 0]) * np.sign(d[..., 1])

# 2D Uneven Capsule
def sdUnevenCapsule2D(p, r1, r2, h):
    """2D Uneven Capsule. p is vec2, r1/r2 are radii, h is height."""
    px = np.abs(p[..., 0])
    py = p[..., 1]
    b = (r1 - r2) / h
    a = np.sqrt(1.0 - b * b)
    k = dot(vec2(px, py), vec2(-b, a))
    cond1 = k < 0.0
    cond2 = k > a * h
    return np.where(cond1, length(vec2(px, py)) - r1,
           np.where(cond2, length(vec2(px, py - h)) - r2,
                    dot(vec2(px, py), vec2(a, b)) - r1))

# 2D Pentagon
def sdPentagon2D(p, r):
    """2D Regular Pentagon. p is vec2, r is circumradius."""
    k = np.array([0.809016994, 0.587785252, 0.726542528])
    px = np.abs(p[..., 0])
    py = p[..., 1]
    d1 = dot(vec2(px, py), vec2(-k[0], k[1]))
    px = px - 2.0 * np.minimum(d1, 0.0) * (-k[0])
    py = py - 2.0 * np.minimum(d1, 0.0) * k[1]
    d2 = dot(vec2(px, py), vec2(k[0], k[1]))
    px = px - 2.0 * np.minimum(d2, 0.0) * k[0]
    py = py - 2.0 * np.minimum(d2, 0.0) * k[1]
    px = px - clamp(px, -r * k[2], r * k[2])
    py = py - r
    return length(vec2(px, py)) * np.sign(py)

# 2D Hexagon
def sdHexagon2D(p, r):
    """2D Regular Hexagon. p is vec2, r is circumradius."""
    k = np.array([-0.866025404, 0.5, 0.577350269])
    p = np.abs(p)
    k_xy = k[:2]
    d = dot(k_xy, p)
    p -= 2.0 * np.minimum(d, 0.0)[..., None] * k_xy
    px_clamped = clamp(p[..., 0], -k[2] * r, k[2] * r)
    sub_vec = vec2(px_clamped, r) # Helper handles broadcasting if r is scalar
    p -= sub_vec
    return length(p) * np.sign(p[..., 1])

# 2D Octogon
def sdOctogon2D(p, r):
    """2D Regular Octagon. p is vec2, r is circumradius."""
    k = np.array([-0.9238795325, 0.3826834323, 0.4142135623])
    p = np.abs(p)
    v1 = np.array([k[0], k[1]])
    d1 = dot(v1, p)
    p -= 2.0 * np.minimum(d1, 0.0)[..., None] * v1
    v2 = np.array([-k[0], k[1]])
    d2 = dot(v2, p)
    p -= 2.0 * np.minimum(d2, 0.0)[..., None] * v2
    px_clamped = clamp(p[..., 0], -k[2] * r, k[2] * r)
    p -= vec2(px_clamped, r)
    return length(p) * np.sign(p[..., 1])


# 2D Hexagram (6-pointed star)
def sdHexagram2D(p, r):
    """2D Hexagram (Star of David). p is vec2, r is radius."""
    k = np.array([-0.5, 0.8660254038, 0.5773502692, 1.7320508076])
    p = np.abs(p)
    k_xy = k[:2]
    d1 = dot(k_xy, p)
    p -= 2.0 * np.minimum(d1, 0.0)[..., None] * k_xy
    k_yx = np.array([k[1], k[0]])
    d2 = dot(k_yx, p)
    p -= 2.0 * np.minimum(d2, 0.0)[..., None] * k_yx
    px_clamped = clamp(p[..., 0], r * k[2], r * k[3])
    p -= vec2(px_clamped, r)
    return length(p) * np.sign(p[..., 1])

# 2D Star (5-pointed)
def sdStar5(p, r, rf):
    """2D 5-pointed Star. p is vec2, r is outer radius, rf is inner factor."""
    k1 = np.array([0.809016994375, -0.587785252292])
    k2 = np.array([-k1[0], k1[1]])
    px = p[..., 0]
    py = p[..., 1]
    px = px - 2.0 * np.maximum(dot(vec2(px, py), k1), 0.0) * k1[0]
    py = py - 2.0 * np.maximum(dot(vec2(px, py), k1), 0.0) * k1[1]
    px = px - 2.0 * np.maximum(dot(vec2(px, py), k2), 0.0) * k2[0]
    py = py - 2.0 * np.maximum(dot(vec2(px, py), k2), 0.0) * k2[1]
    px = px * k1[0] + py * k1[1]
    py = py * k1[0] - px * k1[1]
    px = np.abs(px)
    py = py - r
    ba = rf * vec2(-k1[1], k1[0]) - vec2(0.0, 1.0)
    h = clamp(dot(vec2(px, py), ba) / dot2(ba), 0.0, r)
    return length(vec2(px, py) - ba * h) * np.sign(py * ba[0] - px * ba[1])

# 2D Regular Star (N-pointed)
def sdStar(p, r, n, m):
    """2D N-pointed Star. p is vec2, r is radius, n is points, m is factor."""
    an = np.pi / n
    en = np.pi / m
    acs = vec2(np.cos(an), np.sin(an))
    ecs = vec2(np.cos(en), np.sin(en))
    bn = np.arctan2(np.abs(p[..., 0]), p[..., 1]) % (2.0 * an) - an
    px = length(p) * np.cos(bn)
    py = length(p) * np.abs(np.sin(bn))
    px = px - r * acs[0]
    py = py - r * acs[1]
    px = px + ecs[1] * clamp(-dot(vec2(px, py), ecs), 0.0, r * acs[1] / ecs[1])
    py = py + ecs[0] * clamp(-dot(vec2(px, py), ecs), 0.0, r * acs[1] / ecs[1])
    return length(vec2(px, py)) * np.sign(px)

# 2D Pie (Circular Sector)
def sdPie2D(p, c, r):
    """2D Pie/Circular Sector. p is vec2, c is vec2(sin, cos) of half-angle, r is radius."""
    px = np.abs(p[..., 0])
    py = p[..., 1]
    l = length(p) - r
    m = length(vec2(px, py) - c * clamp(dot(vec2(px, py), c), 0.0, r)[..., None])
    return np.maximum(l, m * np.sign(c[1] * px - c[0] * py))

# 2D Cut Disk (circle with cut)
def sdCutDisk2D(p, r, h):
    """2D Cut Disk. p is vec2, r is radius, h is cut height."""
    w = np.sqrt(r * r - h * h)
    px = np.abs(p[..., 0])
    py = p[..., 1]
    s = np.maximum((h - r) * px * px + w * w * (h + r - 2.0 * py), h * px - w * py)
    cond1 = s < 0.0
    cond2 = px < w
    return np.where(cond1, length(p) - r,
           np.where(cond2, h - py, length(vec2(px, py) - vec2(w, h))))

# 2D Arc
def sdArc2D(p, sc, ra, rb):
    """2D Arc. p is vec2, sc is vec2(sin, cos), ra is radius, rb is thickness."""
    px = np.abs(p[..., 0])
    py = p[..., 1]
    cond = sc[1] * px > sc[0] * py
    return np.where(cond, 
                    length(vec2(px, py) - sc * ra) - rb,
                    np.abs(length(p) - ra) - rb)

# 2D Ring (Annulus)
def sdRing2D(p, r1, r2):
    """2D Ring/Annulus. p is vec2, r1 is inner radius, r2 is outer radius."""
    l = length(p)
    return np.maximum(r1 - l, l - r2)

# 2D Horseshoe
def sdHorseshoe2D(p, c, r, w):
    """2D Horseshoe. p is vec2, c is vec2(sin, cos), r is radius, w is vec2 thickness."""
    px = np.abs(p[..., 0])
    py = p[..., 1]
    l = length(p)
    px = np.where(py > 0.0, px, l) * np.sign(-c[0])
    py = np.where(py > 0.0, py, 0.0)
    px = px - c[0] * r
    py = py - c[1] * r
    q = vec2(length(vec2(np.maximum(px, 0.0), py)), 
             np.where(px < 0.0, py, length(vec2(px, py))))
    d = vec2(q[..., 0] - w[0], q[..., 1] - w[1])
    return np.minimum(np.maximum(d[..., 0], d[..., 1]), 0.0) + length(np.maximum(d, 0.0))

# 2D Vesica (Eye shape)
def sdVesica2D(p, r, d):
    """2D Vesica. p is vec2, r is radius, d is distance between circles."""
    px = np.abs(p[..., 0])
    py = p[..., 1]
    b = np.sqrt(r * r - d * d)
    cond = (py - b) * d > px * b
    return np.where(cond,
                    length(vec2(px, py) - vec2(0.0, b)) * np.sign(d),
                    length(vec2(px, py) - vec2(-d, 0.0)) - r)

# 2D Moon (Crescent)
def sdMoon2D(p, d, ra, rb):
    """2D Moon/Crescent. p is vec2, d is distance, ra/rb are radii."""
    py = np.abs(p[..., 1])
    a = (ra * ra - rb * rb + d * d) / (2.0 * d)
    b = np.sqrt(np.maximum(ra * ra - a * a, 0.0))
    cond = d * (p[..., 0] * b - py * a) > d * d * np.maximum(b - py, 0.0)
    return np.where(cond,
                    length(vec2(p[..., 0], py) - vec2(a, b)),
                    np.maximum(length(p) - ra, -(length(vec2(p[..., 0] - d, py)) - rb)))

# 2D Rounded Cross
def sdRoundedCross2D(p, h):
    """2D Rounded Cross. p is vec2, h is size."""
    k = 0.5 * (h + 1.0 / h)
    p = np.abs(p)
    cond_1 = p[..., 0] < 1.0
    cond_2 = p[..., 1] < p[..., 0] * (k - h) + h
    cond = cond_1 & cond_2
    v1 = p - vec2(1.0, k)
    d_true = k- np.sqrt(dot2(v1)) #circular arc
    v2a = p - vec2(0.0, h)
    v2b = p - vec2(1.0, 0.0)
    d2a = dot2(v2a) #top corner 
    d2b = dot2(v2b) #right corner
    d_false = np.sqrt(np.minimum(d2a, d2b))
    return np.where(cond, d_true, d_false)

# 2D Egg
def sdEgg2D(p, ra, rb):
    """2D Egg. p is vec2, ra/rb are radii."""
    k = np.sqrt(3.0)
    px = np.abs(p[..., 0])
    py = p[..., 1]
    r = ra - rb
    cond1 = py < 0.0
    cond2 = k * (px + r) < py
    d1 = length(vec2(px, py)) - r
    d2 = length(vec2(px, py - k * r))
    d3 = length(vec2(px + r, py)) - 2.0 * r
    return np.where(cond1, d1, np.where(cond2, d2, d3)) - rb

# 2D Heart
def sdHeart2D(p):
    """
    2D Heart shape. p is vec2.
    Corrected for NumPy broadcasting and grid evaluation.
    """
    px = np.abs(p[..., 0])
    py = p[..., 1]
    cond = px + py > 1.0
    #top lobes
    d_top = length(vec2(px - 0.25, py - 0.75)) - np.sqrt(2.0) / 4.0

    #bottom v shape
    d_bottom = length(vec2(px, py) - 0.5 * np.maximum(px + py, 0.0)[..., None]) * np.sign(px - py)

    return np.where(cond,d_top,d_bottom)

# 2D Cross
def sdCross2D(p, b, r):
    """2D Cross. p is vec2, b is vec2 size, r is rounding."""
    p = np.abs(p)
    cond = (p[..., 1] > p[..., 0])[..., None]
    p = np.where(cond, p[..., [1, 0]], p)
    #calculate distance
    q = p - b 
    k = np.maximum(q[..., 1], q[..., 0])

    mask = (k > 0.0)[..., None]

    w_else = vec2(b[1] - p[..., 0], -k)
    w = np.where(mask, q, w_else)
    return np.sign(k) * length(np.maximum(w, 0.0)) + r

# 2D Rounded X
def sdRoundedX2D(p, w, r):
    """2D Rounded X. p is vec2, w is width, r is rounding."""
    p = np.abs(p)
    s = np.minimum(p[..., 0] + p[..., 1], w) * 0.5
    return length(p - s[..., None]) - r

# 2D Polygon (from vertices)
def sdPolygon2D(p, v):
    """2D Polygon from N vertices. p is vec2, v is array of vec2 vertices."""
    N = v.shape[0]
    d = dot2(p - v[0])
    s = 1.0
    for i in range(N):
        j = (i + 1) % N
        e = v[j] - v[i]
        w = p - v[i]
        b = w - e * clamp(dot(w, e) / dot2(e), 0.0, 1.0)[..., None]
        d = np.minimum(d, dot2(b))
        cond = np.array([p[..., 1] >= v[i][1], 
                        p[..., 1] < v[j][1], 
                        e[0] * w[..., 1] > e[1] * w[..., 0]])
        s = np.where(np.all(cond, axis=0) | np.all(~cond, axis=0), -s, s)
    return s * np.sqrt(d)

# 2D Ellipse
def sdEllipse2D(p, ab):
    """2D Ellipse. p is vec2, ab is vec2 semi-axes."""
    px = np.abs(p[..., 0])
    py = np.abs(p[..., 1])
    cond = px > py
    px_new = np.where(cond, px, py)
    py_new = np.where(cond, py, px)
    ab_new = np.where(cond, ab, np.array([ab[1], ab[0]]))
    l = ab_new[1] * ab_new[1] - ab_new[0] * ab_new[0]
    m = ab_new[0] * px_new / l
    n = ab_new[1] * py_new / l
    m2 = m * m
    n2 = n * n
    c = (m2 + n2 - 1.0) / 3.0
    c3 = c * c * c
    d = c3 + m2 * n2
    q = d + m2 * n2
    g = m + m * n2
    co = np.where(d < 0.0,
                  (1.0 / 3.0) * np.arccos(safe_div(q, np.power(np.abs(c3), 0.5))) - (np.pi if c < 0.0 else 0.0),
                  (1.0 / 3.0) * np.log(safe_div(np.sqrt(q) + np.sqrt(d), g)))
    vec2_val = ab_new * vec2(np.cos(co), np.sin(co))
    return length(vec2(px_new, py_new) - vec2_val) * np.sign(py_new - vec2_val[..., 1])

# 2D Parabola
def sdParabola2D(p, k):
    """2D Parabola. p is vec2, k is curvature."""
    pos_x = np.abs(p[..., 0])
    pos_y = p[..., 1]
    ik = 1.0 / k
    p = ik * (pos_y - 0.5 * ik) / 3.0
    q = 0.25 * ik * ik * pos_x
    h = q * q - p * p * p
    r_hpos = cbrt(q + safe_sqrt(h))
    x_hpos = r_hpos + safe_div(p, r_hpos)
    r_hneg = safe_sqrt(p)
    term = safe_div(q, p * r_hneg)
    x_hneg = 2.0 * r_hneg * np.cos(safe_acos(term) / 3.0)
    x = np.where(h > 0.0, x_hpos,x_hneg)
    return length(vec2(pos_x - x, pos_y - k * x * x)) * np.sign(pos_x - x)

# 2D Parabola Segment
def sdParabolaSegment2D(p, wi, he):
    """2D Parabola Segment. p is vec2, wi is width, he is height."""
    px = np.abs(p[..., 0])
    py = p[..., 1]
    ik = wi * wi / he
    p_val = ik * (he - py - 0.5 * ik) / 3.0
    q = px * ik * ik * 0.25
    h = q * q - p_val * p_val* p_val
    r_hpos = cbrt(q + safe_sqrt(h))
    x_hpos = r_hpos + safe_div(p_val, r_hpos)
    r_hneg = safe_sqrt(p_val)
    x_hneg = 2.0 * r_hneg * np.cos(safe_acos(safe_div(q, p_val * r_hneg)) / 3.0)
    x = np.where(h > 0.0, x_hpos, x_hneg)
    x = np.minimum(x, wi)
    return length(vec2(px - x, py - he + x * x / ik)) * np.sign(px - x)
# 2D Bezier (Quadratic)
def sdBezier2D(p, A, B, C):
    """2D Quadratic Bezier. p is vec2, A/B/C are vec2 control points."""
    a = B - A
    b = A - 2.0 * B + C
    c = a * 2.0
    d = A - p
    kk = 1.0 / dot2(b)
    kx = kk * dot(a, b)
    ky = kk * (2.0 * dot2(a) + dot(d, b)) / 3.0
    kz = kk * dot(d, a)
    res = 0.0
    p1 = ky - kx * kx
    p3 = p1 * p1 * p1
    q = kx * (2.0 * kx * kx - 3.0 * ky) + kz
    h = q * q + 4.0 * p3
    h_pos = h >= 0.0
    z = np.where(h_pos[..., None],
                 np.sqrt(h[..., None]),
                 np.array([0.0, 0.0]))
    v = np.sign(q + h_pos * z[..., 0]) * np.power(np.abs(q + h_pos * z[..., 0]), 1.0/3.0)
    u = np.sign(q - h_pos * z[..., 0]) * np.power(np.abs(q - h_pos * z[..., 0]), 1.0/3.0)
    t = clamp(np.where(h_pos, (v + u) - kx, 
                      2.0 * np.cos(np.arctan2(np.sqrt(-h), q) / 3.0) * np.sqrt(-p1) - kx), 0.0, 1.0)
    q = d + (c + b * t[..., None]) * t[..., None]
    return length(q)

# 2D Blobbycross
def sdBlobbyCross2D(p, he):
    """2D Blobby Cross. p is vec2, he is size."""
    px = np.abs(p[..., 0])
    py = np.abs(p[..., 1])
    px = np.where(py > px, py, px)
    py = np.where(py > px, px, py)
    a = px - py
    b = px + py - 2.0 * he
    c1 = a * a
    c2 = b * b + 4.0 * he * he
    d = np.where(a > 0.0, c1, c2)
    return 0.5 * (np.sqrt(d) - he)

# 2D Tunnel
def sdTunnel2D(p, wh):
    """2D Tunnel. p is vec2, wh is vec2 size."""
    px = np.abs(p[..., 0])
    py = -p[..., 1]
    qx = px - wh[0]
    qy = py - wh[1]
    d1 = dot2(vec2(np.maximum(qx, 0.0), qy))
    len_p = length(vec2(px, py))
    qx_new = np.where(py > 0.0, qx, len_p - wh[0])
    d2 = dot2(vec2(qx_new, np.maximum(qy, 0.0)))
    d = safe_sqrt(np.minimum(d1, d2))
    return np.where(np.maximum(qx_new, qy) < 0.0, -d, d)

# 2D Stairs
def sdStairs2D(p, wh, n):
    """2D Stairs. p is vec2, wh is vec2 step size, n is number of steps."""
    ba = wh * n
    # We create the two box segments for the overall stairs bounds
    d = np.minimum(dot2(p - vec2(clamp(p[..., 0], 0.0, ba[0]), 0.0)),
                   dot2(p - vec2(ba[0], clamp(p[..., 1], 0.0, ba[1]))))
    s = np.sign(np.maximum(-p[..., 1], p[..., 0] - ba[0]))
    dia = length(wh)
    # Rotate 45 degrees to align steps
    p_x = (wh[0] * p[..., 0] + wh[1] * p[..., 1]) / dia
    p_y = (-wh[1] * p[..., 0] + wh[0] * p[..., 1]) / dia
    p = np.stack([p_x, p_y], axis=-1)
    id_ = clamp(np.round(p[..., 0] / dia), 0.0, n - 1.0)
    p[..., 0] = p[..., 0] - id_ * dia
    # Inverse Rotation
    p_x = (wh[0] * p[..., 0] - wh[1] * p[..., 1]) / dia
    p_y = (wh[1] * p[..., 0] + wh[0] * p[..., 1]) / dia
    p = np.stack([p_x, p_y], axis=-1)
    hh = wh[1] / 2.0
    p[..., 1] -= hh
    cond_s = p[..., 1] > hh * np.sign(p[..., 0])
    s = np.where(cond_s, 1.0, s)
    cond_p = (id_ < 0.5) | (p[..., 0] > 0.0)
    p = np.where(cond_p[..., None], p, -p)
    d = np.minimum(d, dot2(p - vec2(0.0, clamp(p[..., 1], -hh, hh))))
    d = np.minimum(d, dot2(p - vec2(clamp(p[..., 0], 0.0, wh[0]), hh)))
    return np.sqrt(np.maximum(d, 0.0)) * s

# 2D Quadratic Circle (approximate)
def sdQuadraticCircle2D(p):
    """2D Quadratic Circle approximation. p is vec2."""
    px = np.abs(p[..., 0])
    py = np.abs(p[..., 1])
    cond = py > px
    px_new = np.where(cond, py, px)
    py_new = np.where(cond, px, py)
    a = px_new - py_new
    b = px_new + py_new
    c = (2.0 * b - 1.0) / 3.0
    h = a * a + c * c * c
    h_sqrt = safe_sqrt(h)
    t_hpos = np.sign(h_sqrt - a) * cbrt(h_sqrt - a) - cbrt(h_sqrt + a)
    z = safe_sqrt(-c)
    v = safe_acos(safe_div(a, c * z)) / 3.0
    t_hneg = -z * (np.cos(v) + np.sin(v) * 1.732050808)
    t = np.where(h >= 0.0, t_hpos, t_hneg) * 0.5
    wx = -t + 0.75 - t * t - px_new
    wy = t + 0.75 - t * t - py_new
    return length(vec2(wx, wy)) * np.sign(a * a * 0.5 + b - 1.5)

# 2D Hyperbola
def sdHyperbola2D(p, k, he):
    """2D Hyperbola. p is vec2, k is curvature, he is height."""
    px = np.abs(p[..., 0])
    py = np.abs(p[..., 1])
    px = px - 2.0 * np.minimum(dot(vec2(px, py), vec2(k, 1.0)), 0.0) * k / dot2(vec2(k, 1.0))
    py = py - 2.0 * np.minimum(dot(vec2(px, py), vec2(k, 1.0)), 0.0) / dot2(vec2(k, 1.0))
    x2 = px * px / 16.0
    y2 = py * py / 16.0
    r = dot2(vec2(px * py, he * he * (1.0 - 2.0 * k) * px - k * py * py))
    q = (x2 - y2) * (x2 - y2)
    q = np.where(r != 0.0, (3.0 * y2 - x2) * x2 * x2 + r, q)
    return (length(vec2(px, py - he)) * np.sign(py - he) +
            np.sqrt(np.abs(q)) * np.sign(r) * 0.0625)

# 2D Cool S
def sdCoolS2D(p):
    """2D Cool S shape. p is vec2."""
    # Simplified version - full implementation is complex
    px = np.abs(p[..., 0])
    py = p[..., 1]
    return length(vec2(px, py)) - 1.0  # Placeholder

# 2D Regular N-Gon (Generic)
def sdNGon2D(p, r, n):
    """2D Regular N-sided polygon. p is vec2, r is radius, n is number of sides."""
    an = np.pi / n
    acs = vec2(np.cos(an), np.sin(an))
    bn = np.arctan2(np.abs(p[..., 0]), p[..., 1]) % (2.0 * an) - an
    px = length(p) * np.cos(bn)
    py = length(p) * np.abs(np.sin(bn))
    px = px - r * acs[0]
    py = py - r * acs[1]
    return length(np.maximum(vec2(px, py), 0.0)) + np.minimum(np.maximum(px, py), 0.0)

def opTx2D(p, mat, trans, sdf_func):
    """Apply 2D transformation matrix and translation."""
    p_transformed = np.dot(p, mat.T) - trans
    return sdf_func(p_transformed)
