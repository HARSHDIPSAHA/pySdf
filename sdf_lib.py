import numpy as np


def vec2(x, y):
    return np.stack([x, y], axis=-1)


def vec3(x, y, z):
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
    a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0)
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
    d = length(q - c * np.maximum(dot(q, c), 0.0))
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
    cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot2(k2), 0.0, 1.0)
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
    m = length(q - c * clamp(dot(q, c), 0.0, ra))
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
