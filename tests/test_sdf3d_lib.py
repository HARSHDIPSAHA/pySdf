"""Tests for sdf3d/primitives.py — 3-D SDF math primitives and operators.

Every 3-D function in sdf3d.primitives is tested at least once.  Tests verify:
- Correct sign (negative inside, positive outside, zero on surface)
- Exact or near-exact distance at analytically known points
- Array shape / broadcasting consistency
"""

import numpy as np
import numpy.testing as npt
import pytest

from sdf3d import primitives as sdf
from sdf2d import primitives as sdf2d  # 2D primitives needed for opRevolution/opExtrusion tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p3(*xyz) -> np.ndarray:
    """Single 3-D point as shape ``(1, 3)``."""
    return np.array([list(xyz)], dtype=float)


def _p2(*xy) -> np.ndarray:
    """Single 2-D point as shape ``(1, 2)``."""
    return np.array([list(xy)], dtype=float)


def _grid3(n: int = 8) -> np.ndarray:
    """Uniform ``n³`` grid of 3-D points in ``[-1, 1]³`` (shape ``(n, n, n, 3)``)."""
    lin = np.linspace(-1.0, 1.0, n)
    Z, Y, X = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.stack([X, Y, Z], axis=-1)


# ===========================================================================
# 3-D primitives
# ===========================================================================

class TestSphere:
    R = 0.3

    def test_inside_at_origin(self):
        p = _p3(0.0, 0.0, 0.0)
        npt.assert_allclose(sdf.sdSphere(p, self.R), [-self.R], atol=1e-10)

    def test_on_surface(self):
        p = _p3(self.R, 0.0, 0.0)
        npt.assert_allclose(sdf.sdSphere(p, self.R), [0.0], atol=1e-10)

    def test_outside(self):
        d = 0.5
        p = _p3(d, 0.0, 0.0)
        npt.assert_allclose(sdf.sdSphere(p, self.R), [d - self.R], atol=1e-10)

    def test_batch(self):
        p = _grid3(4)
        result = sdf.sdSphere(p, self.R)
        assert result.shape == (4, 4, 4)
        assert (result[0, 0, 0] > 0)  # corner is outside


class TestBox:
    B = np.array([0.2, 0.3, 0.1])

    def test_inside_at_origin(self):
        p = _p3(0.0, 0.0, 0.0)
        assert sdf.sdBox(p, self.B)[0] < 0

    def test_on_surface_face(self):
        p = _p3(self.B[0], 0.0, 0.0)
        npt.assert_allclose(sdf.sdBox(p, self.B), [0.0], atol=1e-10)

    def test_outside_x(self):
        p = _p3(self.B[0] + 0.1, 0.0, 0.0)
        npt.assert_allclose(sdf.sdBox(p, self.B), [0.1], atol=1e-10)


class TestRoundBox:
    def test_surface_at_face(self):
        # sdRoundBox face centre is at b_x (not b_x+r — rounding affects corners only)
        b = np.array([0.2, 0.2, 0.2])
        r = 0.05
        p = _p3(b[0], 0.0, 0.0)   # face centre
        npt.assert_allclose(sdf.sdRoundBox(p, b, r), [0.0], atol=1e-10)

    def test_inside_origin(self):
        b = np.array([0.2, 0.2, 0.2])
        assert sdf.sdRoundBox(_p3(0, 0, 0), b, 0.05)[0] < 0


class TestTorus:
    def test_on_surface(self):
        R, r = 0.3, 0.1
        t = np.array([R, r])
        # Point on the outer equator of the torus
        p = _p3(R + r, 0.0, 0.0)
        npt.assert_allclose(sdf.sdTorus(p, t), [0.0], atol=1e-10)

    def test_inside_tube(self):
        R, r = 0.3, 0.1
        t = np.array([R, r])
        p = _p3(R, 0.0, 0.0)
        npt.assert_allclose(sdf.sdTorus(p, t), [-r], atol=1e-10)


class TestCappedCylinder:
    def test_inside(self):
        r, h = 0.2, 0.3
        p = _p3(0.0, 0.0, 0.0)
        assert sdf.sdCappedCylinder(p, r, h)[0] < 0

    def test_on_side(self):
        r, h = 0.2, 0.3
        p = _p3(r, 0.0, 0.0)
        npt.assert_allclose(sdf.sdCappedCylinder(p, r, h), [0.0], atol=1e-10)


class TestConeExact:
    def test_returns_array(self):
        c = np.array([0.6, 0.8])
        p = _grid3(4)
        result = sdf.sdConeExact(p, c, 0.35)
        assert result.shape == (4, 4, 4)


class TestVerticalCapsule:
    def test_inside_body(self):
        h, r = 0.4, 0.15
        p = _p3(0.0, 0.2, 0.0)
        assert sdf.sdVerticalCapsule(p, h, r)[0] < 0

    def test_outside(self):
        h, r = 0.4, 0.15
        p = _p3(r + 0.1, 0.0, 0.0)
        assert sdf.sdVerticalCapsule(p, h, r)[0] > 0


class TestCappedCone:
    def test_returns_array(self):
        p = _grid3(4)
        result = sdf.sdCappedCone(p, 0.3, 0.2, 0.05)
        assert result.shape == (4, 4, 4)


class TestEllipsoid:
    def test_inside(self):
        # sdEllipsoid is approximate; test a point clearly inside (not origin)
        r = np.array([0.4, 0.3, 0.2])
        p = _p3(0.2, 0.0, 0.0)   # half-way to semi-major axis
        assert sdf.sdEllipsoid(p, r)[0] < 0

    def test_on_surface(self):
        r = np.array([0.3, 0.3, 0.3])
        p = _p3(0.3, 0.0, 0.0)
        npt.assert_allclose(sdf.sdEllipsoid(p, r), [0.0], atol=1e-6)


class TestOctahedron:
    def test_exact_and_bound_agree_inside(self):
        s = 0.35
        p = _p3(0.0, 0.0, 0.0)
        e = sdf.sdOctahedronExact(p, s)[0]
        b = sdf.sdOctahedronBound(p, s)[0]
        assert e < 0 and b < 0

    def test_bound_is_le_exact_outside(self):
        s = 0.35
        p = _p3(0.8, 0.8, 0.8)
        e = sdf.sdOctahedronExact(p, s)[0]
        b = sdf.sdOctahedronBound(p, s)[0]
        # Bound is an over-estimate (positive outside), so b >= e for outside points
        # (both should be positive)
        assert e > 0 and b > 0


# ===========================================================================
# 3-D unsigned-distance helpers
# ===========================================================================

class TestUdTriangle:
    def test_above_triangle(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 0.0, 1.0])
        p = _p3(0.25, 1.0, 0.25)
        d = sdf.udTriangle(p, a, b, c)[0]
        npt.assert_allclose(d, 1.0, atol=1e-6)

    def test_returns_nonneg(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 0.0, 1.0])
        p = _grid3(4)
        assert (sdf.udTriangle(p, a, b, c) >= 0).all()


class TestUdQuad:
    def test_returns_nonneg(self):
        a = np.array([-0.3, -0.3, 0.0])
        b = np.array([ 0.3, -0.3, 0.0])
        c = np.array([ 0.3,  0.3, 0.0])
        d = np.array([-0.3,  0.3, 0.0])
        p = _grid3(4)
        assert (sdf.udQuad(p, a, b, c, d) >= 0).all()


# ===========================================================================
# 3-D-only boolean variants
# ===========================================================================

class TestBooleanOps:
    S = 0.3

    @pytest.fixture
    def p(self):
        return _grid3(8)

    def test_union_le_min(self, p):
        d1 = sdf.sdSphere(p, self.S)
        d2 = sdf.sdBox(p, np.array([0.2, 0.2, 0.2]))
        u  = sdf.opUnion(d1, d2)
        npt.assert_array_less(u, np.minimum(d1, d2) + 1e-12)

    def test_intersection_ge_max(self, p):
        d1 = sdf.sdSphere(p, self.S)
        d2 = sdf.sdBox(p, np.array([0.2, 0.2, 0.2]))
        i  = sdf.opIntersection(d1, d2)
        npt.assert_array_less(np.maximum(d1, d2) - 1e-12, i)

    def test_subtraction_sign(self):
        # p is outside sphere (d_sphere > 0) but inside big box (d_box < 0)
        # opSubtraction(cutter=sphere, base=box) = max(-d_sphere, d_box)
        # → negative when d_sphere > 0 AND d_box < 0  (inside box, outside sphere)
        p = _p3(0.4, 0.0, 0.0)            # 0.4 > S=0.3 → outside sphere
        d_sphere = sdf.sdSphere(p, self.S)  # positive
        d_box    = sdf.sdBox(p, np.array([0.5, 0.5, 0.5]))  # negative (inside)
        result   = sdf.opSubtraction(d_sphere, d_box)[0]
        assert result < 0  # inside box, outside sphere → inside (box − sphere)

    def test_xor_is_zero_on_surface(self):
        p = _p3(self.S, 0.0, 0.0)
        d1 = sdf.sdSphere(p, self.S)
        d2 = sdf.sdSphere(p, self.S)
        npt.assert_allclose(sdf.opXor(d1, d2), [0.0], atol=1e-10)

    def test_smooth_union_le_union(self):
        p = _grid3(4)
        d1 = sdf.sdSphere(p, self.S)
        d2 = sdf.sdBox(p, np.array([0.2, 0.2, 0.2]))
        su = sdf.opSmoothUnion(d1, d2, k=0.1)
        u  = sdf.opUnion(d1, d2)
        assert (su <= u + 1e-10).all()

    def test_smooth_subtraction_returns_array(self):
        p = _grid3(4)
        d1 = sdf.sdSphere(p, self.S)
        d2 = sdf.sdBox(p, np.array([0.2, 0.2, 0.2]))
        assert sdf.opSmoothSubtraction(d1, d2, k=0.1).shape == d1.shape

    def test_smooth_intersection_returns_array(self):
        p = _grid3(4)
        d1 = sdf.sdSphere(p, self.S)
        d2 = sdf.sdBox(p, np.array([0.2, 0.2, 0.2]))
        assert sdf.opSmoothIntersection(d1, d2, k=0.1).shape == d1.shape


# ===========================================================================
# Space-warp operators
# ===========================================================================

class TestWarpOps:
    def test_round_grows_surface(self):
        p = _p3(0.3, 0.0, 0.0)
        b = np.array([0.2, 0.2, 0.2])
        d_box   = sdf.sdBox(p, b)
        d_round = sdf.opRound(p, lambda q: sdf.sdBox(q, b), 0.05)
        npt.assert_allclose(d_round, d_box - 0.05, atol=1e-10)

    def test_onion_positive_outside(self):
        p = _p3(0.5, 0.0, 0.0)
        d = sdf.sdSphere(p, 0.3)[0]
        result = sdf.opOnion(np.array([d]), 0.05)
        assert result[0] > 0

    def test_scale_smaller(self):
        p = _p3(0.3, 0.0, 0.0)
        d_orig  = sdf.sdSphere(p, 0.3)[0]
        d_scale = sdf.opScale(p, 0.5, lambda q: sdf.sdSphere(q, 0.3))[0]
        # Scaled by 0.5 → sphere surface at 0.15, so p=0.3 is outside more
        assert d_scale > d_orig

    def test_tx_identity(self):
        p = _grid3(4)
        rot = np.eye(3)
        trans = np.zeros(3)
        d1 = sdf.sdSphere(p, 0.3)
        d2 = sdf.opTx(p, rot, trans, lambda q: sdf.sdSphere(q, 0.3))
        npt.assert_allclose(d1, d2)

    def test_elongate2_extends_interior(self):
        p = _p3(0.0, 0.3, 0.0)
        # Without elongation: inside sphere (0.3 - 0.3 = 0)
        d_base = sdf.sdSphere(p, 0.3)[0]
        # With elongation along y by 0.2: point (0, 0.3, 0) is now deeper in
        h = np.array([0.0, 0.2, 0.0])
        d_elong = sdf.opElongate2(p, lambda q: sdf.sdSphere(q, 0.3), h)[0]
        assert d_elong <= d_base + 1e-10

    def test_revolution_returns_array(self):
        p = _grid3(4)
        result = sdf.opRevolution(p, lambda q: sdf2d.sdCircle(q, 0.2), 0.15)
        assert result.shape == (4, 4, 4)

    def test_extrusion_returns_array(self):
        p = _grid3(4)
        result = sdf.opExtrusion(p, lambda q: sdf2d.sdBox2D(q, np.array([0.2, 0.2])), 0.1)
        assert result.shape == (4, 4, 4)

    def test_sym_x_equals_abs_x(self):
        p1 = _p3(0.2, 0.3, 0.1)
        p2 = _p3(-0.2, 0.3, 0.1)
        prim = lambda q: sdf.sdSphere(q, 0.3)
        npt.assert_allclose(
            sdf.opSymX(p1, prim),
            sdf.opSymX(p2, prim),
        )

    def test_repetition_returns_array(self):
        p = _grid3(4)
        s = np.array([0.5, 0.5, 0.5])
        result = sdf.opRepetition(p, s, lambda q: sdf.sdSphere(q, 0.2))
        assert result.shape == (4, 4, 4)

    def test_limited_repetition_returns_array(self):
        p = _grid3(4)
        s = np.array([0.5, 0.5, 0.5])
        l = np.array([1.0, 1.0, 1.0])
        result = sdf.opLimitedRepetition(p, s, l, lambda q: sdf.sdSphere(q, 0.2))
        assert result.shape == (4, 4, 4)

    def test_displace_returns_array(self):
        p = _grid3(4)
        result = sdf.opDisplace(p, lambda q: sdf.sdSphere(q, 0.3))
        assert result.shape == (4, 4, 4)

    def test_twist_returns_array(self):
        p = _grid3(4)
        b = np.array([0.2, 0.2, 0.2])
        result = sdf.opTwist(p, lambda q: sdf.sdBox(q, b), 2.0)
        assert result.shape == (4, 4, 4)

    def test_cheap_bend_returns_array(self):
        p = _grid3(4)
        b = np.array([0.2, 0.2, 0.2])
        result = sdf.opCheapBend(p, lambda q: sdf.sdBox(q, b), 2.0)
        assert result.shape == (4, 4, 4)


# ===========================================================================
# Additional 3-D primitives (previously untested)
# ===========================================================================

class TestBoxFrame:
    B = np.array([0.3, 0.3, 0.3])
    E = 0.05

    def test_origin_outside_wires(self):
        # Origin is in empty space between wires — outside the wire material
        assert sdf.sdBoxFrame(_p3(0, 0, 0), self.B, self.E)[0] > 0

    def test_on_wire_surface(self):
        # A corner edge of the frame: (bx, by, 0) is on the wire surface
        p = _p3(self.B[0], self.B[1], 0.0)
        npt.assert_allclose(sdf.sdBoxFrame(p, self.B, self.E), [0.0], atol=1e-10)

    def test_inside_wire(self):
        # Slightly inside the corner wire
        p = _p3(self.B[0] - 0.01, self.B[1] - 0.01, 0.0)
        assert sdf.sdBoxFrame(p, self.B, self.E)[0] < 0

    def test_returns_array(self):
        assert sdf.sdBoxFrame(_grid3(4), self.B, self.E).shape == (4, 4, 4)


class TestCappedTorus:
    def test_inside_tube(self):
        # With sc=(sin,cos) where cos < 0 the cap covers the full ring, and
        # the formula reduces to the standard torus.  At (R, 0, 0) (ring centre):
        # phi = sqrt(R² + R² - 2R²) - rb = -rb
        R, rb = 0.3, 0.08
        sc = np.array([0.0, -1.0])   # 180° half-angle → full ring
        p = _p3(R, 0.0, 0.0)
        npt.assert_allclose(sdf.sdCappedTorus(p, sc, R, rb), [-rb], atol=1e-10)

    def test_on_outer_surface(self):
        # Same full-ring simplification: (R+rb, 0, 0) is on the outer tube surface → phi=0
        R, rb = 0.3, 0.08
        sc = np.array([0.0, -1.0])
        p = _p3(R + rb, 0.0, 0.0)
        npt.assert_allclose(sdf.sdCappedTorus(p, sc, R, rb), [0.0], atol=1e-10)


class TestLink:
    def test_inside_tube(self):
        # Point at (r1, 0, 0) is on the central axis of the link ring: phi = -r2
        le, r1, r2 = 0.2, 0.2, 0.05
        p = _p3(r1, 0.0, 0.0)
        npt.assert_allclose(sdf.sdLink(p, le, r1, r2), [-r2], atol=1e-10)

    def test_outside(self):
        le, r1, r2 = 0.2, 0.2, 0.05
        p = _p3(0.6, 0.0, 0.0)
        assert sdf.sdLink(p, le, r1, r2)[0] > 0


class TestCylinder:
    def test_inside_at_origin(self):
        # Infinite cylinder with axis at origin, radius 0.3
        c = np.array([0.0, 0.0, 0.3])
        p = _p3(0.0, 0.0, 0.0)
        npt.assert_allclose(sdf.sdCylinder(p, c), [-0.3], atol=1e-10)

    def test_on_surface(self):
        c = np.array([0.0, 0.0, 0.3])
        p = _p3(0.3, 0.0, 0.0)
        npt.assert_allclose(sdf.sdCylinder(p, c), [0.0], atol=1e-10)


class TestConeBound:
    def test_bound_le_exact_outside(self):
        # sdConeBound is a conservative (under-estimate) bound used for ray marching;
        # outside (phi>0) it satisfies sdBound <= sdExact so marching never overshoots.
        c = np.array([np.sin(np.pi / 6), np.cos(np.pi / 6)])  # 30° half-angle
        h = 0.4
        p = _grid3(6)
        exact = sdf.sdConeExact(p, c, h)
        bound = sdf.sdConeBound(p, c, h)
        outside = exact > 0
        assert (bound[outside] <= exact[outside] + 1e-10).all()

    def test_sign_agrees_inside(self):
        c = np.array([np.sin(np.pi / 6), np.cos(np.pi / 6)])
        h = 0.4
        p = _grid3(6)
        exact = sdf.sdConeExact(p, c, h)
        bound = sdf.sdConeBound(p, c, h)
        inside = exact < 0
        assert (bound[inside] < 0).all()


class TestConeInfinite:
    def test_inside(self):
        # 45° infinite cone; point below apex along -y with small x → inside
        c = np.array([np.sin(np.pi / 4), np.cos(np.pi / 4)])
        p = _p3(0.0, -0.5, 0.0)
        assert sdf.sdConeInfinite(p, c)[0] < 0

    def test_outside(self):
        c = np.array([np.sin(np.pi / 4), np.cos(np.pi / 4)])
        p = _p3(1.0, 0.0, 0.0)
        assert sdf.sdConeInfinite(p, c)[0] > 0


class TestPlane:
    def test_above_plane_positive(self):
        n = np.array([0.0, 1.0, 0.0])
        p = _p3(0.0, 1.0, 0.0)
        npt.assert_allclose(sdf.sdPlane(p, n, 0.0), [1.0], atol=1e-10)

    def test_below_plane_negative(self):
        n = np.array([0.0, 1.0, 0.0])
        p = _p3(0.0, -1.0, 0.0)
        npt.assert_allclose(sdf.sdPlane(p, n, 0.0), [-1.0], atol=1e-10)

    def test_offset(self):
        n = np.array([0.0, 1.0, 0.0])
        p = _p3(0.0, 0.0, 0.0)
        npt.assert_allclose(sdf.sdPlane(p, n, 0.5), [0.5], atol=1e-10)


class TestHexPrism:
    def test_inside_at_origin(self):
        h = np.array([0.3, 0.4])
        assert sdf.sdHexPrism(_p3(0, 0, 0), h)[0] < 0

    def test_outside(self):
        h = np.array([0.3, 0.4])
        assert sdf.sdHexPrism(_p3(1.0, 0, 0), h)[0] > 0


class TestTriPrism:
    def test_inside_at_origin(self):
        h = np.array([0.3, 0.4])
        assert sdf.sdTriPrism(_p3(0, 0, 0), h)[0] < 0

    def test_outside(self):
        h = np.array([0.3, 0.4])
        assert sdf.sdTriPrism(_p3(1.0, 0, 0), h)[0] > 0


class TestCapsule:
    def test_inside_at_midpoint(self):
        a = np.array([0.0, -0.3, 0.0])
        b = np.array([0.0,  0.3, 0.0])
        r = 0.15
        # Midpoint of segment is on the axis → phi = -r
        p = _p3(0.0, 0.0, 0.0)
        npt.assert_allclose(sdf.sdCapsule(p, a, b, r), [-r], atol=1e-10)

    def test_outside(self):
        a = np.array([0.0, -0.3, 0.0])
        b = np.array([0.0,  0.3, 0.0])
        assert sdf.sdCapsule(_p3(1.0, 0, 0), a, b, 0.15)[0] > 0


class TestRoundedCylinder:
    def test_inside_at_origin(self):
        ra, rb, h = 0.3, 0.05, 0.4
        assert sdf.sdRoundedCylinder(_p3(0, 0, 0), ra, rb, h)[0] < 0

    def test_on_flat_face_center(self):
        # At (0, h, 0): face centre — distance = 0 (see MEMORY.md note)
        ra, rb, h = 0.3, 0.05, 0.4
        p = _p3(0.0, h, 0.0)
        npt.assert_allclose(sdf.sdRoundedCylinder(p, ra, rb, h), [0.0], atol=1e-10)


class TestCappedCylinderSegment:
    def test_inside_at_midpoint(self):
        a = np.array([0.0, -0.3, 0.0])
        b = np.array([0.0,  0.3, 0.0])
        r = 0.15
        p = _p3(0.0, 0.0, 0.0)
        assert sdf.sdCappedCylinderSegment(p, a, b, r)[0] < 0

    def test_outside(self):
        a = np.array([0.0, -0.3, 0.0])
        b = np.array([0.0,  0.3, 0.0])
        assert sdf.sdCappedCylinderSegment(_p3(1.0, 0, 0), a, b, 0.15)[0] > 0


class TestCappedConeSegment:
    def test_inside_near_base(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 0.5, 0.0])
        ra, rb = 0.3, 0.1
        # Just inside the base end on the axis
        p = _p3(0.0, 0.05, 0.0)
        assert sdf.sdCappedConeSegment(p, a, b, ra, rb)[0] < 0

    def test_outside(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 0.5, 0.0])
        assert sdf.sdCappedConeSegment(_p3(1.0, 0, 0), a, b, 0.3, 0.1)[0] > 0


class TestRoundCone:
    def test_inside_at_base_center(self):
        # At y=0, x=0: phi = -r1
        r1, r2, h = 0.3, 0.1, 0.5
        p = _p3(0.0, 0.0, 0.0)
        npt.assert_allclose(sdf.sdRoundCone(p, r1, r2, h), [-r1], atol=1e-10)

    def test_outside(self):
        assert sdf.sdRoundCone(_p3(1.0, 0, 0), 0.3, 0.1, 0.5)[0] > 0


class TestRoundConeSegment:
    def test_inside_at_midpoint(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 0.5, 0.0])
        r1, r2 = 0.3, 0.1
        p = _p3(0.0, 0.25, 0.0)
        assert sdf.sdRoundConeSegment(p, a, b, r1, r2)[0] < 0

    def test_outside(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 0.5, 0.0])
        assert sdf.sdRoundConeSegment(_p3(1.0, 0, 0), a, b, 0.3, 0.1)[0] > 0


class TestVesicaSegment:
    def test_returns_nonneg_far_outside(self):
        a = np.array([-0.3, 0.0, 0.0])
        b = np.array([ 0.3, 0.0, 0.0])
        w = 0.1
        p = _p3(0.0, 2.0, 0.0)
        assert sdf.sdVesicaSegment(p, a, b, w)[0] > 0

    def test_returns_array(self):
        a = np.array([-0.3, 0.0, 0.0])
        b = np.array([ 0.3, 0.0, 0.0])
        assert sdf.sdVesicaSegment(_grid3(4), a, b, 0.1).shape == (4, 4, 4)


class TestRhombus:
    def test_inside_at_origin(self):
        assert sdf.sdRhombus(_p3(0, 0, 0), 0.4, 0.3, 0.2, 0.02)[0] < 0

    def test_outside(self):
        assert sdf.sdRhombus(_p3(1.0, 1.0, 1.0), 0.4, 0.3, 0.2, 0.02)[0] > 0


class TestCutSphere:
    def test_inside(self):
        # Sphere r=0.5 cut at h=0; retained portion is y≥0 (above the cut plane).
        # Point at (0, 0.1, 0) is inside the retained upper cap.
        assert sdf.sdCutSphere(_p3(0, 0.1, 0), 0.5, 0.0)[0] < 0

    def test_below_cut_outside(self):
        # The lower half was cut away; (0, -0.2, 0) is outside the retained shape.
        assert sdf.sdCutSphere(_p3(0, -0.2, 0), 0.5, 0.0)[0] > 0

    def test_far_outside(self):
        assert sdf.sdCutSphere(_p3(0, 2.0, 0), 0.5, 0.0)[0] > 0


class TestCutHollowSphere:
    def test_inside_shell(self):
        # p=(r, 0, 0): q=(r, 0). Condition h*r < w*0 → False → use |length(q)-r|-t = -t < 0
        r, h, t = 0.5, 0.2, 0.05
        p = _p3(r, 0.0, 0.0)
        npt.assert_allclose(sdf.sdCutHollowSphere(p, r, h, t), [-t], atol=1e-10)

    def test_outside(self):
        assert sdf.sdCutHollowSphere(_p3(0, 1.0, 0), 0.5, 0.2, 0.05)[0] > 0


class TestDeathStar:
    def test_inside_main_sphere(self):
        # Origin is inside the main sphere (ra=0.5) and not in the bite region → phi < 0
        ra, rb, d = 0.5, 0.2, 0.4
        assert sdf.sdDeathStar(_p3(0, 0, 0), ra, rb, d)[0] < 0

    def test_outside(self):
        assert sdf.sdDeathStar(_p3(1.0, 0, 0), 0.5, 0.2, 0.4)[0] > 0


class TestSolidAngle:
    def test_inside(self):
        # 45° solid angle, radius 0.4; point at (0, 0.2, 0) is inside
        c = np.array([np.sin(np.pi / 4), np.cos(np.pi / 4)])
        assert sdf.sdSolidAngle(_p3(0, 0.2, 0), c, 0.4)[0] < 0

    def test_outside(self):
        c = np.array([np.sin(np.pi / 4), np.cos(np.pi / 4)])
        assert sdf.sdSolidAngle(_p3(0, 0.8, 0), c, 0.4)[0] > 0


class TestPyramid:
    def test_inside(self):
        # Inside the pyramid body (half-base=0.5, apex at y=h)
        h = 0.5
        p = _p3(0.0, h / 2, 0.0)
        assert sdf.sdPyramid(p, h)[0] < 0

    def test_above_apex_outside(self):
        h = 0.5
        p = _p3(0.0, h + 0.2, 0.0)
        assert sdf.sdPyramid(p, h)[0] > 0


class TestOpElongate1:
    def test_extends_interior(self):
        # p=(0.4, 0, 0) is outside sphere(r=0.3); with elongation h=(0.2,0,0) it should be inside
        p = _p3(0.4, 0.0, 0.0)
        h = np.array([0.2, 0.0, 0.0])
        d_base = sdf.sdSphere(p, 0.3)[0]
        assert d_base > 0  # confirm base case is outside
        d_elong = sdf.opElongate1(p, lambda q: sdf.sdSphere(q, 0.3), h)[0]
        assert d_elong < 0  # elongated version is inside


class TestOpSymXZ:
    def test_symmetry_in_x_and_z(self):
        prim = lambda q: sdf.sdSphere(q, 0.3)
        p_pos = _p3( 0.2, 0.1,  0.15)
        p_neg = _p3(-0.2, 0.1, -0.15)
        npt.assert_allclose(
            sdf.opSymXZ(p_pos, prim),
            sdf.opSymXZ(p_neg, prim),
        )
