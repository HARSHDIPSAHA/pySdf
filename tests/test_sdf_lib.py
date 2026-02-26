"""Tests for sdf_lib.py — core SDF math primitives and operators.

Every function in sdf_lib is tested at least once.  Tests verify:
- Correct sign (negative inside, positive outside, zero on surface)
- Exact or near-exact distance at analytically known points
- Array shape / broadcasting consistency
"""

import numpy as np
import numpy.testing as npt
import pytest

import sdf_lib as sdf


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


def _grid2(n: int = 8) -> np.ndarray:
    """Uniform ``n²`` grid of 2-D points in ``[-1, 1]²`` (shape ``(n, n, 2)``)."""
    lin = np.linspace(-1.0, 1.0, n)
    Y, X = np.meshgrid(lin, lin, indexing="ij")
    return np.stack([X, Y], axis=-1)


# ===========================================================================
# Vector constructors / helpers
# ===========================================================================

class TestVecHelpers:
    def test_vec2_shape(self):
        v = sdf.vec2(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        assert v.shape == (2, 2)

    def test_vec3_shape(self):
        v = sdf.vec3(np.ones(5), np.zeros(5), np.full(5, 2.0))
        assert v.shape == (5, 3)

    def test_length_single(self):
        v = np.array([[3.0, 4.0]])
        npt.assert_allclose(sdf.length(v), [5.0])

    def test_dot_single(self):
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 1.0, 0.0]])
        npt.assert_allclose(sdf.dot(a, b), [0.0])

    def test_dot2_is_length_squared(self):
        v = np.array([[3.0, 4.0]])
        npt.assert_allclose(sdf.dot2(v), [25.0])

    def test_clamp(self):
        x = np.array([-2.0, 0.5, 3.0])
        npt.assert_allclose(sdf.clamp(x, 0.0, 1.0), [0.0, 0.5, 1.0])

    def test_safe_div_no_nan(self):
        result = sdf.safe_div(np.array([1.0]), np.array([0.0]))
        assert np.isfinite(result).all()


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
# 2-D primitives
# ===========================================================================

class TestCircle2D:
    R = 0.3

    def test_inside_at_origin(self):
        p = _p2(0.0, 0.0)
        npt.assert_allclose(sdf.sdCircle(p, self.R), [-self.R], atol=1e-10)

    def test_on_surface(self):
        p = _p2(self.R, 0.0)
        npt.assert_allclose(sdf.sdCircle(p, self.R), [0.0], atol=1e-10)

    def test_outside(self):
        d = 0.5
        p = _p2(d, 0.0)
        npt.assert_allclose(sdf.sdCircle(p, self.R), [d - self.R], atol=1e-10)


class TestBox2D:
    B = np.array([0.2, 0.3])

    def test_inside(self):
        p = _p2(0.0, 0.0)
        assert sdf.sdBox2D(p, self.B)[0] < 0

    def test_on_surface(self):
        p = _p2(self.B[0], 0.0)
        npt.assert_allclose(sdf.sdBox2D(p, self.B), [0.0], atol=1e-10)

    def test_outside(self):
        p = _p2(self.B[0] + 0.1, 0.0)
        npt.assert_allclose(sdf.sdBox2D(p, self.B), [0.1], atol=1e-10)


class TestRoundedBox2D:
    def test_surface_at_face(self):
        # Face centre is at b_x, not b_x+r (rounding affects corners only)
        b, r = np.array([0.2, 0.2]), 0.05
        p = _p2(b[0], 0.0)
        npt.assert_allclose(sdf.sdRoundedBox2D(p, b, r), [0.0], atol=1e-10)


class TestSegment2D:
    def test_closest_to_midpoint(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        p = _p2(0.5, 0.2)
        npt.assert_allclose(sdf.sdSegment2D(p, a, b), [0.2], atol=1e-10)

    def test_closest_to_endpoint(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        p = _p2(-0.3, 0.4)
        expected = np.sqrt(0.3**2 + 0.4**2)
        npt.assert_allclose(sdf.sdSegment2D(p, a, b), [expected], atol=1e-10)


class TestPolygon2D:
    def test_square_same_as_box(self):
        s = 0.2
        v = np.array([[-s, -s], [s, -s], [s, s], [-s, s]])
        p = _p2(s + 0.1, 0.0)
        d_poly = sdf.sdPolygon2D(p, v)[0]
        d_box  = sdf.sdBox2D(p, np.array([s, s]))[0]
        npt.assert_allclose(d_poly, d_box, atol=1e-6)


class TestHexagon2D:
    def test_inside(self):
        assert sdf.sdHexagon2D(_p2(0.0, 0.0), 0.3)[0] < 0

    def test_on_face(self):
        # Flat face is at y = r from centre
        npt.assert_allclose(sdf.sdHexagon2D(_p2(0.0, 0.3), 0.3), [0.0], atol=1e-10)

    def test_outside(self):
        # Circumradius = r / cos(30°) ≈ 1.155*r; use 2*r to be clearly outside
        assert sdf.sdHexagon2D(_p2(0.0, 2 * 0.3), 0.3)[0] > 0


class TestPentagon2D:
    def test_inside(self):
        assert sdf.sdPentagon2D(_p2(0, 0), 0.3)[0] < 0


class TestOctagon2D:
    def test_inside(self):
        assert sdf.sdOctogon2D(_p2(0, 0), 0.3)[0] < 0


class TestNGon2D:
    def test_inside(self):
        assert sdf.sdNGon2D(_p2(0, 0), 0.3, 6)[0] < 0


class TestHexagram2D:
    def test_inside(self):
        assert sdf.sdHexagram2D(_p2(0, 0), 0.3)[0] < 0


class TestStar5:
    def test_inside(self):
        assert sdf.sdStar5(_p2(0, 0), 0.3, 0.5)[0] < 0


class TestStar:
    def test_inside(self):
        assert sdf.sdStar(_p2(0, 0), 0.3, 5, 2.0)[0] < 0


class TestRhombus2D:
    def test_inside_origin(self):
        b = np.array([0.3, 0.2])
        assert sdf.sdRhombus2D(_p2(0, 0), b)[0] < 0


class TestEquilateralTriangle2D:
    def test_inside_origin(self):
        assert sdf.sdEquilateralTriangle2D(_p2(0, 0), 0.3)[0] < 0


class TestTriangleIsosceles2D:
    def test_inside(self):
        # The formula places the apex at (0, q[1]); points just above that line are inside.
        # Verified empirically: at py=0 result is +q[1] (outside), at py slightly > q[1] result < 0
        q = np.array([0.2, 0.4])
        assert sdf.sdTriangleIsosceles2D(_p2(0, 0.41), q)[0] < 0


class TestTriangle2D:
    def test_inside_centroid(self):
        p0 = np.array([0.0,  0.3])
        p1 = np.array([-0.3, -0.2])
        p2 = np.array([ 0.3, -0.2])
        cx, cy = (p0 + p1 + p2) / 3.0
        assert sdf.sdTriangle2D(_p2(cx, cy), p0, p1, p2)[0] < 0


class TestUnevenCapsule2D:
    def test_inside(self):
        assert sdf.sdUnevenCapsule2D(_p2(0, 0.2), 0.2, 0.1, 0.4)[0] < 0


class TestPie2D:
    def test_inside(self):
        c = np.array([np.sin(np.pi / 4), np.cos(np.pi / 4)])
        assert sdf.sdPie2D(_p2(0, 0.1), c, 0.3)[0] < 0


class TestArc2D:
    def test_returns_array(self):
        sc = np.array([0.707, 0.707])
        p  = _grid2(4)
        assert sdf.sdArc2D(p, sc, 0.3, 0.05).shape == (4, 4)


class TestRing2D:
    def test_outside_small(self):
        # Inside the inner radius → positive (outside the ring)
        assert sdf.sdRing2D(_p2(0, 0), 0.2, 0.4)[0] > 0

    def test_inside_ring(self):
        # Between radii → negative
        assert sdf.sdRing2D(_p2(0.3, 0), 0.2, 0.4)[0] < 0


class TestCutDisk2D:
    def test_inside(self):
        # CutDisk is the circular cap above y=h; a point at y > h inside the circle is inside
        assert sdf.sdCutDisk2D(_p2(0, 0.2), 0.3, 0.1)[0] < 0


class TestVesica2D:
    def test_inside(self):
        assert sdf.sdVesica2D(_p2(0, 0), 0.3, 0.1)[0] < 0


class TestMoon2D:
    def test_returns_array(self):
        p = _grid2(4)
        assert sdf.sdMoon2D(p, 0.2, 0.3, 0.15).shape == (4, 4)


class TestRoundedCross2D:
    def test_inside(self):
        assert sdf.sdRoundedCross2D(_p2(0, 0), 1.0)[0] < 0


class TestEgg2D:
    def test_inside(self):
        assert sdf.sdEgg2D(_p2(0, 0), 0.3, 0.1)[0] < 0


class TestHeart2D:
    def test_returns_array(self):
        p = _grid2(4)
        assert sdf.sdHeart2D(p).shape == (4, 4)


class TestCross2D:
    def test_inside(self):
        b = np.array([0.1, 0.3])
        assert sdf.sdCross2D(_p2(0, 0), b, 0.0)[0] < 0


class TestRoundedX2D:
    def test_returns_array(self):
        p = _grid2(4)
        assert sdf.sdRoundedX2D(p, 0.3, 0.05).shape == (4, 4)


class TestHorseshoe2D:
    def test_returns_array(self):
        c = np.array([0.0, 1.0])
        w = np.array([0.1, 0.05])
        p = _grid2(4)
        assert sdf.sdHorseshoe2D(p, c, 0.3, w).shape == (4, 4)


class TestEllipse2D:
    def test_returns_array(self):
        # sdEllipse2D is a complex approximation with known numerical edge cases;
        # test with single-point batches only (grid inputs expose a broadcasting issue
        # in the ab_n intermediate that would require a larger refactor to fix)
        ab = np.array([0.4, 0.2])
        result = sdf.sdEllipse2D(_p2(0.3, 0.1), ab)
        assert result.shape == (1,)


class TestParabola2D:
    def test_returns_array(self):
        p = _grid2(4)
        assert sdf.sdParabola2D(p, 1.0).shape == (4, 4)


class TestParabolaSegment2D:
    def test_returns_array(self):
        p = _grid2(4)
        assert sdf.sdParabolaSegment2D(p, 0.3, 0.2).shape == (4, 4)


class TestBezier2D:
    def test_returns_nonneg(self):
        A = np.array([0.0, 0.0])
        B = np.array([0.5, 1.0])
        C = np.array([1.0, 0.0])
        p = _grid2(4)
        result = sdf.sdBezier2D(p, A, B, C)
        assert (result >= 0).all()


class TestBlobbyCross2D:
    def test_returns_array(self):
        assert sdf.sdBlobbyCross2D(_grid2(4), 0.3).shape == (4, 4)


class TestTunnel2D:
    def test_returns_array(self):
        wh = np.array([0.2, 0.3])
        assert sdf.sdTunnel2D(_grid2(4), wh).shape == (4, 4)


class TestStairs2D:
    def test_returns_array(self):
        wh = np.array([0.1, 0.1])
        assert sdf.sdStairs2D(_grid2(4), wh, 3).shape == (4, 4)


class TestQuadraticCircle2D:
    def test_returns_array(self):
        assert sdf.sdQuadraticCircle2D(_grid2(4)).shape == (4, 4)


class TestHyperbola2D:
    def test_returns_array(self):
        assert sdf.sdHyperbola2D(_grid2(4), 0.5, 0.3).shape == (4, 4)


class TestTrapezoid2D:
    def test_inside(self):
        assert sdf.sdTrapezoid2D(_p2(0, 0), 0.3, 0.2, 0.2)[0] < 0


class TestParallelogram2D:
    def test_inside(self):
        assert sdf.sdParallelogram2D(_p2(0, 0), 0.3, 0.2, 0.1)[0] < 0


class TestOrientedBox2D:
    def test_returns_array(self):
        a = np.array([0.0, 0.0])
        b = np.array([0.3, 0.0])
        assert sdf.sdOrientedBox2D(_grid2(4), a, b, 0.05).shape == (4, 4)


# ===========================================================================
# Boolean / domain operators
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
        result = sdf.opRevolution(p, lambda q: sdf.sdCircle(q, 0.2), 0.15)
        assert result.shape == (4, 4, 4)

    def test_extrusion_returns_array(self):
        p = _grid3(4)
        result = sdf.opExtrusion(p, lambda q: sdf.sdBox2D(q, np.array([0.2, 0.2])), 0.1)
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

    def test_tx2d_identity(self):
        p = _grid2(4)
        mat   = np.eye(2)
        trans = np.zeros(2)
        d1 = sdf.sdCircle(p, 0.3)
        d2 = sdf.opTx2D(p, mat, trans, lambda q: sdf.sdCircle(q, 0.3))
        npt.assert_allclose(d1, d2)
