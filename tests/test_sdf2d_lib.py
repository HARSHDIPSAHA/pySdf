"""Tests for sdf2d/primitives.py — 2-D SDF math primitives and operators.

Every 2-D function in sdf2d.primitives is tested at least once.  Tests verify:
- Correct sign (negative inside, positive outside, zero on surface)
- Exact or near-exact distance at analytically known points
- Array shape / broadcasting consistency
"""

import numpy as np
import numpy.testing as npt
import pytest

from sdf2d import primitives as sdf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p2(*xy) -> np.ndarray:
    """Single 2-D point as shape ``(1, 2)``."""
    return np.array([list(xy)], dtype=float)


def _grid2(n: int = 8) -> np.ndarray:
    """Uniform ``n²`` grid of 2-D points in ``[-1, 1]²`` (shape ``(n, n, 2)``)."""
    lin = np.linspace(-1.0, 1.0, n)
    Y, X = np.meshgrid(lin, lin, indexing="ij")
    return np.stack([X, Y], axis=-1)


# ===========================================================================
# Shared helpers (re-exported from _sdf_common)
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
        assert sdf.sdOctagon2D(_p2(0, 0), 0.3)[0] < 0


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
        # IQ formula: apex at (0,0), base at y=q[1]; interior is between them
        q = np.array([0.2, 0.4])
        assert sdf.sdTriangleIsosceles2D(_p2(0, 0.2), q)[0] < 0


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
# 2-D transform operator
# ===========================================================================

class TestTx2D:
    def test_identity(self):
        p = _grid2(4)
        mat   = np.eye(2)
        trans = np.zeros(2)
        d1 = sdf.sdCircle(p, 0.3)
        d2 = sdf.opTx2D(p, mat, trans, lambda q: sdf.sdCircle(q, 0.3))
        npt.assert_allclose(d1, d2)
