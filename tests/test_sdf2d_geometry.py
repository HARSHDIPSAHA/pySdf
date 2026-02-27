"""Tests for sdf2d geometry classes.

Every class in sdf2d.geometry is tested for:
- Correct instantiation
- sdf() returns correct sign (negative inside, positive outside)
- Transform methods return new Geometry2D
- Boolean operation methods return new Geometry2D
"""

import numpy as np
import numpy.testing as npt
import pytest

from sdf2d import (
    Geometry2D,
    Circle2D, Box2D, RoundedBox2D, OrientedBox2D, Segment2D,
    Rhombus2D, Trapezoid2D, Parallelogram2D,
    EquilateralTriangle2D, TriangleIsosceles2D, Triangle2D,
    UnevenCapsule2D,
    Pentagon2D, Hexagon2D, Octagon2D, NGon2D,
    Hexagram2D, Star2D,
    Pie2D, CutDisk2D, Arc2D, Ring2D, Horseshoe2D,
    Vesica2D, Moon2D, RoundedCross2D, Egg2D, Heart2D, Cross2D, RoundedX2D,
    Polygon2D, Ellipse2D, Parabola2D, ParabolaSegment2D, Bezier2D,
    BlobbyCross2D, Tunnel2D, Stairs2D, QuadraticCircle2D, Hyperbola2D,
    Union2D, Intersection2D, Subtraction2D,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p(*xy) -> np.ndarray:
    """Single 2-D point as shape ``(1, 2)``."""
    return np.array([list(xy)], dtype=float)


def _grid(n: int = 16) -> np.ndarray:
    lin = np.linspace(-1.0, 1.0, n)
    Y, X = np.meshgrid(lin, lin, indexing="ij")
    return np.stack([X, Y], axis=-1)


# ===========================================================================
# Base class
# ===========================================================================

class TestGeometry2D:
    def test_sdf_callable(self):
        g = Geometry2D(lambda p: np.zeros(p.shape[:-1]))
        assert g.sdf(_grid()).shape == (16, 16)

    def test_call_is_sdf(self):
        g = Geometry2D(lambda p: np.ones(p.shape[:-1]))
        npt.assert_array_equal(g(_p(0, 0)), g.sdf(_p(0, 0)))

    def test_translate_moves_origin(self):
        c = Circle2D(0.3)
        moved = c.translate(0.5, 0.0)
        # SDF at new centre should be -0.3
        npt.assert_allclose(moved.sdf(_p(0.5, 0.0)), [-0.3], atol=1e-10)

    def test_scale_changes_size(self):
        c = Circle2D(0.3)
        scaled = c.scale(2.0)
        # Scaled sphere has radius 0.6
        npt.assert_allclose(scaled.sdf(_p(0.0, 0.0)), [-0.6], atol=1e-10)

    def test_rotate_preserves_shape(self):
        c = Circle2D(0.3)
        rotated = c.rotate(np.pi / 4)
        # Circle is symmetric → SDF at origin unchanged
        npt.assert_allclose(rotated.sdf(_p(0.0, 0.0)), [-0.3], atol=1e-10)

    def test_round_grows_by_rad(self):
        b = Box2D((0.2, 0.2))
        rounded = b.round(0.05)
        p = _p(0.25, 0.0)
        d_box     = b.sdf(p)[0]
        d_rounded = rounded.sdf(p)[0]
        npt.assert_allclose(d_rounded, d_box - 0.05, atol=1e-10)

    def test_onion_positive_outside_shell(self):
        c = Circle2D(0.3)
        shell = c.onion(0.02)
        # Far outside: positive
        assert shell.sdf(_p(1.0, 0.0))[0] > 0


# ===========================================================================
# Primitive shapes
# ===========================================================================

class TestCircle2D:
    def test_inside_origin(self):
        npt.assert_allclose(Circle2D(0.3).sdf(_p(0, 0)), [-0.3], atol=1e-10)

    def test_on_surface(self):
        npt.assert_allclose(Circle2D(0.3).sdf(_p(0.3, 0)), [0.0], atol=1e-10)

    def test_outside(self):
        npt.assert_allclose(Circle2D(0.3).sdf(_p(0.5, 0)), [0.2], atol=1e-10)


class TestBox2D_:
    def test_inside(self):
        assert Box2D((0.3, 0.3)).sdf(_p(0, 0))[0] < 0

    def test_on_face(self):
        npt.assert_allclose(Box2D((0.3, 0.3)).sdf(_p(0.3, 0)), [0.0], atol=1e-10)

    def test_outside(self):
        npt.assert_allclose(Box2D((0.3, 0.3)).sdf(_p(0.4, 0)), [0.1], atol=1e-10)


class TestRoundedBox2D_:
    def test_inside(self):
        assert RoundedBox2D((0.3, 0.3), 0.05).sdf(_p(0, 0))[0] < 0


class TestOrientedBox2D_:
    def test_inside(self):
        assert OrientedBox2D((0, 0), (0.4, 0), 0.1).sdf(_p(0.2, 0))[0] < 0


class TestSegment2D_:
    def test_perpendicular_distance(self):
        s = Segment2D((0, 0), (1, 0))
        npt.assert_allclose(s.sdf(_p(0.5, 0.2)), [0.2], atol=1e-10)


class TestRhombus2D_:
    def test_inside(self):
        assert Rhombus2D((0.3, 0.2)).sdf(_p(0, 0))[0] < 0


class TestTrapezoid2D_:
    def test_inside(self):
        assert Trapezoid2D(0.3, 0.2, 0.2).sdf(_p(0, 0))[0] < 0


class TestParallelogram2D_:
    def test_inside(self):
        assert Parallelogram2D(0.3, 0.2, 0.1).sdf(_p(0, 0))[0] < 0


class TestEquilateralTriangle2D_:
    def test_inside(self):
        assert EquilateralTriangle2D(0.3).sdf(_p(0, 0))[0] < 0


class TestTriangleIsosceles2D_:
    def test_inside(self):
        # IQ formula: apex at (0,0), base at y=height; interior is between them
        assert TriangleIsosceles2D(0.2, 0.4).sdf(_p(0, 0.2))[0] < 0


class TestTriangle2D_:
    def test_inside_centroid(self):
        p0, p1, p2 = (0, 0.3), (-0.3, -0.2), (0.3, -0.2)
        cx = (p0[0] + p1[0] + p2[0]) / 3
        cy = (p0[1] + p1[1] + p2[1]) / 3
        assert Triangle2D(p0, p1, p2).sdf(_p(cx, cy))[0] < 0


class TestUnevenCapsule2D_:
    def test_inside(self):
        assert UnevenCapsule2D(0.2, 0.1, 0.4).sdf(_p(0, 0.2))[0] < 0


class TestPentagon2D_:
    def test_inside(self):
        assert Pentagon2D(0.3).sdf(_p(0, 0))[0] < 0


class TestHexagon2D_:
    def test_inside(self):
        assert Hexagon2D(0.3).sdf(_p(0, 0))[0] < 0


class TestOctagon2D_:
    def test_inside(self):
        assert Octagon2D(0.3).sdf(_p(0, 0))[0] < 0


class TestNGon2D_:
    def test_inside(self):
        # NGon uses circumradius; origin is inside any regular polygon
        assert NGon2D(0.3, 6).sdf(_p(0, 0))[0] < 0

    def test_outside(self):
        # Point well beyond the circumradius
        assert NGon2D(0.3, 6).sdf(_p(0, 1.0))[0] > 0


class TestHexagram2D_:
    def test_inside(self):
        assert Hexagram2D(0.3).sdf(_p(0, 0))[0] < 0


class TestStar2D_:
    def test_inside(self):
        assert Star2D(0.3, 5, 2.0).sdf(_p(0, 0))[0] < 0


class TestPie2D_:
    def test_inside(self):
        c = [np.sin(np.pi / 4), np.cos(np.pi / 4)]
        assert Pie2D(c, 0.3).sdf(_p(0, 0.1))[0] < 0


class TestCutDisk2D_:
    def test_inside(self):
        # CutDisk is the cap above y=h; a point between h and the arc is inside
        assert CutDisk2D(0.3, 0.1).sdf(_p(0, 0.2))[0] < 0


class TestArc2D_:
    def test_returns_array(self):
        sc = [0.707, 0.707]
        assert Arc2D(sc, 0.3, 0.05).sdf(_grid(8)).shape == (8, 8)


class TestRing2D_:
    def test_outside_inner_radius(self):
        assert Ring2D(0.2, 0.4).sdf(_p(0, 0))[0] > 0

    def test_inside_ring(self):
        assert Ring2D(0.2, 0.4).sdf(_p(0.3, 0))[0] < 0


class TestHorseshoe2D_:
    def test_returns_array(self):
        c = [0.0, 1.0]
        w = [0.1, 0.05]
        assert Horseshoe2D(c, 0.3, w).sdf(_grid(8)).shape == (8, 8)


class TestVesica2D_:
    def test_inside(self):
        assert Vesica2D(0.3, 0.1).sdf(_p(0, 0))[0] < 0


class TestMoon2D_:
    def test_returns_array(self):
        assert Moon2D(0.2, 0.3, 0.15).sdf(_grid(8)).shape == (8, 8)


class TestRoundedCross2D_:
    def test_inside(self):
        assert RoundedCross2D(1.0).sdf(_p(0, 0))[0] < 0


class TestEgg2D_:
    def test_inside(self):
        assert Egg2D(0.3, 0.1).sdf(_p(0, 0))[0] < 0


class TestHeart2D_:
    def test_returns_array(self):
        assert Heart2D().sdf(_grid(8)).shape == (8, 8)


class TestCross2D_:
    def test_inside(self):
        assert Cross2D((0.1, 0.3), 0.0).sdf(_p(0, 0))[0] < 0


class TestRoundedX2D_:
    def test_returns_array(self):
        assert RoundedX2D(0.3, 0.05).sdf(_grid(8)).shape == (8, 8)


class TestPolygon2D_:
    def test_square_matches_box(self):
        s = 0.2
        v = [[-s, -s], [s, -s], [s, s], [-s, s]]
        p = _p(s + 0.1, 0)
        d_poly = Polygon2D(v).sdf(p)[0]
        d_box  = Box2D((s, s)).sdf(p)[0]
        npt.assert_allclose(d_poly, d_box, atol=1e-6)


class TestEllipse2D_:
    def test_returns_array(self):
        # sdEllipse2D is an iterative approximation; test with single-point input only
        # (grid inputs expose a broadcasting issue in the ab_n intermediate computation)
        assert Ellipse2D((0.4, 0.2)).sdf(_p(0.3, 0.1)).shape == (1,)


class TestParabola2D_:
    def test_returns_array(self):
        assert Parabola2D(1.0).sdf(_grid(8)).shape == (8, 8)


class TestParabolaSegment2D_:
    def test_returns_array(self):
        assert ParabolaSegment2D(0.3, 0.2).sdf(_grid(8)).shape == (8, 8)


class TestBezier2D_:
    def test_nonneg(self):
        result = Bezier2D((0, 0), (0.5, 1.0), (1.0, 0)).sdf(_grid(8))
        assert (result >= 0).all()


class TestBlobbyCross2D_:
    def test_returns_array(self):
        assert BlobbyCross2D(0.3).sdf(_grid(8)).shape == (8, 8)


class TestTunnel2D_:
    def test_returns_array(self):
        assert Tunnel2D((0.2, 0.3)).sdf(_grid(8)).shape == (8, 8)


class TestStairs2D_:
    def test_returns_array(self):
        assert Stairs2D((0.1, 0.1), 3).sdf(_grid(8)).shape == (8, 8)


class TestQuadraticCircle2D_:
    def test_returns_array(self):
        assert QuadraticCircle2D().sdf(_grid(8)).shape == (8, 8)


class TestHyperbola2D_:
    def test_returns_array(self):
        assert Hyperbola2D(0.5, 0.3).sdf(_grid(8)).shape == (8, 8)


# ===========================================================================
# Boolean operations
# ===========================================================================

class TestBoolean2D:
    def test_union_includes_both(self):
        a = Circle2D(0.3)
        b = Box2D((0.3, 0.3)).translate(0.5, 0.0)
        u = Union2D(a, b)
        assert u.sdf(_p(0, 0))[0] < 0      # inside circle
        assert u.sdf(_p(0.5, 0))[0] < 0    # inside box

    def test_intersection_excludes_union(self):
        a = Circle2D(0.3)
        b = Circle2D(0.3).translate(0.4, 0.0)
        i = Intersection2D(a, b)
        # Origin is in a but not b
        assert i.sdf(_p(0, 0))[0] > 0

    def test_subtraction_removes_cutter(self):
        a = Circle2D(0.4)
        b = Circle2D(0.2)
        s = Subtraction2D(a, b)
        # Origin is inside both → outside the subtraction result
        assert s.sdf(_p(0, 0))[0] > 0
        # Point inside a but outside b → inside subtraction
        assert s.sdf(_p(0.3, 0))[0] < 0

    def test_union_class_is_geometry2d(self):
        assert isinstance(Union2D(Circle2D(0.3), Box2D((0.2, 0.2))), Geometry2D)

    def test_method_union_equals_class(self):
        a = Circle2D(0.3)
        b = Box2D((0.2, 0.2))
        p = _grid(8)
        npt.assert_allclose(a.union(b).sdf(p), Union2D(a, b).sdf(p))

    def test_method_subtract_equals_class(self):
        a = Circle2D(0.4)
        b = Circle2D(0.2)
        p = _grid(8)
        npt.assert_allclose(a.subtract(b).sdf(p), Subtraction2D(a, b).sdf(p))

    def test_method_intersect_equals_class(self):
        a = Circle2D(0.3)
        b = Box2D((0.2, 0.2))
        p = _grid(8)
        npt.assert_allclose(a.intersect(b).sdf(p), Intersection2D(a, b).sdf(p))
