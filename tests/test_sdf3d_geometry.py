"""Tests for sdf3d geometry classes."""

import numpy as np
import numpy.testing as npt
import pytest

from sdf3d import (
    Geometry3D,
    Sphere3D, Box3D, RoundBox3D, Cylinder3D, ConeExact3D, Torus3D,
    Union3D, Intersection3D, Subtraction3D,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p(*xyz) -> np.ndarray:
    return np.array([list(xyz)], dtype=float)


def _grid(n: int = 8) -> np.ndarray:
    lin = np.linspace(-1.0, 1.0, n)
    Z, Y, X = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.stack([X, Y, Z], axis=-1)


# ===========================================================================
# Base class
# ===========================================================================

class TestGeometry3D:
    def test_translate_moves_origin(self):
        s = Sphere3D(0.3)
        moved = s.translate(0.5, 0.0, 0.0)
        npt.assert_allclose(moved.sdf(_p(0.5, 0, 0)), [-0.3], atol=1e-10)

    def test_scale_doubles_radius(self):
        s = Sphere3D(0.3)
        scaled = s.scale(2.0)
        npt.assert_allclose(scaled.sdf(_p(0, 0, 0)), [-0.6], atol=1e-10)

    def test_elongate_extends_interior(self):
        s = Sphere3D(0.3)
        e = s.elongate(0.0, 0.0, 0.2)
        # Along Z, point at (0, 0, 0.4) should be inside the elongated sphere
        assert e.sdf(_p(0, 0, 0.4))[0] < 0
        # But (0, 0, 0.75) should be outside
        assert e.sdf(_p(0, 0, 0.75))[0] > 0

    def test_rotate_x_preserves_sphere(self):
        s = Sphere3D(0.3)
        r = s.rotate_x(np.pi / 3)
        npt.assert_allclose(r.sdf(_p(0, 0, 0)), [-0.3], atol=1e-10)

    def test_rotate_y_preserves_sphere(self):
        s = Sphere3D(0.3)
        r = s.rotate_y(np.pi / 3)
        npt.assert_allclose(r.sdf(_p(0, 0, 0)), [-0.3], atol=1e-10)

    def test_rotate_z_preserves_sphere(self):
        s = Sphere3D(0.3)
        r = s.rotate_z(np.pi / 3)
        npt.assert_allclose(r.sdf(_p(0, 0, 0)), [-0.3], atol=1e-10)

    def test_round_grows_surface(self):
        b = Box3D((0.2, 0.2, 0.2))
        r = b.round(0.05)
        p = _p(0.25, 0, 0)
        npt.assert_allclose(r.sdf(p), b.sdf(p) - 0.05, atol=1e-10)

    def test_onion_creates_shell(self):
        s = Sphere3D(0.3)
        shell = s.onion(0.02)
        # On the sphere surface → thin positive layer
        assert shell.sdf(_p(1.0, 0, 0))[0] > 0


# ===========================================================================
# Primitive shapes
# ===========================================================================

class TestSphere3D:
    def test_inside_origin(self):
        npt.assert_allclose(Sphere3D(0.3).sdf(_p(0, 0, 0)), [-0.3], atol=1e-10)

    def test_on_surface(self):
        npt.assert_allclose(Sphere3D(0.3).sdf(_p(0.3, 0, 0)), [0.0], atol=1e-10)

    def test_outside(self):
        npt.assert_allclose(Sphere3D(0.3).sdf(_p(0.5, 0, 0)), [0.2], atol=1e-10)

    def test_batch_shape(self):
        phi = Sphere3D(0.3).sdf(_grid(4))
        assert phi.shape == (4, 4, 4)


class TestBox3D_:
    def test_inside(self):
        assert Box3D((0.3, 0.3, 0.3)).sdf(_p(0, 0, 0))[0] < 0

    def test_on_face(self):
        npt.assert_allclose(Box3D((0.3, 0.3, 0.3)).sdf(_p(0.3, 0, 0)), [0.0], atol=1e-10)

    def test_outside(self):
        npt.assert_allclose(Box3D((0.3, 0.3, 0.3)).sdf(_p(0.4, 0, 0)), [0.1], atol=1e-10)


class TestRoundBox3D_:
    def test_inside(self):
        assert RoundBox3D((0.3, 0.3, 0.3), 0.05).sdf(_p(0, 0, 0))[0] < 0

    def test_surface_at_face(self):
        # Face centre is at b_x (rounding affects corners, not face centres)
        b, r = (0.2, 0.2, 0.2), 0.05
        p = _p(b[0], 0, 0)
        npt.assert_allclose(RoundBox3D(b, r).sdf(p), [0.0], atol=1e-10)


class TestCylinder3D_:
    def test_inside(self):
        c = Cylinder3D(axis_offset=[0.0, 0.0], radius=0.3)
        assert c.sdf(_p(0, 0.5, 0))[0] < 0

    def test_on_surface(self):
        c = Cylinder3D(axis_offset=[0.0, 0.0], radius=0.3)
        npt.assert_allclose(c.sdf(_p(0.3, 0, 0)), [0.0], atol=1e-10)


class TestConeExact3D_:
    def test_returns_array(self):
        c = ConeExact3D(sincos=[0.6, 0.8], height=0.35)
        assert c.sdf(_grid(4)).shape == (4, 4, 4)


class TestTorus3D_:
    def test_on_outer_equator(self):
        R, r = 0.3, 0.1
        t = Torus3D((R, r))
        npt.assert_allclose(t.sdf(_p(R + r, 0, 0)), [0.0], atol=1e-10)

    def test_inside_tube(self):
        R, r = 0.3, 0.1
        t = Torus3D((R, r))
        npt.assert_allclose(t.sdf(_p(R, 0, 0)), [-r], atol=1e-10)


# ===========================================================================
# Boolean operations
# ===========================================================================

class TestBoolean3D:
    def test_union_includes_both(self):
        a = Sphere3D(0.3)
        b = Box3D((0.2, 0.2, 0.2)).translate(0.5, 0, 0)
        u = Union3D(a, b)
        assert u.sdf(_p(0, 0, 0))[0] < 0      # inside sphere
        assert u.sdf(_p(0.5, 0, 0))[0] < 0    # inside box

    def test_intersection_requires_both(self):
        a = Sphere3D(0.3)
        b = Sphere3D(0.3).translate(0.4, 0, 0)
        i = Intersection3D(a, b)
        assert i.sdf(_p(0, 0, 0))[0] > 0      # in a but not b

    def test_subtraction_removes_cutter(self):
        a = Sphere3D(0.4)
        b = Sphere3D(0.2)
        s = Subtraction3D(a, b)
        assert s.sdf(_p(0, 0, 0))[0] > 0      # origin inside both → removed
        assert s.sdf(_p(0.3, 0, 0))[0] < 0    # in a, outside b

    def test_method_union_matches_class(self):
        a = Sphere3D(0.3)
        b = Box3D((0.2, 0.2, 0.2))
        p = _grid(4)
        npt.assert_allclose(a.union(b).sdf(p), Union3D(a, b).sdf(p))

    def test_method_subtract_matches_class(self):
        a = Sphere3D(0.4)
        b = Sphere3D(0.2)
        p = _grid(4)
        npt.assert_allclose(a.subtract(b).sdf(p), Subtraction3D(a, b).sdf(p))

    def test_method_intersect_matches_class(self):
        a = Sphere3D(0.3)
        b = Box3D((0.2, 0.2, 0.2))
        p = _grid(4)
        npt.assert_allclose(a.intersect(b).sdf(p), Intersection3D(a, b).sdf(p))

    def test_union_multiple_args(self):
        shapes = [Sphere3D(0.2).translate(i * 0.5, 0, 0) for i in range(3)]
        u = Union3D(*shapes)
        # Centre of each sphere should be inside
        for i in range(3):
            assert u.sdf(_p(i * 0.5, 0, 0))[0] < 0
