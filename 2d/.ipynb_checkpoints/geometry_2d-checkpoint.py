import numpy as np
import sdf_lib as sdf

class Geometry2D:
    """Base class for 2D geometries using SDFs."""
    def __init__(self, func):
        self._func = func

    def sdf(self, p):
        """Evaluate SDF at points p (vec2 array)."""
        return self._func(p)

    def __call__(self, p):
        return self._func(p)

    def union(self, other):
        return Geometry2D(lambda p: sdf.opUnion(self.sdf(p), other.sdf(p)))

    def subtract(self, other):
        return Geometry2D(lambda p: sdf.opSubtraction(self.sdf(p), other.sdf(p)))

    def intersect(self, other):
        return Geometry2D(lambda p: sdf.opIntersection(self.sdf(p), other.sdf(p)))

    def round(self, rad):
        return Geometry2D(lambda p: sdf.opRound(p, self.sdf, rad))

    def onion(self, thickness):
        return Geometry2D(lambda p: sdf.opOnion(self.sdf(p), thickness))

    def translate(self, tx, ty):
        t = np.array([tx, ty])
        return Geometry2D(lambda p: self.sdf(p - t))

    def scale(self, s):
        return Geometry2D(lambda p: sdf.opScale(p, s, self.sdf))

    def rotate(self, angle_rad):
        """Rotate 2D shape by angle in radians."""
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rot = np.array([[c, -s], [s, c]])
        return Geometry2D(lambda p: sdf.opTx2D(p, rot, np.zeros(2), self.sdf))


# ============================================================================
# 2D PRIMITIVE SHAPES
# ============================================================================

class Circle(Geometry2D):
    def __init__(self, radius):
        super().__init__(lambda p: sdf.sdCircle(p, radius))


class Box2D(Geometry2D):
    def __init__(self, half_size):
        b = np.array(half_size, dtype=float)
        super().__init__(lambda p: sdf.sdBox2D(p, b))


class RoundedBox2D(Geometry2D):
    def __init__(self, half_size, radius):
        b = np.array(half_size, dtype=float)
        super().__init__(lambda p: sdf.sdRoundedBox2D(p, b, radius))


class OrientedBox2D(Geometry2D):
    def __init__(self, corner_a, corner_b, thickness):
        a = np.array(corner_a, dtype=float)
        b = np.array(corner_b, dtype=float)
        super().__init__(lambda p: sdf.sdOrientedBox2D(p, a, b, thickness))


class Segment2D(Geometry2D):
    def __init__(self, point_a, point_b):
        a = np.array(point_a, dtype=float)
        b = np.array(point_b, dtype=float)
        super().__init__(lambda p: sdf.sdSegment2D(p, a, b))


class Rhombus2D(Geometry2D):
    def __init__(self, half_size):
        b = np.array(half_size, dtype=float)
        super().__init__(lambda p: sdf.sdRhombus2D(p, b))


class Trapezoid2D(Geometry2D):
    def __init__(self, r1, r2, height):
        super().__init__(lambda p: sdf.sdTrapezoid2D(p, r1, r2, height))


class Parallelogram2D(Geometry2D):
    def __init__(self, width, height, skew):
        super().__init__(lambda p: sdf.sdParallelogram2D(p, width, height, skew))


class EquilateralTriangle2D(Geometry2D):
    def __init__(self, radius):
        super().__init__(lambda p: sdf.sdEquilateralTriangle2D(p, radius))


class TriangleIsosceles2D(Geometry2D):
    def __init__(self, width, height):
        q = np.array([width, height], dtype=float)
        super().__init__(lambda p: sdf.sdTriangleIsosceles2D(p, q))


class Triangle2D(Geometry2D):
    def __init__(self, p0, p1, p2):
        v0 = np.array(p0, dtype=float)
        v1 = np.array(p1, dtype=float)
        v2 = np.array(p2, dtype=float)
        super().__init__(lambda p: sdf.sdTriangle2D(p, v0, v1, v2))


class UnevenCapsule2D(Geometry2D):
    def __init__(self, r1, r2, height):
        super().__init__(lambda p: sdf.sdUnevenCapsule2D(p, r1, r2, height))


class Pentagon2D(Geometry2D):
    def __init__(self, radius):
        super().__init__(lambda p: sdf.sdPentagon2D(p, radius))


class Hexagon2D(Geometry2D):
    def __init__(self, radius):
        super().__init__(lambda p: sdf.sdHexagon2D(p, radius))


class Octogon2D(Geometry2D):
    def __init__(self, radius):
        super().__init__(lambda p: sdf.sdOctogon2D(p, radius))


class Hexagram2D(Geometry2D):
    def __init__(self, radius):
        super().__init__(lambda p: sdf.sdHexagram2D(p, radius))


class Star5(Geometry2D):
    def __init__(self, outer_radius, inner_factor):
        super().__init__(lambda p: sdf.sdStar5(p, outer_radius, inner_factor))


class Star(Geometry2D):
    def __init__(self, radius, n_points, factor):
        super().__init__(lambda p: sdf.sdStar(p, radius, n_points, factor))


class Pie2D(Geometry2D):
    def __init__(self, angle_sc, radius):
        """angle_sc is vec2(sin(half_angle), cos(half_angle))"""
        c = np.array(angle_sc, dtype=float)
        super().__init__(lambda p: sdf.sdPie2D(p, c, radius))


class CutDisk2D(Geometry2D):
    def __init__(self, radius, cut_height):
        super().__init__(lambda p: sdf.sdCutDisk2D(p, radius, cut_height))


class Arc2D(Geometry2D):
    def __init__(self, angle_sc, radius, thickness):
        """angle_sc is vec2(sin(half_angle), cos(half_angle))"""
        sc = np.array(angle_sc, dtype=float)
        super().__init__(lambda p: sdf.sdArc2D(p, sc, radius, thickness))


class Ring2D(Geometry2D):
    def __init__(self, inner_radius, outer_radius):
        super().__init__(lambda p: sdf.sdRing2D(p, inner_radius, outer_radius))


class Horseshoe2D(Geometry2D):
    def __init__(self, angle_sc, radius, thickness_vec2):
        c = np.array(angle_sc, dtype=float)
        w = np.array(thickness_vec2, dtype=float)
        super().__init__(lambda p: sdf.sdHorseshoe2D(p, c, radius, w))


class Vesica2D(Geometry2D):
    def __init__(self, radius, distance):
        super().__init__(lambda p: sdf.sdVesica2D(p, radius, distance))


class Moon2D(Geometry2D):
    def __init__(self, distance, radius_a, radius_b):
        super().__init__(lambda p: sdf.sdMoon2D(p, distance, radius_a, radius_b))


class RoundedCross2D(Geometry2D):
    def __init__(self, size):
        super().__init__(lambda p: sdf.sdRoundedCross2D(p, size))


class Egg2D(Geometry2D):
    def __init__(self, radius_a, radius_b):
        super().__init__(lambda p: sdf.sdEgg2D(p, radius_a, radius_b))


class Heart2D(Geometry2D):
    def __init__(self):
        super().__init__(lambda p: sdf.sdHeart2D(p))


class Cross2D(Geometry2D):
    def __init__(self, size_vec2, rounding):
        b = np.array(size_vec2, dtype=float)
        super().__init__(lambda p: sdf.sdCross2D(p, b, rounding))


class RoundedX2D(Geometry2D):
    def __init__(self, width, rounding):
        super().__init__(lambda p: sdf.sdRoundedX2D(p, width, rounding))


class Polygon2D(Geometry2D):
    def __init__(self, vertices):
        """vertices is list/array of vec2 points"""
        v = np.array(vertices, dtype=float)
        super().__init__(lambda p: sdf.sdPolygon2D(p, v))


class Ellipse2D(Geometry2D):
    def __init__(self, semi_axes):
        """semi_axes is vec2 (a, b)"""
        ab = np.array(semi_axes, dtype=float)
        super().__init__(lambda p: sdf.sdEllipse2D(p, ab))


class Parabola2D(Geometry2D):
    def __init__(self, curvature):
        super().__init__(lambda p: sdf.sdParabola2D(p, curvature))


class ParabolaSegment2D(Geometry2D):
    def __init__(self, width, height):
        super().__init__(lambda p: sdf.sdParabolaSegment2D(p, width, height))


class Bezier2D(Geometry2D):
    def __init__(self, p0, p1, p2):
        """Quadratic Bezier with control points p0, p1, p2"""
        A = np.array(p0, dtype=float)
        B = np.array(p1, dtype=float)
        C = np.array(p2, dtype=float)
        super().__init__(lambda p: sdf.sdBezier2D(p, A, B, C))


class BlobbyCross2D(Geometry2D):
    def __init__(self, size):
        super().__init__(lambda p: sdf.sdBlobbyCross2D(p, size))


class Tunnel2D(Geometry2D):
    def __init__(self, size_vec2):
        wh = np.array(size_vec2, dtype=float)
        super().__init__(lambda p: sdf.sdTunnel2D(p, wh))


class Stairs2D(Geometry2D):
    def __init__(self, step_size, num_steps):
        wh = np.array(step_size, dtype=float)
        super().__init__(lambda p: sdf.sdStairs2D(p, wh, num_steps))


class QuadraticCircle2D(Geometry2D):
    def __init__(self):
        super().__init__(lambda p: sdf.sdQuadraticCircle2D(p))


class Hyperbola2D(Geometry2D):
    def __init__(self, curvature, height):
        super().__init__(lambda p: sdf.sdHyperbola2D(p, curvature, height))


class NGon2D(Geometry2D):
    def __init__(self, radius, n_sides):
        """Regular N-sided polygon"""
        super().__init__(lambda p: sdf.sdNGon2D(p, radius, n_sides))


# ============================================================================
# BOOLEAN OPERATIONS
# ============================================================================

class Union2D(Geometry2D):
    def __init__(self, *geoms):
        def _sdf(p):
            d = geoms[0].sdf(p)
            for g in geoms[1:]:
                d = sdf.opUnion(d, g.sdf(p))
            return d
        super().__init__(_sdf)


class Intersection2D(Geometry2D):
    def __init__(self, *geoms):
        def _sdf(p):
            d = geoms[0].sdf(p)
            for g in geoms[1:]:
                d = sdf.opIntersection(d, g.sdf(p))
            return d
        super().__init__(_sdf)


class Subtraction2D(Geometry2D):
    def __init__(self, base, cutter):
        super().__init__(lambda p: sdf.opSubtraction(base.sdf(p), cutter.sdf(p)))
