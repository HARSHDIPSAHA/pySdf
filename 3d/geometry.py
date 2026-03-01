import numpy as np
import sdf_lib as sdf


class Geometry:
    def __init__(self, func):
        self._func = func

    def sdf(self, p):
        return self._func(p)

    def __call__(self, p):
        return self._func(p)

    def union(self, other):
        return Geometry(lambda p: sdf.opUnion(self.sdf(p), other.sdf(p)))

    def subtract(self, other):
        return Geometry(lambda p: sdf.opSubtraction(self.sdf(p), other.sdf(p)))

    def intersect(self, other):
        return Geometry(lambda p: sdf.opIntersection(self.sdf(p), other.sdf(p)))

    def round(self, rad):
        return Geometry(lambda p: sdf.opRound(p, self.sdf, rad))

    def onion(self, thickness):
        return Geometry(lambda p: sdf.opOnion(self.sdf(p), thickness))

    def translate(self, tx, ty, tz):
        t = np.array([tx, ty, tz])
        return Geometry(lambda p: self.sdf(p - t))

    def scale(self, s):
        return Geometry(lambda p: sdf.opScale(p, s, self.sdf))

    def elongate(self, hx, hy, hz):
        h = np.array([hx, hy, hz])
        return Geometry(lambda p: sdf.opElongate2(p, self.sdf, h))

    def rotate_x(self, angle_rad):
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rot = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        return Geometry(lambda p: sdf.opTx(p, rot, np.zeros(3), self.sdf))

    def rotate_y(self, angle_rad):
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        return Geometry(lambda p: sdf.opTx(p, rot, np.zeros(3), self.sdf))

    def rotate_z(self, angle_rad):
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return Geometry(lambda p: sdf.opTx(p, rot, np.zeros(3), self.sdf))


class Sphere(Geometry):
    def __init__(self, radius):
        super().__init__(lambda p: sdf.sdSphere(p, radius))


class Box(Geometry):
    def __init__(self, half_size):
        b = np.array(half_size, dtype=float)
        super().__init__(lambda p: sdf.sdBox(p, b))


class RoundBox(Geometry):
    def __init__(self, half_size, radius):
        b = np.array(half_size, dtype=float)
        super().__init__(lambda p: sdf.sdRoundBox(p, b, radius))


class Cylinder(Geometry):
    def __init__(self, axis_offset, radius):
        c = np.array([axis_offset[0], axis_offset[1], radius], dtype=float)
        super().__init__(lambda p: sdf.sdCylinder(p, c))


class ConeExact(Geometry):
    def __init__(self, sincos, height):
        c = np.array(sincos, dtype=float)
        super().__init__(lambda p: sdf.sdConeExact(p, c, height))


class Torus(Geometry):
    def __init__(self, major_minor):
        t = np.array(major_minor, dtype=float)
        super().__init__(lambda p: sdf.sdTorus(p, t))


class Union(Geometry):
    def __init__(self, *geoms):
        def _sdf(p):
            d = geoms[0].sdf(p)
            for g in geoms[1:]:
                d = sdf.opUnion(d, g.sdf(p))
            return d

        super().__init__(_sdf)


class Intersection(Geometry):
    def __init__(self, *geoms):
        def _sdf(p):
            d = geoms[0].sdf(p)
            for g in geoms[1:]:
                d = sdf.opIntersection(d, g.sdf(p))
            return d

        super().__init__(_sdf)


class Subtraction(Geometry):
    def __init__(self, base, cutter):
        super().__init__(lambda p: sdf.opSubtraction(base.sdf(p), cutter.sdf(p)))
