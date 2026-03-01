from ._loader import load_module

_geom = load_module("sdf3d._geom", "3d/geometry.py")
_amrex = load_module("sdf3d._amrex", "3d/amrex_sdf.py")

Geometry = _geom.Geometry
Sphere = _geom.Sphere
Box = _geom.Box
RoundBox = _geom.RoundBox
Cylinder = _geom.Cylinder
ConeExact = _geom.ConeExact
Torus = _geom.Torus
Union = _geom.Union
Intersection = _geom.Intersection
Subtraction = _geom.Subtraction
SDFLibrary = _amrex.SDFLibrary

__all__ = [
    "Geometry",
    "Sphere",
    "Box",
    "RoundBox",
    "Cylinder",
    "ConeExact",
    "Torus",
    "Union",
    "Intersection",
    "Subtraction",
    "SDFLibrary",
]
