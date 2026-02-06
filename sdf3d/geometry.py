from ._loader import load_module

_geom = load_module("sdf3d._geom", "3d/geometry.py")

# Try to load AMReX-dependent modules, but make them optional
try:
    _amrex = load_module("sdf3d._amrex", "3d/amrex_sdf.py")
    SDFLibrary = _amrex.SDFLibrary
    _HAS_AMREX = True
except ImportError:
    # AMReX not available, SDFLibrary will not be available
    SDFLibrary = None
    _HAS_AMREX = False

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
]

if _HAS_AMREX:
    __all__.append("SDFLibrary")
