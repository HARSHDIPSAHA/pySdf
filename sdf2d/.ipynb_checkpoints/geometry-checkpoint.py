"""2D geometry primitives and operations."""
from ._loader import load_module

# Load implementations
_geom = load_module("sdf2d._geom", "2d/geometry_2d.py")
_amrex = load_module("sdf2d._amrex", "2d/amrex_sdf_2d.py")

# Base class
Geometry2D = _geom.Geometry2D

# Primitive shapes
Circle = _geom.Circle
Box2D = _geom.Box2D
RoundedBox2D = _geom.RoundedBox2D
OrientedBox2D = _geom.OrientedBox2D
Segment2D = _geom.Segment2D
Rhombus2D = _geom.Rhombus2D
Trapezoid2D = _geom.Trapezoid2D
Parallelogram2D = _geom.Parallelogram2D

# Triangles
EquilateralTriangle2D = _geom.EquilateralTriangle2D
TriangleIsosceles2D = _geom.TriangleIsosceles2D
Triangle2D = _geom.Triangle2D

# Capsules
UnevenCapsule2D = _geom.UnevenCapsule2D

# Regular polygons
Pentagon2D = _geom.Pentagon2D
Hexagon2D = _geom.Hexagon2D
Octogon2D = _geom.Octogon2D
NGon2D = _geom.NGon2D

# Stars
Hexagram2D = _geom.Hexagram2D
Star5 = _geom.Star5
Star = _geom.Star

# Arcs and sectors
Pie2D = _geom.Pie2D
CutDisk2D = _geom.CutDisk2D
Arc2D = _geom.Arc2D
Ring2D = _geom.Ring2D
Horseshoe2D = _geom.Horseshoe2D

# Special shapes
Vesica2D = _geom.Vesica2D
Moon2D = _geom.Moon2D
RoundedCross2D = _geom.RoundedCross2D
Egg2D = _geom.Egg2D
Heart2D = _geom.Heart2D
Cross2D = _geom.Cross2D
RoundedX2D = _geom.RoundedX2D

# Complex shapes
Polygon2D = _geom.Polygon2D
Ellipse2D = _geom.Ellipse2D
Parabola2D = _geom.Parabola2D
ParabolaSegment2D = _geom.ParabolaSegment2D
Bezier2D = _geom.Bezier2D
BlobbyCross2D = _geom.BlobbyCross2D
Tunnel2D = _geom.Tunnel2D
Stairs2D = _geom.Stairs2D
QuadraticCircle2D = _geom.QuadraticCircle2D
Hyperbola2D = _geom.Hyperbola2D

# Boolean operations
Union2D = _geom.Union2D
Intersection2D = _geom.Intersection2D
Subtraction2D = _geom.Subtraction2D

# AMReX integration
SDFLibrary2D = _amrex.SDFLibrary2D

__all__ = [
    # Base
    "Geometry2D",
    
    # Basic shapes
    "Circle",
    "Box2D",
    "RoundedBox2D",
    "OrientedBox2D",
    "Segment2D",
    "Rhombus2D",
    "Trapezoid2D",
    "Parallelogram2D",
    
    # Triangles
    "EquilateralTriangle2D",
    "TriangleIsosceles2D",
    "Triangle2D",
    
    # Capsules
    "UnevenCapsule2D",
    
    # Regular polygons
    "Pentagon2D",
    "Hexagon2D",
    "Octogon2D",
    "NGon2D",
    
    # Stars
    "Hexagram2D",
    "Star5",
    "Star",
    
    # Arcs and sectors
    "Pie2D",
    "CutDisk2D",
    "Arc2D",
    "Ring2D",
    "Horseshoe2D",
    
    # Special shapes
    "Vesica2D",
    "Moon2D",
    "RoundedCross2D",
    "Egg2D",
    "Heart2D",
    "Cross2D",
    "RoundedX2D",
    
    # Complex shapes
    "Polygon2D",
    "Ellipse2D",
    "Parabola2D",
    "ParabolaSegment2D",
    "Bezier2D",
    "BlobbyCross2D",
    "Tunnel2D",
    "Stairs2D",
    "QuadraticCircle2D",
    "Hyperbola2D",
    
    # Boolean operations
    "Union2D",
    "Intersection2D",
    "Subtraction2D",
    
    # AMReX
    "SDFLibrary2D",
]
