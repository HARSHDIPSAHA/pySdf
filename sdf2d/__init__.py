"""
sdf2d - 2D Signed Distance Function Library
============================================

A comprehensive library for creating and manipulating 2D signed distance fields (SDFs).
Based on Inigo Quilez's distance function collection.

Basic Usage
-----------

Creating shapes:
    >>> from sdf2d import Circle, Box2D, Union2D
    >>> circle = Circle(radius=0.3)
    >>> box = Box2D(half_size=(0.2, 0.2)).translate(0.4, 0.0)
    >>> combined = Union2D(circle, box)

Sampling to grid:
    >>> from sdf2d import sample_levelset_2d
    >>> bounds = ((-1, 1), (-1, 1))
    >>> resolution = (512, 512)
    >>> phi = sample_levelset_2d(combined, bounds, resolution)

AMReX integration:
    >>> import amrex.space2d as amr
    >>> from sdf2d import SDFLibrary2D
    >>> amr.initialize([])
    >>> # ... setup geometry, ba, dm ...
    >>> lib = SDFLibrary2D(geom, ba, dm)
    >>> levelset = lib.circle(center=(0, 0), radius=0.3)
"""

from .geometry import (
    # Base
    Geometry2D,
    # Basic shapes
    Circle,
    Box2D,
    RoundedBox2D,
    OrientedBox2D,
    Segment2D,
    Rhombus2D,
    Trapezoid2D,
    Parallelogram2D,
    # Triangles
    EquilateralTriangle2D,
    TriangleIsosceles2D,
    Triangle2D,
    # Capsules
    UnevenCapsule2D,
    # Regular polygons
    Pentagon2D,
    Hexagon2D,
    Octogon2D,
    NGon2D,
    # Stars
    Hexagram2D,
    Star5,
    Star,
    # Arcs and sectors
    Pie2D,
    CutDisk2D,
    Arc2D,
    Ring2D,
    Horseshoe2D,
    # Special shapes
    Vesica2D,
    Moon2D,
    RoundedCross2D,
    Egg2D,
    Heart2D,
    Cross2D,
    RoundedX2D,
    # Complex shapes
    Polygon2D,
    Ellipse2D,
    Parabola2D,
    ParabolaSegment2D,
    Bezier2D,
    BlobbyCross2D,
    Tunnel2D,
    Stairs2D,
    QuadraticCircle2D,
    Hyperbola2D,
    # Boolean operations
    Union2D,
    Intersection2D,
    Subtraction2D,
    # AMReX
    SDFLibrary2D,
)

from .grid import sample_levelset_2d, save_npy
from ._loader import load_module

# Load visualization module
_viz = load_module("sdf2d._viz", "2d/visualization_2d.py")
save_levelset_html_2d = _viz.save_levelset_html_2d

__version__ = "0.1.0"

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
    # Grid utilities
    "sample_levelset_2d",
    "save_npy",
    "save_levelset_html_2d",
]
