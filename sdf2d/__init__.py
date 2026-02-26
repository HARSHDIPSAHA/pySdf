"""
sdf2d — 2D Signed Distance Function Library
============================================

A library for creating and composing 2D signed distance fields (SDFs),
based on Inigo Quilez's distance function collection.

Implemented features
--------------------
- Primitive shapes: Circle, Box, triangles, polygons, stars, arcs, ...
- Boolean operations: Union, Intersection, Subtraction
- Transforms: translate, rotate, scale, round, onion
- Grid sampling: :func:`sample_levelset_2d`
- AMReX MultiFab output: :class:`SDFLibrary2D` (requires pyAMReX 2-D build)

Quick start
-----------

NumPy mode (no AMReX required)::

    from sdf2d import Circle2D, Box2D, Union2D, sample_levelset_2d
    import numpy as np

    circle = Circle2D(radius=0.3)
    box    = Box2D(half_size=(0.2, 0.2)).translate(0.4, 0.0)
    shape  = Union2D(circle, box)

    bounds     = ((-1.0, 1.0), (-1.0, 1.0))
    resolution = (512, 512)
    phi = sample_levelset_2d(shape, bounds, resolution)

AMReX mode::

    import amrex.space2d as amr
    from sdf2d import SDFLibrary2D

    amr.initialize([])
    # ... set up geom, ba, dm ...
    lib      = SDFLibrary2D(geom, ba, dm)
    levelset = lib.circle(center=(0.0, 0.0), radius=0.3)
    amr.finalize()
"""

from .geometry import (
    # Base class
    Geometry2D,

    # Primitive shapes
    Circle2D,
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
    Octagon2D,
    Octogon2D,   # backward-compat alias
    NGon2D,

    # Stars
    Hexagram2D,
    Star5_2D,
    Star2D,

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
)

from .grid import sample_levelset_2d, save_npy
from .amrex import SDFLibrary2D

__version__ = "0.2.0"

__all__ = [
    # Base
    "Geometry2D",

    # Primitive shapes
    "Circle2D",
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
    "Octagon2D",
    "Octogon2D",
    "NGon2D",

    # Stars
    "Hexagram2D",
    "Star5_2D",
    "Star2D",

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

    # Grid utilities
    "sample_levelset_2d",
    "save_npy",

    # AMReX integration
    "SDFLibrary2D",
]
