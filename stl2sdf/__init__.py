"""stl2sdf — STL mesh to Signed Distance Field.

Converts a triangulated surface mesh (STL file) into a
:class:`sdf3d.geometry.Geometry3D` that composes with analytic primitives.

Quick start
-----------
>>> from stl2sdf import stl_to_geometry
>>> from sdf3d import Sphere3D
>>> from sdf3d.grid import sample_levelset_3d
>>>
>>> wheel = stl_to_geometry("mars_wheel.stl")
>>> hollowed = wheel.subtract(Sphere3D(0.3))
>>> phi = sample_levelset_3d(hollowed, bounds=((-1,1),(-1,1),(-1,1)), resolution=(32,32,32))

Watertight requirement
----------------------
Sign determination uses Möller–Trumbore ray casting (parity of crossings).
Results are only correct for closed, 2-manifold meshes.

Performance
-----------
O(F × N) per .sdf() call where F = triangles and N = query points.
Use coarse grids for large meshes; a BVH would be needed for production scale.
"""

from .geometry import stl_to_geometry

__all__ = ["stl_to_geometry"]
