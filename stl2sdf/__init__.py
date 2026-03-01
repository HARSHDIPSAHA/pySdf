"""stl2sdf — STL mesh to Signed Distance Field (pure numpy).

Converts triangulated surface meshes stored in STL format into volumetric
signed distance fields sampled on a uniform Cartesian grid.

Quick start
-----------
>>> from stl2sdf import sample_sdf_from_stl
>>> phi = sample_sdf_from_stl(
...     "my_mesh.stl",
...     bounds=((0, 1), (0, 1), (0, 1)),
...     resolution=(32, 32, 32),
... )
>>> phi.shape
(32, 32, 32)

Watertight requirement
----------------------
Sign determination uses Möller–Trumbore ray casting (parity of ray-triangle
intersections).  The result is only correct for **watertight** (closed, 2-manifold)
meshes.  Near gaps or non-manifold edges the sign will be wrong or inconsistent.

Performance
-----------
Complexity is O(F × N) where F = number of triangles and N = number of grid
cells.  A 14 K-triangle mesh on a 40³ grid (~64 K points) takes 2–5 min on a
single CPU core.  Use ``resolution=(20, 20, 20)`` for rapid tests (~15 s).
For production workloads a BVH/KD-tree would be necessary.
"""

from .mesh_sdf import load_stl, mesh_to_sdf, sample_sdf_from_stl

__all__ = ["load_stl", "mesh_to_sdf", "sample_sdf_from_stl"]
