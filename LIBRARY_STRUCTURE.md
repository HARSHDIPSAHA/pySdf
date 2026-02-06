# Library Structure

- `3d/` - the 3D geometry to level-set system
- `2d/` - the 2D geometry to level-set system

Packages: (installed via setup.py)
- `sdf3d/` - Python package for `3d/` (can be installed via pip)
- `sdf2d/` - Python package for `2d/` (can be installed via pip)

Other folders: (not part of the library)
- `scripts/` - utility scripts
- `outputs/` - generated images and data
- `examples/` - example usage scripts for testing and demonstration

# `3d/` contents
In order of dependency:

- `geometry.py`  
  Core geometry classes (Sphere, Box, Torus, etc.) and boolean operations.
  Each geometry exposes `sdf(p)` for signed distance evaluation.

- `grid.py`  
  Utilities for sampling a geometry on a 3D grid and saving the level-set
  data to disk (e.g., `output/levelset.npy`).

- `amrex_sdf.py`  
  AMReX-native level-set generation. Creates and combines `MultiFab` fields
  directly (solver-ready output).

- `SDFLibrary` output is always an AMReX `MultiFab`. Visualization scripts
  live in `scripts/` and do not affect the library API.

- `examples.py`  
  Minimal example showing how to build a composite geometry and generate a
  level-set field.

- `__init__.py`  
  Exposes the main classes and helpers for easy imports.

# `2d/` contents
In order of dependency:

- `sdf_lib.py` (top-level, shared with `3d/`, not in `2d/` folder)  
  Numpy implementations of functions for vector math and SDF operations (e.g., vector creation, dot product, length) shared by both 2D and 3D; the 2D modules import this top-level module rather than a local copy.

- `geometry_2d.py`  
  Core 2D geometry classes (Circle, Box, Polygon, etc.) and boolean operations.
  Each geometry exposes `sdf(p)` for signed distance evaluation in 2D.

- `grid_2d.py`  
  Utilities for sampling a 2D geometry on a grid and saving the level-set
  data to disk (e.g., `output/levelset_2d.npy`).

- `amrex_sdf_2d.py`  
  AMReX-native level-set generation for 2D. Creates and combines fields
  directly for solver-ready output.

- `examples_2d.py`  
  Minimal example showing how to build a composite 2D geometry and generate a
  level-set field.

- `eg2dd.py`  
  Additional 2D example or demonstration script.

- `__init__.py`  
  Exposes the main 2D classes and helpers for easy imports.
