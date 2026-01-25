## Geometry to Level-Set Library Structure

This project now separates the library into two folders:

- `3d/` - the 3D geometry to level-set system (implemented now)
- `2d/` - placeholder for the 2D version (to be implemented later)
- `scripts/` - rendering/utility scripts (not part of the core library)
- `outputs/` - generated images and data (ignored by the library API)

### `3d/` contents

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

### `2d/` contents

Currently empty. This folder is reserved for the 2D geometry system, which
will mirror the same structure as `3d/`.

### `scripts/` contents

Utility scripts for visualization and plotfile workflows. These are optional
and do not affect the level-set library itself.

### `outputs/` contents

Generated images or derived artifacts (PNG renders, etc.). Safe to delete.

## Intent

The goal is to act like a small CAD-like compiler:

User geometry and operations  
→ Signed Distance Function  
→ Sampled level-set field on a bounding box grid

Visualization is optional and treated as post-processing.
