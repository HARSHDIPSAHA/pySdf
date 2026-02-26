## pySdf Library Structure

A 2D and 3D Signed Distance Function (SDF) library with optional pyAMReX integration.

---

### Top-level

- `sdf_lib.py` — NumPy implementations of all SDF primitives and operators. Both 2D and 3D functions live here. No AMReX dependency.
- `sdf2d/` — 2D geometry package (fully implemented).
- `sdf3d/` — 3D geometry package (fully implemented).
- `tests/` — pytest test suite (no AMReX required to run).
- `scripts/` — visualization and rendering utilities (optional, not part of the library API).
- `gallery_2d.png`, `gallery_3d.png` — pre-rendered shape galleries (repo root).

---

### `sdf2d/` contents

The 2D API. All code lives directly in this package.

- `geometry.py` — All 2D geometry classes. Includes primitives (`Circle2D`, `Box2D`, `Hexagon2D`, …), boolean ops (`Union2D`, `Intersection2D`, `Subtraction2D`), and transforms (`translate`, `rotate`, `scale`, `round`, `onion`). Every class wraps a lambda over `sdf_lib` and exposes `sdf(p)`.
- `grid.py` — `sample_levelset_2d(geom, bounds, resolution)` samples a `Geometry2D` object onto a cell-centred 2D NumPy grid. Also provides `save_npy`.
- `amrex.py` — `SDFLibrary2D` builds AMReX `MultiFab` level-set fields for a 2D domain. Requires a pyAMReX 2-D build (`amrex.space2d`). Import is guarded so the module loads even without AMReX.
- `__init__.py` — Re-exports every public symbol from the three modules above.

### `sdf3d/` contents

The 3D API. All code lives directly in this package.

- `geometry.py` — All 3D geometry classes: `Sphere3D`, `Box3D`, `RoundBox3D`, `Cylinder3D`, `ConeExact3D`, `Torus3D`, and boolean ops `Union3D` / `Intersection3D` / `Subtraction3D`. Transforms: `translate`, `rotate_x/y/z`, `scale`, `elongate`, `round`, `onion`.
- `grid.py` — `sample_levelset_3d(geom, bounds, resolution)` samples a `Geometry3D` onto a cell-centred 3D NumPy grid (shape `(nz, ny, nx)`). Also provides `save_npy`.
- `amrex.py` — `SDFLibrary3D` builds AMReX `MultiFab` level-set fields for a 3D domain. Requires a pyAMReX 3-D build (`amrex.space3d`). Import is guarded so the module loads even without AMReX.
- `examples/` — High-level geometry assemblies:
  - `nato_stanag.py` — `NATOFragment(lib, …)` — a NATO STANAG fragmentation cylinder with conical nose, parameterised by diameter and L/D ratio.
  - `rocket_assembly.py` — `RocketAssembly(lib, …)` — a multi-part rocket: cylindrical body, nose sphere, and N fins arranged symmetrically.
  - Both return `(MultiFab | Geometry3D, Geometry3D)` depending on the `lib` argument.
- `__init__.py` — Re-exports all public symbols.

---

### `tests/` contents

All tests run with `pytest` and require only `numpy`. AMReX is not needed.

| File | What it tests |
|------|--------------|
| `test_sdf_lib.py` | Every function in `sdf_lib.py` at analytically known points |
| `test_sdf2d_geometry.py` | Every 2D geometry class: sign correctness, transforms, booleans |
| `test_sdf2d_grid.py` | `sample_levelset_2d` shape/sign, `save_npy` round-trip |
| `test_sdf3d_geometry.py` | Every 3D geometry class: sign correctness, transforms, booleans |
| `test_sdf3d_grid.py` | `sample_levelset_3d` shape/sign, `save_npy` round-trip |
| `test_complex.py` | `NATOFragment` and `RocketAssembly` (using a mock lib, no AMReX) |
| `test_amrex.py` | `SDFLibrary2D` and `SDFLibrary3D` — **skipped automatically without pyAMReX** |

Run:
```bash
pytest tests/ -v
```

---

### `scripts/` contents

Rendering and visualization utilities. These are not part of the library API and are not installed.

- `gallery_2d.py` — Renders all `sdf2d` shapes on a single page (requires matplotlib).
- `gallery_3d.py` — Renders all `sdf_lib` 3D primitives on a single page (requires matplotlib + scikit-image).
- `render_surface_from_plotfile.py` — Renders an AMReX plotfile SDF=0 surface (requires pyAMReX + yt).

---

### Design

```
User parameters
      ↓
Geometry classes  (sdf2d / sdf3d)
      ↓
SDF evaluation    (sdf_lib.py — pure NumPy)
      ↓
Level-set field   φ(x, y[, z]) on a grid
      ↓
Output:  NumPy array  OR  AMReX MultiFab
```

- **NumPy path** (no AMReX): `sample_levelset_2d` / `sample_levelset_3d` → `np.ndarray`
- **AMReX path**: `SDFLibrary2D` / `SDFLibrary3D` → `amr.MultiFab` (solver-ready)
