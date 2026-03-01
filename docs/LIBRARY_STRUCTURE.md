# pySdf Library Structure

A 2D and 3D Signed Distance Function library with STL mesh conversion and optional pyAMReX integration.

---

### Top-level

- `_sdf_common.py` — Shared math helpers (`vec2`/`vec3`, `length`, `dot`, `clamp`, boolean ops) used by all packages.
- `sdf2d/` — 2D geometry package.
- `sdf3d/` — 3D geometry package.
- `stl2sdf/` — STL mesh → SDF package (pure NumPy).
- `tests/` — pytest suite (308 pass, 1 skip); no AMReX required.
- `scripts/` — Gallery and plotfile rendering utilities (not part of the library API).
- `examples/` — Standalone runnable demos; outputs written to this folder.
- `gallery_2d.png`, `gallery_3d.png` — Pre-rendered shape galleries.

---

### `sdf2d/` contents

- `primitives.py` — NumPy implementations of all ~50 2D SDF formulas (`sdCircle`, `sdBox2D`, …) and `opTx2D`. Re-exports everything from `_sdf_common`. No AMReX dependency.
- `geometry.py` — All 2D geometry classes: `Circle2D`, `Box2D`, `Hexagon2D`, … plus `Union2D`, `Intersection2D`, `Subtraction2D`. Transforms: `translate`, `rotate`, `scale`, `round`, `onion`. Every class wraps a lambda over `primitives` and exposes `sdf(p)`.
- `grid.py` — `sample_levelset_2d(geom, bounds, resolution)` → `ndarray` of shape `(ny, nx)`. Also provides `save_npy`.
- `amrex.py` — `SDFLibrary2D` for AMReX `MultiFab` output. Requires `amrex.space2d`. Import-guarded so the module loads without AMReX.
- `__init__.py` — Re-exports all public symbols.

---

### `sdf3d/` contents

- `primitives.py` — NumPy implementations of all ~30 3D SDF formulas (`sdSphere`, `sdBox`, …), smooth boolean ops (`opSmoothUnion`, …), and space-warps (`opElongate`, `opRevolution`, `opExtrusion`, `opTwist`, …). Re-exports everything from `_sdf_common`. No AMReX dependency.
- `geometry.py` — All 3D geometry classes: `Sphere3D`, `Box3D`, `RoundBox3D`, `Cylinder3D`, `ConeExact3D`, `Torus3D`, and `Union3D` / `Intersection3D` / `Subtraction3D`. Transforms: `translate`, `rotate_x/y/z`, `scale`, `elongate`, `round`, `onion`.
- `grid.py` — `sample_levelset_3d(geom, bounds, resolution)` → `ndarray` of shape `(nz, ny, nx)`. Also provides `save_npy`.
- `amrex.py` — `SDFLibrary3D` for AMReX `MultiFab` output. Requires `amrex.space3d`. Import-guarded so the module loads without AMReX.
- `examples/` — High-level geometry assemblies:
  - `nato_stanag.py` — `NATOFragment(lib, …)`: NATO STANAG fragmentation cylinder with conical nose.
  - `rocket_assembly.py` — `RocketAssembly(lib, …)`: multi-part rocket with body, nose, and fins.
- `__init__.py` — Re-exports all public symbols.

---

### `stl2sdf/` contents

Converts binary or ASCII STL meshes into signed distance fields via pure NumPy.
Requires a **watertight** (closed, 2-manifold) mesh for correct sign determination.

- `mesh_sdf.py` — All logic:
  - `load_stl(path)` → `(F, 3, 3)` float64 — binary + ASCII loader
  - `mesh_to_sdf(points, triangles)` → `(N,)` — Ericson closest-point + Möller–Trumbore sign
  - `sample_sdf_from_stl(path, bounds, resolution)` → `(nz, ny, nx)` — grid API matching `sample_levelset_3d`
- `__init__.py` — Re-exports `load_stl`, `mesh_to_sdf`, `sample_sdf_from_stl`.

**Algorithms:**
- *Unsigned distance*: Christer Ericson's Voronoi-region closest-point (7 regions, O(F×N))
- *Sign*: Möller–Trumbore ray casting — odd hit count → inside (φ < 0)

---

### `tests/` contents

All tests pass with `pytest` and require only `numpy`. AMReX is not needed.

| File | What it tests |
|------|--------------|
| `test_sdf2d_lib.py` | Every function in `sdf2d/primitives.py` at analytically known points |
| `test_sdf3d_lib.py` | Every function in `sdf3d/primitives.py` at analytically known points |
| `test_sdf2d_geometry.py` | Every 2D geometry class: sign correctness, transforms, booleans |
| `test_sdf2d_grid.py` | `sample_levelset_2d` shape/sign, `save_npy` round-trip |
| `test_sdf3d_geometry.py` | Every 3D geometry class: sign correctness, transforms, booleans |
| `test_sdf3d_grid.py` | `sample_levelset_3d` shape/sign, `save_npy` round-trip |
| `test_complex.py` | `NATOFragment` and `RocketAssembly` (mock lib, no AMReX) |
| `test_stl2sdf.py` | `load_stl` (binary + ASCII), `_sq_dist_to_tri` (7 Voronoi regions), `_mt_ray_hits`, `mesh_to_sdf`, `sample_sdf_from_stl` — all synthetic, no downloads |
| `test_amrex.py` | `SDFLibrary2D` and `SDFLibrary3D` — **skipped automatically without pyAMReX** |

```bash
uv run pytest tests/ -v
```

---

### `scripts/` contents

Rendering and visualization utilities. Not part of the library API; not installed.

- `gallery_2d.py` — Renders all `sdf2d` shapes on a single page (requires matplotlib).
- `gallery_3d.py` — Renders all `sdf3d` primitives (requires matplotlib + scikit-image).
- `render_surface_from_plotfile.py` — Renders an AMReX plotfile SDF=0 surface (requires pyAMReX + yt).

---

### `examples/` contents

Standalone runnable demos. Outputs (PNG, HTML, NPY) are written to `examples/`.

| File | Description |
|------|-------------|
| `union_example.py` | Two spheres joined with `Union3D` |
| `intersection_example.py` | Sphere–sphere intersection |
| `subtraction_example.py` | Sphere with spherical cavity via `Subtraction3D` |
| `elongation_example.py` | Sphere elongated into a capsule |
| `complex_example.py` | Chains all four operations, one PNG per step |
| `nato_stanag_4496_test.py` | NATO fragment impact scene (no AMReX required) |
| `stl_sdf_demo.py` | Downloads ISS wrench STL, samples SDF, saves interactive Plotly HTML |

---

### Design

```
User parameters
      ↓
Geometry classes      STL file
(sdf2d / sdf3d)          ↓
      ↓              stl2sdf.load_stl
SDF evaluation              ↓
(primitives.py)      stl2sdf.mesh_to_sdf
      ↓                     ↓
Level-set field   φ(x, y[, z]) on a grid
      ↓
Output:  NumPy ndarray  OR  AMReX MultiFab
```

- **NumPy path** (no AMReX): `sample_levelset_2d` / `sample_levelset_3d` / `sample_sdf_from_stl` → `np.ndarray`
- **AMReX path**: `SDFLibrary2D` / `SDFLibrary3D` → `amr.MultiFab`
