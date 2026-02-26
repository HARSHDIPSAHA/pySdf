# Signed Distance Functions with pyAMReX

This repo implements the signed distance functions and operators listed on
the Inigo Quilez "Distance Functions" article, and evaluates them with
pyAMReX on a 2D grid (z = 0 slice) to generate visualization PNGs. The
visualization workflow uses pyAMReX to build a structured grid, split it into
boxes, distribute those boxes, and store the SDF values in a `MultiFab`.
[pyAMReX](https://pyamrex.readthedocs.io/en/latest/) and the original SDF
formulas are referenced from
[iquilezles.org](https://iquilezles.org/articles/distfunctions/).


<img width="746" height="636" alt="Screenshot 2026-02-03 191159" src="https://github.com/user-attachments/assets/1ca854f8-edfe-4094-8316-355e621f5056" />

⚠️ WARNING: This branch is a work in progress and mostly vibe-coded. Todolist:
- EqTriangle, TriangleIsosceles, Hexagram, RoundedCross, Egg, Heart, Cross, Ellipse, Parabola, Stairs, QuadraticCircle, Hyperbola; these do not render properly in the gallery, and many of their formulas had to deviate from the original iquilez formulas to avoid NaNs or other issues.
- Some of the renders in gallery_3d render as elongated spheroids instead of spheres.
- The scripts running in the examples folder do not work anymore; needs updating.

## Installation

### Core library (NumPy only — no AMReX required)

```bash
pip install -e .
```

This gives you full access to `sdf2d`, `sdf3d`, `sdf_lib`, and all tests.

### With visualization extras (galleries, scikit-image for marching cubes)

```bash
pip install -e .[viz]
# or just: pip install matplotlib scikit-image
```

### With pyAMReX (optional — for MultiFab/parallel grid output)

**pyAMReX is not on PyPI.** Choose one of the methods below.

#### Option A — conda (CPU only, easiest)

```bash
conda create -n pyamrex -c conda-forge pyamrex
conda activate pyamrex
pip install -e .
```

#### Option B — build from source (GPU / MPI / custom dimensions)

```bash
# Prerequisites
python3 -m pip install -U pip build packaging setuptools wheel pytest
git clone https://github.com/AMReX-Codes/pyamrex.git $HOME/src/pyamrex
cd $HOME/src/pyamrex

# Configure — build all three space dimensions in one go
cmake -S . -B build -DAMReX_SPACEDIM="1;2;3"
cmake --build build -j 4 --target pip_install

# Optional flags
#   -DAMReX_GPU_BACKEND=CUDA   (or HIP, SYCL)
#   -DAMReX_MPI=ON
#   -DAMReX_OMP=ON
#   -DCMAKE_BUILD_TYPE=Release
```

> **Dimension constraint:** Only one space dimension can be imported per Python
> process (`amrex.space2d` *or* `amrex.space3d`, not both simultaneously).

See the full guides:
[conda install](https://pyamrex.readthedocs.io/en/latest/install/users.html) ·
[cmake build](https://pyamrex.readthedocs.io/en/latest/install/cmake.html)

> **Note:** The `[amrex]` extra in `setup.py` is a documentation placeholder —
> pyAMReX must be installed through conda or built from source, not pip.

After installation, both **2D** (`sdf2d`) and **3D** (`sdf3d`) APIs are available.

## Documentation

- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)**: Complete API reference with all functions, operations, and examples
- **[LIBRARY_STRUCTURE.md](LIBRARY_STRUCTURE.md)**: Folder structure and design overview
- **[examples/](examples/)**: Working examples with mathematical verification

## Files

- `sdf_lib.py`: numpy implementations of all SDF primitives and operators.
- `sdf2d/`: 2D geometry package (`Circle2D`, `Box2D`, `Hexagon2D`, …).
- `sdf3d/`: 3D geometry package (`Sphere3D`, `Box3D`, `Torus3D`, …).
- `sdf3d/examples/`: high-level assemblies (`NATOFragment`, `RocketAssembly`).
- `tests/`: pytest suite — no AMReX required.
- `scripts/`: visualization and plotfile utilities (optional).
- `gallery_2d.png`, `gallery_3d.png`: pre-rendered shape galleries (see below).

## Render galleries

Generate a single-page PNG showing every shape in the library:

```bash
# All 43 sdf2d shapes (requires matplotlib only)
python scripts/gallery_2d.py --out gallery_2d.png

# All sdf_lib 3D primitives (requires matplotlib + scikit-image)
python scripts/gallery_3d.py --out gallery_3d.png --res 64
```

Both scripts run without AMReX.

### 2D shape gallery (`sdf2d` — 43 shapes)

![sdf2d gallery](gallery_2d.png)

_Blue = inside (φ < 0), red = outside (φ > 0), white contour = surface (φ = 0)._

### 3D shape gallery (`sdf_lib` 3D primitives)

![sdf_lib 3D gallery](gallery_3d.png)

_Gold isosurfaces extracted from 3D SDF grids using marching cubes._

## Library usage in Python

**3D API** (import via `sdf3d`):
```python
from sdf3d import Sphere3D, sample_levelset_3d, SDFLibrary3D
```

**2D API** (import via `sdf2d`):
```python
from sdf2d import Circle2D, Box2D, sample_levelset_2d, SDFLibrary2D
```

If you want AMReX-native output (MultiFab instead of NumPy), use
`SDFLibrary3D`:

```python
import amrex.space3d as amr
from sdf3d import SDFLibrary3D

# Always initialize/finalize AMReX in scripts
amr.initialize([])

# Build AMReX grid objects
real_box = amr.RealBox([-1, -1, -1], [1, 1, 1])
domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(63, 63, 63))
geom = amr.Geometry(domain, real_box, 0, [0, 0, 0])
ba = amr.BoxArray(domain)
ba.max_size(32)
dm = amr.DistributionMapping(ba)

lib = SDFLibrary3D(geom, ba, dm)
mf = lib.sphere(center=(0.0, 0.0, 0.0), radius=0.3)

amr.finalize()
```

### Example: MultiFab union

```python
import amrex.space3d as amr
from sdf3d import SDFLibrary3D

amr.initialize([])
try:
    real_box = amr.RealBox([-1, -1, -1], [1, 1, 1])
    domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(63, 63, 63))
    geom = amr.Geometry(domain, real_box, 0, [0, 0, 0])
    ba = amr.BoxArray(domain)
    ba.max_size(32)
    dm = amr.DistributionMapping(ba)

    lib = SDFLibrary3D(geom, ba, dm)
    a = lib.sphere(center=(-0.3, 0.0, 0.0), radius=0.25)
    b = lib.sphere(center=(0.3, 0.0, 0.0), radius=0.25)
    u = lib.union(a, b)

    mins, maxs = [], []
    for mfi in u:
        arr = u.array(mfi).to_numpy()
        vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
        mins.append(vals.min())
        maxs.append(vals.max())
    print("union min/max:", min(mins), max(maxs))
finally:
    amr.finalize()
```

### Quick correctness check (beginner friendly)

Run this small script to verify the SDF output for a sphere:

```bash
python - << "PY"
import numpy as np
from sdf3d import Sphere3D, sample_levelset_3d

sphere = Sphere3D(0.3)
bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
res = (64, 64, 64)

phi = sample_levelset_3d(sphere, bounds, res)

center = tuple(r // 2 for r in res)
print("phi(center) ~ -0.3:", phi[center])
print("phi(corner) > 0:", phi[0, 0, 0])
print("any near zero:", (np.abs(phi) < 0.02).any())
PY
```

Expected:
- `phi(center)` is close to `-0.3`
- `phi(corner)` is positive
- `any near zero` is `True`

Why `phi(center)` may not be exactly `-0.3`:
the grid is cell-centered, so the "center cell" is slightly offset from
`(0, 0, 0)`. If you want a value closer to `-0.3`, use an odd resolution
such as `res = (65, 65, 65)`.

### Render a single AMReX plotfile (requires pyAMReX + yt)

```bash
python scripts/render_surface_from_plotfile.py plotfiles/sdSphere --out out.png
```

## Color guide

**2D gallery** uses a diverging colormap: blue = inside (φ < 0), red = outside (φ > 0), white contour = surface (φ = 0).

**3D gallery** shows the φ = 0 isosurface extracted with marching cubes and rendered with diffuse shading.

## How pyAMReX helps (optional path)

The library has two independent evaluation paths:

| Path | Requires | Use case |
|------|----------|----------|
| **NumPy** | `numpy` only | geometry design, visualization, testing |
| **AMReX** | pyAMReX via conda | parallel grid output for solvers |

When AMReX is installed, `SDFLibrary2D`/`SDFLibrary3D` use the same numpy SDF
formulas but wrap them in AMReX `MultiFab` infrastructure for parallel execution.

pyAMReX provides the grid/mesh infrastructure:

- `BoxArray` defines how the domain is split into tiles.
- `DistributionMapping` assigns each tile to a compute resource.
- `MultiFab` stores grid-aligned data (the SDF values) for each tile.

```python
sdf_mf = amr.MultiFab(ba, dm, 1, 0)
```

`MultiFab(BoxArray, DistributionMapping, ncomp, ngrow)` means:

- `ba`: how the domain is split into boxes.
- `dm`: who owns each box (CPU/GPU/threads).
- `1`: one component per cell (the SDF value).
- `0`: no ghost cells.

Because `ncomp = 1` and `ngrow = 0`, each tile is accessed as
`arr[:, :, 0, 0]`, which is the scalar SDF field for that box.

## Short syntax guide

These are the core AMReX objects, explained in plain terms:

- `amr.RealBox(prob_lo, prob_hi)`: defines the physical bounds of the domain
  (e.g., x and y from 0 to 1).
- `amr.Box(lo, hi)`: defines the integer index region (grid indices), like
  i = 0..n-1 and j = 0..n-1.
- `amr.Geometry(domain, real_box, coord, is_periodic)`: ties index space to
  physical space and stores geometry info such as cell spacing (`dx`).
- `amr.BoxArray(domain)`: describes how the index domain is split into boxes.
  `max_size` limits each box size for parallelism and cache efficiency.
- `amr.DistributionMapping(ba)`: assigns those boxes to compute resources.
- `amr.MultiFab(ba, dm, ncomp, ngrow)`: stores grid data for each box.
  Here it stores one SDF value per cell with no ghost cells.

## Notes

- All 3D SDFs are evaluated on the z = 0 slice for visualization.
- `udTriangle` and `udQuad` are unsigned distance fields by definition.

## Library Flow

```
User parameters / GUI
        ↓
SDF Geometry Library
        ↓
Compose shapes + operations
        ↓
Evaluate SDF on bounding box grid
        ↓
Output: ϕ(x, y, z) level-set data
        ↓
Solver reads it
```
