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

## Installation

See **[INSTALLATION.md](INSTALLATION.md)** for detailed installation instructions.

**Quick install:**
```bash
# Basic
pip install -e .

# With visualization (plotly, matplotlib, scikit-image)
pip install -e .[viz]

# With AMReX support
pip install -e .[amrex]

# All features
pip install -e .[viz,amrex]
```

After installation, both **2D** (`sdf2d`) and **3D** (`sdf3d`) APIs are available.

## Documentation

- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)**: Complete API reference with all functions, operations, and examples
- **[LIBRARY_STRUCTURE.md](LIBRARY_STRUCTURE.md)**: Folder structure and design overview
- **[examples/](examples/)**: Working examples with mathematical verification

## Files

- `sdf_lib.py`: numpy implementations of the SDF primitives and operators.
- `scripts/`: visualization and plotfile utilities (optional).
- `outputs/`: generated images (not required for the library).
- `sdf3d/complex`: contains complex 3D geometries like Rocket assembly and Nato Stanag fragment.

## Run

```bash
python scripts/render_all_sdfs.py
```

For library usage in Python:

**3D API** (import via `sdf3d`):
```python
from sdf3d import Sphere, sample_levelset, SDFLibrary
```

**2D API** (import via `sdf2d`):
```python
from sdf2d import Circle, Box2D, sample_levelset_2d, SDFLibrary2D
```

If you want AMReX-native output (MultiFab instead of NumPy), use
`SDFLibrary`:

```python
import amrex.space3d as amr
from sdf3d import SDFLibrary

# Always initialize/finalize AMReX in scripts
amr.initialize([])

# Build AMReX grid objects
real_box = amr.RealBox([-1, -1, -1], [1, 1, 1])
domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(63, 63, 63))
geom = amr.Geometry(domain, real_box, 0, [0, 0, 0])
ba = amr.BoxArray(domain)
ba.max_size(32)
dm = amr.DistributionMapping(ba)

lib = SDFLibrary(geom, ba, dm)
mf = lib.sphere(center=(0.0, 0.0, 0.0), radius=0.3)

amr.finalize()
```

### Example: MultiFab union

```python
import amrex.space3d as amr
from sdf3d import SDFLibrary

amr.initialize([])
try:
    real_box = amr.RealBox([-1, -1, -1], [1, 1, 1])
    domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(63, 63, 63))
    geom = amr.Geometry(domain, real_box, 0, [0, 0, 0])
    ba = amr.BoxArray(domain)
    ba.max_size(32)
    dm = amr.DistributionMapping(ba)

    lib = SDFLibrary(geom, ba, dm)
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
from sdf3d import Sphere, sample_levelset

sphere = Sphere(0.3)
bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
res = (64, 64, 64)

phi = sample_levelset(sphere, bounds, res)

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

### 3D volume renders (yt)

This uses yt volume rendering to make 3D snapshots and saves them to
`outputs/vis3d/`:

```bash
python scripts/render_all_sdfs_3d.py
```

Notes:
- Requires `yt` (`pip install yt`).
- Rendering all shapes can take a while; reduce `n` in `render_all_sdfs_3d.py`
  if you need faster previews.

### 3D renders from AMReX plotfiles (yt)

If you have AMReX plotfiles (e.g., `plt00000/`), yt can load them directly
and you can volume render or slice them. Example:

```python
import yt

ds = yt.load("./plt00000/")
sc = yt.create_scene(ds, field="sdf")
sc.save("rendering.png")
```

This is useful when you want to render the full AMReX dataset (including
refinement) instead of the uniform grid used in `render_all_sdfs_3d.py`.

## Output

Generated images are stored under `outputs/`:

- `outputs/vis/<name>.png` (2D slices)
- `outputs/vis3d/<name>.png` (3D renders)

### Plotfiles + 3D snapshots (yt)

This workflow writes AMReX plotfiles for each SDF and then renders each
plotfile to a PNG:

```bash
python scripts/render_all_sdfs_plotfiles.py
```

Outputs:
- `plotfiles/<name>/` (AMReX plotfiles)
- `outputs/vis3d_plotfile/<name>.png` (renders)

Notes:
- Requires pyAMReX built in 3D and `yt` installed.

## Color guide for `outputs/vis` images

The PNGs use a diverging colormap:

- Blue tones: negative SDF values (inside the shape).
- Red tones: positive SDF values (outside the shape).
- White/near zero: near the surface.
- Black contour line: the zero level set (the shape boundary).

For `outputs/vis3d`, the volume render highlights values near the zero level set,
so the visible surface corresponds to the SDF = 0 boundary.

## How pyAMReX helps

pyAMReX provides the grid/mesh infrastructure that makes it easy to evaluate
an SDF on a structured domain and scale the work later if needed:

- `BoxArray` defines how the domain is split into tiles.
- `DistributionMapping` assigns each tile to a compute resource.
- `MultiFab` stores grid-aligned data (the SDF values) for each tile.

In `scripts/render_all_sdfs.py`, the line below allocates the SDF storage:

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

These are the core AMReX objects used in `render_all_sdfs.py`, explained in
plain terms:

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
