# pySdf — Signed Distance Functions in Python

A library of signed distance functions (SDFs) for 2D and 3D geometry, implemented in pure NumPy.
SDF formulas are adapted from [iquilezles.org](https://iquilezles.org/articles/distfunctions/).
Optional [pyAMReX](https://pyamrex.readthedocs.io/en/latest/) integration provides `MultiFab` output for parallel solvers.

## Documentation

- **[INSTALLATION.md](docs/INSTALLATION.md)**: Detailed installation and troubleshooting
- **[API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)**: Complete API reference for all packages
- **[LIBRARY_STRUCTURE.md](docs/LIBRARY_STRUCTURE.md)**: Folder structure and design overview
- **[examples/](examples/)**: Standalone runnable examples

## Packages overview

| Package | Purpose |
|---------|---------|
| `sdf2d` | 2D geometry classes, grid sampling, optional AMReX output |
| `sdf3d` | 3D geometry classes, grid sampling, optional AMReX output |
| `stl2sdf` | Convert STL mesh → SDF grid (pure NumPy, watertight meshes only) |

## Files

- `_sdf_common.py` — Shared math helpers (`vec2`/`vec3`, `opUnion`, `opSubtraction`, …)
- `sdf2d/` — 2D package: `Circle2D`, `Box2D`, `Hexagon2D`, … (~50 shapes)
- `sdf3d/` — 3D package: `Sphere3D`, `Box3D`, `Torus3D`, … (~30 shapes + warps)
- `sdf3d/examples/` — High-level assemblies (`NATOFragment`, `RocketAssembly`)
- `stl2sdf/` — STL mesh → SDF: `stl_to_geometry`
- `tests/` — pytest suite; no AMReX required (`test_amrex.py` skips automatically)
- `scripts/` — Gallery scripts and AMReX plotfile renderer
- `examples/` — Standalone demos; outputs written to `examples/`

## Running tests

```bash
uv run pytest tests/ -v
```

All tests pass without AMReX. `tests/test_amrex.py` skips automatically via `pytest.importorskip`.

## Shapes

![sdf2d gallery](docs/gallery_2d.png)

_Blue = inside (φ < 0), red = outside (φ > 0), white contour = surface (φ = 0)._

![sdf3d 3D gallery](docs/gallery_3d.png)

_Gold isosurfaces extracted from 3D SDF grids using marching cubes._

## Library usage

### `sdf2d` — 2D geometry

```python
from sdf2d import Circle2D, Box2D, Union2D, sample_levelset_2d

circle = Circle2D(radius=0.3)
box    = Box2D(half_size=(0.4, 0.2)).translate(0.5, 0.0)
shape  = circle.union(box)

phi = sample_levelset_2d(shape, bounds=((-1,1), (-1,1)), resolution=(128, 128))
# phi.shape == (128, 128);  phi < 0 inside, phi > 0 outside
```

### `sdf3d` — 3D geometry

```python
from sdf3d import Sphere3D, Box3D, Union3D, sample_levelset_3d

sphere = Sphere3D(radius=0.3)
box    = Box3D(half_size=(0.2, 0.2, 0.2)).translate(0.4, 0.0, 0.0)
shape  = Union3D(sphere, box)

phi = sample_levelset_3d(shape, bounds=((-1,1),(-1,1),(-1,1)), resolution=(64,64,64))
# phi.shape == (64, 64, 64);  phi < 0 inside, phi > 0 outside
```

### `stl2sdf` — STL mesh to SDF

```python
from stl2sdf import sample_sdf_from_stl

phi = sample_sdf_from_stl(
    "my_mesh.stl",
    bounds=((x0,x1), (y0,y1), (z0,z1)),
    resolution=(64, 64, 64),
)
# phi.shape == (64, 64, 64);  requires watertight mesh
```

See `examples/stl_sdf_demo.py` for a full demo that downloads the ISS ratchet wrench
STL (the first object 3D-printed in space, Dec 2014) and renders an interactive Plotly
figure.

```bash
uv run python examples/stl_sdf_demo.py --res 20   # quick draft
uv run python examples/stl_sdf_demo.py --res 40   # full quality
```

### AMReX output (optional)

```python
import amrex.space3d as amr
from sdf3d import SDFLibrary3D

amr.initialize([])
try:
    real_box = amr.RealBox([-1,-1,-1], [1,1,1])
    domain   = amr.Box(amr.IntVect(0,0,0), amr.IntVect(63,63,63))
    geom     = amr.Geometry(domain, real_box, 0, [0,0,0])
    ba       = amr.BoxArray(domain); ba.max_size(32)
    dm       = amr.DistributionMapping(ba)

    lib = SDFLibrary3D(geom, ba, dm)
    mf  = lib.sphere(center=(0,0,0), radius=0.3)
    # mf is an amr.MultiFab
finally:
    amr.finalize()
```

| Path | Requires | Returns | Use case |
|------|----------|---------|----------|
| **NumPy** | `numpy` only | `np.ndarray` | design, testing, visualization |
| **AMReX** | pyAMReX via conda | `amr.MultiFab` | parallel solver input |

Both paths use identical SDF math from `primitives.py`.