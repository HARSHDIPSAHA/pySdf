# CLAUDE.md — Project context for Claude Code

## What this project is

`pySdf` is a Python library of signed distance functions (SDFs) for 2D and 3D geometry.
It has two modes of operation:

- **Pure numpy** (no external dependencies beyond numpy/scipy): evaluate SDFs on grids
  using `sample_levelset_2d` / `sample_levelset_3d`.
- **AMReX** (optional): fill `MultiFab` grids via `SDFLibrary2D` / `SDFLibrary3D`.

## Repository layout

```
pySdf/
├── sdf_lib.py            # All SDF math — pure numpy, no AMReX
├── sdf2d/
│   ├── __init__.py       # Exports all 2D classes
│   ├── geometry.py       # Circle2D, Box2D, ... + Union2D, Intersection2D, Subtraction2D
│   ├── grid.py           # sample_levelset_2d(geom, bounds, resolution) -> ndarray
│   └── amrex.py          # SDFLibrary2D (requires amrex.space2d)
├── sdf3d/
│   ├── __init__.py       # Exports all 3D classes
│   ├── geometry.py       # Sphere3D, Box3D, ... + Union3D, Intersection3D, Subtraction3D
│   ├── grid.py           # sample_levelset_3d(geom, bounds, resolution) -> ndarray
│   ├── amrex.py          # SDFLibrary3D (requires amrex.space3d)
│   └── examples/
│       ├── nato_stanag.py      # NATOFragment(lib, diameter, L_over_D, cone_angle_deg)
│       └── rocket_assembly.py  # RocketAssembly(lib, body_radius, ...)
├── tests/                # 215 passed, 1 skipped (AMReX)
├── scripts/
│   ├── gallery_2d.py           # All sdf2d shapes on one matplotlib page
│   ├── gallery_3d.py           # All sdf_lib 3D shapes (marching cubes)
│   └── render_surface_from_plotfile.py  # AMReX plotfile -> PNG (needs pyAMReX+yt)
├── examples/             # Standalone runnable examples (no AMReX required)
├── gallery_2d.png        # Pre-rendered 2D gallery (repo root)
└── gallery_3d.png        # Pre-rendered 3D gallery (repo root)
```

## SDF sign convention

- `phi < 0` — inside the solid
- `phi = 0` — on the surface
- `phi > 0` — outside the solid

## Key naming conventions

- 3D geometry: `Sphere3D`, `Box3D`, `Union3D`, `Intersection3D`, `Subtraction3D`
- 2D geometry: `Circle2D`, `Box2D`, `Union2D`, `Intersection2D`, `Subtraction2D`
- Grid functions: `sample_levelset_2d` / `sample_levelset_3d`
- AMReX classes: `SDFLibrary2D` / `SDFLibrary3D`

## Running tests

```bash
pytest tests/        # 215 passed, 1 skipped (test_amrex.py without pyAMReX)
```

All tests pass without AMReX. `tests/test_amrex.py` skips automatically via
`pytest.importorskip`.

## Running the gallery scripts

```bash
python scripts/gallery_2d.py          # saves gallery_2d.png
python scripts/gallery_3d.py          # saves gallery_3d.png
python scripts/gallery_3d.py --res 48 # faster draft render
```

## AMReX installation

pyAMReX is **not on PyPI**. Install via conda:

```bash
conda create -n pyamrex -c conda-forge pyamrex
```

Or build from source: https://pyamrex.readthedocs.io/en/latest/install/cmake.html

## Critical design decisions

### opSubtraction argument order
`opSubtraction(d1, d2) = max(-d1, d2)` — d1 is the CUTTER, d2 is the BASE.
- `Subtraction3D(base, cutter)` calls `opSubtraction(cutter.sdf(p), base.sdf(p))`
- `a.subtract(b)` means "subtract b from a" — b is the cutter

### GLSL-to-numpy simultaneous update
`p -= 2.0*min(dot(k,p),0.0)*k` in GLSL updates both components simultaneously.
Python sequential `px=...; py=...` is wrong. Fix: compute scalar once, then apply.
Affects: `sdPentagon2D`, `sdHexagon2D`, `sdOctagon2D`, `sdHexagram2D`, `sdStar5`

### np.where evaluates both branches
`np.where(cond, A, B)` computes both A and B for all elements. Operations like
`sqrt` or `arccos` on values only valid for some elements produce NaN in the
other branch that can leak through. Guard with `np.maximum(..., 0)` and `np.clip`.
