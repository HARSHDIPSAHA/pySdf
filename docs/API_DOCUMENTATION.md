# SDF Library API Documentation

Reference guide for the `sdf2d`, `sdf3d`, and `stl2sdf` packages.

## Table of Contents

1. [Installation](#installation)
2. [What is implemented](#what-is-implemented)
3. [Quick start](#quick-start)
4. [2D API — `sdf2d`](#2d-api--sdf2d)
5. [3D API — `sdf3d`](#3d-api--sdf3d)
6. [STL → SDF — `stl2sdf`](#stl--sdf--stl2sdf)
7. [AMReX integration](#amrex-integration)
8. [Low-level math — `sdf2d.primitives` / `sdf3d.primitives`](#low-level-math--sdf2dprimitives--sdf3dprimitives)
9. [Tips](#tips)

---

## Installation

```bash
# Core library (numpy only)
uv sync

# With visualization (plotly, matplotlib, scikit-image)
uv sync --extra viz

# With dev tools
uv sync --extra dev
```

pyAMReX is not on PyPI. See [INSTALLATION.md](INSTALLATION.md) for all three
methods (conda, source build, wheel injection).

---

## What is implemented

| Feature | Where |
|---------|-------|
| ~50 2D SDF primitives | `sdf2d.primitives`, `sdf2d.geometry` |
| ~30 3D SDF primitives + warps + smooth ops | `sdf3d.primitives`, `sdf3d.geometry` |
| Boolean ops (Union, Intersection, Subtraction) | both packages |
| Smooth boolean ops (smooth union/subtraction/intersection) | `sdf3d.primitives` |
| Transforms (translate, rotate, scale, round, onion, elongate) | both packages |
| Grid sampling to NumPy arrays | `sdf2d.grid`, `sdf3d.grid` |
| STL mesh → SDF grid | `stl2sdf` |
| AMReX MultiFab output | `sdf2d.amrex`, `sdf3d.amrex` |
| Complex assemblies | `sdf3d.examples` (`NATOFragment`, `RocketAssembly`) |
| 308 unit tests | `tests/` |

---

## Quick start

### NumPy mode (no AMReX)

```python
from sdf3d import Sphere3D, Box3D, Union3D, sample_levelset_3d
import numpy as np

sphere = Sphere3D(radius=0.3)
box    = Box3D(half_size=(0.2, 0.2, 0.2)).translate(0.4, 0.0, 0.0)
shape  = Union3D(sphere, box)

phi = sample_levelset_3d(shape, bounds=((-1,1),(-1,1),(-1,1)), resolution=(64,64,64))
# phi.shape == (64, 64, 64);  phi < 0 inside, phi > 0 outside
```

### AMReX mode

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

    lib      = SDFLibrary3D(geom, ba, dm)
    levelset = lib.sphere(center=(0,0,0), radius=0.3)
    # levelset is an amr.MultiFab
finally:
    amr.finalize()
```

---

## 2D API — `sdf2d`

### Base class

```python
from sdf2d import Geometry2D
```

`Geometry2D(func)` wraps any callable `func(p: ndarray) -> ndarray` where `p` has shape `(..., 2)`.

#### Methods on every geometry

| Method | Signature | Returns |
|--------|-----------|---------|
| `sdf` | `sdf(p)` | signed distance values |
| `translate` | `translate(tx, ty)` | `Geometry2D` |
| `rotate` | `rotate(angle_rad)` | `Geometry2D` |
| `scale` | `scale(factor)` | `Geometry2D` |
| `round` | `round(radius)` | `Geometry2D` |
| `onion` | `onion(thickness)` | `Geometry2D` |
| `union` | `union(other)` | `Geometry2D` |
| `subtract` | `subtract(other)` | `Geometry2D` |
| `intersect` | `intersect(other)` | `Geometry2D` |

### Primitive shapes

```python
from sdf2d import (
    Circle2D, Box2D, RoundedBox2D, OrientedBox2D,
    Segment2D, Rhombus2D, Trapezoid2D, Parallelogram2D,
    EquilateralTriangle2D, TriangleIsosceles2D, Triangle2D,
    UnevenCapsule2D,
    Pentagon2D, Hexagon2D, Octagon2D, NGon2D,
    Hexagram2D, Star2D,
    Pie2D, CutDisk2D, Arc2D, Ring2D, Horseshoe2D,
    Vesica2D, Moon2D, RoundedCross2D, Egg2D, Heart2D,
    Cross2D, RoundedX2D,
    Polygon2D, Ellipse2D, Parabola2D, ParabolaSegment2D,
    Bezier2D, BlobbyCross2D, Tunnel2D, Stairs2D,
    QuadraticCircle2D, Hyperbola2D,
)
```

| Class | Constructor |
|-------|-------------|
| `Circle2D` | `Circle2D(radius)` |
| `Box2D` | `Box2D(half_size)` — `half_size = (hx, hy)` |
| `RoundedBox2D` | `RoundedBox2D(half_size, radius)` |
| `OrientedBox2D` | `OrientedBox2D(point_a, point_b, width)` |
| `Segment2D` | `Segment2D(a, b)` — unsigned distance to segment |
| `Rhombus2D` | `Rhombus2D(half_size)` |
| `Trapezoid2D` | `Trapezoid2D(r1, r2, height)` |
| `Parallelogram2D` | `Parallelogram2D(width, height, skew)` |
| `EquilateralTriangle2D` | `EquilateralTriangle2D(size)` |
| `TriangleIsosceles2D` | `TriangleIsosceles2D(width, height)` |
| `Triangle2D` | `Triangle2D(p0, p1, p2)` |
| `UnevenCapsule2D` | `UnevenCapsule2D(r1, r2, height)` |
| `Pentagon2D` | `Pentagon2D(radius)` |
| `Hexagon2D` | `Hexagon2D(radius)` — `radius` is the inradius (flat-face distance) |
| `Octagon2D` | `Octagon2D(radius)` |
| `NGon2D` | `NGon2D(radius, n_sides)` |
| `Hexagram2D` | `Hexagram2D(radius)` |
| `Star2D` | `Star2D(radius, n_points, m)` — `2 ≤ m ≤ n_points`; m=2 sharpest, m=n_points → regular polygon |
| `Pie2D` | `Pie2D(sin_cos, radius)` |
| `CutDisk2D` | `CutDisk2D(radius, cut_height)` — the circular cap above `y = cut_height` |
| `Arc2D` | `Arc2D(sin_cos, radius, thickness)` |
| `Ring2D` | `Ring2D(inner_radius, outer_radius)` |
| `Horseshoe2D` | `Horseshoe2D(sin_cos, radius, widths)` |
| `Vesica2D` | `Vesica2D(radius, offset)` |
| `Moon2D` | `Moon2D(d, ra, rb)` |
| `RoundedCross2D` | `RoundedCross2D(size)` |
| `Egg2D` | `Egg2D(ra, rb)` |
| `Heart2D` | `Heart2D()` |
| `Cross2D` | `Cross2D(size, r)` |
| `RoundedX2D` | `RoundedX2D(size, r)` |
| `Polygon2D` | `Polygon2D(vertices)` — list of `[x, y]` |
| `Ellipse2D` | `Ellipse2D(radii)` — `radii = (rx, ry)` |
| `Parabola2D` | `Parabola2D(k)` |
| `ParabolaSegment2D` | `ParabolaSegment2D(width, height)` |
| `Bezier2D` | `Bezier2D(a, b, c)` — quadratic Bézier, unsigned |
| `BlobbyCross2D` | `BlobbyCross2D(size)` |
| `Tunnel2D` | `Tunnel2D(size)` — `size = (wx, wy)` |
| `Stairs2D` | `Stairs2D(size, n)` — `size = (tread, rise)` |
| `QuadraticCircle2D` | `QuadraticCircle2D()` |
| `Hyperbola2D` | `Hyperbola2D(k, he)` |

### Boolean operations

```python
from sdf2d import Union2D, Intersection2D, Subtraction2D

u = Union2D(a, b)            # SDF = min(a, b)
i = Intersection2D(a, b)     # SDF = max(a, b)
s = Subtraction2D(base, cutter)   # SDF = max(-cutter, base)

# Equivalent method syntax:
u = a.union(b)
i = a.intersect(b)
s = a.subtract(b)   # "subtract b from a"
```

`Union2D` accepts more than two arguments: `Union2D(a, b, c, ...)`.

### Grid sampling

```python
from sdf2d import sample_levelset_2d, save_npy

phi = sample_levelset_2d(
    geom,                          # Geometry2D
    bounds=((-1,1), (-1,1)),       # ((xlo,xhi), (ylo,yhi))
    resolution=(nx, ny),
)
# phi.shape == (ny, nx)  — y-first

save_npy("output/levelset.npy", phi)  # creates parent dirs automatically
```

### AMReX (2D)

```python
from sdf2d import SDFLibrary2D
import amrex.space2d as amr

lib = SDFLibrary2D(geom, ba, dm)

mf = lib.circle(center=(cx, cy), radius=r)
mf = lib.box(center=(cx, cy), size=(hx, hy))
mf = lib.rounded_box(center=(cx, cy), size=(hx, hy), radius=r)
mf = lib.hexagon(center=(cx, cy), radius=r)
mf = lib.from_geometry(geom_obj)   # any Geometry2D

mf = lib.union(mf1, mf2)
mf = lib.subtract(base, cutter)
mf = lib.intersect(mf1, mf2)
mf = lib.negate(mf)
```

---

## 3D API — `sdf3d`

### Base class

```python
from sdf3d import Geometry3D
```

`Geometry3D(func)` wraps any callable `func(p: ndarray) -> ndarray` where `p` has shape `(..., 3)`.

#### Methods on every geometry

| Method | Signature | Returns |
|--------|-----------|---------|
| `sdf` | `sdf(p)` | signed distance values |
| `translate` | `translate(tx, ty, tz)` | `Geometry3D` |
| `rotate_x` | `rotate_x(angle_rad)` | `Geometry3D` |
| `rotate_y` | `rotate_y(angle_rad)` | `Geometry3D` |
| `rotate_z` | `rotate_z(angle_rad)` | `Geometry3D` |
| `scale` | `scale(factor)` | `Geometry3D` |
| `elongate` | `elongate(hx, hy, hz)` | `Geometry3D` |
| `round` | `round(radius)` | `Geometry3D` |
| `onion` | `onion(thickness)` | `Geometry3D` |
| `union` | `union(other)` | `Geometry3D` |
| `subtract` | `subtract(other)` | `Geometry3D` |
| `intersect` | `intersect(other)` | `Geometry3D` |

### Primitive shapes

```python
from sdf3d import Sphere3D, Box3D, RoundBox3D, Cylinder3D, ConeExact3D, Torus3D
```

| Class | Constructor | Notes |
|-------|-------------|-------|
| `Sphere3D` | `Sphere3D(radius)` | Exact SDF |
| `Box3D` | `Box3D(half_size)` — `(hx, hy, hz)` | Exact SDF |
| `RoundBox3D` | `RoundBox3D(half_size, radius)` | Rounded corners only; flat faces stay at `half_size` |
| `Cylinder3D` | `Cylinder3D(axis_offset, radius)` | Infinite cylinder along Z |
| `ConeExact3D` | `ConeExact3D(sincos, height)` — `[sin θ, cos θ]` | Finite cone |
| `Torus3D` | `Torus3D(radii)` — `(R, r)` | Major/minor radii; lies in XZ plane |

### Boolean operations

```python
from sdf3d import Union3D, Intersection3D, Subtraction3D

u = Union3D(a, b)
i = Intersection3D(a, b)
s = Subtraction3D(base, cutter)   # SDF = max(-cutter, base)

# Method syntax:
u = a.union(b)
i = a.intersect(b)
s = a.subtract(b)   # "subtract b from a"
```

`Union3D` accepts more than two arguments.

### Grid sampling

```python
from sdf3d import sample_levelset_3d, save_npy

phi = sample_levelset_3d(
    geom,
    bounds=((-1,1), (-1,1), (-1,1)),   # ((xlo,xhi), (ylo,yhi), (zlo,zhi))
    resolution=(nx, ny, nz),
)
# phi.shape == (nz, ny, nx)  — z-first

save_npy("output/levelset.npy", phi)
```

### Complex assemblies

```python
from sdf3d.examples import NATOFragment, RocketAssembly
```

Both accept a `lib` argument: pass a real `SDFLibrary3D` for AMReX output,
or a mock object for pure-numpy use (see `tests/test_complex.py`).

#### `NATOFragment`

```python
geom, _ = NATOFragment(
    lib,
    diameter=14.30e-3,    # fragment diameter (m)
    L_over_D=1.09,        # length-to-diameter ratio
    cone_angle_deg=20.0,  # nose cone half-angle (degrees)
)
```

#### `RocketAssembly`

```python
geom, _ = RocketAssembly(
    lib,
    body_radius=0.15,
    L_extra=0.40,
    nose_len=0.25,
    fin_span=0.12,
    fin_height=0.18,
    fin_thickness=0.03,
    n_fins=4,
)
```

### AMReX (3D)

```python
from sdf3d import SDFLibrary3D
import amrex.space3d as amr

lib = SDFLibrary3D(geom, ba, dm)

mf = lib.sphere(center=(cx,cy,cz), radius=r)
mf = lib.box(center=(cx,cy,cz), size=(hx,hy,hz))
mf = lib.round_box(center=(cx,cy,cz), size=(hx,hy,hz), radius=r)
mf = lib.from_geometry(geom_obj)

mf = lib.union(mf1, mf2)
mf = lib.subtract(base, cutter)
mf = lib.intersect(mf1, mf2)
mf = lib.negate(mf)
```

---

## STL → SDF — `stl2sdf`

Converts binary or ASCII STL meshes into SDF grids using pure NumPy.
Requires a **watertight** (closed, 2-manifold) mesh for correct sign determination.
Complexity is O(F × N) — no BVH acceleration.

```python
from stl2sdf import load_stl, mesh_to_sdf, sample_sdf_from_stl
```

### `load_stl`

```python
triangles = load_stl("mesh.stl")
# triangles.shape == (F, 3, 3)  — F triangles, 3 vertices, 3 coords (x,y,z)
# supports binary and ASCII STL
```

### `mesh_to_sdf`

```python
import numpy as np

points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # (N, 3)
phi    = mesh_to_sdf(points, triangles)
# phi.shape == (N,);  phi < 0 inside, phi > 0 outside
```

Optional `ray_dir` parameter overrides the default irrational ray direction used for
sign determination. Avoid axis-aligned directions with axis-aligned meshes.

### `sample_sdf_from_stl`

Grid API matching `sample_levelset_3d` exactly — same cell-centre formula,
same `(nz, ny, nx)` output shape:

```python
phi = sample_sdf_from_stl(
    "mesh.stl",
    bounds=((x0,x1), (y0,y1), (z0,z1)),
    resolution=(nx, ny, nz),
)
# phi.shape == (nz, ny, nx)
```

### Demo

```bash
uv run python examples/stl_sdf_demo.py --res 20   # downloads ISS wrench STL, ~15 s
uv run python examples/stl_sdf_demo.py --res 40   # cleaner surface, ~2-5 min
```

Outputs `examples/wrench_sdf.npy` and `examples/wrench_sdf.html` (interactive Plotly:
2D mid-Z heatmap + 3D isosurface).

---

## AMReX integration

Both `SDFLibrary2D` and `SDFLibrary3D` require AMReX to be installed and initialized:

```python
import amrex.space3d as amr   # or amrex.space2d for 2D

amr.initialize([])
try:
    real_box = amr.RealBox([xlo,ylo,zlo], [xhi,yhi,zhi])
    domain   = amr.Box(amr.IntVect(0,0,0), amr.IntVect(nx-1,ny-1,nz-1))
    geom     = amr.Geometry(domain, real_box, 0, [0,0,0])
    ba       = amr.BoxArray(domain); ba.max_size(32)
    dm       = amr.DistributionMapping(ba)

    from sdf3d import SDFLibrary3D
    lib = SDFLibrary3D(geom, ba, dm)
    mf  = lib.sphere(center=(0,0,0), radius=0.3)

    varnames = amr.Vector_string(["phi"])
    amr.write_single_level_plotfile("output/levelset", mf, varnames, geom, 0.0, 0)
finally:
    amr.finalize()
```

### Reading MultiFab values

```python
for mfi in mf:
    arr = mf.array(mfi).to_numpy()
    phi = arr[..., 0]   # shape (ny, nx[, nz]) — one component, no ghost cells
```

See [INSTALLATION.md](INSTALLATION.md) for all three pyAMReX installation methods.

---

## Low-level math — `sdf2d.primitives` / `sdf3d.primitives`

```python
from sdf3d import primitives as sdf3d
from sdf2d import primitives as sdf2d

p = np.array([0.0, 0.0, 0.0])
d = sdf3d.sdSphere(p, 0.3)    # == -0.3

q = np.array([0.0, 0.0])
d = sdf2d.sdCircle(q, 0.3)    # == -0.3
```

Functions follow `sd<Shape>` (signed) or `ud<Shape>` (unsigned); operators use `op` prefix.

**3D primitives:** `sdSphere`, `sdBox`, `sdRoundBox`, `sdBoxFrame`, `sdTorus`, `sdCappedTorus`, `sdLink`, `sdCylinder`, `sdConeExact`, `sdConeBound`, `sdConeInfinite`, `sdPlane`, `sdHexPrism`, `sdTriPrism`, `sdCapsule`, `sdVerticalCapsule`, `sdCappedCylinder`, `sdCappedCylinderSegment`, `sdRoundedCylinder`, `sdCappedCone`, `sdCappedConeSegment`, `sdSolidAngle`, `sdCutSphere`, `sdCutHollowSphere`, `sdDeathStar`, `sdRoundCone`, `sdRoundConeSegment`, `sdEllipsoid`, `sdVesicaSegment`, `sdRhombus`, `sdOctahedronExact`, `sdOctahedronBound`, `sdPyramid`

**3D unsigned:** `udTriangle`, `udQuad`

**2D primitives:** `sdCircle`, `sdBox2D`, `sdRoundedBox2D`, `sdOrientedBox2D`, `sdSegment`, `sdRhombus2D`, `sdTrapezoid2D`, `sdParallelogram2D`, `sdEquilateralTriangle`, `sdTriangleIsosceles`, `sdTriangle2D`, `sdUnevenCapsule2D`, `sdPentagon`, `sdHexagon`, `sdOctagon`, `sdNGon`, `sdHexagram`, `sdStar`, `sdPie2D`, `sdCutDisk`, `sdArc`, `sdRing`, `sdHorseshoe`, `sdVesica2D`, `sdMoon`, `sdRoundedCross`, `sdEgg`, `sdHeart`, `sdCross`, `sdRoundedX`, `sdPolygon`, `sdEllipse2D`, `sdParabola`, `sdParabolaSegment`, `sdBezier`, `sdBlobbyCross`, `sdTunnel`, `sdStairs`, `sdQuadraticCircle`, `sdHyperbola`

**Boolean:** `opUnion`, `opSubtraction`, `opIntersection`, `opSmoothUnion`, `opSmoothSubtraction`, `opSmoothIntersection`

**Warp/space:** `opRound`, `opOnion`, `opElongate1`, `opElongate2`, `opRevolution`, `opExtrusion`, `opTwist`, `opCheapBend`, `opTx`, `opTx2D`, `opScale`, `opSymX`, `opSymXZ`, `opRepetition`, `opLimitedRepetition`, `opDisplace`

---

## Tips

- **Sign convention:** `phi < 0` inside, `phi = 0` on surface, `phi > 0` outside.
- **Grid layout:** `sample_levelset_3d` / `sample_sdf_from_stl` both return shape `(nz, ny, nx)`. Access as `phi[iz, iy, ix]`.
- **Subtraction argument order:** `Subtraction3D(base, cutter)` — first arg is the base shape, second is what gets cut away. `a.subtract(b)` means "cut b from a".
- **AMReX initialize/finalize:** Always wrap in `try/finally` with `amr.finalize()`.
- **Chaining transforms:**
  ```python
  shape = Sphere3D(0.3).translate(0.5, 0, 0).rotate_z(np.pi/4).scale(1.2)
  ```
- **stl2sdf resolution:** O(F×N) — use `--res 20` for drafts (~15 s for a 14 K-triangle mesh), `--res 40` for quality renders.
- **Watertight check:** Verify your STL has no boundary edges before using `mesh_to_sdf`. See the troubleshooting section in [INSTALLATION.md](INSTALLATION.md).
