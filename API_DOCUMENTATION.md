# SDF Library API Documentation

Complete reference guide for using the Signed Distance Function (SDF) library with pyAMReX.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Importing the Library](#importing-the-library)
3. [AMReX MultiFab Mode (Solver-Native)](#amrex-multifab-mode-solver-native)
4. [Geometry API Mode (NumPy Arrays)](#geometry-api-mode-numpy-arrays)
5. [Available Primitives](#available-primitives)
6. [Boolean Operations](#boolean-operations)
7. [Transform Operations](#transform-operations)
8. [Output Formats](#output-formats)
9. [Complete Examples](#complete-examples)

---

## Quick Start

### AMReX MultiFab Mode (Recommended for Solvers)

```python
import amrex.space3d as amr
from sdf3d import SDFLibrary

amr.initialize([])
try:
    # Setup grid
    real_box = amr.RealBox([-1, -1, -1], [1, 1, 1])
    domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(63, 63, 63))
    geom = amr.Geometry(domain, real_box, 0, [0, 0, 0])
    ba = amr.BoxArray(domain)
    ba.max_size(32)
    dm = amr.DistributionMapping(ba)

    # Create library instance
    lib = SDFLibrary(geom, ba, dm)

    # Generate SDF field
    mf = lib.sphere(center=(0, 0, 0), radius=0.3)
    # mf is an AMReX MultiFab, ready for solvers
finally:
    amr.finalize()
```

### Geometry API Mode (For Visualization/Testing)

```python
from sdf3d import Sphere, sample_levelset
import numpy as np

# Create geometry
sphere = Sphere(0.3)

# Sample on grid
bounds = ((-1, 1), (-1, 1), (-1, 1))
res = (64, 64, 64)
phi = sample_levelset(sphere, bounds, res)
# phi is a numpy array
```

---

## Importing the Library

### For AMReX MultiFab Usage

```python
import amrex.space3d as amr
from sdf3d import SDFLibrary
```

### For Geometry API Usage

```python
from sdf3d import (
    Sphere, Box, RoundBox,
    Cylinder, Cone, Torus,
    # ... see full list below
    sample_levelset
)
```

---

## AMReX MultiFab Mode (Solver-Native)

This mode generates **AMReX MultiFab** objects directly, which is the native format for physics solvers.

### Initialization

```python
import amrex.space3d as amr
from sdf3d import SDFLibrary

amr.initialize([])  # Required!

# Define domain
real_box = amr.RealBox([xlo, ylo, zlo], [xhi, yhi, zhi])
domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(nx-1, ny-1, nz-1))
geom = amr.Geometry(domain, real_box, 0, [0, 0, 0])

# Create box array and distribution mapping
ba = amr.BoxArray(domain)
ba.max_size(max_box_size)  # e.g., 32
dm = amr.DistributionMapping(ba)

# Create SDF library instance
lib = SDFLibrary(geom, ba, dm)
```

### Available Primitives (MultiFab Mode)

#### Sphere

```python
mf = lib.sphere(center=(x, y, z), radius=r)
```

**Parameters:**
- `center`: Tuple `(x, y, z)` - center position
- `radius`: Float - sphere radius

**Returns:** `amr.MultiFab` with SDF values

**Example:**
```python
mf = lib.sphere(center=(0.0, 0.0, 0.0), radius=0.3)
```

#### Box

```python
mf = lib.box(center=(x, y, z), size=(wx, wy, wz))
```

**Parameters:**
- `center`: Tuple `(x, y, z)` - center position
- `size`: Tuple `(wx, wy, wz)` - half-widths in each direction

**Returns:** `amr.MultiFab` with SDF values

**Example:**
```python
mf = lib.box(center=(0.0, 0.0, 0.0), size=(0.25, 0.2, 0.15))
```

#### RoundBox

```python
mf = lib.round_box(center=(x, y, z), size=(wx, wy, wz), radius=r)
```

**Parameters:**
- `center`: Tuple `(x, y, z)` - center position
- `size`: Tuple `(wx, wy, wz)` - half-widths
- `radius`: Float - rounding radius

**Example:**
```python
mf = lib.round_box(center=(0, 0, 0), size=(0.25, 0.25, 0.25), radius=0.05)
```

### Boolean Operations (MultiFab Mode)

All operations work on `MultiFab` objects and return new `MultiFab` objects.

#### Union

```python
result = lib.union(mf1, mf2)
```

**Mathematical Formula:** `min(SDF1, SDF2)`

**Example:**
```python
a = lib.sphere(center=(-0.3, 0, 0), radius=0.25)
b = lib.sphere(center=(0.3, 0, 0), radius=0.25)
combined = lib.union(a, b)
```

#### Intersection

```python
result = lib.intersect(mf1, mf2)
```

**Mathematical Formula:** `max(SDF1, SDF2)`

**Example:**
```python
a = lib.sphere(center=(0, 0, 0), radius=0.35)
b = lib.sphere(center=(0.2, 0, 0), radius=0.35)
overlap = lib.intersect(a, b)
```

#### Subtraction

```python
result = lib.subtract(base, cutter)
```

**Mathematical Formula:** `max(-SDF_base, SDF_cutter)`

**Example:**
```python
base = lib.sphere(center=(0, 0, 0), radius=0.4)
hole = lib.sphere(center=(0.2, 0, 0), radius=0.25)
result = lib.subtract(base, hole)  # Creates a hole
```

#### Negation

```python
result = lib.negate(mf)
```

**Mathematical Formula:** `-SDF`

**Example:**
```python
mf = lib.sphere(center=(0, 0, 0), radius=0.3)
inverted = lib.negate(mf)  # Inside becomes outside
```

---

## Geometry API Mode (NumPy Arrays)

This mode uses geometry objects that can be sampled to NumPy arrays. Useful for visualization and testing.

### Available Primitives (Geometry API)

#### Sphere

```python
from sdf3d import Sphere

sphere = Sphere(radius)
```

**Parameters:**
- `radius`: Float - sphere radius

**Example:**
```python
sphere = Sphere(0.3)
```

#### Box

```python
from sdf3d import Box

box = Box(size)
```

**Parameters:**
- `size`: Tuple `(wx, wy, wz)` - half-widths

**Example:**
```python
box = Box((0.25, 0.2, 0.15))
```

#### RoundBox

```python
from sdf3d import RoundBox

round_box = RoundBox(size, radius)
```

**Parameters:**
- `size`: Tuple `(wx, wy, wz)` - half-widths
- `radius`: Float - rounding radius

**Example:**
```python
rb = RoundBox((0.25, 0.25, 0.25), 0.05)
```

#### Cylinder

```python
from sdf3d import Cylinder

cylinder = Cylinder(radius, height)
```

**Parameters:**
- `radius`: Float - cylinder radius
- `height`: Float - cylinder height

**Example:**
```python
cyl = Cylinder(0.2, 0.4)
```

#### Torus

```python
from sdf3d import Torus

torus = Torus(major_radius, minor_radius)
```

**Parameters:**
- `major_radius`: Float - distance from center to tube center
- `minor_radius`: Float - tube radius

**Example:**
```python
torus = Torus(0.25, 0.08)
```

### Transform Operations (Geometry API)

All geometry objects support chaining operations.

#### Translation

```python
translated = geometry.translate(dx, dy, dz)
```

**Example:**
```python
sphere = Sphere(0.3)
moved = sphere.translate(0.2, 0.1, 0.0)
```

#### Rotation

```python
rotated = geometry.rotate_x(angle_radians)
rotated = geometry.rotate_y(angle_radians)
rotated = geometry.rotate_z(angle_radians)
```

**Example:**
```python
import numpy as np
box = Box((0.25, 0.25, 0.25))
rotated = box.rotate_z(np.pi / 4)  # 45 degrees
```

#### Scaling

```python
scaled = geometry.scale(factor)
```

**Example:**
```python
sphere = Sphere(0.3)
bigger = sphere.scale(1.5)
```

#### Elongation

```python
elongated = geometry.elongate(dx, dy, dz)
```

**Example:**
```python
sphere = Sphere(0.25)
capsule = sphere.elongate(0.3, 0.0, 0.0)  # Elongate in x-direction
```

### Boolean Operations (Geometry API)

#### Union

```python
from sdf3d import Union

combined = Union(geom1, geom2)
```

**Example:**
```python
a = Sphere(0.25)
b = Box((0.2, 0.2, 0.2))
combined = Union(a, b)
```

#### Intersection

```python
from sdf3d import Intersection

overlap = Intersection(geom1, geom2)
```

**Example:**
```python
a = Sphere(0.35)
b = Sphere(0.35).translate(0.2, 0, 0)
overlap = Intersection(a, b)
```

#### Subtraction

```python
from sdf3d import Subtraction

result = Subtraction(base, cutter)
```

**Example:**
```python
base = Sphere(0.4)
hole = Sphere(0.25).translate(0.2, 0, 0)
result = Subtraction(base, hole)
```

### Sampling Geometry to NumPy Array

```python
from sdf3d import sample_levelset

phi = sample_levelset(geometry, bounds, resolution)
```

**Parameters:**
- `geometry`: Geometry object (Sphere, Box, etc. or combined)
- `bounds`: Tuple of tuples `((xlo, xhi), (ylo, yhi), (zlo, zhi))`
- `resolution`: Tuple `(nx, ny, nz)` - grid resolution

**Returns:** NumPy array of shape `(nx, ny, nz)` with SDF values

**Example:**
```python
sphere = Sphere(0.3)
bounds = ((-1, 1), (-1, 1), (-1, 1))
res = (64, 64, 64)
phi = sample_levelset(sphere, bounds, res)
print(phi.shape)  # (64, 64, 64)
print(phi.min(), phi.max())  # negative, positive
```

---

## Output Formats

### AMReX MultiFab

When using `SDFLibrary`, output is an `amr.MultiFab` object.

**Accessing values:**
```python
mf = lib.sphere(center=(0, 0, 0), radius=0.3)

# Iterate over boxes
for mfi in mf:
    arr = mf.array(mfi).to_numpy()
    # arr shape: (ny, nx, nz, ncomp) or (ny, nx, nz, ncomp, ngrow)
    vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
    # Process vals (numpy array for this box)
```

**Writing to plotfile:**
```python
import amrex.space3d as amr

varnames = amr.Vector_string(["sdf"])
amr.write_single_level_plotfile("output/plt00000", mf, varnames, geom, 0.0, 0)
```

### NumPy Array

When using `sample_levelset`, output is a NumPy array.

**Saving:**
```python
import numpy as np

phi = sample_levelset(sphere, bounds, res)
np.save("output/levelset.npy", phi)
```

**Loading:**
```python
phi = np.load("output/levelset.npy")
```

---

## Complete Examples

### Example 1: Complex Shape with AMReX

```python
import amrex.space3d as amr
from sdf3d import SDFLibrary

amr.initialize([])
try:
    # Setup
    real_box = amr.RealBox([-1, -1, -1], [1, 1, 1])
    domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(63, 63, 63))
    geom = amr.Geometry(domain, real_box, 0, [0, 0, 0])
    ba = amr.BoxArray(domain)
    ba.max_size(32)
    dm = amr.DistributionMapping(ba)

    lib = SDFLibrary(geom, ba, dm)

    # Build complex shape: sphere with box cutout
    base = lib.sphere(center=(0, 0, 0), radius=0.4)
    cutter = lib.box(center=(0.2, 0, 0), size=(0.15, 0.15, 0.15))
    result = lib.subtract(base, cutter)

    # Save to plotfile
    varnames = amr.Vector_string(["sdf"])
    amr.write_single_level_plotfile("output/complex", result, varnames, geom, 0.0, 0)
finally:
    amr.finalize()
```

### Example 2: Chained Operations with Geometry API

```python
from sdf3d import Sphere, Box, Union, sample_levelset
import numpy as np

# Create two shapes
sphere = Sphere(0.25).translate(-0.3, 0, 0)
box = Box((0.2, 0.2, 0.2)).translate(0.3, 0, 0)

# Combine
combined = Union(sphere, box)

# Sample
bounds = ((-1, 1), (-1, 1), (-1, 1))
res = (128, 128, 128)
phi = sample_levelset(combined, bounds, res)

# Save
np.save("output/combined.npy", phi)
```

### Example 3: Rocket Shape (Multiple Operations)

```python
import amrex.space3d as amr
from sdf3d import SDFLibrary

amr.initialize([])
try:
    # Setup grid
    real_box = amr.RealBox([-1, -1, -1], [1, 1, 1])
    domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(127, 127, 127))
    geom = amr.Geometry(domain, real_box, 0, [0, 0, 0])
    ba = amr.BoxArray(domain)
    ba.max_size(32)
    dm = amr.DistributionMapping(ba)

    lib = SDFLibrary(geom, ba, dm)

    # Body (cylinder-like using elongated sphere)
    body = lib.sphere(center=(0, 0, -0.3), radius=0.15)
    
    # Nose cone (smaller sphere)
    nose = lib.sphere(center=(0, 0, 0.2), radius=0.1)
    
    # Fins (boxes)
    fin1 = lib.box(center=(-0.1, 0.2, -0.2), size=(0.05, 0.15, 0.1))
    fin2 = lib.box(center=(0.1, 0.2, -0.2), size=(0.05, 0.15, 0.1))
    fin3 = lib.box(center=(0, -0.2, -0.2), size=(0.15, 0.05, 0.1))

    # Combine: body + nose + fins
    rocket = lib.union(body, nose)
    rocket = lib.union(rocket, fin1)
    rocket = lib.union(rocket, fin2)
    rocket = lib.union(rocket, fin3)

    # Save
    varnames = amr.Vector_string(["sdf"])
    amr.write_single_level_plotfile("output/rocket", rocket, varnames, geom, 0.0, 0)
finally:
    amr.finalize()
```

---

## Function Reference Summary

### SDFLibrary Methods (AMReX Mode)

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `sphere` | `center, radius` | `MultiFab` | Sphere primitive |
| `box` | `center, size` | `MultiFab` | Box primitive |
| `round_box` | `center, size, radius` | `MultiFab` | Rounded box |
| `union` | `mf1, mf2` | `MultiFab` | Boolean union |
| `intersect` | `mf1, mf2` | `MultiFab` | Boolean intersection |
| `subtract` | `base, cutter` | `MultiFab` | Boolean subtraction |
| `negate` | `mf` | `MultiFab` | Negate SDF |

### Geometry Classes (NumPy Mode)

| Class | Parameters | Methods |
|-------|-----------|---------|
| `Sphere` | `radius` | `translate`, `rotate_*`, `scale`, `elongate` |
| `Box` | `size` | `translate`, `rotate_*`, `scale`, `elongate` |
| `RoundBox` | `size, radius` | `translate`, `rotate_*`, `scale`, `elongate` |
| `Cylinder` | `radius, height` | `translate`, `rotate_*`, `scale`, `elongate` |
| `Torus` | `major_r, minor_r` | `translate`, `rotate_*`, `scale`, `elongate` |
| `Union` | `geom1, geom2` | `translate`, `rotate_*`, `scale`, `elongate` |
| `Intersection` | `geom1, geom2` | `translate`, `rotate_*`, `scale`, `elongate` |
| `Subtraction` | `base, cutter` | `translate`, `rotate_*`, `scale`, `elongate` |

### Utility Functions

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `sample_levelset` | `geometry, bounds, resolution` | `ndarray` | Sample geometry to NumPy array |

---

## Tips and Best Practices

1. **Always initialize/finalize AMReX:**
   ```python
   amr.initialize([])
   try:
       # Your code
   finally:
       amr.finalize()
   ```

2. **Use appropriate grid resolution:**
   - For testing: 64×64×64
   - For production: 128×128×128 or higher
   - Balance between accuracy and memory

3. **Box size for parallelism:**
   - `ba.max_size(32)` works well for most cases
   - Smaller boxes = better parallelism but more overhead

4. **Chaining operations:**
   - Geometry API: Can chain transforms
   - MultiFab mode: Store intermediate results, then combine

5. **Memory management:**
   - MultiFab operations create new objects
   - Delete intermediate results if memory is limited

---

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'sdf3d'`:
- Make sure you're running from the project root
- Or add project root to `PYTHONPATH`

### AMReX Errors

If you get segmentation faults:
- Make sure `amr.initialize([])` is called before any AMReX operations
- Make sure `amr.finalize()` is called in a `finally` block

### Wrong Values

If SDF values seem incorrect:
- Check that your domain bounds include the geometry
- Verify grid resolution is sufficient
- Check that cell-centered coordinates are used correctly

---

For more examples, see the `examples/` folder in the project root.
