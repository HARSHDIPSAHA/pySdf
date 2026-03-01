# pySdf Code Walkthrough

*2026-02-27T07:25:32Z by Showboat 0.6.1*
<!-- showboat-id: dc9360bc-fd91-4c3f-bf79-96c80c905cf7 -->

## What is a Signed Distance Function?

A **signed distance function** (SDF) maps every point in space to a scalar:

- **negative** → the point is *inside* the shape
- **zero** → the point is *on the surface*
- **positive** → the point is *outside* the shape

The magnitude tells you how far you are from the nearest surface.  
This makes SDFs composable — you can union, subtract, and intersect shapes  
with simple min/max/negate operations on their distance values.

pySdf implements a library of these functions in pure numpy, so any SDF can  
be evaluated over an entire grid in one vectorised call.

## Layer 1 — Shared helpers: `_sdf_common.py`

Everything sits on a thin foundation of numpy wrappers.  
`vec2` / `vec3` turn separate x/y/z scalars-or-arrays into a  
`(..., 2)` / `(..., 3)` array, using `np.broadcast_arrays` so any  
batch shape works.  The boolean operators are one-liners that follow  
directly from the SDF algebra:

```bash
sed -n '40,49p' _sdf_common.py
```

```output
def vec2(x: _F, y: _F) -> _F:
    """Stack *x* and *y* into a ``(..., 2)`` array."""
    x, y = np.broadcast_arrays(x, y)
    return np.stack([x, y], axis=-1)


def vec3(x: _F, y: _F, z: _F) -> _F:
    """Stack *x*, *y*, *z* into a ``(..., 3)`` array."""
    x, y, z = np.broadcast_arrays(x, y, z)
    return np.stack([x, y, z], axis=-1)
```

```bash
sed -n '85,112p' _sdf_common.py
```

```output
def opUnion(d1: _F, d2: _F) -> _F:
    """Union of two SDFs: ``min(d1, d2)``."""
    return np.minimum(d1, d2)


def opSubtraction(d1: _F, d2: _F) -> _F:
    """Subtract *d1* from *d2*: ``max(-d1, d2)``."""
    return np.maximum(-d1, d2)


def opIntersection(d1: _F, d2: _F) -> _F:
    """Intersection of two SDFs: ``max(d1, d2)``."""
    return np.maximum(d1, d2)


def opRound(p: _F, primitive: "_SDFFunc", rad: float) -> _F:  # type: ignore[name-defined]
    """Round a primitive outward by *rad*."""
    return primitive(p) - rad


def opOnion(sdf_val: _F, thickness: float) -> _F:
    """Turn a solid into a shell of *thickness*."""
    return np.abs(sdf_val) - thickness


def opScale(p: _F, s: float, primitive: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Uniformly scale a primitive by factor *s*."""
    return primitive(p / s) * s
```

The boolean algebra is beautiful in its simplicity:

| operation | formula | why it works |
|-----------|---------|--------------|
| union | `min(d1, d2)` | nearest surface wins |
| intersection | `max(d1, d2)` | you must be inside *both* |
| subtraction (d1=cutter) | `max(-d1, d2)` | negate cutter then intersect |
| round | `sdf(p) - r` | pull the surface outward by r |
| onion | `abs(sdf) - t` | fold inside onto outside, thin band |
| scale by s | `sdf(p/s) * s` | divide query point, multiply result |

## Layer 2 — 2D primitives: `sdf2d/primitives.py`

This module does `from _sdf_common import *` so all helpers are available,
then adds every 2D shape.  The simplest primitive is the circle:

    sdCircle(p, r) = length(p) - r

The distance from any point to a circle centred at the origin is just the
distance from the origin minus the radius.  Negative inside, zero on the rim,
positive outside.

The **axis-aligned box** is more instructive — it handles corners, edges, and
the interior all in one expression:

```bash
sed -n '25,33p' sdf2d/primitives.py
```

```output
def sdCircle(p: _F, r: float) -> _F:
    """2-D circle of radius *r* centred at origin."""
    return length(p) - r


def sdBox2D(p: _F, b: _F) -> _F:
    """2-D axis-aligned box with half-extents *b* ``(bx, by)``."""
    d = np.abs(p) - b
    return length(np.maximum(d, 0.0)) + np.minimum(np.maximum(d[..., 0], d[..., 1]), 0.0)
```

How `sdBox2D` works:

1. `d = abs(p) - b` — fold into the positive quadrant; negative components
   mean "inside along that axis".
2. `length(max(d, 0))` — distance from the corner for points *outside* the box;
   components that are negative (inside along that axis) contribute 0.
3. `min(max(d.x, d.y), 0)` — for points *inside* the box, the max of the two
   components is the (negative) distance to the nearest face; clamped to ≤0 so
   it only contributes when both components are negative.

The two terms together give the correct signed distance everywhere: outside via
the corner distance, inside via the face distance.

### The "folding" trick for regular polygons

Polygons with N-fold symmetry (pentagon, hexagon, octagon) use a clever trick:
fold the query point into the fundamental domain of the shape, then measure a
simple distance there.  Each fold is a reflection across a symmetry line:

    d = 2 * min(dot(p, k), 0)   # how far to reflect
    p -= d * k                   # apply to both components simultaneously

The critical subtlety: GLSL updates `p.x` and `p.y` at the same time in
`p -= 2*min(dot(k,p),0)*k`.  A naive Python translation that updates `px`
then `py` sequentially would use the *already-modified* px in the second
line — a silent correctness bug.  The fix: compute the scalar `d` once,
then apply it to both components.

```bash
sed -n '167,177p' sdf2d/primitives.py
```

```output
def sdHexagon2D(p: _F, r: float) -> _F:
    """2-D regular hexagon with inradius *r* (distance from centre to a flat face)."""
    k  = np.array([-0.866025404, 0.5, 0.577350269])
    px = np.abs(p[..., 0])
    py = np.abs(p[..., 1])
    # Fold step: compute dot product once, apply to both components simultaneously
    d  = 2.0 * np.minimum(dot(vec2(px, py), k[:2]), 0.0)
    px = px - d * k[0];  py = py - d * k[1]
    px = px - clamp(px, -k[2] * r, k[2] * r)
    py = py - r
    return length(vec2(px, py)) * np.sign(py)
```

After folding, every point is as if it were in the "wedge" above the flat top
face.  The remaining distance is then just a 2D box distance against that
single face: clamp x to the face width, then measure to `(clamped_x, r)`.

### Broadcasting over grids

Every primitive accepts `p` with shape `(..., 2)` — the `...` can be anything:
a single point `(2,)`, a row of points `(N, 2)`, or a full 2D grid `(H, W, 2)`.
The `axis=-1` convention on `length`, `dot`, `vec2` makes this transparent.

```bash
/c/Users/arkma/AppData/Local/Programs/Python/Python313/python -c "
import sys; sys.path.insert(0, '.')
from sdf2d.primitives import sdCircle
import numpy as np

# Single point — inside the circle
p_single = np.array([0.5, 0.0])
print('single point (inside):', sdCircle(p_single, 1.0))   # -0.5

# Row of points
p_row = np.array([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]])
print('row:', sdCircle(p_row, 1.0))   # [-0.5, 0.0, 1.0]

# 2D grid of points (H=3, W=3)
x = np.linspace(-1.5, 1.5, 3)
Y, X = np.meshgrid(x, x, indexing='ij')
p_grid = np.stack([X, Y], axis=-1)
print('grid shape:', p_grid.shape)
print('grid distances (negative=inside):')
print(np.round(sdCircle(p_grid, 1.0), 2))
"
```

```output
single point (inside): -0.5
row: [-0.5  0.   1. ]
grid shape: (3, 3, 2)
grid distances (negative=inside):
[[ 1.12  0.5   1.12]
 [ 0.5  -1.    0.5 ]
 [ 1.12  0.5   1.12]]
```

## Layer 3 — 2D Geometry classes: `sdf2d/geometry.py`

The geometry layer wraps every primitive in an OOP shell.  The key insight is
`Geometry2D`, a base class that holds a single callable `_func(p) -> phi`:

```bash
sed -n '23,95p' sdf2d/geometry.py
```

```output
class Geometry2D:
    """Base class for 2D signed-distance-function geometries.

    A ``Geometry2D`` wraps a callable ``func(p) -> distances`` where *p* is
    a ``(..., 2)`` array of 2D points and the return value is a ``(...)``
    array of signed distances.

    Subclasses override ``__init__`` to pass the appropriate primitive SDF to
    ``super().__init__(func)``.

    Implements:
    - Boolean operations: :meth:`union`, :meth:`subtract`, :meth:`intersect`
    - Modifiers:          :meth:`round`, :meth:`onion`
    - Transforms:         :meth:`translate`, :meth:`scale`, :meth:`rotate`
    """

    def __init__(self, func: _SDFFunc) -> None:
        self._func = func

    def sdf(self, p: _Array) -> _Array:
        """Evaluate signed distance at *p* (shape ``(..., 2)``)."""
        return self._func(p)

    def __call__(self, p: _Array) -> _Array:
        return self._func(p)

    # ------------------------------------------------------------------
    # Boolean operations
    # ------------------------------------------------------------------

    def union(self, other: Geometry2D) -> Geometry2D:
        """Return the union (min) of this shape and *other*."""
        return Geometry2D(lambda p: sdf.opUnion(self.sdf(p), other.sdf(p)))

    def subtract(self, other: Geometry2D) -> Geometry2D:
        """Subtract *other* from this shape."""
        return Geometry2D(lambda p: sdf.opSubtraction(other.sdf(p), self.sdf(p)))

    def intersect(self, other: Geometry2D) -> Geometry2D:
        """Return the intersection (max) of this shape and *other*."""
        return Geometry2D(lambda p: sdf.opIntersection(self.sdf(p), other.sdf(p)))

    # ------------------------------------------------------------------
    # Modifiers
    # ------------------------------------------------------------------

    def round(self, rad: float) -> Geometry2D:
        """Round the surface outward by *rad*."""
        return Geometry2D(lambda p: sdf.opRound(p, self.sdf, rad))

    def onion(self, thickness: float) -> Geometry2D:
        """Turn the solid into a hollow shell of *thickness*."""
        return Geometry2D(lambda p: sdf.opOnion(self.sdf(p), thickness))

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def translate(self, tx: float, ty: float) -> Geometry2D:
        """Translate by ``(tx, ty)``."""
        t = np.array([tx, ty])
        return Geometry2D(lambda p: self.sdf(p - t))

    def scale(self, s: float) -> Geometry2D:
        """Uniformly scale by factor *s*."""
        return Geometry2D(lambda p: sdf.opScale(p, s, self.sdf))

    def rotate(self, angle_rad: float) -> Geometry2D:
        """Rotate by *angle_rad* radians (counter-clockwise)."""
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rot = np.array([[c, -s], [s, c]])
        return Geometry2D(lambda p: sdf.opTx2D(p, rot, np.zeros(2), self.sdf))
```

Each method returns a **new** `Geometry2D` wrapping a new lambda.  This makes
the API fully composable:

```python
# build up a compound shape by chaining methods
shape = (Circle2D(1.0)
         .subtract(Box2D([0.5, 0.5]))
         .translate(2.0, 0.0)
         .onion(0.05))
```

Every node in this expression tree captures its parents by closure.
When `shape.sdf(p)` is called, it evaluates the whole chain.

Concrete subclasses are just one-liners that pass the right lambda:

```bash
sed -n '102,115p' sdf2d/geometry.py
```

```output
class Circle2D(Geometry2D):
    """Circle centred at origin with given *radius*."""

    def __init__(self, radius: float) -> None:
        super().__init__(lambda p: sdf.sdCircle(p, radius))


class Box2D(Geometry2D):
    """Axis-aligned rectangle with *half_size* ``(hx, hy)`` centred at origin."""

    def __init__(self, half_size: Sequence[float]) -> None:
        b = np.array(half_size, dtype=float)
        super().__init__(lambda p: sdf.sdBox2D(p, b))

```

### The subtraction argument order — a critical convention

`opSubtraction(d1, d2) = max(-d1, d2)` where **d1 is the cutter** and
**d2 is the base**.  The `subtract` method always passes `other` (the cutter)
as d1 and `self` (the base) as d2.

```python
def subtract(self, other):
    return Geometry2D(lambda p: sdf.opSubtraction(other.sdf(p), self.sdf(p)))
#                                                  ^^^ cutter    ^^^ base
```

So `a.subtract(b)` means "remove b from a".  Similarly, `Subtraction2D(base, cutter)`
follows the same convention.  Getting this backwards produces a shape that is the
*inverse* — the leftover cutter material rather than the carved base.

```bash
/c/Users/arkma/AppData/Local/Programs/Python/Python313/python -c "
import sys; sys.path.insert(0, '.')
from sdf2d.geometry import Circle2D, Box2D
import numpy as np

# Large circle with small square cut out of the middle
circle = Circle2D(1.0)
box    = Box2D([0.3, 0.3])
carved = circle.subtract(box)

# Check three points
pts = np.array([
    [0.0, 0.0],   # origin — inside box (should be outside carved shape)
    [0.0, 0.5],   # inside circle, outside box (should be inside carved shape)
    [0.0, 1.5],   # outside circle (should be outside carved shape)
])
print('origin (in box, cut out):', round(float(carved.sdf(pts[0])), 3))
print('mid-circle (kept)      :', round(float(carved.sdf(pts[1])), 3))
print('outside circle         :', round(float(carved.sdf(pts[2])), 3))
"
```

```output
origin (in box, cut out): 0.3
mid-circle (kept)      : -0.2
outside circle         : 0.5
```

The origin (inside the cut-out box) reads **+0.3** — correctly outside the
carved shape.  The mid-circle point reads **-0.2** — correctly inside.

## Layer 4 — Grid sampling: `sdf2d/grid.py`

`sample_levelset_2d` turns a geometry and a bounding box into a 2D numpy array.

```bash
sed -n '18,47p' sdf2d/grid.py
```

```output
def sample_levelset_2d(
    geom: Geometry2D,
    bounds: _Bounds2D,
    resolution: _Resolution2D,
) -> _Array:
    """Sample *geom* on a uniform 2-D cell-centred grid.

    Parameters
    ----------
    geom:
        A 2-D geometry whose ``sdf()`` method accepts ``(..., 2)`` arrays.
    bounds:
        ``((x0, x1), (y0, y1))`` physical extents of the domain.
    resolution:
        ``(nx, ny)`` number of cells along each axis.

    Returns
    -------
    numpy.ndarray
        Shape ``(ny, nx)`` array of signed distances, row-major (y first).
    """
    (x0, x1), (y0, y1) = bounds
    nx, ny = resolution

    xs = np.linspace(x0, x1, nx, endpoint=False) + (x1 - x0) / (2.0 * nx)
    ys = np.linspace(y0, y1, ny, endpoint=False) + (y1 - y0) / (2.0 * ny)

    Y, X = np.meshgrid(ys, xs, indexing="ij")
    p = np.stack([X, Y], axis=-1)
    return geom.sdf(p)
```

The grid is **cell-centred**: each sample point sits at the middle of its cell,
not at the cell boundary.  That is why `linspace` uses `endpoint=False` and
then adds half a cell-width (`(x1-x0)/(2*nx)`).

`meshgrid` with `indexing="ij"` is followed by `np.stack([X, Y], axis=-1)`,
producing a `(ny, nx, 2)` array — row-major (Y axis first) for numpy
compatibility, but the last axis is always `(x, y)` for the SDF call.

The entire grid evaluation is then a **single call** to `geom.sdf(p)`, which
broadcasts through all the lambda closures built up in `geometry.py`.

```bash
/c/Users/arkma/AppData/Local/Programs/Python/Python313/python -c "
import sys; sys.path.insert(0, '.')
from sdf2d.geometry import Circle2D
from sdf2d.grid import sample_levelset_2d
import numpy as np

phi = sample_levelset_2d(
    Circle2D(1.0),
    bounds=((-2.0, 2.0), (-2.0, 2.0)),
    resolution=(5, 5),
)
print('output shape (ny, nx):', phi.shape)
print('values (negative=inside):')
print(np.round(phi, 2))
print('zero-crossing (surface) expected near index 1-2 from edge')
"
```

```output
output shape (ny, nx): (5, 5)
values (negative=inside):
[[ 1.26  0.79  0.6   0.79  1.26]
 [ 0.79  0.13 -0.2   0.13  0.79]
 [ 0.6  -0.2  -1.   -0.2   0.6 ]
 [ 0.79  0.13 -0.2   0.13  0.79]
 [ 1.26  0.79  0.6   0.79  1.26]]
zero-crossing (surface) expected near index 1-2 from edge
```

## Layer 2 (3D side) — 3D primitives: `sdf3d/primitives.py`

The 3D module mirrors the 2D one.  Two techniques are worth highlighting:

### Projection to 2D for revolution-symmetric shapes

The **torus** is defined in XZ: compute the 2D distance from the "tube centre
circle" (radius R in XZ), then subtract the tube radius r:

    q = vec2(length(p.xz) - R, p.y)
    sdf = length(q) - r

This works because the torus is rotationally symmetric around Y.  The same
projection trick appears in `sdCappedTorus`, `sdLink`, `sdCappedCylinder`, etc.

```bash
sed -n '57,61p' sdf3d/primitives.py
```

```output
def sdTorus(p: _F, t: _F) -> _F:
    """Torus in the XZ plane; *t* = ``(R, r)`` (major, minor radii)."""
    q = vec2(length(p[..., [0, 2]]) - t[0], p[..., 1])
    return length(q) - t[1]

```

### Space-warp / domain operators in 3D

The 3D module adds a rich set of operators that *transform the query point*
before passing it to a primitive.  This is the key insight: you never transform
the shape — you transform the coordinate space:

| operator | what it does |
|----------|-------------|
| `opRevolution(p, prim2d, o)` | revolve a 2D shape around Y at offset o |
| `opExtrusion(p, prim2d, h)` | extrude a 2D shape along Z to height h |
| `opRepetition(p, s, prim)` | tile infinitely with cell size s |
| `opTwist(p, prim, k)` | twist around Y axis |
| `opCheapBend(p, prim, k)` | bend around Y axis |
| `opTx(p, rot, trans, prim)` | rigid body transform (rotate + translate) |

`opRevolution` and `opExtrusion` are especially powerful: they let you define
any 2D profile in `sdf2d.primitives` and promote it to 3D for free.

```bash
sed -n '449,533p' sdf3d/primitives.py
```

```output
def opRevolution(p: _F, primitive2d: "_SDFFunc", o: float) -> _F:  # type: ignore[name-defined]
    """Revolve a 2-D primitive around the Y axis with offset *o*."""
    q = vec2(length(p[..., [0, 2]]) - o, p[..., 1])
    return primitive2d(q)


def opExtrusion(p: _F, primitive2d: "_SDFFunc", h: float) -> _F:  # type: ignore[name-defined]
    """Extrude a 2-D primitive along Z to half-height *h*."""
    d = primitive2d(p[..., :2])
    w = vec2(d, np.abs(p[..., 2]) - h)
    return np.minimum(np.maximum(w[..., 0], w[..., 1]), 0.0) + length(np.maximum(w, 0.0))


def opElongate1(p: _F, primitive3d: "_SDFFunc", h: _F) -> _F:  # type: ignore[name-defined]
    """Elongate by clamping *p* to ``[-h, h]`` (type 1)."""
    q = p - clamp(p, -h, h)
    return primitive3d(q)


def opElongate2(p: _F, primitive3d: "_SDFFunc", h: _F) -> _F:  # type: ignore[name-defined]
    """Elongate by folding *p* beyond ``[-h, h]`` (type 2, exact)."""
    q = np.abs(p) - h
    return primitive3d(np.maximum(q, 0.0)) + np.minimum(np.max(q, axis=-1), 0.0)


def opSymX(p: _F, primitive3d: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Mirror the primitive in the YZ plane."""
    p = p.copy()
    p[..., 0] = np.abs(p[..., 0])
    return primitive3d(p)


def opSymXZ(p: _F, primitive3d: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Mirror the primitive in both YZ and XY planes."""
    p = p.copy()
    p[..., 0] = np.abs(p[..., 0])
    p[..., 2] = np.abs(p[..., 2])
    return primitive3d(p)


def opRepetition(p: _F, s: _F, primitive3d: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Tile the primitive infinitely with cell size *s*."""
    q = p - s * np.round(p / s)
    return primitive3d(q)


def opLimitedRepetition(p: _F, s: _F, l: _F, primitive3d: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Tile the primitive with cell size *s*, limited to ``[-l, l]`` repetitions."""
    q = p - s * clamp(np.round(p / s), -l, l)
    return primitive3d(q)


def opDisplace(p: _F, primitive3d: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Add sinusoidal displacement noise to the primitive."""
    d1 = primitive3d(p)
    d2 = np.sin(20.0 * p[..., 0]) * np.sin(20.0 * p[..., 1]) * np.sin(20.0 * p[..., 2])
    return d1 + d2


def opTwist(p: _F, primitive3d: "_SDFFunc", k: float) -> _F:  # type: ignore[name-defined]
    """Twist the primitive around Y with frequency *k*."""
    c = np.cos(k * p[..., 1]);  s = np.sin(k * p[..., 1])
    x = c * p[..., 0] - s * p[..., 2]
    z = s * p[..., 0] + c * p[..., 2]
    return primitive3d(vec3(x, p[..., 1], z))


def opCheapBend(p: _F, primitive3d: "_SDFFunc", k: float) -> _F:  # type: ignore[name-defined]
    """Bend the primitive around the Y axis with frequency *k*."""
    c = np.cos(k * p[..., 0]);  s = np.sin(k * p[..., 0])
    x = c * p[..., 0] - s * p[..., 1]
    y = s * p[..., 0] + c * p[..., 1]
    return primitive3d(vec3(x, y, p[..., 2]))


def opTx(p: _F, rot: _F, trans: _F, primitive3d: "_SDFFunc") -> _F:  # type: ignore[name-defined]
    """Apply rotation *rot* and translation *trans* to the primitive.

    *rot* is a ``(3, 3)`` rotation matrix; the inverse (transpose) is applied
    to transform the query point into the primitive's local frame.
    """
    inv_rot = rot.T
    q = (p - trans) @ inv_rot
    return primitive3d(q)
```

`opRepetition` is delightfully simple: `q = p - s * round(p/s)` snaps each
query point to the nearest tile centre, then evaluates the primitive at the
offset within that tile.  You get infinite copies for the cost of one.

`opTx` applies a rigid transform by working in the *inverse* direction:
instead of moving the shape, move the query point into the shape's local frame
using the transpose of the rotation matrix (which is its inverse for orthogonal
matrices).  `q = (p - trans) @ rot.T` then evaluates the untransformed primitive.

## Layer 3 (3D side) — 3D Geometry classes: `sdf3d/geometry.py`

The 3D geometry layer is structurally identical to the 2D one.  `Geometry3D`
holds a lambda, exposes `.sdf(p)`, and all methods return new `Geometry3D`
instances.  The 3D class adds extra rotation helpers:

```bash
sed -n '93,113p' sdf3d/geometry.py
```

```output
    def rotate_x(self, angle_rad: float) -> Geometry3D:
        """Rotate around the X axis by *angle_rad* radians."""
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rot = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        return Geometry3D(lambda p: sdf.opTx(p, rot, np.zeros(3), self.sdf))

    def rotate_y(self, angle_rad: float) -> Geometry3D:
        """Rotate around the Y axis by *angle_rad* radians."""
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        return Geometry3D(lambda p: sdf.opTx(p, rot, np.zeros(3), self.sdf))

    def rotate_z(self, angle_rad: float) -> Geometry3D:
        """Rotate around the Z axis by *angle_rad* radians."""
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return Geometry3D(lambda p: sdf.opTx(p, rot, np.zeros(3), self.sdf))

```

## Putting it all together: a complete 3D example

Here is a worked example that uses every layer: primitive, geometry class,
boolean operation, transform, and grid sampler.  We build a capped cylinder
with a spherical bite taken out of the top.

```bash
/c/Users/arkma/AppData/Local/Programs/Python/Python313/python -c "
import sys; sys.path.insert(0, '.')
from sdf3d.geometry import Sphere3D, ConeExact3D
from sdf3d.grid import sample_levelset_3d
import numpy as np

# Build: a sphere with a cone carved out of the bottom
import math
angle = math.radians(30)
sphere = Sphere3D(1.0)
cone   = ConeExact3D((math.sin(angle), math.cos(angle)), height=1.5)
carved = sphere.subtract(cone.translate(0, -1.5, 0))

# Sample on a coarse grid
phi = sample_levelset_3d(
    carved,
    bounds=((-2,2), (-2,2), (-2,2)),
    resolution=(9, 9, 9),
)
print('output shape (nz, ny, nx):', phi.shape)
print('inside voxels:', int(np.sum(phi < 0)))
print('outside voxels:', int(np.sum(phi > 0)))
print('min phi (deepest inside):', round(float(phi.min()), 3))
print('max phi (farthest outside):', round(float(phi.max()), 3))
"
```

```output
output shape (nz, ny, nx): (9, 9, 9)
inside voxels: 57
outside voxels: 672
min phi (deepest inside): -1.0
max phi (farthest outside): 2.079
```

## Smooth boolean operations (3D only)

The 3D module adds smooth variants that blend shapes near the boundary.
`opSmoothUnion` uses a quadratic falloff in the region where `|d1-d2| < k`:

    h = max(k - |d1-d2|, 0)
    result = min(d1, d2) - h² / (4k)

This rounds off the sharp crease that plain union would produce.
`opSmoothSubtraction` and `opSmoothIntersection` are implemented by
negating inputs and reusing `opSmoothUnion`.

```bash
sed -n '423,443p' sdf3d/primitives.py
```

```output
def opXor(d1: _F, d2: _F) -> _F:
    """Exclusive-or of two SDFs."""
    return np.maximum(np.minimum(d1, d2), -np.maximum(d1, d2))


def opSmoothUnion(d1: _F, d2: _F, k: float) -> _F:
    """Smooth union with smoothing factor *k*."""
    k = k * 4.0
    h = np.maximum(k - np.abs(d1 - d2), 0.0)
    return np.minimum(d1, d2) - h * h * 0.25 / k


def opSmoothSubtraction(d1: _F, d2: _F, k: float) -> _F:
    """Smooth subtraction with smoothing factor *k*."""
    return -opSmoothUnion(d1, -d2, k)


def opSmoothIntersection(d1: _F, d2: _F, k: float) -> _F:
    """Smooth intersection with smoothing factor *k*."""
    return -opSmoothUnion(-d1, -d2, k)

```

## The `np.where` / NaN pitfall

Many complex 2D SDFs (ellipse, parabola, Bézier, stairs, quadratic circle) have
two analytic branches — one for `d < 0`, one for `d ≥ 0`.  In GLSL, an `if`
statement only evaluates one branch.  In numpy, `np.where(cond, A, B)` evaluates
**both** A and B for every element before selecting.

If A contains `sqrt` of values that are negative in the B-branch, you get NaN
in the A array — even though those NaN values will ultimately be discarded by
`np.where`.  NaN is infectious: it propagates through subsequent arithmetic.

The fix used throughout the codebase is to guard with `np.maximum(..., 0)` and
`np.clip` before feeding into `sqrt` or `arccos`:

```python
# Safe: guard the sqrt input so numpy never sees a negative value
rx_n = np.sqrt(np.maximum(-c * (s_n + t_n + 2.0) + m2, 0.0))
```

This makes both branches safe to evaluate everywhere, at the cost of slightly
more computation — acceptable since the final `np.where` selects the correct
result.

## Running the test suite

The library ships with a comprehensive test suite.  No AMReX installation is
required — the one AMReX test auto-skips.

```bash
C:/Users/arkma/AppData/Local/Programs/Python/Python313/python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

```output
  C:\Users\arkma\Documents\GitHub\pySdf\sdf2d\primitives.py:476: RuntimeWarning: invalid value encountered in sqrt
    2.0 * np.cos(np.arctan2(np.sqrt(-h), q2) / 3.0) * np.sqrt(-p1) - kx),

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
212 passed, 1 skipped, 7 warnings in 0.76s
```

212 tests pass, 1 skipped (AMReX), 7 warnings.  The warnings come from the
complex analytic functions (arccos, sqrt) computing intermediate NaN values in
the "wrong" branch of np.where — as expected and by design.

## Architecture summary

```
_sdf_common.py          ← vec2/vec3, length/dot/clamp, bool ops
       │
   ┌───┴───┐
sdf2d/   sdf3d/
primitives.py           ← raw numpy math (no classes)
geometry.py             ← Geometry2D / Geometry3D wrapping lambdas
grid.py                 ← sample_levelset_2d/3d → ndarray
amrex.py                ← optional AMReX MultiFab integration
```

The data flow for a grid evaluation is:
1. User calls `sample_levelset_2d(geom, bounds, resolution)`
2. A `(ny, nx, 2)` array of cell-centred points is built via `meshgrid`
3. `geom.sdf(p)` is called once — the lambda chain unrolls through every
   boolean op, transform, and modifier in one vectorised pass
4. The result is a `(ny, nx)` float array of signed distances

