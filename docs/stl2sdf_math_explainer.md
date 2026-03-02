# How `stl2sdf` Works

An STL file is a list of triangles.  Each triangle has three **vertices**
(3-D coordinates), no metadata.  The library reads it into a NumPy array of shape `(F, 3, 3)`:
`F` triangles × 3 vertices × 3 coordinates (x, y, z).

### Binary vs ASCII parsing

STL comes in two flavours.  The library peeks at the first 5 bytes:

- If they spell `"solid"` → ASCII format, parse line-by-line for `vertex` keywords.
- Otherwise → binary format, read with a fixed NumPy `dtype` (80-byte header, then
  50-byte records: 12-byte normal + 36-byte vertices + 2-byte attribute).

## The two sub-problems

Computing a signed distance breaks into two independent questions:

| Question | Answer |
|---|---|
| **How far** is point P from the mesh? | Unsigned distance — §1 |
| **Is P inside or outside** the mesh? | Sign determination — §2 |

## §1 Unsigned Distance: Ericson's Voronoi-Region Method

**Source:** *Real-Time Collision Detection* by Christer Ericson, §5.1.5.

### The goal

Given a query point **P** and one triangle **T = {A, B, C}**, find the point
**Q** on or inside the triangle that is closest to **P**.  Then
`dist(P, T) = ‖P − Q‖`.

### Why the closest point might not be on the interior

The closest point to a plane-projected position might fall *outside* the
triangle.  In that case we "snap" it back to the nearest edge or vertex.
The triangle's interior plus three edges plus three vertices forms seven
**Voronoi regions** — each region of space claims the closest feature of the
triangle.

```
         C
        / \
   AC  /   \  BC
      /  T  \
     /       \
    A----AB----B
```

- If P projects above vertex A: closest point = A.
- If P projects above vertex B: closest point = B.
- If P projects above vertex C: closest point = C.
- If P projects near edge AB: closest point = some point on segment AB.
- If P projects near edge AC: closest point = some point on segment AC.
- If P projects near edge BC: closest point = some point on segment BC.
- Otherwise: P projects inside T, closest point = the projection itself.

### The dot products d1 – d6

Let:
```
AB = B - A
AC = C - A
AP = P - A
```

We compute six dot products:
```
d1 = AP · AB      (how far P's offset from A projects onto AB)
d2 = AP · AC      (how far P's offset from A projects onto AC)
d3 = (P-B) · AB   (offset from B projected onto AB — negative means P is "past" B)
d4 = (P-B) · AC
d5 = (P-C) · AB
d6 = (P-C) · AC
```

These six numbers encode all the geometry we need, without ever computing a
square root.

### Region tests (pure comparisons)

```python
cond_A  = (d1 <= 0) and (d2 <= 0)          # P is "behind" both AB and AC from A
cond_B  = (d3 >= 0) and (d4 <= d3)         # P is past B in both directions
cond_C  = (d6 >= 0) and (d5 <= d6)
cond_AB = (vc <= 0) and (d1 >= 0) and (d3 <= 0)   # P is in the AB edge strip
cond_AC = (vb <= 0) and (d2 >= 0) and (d6 <= 0)
cond_BC = (va <= 0) and ((d4-d3) >= 0) and ((d5-d6) >= 0)
# Interior: anything that didn't match the above
```

`va`, `vb`, `vc` are **cross-term determinants** — essentially signed areas
that tell us which side of each edge the point falls on:

```
vc = d1·d4 − d3·d2
vb = d5·d2 − d1·d6
va = d3·d6 − d5·d4
```

Geometrically, `va + vb + vc = area(ABC)^2` (unnormalised).  Each individual
term is the barycentric coordinate numerator for one vertex.

### Closest-point formulas per region

| Region | Closest point formula |
|--------|----------------------|
| A vertex | Q = A |
| B vertex | Q = B |
| C vertex | Q = C |
| AB edge | Q = A + t·AB,  t = clamp(d1 / (d1−d3), 0, 1) |
| AC edge | Q = A + t·AC,  t = clamp(d2 / (d2−d6), 0, 1) |
| BC edge | Q = B + t·(C−B), t = clamp((d4−d3) / denom, 0, 1) |
| Interior | Q = A + w_v·AC + w_w·AB  (barycentric weights) |

For the interior:
```
w_v = vb / (va + vb + vc)
w_w = vc / (va + vb + vc)
```
These are the barycentric coordinates of the projection.

### What the code actually does

```python
sq = np.select(
    [cond_A, cond_B, cond_C, cond_AB, cond_AC, cond_BC],
    [_sq(A), _sq(B), _sq(C), _sq(cp_AB), _sq(cp_AC), _sq(cp_BC)],
    default=_sq(cp_int),
)
```

`np.select` evaluates **all** branches (this is a NumPy constraint — it
cannot short-circuit conditionally).  Every `_sq(cp_*)` is computed for all N
points, then the conditions select the right one.  This is why all denominators
are guarded with `np.maximum(..., 1e-30)`: a division by zero in a branch that
won't be selected still happens numerically, and we must prevent NaN from
polluting the chosen branch's result.

### Loop over all triangles

```python
sq_min = np.full(N, np.inf)
for tri in tris:
    sq = _sq_dist_to_tri(P, tri)         # (N,) squared distances for this tri
    sq_min = np.minimum(sq_min, sq)      # keep the minimum over all triangles
```

After the loop: `unsigned_dist = sqrt(sq_min)`.  We work in squared distance
to avoid N expensive square roots during the loop.

> **Complexity:** O(F × N).  For 14 000 triangles and 64 000 grid points that
> is ~900 million operations.  A BVH (Bounding Volume Hierarchy) or KD-tree
> would reduce this to O(N log F), but is not implemented here.

---

## §2 Sign: Möller–Trumbore Ray Casting

### The idea: parity of intersections

The **Jordan Curve Theorem** (in 3-D: the Poincaré–Lefschetz theorem) says:
a ray from any point P that goes to infinity crosses the surface an **even**
number of times if P is outside, and an **odd** number of times if P is inside.

```
   outside →  0 or 2 crossings
   inside  →  1 or 3 crossings
```

So: cast a ray from P in a fixed direction, count how many triangles it hits.
If the count is odd, P is inside.

### Why the irrational ray direction?

```python
_RAY_DIR = [sqrt(2)-1, sqrt(3)-1, 1/sqrt(3)]   # then normalised
```

An axis-aligned ray (e.g. `[1, 0, 0]`) has a higher chance of grazing a
triangle edge or vertex exactly, producing ambiguous intersection counts.
Irrational components make it essentially impossible to hit an edge or vertex
exactly, so every intersection is a clean face crossing.

### Möller–Trumbore algorithm

For one ray origin **P** and one triangle **{v0, v1, v2}**:

#### Set up

```
e1 = v1 - v0
e2 = v2 - v0
h  = ray_dir × e2          (cross product — a (3,) vector)
det = e1 · h
```

`det` is the scalar triple product `(ray_dir, e1, e2)`.  If `|det| < ε`, the
ray is **parallel** to the triangle plane → no intersection.

#### Solve the parametric system

We want to find parameters `(t, u, v)` such that:

```
P + t·ray_dir = v0 + u·e1 + v·e2
```

This is a 3×3 linear system (3 equations, 3 unknowns).  Möller–Trumbore solves
it with Cramer's Rule:

```
inv_det = 1 / det
s = P - v0
u = inv_det · (s · h)         ← barycentric coord along e1
q = s × e1
v = inv_det · (q · ray_dir)   ← barycentric coord along e2
t = inv_det · (q · e2)        ← ray parameter (distance along the ray)
```

The point of intersection lies **inside** the triangle if and only if:
```
u >= 0
v >= 0
u + v <= 1
```
And the intersection is **in front of** the ray origin (not behind) if `t > ε`.

#### All conditions together

```python
hit = (u >= 0) & (v >= 0) & (u + v <= 1.0) & (t > 1e-10)
```

### NumPy vectorisation over all query points

The ray direction is the same for all N points, but the origins differ.
The per-triangle edge vectors `e1`, `e2`, `h`, and `det` are scalars (or fixed
vectors) computed once.  Then `s`, `u`, `q`, `v`, `t` all have shape `(N,)`,
giving a vectorised test over all query points in one pass.

```python
for tri in tris:
    hits += _mt_ray_hits(P, ray_dir, tri)   # accumulate intersection counts
```

### Applying the sign

```python
sign = np.where(hits % 2 == 1, -1.0, 1.0)
return sign * unsigned_dist
```

Odd hit count → inside → negative distance.

---

## §3 Grid Sampling

`sample_sdf_from_stl` builds a uniform Cartesian grid of **cell centres**:

```python
xs = linspace(x0, x1, nx, endpoint=False) + (x1-x0)/(2*nx)
```

The `endpoint=False` + half-cell offset puts samples at cell centres (not cell
corners).  The same convention is used by `sample_levelset_3d` in `sdf3d`.

`np.meshgrid(..., indexing="ij")` creates 3-D arrays; `reshape(-1, 3)` flattens
them into `(N, 3)` to pass to `mesh_to_sdf`.  The output is reshaped back to
`(nz, ny, nx)` — z-major ordering.

## Putting it all together

```
STL file  ──load_stl──►  triangles (F, 3, 3)
                             │
query grid  ──meshgrid──►  points (N, 3)
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
     _sq_dist_to_tri (Ericson)    _mt_ray_hits (Möller-Trumbore)
          for each triangle            for each triangle
               │                           │
         sq_min (N,)                  hits (N,)
               │                           │
         sqrt(sq_min)             np.where(hits%2==1, -1, +1)
               │                           │
               └──────── multiply ──────────┘
                              │
                         phi (N,)  →  reshape  →  (nz, ny, nx)
```

## Key limitations

| Limitation | Why it exists |
|---|---|
| **Watertight mesh required** | Ray parity only works if every ray crosses the surface an even or odd number of times with no ambiguity. A hole in the mesh lets rays escape without a matching exit crossing. |
| **O(F × N) complexity** | No spatial acceleration structure (BVH, KD-tree). Every query point checks every triangle. |
| **Single ray direction** | One ray can occasionally give the wrong parity if grazing a sharp feature. Using multiple rays and voting would be more robust but slower. |

## Glossary

| Term | Meaning |
|---|---|
| **SDF** | Signed Distance Field — a scalar field encoding distance to a surface, negative inside |
| **STL** | STereoLithography — a triangle mesh file format |
| **Voronoi region** | The region of space closest to a given feature (vertex, edge, face) of a triangle |
| **Barycentric coordinates** | A way to express a point inside a triangle as a weighted sum of its vertices: `P = α·A + β·B + γ·C` with `α+β+γ=1` |
| **Watertight mesh** | A closed 2-manifold — no holes, every edge shared by exactly two triangles |
| **Möller–Trumbore** | A fast ray-triangle intersection algorithm using Cramer's Rule on a parametric linear system |
| **Jordan Curve / Parity** | The topological result that says "odd crossing count ↔ inside closed surface" |
| **Irrational ray direction** | A direction with irrational components, used to avoid numerical degeneracy when the ray exactly grazes an edge |
