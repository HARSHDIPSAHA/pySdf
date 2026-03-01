# pySdf — Examples

Standalone scripts demonstrating the `sdf2d`, `sdf3d`, and `stl2sdf` APIs.
**No AMReX required** — all geometry is evaluated in pure NumPy.

Output files (PNG, HTML, NPY) are written to **this folder** (`examples/`).

```bash
# Run from the repo root:
uv run python examples/union_example.py
uv run python examples/intersection_example.py
uv run python examples/subtraction_example.py
uv run python examples/elongation_example.py
uv run python examples/complex_example.py
uv run python examples/nato_stanag_4496_test.py
uv run python examples/stl_sdf_demo.py --res 20   # quick draft
uv run python examples/stl_sdf_demo.py --res 40   # full quality
```

---

## Examples

### `union_example.py`
Two overlapping spheres combined with `Union3D`.
Verifies: `Union(A,B)(p) == min(A(p), B(p))`

### `intersection_example.py`
Intersection of two overlapping spheres via `Intersection3D`.
Verifies: `Intersection(A,B)(p) == max(A(p), B(p))`

### `subtraction_example.py`
Sphere with a spherical cavity cut using `Subtraction3D(base, cutter)`.
Verifies: `Subtraction(base,cutter)(p) == max(-cutter(p), base(p))`

### `elongation_example.py`
Sphere elongated along X into a capsule with `.elongate(h, 0, 0)`.

### `complex_example.py`
Chains all four boolean operations in sequence, saving a PNG per step:

| File | Description |
|------|-------------|
| `complex_example_step1.png` | Base box |
| `complex_example_step2.png` | Elongated sphere (capsule) |
| `complex_example_step3.png` | Union: box ∪ capsule |
| `complex_example_step4.png` | Intersection: rounded top |
| `complex_example_final.png` | Subtraction: central cavity |

### `nato_stanag_4496_test.py`
NATO STANAG-4496 fragment impact scene.
Builds the fragment via `sdf3d.examples.NATOFragment`, positions it 20 mm
in front of a target block at a 5° yaw angle, then unions them.

| File | Description |
|------|-------------|
| `nato_fragment.png`     | Fragment geometry alone |
| `nato_impact_scene.png` | Fragment + target, impact position |

### `stl_sdf_demo.py`
Downloads the ISS ratchet wrench STL (the first object 3D-printed in space,
Dec 2014) from NASA's GitHub archive, computes its SDF on a uniform grid,
and saves an interactive Plotly figure.

| File | Description |
|------|-------------|
| `wrench.stl`      | Downloaded STL (711 KB, 14 564 triangles) |
| `wrench_sdf.npy`  | SDF field — shape `(nz, ny, nx)` float64 |
| `wrench_sdf.html` | Interactive Plotly figure: 2D mid-Z heatmap + 3D isosurface |

```bash
uv run python examples/stl_sdf_demo.py --res 20   # ~15 s
uv run python examples/stl_sdf_demo.py --res 40   # ~2-5 min, cleaner surface
```

**Note:** `stl2sdf` uses O(F × N) brute-force (no BVH). Resolution 20 is fast;
resolution 40+ is suitable for quality renders. Requires a **watertight** mesh
for correct sign determination — the wrench passes this check.

---

## SDF sign convention

| Value | Meaning |
|-------|---------|
| φ < 0 | inside the solid |
| φ = 0 | on the surface |
| φ > 0 | outside the solid |
