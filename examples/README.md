# SDF Library — Examples

Standalone scripts demonstrating the `sdf2d` / `sdf3d` API.
**No AMReX required** — all geometry is evaluated with `sample_levelset_3d`
and rendered to PNG via matplotlib + scikit-image.

```bash
# Run from the repo root or from this folder:
python examples/union_example.py
python examples/intersection_example.py
python examples/subtraction_example.py
python examples/elongation_example.py
python examples/complex_example.py
python examples/nato_stanag_4496_test.py
```

Output PNGs are written to **this folder** (`examples/`).

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
Note the argument order: first arg is the base, second is the cutter.

### `elongation_example.py`
Sphere elongated along X into a capsule with `.elongate(h, 0, 0)`.
Spot-checks that the SDF value at the elongation boundary equals `-radius`.

### `complex_example.py`
Chains all four operations in sequence, saving a PNG per step:

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
in front of a 50 mm target block at a 5° yaw angle, then unions them.

| File | Description |
|------|-------------|
| `nato_fragment.png`     | Fragment geometry alone |
| `nato_impact_scene.png` | Fragment + target, impact position |

In production, replace `_MockLib` with an `SDFLibrary3D` instance to obtain
an AMReX `MultiFab` for solver input.

---

## SDF sign convention

| Value | Meaning |
|-------|---------|
| φ < 0 | inside the solid |
| φ = 0 | on the surface |
| φ > 0 | outside the solid |
