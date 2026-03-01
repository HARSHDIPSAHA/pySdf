# pySdf Performance Notes

Benchmarked on Windows 11, AMD/Intel CPU, single thread, AMReX 26.01.

## numpy path — grid size scaling

Single `Sphere3D`, varying resolution:

| Resolution | Time (ms) | Throughput |
|---|---|---|
| 16³ | 0.1 | ~47 MVox/s |
| 32³ | 1.1 | ~25 MVox/s |
| 64³ | 10–12 | ~21–25 MVox/s |
| 96³ | 38–43 | ~21–24 MVox/s |
| 128³ | 90–100 | ~21–23 MVox/s |

Throughput plateaus at ~21 MVox/s for 64³ and above. The spike at 16³ is startup noise.

## numpy path — geometry complexity at 64³

| Geometry | Time (ms) | Throughput |
|---|---|---|
| Sphere3D | 10 | 26 MVox/s |
| Cylinder3D | 12 | 22 MVox/s |
| Torus3D | 15 | 18 MVox/s |
| Box3D | 22 | 12 MVox/s |
| RoundBox3D | 24 | 11 MVox/s |
| elongate(sphere) | 21 | 12 MVox/s |
| round(box) | 22 | 12 MVox/s |
| translate + rotate_y | 18 | 15 MVox/s |
| Union3D / Intersection3D / Subtraction3D | 20–21 | 12–13 MVox/s |
| Union3D(Box3D, Torus3D) | 33 | 7.9 MVox/s |

`Box3D` is ~2× slower than `Sphere3D` because `max(abs(p)-b, 0)` involves more
operations than a single subtract-and-length. Boolean ops add minimal overhead
on their own — their cost is dominated by evaluating both child SDFs.

## numpy vs AMReX at equal resolution

Single `Sphere3D`:

| Resolution | numpy (ms) | AMReX (ms) | Overhead |
|---|---|---|---|
| 32³ | 1.1 | 1.3 | +18% |
| 64³ | 10.3 | 12.1 | +17% |
| 96³ | 38.3 | 41.6 | +9% |
| 128³ | 89.1 | 104.5 | +17% |

`Union3D(Sphere3D, Box3D)`:

| Resolution | numpy (ms) | AMReX (ms) | Overhead |
|---|---|---|---|
| 64³ | 30.6 | 31.1 | +2% |
| 128³ | 288.2 | 295.7 | +3% |

## Why AMReX is slower here

`SDFLibrary3D.from_geometry` calls the same Python/numpy SDF functions as
`sample_levelset_3d` — there are no native C++ SDF kernels. The extra cost
comes from the MFIter loop and `to_numpy()` round-trips on each box.

For a single-level full-domain grid, AMReX adds overhead with no benefit.
It pays off in AMR workflows where only specific refinement-level patch boxes
need to be filled rather than the whole domain.
