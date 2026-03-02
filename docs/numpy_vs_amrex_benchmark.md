# pySdf — numpy path vs AMReX path

## How each path works

**numpy path (`sample_levelset_3d`)** allocates one contiguous point array of shape
`(nz, ny, nx, 3)` covering the entire domain, evaluates the SDF on it, and returns a
single `(nz, ny, nx)` array. Both arrays must fit in RAM simultaneously.

**AMReX path (`SDFLibrary3D.from_geometry`)** iterates over MultiFab patch boxes via
`MFIter`. For each box it allocates a small local point array, evaluates the SDF on that
box only, and writes the result back. No single array covering the full domain is ever
allocated.

## Memory scaling

At resolution N³ the numpy path requires a contiguous allocation of
`N³ × 3 × 8 bytes` for the point array plus `N³ × 8 bytes` for the output — roughly
`32 × N³` bytes total.

| Resolution | numpy allocation | AMReX peak per box (32³ box) |
|---|---|---|
| 64³ | 12 MB | < 1 MB |
| 256³ | 0.8 GB | < 1 MB |
| 512³ | 6.4 GB | < 1 MB |

At large resolutions the numpy path requires a single multi-gigabyte contiguous
allocation, which will fail or cause swapping before computation time becomes the
bottleneck. The AMReX path's per-box allocations remain small regardless of total
domain size.

## When to use each path

| Use case | Recommended path |
|---|---|
| Quick visualisation, small grids (≤ 128³) | `sample_levelset_3d` |
| Production grids, large domains (≥ 256³) | `SDFLibrary3D` |
| AMR workflows (fill only specific patch boxes) | `SDFLibrary3D` |

## Note on compute speed

Both paths call the same Python/numpy SDF code — there are no native C++ kernels.
AMReX adds ~10–17% overhead from the MFIter loop and `to_numpy()` round-trips on each
box. The AMReX path is not faster; it is memory-safe at scale.
