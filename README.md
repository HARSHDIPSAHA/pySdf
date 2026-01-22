# Signed Distance Functions with pyAMReX

This repo implements the signed distance functions and operators listed on
the Inigo Quilez "Distance Functions" article, and evaluates them with
pyAMReX on a 2D grid (z = 0 slice) to generate visualization PNGs. The
visualization workflow uses pyAMReX to build a structured grid, split it into
boxes, distribute those boxes, and store the SDF values in a `MultiFab`.
[pyAMReX](https://pyamrex.readthedocs.io/en/latest/) and the original SDF
formulas are referenced from
[iquilezles.org](https://iquilezles.org/articles/distfunctions/).

## Files

- `sdf_lib.py`: numpy implementations of the SDF primitives and operators.
- `render_all_sdfs.py`: builds a 2D AMReX grid, evaluates each SDF, and saves
  images into `vis/`.

## Run

```bash
python render_all_sdfs.py
```

### 3D volume renders (yt)

This uses yt volume rendering to make 3D snapshots and saves them to `vis3d/`:

```bash
python render_all_sdfs_3d.py
```

Notes:
- Requires `yt` (`pip install yt`).
- Rendering all shapes can take a while; reduce `n` in `render_all_sdfs_3d.py`
  if you need faster previews.

### 3D renders from AMReX plotfiles (yt)

If you have AMReX plotfiles (e.g., `plt00000/`), yt can load them directly
and you can volume render or slice them. Example:

```python
import yt

ds = yt.load("./plt00000/")
sc = yt.create_scene(ds, field="sdf")
sc.save("rendering.png")
```

This is useful when you want to render the full AMReX dataset (including
refinement) instead of the uniform grid used in `render_all_sdfs_3d.py`.

## Output

Each SDF produces a `vis/<name>.png` visualization with a diverging colormap
and the zero level-set contour drawn in black.

The 3D volume renders are saved to `vis3d/<name>.png`.

### Plotfiles + 3D snapshots (yt)

This workflow writes AMReX plotfiles for each SDF and then renders each
plotfile to a PNG:

```bash
python render_all_sdfs_plotfiles.py
```

Outputs:
- `plotfiles/<name>/` (AMReX plotfiles)
- `vis3d_plotfile/<name>.png` (volume renders)

Notes:
- Requires pyAMReX built in 3D and `yt` installed.

## Color guide for `vis/` images

The PNGs use a diverging colormap:

- Blue tones: negative SDF values (inside the shape).
- Red tones: positive SDF values (outside the shape).
- White/near zero: near the surface.
- Black contour line: the zero level set (the shape boundary).

For `vis3d/`, the volume render highlights values near the zero level set,
so the visible surface corresponds to the SDF = 0 boundary.

## How pyAMReX helps

pyAMReX provides the grid/mesh infrastructure that makes it easy to evaluate
an SDF on a structured domain and scale the work later if needed:

- `BoxArray` defines how the domain is split into tiles.
- `DistributionMapping` assigns each tile to a compute resource.
- `MultiFab` stores grid-aligned data (the SDF values) for each tile.

In `render_all_sdfs.py`, the line below allocates the SDF storage:

```python
sdf_mf = amr.MultiFab(ba, dm, 1, 0)
```

`MultiFab(BoxArray, DistributionMapping, ncomp, ngrow)` means:

- `ba`: how the domain is split into boxes.
- `dm`: who owns each box (CPU/GPU/threads).
- `1`: one component per cell (the SDF value).
- `0`: no ghost cells.

Because `ncomp = 1` and `ngrow = 0`, each tile is accessed as
`arr[:, :, 0, 0]`, which is the scalar SDF field for that box.

## Short syntax guide

These are the core AMReX objects used in `render_all_sdfs.py`, explained in
plain terms:

- `amr.RealBox(prob_lo, prob_hi)`: defines the physical bounds of the domain
  (e.g., x and y from 0 to 1).
- `amr.Box(lo, hi)`: defines the integer index region (grid indices), like
  i = 0..n-1 and j = 0..n-1.
- `amr.Geometry(domain, real_box, coord, is_periodic)`: ties index space to
  physical space and stores geometry info such as cell spacing (`dx`).
- `amr.BoxArray(domain)`: describes how the index domain is split into boxes.
  `max_size` limits each box size for parallelism and cache efficiency.
- `amr.DistributionMapping(ba)`: assigns those boxes to compute resources.
- `amr.MultiFab(ba, dm, ncomp, ngrow)`: stores grid data for each box.
  Here it stores one SDF value per cell with no ghost cells.

## Notes

- All 3D SDFs are evaluated on the z = 0 slice for visualization.
- `udTriangle` and `udQuad` are unsigned distance fields by definition.
