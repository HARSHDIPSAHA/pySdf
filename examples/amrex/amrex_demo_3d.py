"""3D AMReX SDF demo.

Demonstrates the full AMReX-native visualization pipeline for 3D shapes:

    Geometry → SDFLibrary3D → MultiFab → write_single_level_plotfile
    → yt (SurfaceSource or marching-cubes fallback) → PNG

Run with:
    conda run -n pyamrex python examples/amrex/amrex_demo_3d.py

Outputs PNGs to examples/amrex/output/.

IMPORTANT: Run this script in its own process — amrex.space3d cannot coexist
with amrex.space2d in the same Python interpreter (pybind11 type-name conflict).
"""

import os
import sys

# ---------------------------------------------------------------------------
# AMReX + renderer imports
# ---------------------------------------------------------------------------
try:
    import amrex.space3d as amr
except ImportError as exc:
    raise SystemExit(
        "amrex.space3d not found. Install via conda:\n"
        "  conda install pyamrex -c conda-forge"
    ) from exc

# render_surface lives in the same directory as this script
sys.path.insert(0, os.path.dirname(__file__))
from render_surface_from_plotfile import render_surface

# pySdf root so we can import SDFLibrary3D
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sdf3d.amrex import SDFLibrary3D

# ---------------------------------------------------------------------------
# AMReX initialisation
# ---------------------------------------------------------------------------
amr.initialize([])

# Grid: 48^3 cells covering [-0.6, 0.6]^3
# 48 keeps render time modest; bump to 64+ for higher fidelity
n = 48
lo, hi = -0.6, 0.6
real_box = amr.RealBox([lo, lo, lo], [hi, hi, hi])
domain   = amr.Box([0, 0, 0], [n - 1, n - 1, n - 1])

# coord_sys=0 → Cartesian; is_periodic=[0,0,0] → non-periodic
geom = amr.Geometry(domain, real_box, 0, [0, 0, 0])

ba = amr.BoxArray(domain)
ba.max_size(n // 2)
dm = amr.DistributionMapping(ba)

lib = SDFLibrary3D(geom, ba, dm)

# ---------------------------------------------------------------------------
# Build MultiFabs for each shape
# ---------------------------------------------------------------------------
mf_sphere    = lib.sphere((0, 0, 0), 0.3)
mf_box       = lib.box((0, 0, 0), (0.25, 0.2, 0.15))
mf_round_box = lib.round_box((0, 0, 0), (0.25, 0.2, 0.15), 0.05)
mf_union     = lib.union(mf_sphere, mf_box)
mf_subtract  = lib.subtract(mf_box, mf_sphere)  # box with sphere subtracted

shapes = [
    ("sphere",    mf_sphere),
    ("box",       mf_box),
    ("round_box", mf_round_box),
    ("union",     mf_union),
    ("subtract",  mf_subtract),
]

# ---------------------------------------------------------------------------
# Write plotfiles and render PNGs
# ---------------------------------------------------------------------------
out_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(out_dir, exist_ok=True)

for name, mf in shapes:
    pf_dir  = os.path.join(out_dir, f"plt_{name}")
    png_out = os.path.join(out_dir, f"{name}.png")

    # --- Step 1: write AMReX plotfile directory ---
    amr.write_single_level_plotfile(pf_dir, mf, amr.Vector_string(["sdf"]), geom, 0.0, 0)

    # --- Step 2: render the SDF=0 isosurface ---
    # render_surface tries yt SurfaceSource first; falls back to marching cubes
    # (scikit-image) if SurfaceSource is unavailable in this yt build.
    render_surface(pf_dir, png_out)
    print(f"Saved {name}.png")

amr.finalize()
print("Done.")
