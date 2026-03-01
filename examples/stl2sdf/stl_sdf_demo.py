"""stl_sdf_demo.py — ISS Multi-Tool Wrench → SDF demo.

Downloads the Wrench.stl from NASA's public 3D-printing archive (the first
object ever 3D-printed in space, Dec 2014) and computes its signed distance
field on a uniform grid.

Usage
-----
python examples/stl_sdf_demo.py           # default --res 40 (~2-5 min)
python examples/stl_sdf_demo.py --res 20  # quick draft (~15 s)
python examples/stl_sdf_demo.py --res 60  # higher quality (~20 min)

Outputs
-------
wrench_sdf.npy   — (nz, ny, nx) float64 signed distance field
wrench_sdf.html  — interactive Plotly figure:
                     left panel  — 2D mid-Z SDF heatmap
                     right panel — 3D isosurface at φ = 0
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# STL download
# ---------------------------------------------------------------------------
_WRENCH_URL = (
    "https://raw.githubusercontent.com/nasa/NASA-3D-Resources"
    "/master/3D%20Printing/Wrench/Wrench.stl"
)
_EXAMPLES_DIR = Path(__file__).parent
_LOCAL_STL = _EXAMPLES_DIR / "wrench.stl"


def _download_stl(url: str, dest: Path) -> None:
    print(f"Downloading {url} ...", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest} ({dest.stat().st_size // 1024} KB)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Wrench STL → SDF demo")
    parser.add_argument(
        "--res", type=int, default=40,
        help="Cubic grid resolution (default 40; use 20 for quick test)"
    )
    parser.add_argument(
        "--stl", type=Path, default=_LOCAL_STL,
        help="Path to STL file (downloaded if not present)"
    )
    parser.add_argument(
        "--out", type=Path, default=_EXAMPLES_DIR / "wrench_sdf.npy",
        help="Output .npy path"
    )
    args = parser.parse_args()

    # --- ensure STL exists ---
    if not args.stl.exists():
        try:
            _download_stl(_WRENCH_URL, args.stl)
        except Exception as exc:
            print(f"ERROR: Could not download STL: {exc}", file=sys.stderr)
            print("Please download manually and pass --stl <path>", file=sys.stderr)
            sys.exit(1)

    # --- load and inspect ---
    from stl2sdf import load_stl, sample_sdf_from_stl

    triangles = load_stl(args.stl)
    print(f"Loaded {len(triangles)} triangles from {args.stl}", flush=True)

    # --- auto bounds from mesh bbox + 10% padding ---
    verts = triangles.reshape(-1, 3)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    pad    = 0.1 * (hi - lo)
    lo    -= pad
    hi    += pad
    bounds = tuple(zip(lo.tolist(), hi.tolist()))  # ((x0,x1),(y0,y1),(z0,z1))
    print(f"Bounds (with 10% pad): {bounds}", flush=True)

    res = args.res
    print(f"Sampling {res}³ = {res**3:,} points × {len(triangles):,} triangles ...", flush=True)
    print("  (This is O(F×N) — use --res 20 for a ~15 s quick run)", flush=True)

    phi = sample_sdf_from_stl(args.stl, bounds, (res, res, res))
    print(f"Done.  phi.shape={phi.shape}  min={phi.min():.4f}  max={phi.max():.4f}")

    np.save(args.out, phi)
    print(f"Saved SDF to {args.out}")

    # --- plot ---
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed; skipping plot  (uv sync --extra viz)", file=sys.stderr)
        return

    (x0, x1), (y0, y1), (z0, z1) = bounds
    xs = np.linspace(x0, x1, res, endpoint=False) + (x1 - x0) / (2.0 * res)
    ys = np.linspace(y0, y1, res, endpoint=False) + (y1 - y0) / (2.0 * res)
    zs = np.linspace(z0, z1, res, endpoint=False) + (z1 - z0) / (2.0 * res)

    # Build flat coordinate arrays for Isosurface
    Z3, Y3, X3 = np.meshgrid(zs, ys, xs, indexing="ij")

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=[
            f"Mid-Z SDF slice  (z ≈ {zs[res // 2]:.1f} mm)",
            "3D isosurface  (φ = 0)",
        ],
        horizontal_spacing=0.08,
    )

    # --- left: 2D heatmap of mid-Z slice ---
    mid_z  = phi[res // 2]          # (ny, nx)
    clim   = float(np.abs(mid_z).max())
    fig.add_trace(
        go.Heatmap(
            z=mid_z,
            x=xs,
            y=ys,
            colorscale="RdBu",
            reversescale=True,       # red = outside (+), blue = inside (-)
            zmid=0.0,
            zmin=-clim,
            zmax=clim,
            colorbar=dict(
                title=dict(text="φ (mm)", side="right"),
                x=0.44,
                len=0.8,
            ),
        ),
        row=1, col=1,
    )
    fig.update_xaxes(title_text="X (mm)", row=1, col=1, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="Y (mm)", row=1, col=1)

    # --- right: 3D isosurface at phi=0 ---
    fig.add_trace(
        go.Isosurface(
            x=X3.ravel(),
            y=Y3.ravel(),
            z=Z3.ravel(),
            value=phi.ravel(),
            isomin=0.0,
            isomax=0.0,
            surface_count=1,
            colorscale=[[0, "#4a90d9"], [1, "#4a90d9"]],  # flat steel-blue
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.5),
            lightposition=dict(x=100, y=200, z=300),
        ),
        row=1, col=2,
    )
    fig.update_scenes(
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        zaxis_title="Z (mm)",
        aspectmode="data",
        row=1, col=2,
    )

    fig.update_layout(
        title=dict(
            text=f"ISS Wrench SDF — {res}³ grid",
            font=dict(size=16),
        ),
        width=1200,
        height=650,
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
    )

    out_html = args.out.with_suffix(".html")
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved interactive plot to {out_html}")


if __name__ == "__main__":
    main()
