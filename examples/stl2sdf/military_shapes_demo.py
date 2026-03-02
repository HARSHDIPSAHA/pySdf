"""military_shapes_demo.py — Artillery shell & missile STL → SDF demo.

Loads the two STL files from the same directory, computes their signed distance
fields on a regular 3-D grid, and produces an interactive Plotly HTML report
showing the φ = 0 isosurface for each shape.

Usage
-----
uv run python examples/stl2sdf/military_shapes_demo.py            # res=50
uv run python examples/stl2sdf/military_shapes_demo.py --res 80  # higher quality
uv run python examples/stl2sdf/military_shapes_demo.py --out my_report.html
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_EXAMPLES_DIR = Path(__file__).parent
_OUTPUT_DIR   = _EXAMPLES_DIR / "output"

SHAPES = [
    {"name": "Artillery Shell", "stem": "artillery_shell"},
    {"name": "Missile",         "stem": "missile"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_bounds(triangles: np.ndarray, pad_frac: float = 0.10):
    """Return ((x0,x1),(y0,y1),(z0,z1)) with fractional padding around mesh."""
    verts = triangles.reshape(-1, 3)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    pad    = pad_frac * (hi - lo)
    return tuple(zip((lo - pad).tolist(), (hi + pad).tolist()))


def _process_shape(shape: dict, res: int) -> dict | None:
    from stl2sdf import stl_to_geometry
    from stl2sdf._math import _stl_to_triangles
    from sdf3d.grid import sample_levelset_3d

    stl_path = _EXAMPLES_DIR / f"{shape['stem']}.stl"
    npy_path = _OUTPUT_DIR   / f"{shape['stem']}_sdf.npy"

    print(f"\n{'='*60}", flush=True)
    print(f"  {shape['name']}", flush=True)
    print(f"{'='*60}", flush=True)

    if not stl_path.exists():
        print(f"  SKIP: {stl_path.name} not found in examples/stl2sdf/", flush=True)
        return None

    triangles = _stl_to_triangles(stl_path)
    n_tri     = len(triangles)
    bounds    = _auto_bounds(triangles)
    print(f"  Triangles : {n_tri:>10,}", flush=True)
    print(f"  Grid      : {res}³ = {res**3:,} points", flush=True)

    geom = stl_to_geometry(stl_path)
    t0   = time.perf_counter()
    phi  = sample_levelset_3d(geom, bounds, (res, res, res))
    elapsed = time.perf_counter() - t0

    print(f"  Done in {elapsed:.1f} s", flush=True)
    print(f"  Inside fraction: {(phi < 0).mean() * 100:.1f}%", flush=True)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, phi)
    print(f"  Saved: {npy_path}", flush=True)

    return {
        "name":   shape["name"],
        "phi":    phi,
        "bounds": bounds,
        "n_tri":  n_tri,
        "res":    res,
        "time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _build_report(results: list[dict], out_html: Path) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("\nplotly not installed; skipping HTML report (uv sync --extra viz)",
              file=sys.stderr)
        return

    n = len(results)
    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "scene"}] * n],
        subplot_titles=[
            f"{r['name']} — {r['n_tri']:,} tris | {r['res']}³ | {r['time_s']:.1f}s"
            for r in results
        ],
        horizontal_spacing=0.08,
    )

    for col, r in enumerate(results, start=1):
        phi    = r["phi"]
        bounds = r["bounds"]
        res    = r["res"]
        (x0, x1), (y0, y1), (z0, z1) = bounds

        xs = np.linspace(x0, x1, res, endpoint=False) + (x1 - x0) / (2.0 * res)
        ys = np.linspace(y0, y1, res, endpoint=False) + (y1 - y0) / (2.0 * res)
        zs = np.linspace(z0, z1, res, endpoint=False) + (z1 - z0) / (2.0 * res)
        Z3, Y3, X3 = np.meshgrid(zs, ys, xs, indexing="ij")

        fig.add_trace(
            go.Isosurface(
                x=X3.ravel(), y=Y3.ravel(), z=Z3.ravel(),
                value=phi.ravel(),
                isomin=0.0, isomax=0.0, surface_count=1,
                colorscale=[[0, "#c0392b"], [1, "#c0392b"]],
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.4, roughness=0.3),
                lightposition=dict(x=1000, y=1000, z=2000),
            ),
            row=1, col=col,
        )

    fig.update_scenes(aspectmode="data")
    fig.update_layout(
        title=dict(
            text="Military Shape Gallery — φ = 0 Surfaces",
            font=dict(size=18),
        ),
        width=600 * n,
        height=600,
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
    )

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"\nSaved interactive report: {out_html}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Artillery shell & missile STL → SDF demo"
    )
    parser.add_argument(
        "--res", type=int, default=50,
        help="Grid resolution per axis (default 50)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=_OUTPUT_DIR / "military_shapes_report.html",
        help="Output HTML report path",
    )
    args = parser.parse_args()

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Military SDF Demo — {len(SHAPES)} shape(s) at res={args.res}")
    results = []
    for shape in SHAPES:
        result = _process_shape(shape, args.res)
        if result is not None:
            results.append(result)

    if results:
        _build_report(results, args.out)

    print(f"\nAll done.  {len(results)}/{len(SHAPES)} shape(s) succeeded.")


if __name__ == "__main__":
    main()
