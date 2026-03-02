"""nasa_boolean_demo.py — Boolean operations between a NASA STL mesh and analytic shapes.

Demonstrates that stl_to_geometry returns a plain Geometry3D that composes
with the same union / subtraction API as analytic primitives (Sphere3D, Box3D).

Shapes
------
* Mars Rover Wheel  (mars_wheel.stl, ~45 K triangles)
* Sphere3D(0.20) shifted to (0.35, 0, 0) — overlaps the wheel rim.

Operations shown (4 panels)
----------------------------
1. Wheel alone
2. Sphere alone
3. Wheel union Sphere        — wheel with a sphere fused to its rim
4. Wheel subtract Sphere     — wheel with a spherical bite taken from its rim

Usage
-----
uv run python examples/stl2sdf/nasa_boolean_demo.py           # res=40
uv run python examples/stl2sdf/nasa_boolean_demo.py --res 60  # higher quality
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from sdf3d import Sphere3D
from stl2sdf import stl_to_geometry

_HERE = Path(__file__).parent
_WHEEL_STL = _HERE / "mars_wheel.stl"

# Sphere positioned to overlap the wheel rim by ~50 % in X
_SPHERE_CENTRE = (0.35, 0.0, 0.0)
_SPHERE_RADIUS = 0.20

# Bounds covering wheel (x,y in [-0.5,0.5], z in [-0.30,0.30]) + sphere + 15 % pad
_BOUNDS = ((-0.65, 0.65), (-0.65, 0.65), (-0.45, 0.45))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_wheel():
    """Return a centred, unit-scale Geometry3D for mars_wheel.stl."""
    from stl2sdf._math import _stl_to_triangles

    tris  = _stl_to_triangles(_WHEEL_STL)
    verts = tris.reshape(-1, 3)
    lo, hi = verts.min(0), verts.max(0)
    centre = (lo + hi) / 2.0
    extent = (hi - lo).max()

    print(f"  Mars Rover Wheel: {len(tris):,} tris", flush=True)
    return stl_to_geometry(_WHEEL_STL).translate(*(-centre)).scale(1.0 / extent)


def _sample(geom, res: int) -> np.ndarray:
    (x0, x1), (y0, y1), (z0, z1) = _BOUNDS
    xs = np.linspace(x0, x1, res, endpoint=False) + (x1 - x0) / (2 * res)
    ys = np.linspace(y0, y1, res, endpoint=False) + (y1 - y0) / (2 * res)
    zs = np.linspace(z0, z1, res, endpoint=False) + (z1 - z0) / (2 * res)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    return geom.sdf(np.stack([X, Y, Z], axis=-1))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _build_report(panels: list[dict], out_html: Path) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("\nplotly not installed — skipping HTML (uv sync --extra viz)",
              file=sys.stderr)
        return

    n   = len(panels)
    res = panels[0]["res"]

    (x0, x1), (y0, y1), (z0, z1) = _BOUNDS
    xs = np.linspace(x0, x1, res, endpoint=False) + (x1 - x0) / (2 * res)
    ys = np.linspace(y0, y1, res, endpoint=False) + (y1 - y0) / (2 * res)
    zs = np.linspace(z0, z1, res, endpoint=False) + (z1 - z0) / (2 * res)
    Z3, Y3, X3 = np.meshgrid(zs, ys, xs, indexing="ij")

    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "scene"}] * n],
        subplot_titles=[p["label"] for p in panels],
        horizontal_spacing=0.04,
    )

    for col, p in enumerate(panels, start=1):
        fig.add_trace(
            go.Isosurface(
                x=X3.ravel(), y=Y3.ravel(), z=Z3.ravel(),
                value=p["phi"].ravel(),
                isomin=0.0, isomax=0.0, surface_count=1,
                colorscale=[[0, "#4a90d9"], [1, "#4a90d9"]],
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),
            ),
            row=1, col=col,
        )

    fig.update_scenes(aspectmode="data")
    fig.update_layout(
        title=dict(
            text="Mars Wheel + Sphere3D: boolean operations",
            font=dict(size=15),
        ),
        width=380 * n,
        height=550,
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"\nSaved: {out_html}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Boolean ops: Mars Wheel STL mesh + Sphere3D"
    )
    parser.add_argument(
        "--res", type=int, default=40,
        help="Grid resolution per axis (default 40)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=_HERE / "output" / "nasa_boolean_report.html",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("Loading mesh ...\n")
    wheel  = _load_wheel()
    sphere = Sphere3D(_SPHERE_RADIUS).translate(*_SPHERE_CENTRE)

    operations = [
        ("Wheel",               wheel),
        ("Sphere",              sphere),
        ("Wheel union Sphere",  wheel.union(sphere)),
        ("Wheel - Sphere",      wheel.subtract(sphere)),
    ]

    panels: list[dict] = []
    for label, geom in operations:
        print(f"Sampling '{label}' at res={args.res} ...", flush=True)
        t0  = time.perf_counter()
        phi = _sample(geom, args.res)
        dt  = time.perf_counter() - t0
        print(f"  {dt:.1f}s  inside={(phi < 0).mean()*100:.1f}%", flush=True)
        panels.append({"label": label, "phi": phi, "res": args.res, "time_s": dt})

    _build_report(panels, args.out)
    print("\nAll done.")


if __name__ == "__main__":
    main()
