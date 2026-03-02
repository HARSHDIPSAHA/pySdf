"""nasa_boolean_demo.py — Boolean operations between a NASA STL mesh and analytic shapes.

Demonstrates that geometry_from_stl returns a plain Geometry3D that composes
with the same union / subtraction API as analytic primitives (Sphere3D, Box3D).

Shapes
------
* Mars 2020 Wheel  (mars_wheel.stl, ~45 K triangles)
  Normalised aspect: 1.0 x 1.0 x 0.58 — squat cylinder with tread detail.
  Must be present in the same directory as this script.

* Sphere3D(0.3)  shifted to (0.5, 0, 0) — 50% overlap with the wheel in X.

Operations shown (4 panels)
----------------------------
1. Wheel alone
2. Sphere alone
3. Wheel union Sphere        — wheel with sphere bulging from its rim
4. Wheel subtract Sphere     — wheel with a spherical bite taken from its rim

Usage
-----
python examples/stl2sdf/nasa_boolean_demo.py           # res=8,  ~90 s
python examples/stl2sdf/nasa_boolean_demo.py --res 12  # higher quality, ~3 min
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

# Sphere positioned so it overlaps the wheel rim by ~50 % in X
_SPHERE_CENTRE = (0.5, 0.0, 0.0)
_SPHERE_RADIUS = 0.3

# Bounds covering wheel (x in [-0.5,0.5]) + sphere (x in [0.2,0.8]) + 15 % pad
_BOUNDS = ((-0.65, 0.75), (-0.65, 0.65), (-0.65, 0.65))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_wheel():
    """Return a centred, unit-scale Geometry3D for mars_wheel.stl."""
    from stl2sdf._math import _stl_to_triangles

    if not _WHEEL_STL.exists():
        raise FileNotFoundError(
            f"{_WHEEL_STL} not found. "
            "Place mars_wheel.stl in the examples/stl2sdf/ directory."
        )

    tris  = _stl_to_triangles(_WHEEL_STL)
    verts = tris.reshape(-1, 3)
    lo, hi = verts.min(0), verts.max(0)
    centre = (lo + hi) / 2.0
    extent = (hi - lo).max()

    print(
        f"  Mars 2020 Wheel: {len(tris):,} tris  "
        f"normalised aspect {((hi - lo) / extent).round(2).tolist()}",
        flush=True,
    )
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
        rows=2, cols=n,
        specs=[[{"type": "xy"}] * n, [{"type": "scene"}] * n],
        subplot_titles=[
            *[f"{p['label']}<br>{p['time_s']:.0f}s" for p in panels],
            *["phi=0 surface"] * n,
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.04,
    )

    for col, p in enumerate(panels, start=1):
        phi = p["phi"]

        # Mid-Z slice: z=0 cuts through full wheel width and sphere equator
        mid  = phi[res // 2, :, :]   # (ny, nx)
        clim = float(np.abs(mid).max()) or 1.0

        fig.add_trace(
            go.Heatmap(
                z=mid, x=xs, y=ys,
                colorscale="RdBu", reversescale=True,
                zmid=0.0, zmin=-clim, zmax=clim,
                showscale=(col == 1),
                colorbar=dict(title=dict(text="phi"), x=-0.02) if col == 1 else None,
            ),
            row=1, col=col,
        )
        fig.update_xaxes(title_text="x", row=1, col=col)
        fig.update_yaxes(
            title_text="y",
            scaleanchor=f"x{col if col > 1 else ''}",
            row=1, col=col,
        )

        fig.add_trace(
            go.Isosurface(
                x=X3.ravel(), y=Y3.ravel(), z=Z3.ravel(),
                value=phi.ravel(),
                isomin=0.0, isomax=0.0, surface_count=1,
                colorscale=[[0, "#4a90d9"], [1, "#4a90d9"]],
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),
            ),
            row=2, col=col,
        )

    fig.update_layout(
        title=dict(
            text="geometry_from_stl + Sphere3D: Mars Wheel boolean ops",
            font=dict(size=15),
        ),
        width=340 * n,
        height=900,
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
        "--res", type=int, default=8,
        help="Grid resolution per axis (default 8; use 12 for better quality)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=_HERE / "nasa_boolean_report.html",
    )
    args = parser.parse_args()

    print("Loading mesh ...\n")
    wheel  = _load_wheel()
    sphere = Sphere3D(_SPHERE_RADIUS).translate(*_SPHERE_CENTRE)

    operations = [
        ("Wheel",              wheel),
        ("Sphere",             sphere),
        ("Wheel union Sphere", wheel.union(sphere)),
        ("Wheel - Sphere",     wheel.subtract(sphere)),
    ]

    panels: list[dict] = []
    for label, geom in operations:
        print(f"\nSampling '{label}' at res={args.res} ...", flush=True)
        t0  = time.perf_counter()
        phi = _sample(geom, args.res)
        dt  = time.perf_counter() - t0
        print(
            f"  {dt:.1f}s  phi=[{phi.min():.3f}, {phi.max():.3f}]"
            f"  inside={( phi < 0).mean()*100:.1f}%",
            flush=True,
        )
        panels.append({"label": label, "phi": phi, "res": args.res, "time_s": dt})

    _build_report(panels, args.out)
    print("\nAll done.")


if __name__ == "__main__":
    main()
