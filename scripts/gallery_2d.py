"""Render every sdf2d geometry class as a signed-distance heatmap on one page.

Usage::

    python scripts/gallery_2d.py                   # saves gallery_2d.png
    python scripts/gallery_2d.py --out my_file.png # custom output path

Requirements: numpy, matplotlib  (no AMReX needed)
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Sequence

# Ensure the repo root (parent of scripts/) is importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from sdf2d import sample_levelset_2d
import sdf2d as _sdf2d

# ---------------------------------------------------------------------------
# Shape catalogue — (label, geometry_object)
# ---------------------------------------------------------------------------

def _make_shapes() -> list[tuple[str, object]]:
    from sdf2d import (
        Circle2D, Box2D, RoundedBox2D, OrientedBox2D, Segment2D,
        Rhombus2D, Trapezoid2D, Parallelogram2D,
        EquilateralTriangle2D, TriangleIsosceles2D, Triangle2D,
        UnevenCapsule2D,
        Pentagon2D, Hexagon2D, Octagon2D, NGon2D,
        Hexagram2D, Star5_2D, Star2D,
        Pie2D, CutDisk2D, Arc2D, Ring2D, Horseshoe2D,
        Vesica2D, Moon2D, RoundedCross2D, Egg2D, Heart2D, Cross2D, RoundedX2D,
        Polygon2D, Ellipse2D, Parabola2D, ParabolaSegment2D, Bezier2D,
        BlobbyCross2D, Tunnel2D, Stairs2D, QuadraticCircle2D, Hyperbola2D,
        Union2D, Subtraction2D,
    )

    sc = [0.5 * np.sqrt(2), 0.5 * np.sqrt(2)]

    shapes = [
        ("Circle2D",             Circle2D(0.6)),
        ("Box2D",                Box2D((0.5, 0.4))),
        ("RoundedBox2D",         RoundedBox2D((0.45, 0.35), 0.1)),
        ("OrientedBox2D",        OrientedBox2D((-0.35, 0.0), (0.35, 0.0), 0.25)),
        ("Segment2D",            Segment2D((-0.7, -0.3), (0.7, 0.3))),
        ("Rhombus2D",            Rhombus2D((0.5, 0.7))),
        ("Trapezoid2D",          Trapezoid2D(0.55, 0.25, 0.55)),
        ("Parallelogram2D",      Parallelogram2D(0.45, 0.45, 0.25)),
        ("EqTriangle2D",         EquilateralTriangle2D(0.65)),
        ("TriangleIsosceles2D",  TriangleIsosceles2D(0.5, 0.8)),
        ("Triangle2D",           Triangle2D((0.0, 0.7), (-0.6, -0.4), (0.6, -0.4))),
        ("UnevenCapsule2D",      UnevenCapsule2D(0.35, 0.15, 0.7)),
        ("Pentagon2D",           Pentagon2D(0.65)),
        ("Hexagon2D",            Hexagon2D(0.6)),
        ("Octagon2D",            Octagon2D(0.65)),
        ("NGon2D(7)",            NGon2D(0.65, 7)),
        ("Hexagram2D",           Hexagram2D(0.55)),
        ("Star5_2D",             Star5_2D(0.65, 0.4)),
        ("Star2D(5)",            Star2D(0.65, 5, 2.0)),
        ("Pie2D",                Pie2D(sc, 0.7)),
        ("CutDisk2D",            CutDisk2D(0.7, 0.1)),
        ("Arc2D",                Arc2D([0.707, 0.707], 0.6, 0.1)),
        ("Ring2D",               Ring2D(0.35, 0.65)),
        ("Horseshoe2D",          Horseshoe2D([0.0, 1.0], 0.5, [0.15, 0.1])),
        ("Vesica2D",             Vesica2D(0.65, 0.3)),
        ("Moon2D",               Moon2D(0.35, 0.6, 0.4)),
        ("RoundedCross2D",       RoundedCross2D(0.5).scale(0.55)),
        ("Egg2D",                Egg2D(0.5, 0.2)),
        ("Heart2D",              Heart2D()),
        ("Cross2D",              Cross2D((0.2, 0.6), 0.0)),
        ("RoundedX2D",           RoundedX2D(0.5, 0.36)),
        ("Polygon2D",            Polygon2D([(-0.5, -0.5), (0.5, -0.4), (0.6, 0.3), (0.0, 0.6), (-0.5, 0.4)])),
        ("Ellipse2D",            Ellipse2D((0.7, 0.4))),
        ("Parabola2D",           Parabola2D(1.5)),
        ("ParabolaSegment2D",    ParabolaSegment2D(0.55, 0.45)),
        ("Bezier2D",             Bezier2D((-0.6, -0.4), (0.0, 1.0), (0.6, -0.4))),
        ("BlobbyCross2D",        BlobbyCross2D(0.5)),
        ("Tunnel2D",             Tunnel2D((0.45, 0.55))),
        ("Stairs2D",             Stairs2D((0.22, 0.22), 3).translate(-0.33, -0.33)),
        ("QuadraticCircle2D",    QuadraticCircle2D()),
        ("Hyperbola2D",          Hyperbola2D(0.6, 0.4)),
        # Boolean example
        ("Union2D",              Union2D(Circle2D(0.45), Box2D((0.35, 0.35)).translate(0.3, 0.3))),
        ("Subtraction2D",        Subtraction2D(Circle2D(0.65), Circle2D(0.35).translate(0.25, 0.0))),
    ]
    return shapes


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0))
_RES    = (512, 512)
_EXTENT = [-1, 1, -1, 1]


def _eval_shape(geom) -> np.ndarray | None:
    """Evaluate SDF on grid; return None on failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sample_levelset_2d(geom, _BOUNDS, _RES)
    except Exception:
        return None


def render_gallery(shapes: list[tuple[str, object]], out_path: str, ncols: int = 7) -> None:
    nrows = (len(shapes) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.2, nrows * 3.2),
        facecolor="#111111",
    )
    axes = np.asarray(axes).ravel()

    for ax, (label, geom) in zip(axes, shapes):
        phi = _eval_shape(geom)
        ax.set_facecolor("#111111")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, color="white", fontsize=7, pad=3)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        if phi is None:
            ax.text(0.5, 0.5, "eval error", ha="center", va="center",
                    color="red", transform=ax.transAxes, fontsize=8)
            continue

        lim = max(np.nanmax(np.abs(phi)), 1e-6)
        ax.imshow(phi, origin="lower", extent=_EXTENT,
                  cmap="seismic", vmin=-lim, vmax=lim, interpolation="bilinear")
        try:
            ax.contour(phi, levels=[0.0], colors="white", linewidths=1.0,
                       extent=_EXTENT)
        except Exception:
            pass

    # Hide unused axes
    for ax in axes[len(shapes):]:
        ax.set_visible(False)

    fig.suptitle("sdf2d — Signed Distance Function Gallery", color="white",
                 fontsize=13, y=1.002)
    plt.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render all sdf2d shapes to a single PNG gallery.")
    parser.add_argument("--out", default="gallery_2d.png", help="Output PNG path")
    parser.add_argument("--cols", type=int, default=7, help="Number of columns (default 7)")
    args = parser.parse_args()

    shapes = _make_shapes()
    render_gallery(shapes, args.out, ncols=args.cols)


if __name__ == "__main__":
    main()
