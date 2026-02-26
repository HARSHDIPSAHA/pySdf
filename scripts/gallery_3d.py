"""Render every sdf_lib 3D primitive as an isosurface on one page.

Uses marching cubes (scikit-image) to extract the SDF=0 surface and
matplotlib's 3-D axes to display it — no AMReX or yt required.

Usage::

    python scripts/gallery_3d.py                   # saves gallery_3d.png
    python scripts/gallery_3d.py --out my_file.png
    python scripts/gallery_3d.py --res 48          # faster, lower quality

Requirements: numpy, matplotlib, scikit-image
    pip install scikit-image
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

# Ensure the repo root (parent of scripts/) is importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sdf3d import sdf_lib as sdf
from sdf3d.examples import NATOFragment, RocketAssembly


# ---------------------------------------------------------------------------
# Minimal stand-in for SDFLibrary3D (no AMReX required)
# ---------------------------------------------------------------------------

class _MockLib:
    def from_geometry(self, geom):
        return geom


# ---------------------------------------------------------------------------
# Shape catalogue  (label, sdf_func)
# ---------------------------------------------------------------------------

def _make_shapes() -> list[tuple[str, object]]:
    S  = 0.3
    bh = np.array([0.25, 0.2, 0.15])
    r  = 0.05

    base_sphere     = lambda p: sdf.sdSphere(p, S)
    base_box        = lambda p: sdf.sdBox(p, bh)
    base_round_box  = lambda p: sdf.sdRoundBox(p, bh, r)

    def rot_z(th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    shapes = [
        # --- primitives ---
        ("sdSphere",
         lambda p: sdf.sdSphere(p, S)),
        ("sdBox",
         lambda p: sdf.sdBox(p, bh)),
        ("sdRoundBox",
         lambda p: sdf.sdRoundBox(p, bh, r)),
        ("sdBoxFrame",
         lambda p: sdf.sdBoxFrame(p, np.array([0.25, 0.25, 0.25]), 0.06)),
        ("sdTorus",
         lambda p: sdf.sdTorus(p, np.array([0.22, 0.08]))),
        ("sdCappedTorus",
         lambda p: sdf.sdCappedTorus(p, np.array([0.5, 0.866]), 0.22, 0.06)),
        ("sdLink",
         lambda p: sdf.sdLink(p, 0.15, 0.18, 0.06)),
        ("sdCylinder",
         lambda p: sdf.sdCylinder(p, np.array([0., 0., 0.28]))),
        ("sdConeExact",
         lambda p: sdf.sdConeExact(p, np.array([0.6, 0.8]), 0.35)),
        ("sdConeBound",
         lambda p: sdf.sdConeBound(p, np.array([0.6, 0.8]), 0.35)),
        ("sdHexPrism",
         lambda p: sdf.sdHexPrism(p, np.array([0.25, 0.22]))),
        ("sdTriPrism",
         lambda p: sdf.sdTriPrism(p, np.array([0.3, 0.22]))),
        ("sdCapsule",
         lambda p: sdf.sdCapsule(p, np.array([-0.3, 0., 0.]), np.array([0.3, 0., 0.]), 0.15)),
        ("sdVertCapsule",
         lambda p: sdf.sdVerticalCapsule(p, 0.4, 0.15)),
        ("sdCappedCyl",
         lambda p: sdf.sdCappedCylinder(p, 0.22, 0.28)),
        ("sdCappedCylSeg",
         lambda p: sdf.sdCappedCylinderSegment(
             p, np.array([0., -0.22, 0.]), np.array([0., 0.22, 0.]), 0.2)),
        ("sdRoundedCyl",
         lambda p: sdf.sdRoundedCylinder(p, 0.2, 0.05, 0.25)),
        ("sdCappedCone",
         lambda p: sdf.sdCappedCone(p, 0.3, 0.2, 0.05)),
        ("sdCappedConeSeg",
         lambda p: sdf.sdCappedConeSegment(
             p, np.array([0., -0.28, 0.]), np.array([0., 0.28, 0.]), 0.2, 0.05)),
        ("sdSolidAngle",
         lambda p: sdf.sdSolidAngle(p, np.array([0.5, 0.866]), 0.38)),
        ("sdCutSphere",
         lambda p: sdf.sdCutSphere(p, 0.35, 0.1)),
        ("sdCutHollowSphere",
         lambda p: sdf.sdCutHollowSphere(p, 0.35, 0.1, 0.04)),
        ("sdDeathStar",
         lambda p: sdf.sdDeathStar(p, 0.32, 0.18, 0.28)),
        ("sdRoundCone",
         lambda p: sdf.sdRoundCone(p, 0.25, 0.05, 0.4)),
        ("sdRoundConeSeg",
         lambda p: sdf.sdRoundConeSegment(
             p, np.array([0., -0.25, 0.]), np.array([0., 0.25, 0.]), 0.2, 0.05)),
        ("sdEllipsoid",
         lambda p: sdf.sdEllipsoid(p, np.array([0.35, 0.25, 0.2]))),
        ("sdVesicaSeg",
         lambda p: sdf.sdVesicaSegment(
             p, np.array([-0.25, 0., 0.]), np.array([0.25, 0., 0.]), 0.2)),
        ("sdRhombus",
         lambda p: sdf.sdRhombus(p, 0.3, 0.22, 0.15, 0.05)),
        ("sdOctahedronExact",
         lambda p: sdf.sdOctahedronExact(p, 0.38)),
        ("sdOctahedronBound",
         lambda p: sdf.sdOctahedronBound(p, 0.38)),
        ("sdPyramid",
         lambda p: sdf.sdPyramid(p, 0.4)),
        # udTriangle / udQuad are unsigned (no interior) — skip in gallery

        # --- boolean operations ---
        ("opUnion",
         lambda p: sdf.opUnion(base_sphere(p), base_box(p))),
        ("opSubtraction",
         lambda p: sdf.opSubtraction(base_sphere(p), base_box(p))),
        ("opIntersection",
         lambda p: sdf.opIntersection(base_sphere(p), base_box(p))),
        ("opXor",
         lambda p: sdf.opXor(base_round_box(p), base_sphere(p))),
        ("opSmoothUnion",
         lambda p: sdf.opSmoothUnion(base_sphere(p), base_box(p), 0.15)),
        ("opSmoothSubtract",
         # subtract sphere from a larger box so the sphere is not fully enclosed
         lambda p: sdf.opSmoothSubtraction(base_sphere(p), sdf.sdBox(p, np.array([0.42, 0.42, 0.42])), 0.12)),
        ("opSmoothIntersect",
         lambda p: sdf.opSmoothIntersection(base_sphere(p), base_box(p), 0.15)),
        # --- domain operations ---
        ("opRevolution",
         lambda p: sdf.opRevolution(p, lambda q: sdf.sdCircle(q, 0.2), 0.18)),
        ("opExtrusion",
         lambda p: sdf.opExtrusion(p, lambda q: sdf.sdBox(q, np.array([0.2, 0.2])), 0.12)),
        ("opElongate1",
         lambda p: sdf.opElongate1(p, base_sphere, np.array([0.2, 0.1, 0.]))),
        ("opElongate2",
         lambda p: sdf.opElongate2(p, base_sphere, np.array([0.2, 0.1, 0.]))),
        ("opRound",
         lambda p: sdf.opRound(p, base_box, 0.06)),
        ("opOnion",
         lambda p: sdf.opOnion(base_sphere(p), 0.06)),
        ("opScale",
         lambda p: sdf.opScale(p, 0.8, base_sphere)),
        ("opSymX",
         lambda p: sdf.opSymX(p, base_box)),
        ("opSymXZ",
         lambda p: sdf.opSymXZ(p, base_box)),
        ("opRepetition",
         lambda p: sdf.opRepetition(p, np.array([0.55, 0.55, 0.55]), base_sphere)),
        ("opLimitedRep",
         lambda p: sdf.opLimitedRepetition(
             p, np.array([0.55, 0.55, 0.55]), np.array([1., 1., 1.]), base_sphere)),
        ("opDisplace",
         lambda p: sdf.opDisplace(p, base_sphere)),
        ("opTwist",
         lambda p: sdf.opTwist(p, base_round_box, 5.0)),
        ("opCheapBend",
         lambda p: sdf.opCheapBend(p, base_round_box, 5.0)),
        ("opTx",
         lambda p: sdf.opTx(p, rot_z(np.deg2rad(30.)), np.array([0.1, 0.1, 0.]), base_box)),
        # --- examples ---
        # NATOFragment: diameter=0.3 → total_length=0.75; shift by -0.375 to centre in Z
        *_make_example_shapes(),
    ]
    return shapes


def _make_example_shapes() -> list[tuple[str, object]]:
    lib = _MockLib()

    # NATOFragment: diameter=0.3 → spans Z=[0, 0.75]; centre at Z=0.375
    _, nato_geom = NATOFragment(lib, diameter=0.3)
    _nato_offset = np.array([0.0, 0.0, -0.375])
    nato_func = lambda p: nato_geom.sdf(p - _nato_offset)

    # RocketAssembly: scaled down to fit in [-0.55, 0.55] domain
    _, rocket_geom = RocketAssembly(
        lib, body_radius=0.08, L_extra=0.18, nose_len=0.12,
        fin_span=0.06, fin_height=0.08, fin_thickness=0.018,
    )
    rocket_func = lambda p: rocket_geom.sdf(p)

    return [
        ("NATOFragment", nato_func),
        ("RocketAssembly", rocket_func),
    ]


# ---------------------------------------------------------------------------
# Evaluation + marching cubes
# ---------------------------------------------------------------------------

def _build_volume(n: int, lo: float = -0.55, hi: float = 0.55) -> np.ndarray:
    coords = np.linspace(lo, hi, n)
    Z, Y, X = np.meshgrid(coords, coords, coords, indexing="ij")
    return sdf.vec3(X, Y, Z)


def _eval_surface(sdf_func, p_vol: np.ndarray, n: int):
    """Return (verts, faces) of the zero isosurface, or None on failure."""
    try:
        from skimage import measure
    except ImportError:
        raise SystemExit(
            "scikit-image is required for 3-D rendering.\n"
            "  pip install scikit-image"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vals = sdf_func(p_vol).astype(float)
        # marching cubes needs at least one pos and neg value
        if vals.max() <= 0 or vals.min() >= 0:
            return None
        verts, faces, _, _ = measure.marching_cubes(vals, level=0.0)
        # map voxel indices → world coords [-0.55, 0.55]
        lo, hi = -0.55, 0.55
        scale = (hi - lo) / (n - 1)
        verts = verts * scale + lo
        return verts, faces
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_gallery(shapes, out_path: str, ncols: int = 8, res: int = 80) -> None:
    nrows = (len(shapes) + ncols - 1) // ncols
    fig = plt.figure(figsize=(ncols * 3.0, nrows * 3.0), facecolor="#111111")

    p_vol = _build_volume(res)

    _FACE_COLOR   = np.array([1.0, 0.82, 0.2])   # warm gold
    _EDGE_COLOR   = "none"
    _VIEW_ELEV    = 20
    _VIEW_AZIM    = 35

    for idx, (label, sdf_func) in enumerate(shapes):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        ax.set_facecolor("#111111")
        ax.set_axis_off()
        ax.set_title(label, color="white", fontsize=6.5, pad=1)

        result = _eval_surface(sdf_func, p_vol, res)
        if result is None:
            ax.text2D(0.5, 0.5, "no surface", ha="center", va="center",
                      color="gray", transform=ax.transAxes, fontsize=7)
            continue

        verts, faces = result
        # Compute face normals for diffuse shading
        tris  = verts[faces]           # (F, 3, 3)
        e1    = tris[:, 1] - tris[:, 0]
        e2    = tris[:, 2] - tris[:, 0]
        norms = np.cross(e1, e2)
        nlen  = np.linalg.norm(norms, axis=1, keepdims=True)
        norms = norms / np.where(nlen > 0, nlen, 1.0)
        light = np.array([0.577, 0.577, 0.577])   # diagonal illumination
        diffuse = np.clip(norms @ light, 0.0, 1.0)
        shade = 0.3 + 0.7 * diffuse               # ambient + diffuse
        face_colors = np.outer(shade, _FACE_COLOR)
        mesh = Poly3DCollection(verts[faces], facecolors=face_colors,
                                edgecolors="none", alpha=1.0)
        ax.add_collection3d(mesh)

        lo, hi = -0.55, 0.55
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=_VIEW_ELEV, azim=_VIEW_AZIM)

    # Hide empty trailing axes (subplot adds blank ones automatically in some mpl versions)
    total_slots = nrows * ncols
    for idx in range(len(shapes), total_slots):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        ax.set_visible(False)

    fig.suptitle("sdf_lib — 3D Signed Distance Function Gallery", color="white",
                 fontsize=13, y=1.002)
    plt.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render all sdf_lib 3D primitives to a single PNG gallery."
    )
    parser.add_argument("--out",  default="gallery_3d.png", help="Output PNG path")
    parser.add_argument("--cols", type=int, default=8, help="Number of columns (default 8)")
    parser.add_argument("--res",  type=int, default=48,
                        help="Grid resolution per axis (default 80, use 96+ for highest quality)")
    args = parser.parse_args()

    shapes = _make_shapes()
    render_gallery(shapes, args.out, ncols=args.cols, res=args.res)


if __name__ == "__main__":
    main()
