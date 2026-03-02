"""Intersection of two overlapping spheres.

Demonstrates: Intersection3D, sample_levelset_3d
Output:       examples/intersection_example.png

Mathematical identity verified:
    Intersection(A, B)(p) == max(A(p), B(p))
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sdf3d import Sphere3D, Intersection3D, sample_levelset_3d

_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
_RES    = (64, 64, 64)
_OUT    = os.path.join(os.path.dirname(__file__), "intersection_example.png")


def _render_png(phi, out_path, title=""):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from skimage import measure
    except ImportError:
        print("  scikit-image / matplotlib not available — skipping PNG")
        return

    lo = _BOUNDS[0][0]
    spacing = (_BOUNDS[0][1] - lo) / phi.shape[0]
    if phi.min() >= 0 or phi.max() <= 0:
        print("  No zero crossing — cannot render isosurface.")
        return

    verts, faces, _, _ = measure.marching_cubes(phi, level=0, spacing=(spacing,) * 3)
    verts += lo

    tris  = verts[faces]
    norms = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nlen  = np.linalg.norm(norms, axis=1, keepdims=True)
    norms = norms / np.where(nlen > 0, nlen, 1.0)
    shade = 0.3 + 0.7 * np.clip(norms @ np.array([0.577, 0.577, 0.577]), 0, 1)
    fc    = np.column_stack([shade * 0.3, shade * 0.7, shade * 1.0, np.ones_like(shade)])

    fig = plt.figure(figsize=(5, 5), facecolor="#111")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111"); ax.set_axis_off(); ax.set_box_aspect([1, 1, 1])
    ax.add_collection3d(Poly3DCollection(verts[faces], facecolors=fc, edgecolors="none"))
    hi = _BOUNDS[0][1]
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
    ax.set_title(title, color="white", fontsize=10)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    print("=" * 60)
    print("INTERSECTION: two overlapping spheres")
    print("  S1: centre (0,   0, 0)  radius 0.35")
    print("  S2: centre (0.2, 0, 0)  radius 0.35")
    print("=" * 60)

    s1 = Sphere3D(0.35)
    s2 = Sphere3D(0.35).translate(0.2, 0.0, 0.0)
    geom = Intersection3D(s1, s2)

    phi    = sample_levelset_3d(geom, _BOUNDS, _RES)
    phi_s1 = sample_levelset_3d(s1,   _BOUNDS, _RES)
    phi_s2 = sample_levelset_3d(s2,   _BOUNDS, _RES)

    # --- mathematical verification ---
    expected = np.maximum(phi_s1, phi_s2)
    max_diff = np.abs(phi - expected).max()

    print(f"\nSDF range : [{phi.min():.4f}, {phi.max():.4f}]")
    print(f"Inside (phi<0): {(phi < 0).any()}   Outside (phi>0): {(phi > 0).any()}")
    print(f"max |Intersection - max(S1,S2)| = {max_diff:.2e}  (should be ~0)")

    # --- spot check: origin is inside both spheres ---
    p = np.array([[[0.0, 0.0, 0.0]]])
    v_s1 = float(s1.sdf(p).flat[0]); v_s2 = float(s2.sdf(p).flat[0]); v_i = float(geom.sdf(p).flat[0])
    print(f"\nAt origin: S1={v_s1:.4f}  S2={v_s2:.4f}  Intersection={v_i:.4f}  (expected {max(v_s1,v_s2):.4f})")

    ok = max_diff < 1e-5 and phi.min() < 0 and phi.max() > 0
    print("\n" + ("PASSED PASSED" if ok else "FAILED FAILED"))

    _render_png(phi, _OUT, "Intersection: S1 intersect S2")


if __name__ == "__main__":
    main()
