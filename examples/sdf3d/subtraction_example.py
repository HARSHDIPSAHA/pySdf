"""Subtraction: sphere with a spherical cavity.

Demonstrates: Subtraction3D (base.subtract(cutter)), sample_levelset_3d
Output:       examples/subtraction_example.png

Argument order reminder:
    opSubtraction(d1, d2) = max(-d1, d2)   where d1=CUTTER, d2=BASE
    Subtraction3D(base, cutter)             same convention
    base.subtract(cutter)                   fluent form

Mathematical identity verified:
    Subtraction(base, cutter)(p) == max(-cutter(p), base(p))
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sdf3d import Sphere3D, Subtraction3D, sample_levelset_3d

_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
_RES    = (64, 64, 64)
_OUT    = os.path.join(os.path.dirname(__file__), "subtraction_example.png")


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
    fc    = np.column_stack([shade * 1.0, shade * 0.4, shade * 0.3, np.ones_like(shade)])

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
    print("SUBTRACTION: sphere with spherical cavity")
    print("  Base:   centre (0,   0, 0)  radius 0.40")
    print("  Cutter: centre (0.2, 0, 0)  radius 0.25")
    print("=" * 60)

    base   = Sphere3D(0.40)
    cutter = Sphere3D(0.25).translate(0.2, 0.0, 0.0)
    geom   = Subtraction3D(base, cutter)     # max(-cutter, base)

    phi      = sample_levelset_3d(geom,   _BOUNDS, _RES)
    phi_base = sample_levelset_3d(base,   _BOUNDS, _RES)
    phi_cut  = sample_levelset_3d(cutter, _BOUNDS, _RES)

    # --- mathematical verification ---
    expected = np.maximum(-phi_cut, phi_base)
    max_diff = np.abs(phi - expected).max()

    print(f"\nSDF range : [{phi.min():.4f}, {phi.max():.4f}]")
    print(f"Inside (phi<0): {(phi < 0).any()}   Outside (phi>0): {(phi > 0).any()}")
    print(f"max |Subtraction - max(-cutter, base)| = {max_diff:.2e}  (should be ~0)")

    # --- spot checks ---
    # At origin: base=-0.4, cutter=0.2-0.25=-0.05 -> result = max(0.05, -0.4) = 0.05 (outside)
    p   = np.array([[[0.0, 0.0, 0.0]]])
    v_b = float(base.sdf(p).flat[0]); v_c = float(cutter.sdf(p).flat[0]); v_r = float(geom.sdf(p).flat[0])
    print(f"\nAt origin:         base={v_b:.4f}  cutter={v_c:.4f}  result={v_r:.4f}  (expected {max(-v_c,v_b):.4f})")

    # At (-0.4, 0, 0): outside cutter, inside base -> stays inside
    p2  = np.array([[[-0.4, 0.0, 0.0]]])
    v_b2 = float(base.sdf(p2).flat[0]); v_c2 = float(cutter.sdf(p2).flat[0]); v_r2 = float(geom.sdf(p2).flat[0])
    print(f"At (-0.4, 0, 0):   base={v_b2:.4f}  cutter={v_c2:.4f}  result={v_r2:.4f}  (expected {max(-v_c2,v_b2):.4f})")

    ok = max_diff < 1e-5 and (phi < 0).any() and (phi > 0).any()
    print("\n" + ("PASSED PASSED" if ok else "FAILED FAILED"))

    _render_png(phi, _OUT, "Subtraction: base - cutter")


if __name__ == "__main__":
    main()
