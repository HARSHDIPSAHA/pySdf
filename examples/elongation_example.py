"""Elongation: sphere stretched into a capsule.

Demonstrates: Sphere3D.elongate(), sample_levelset_3d
Output:       examples/elongation_example.png

Elongation formula:
    q = p - clamp(p, -h, h)      (collapse the middle band to the sphere)
    phi_elongated(p) = phi_sphere(q)

This preserves the SDF metric: the distance gradient magnitude stays 1
everywhere, making the result a true signed distance function.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sdf3d import Sphere3D, sample_levelset_3d

_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
_RES    = (64, 64, 64)
_OUT    = os.path.join(os.path.dirname(__file__), "elongation_example.png")


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
    fc    = np.column_stack([shade * 0.3, shade * 0.9, shade * 0.4, np.ones_like(shade)])

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
    R = 0.25
    H = 0.30   # elongation half-length along X

    print("=" * 60)
    print(f"ELONGATION: sphere (r={R}) elongated along X by h={H}")
    print(f"  Result is a capsule spanning X in [{-(R+H):.2f}, {R+H:.2f}]")
    print("=" * 60)

    sphere    = Sphere3D(R)
    elongated = sphere.elongate(H, 0.0, 0.0)

    phi = sample_levelset_3d(elongated, _BOUNDS, _RES)

    print(f"\nSDF range : [{phi.min():.4f}, {phi.max():.4f}]")
    print(f"Inside (phi<0): {(phi < 0).any()}   Outside (phi>0): {(phi > 0).any()}")

    # --- spot checks ---
    # At origin: q=(0,0,0) -> sphere gives -R
    p0 = np.array([[[0.0, 0.0, 0.0]]])
    print(f"\nAt origin (expected ~ {-R:.2f}):    phi = {float(elongated.sdf(p0).flat[0]):.4f}")

    # At (H, 0, 0): q=(0,0,0) -> same as origin
    p1 = np.array([[[H, 0.0, 0.0]]])
    print(f"At (H, 0, 0) (expected ~ {-R:.2f}): phi = {float(elongated.sdf(p1).flat[0]):.4f}")

    # At (H+R, 0, 0): on the surface -> 0
    p2 = np.array([[[H + R, 0.0, 0.0]]])
    print(f"At (H+R, 0, 0) (expected ~  0.00): phi = {float(elongated.sdf(p2).flat[0]):.4f}")

    # At (H+R+0.1, 0, 0): just outside -> +0.1
    p3 = np.array([[[H + R + 0.1, 0.0, 0.0]]])
    print(f"At (H+R+0.1, 0, 0) (expected ~ +0.10): phi = {float(elongated.sdf(p3).flat[0]):.4f}")

    ok = phi.min() < -R * 0.8 and phi.max() > 0
    print("\n" + ("PASSED PASSED" if ok else "FAILED FAILED"))

    _render_png(phi, _OUT, f"Elongation: sphere r={R}, h={H}")


if __name__ == "__main__":
    main()
