"""Complex example: chaining Union, Elongation, Intersection, and Subtraction.

Demonstrates: Box3D, Sphere3D, Union3D, Intersection3D, Subtraction3D,
              Geometry3D.elongate(), sample_levelset_3d
Output:       examples/complex_example_step*.png  +  examples/complex_example_final.png

Build sequence
--------------
1. Base box at origin
2. Capsule: sphere elongated along X
3. Union: box union capsule
4. Rounded top: intersect with a large sphere offset in +Y
5. Cavity: subtract a small central box
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sdf3d import Sphere3D, Box3D, Union3D, Intersection3D, Subtraction3D, sample_levelset_3d

_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
_RES    = (64, 64, 64)
_DIR    = os.path.dirname(__file__)


def _render_png(phi, out_path, title="", color=(0.4, 0.7, 1.0)):
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
        print(f"  No zero crossing — skipping {os.path.basename(out_path)}")
        return

    verts, faces, _, _ = measure.marching_cubes(phi, level=0, spacing=(spacing,) * 3)
    verts += lo

    tris  = verts[faces]
    norms = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nlen  = np.linalg.norm(norms, axis=1, keepdims=True)
    norms = norms / np.where(nlen > 0, nlen, 1.0)
    shade = 0.3 + 0.7 * np.clip(norms @ np.array([0.577, 0.577, 0.577]), 0, 1)
    fc    = np.column_stack([shade * color[0], shade * color[1], shade * color[2],
                             np.ones_like(shade)])

    fig = plt.figure(figsize=(5, 5), facecolor="#111")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111"); ax.set_axis_off(); ax.set_box_aspect([1, 1, 1])
    ax.add_collection3d(Poly3DCollection(verts[faces], facecolors=fc, edgecolors="none"))
    hi = _BOUNDS[0][1]
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
    ax.set_title(title, color="white", fontsize=9)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"  Saved: {out_path}")


def _info(label, phi):
    print(f"  {label:30s}  range=[{phi.min():+.3f}, {phi.max():+.3f}]"
          f"  inside={( phi<0).any()}  outside={(phi>0).any()}")


def main():
    print("=" * 60)
    print("COMPLEX EXAMPLE: chain of SDF operations")
    print("=" * 60)

    # Step 1 — base box
    base = Box3D([0.30, 0.30, 0.30])
    phi1 = sample_levelset_3d(base, _BOUNDS, _RES)
    _info("Step 1 — box", phi1)
    _render_png(phi1, os.path.join(_DIR, "complex_example_step1.png"),
                "Step 1: Box", color=(0.5, 0.7, 1.0))

    # Step 2 — capsule (elongated sphere)
    capsule = Sphere3D(0.18).elongate(0.25, 0.0, 0.0)
    phi2    = sample_levelset_3d(capsule, _BOUNDS, _RES)
    _info("Step 2 — capsule", phi2)
    _render_png(phi2, os.path.join(_DIR, "complex_example_step2.png"),
                "Step 2: Capsule", color=(0.3, 0.9, 0.4))

    # Step 3 — union
    union_geom = Union3D(base, capsule)
    phi3       = sample_levelset_3d(union_geom, _BOUNDS, _RES)
    _info("Step 3 — union", phi3)
    _render_png(phi3, os.path.join(_DIR, "complex_example_step3.png"),
                "Step 3: Union", color=(0.9, 0.8, 0.2))

    # Step 4 — intersection with a large sphere -> rounds the top
    rounder    = Sphere3D(0.60).translate(0.0, 0.2, 0.0)
    rounded    = Intersection3D(union_geom, rounder)
    phi4       = sample_levelset_3d(rounded, _BOUNDS, _RES)
    _info("Step 4 — intersection (round top)", phi4)
    _render_png(phi4, os.path.join(_DIR, "complex_example_step4.png"),
                "Step 4: Intersection (rounded)", color=(1.0, 0.5, 0.3))

    # Step 5 — subtract a small central box -> cavity
    cutter     = Box3D([0.08, 0.08, 0.08]).translate(0.0, 0.05, 0.0)
    final      = Subtraction3D(rounded, cutter)
    phi5       = sample_levelset_3d(final, _BOUNDS, _RES)
    _info("Step 5 — subtraction (cavity)", phi5)
    _render_png(phi5, os.path.join(_DIR, "complex_example_final.png"),
                "Final: with cavity", color=(0.7, 0.4, 1.0))

    print("\nFinal shape:")
    print(f"  Inside voxels : {(phi5 < 0).sum():,}")
    print(f"  On surface    : {(np.abs(phi5) < 0.02).sum():,}")

    ok = phi5.min() < 0 and phi5.max() > 0
    print("\n" + ("PASSED PASSED" if ok else "FAILED FAILED"))


if __name__ == "__main__":
    main()
