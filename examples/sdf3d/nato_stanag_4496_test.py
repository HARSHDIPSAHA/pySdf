"""NATO STANAG-4496 fragment impact geometry.

Demonstrates: NATOFragment, Box3D, Union3D, Geometry3D transforms,
              sample_levelset_3d
Output:       examples/nato_fragment.png
              examples/nato_impact_scene.png

Scene setup
-----------
1. Build the fragment geometry (cylinder + sharp cone nose) via NATOFragment.
2. Build a 50 mm target block.
3. Position the fragment 20 mm in front of the target at a 5 deg yaw angle.
4. Union fragment + target into a single SDF for the solver.

No AMReX required — geometry is evaluated via sample_levelset_3d.
In production, pass an SDFLibrary3D instance to NATOFragment instead of
the MockLib used here.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sdf3d import Box3D, Union3D, sample_levelset_3d
from sdf3d.examples import NATOFragment

_DIR = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# Mock library — returns the geometry object directly (no AMReX MultiFab)
# ---------------------------------------------------------------------------

class _MockLib:
    def from_geometry(self, geom):
        return geom


# ---------------------------------------------------------------------------
# Render helper
# ---------------------------------------------------------------------------

def _render_png(phi, bounds, out_path, title="", color=(0.6, 0.7, 0.9)):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from skimage import measure
    except ImportError:
        print("  scikit-image / matplotlib not available — skipping PNG")
        return

    lo, hi = bounds[0][0], bounds[0][1]
    spacing = (hi - lo) / phi.shape[0]
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

    fig = plt.figure(figsize=(6, 5), facecolor="#111")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111"); ax.set_axis_off(); ax.set_box_aspect([1, 1, 1])
    ax.add_collection3d(Poly3DCollection(verts[faces], facecolors=fc, edgecolors="none"))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
    ax.set_title(title, color="white", fontsize=9)
    ax.view_init(elev=20, azim=45)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("NATO STANAG-4496 FRAGMENT IMPACT SCENE")
    print("=" * 65)

    # Fragment dimensions (STANAG-4496 standard)
    diameter = 14.30e-3   # 14.30 mm
    lib = _MockLib()
    _, fragment_geom = NATOFragment(lib, diameter=diameter, L_over_D=1.09,
                                    cone_angle_deg=20.0)

    total_length = diameter * 1.09   # ~ 15.56 mm

    print(f"\nFragment: dia{diameter*1e3:.2f} mm x L={total_length*1e3:.2f} mm")
    print(f"  Cone half-angle: 20 deg")

    # ------------------------------------------------------------------
    # Visualise fragment alone (small domain, metres)
    # ------------------------------------------------------------------
    frag_bounds = (
        (-diameter, diameter),
        (-diameter, diameter),
        (-diameter * 0.2, total_length + diameter * 0.2),
    )
    phi_frag = sample_levelset_3d(fragment_geom, frag_bounds, (48, 48, 80))
    inside   = (phi_frag < 0).sum()
    print(f"\nFragment grid ({48}x{48}x{80}):")
    print(f"  SDF range : [{phi_frag.min():.4e}, {phi_frag.max():.4e}] m")
    print(f"  Inside voxels: {inside}")

    _render_png(phi_frag, frag_bounds,
                os.path.join(_DIR, "nato_fragment.png"),
                title=f"STANAG-4496 fragment  dia{diameter*1e3:.1f} mm",
                color=(0.7, 0.75, 0.8))

    # ------------------------------------------------------------------
    # Impact scene: fragment positioned 20 mm in front of 50 mm target
    # ------------------------------------------------------------------
    target_half  = 0.025           # 50 mm cube
    target_z     = 0.060           # target centre at z = 60 mm
    gap          = 0.020           # 20 mm clearance before tip

    # Fragment: base at z=0, tip at z=total_length.
    # Shift so the tip is gap ahead of the target front face.
    target_front = target_z - target_half
    z_tip_target = target_front - gap
    z_shift      = z_tip_target - total_length

    positioned = (fragment_geom
                  .rotate_y(np.radians(5.0))      # 5 deg yaw
                  .translate(0.0, 0.0, z_shift))

    target = Box3D([target_half] * 3).translate(0.0, 0.0, target_z)
    scene  = Union3D(positioned, target)

    scene_lo = -0.04
    scene_hi =  0.10
    scene_bounds = ((scene_lo, scene_hi),) * 3
    phi_scene = sample_levelset_3d(scene, scene_bounds, (64, 64, 64))

    print(f"\nImpact scene (fragment + target):")
    print(f"  Target front face : z = {target_front*1e3:.1f} mm")
    print(f"  Fragment tip      : z = {(z_shift + total_length)*1e3:.1f} mm")
    print(f"  Standoff gap      : {gap*1e3:.0f} mm")
    print(f"  Yaw angle         : 5 deg")
    print(f"  SDF range : [{phi_scene.min():.4e}, {phi_scene.max():.4e}] m")

    _render_png(phi_scene, scene_bounds,
                os.path.join(_DIR, "nato_impact_scene.png"),
                title="Impact scene: fragment + target",
                color=(0.5, 0.65, 0.9))

    ok = (phi_frag < 0).any() and (phi_scene < 0).any()
    print("\n" + ("PASSED PASSED" if ok else "FAILED FAILED"))


if __name__ == "__main__":
    main()
