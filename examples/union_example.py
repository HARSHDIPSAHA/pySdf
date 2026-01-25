"""
Union Example: Two overlapping spheres

Mathematical expectation:
- Two spheres: S1 at (-0.3, 0, 0) radius 0.25, S2 at (0.3, 0, 0) radius 0.25
- Union = min(S1, S2) at each point
- At origin (0,0,0):
  - S1 distance = sqrt(0.3^2) - 0.25 = 0.3 - 0.25 = 0.05 (outside)
  - S2 distance = sqrt(0.3^2) - 0.25 = 0.05 (outside)
  - Union = min(0.05, 0.05) = 0.05 (outside, as expected)
- At (-0.3, 0, 0) (center of S1):
  - S1 distance = 0 - 0.25 = -0.25 (inside)
  - S2 distance = sqrt(0.6^2) - 0.25 = 0.6 - 0.25 = 0.35 (outside)
  - Union = min(-0.25, 0.35) = -0.25 (inside, as expected)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import amrex.space3d as amr
from sdf3d import SDFLibrary
import numpy as np


def main():
    amr.initialize([])
    try:
        # Setup grid
        real_box = amr.RealBox([-1, -1, -1], [1, 1, 1])
        domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(63, 63, 63))
        geom = amr.Geometry(domain, real_box, 0, [0, 0, 0])
        ba = amr.BoxArray(domain)
        ba.max_size(32)
        dm = amr.DistributionMapping(ba)

        lib = SDFLibrary(geom, ba, dm)

        # Create two overlapping spheres
        s1 = lib.sphere(center=(-0.3, 0.0, 0.0), radius=0.25)
        s2 = lib.sphere(center=(0.3, 0.0, 0.0), radius=0.25)
        union = lib.union(s1, s2)

        # Gather values for verification
        all_vals = []
        for mfi in union:
            arr = union.array(mfi).to_numpy()
            vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
            all_vals.append(vals.flatten())
        phi = np.concatenate(all_vals)

        # Expected checks
        print("=" * 60)
        print("UNION EXAMPLE: Two overlapping spheres")
        print("=" * 60)
        print(f"Min value (should be < 0, inside): {phi.min():.6f}")
        print(f"Max value (should be > 0, outside): {phi.max():.6f}")
        print(f"Has negative values (inside): {(phi < 0).any()}")
        print(f"Has positive values (outside): {(phi > 0).any()}")
        print(f"Has near-zero (surface): {(np.abs(phi) < 0.05).any()}")

        # Mathematical verification at origin
        # At (0,0,0): distance to S1 center = 0.3, so S1 SDF = 0.3 - 0.25 = 0.05
        # Similarly S2 SDF = 0.05, so union = min(0.05, 0.05) = 0.05
        # We expect some cells near origin to have values around 0.05
        near_origin = (np.abs(phi - 0.05) < 0.1).any()
        print(f"Has values near expected origin value (0.05): {near_origin}")

        # Success criteria
        success = (
            phi.min() < 0 and
            phi.max() > 0 and
            (phi < 0).any() and
            (phi > 0).any()
        )
        print("\n" + "=" * 60)
        if success:
            print("✅ UNION TEST PASSED: Output matches expected behavior")
        else:
            print("❌ UNION TEST FAILED: Unexpected values")
        print("=" * 60)

    finally:
        amr.finalize()


if __name__ == "__main__":
    main()
