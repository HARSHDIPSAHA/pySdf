"""
Subtraction Example: Sphere with a hole cut out

Mathematical expectation:
- Base sphere at (0, 0, 0) radius 0.4
- Cutter sphere at (0.2, 0, 0) radius 0.25
- Subtraction = max(-base, cutter) = max(-base, cutter)
- At origin (0,0,0):
  - Base distance = 0 - 0.4 = -0.4 (inside)
  - Cutter distance = sqrt(0.2^2) - 0.25 = 0.2 - 0.25 = -0.05 (inside cutter)
  - Subtraction = max(-(-0.4), -0.05) = max(0.4, -0.05) = 0.4 (outside result)
- At (0.2, 0, 0) (cutter center):
  - Base distance = sqrt(0.2^2) - 0.4 = 0.2 - 0.4 = -0.2 (inside base)
  - Cutter distance = 0 - 0.25 = -0.25 (inside cutter)
  - Subtraction = max(0.2, -0.25) = 0.2 (outside result, hole created)
- At (0.5, 0, 0) (far from both):
  - Base distance = 0.5 - 0.4 = 0.1 (outside)
  - Cutter distance = sqrt(0.3^2) - 0.25 = 0.3 - 0.25 = 0.05 (outside)
  - Subtraction = max(0.1, 0.05) = 0.1 (outside)
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

        # Create base sphere and cutter
        base = lib.sphere(center=(0.0, 0.0, 0.0), radius=0.4)
        cutter = lib.sphere(center=(0.2, 0.0, 0.0), radius=0.25)
        sub = lib.subtract(base, cutter)

        # Gather values
        all_vals = []
        for mfi in sub:
            arr = sub.array(mfi).to_numpy()
            vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
            all_vals.append(vals.flatten())
        phi = np.concatenate(all_vals)

        print("=" * 60)
        print("SUBTRACTION EXAMPLE: Sphere with hole cut out")
        print("=" * 60)
        print(f"Min value: {phi.min():.6f}")
        print(f"Max value: {phi.max():.6f}")
        print(f"Has negative values: {(phi < 0).any()}")
        print(f"Has positive values: {(phi > 0).any()}")

        # Mathematical verification: subtraction = max(-base, cutter)
        base_vals = []
        cutter_vals = []
        for mfi in base:
            arr = base.array(mfi).to_numpy()
            vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
            base_vals.append(vals.flatten())
        for mfi in cutter:
            arr = cutter.array(mfi).to_numpy()
            vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
            cutter_vals.append(vals.flatten())
        base_phi = np.concatenate(base_vals)
        cutter_phi = np.concatenate(cutter_vals)

        expected_sub = np.maximum(-base_phi, cutter_phi)
        max_diff = np.abs(phi - expected_sub).max()
        print(f"Max difference from expected max(-base, cutter): {max_diff:.6e}")

        success = (
            max_diff < 1e-5 and
            (phi > 0).any()  # Should have outside regions
        )
        print("\n" + "=" * 60)
        if success:
            print("✅ SUBTRACTION TEST PASSED: Matches max(-base, cutter) exactly")
        else:
            print("❌ SUBTRACTION TEST FAILED")
        print("=" * 60)

    finally:
        amr.finalize()


if __name__ == "__main__":
    main()
