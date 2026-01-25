"""
Elongation Example: Elongated sphere using geometry API

Mathematical expectation:
- Base sphere radius 0.25
- Elongated by (0.3, 0.0, 0.0) in x-direction
- Elongation operation: q = p - clamp(p, -h, h), then evaluate sphere(q)
- At origin (0,0,0):
  - q = (0,0,0) - clamp((0,0,0), (-0.3,0,0), (0.3,0,0)) = (0,0,0) - (0,0,0) = (0,0,0)
  - Distance = 0 - 0.25 = -0.25 (inside)
- At (0.4, 0, 0) (beyond elongation):
  - q = (0.4,0,0) - clamp((0.4,0,0), (-0.3,0,0), (0.3,0,0)) = (0.4,0,0) - (0.3,0,0) = (0.1,0,0)
  - Distance = 0.1 - 0.25 = -0.15 (still inside elongated shape)
- At (0.5, 0, 0):
  - q = (0.5,0,0) - (0.3,0,0) = (0.2,0,0)
  - Distance = 0.2 - 0.25 = -0.05 (inside)
- At (0.3, 0, 0) (at elongation boundary):
  - q = (0.3,0,0) - (0.3,0,0) = (0,0,0)
  - Distance = 0 - 0.25 = -0.25 (inside)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sdf3d import Sphere, sample_levelset


def main():
    print("=" * 60)
    print("ELONGATION EXAMPLE: Sphere elongated in x-direction")
    print("=" * 60)

    # Create elongated sphere using geometry API
    sphere = Sphere(0.25)
    elongated = sphere.elongate(0.3, 0.0, 0.0)

    # Sample on grid
    bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    res = (64, 64, 64)
    phi = sample_levelset(elongated, bounds, res)

    print(f"Min value (should be < 0, inside): {phi.min():.6f}")
    print(f"Max value (should be > 0, outside): {phi.max():.6f}")
    print(f"Has negative values (inside): {(phi < 0).any()}")
    print(f"Has positive values (outside): {(phi > 0).any()}")

    # Mathematical verification at specific points
    # Grid coordinates: cell centers
    dx = (bounds[0][1] - bounds[0][0]) / res[0]
    coords = np.linspace(bounds[0][0] + dx/2, bounds[0][1] - dx/2, res[0])

    # Find index closest to origin
    origin_idx = np.argmin(np.abs(coords))
    origin_val = phi[origin_idx, origin_idx, origin_idx]
    print(f"\nValue at origin (expected ~ -0.25): {origin_val:.6f}")

    # Find index closest to (0.3, 0, 0)
    x03_idx = np.argmin(np.abs(coords - 0.3))
    val_at_03 = phi[x03_idx, origin_idx, origin_idx]
    print(f"Value at (0.3, 0, 0) (expected ~ -0.25): {val_at_03:.6f}")

    # Find index closest to (0.5, 0, 0)
    x05_idx = np.argmin(np.abs(coords - 0.5))
    val_at_05 = phi[x05_idx, origin_idx, origin_idx]
    print(f"Value at (0.5, 0, 0) (expected ~ -0.05): {val_at_05:.6f}")

    # Success criteria
    success = (
        phi.min() < 0 and
        phi.max() > 0 and
        (phi < 0).any() and
        (phi > 0).any() and
        origin_val < -0.2 and origin_val > -0.3  # Should be inside
    )

    print("\n" + "=" * 60)
    if success:
        print("✅ ELONGATION TEST PASSED: Values match expected behavior")
    else:
        print("❌ ELONGATION TEST FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
