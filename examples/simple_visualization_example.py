#!/usr/bin/env python3
"""
Example: Using save_levelset_html for easy visualization

This demonstrates the simplified visualization API requested in the feature.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sdf3d import Sphere, sample_levelset, save_levelset_html  # noqa: E402

# Create an elongated sphere (similar to the issue example)
sphere = Sphere(0.25).elongate(0.3, 0.0, 0.0)

# Sample the level set
bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
res = (64, 64, 64)
phi = sample_levelset(sphere, bounds, res)

# Save directly to HTML - as simple as requested!
save_levelset_html(phi, bounds=(-1, 1), filename="outputs/elongated_sphere.html")

print("✅ Visualization complete!")
print(
    "   Open outputs/elongated_sphere.html in your browser to view "
    "the 3D interactive visualization."
)
