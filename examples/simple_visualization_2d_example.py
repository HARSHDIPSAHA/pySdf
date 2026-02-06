#!/usr/bin/env python3
"""
Example: Using save_levelset_html_2d for easy 2D visualization

This demonstrates the simplified 2D visualization API.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sdf2d import (  # noqa: E402
    Box2D,
    Circle,
    Union2D,
    sample_levelset_2d,
    save_levelset_html_2d,
)

# Create a combined shape
circle = Circle(0.3).translate(-0.3, 0)
box = Box2D((0.25, 0.25)).translate(0.3, 0)
combined = Union2D(circle, box)

# Sample the level set
bounds = ((-1.0, 1.0), (-1.0, 1.0))
res = (256, 256)
phi = sample_levelset_2d(combined, bounds, res)

# Save directly to HTML - simple and easy!
save_levelset_html_2d(phi, bounds=(-1, 1), filename="outputs/combined_shape_2d.html")

print("✅ 2D visualization complete!")
print(
    "   Open outputs/combined_shape_2d.html in your browser to view "
    "the interactive visualization."
)
