import numpy as np
import matplotlib.pyplot as plt
import geometry_2d as geo
import grid_2d as grid
import os


def visualize_shape(phi, bounds, title, filename):
    """Helper function to create clean visualization"""
    x = np.linspace(bounds[0][0], bounds[0][1], phi.shape[1])
    y = np.linspace(bounds[1][0], bounds[1][1], phi.shape[0])
    X, Y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Fill the shape
    ax.contourf(X, Y, phi.T, levels=[-1, 0], colors=['#2E86AB'], alpha=0.8)
    
    # Draw boundary
    ax.contour(X, Y, phi.T, levels=[0], colors='#A23B72', linewidths=3)
    
    # Distance isolines
    negative_levels = [-0.2, -0.15, -0.1, -0.05]
    positive_levels = [0.05, 0.1, 0.15, 0.2, 0.3]
    ax.contour(X, Y, phi.T, levels=negative_levels, colors='cyan', linewidths=0.8, alpha=0.5)
    ax.contour(X, Y, phi.T, levels=positive_levels, colors='orange', linewidths=0.8, alpha=0.5)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    
    plt.tight_layout()
    plt.savefig(f'output/{filename}', dpi=150, bbox_inches='tight')
    print(f"✓ Saved output/{filename}")
    plt.close()


def main():
    bounds = ((-1.0, 1.0), (-1.0, 1.0))
    resolution = (512, 512)
    os.makedirs("output", exist_ok=True)
    
    print("Generating simple 2D shape examples...\n")
    
    # ===================================================================
    # Example 1: Single Circle
    # ===================================================================
    print("1. Single Circle")
    circle = geo.Circle(radius=0.4)
    phi = grid.sample_levelset_2d(circle, bounds, resolution)
    visualize_shape(phi, bounds, "Circle (radius=0.4)", "01_circle.png")
    
    # ===================================================================
    # Example 2: Single Square
    # ===================================================================
    print("2. Single Square")
    square = geo.Box2D(half_size=(0.3, 0.3))
    phi = grid.sample_levelset_2d(square, bounds, resolution)
    visualize_shape(phi, bounds, "Square (0.6 × 0.6)", "02_square.png")
    
    # ===================================================================
    # Example 3: Single Hexagon
    # ===================================================================
    print("3. Single Hexagon")
    hexagon = geo.Hexagon2D(radius=0.4)
    phi = grid.sample_levelset_2d(hexagon, bounds, resolution)
    visualize_shape(phi, bounds, "Regular Hexagon", "03_hexagon.png")
    
    # ===================================================================
    # Example 4: Single Star
    # ===================================================================
    print("4. Single 5-Pointed Star")
    star = geo.Star5(outer_radius=0.4, inner_factor=0.5)
    phi = grid.sample_levelset_2d(star, bounds, resolution)
    visualize_shape(phi, bounds, "5-Pointed Star", "04_star.png")
    
    # ===================================================================
    # Example 5: Circle + Circle (Union)
    # ===================================================================
    print("5. Two Circles (Union)")
    circle1 = geo.Circle(radius=0.3).translate(-0.3, 0.0)
    circle2 = geo.Circle(radius=0.3).translate(0.3, 0.0)
    union = geo.Union2D(circle1, circle2)
    phi = grid.sample_levelset_2d(union, bounds, resolution)
    visualize_shape(phi, bounds, "Two Overlapping Circles", "05_two_circles.png")
    
    # ===================================================================
    # Example 6: Circle - Circle (Subtraction)
    # ===================================================================
    print("6. Circle with Bite (Subtraction)")
    big_circle = geo.Circle(radius=0.5)
    small_circle = geo.Circle(radius=0.3).translate(0.3, 0.0)
    subtraction = geo.Subtraction2D(big_circle, small_circle)
    phi = grid.sample_levelset_2d(subtraction, bounds, resolution)
    visualize_shape(phi, bounds, "Circle with Bite Removed", "06_subtraction.png")
    
    # ===================================================================
    # Example 7: Circle ∩ Square (Intersection)
    # ===================================================================
    print("7. Circle ∩ Square (Intersection)")
    circle = geo.Circle(radius=0.4)
    square = geo.Box2D(half_size=(0.35, 0.35))
    intersection = geo.Intersection2D(circle, square)
    phi = grid.sample_levelset_2d(intersection, bounds, resolution)
    visualize_shape(phi, bounds, "Circle ∩ Square", "07_intersection.png")
    
    # ===================================================================
    # Example 8: Rounded Square
    # ===================================================================
    print("8. Rounded Square")
    rounded_square = geo.RoundedBox2D(half_size=(0.3, 0.3), radius=0.1)
    phi = grid.sample_levelset_2d(rounded_square, bounds, resolution)
    visualize_shape(phi, bounds, "Rounded Square", "08_rounded_square.png")
    
    # ===================================================================
    # Example 9: Heart Shape
    # ===================================================================
    print("9. Heart")
    heart = geo.Heart2D()
    phi = grid.sample_levelset_2d(heart, bounds, resolution)
    visualize_shape(phi, bounds, "Heart Shape", "09_heart.png")
    
    # ===================================================================
    # Example 10: Smiley Face
    # ===================================================================
    # ===================================================================
# Example 10: Smiley Face (FIXED)
# ===================================================================
    print("10. Smiley Face")

# Face (outer circle)
    face = geo.Circle(radius=0.5)

# Eyes - just two small circles
    left_eye = geo.Circle(radius=0.08).translate(-0.15, 0.12)
    right_eye = geo.Circle(radius=0.08).translate(0.15, 0.12)

# Smile - create arc using intersection
# Method: Take bottom half of a circle
    smile_circle = geo.Circle(radius=0.25).translate(0.0, 0.0)
    smile_cutoff_top = geo.Box2D(half_size=(0.5, 0.5)).translate(0.0, 0.3)
    smile_cutoff_bottom = geo.Box2D(half_size=(0.5, 0.5)).translate(0.0, -0.4)

# Keep only the middle arc
    smile_band = geo.Subtraction2D(geo.Subtraction2D(smile_circle, smile_cutoff_top), smile_cutoff_bottom)

# Make it thick (ring)
    smile_inner = geo.Circle(radius=0.20).translate(0.0, 0.0)
    smile_ring = geo.Subtraction2D(smile_band, smile_inner)

# Alternative: Simple crescent smile
    smile_outer = geo.Circle(radius=0.28).translate(0.0, 0.15)
    smile_inner2 = geo.Circle(radius=0.25).translate(0.0, 0.18)
    smile = geo.Subtraction2D(smile_outer, smile_inner2)
    smile_bottom_cut = geo.Box2D(half_size=(0.5, 0.15)).translate(0.0, -0.05)
    smile_final = geo.Subtraction2D(smile, smile_bottom_cut)

# Assemble: Face with holes for eyes and smile
    smiley = geo.Subtraction2D(
    geo.Subtraction2D(
        geo.Subtraction2D(face, left_eye), 
        right_eye
    ), 
    smile_final
)

    phi = grid.sample_levelset_2d(smiley, bounds, resolution)
    visualize_shape(phi, bounds, "Smiley Face", "10_smiley.png")

    
    # ===================================================================
    # Example 11: Ring (Donut)
    # ===================================================================
    print("11. Ring (Annulus)")
    ring = geo.Ring2D(inner_radius=0.2, outer_radius=0.4)
    phi = grid.sample_levelset_2d(ring, bounds, resolution)
    visualize_shape(phi, bounds, "Ring / Donut", "11_ring.png")
    
    # ===================================================================
    # Example 12: Triangle
    # ===================================================================
    print("12. Triangle")
    triangle = geo.EquilateralTriangle2D(radius=0.5)
    phi = grid.sample_levelset_2d(triangle, bounds, resolution)
    visualize_shape(phi, bounds, "Equilateral Triangle", "12_triangle.png")
    
    # ===================================================================
    # Example 13: Cross
    # ===================================================================
    print("13. Cross")
    cross = geo.Cross2D(size_vec2=(0.4, 0.15), rounding=0.05)
    phi = grid.sample_levelset_2d(cross, bounds, resolution)
    visualize_shape(phi, bounds, "Rounded Cross", "13_cross.png")
    
    # ===================================================================
    # Example 14: Pill Shape (Stadium)
    # ===================================================================
    print("14. Pill / Stadium")
    pill = geo.UnevenCapsule2D(r1=0.15, r2=0.15, height=0.5)
    phi = grid.sample_levelset_2d(pill, bounds, resolution)
    visualize_shape(phi, bounds, "Pill / Stadium Shape", "14_pill.png")
    
    # ===================================================================
    # Example 15: Gear-like (Multiple Circles)
    # ===================================================================
    print("15. Flower Pattern")
    center = geo.Circle(radius=0.2)
    petals = []
    n_petals = 6
    for i in range(n_petals):
        angle = 2 * np.pi * i / n_petals
        x = 0.3 * np.cos(angle)
        y = 0.3 * np.sin(angle)
        petals.append(geo.Circle(radius=0.15).translate(x, y))
    
    flower = geo.Union2D(center, *petals)
    phi = grid.sample_levelset_2d(flower, bounds, resolution)
    visualize_shape(phi, bounds, "Flower Pattern (6 petals)", "15_flower.png")
    
    print("\n" + "="*60)
    print("✅ All examples generated!")
    print("="*60)
    print("Check the 'output' folder for 15 different shape examples")
    print("Files: 01_circle.png through 15_flower.png")


if __name__ == "__main__":
    main()
