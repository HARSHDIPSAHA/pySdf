import numpy as np
import matplotlib.pyplot as plt
import geometry_2d as geo
import grid_2d as grid
import os

def main():
    # Build a 2D union of shapes
    circle = geo.Circle(radius=0.3)
    box = geo.Box2D(half_size=(0.2, 0.2)).translate(0.4, 0.0)
    hexagon = geo.Hexagon2D(radius=0.25).translate(-0.4, 0.3)
    star = geo.Star5(outer_radius=0.2, inner_factor=0.5).translate(0.0, -0.4)
    moon = geo.Moon2D(distance=0.15, radius_a=0.25, radius_b=0.2).translate(0.4, 0.4)
    
    geom = geo.Union2D(circle, box, hexagon, star, moon).rotate(np.deg2rad(15.0))

    bounds = ((-1.0, 1.0), (-1.0, 1.0))
    resolution = (512, 512)

    print("Computing level set...")
    phi = grid.sample_levelset_2d(geom, bounds, resolution)
    
    os.makedirs("output", exist_ok=True)
    grid.save_npy("output/levelset_2d.npy", phi)
    
    print("✓ Saved output/levelset_2d.npy")
    print(f"  Shape: {phi.shape}")
    print(f"  Range: [{phi.min():.4f}, {phi.max():.4f}]")
    
    # Create visualization
    print("\nGenerating visualizations...")
    
    # Setup coordinate grid for plotting
    x = np.linspace(bounds[0][0], bounds[0][1], resolution[0])
    y = np.linspace(bounds[1][0], bounds[1][1], resolution[1])
    X, Y = np.meshgrid(x, y)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Filled contour plot (distance field)
    ax = axes[0, 0]
    levels = np.linspace(phi.min(), phi.max(), 30)
    cf = ax.contourf(X, Y, phi.T, levels=levels, cmap='RdBu_r')
    ax.contour(X, Y, phi.T, levels=[0], colors='black', linewidths=2.5)
    plt.colorbar(cf, ax=ax, label='Signed Distance')
    ax.set_title('Signed Distance Field', fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 2. Binary mask (inside/outside)
    ax = axes[0, 1]
    mask = phi < 0  # Inside shapes
    ax.imshow(mask.T, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]], 
              origin='lower', cmap='gray', interpolation='nearest')
    ax.contour(X, Y, phi.T, levels=[0], colors='red', linewidths=2)
    ax.set_title('Binary Mask (Black = Inside)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 3. 3D surface plot of distance field
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    surf = ax.plot_surface(X, Y, phi.T, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.set_title('3D Distance Field', fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Distance')
    plt.colorbar(surf, ax=ax, shrink=0.5, label='Distance')
    
    # 4. Contour lines only
    ax = axes[1, 1]
    contour_levels = np.linspace(phi.min(), phi.max(), 20)
    cs = ax.contour(X, Y, phi.T, levels=contour_levels, cmap='coolwarm', linewidths=1)
    ax.contour(X, Y, phi.T, levels=[0], colors='black', linewidths=3)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    ax.set_title('Distance Contours (Black = Boundary)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/levelset_2d_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved output/levelset_2d_visualization.png")
    
    # Additional: Just the shapes (high quality)
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    
    # Create smooth filled shape
    ax2.contourf(X, Y, phi.T, levels=[-1, 0], colors=['#2E86AB'], alpha=0.7)
    ax2.contour(X, Y, phi.T, levels=[0], colors='#A23B72', linewidths=3)
    
    # Add some iso-distance contours for effect
    positive_levels = [0.05, 0.1, 0.15, 0.2, 0.3]
    negative_levels = [-0.2, -0.15, -0.1, -0.05]
    ax2.contour(X, Y, phi.T, levels=positive_levels, colors='orange', linewidths=0.8, alpha=0.5)
    ax2.contour(X, Y, phi.T, levels=negative_levels, colors='cyan', linewidths=0.8, alpha=0.5)
    
    ax2.set_title('2D Shapes with Distance Isolines', fontsize=16, fontweight='bold')
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(bounds[0])
    ax2.set_ylim(bounds[1])
    
    plt.savefig('output/levelset_2d_shapes.png', dpi=200, bbox_inches='tight')
    print("✓ Saved output/levelset_2d_shapes.png")
    
    print("\n✅ All done! Check the 'output' folder for:")
    print("   - levelset_2d.npy (raw data)")
    print("   - levelset_2d_visualization.png (4-panel analysis)")
    print("   - levelset_2d_shapes.png (clean shape visualization)")

if __name__ == "__main__":
    main()

