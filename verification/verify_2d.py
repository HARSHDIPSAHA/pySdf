import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import sdf_lib as sdf

def render_sdf_dynamic(sdf_func, name, bounds=((-2.0, 2.0), (-2.0, 2.0)), res=(512, 512)):
    """
    Renders an SDF with dynamic bounds using the STANDARD visual style.
    """
    
    print(f"Generating visualization for: {name}...")
    
    x_range = np.linspace(bounds[0][0], bounds[0][1], res[0])
    y_range = np.linspace(bounds[1][0], bounds[1][1], res[1])
    X, Y = np.meshgrid(x_range, y_range)
    
    p_grid = sdf.vec2(X, Y) 
    
    try:
        phi = sdf_func(p_grid)
    except Exception as e:
        print(f"❌ ERROR: {name} crashed. Reason: {e}")
        return
    
    output_dir = "verification/outputs/2d"
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Filled contour plot
    ax = axes[0, 0]
    levels = np.linspace(phi.min(), phi.max(), 30)
    cf = ax.contourf(X, Y, phi, levels=levels, cmap='RdBu_r')
    ax.contour(X, Y, phi, levels=[0], colors='black', linewidths=2.5)
    plt.colorbar(cf, ax=ax, label='Signed Distance')
    ax.set_title('Signed Distance Field', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')

    # 2. Binary mask
    ax = axes[0, 1]
    mask = phi < 0 
    ax.imshow(mask, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]], 
              origin='lower', cmap='gray', interpolation='nearest')
    ax.contour(X, Y, phi, levels=[0], colors='red', linewidths=2)
    ax.set_title('Binary Mask (Black = Inside)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')

    # 3. 3D surface plot
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    surf = ax.plot_surface(X, Y, phi, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.set_title('3D Distance Field', fontsize=14, fontweight='bold')
    plt.colorbar(surf, ax=ax, shrink=0.5)

    # 4. Contour lines
    ax = axes[1, 1]
    contour_levels = np.linspace(phi.min(), phi.max(), 20)
    cs = ax.contour(X, Y, phi, levels=contour_levels, cmap='coolwarm', linewidths=1)
    ax.contour(X, Y, phi, levels=[0], colors='black', linewidths=3)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    ax.set_title('Distance Contours', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dashboard_{name}.png", dpi=100)
    plt.close(fig)



test_cases = [
    {"name": "01_Circle",           "func": lambda p: sdf.sdCircle(p, 0.5)},
    {"name": "02_Box2D",            "func": lambda p: sdf.sdBox2D(p, np.array([0.4, 0.3]))},
    {"name": "03_RoundedBox2D",     "func": lambda p: sdf.sdRoundedBox2D(p, np.array([0.4, 0.3]), 0.1)},
    {"name": "04_OrientedBox2D",    "func": lambda p: sdf.sdOrientedBox2D(p, np.array([-0.4, -0.4]), np.array([0.4, 0.4]), 0.1)},
    {"name": "05_Segment2D",        "func": lambda p: sdf.sdSegment2D(p, np.array([-0.4, -0.3]), np.array([0.4, 0.3]))},
    
    # === B. POLYGONS ===
    {"name": "06_Rhombus2D",        "func": lambda p: sdf.sdRhombus2D(p, np.array([0.4, 0.2]))},
    {"name": "07_Trapezoid2D",      "func": lambda p: sdf.sdTrapezoid2D(p, 0.2, 0.5, 0.4)},
    {"name": "08_Parallelogram2D",  "func": lambda p: sdf.sdParallelogram2D(p, 0.5, 0.3, 0.2)},
    {"name": "09_EquilateralTri",   "func": lambda p: sdf.sdEquilateralTriangle2D(p, 0.5)},
    {"name": "10_IsoTriangle",      "func": lambda p: sdf.sdTriangleIsosceles2D(p, np.array([0.3, 0.5]))},
    {"name": "11_Triangle",         "func": lambda p: sdf.sdTriangle2D(p, np.array([0.0, 0.5]), np.array([0.5, -0.5]), np.array([-0.5, -0.5]))},
    {"name": "12_Pentagon2D",       "func": lambda p: sdf.sdPentagon2D(p, 0.5)},
    {"name": "13_Hexagon2D",        "func": lambda p: sdf.sdHexagon2D(p, 0.5)},
    {"name": "14_Octogon2D",        "func": lambda p: sdf.sdOctogon2D(p, 0.5)},
    {"name": "15_NGon2D_7",         "func": lambda p: sdf.sdNGon2D(p, 0.5, 7.0)},

    # === C. CURVED & COMPLEX SHAPES ===
    {"name": "16_UnevenCapsule",    "func": lambda p: sdf.sdUnevenCapsule2D(p, 0.2, 0.4, 0.6)},
    {"name": "17_Egg2D",            "func": lambda p: sdf.sdEgg2D(p, 0.4, 0.1)},
    {"name": "18_Pie2D",            "func": lambda p: sdf.sdPie2D(p, np.array([np.sin(0.5), np.cos(0.5)]), 0.5)},
    {"name": "19_Arc2D",            "func": lambda p: sdf.sdArc2D(p, np.array([np.sin(0.5), np.cos(0.5)]), 0.5, 0.05)},
    {"name": "20_Ring2D",           "func": lambda p: sdf.sdRing2D(p, 0.3, 0.5)},
    {"name": "21_Horseshoe2D",      "func": lambda p: sdf.sdHorseshoe2D(p, np.array([np.sin(2.0), np.cos(2.0)]), 0.5, np.array([0.1, 0.1]))},
    {"name": "22_Vesica2D",         "func": lambda p: sdf.sdVesica2D(p, 0.5, 0.2)},
    {"name": "23_Moon2D",           "func": lambda p: sdf.sdMoon2D(p, 0.2, 0.5, 0.4)},
    {"name": "24_CutDisk2D",        "func": lambda p: sdf.sdCutDisk2D(p, 0.5, -0.2)},
    
    {"name": "25_Star5",            "func": lambda p: sdf.sdStar5(p, 0.5, 0.2)},
    {"name": "26_Hexagram2D",       "func": lambda p: sdf.sdHexagram2D(p, 0.4)},
    {"name": "27_Star_Regular",     "func": lambda p: sdf.sdStar(p, 0.5, 6.0, 2.0)},
    {"name": "28_Heart2D",          "func": lambda p: sdf.sdHeart2D(p)},
    {"name": "29_Cross2D",          "func": lambda p: sdf.sdCross2D(p, np.array([0.5, 0.15]), 0.05)},
    {"name": "30_RoundedX2D",       "func": lambda p: sdf.sdRoundedX2D(p, 0.4, 0.1)},
    {"name": "31_RoundedCross2D",   "func": lambda p: sdf.sdRoundedCross2D(p, 0.3)},
    {"name": "32_BlobbyCross2D",    "func": lambda p: sdf.sdBlobbyCross2D(p, 0.4)},
    {"name": "33_CoolS2D",          "func": lambda p: sdf.sdCoolS2D(p)},
    
    {"name": "34_Ellipse2D",        "func": lambda p: sdf.sdEllipse2D(p, np.array([0.5, 0.3]))},
    {"name": "35_Parabola2D",       "func": lambda p: sdf.sdParabola2D(p, 0.5)},
    {"name": "36_Hyperbola2D",      "func": lambda p: sdf.sdHyperbola2D(p, 1.0, 0.5)},
    {"name": "37_Tunnel2D",         "func": lambda p: sdf.sdTunnel2D(p, np.array([0.4, 0.3]))},
    {"name": "38_Stairs2D",         "func": lambda p: sdf.sdStairs2D(p, np.array([0.2, 0.2]), 3.0)},
    {"name": "39_QuadraticCircle",  "func": lambda p: sdf.sdQuadraticCircle2D(p)},
    {"name": "40_Bezier2D",         "func": lambda p: sdf.sdBezier2D(p, np.array([-0.5, -0.5]), np.array([0.0, 0.5]), np.array([0.5, -0.5]))},
    {"name": "43_ParabolaSegment",  "func": lambda p: sdf.sdParabolaSegment2D(p, 0.6, 0.8),"bounds": ((-1.0, 1.0), (-0.5, 1.5))}, # Adjusting Y-bounds to see the peak
    {"name": "41_Polygon2D_Diamond","func": lambda p: sdf.sdPolygon2D(p, np.array([[0.0, 0.5],[0.5, 0.0],[0.0, -0.5],[-0.5, 0.0]]))},
    {"name": "42_Polygon2D_Star",   "func": lambda p: sdf.sdPolygon2D(p, np.array([[0.0, 0.6], [0.2, 0.2], [0.6, 0.2], [0.3, -0.1],[0.4, -0.6], [0.0, -0.3], [-0.4, -0.6], [-0.3, -0.1], [-0.6, 0.2], [-0.2, 0.2]]))}
]

# --- 4. EXECUTION LOOP ---
def main():
    print("🚀 Starting Generalized 2D Verification...")
    
    DEFAULT_BOUNDS = ((-1.5, 1.5), (-1.5, 1.5))
    
    for case in test_cases:
        bounds = case.get("bounds", DEFAULT_BOUNDS)
        render_sdf_dynamic(case["func"], case["name"], bounds=bounds)

    print("\n✅ Verification Complete. Check 'verification/outputs/2d/'")

if __name__ == "__main__":
    main()