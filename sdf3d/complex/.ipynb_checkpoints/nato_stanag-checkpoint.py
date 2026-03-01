"""
NATO STANAG-4496 Fragment - FULLY DYNAMIC
nato_mf, nato_geom = NATOFragment(lib, diameter=14.30e-3, L_over_D=1.09)
"""

from sdf3d import Cylinder, Box, Intersection, Union, Geometry
import sdf_lib as sdf
import numpy as np

def NATOFragment(lib, diameter=14.30e-3, L_over_D=1.09, cone_angle_deg=20.0):
    """
    NATO STANAG-4496 Fragment - FULLY PARAMETRIC
    
    Args:
        diameter: Fragment diameter (m) → 14.30e-3 = 14.3mm
        L_over_D: Length/Diameter ratio → 1.09 (15.56/14.3)
        cone_angle_deg: Cone half-angle → 20°
    """
    print("\n📐 NATO STANAG-4496 FRAGMENT (DYNAMIC)...")
    
    # DYNAMIC DIMENSIONS
    fragment_radius = diameter / 2.0
    total_length = diameter * L_over_D           # 🔥 DYNAMIC LENGTH
    cylinder_height = diameter                   # 🔥 DYNAMIC CYLINDER (was hardcoded 14.3e-3)
    cone_height = total_length - cylinder_height # 🔥 DYNAMIC CONE
    
    print(f"  🔧 diameter={diameter*1000:5.1f}mm, L/D={L_over_D:.2f}")
    print(f"  📦 Cylinder: r={fragment_radius*1000:5.2f}mm × h={cylinder_height*1000:5.2f}mm")
    print(f"  🔺 Cone:    h={cone_height*1000:5.2f}mm")
    
    # CYLINDER - DYNAMIC
    cyl_inf = Cylinder(axis_offset=[0.0, 0.0], radius=fragment_radius)
    cyl_box = Box(half_size=[fragment_radius*1.2, cylinder_height/2, fragment_radius*1.2])
    cyl_geom = (Intersection(cyl_inf, cyl_box)
               .rotate_x(np.pi/2)
               .translate(0.0, 0.0, cylinder_height/2))
    
    # CONE - DYNAMIC
    def sharp_cone_sdf(p):
        return sdf.sdCappedCone(p, cone_height, 0.0, fragment_radius)
    
    cone_geom = (Geometry(sharp_cone_sdf)
                .rotate_x(np.pi/2)
                .translate(0.0, 0.0, cylinder_height + cone_height))
    
    # UNION
    fragment_geom = Union(cyl_geom, cone_geom)
    fragment_mf = lib.from_geometry(fragment_geom)
    
    print(f"✅ NATO: {diameter*1000:.1f}mm × {total_length*1000:.1f}mm (L/D={L_over_D:.2f})")
    return fragment_mf, fragment_geom
