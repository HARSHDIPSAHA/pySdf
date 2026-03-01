"""
🚀 DYNAMIC ROCKET ASSEMBLY - FULLY PARAMETRIC
rocket_mf, rocket_geom = RocketAssembly(lib, body_radius=0.15, L_extra=0.40, nose_len=0.25)
"""

from sdf3d import Sphere, Box, Union, Geometry
import sdf_lib as sdf
import numpy as np

def RocketAssembly(lib, body_radius=0.15, L_extra=0.40, nose_len=0.25, 
                   fin_span=0.12, fin_height=0.18, fin_thickness=0.03, n_fins=4):
    """
    🚀 Fully Dynamic Rocket Assembly
    
    Args:
        body_radius: Body radius (m)
        L_extra: Body elongation length (m)  
        nose_len: Nose cone length (m)
        fin_span: Fin span (m)
        fin_height: Fin height (m)
        fin_thickness: Fin thickness (m)
        n_fins: Number of fins (default 4)
    """
    print("\n🚀 DYNAMIC ROCKET ASSEMBLY...")
    print(f"  🔧 R={body_radius:.0f}cm, L_extra={L_extra:.0f}cm, Nose={nose_len:.0f}cm")
    
    # -----------------------------------------------------
    # BODY (Capsule) - DYNAMIC
    # -----------------------------------------------------
    R = body_radius
    body_geom = Sphere(R).elongate(0.0, 0.0, L_extra)
    
    # -----------------------------------------------------
    # NOSE CONE - DYNAMIC
    # -----------------------------------------------------
    z_body_top = (L_extra / 2.0) + R
    z_cone_center = z_body_top + nose_len / 2.0
    h_cone = nose_len / 2.0

    def nose_sdf(p):
        qx = p[..., 0]
        qy = p[..., 1] 
        qz = p[..., 2] - z_cone_center
        q = np.stack([qx, qz, qy], axis=-1)
        return sdf.sdCappedCone(q, h_cone, 0.0, R)
    
    nose_geom = Geometry(nose_sdf)
    
    # -----------------------------------------------------
    # FINS - DYNAMIC
    # -----------------------------------------------------
    fin_half = [fin_span / 2.0, fin_thickness / 2.0, fin_height / 2.0]
    z_fin_center = -0.18  # Fixed relative to body
    
    fins_geom = None
    for i in range(n_fins):
        angle = i * (2 * np.pi / n_fins)  # Dynamic fin spacing
        radial_dist = R + fin_half[0]
        dx = radial_dist * np.cos(angle)
        dy = radial_dist * np.sin(angle)
        
        single_fin = (Box(half_size=fin_half)
                     .rotate_z(angle)
                     .translate(dx, dy, z_fin_center))
        
        if fins_geom is None:
            fins_geom = single_fin
        else:
            fins_geom = Union(fins_geom, single_fin)
    
    # -----------------------------------------------------
    # FULL ASSEMBLY - DYNAMIC
    # -----------------------------------------------------
    rocket = Union(body_geom, nose_geom)
    rocket = Union(rocket, fins_geom)
    
    rocket_mf = lib.from_geometry(rocket)
    
    total_length = L_extra + 2*R + nose_len
    print(f"✅ Rocket: {total_length*100:.0f}cm long, {n_fins} fins")
    return rocket_mf, rocket

# Legacy compatibility
Rocket = RocketAssembly
