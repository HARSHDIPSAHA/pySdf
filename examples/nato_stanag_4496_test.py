"""
NATO STANAG-4496 Fragment Impact Test Geometry

This test case implements the standard fragment impact test geometry as specified
in NATO STANAG-4496, demonstrating the library's capability to handle real-world
experimental test geometries.

Specifications:
- Standard Fragment:
  * Diameter: 14.30 mm (radius: 7.15 mm)
  * Overall Length: 15.56 mm
  * Cone Angle: 20° (half-angle)
  * Shape: Conical-ended cylinder (L/D > 1 for stability)
  * Mass: 18.6 g
  * Material: Mild carbon steel (HB 190-270)

- Target Material:
  * Solid block/sample for impact testing

- Impact Configuration:
  * Angular deviation tolerance: ±10°
  * Impact velocities: 2530 ± 90 m/s (Method 1) or 1830 ± 60 m/s (Method 2)

This geometry is used for:
- High-velocity impact simulations
- Fragment penetration studies
- Munition response testing
- Shock physics simulations
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import amrex.space3d as amr
from sdf3d import SDFLibrary, Cylinder, ConeExact, Box, Intersection, Union
import numpy as np

try:
    from skimage import measure
    import plotly.graph_objects as go
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("⚠️  plotly/scikit-image not available, skipping 3D visualization")


def gather_multifab_to_array(mf, shape):
    """Convert MultiFab to full numpy array"""
    full = np.zeros(shape, dtype=np.float32)
    for mfi in mf:
        arr = mf.array(mfi).to_numpy()
        vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
        bx = mfi.validbox()
        i_lo, j_lo, k_lo = bx.lo_vect
        i_hi, j_hi, k_hi = bx.hi_vect
        full[k_lo:k_hi+1, j_lo:j_hi+1, i_lo:i_hi+1] = vals
    return full


def save_3d_html(values, name, bounds, out_dir="outputs/vis3d_plotly"):
    """Generate interactive 3D HTML visualization using plotly"""
    if not HAS_VIZ:
        return
    
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = bounds
    spacing = (hi - lo) / values.shape[0]
    
    # Check if level=0 is within the data range
    if values.min() >= 0 or values.max() <= 0:
        print(f"  ⚠️  {name}: Cannot extract isosurface - no zero crossing")
        print(f"      Data range: [{values.min():.6f}, {values.max():.6f}]")
        return
    
    try:
        verts, faces, _, _ = measure.marching_cubes(
            values, level=0.0, spacing=(spacing, spacing, spacing)
        )
    except ValueError as e:
        print(f"  ⚠️  {name}: Error extracting isosurface - {e}")
        print(f"      Data range: [{values.min():.6f}, {values.max():.6f}]")
        return
    
    verts += np.array([lo, lo, lo])
    
    # Filter out small disconnected fragments
    if len(verts) > 0 and len(faces) > 0:
        vertex_face_count = np.zeros(len(verts))
        for face in faces:
            vertex_face_count[face] += 1
        
        min_faces = max(2, int(len(faces) / len(verts) * 0.1))
        valid_vertices = vertex_face_count >= min_faces
        
        if valid_vertices.sum() > len(verts) * 0.3:
            vertex_map = np.full(len(verts), -1, dtype=int)
            new_idx = 0
            for i, valid in enumerate(valid_vertices):
                if valid:
                    vertex_map[i] = new_idx
                    new_idx += 1
            
            valid_faces = valid_vertices[faces].all(axis=1)
            faces = faces[valid_faces]
            faces = np.array([[vertex_map[v] for v in face] for face in faces])
            verts = verts[valid_vertices]
    
    i, j, k = faces.T
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
            i=i, j=j, k=k, opacity=1.0, color='steelblue', flatshading=True
        )
    ])
    
    fig.update_layout(
        title=f"{name} (SDF=0 isosurface)",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    out_path = os.path.join(out_dir, f"{name}_3d.html")
    fig.write_html(out_path)
    print(f"  ✅ Visualization: {out_path}")


def create_fragment_geometry(lib):
    """
    Step 1: Build the NATO standard fragment (conical-ended cylinder)
    
    Specifications:
    - Diameter: 14.30 mm → radius: 7.15 mm = 7.15e-3 m
    - Overall length: 15.56 mm = 15.56e-3 m
    - Cone angle: 20° (half-angle)
    - Cylinder height: ~10 mm
    - Cone height: ~5.56 mm
    """
    print("\n" + "=" * 70)
    print("STEP 1: Building NATO Standard Fragment")
    print("=" * 70)
    
    # Fragment dimensions (in meters)
    fragment_radius = 7.15e-3  # 7.15 mm radius
    cylinder_height = 10e-3     # 10 mm cylinder
    cone_height = 5.56e-3       # 5.56 mm cone (total 15.56 mm)
    # Cone angle: 20° half-angle (or 10° if "10/20" means full angle)
    # Using 20° half-angle as per NATO spec
    cone_angle_deg = 20.0       # 20° cone half-angle
    
    # Convert cone angle to sin/cos for sdConeExact
    # sdConeExact uses: c = [sin(angle), cos(angle)]
    cone_angle_rad = np.deg2rad(cone_angle_deg)
    cone_sincos = [np.sin(cone_angle_rad), np.cos(cone_angle_rad)]
    
    print(f"  Fragment specifications:")
    print(f"    Diameter: {2*fragment_radius*1000:.2f} mm")
    print(f"    Radius: {fragment_radius*1000:.2f} mm")
    print(f"    Cylinder height: {cylinder_height*1000:.2f} mm")
    print(f"    Cone height: {cone_height*1000:.2f} mm")
    print(f"    Total length: {(cylinder_height + cone_height)*1000:.2f} mm")
    print(f"    Cone angle: {cone_angle_deg}°")
    
    # Create cylinder (body of fragment)
    # sdCylinder is aligned with y-axis (infinite along y)
    # We need to make it finite by intersecting with a box
    cylinder_infinite = Cylinder(axis_offset=[0.0, 0.0], radius=fragment_radius)
    # Create a box to limit cylinder height along y-axis
    # Box extends in x, y, z directions - y is the cylinder axis
    cylinder_box = Box(half_size=[fragment_radius*2, cylinder_height/2, fragment_radius*2])
    cylinder_geom = Intersection(cylinder_infinite, cylinder_box)
    # Rotate to align with z-axis (rotate 90° around x-axis: y -> z)
    cylinder_geom = cylinder_geom.rotate_x(np.pi/2)
    # Translate to position cylinder: base at z=0, top at z=cylinder_height
    # After rotation, cylinder extends from z=-cylinder_height/2 to z=+cylinder_height/2
    # Translate to position: base at z=0, top at z=cylinder_height
    cylinder_geom = cylinder_geom.translate(0.0, 0.0, cylinder_height/2)
    
    # Create smooth cone (tip of fragment) with 20° half-angle
    # Use sdCappedCone to create a truncated cone (frustum) that smoothly connects
    # sdCappedCone(p, h, r1, r2): h=height, r1=tip_radius, r2=base_radius
    # For smooth connection: base_radius (r2) = fragment_radius
    # Tip radius (r1) = base_radius - height * tan(angle)
    import sdf_lib as sdf
    
    # Calculate tip radius for 20° half-angle cone
    # Note: sdCappedCone uses h as half-height, so total height is 2*h
    # We want total height = cone_height, so use h = cone_height/2
    cone_half_height = cone_height / 2.0
    tip_radius = fragment_radius - cone_height * np.tan(cone_angle_rad)
    tip_radius = max(0.0, tip_radius)  # Ensure non-negative
    
    # Create capped cone (truncated cone) using sdCappedCone
    # sdCappedCone has base at y=+h, tip at y=-h
    # We want base at y=-h, tip at y=+h, so flip y-coordinate in SDF
    def capped_cone_sdf(p):
        # Flip y-coordinate to reverse orientation: base at bottom, tip at top
        p_flipped = np.stack([p[..., 0], -p[..., 1], p[..., 2]], axis=-1)
        # sdCappedCone: p, h (half-height), r1 (tip), r2 (base)
        return sdf.sdCappedCone(p_flipped, cone_half_height, tip_radius, fragment_radius)
    
    from sdf3d.geometry import Geometry
    cone_geom = Geometry(capped_cone_sdf)
    
    # Rotate to align with z-axis (rotate 90° around x-axis: y -> z)
    # After flipping y in SDF: base at y=-h, tip at y=+h (where h=cone_height/2)
    # After rotation: base at z=-h, tip at z=+h
    cone_geom = cone_geom.rotate_x(np.pi/2)
    
    # Translate to position cone base exactly at cylinder top
    # Base is at z=-h = -cone_height/2 after rotation
    # We want base at z=cylinder_height, so translate by z=cylinder_height + h
    # This puts base at cylinder_height, tip at cylinder_height + cone_height
    cone_geom = cone_geom.translate(0.0, 0.0, cylinder_height + cone_half_height)
    
    print(f"    Cone: smooth truncated cone (frustum)")
    print(f"    Cone base radius: {fragment_radius*1000:.2f} mm (matches cylinder)")
    print(f"    Cone tip radius: {tip_radius*1000:.2f} mm")
    print(f"    Cone height: {cone_height*1000:.2f} mm")
    print(f"    Cone angle: {cone_angle_deg}° half-angle")
    
    # Union cylinder and cone
    fragment_geom = Union(cylinder_geom, cone_geom)
    
    # Convert geometry to MultiFab for visualization and further operations
    fragment_mf = lib.from_geometry(fragment_geom)
    
    print(f"  ✅ Fragment created: Cylinder + Cone union")
    
    return fragment_mf, fragment_geom


def create_target_geometry(lib):
    """
    Step 2: Build the target material block
    """
    print("\n" + "=" * 70)
    print("STEP 2: Building Target Material Block")
    print("=" * 70)
    
    # Target dimensions (50mm cube)
    target_size = 0.05  # 50 mm = 0.05 m
    target_center = [0.0, 0.0, 0.05]  # Positioned at z=0.05 m
    
    target = lib.box(
        center=target_center,
        half_size=(target_size/2, target_size/2, target_size/2)
    )
    
    print(f"  Target specifications:")
    print(f"    Size: {target_size*1000:.1f} mm cube")
    print(f"    Center: ({target_center[0]*1000:.1f}, {target_center[1]*1000:.1f}, {target_center[2]*1000:.1f}) mm")
    print(f"  ✅ Target block created")
    
    return target


def position_fragment_for_impact(fragment_geom, impact_angle_deg=0.0, distance_from_target=0.1):
    """
    Step 3: Position fragment in front of target with impact orientation
    
    Args:
        impact_angle_deg: Angular deviation from normal (yaw/pitch), ±10° tolerance
        distance_from_target: Distance from fragment to target (m)
    """
    print("\n" + "=" * 70)
    print("STEP 3: Positioning Fragment for Impact")
    print("=" * 70)
    
    # Rotate fragment around y-axis (pitch) for impact angle
    impact_angle_rad = np.deg2rad(impact_angle_deg)
    fragment_rotated = fragment_geom.rotate_y(impact_angle_rad)
    
    # Translate fragment to position in front of target
    # Fragment starts at origin, move it to negative z (in front of target at z=0.05)
    fragment_positioned = fragment_rotated.translate(0.0, 0.0, -distance_from_target)
    
    print(f"  Impact configuration:")
    print(f"    Impact angle: {impact_angle_deg:.1f}° (within ±10° tolerance)")
    print(f"    Distance from target: {distance_from_target*1000:.1f} mm")
    print(f"  ✅ Fragment positioned for impact")
    
    return fragment_positioned


def create_full_domain(lib, fragment_geom, target_mf):
    """
    Step 4: Create full domain geometry (fragment + target)
    
    The solver needs a single SDF representing all solid material.
    Everything else is automatically void.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Creating Full Domain Geometry")
    print("=" * 70)
    
    # Convert positioned fragment geometry to MultiFab
    fragment_mf = lib.from_geometry(fragment_geom)
    
    # Union fragment and target
    solid = lib.union(fragment_mf, target_mf)
    
    print(f"  Domain composition:")
    print(f"    Fragment (projectile)")
    print(f"    Target (material block)")
    print(f"    Void (everything else)")
    print(f"  ✅ Full domain geometry created")
    
    return solid, fragment_mf


def main():
    amr.initialize([])
    try:
        print("=" * 70)
        print("NATO STANAG-4496 Fragment Impact Test Geometry")
        print("=" * 70)
        print("\nThis test case implements the standard fragment impact test")
        print("geometry as specified in NATO STANAG-4496.")
        print("\nOutput: Level set field φ(x,y,z) for solver input")
        
        # Setup grid (domain: -0.1 to 0.1 m in all directions)
        domain_size = 0.2  # 200 mm domain
        real_box = amr.RealBox([-domain_size, -domain_size, -domain_size], 
                              [domain_size, domain_size, domain_size])
        # Higher resolution for accurate geometry
        domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(255, 255, 255))  # 256^3
        geom = amr.Geometry(domain, real_box, 0, [0, 0, 0])
        ba = amr.BoxArray(domain)
        ba.max_size(32)
        dm = amr.DistributionMapping(ba)
        
        lib = SDFLibrary(geom, ba, dm)
        n = 256
        bounds = (-domain_size, domain_size)
        
        # Step 1: Build fragment
        fragment_mf, fragment_geom = create_fragment_geometry(lib)
        
        # Step 2: Build target
        target_mf = create_target_geometry(lib)
        
        # Step 3: Position fragment for impact (with 5° impact angle as example)
        impact_angle = 5.0  # Within ±10° tolerance
        fragment_positioned = position_fragment_for_impact(
            fragment_geom, 
            impact_angle_deg=impact_angle,
            distance_from_target=0.05  # 50 mm from target
        )
        
        # Step 4: Create full domain
        solid_mf, fragment_positioned_mf = create_full_domain(lib, fragment_positioned, target_mf)
        
        # Gather statistics
        all_vals = []
        for mfi in solid_mf:
            arr = solid_mf.array(mfi).to_numpy()
            vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
            all_vals.append(vals.flatten())
        phi = np.concatenate(all_vals)
        
        print("\n" + "=" * 70)
        print("FINAL RESULT: Level Set Field φ(x,y,z)")
        print("=" * 70)
        print(f"  Min value (deepest inside): {phi.min():.6e} m")
        print(f"  Max value (furthest outside): {phi.max():.6e} m")
        print(f"  Has inside regions (negative): {(phi < 0).any()}")
        print(f"  Has outside regions (positive): {(phi > 0).any()}")
        print(f"  Has surface (near zero): {(np.abs(phi) < 0.001).any()}")
        print(f"\n  This level set field is ready for solver input:")
        print(f"    - Fragment geometry encoded")
        print(f"    - Target geometry encoded")
        print(f"    - Impact configuration encoded")
        print(f"    - All in a single φ(x,y,z) field")
        
        # Generate visualizations
        if HAS_VIZ:
            print("\n" + "=" * 70)
            print("Generating 3D Visualizations")
            print("=" * 70)
            
            # Visualize fragment alone
            fragment_array = gather_multifab_to_array(fragment_mf, (n, n, n))
            save_3d_html(fragment_array, "nato_fragment", bounds)
            
            # Visualize target alone
            target_array = gather_multifab_to_array(target_mf, (n, n, n))
            save_3d_html(target_array, "nato_target", bounds)
            
            # Visualize positioned fragment
            fragment_pos_array = gather_multifab_to_array(fragment_positioned_mf, (n, n, n))
            save_3d_html(fragment_pos_array, "nato_fragment_positioned", bounds)
            
            # Visualize full domain (fragment + target)
            solid_array = gather_multifab_to_array(solid_mf, (n, n, n))
            save_3d_html(solid_array, "nato_full_domain", bounds)
            
            print("\n  Visualizations saved to: outputs/vis3d_plotly/")
            print("    - nato_fragment_3d.html: Fragment geometry")
            print("    - nato_target_3d.html: Target block")
            print("    - nato_fragment_positioned_3d.html: Fragment with impact orientation")
            print("    - nato_full_domain_3d.html: Complete test geometry")
        
        # Save to plotfile for solver input
        print("\n" + "=" * 70)
        print("Saving to AMReX Plotfile")
        print("=" * 70)
        try:
            plotfile_dir = "plotfiles"
            os.makedirs(plotfile_dir, exist_ok=True)
            plotfile_name = os.path.join(plotfile_dir, "nato_stanag_4496_test")
            
            varnames = amr.Vector_string(["phi"])
            amr.write_single_level_plotfile(
                plotfile_name,
                solid_mf,
                varnames,
                geom,
                0.0,  # time
                0     # level
            )
            print(f"  ✅ Plotfile saved: {plotfile_name}")
            print(f"     Ready for solver input (hydrodynamics, solid mechanics, etc.)")
        except Exception as e:
            print(f"  ⚠️  Could not save plotfile: {e}")
        
        print("\n" + "=" * 70)
        print("✅ NATO STANAG-4496 Test Case Complete")
        print("=" * 70)
        print("\nThis geometry can now be used for:")
        print("  - High-velocity impact simulations")
        print("  - Fragment penetration studies")
        print("  - Munition response testing")
        print("  - Shock physics simulations")
        print("\nNo STL, no CAD, no meshing - pure level set geometry!")
        
    finally:
        amr.finalize()


if __name__ == "__main__":
    main()
