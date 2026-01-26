"""
Complex Example: Combining Union, Intersection, Subtraction, and Elongation

This example demonstrates all major SDF operations in sequence:
1. Base shape (box)
2. Union with elongated sphere (capsule)
3. Intersection with a larger sphere (rounding)
4. Subtraction of a smaller box (creating a cavity)

Expected Final Shape:
- A rounded box-like structure with:
  - A cylindrical/capsule extension on one side (from union + elongation)
  - Rounded top and edges (from intersection with large sphere)
  - A rectangular cavity/hole (from subtraction)
- The shape should look like a complex mechanical part or architectural element

Mathematical Operations:
- Step 1: base = box(center=(0,0,0), half_size=(0.3,0.3,0.3))
- Step 2: capsule = elongated_sphere(radius=0.2, elongation=(0.3,0,0))
- Step 3: union_result = union(base, capsule)
- Step 4: rounder = sphere(center=(0,0.2,0), radius=0.6)  # Large sphere for rounding
- Step 5: rounded = intersect(union_result, rounder)
- Step 6: cutter = box(center=(0,-0.1,0), half_size=(0.15,0.15,0.15))
- Step 7: final = subtract(rounded, cutter)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import amrex.space3d as amr
from sdf3d import SDFLibrary
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


def save_3d_html(values, name, step_name, bounds, out_dir="outputs/vis3d_plotly"):
    """Generate interactive 3D HTML visualization using plotly"""
    if not HAS_VIZ:
        return
    
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = bounds
    spacing = (hi - lo) / values.shape[0]
    
    verts, faces, _, _ = measure.marching_cubes(
        values, level=0.0, spacing=(spacing, spacing, spacing)
    )
    verts += np.array([lo, lo, lo])
    
    # Filter out small disconnected fragments (artifacts from low resolution)
    if len(verts) > 0 and len(faces) > 0:
        vertex_face_count = np.zeros(len(verts))
        for face in faces:
            vertex_face_count[face] += 1
        
        # Keep vertices that appear in at least 2 faces (removes tiny fragments)
        min_faces = max(2, int(len(faces) / len(verts) * 0.1))  # Adaptive threshold
        valid_vertices = vertex_face_count >= min_faces
        
        if valid_vertices.sum() > len(verts) * 0.3:  # Only filter if we keep >30% of vertices
            # Remap vertex indices
            vertex_map = np.full(len(verts), -1, dtype=int)
            new_idx = 0
            for i, valid in enumerate(valid_vertices):
                if valid:
                    vertex_map[i] = new_idx
                    new_idx += 1
            
            # Filter faces: keep only faces where all vertices are valid
            valid_faces = valid_vertices[faces].all(axis=1)
            faces = faces[valid_faces]
            
            # Remap vertex indices in faces
            faces = np.array([[vertex_map[v] for v in face] for face in faces])
            
            # Filter vertices
            verts = verts[valid_vertices]
    
    i, j, k = faces.T
    
    # Color scheme for different steps
    colors = {
        'step1': 'lightblue',
        'step2': 'lightgreen',
        'step3': 'gold',
        'step4': 'coral',
        'step5': 'plum',
        'final': 'mediumpurple'
    }
    color = colors.get(step_name, 'lightgray')
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
            i=i, j=j, k=k, opacity=1.0, color=color, flatshading=True
        )
    ])
    
    fig.update_layout(
        title=f"{name} - {step_name} (SDF=0 isosurface)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="cube"),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    out_path = os.path.join(out_dir, f"{name}_{step_name}_3d.html")
    fig.write_html(out_path)
    print(f"  ✅ Step visualization: {out_path}")


def print_step_info(step_num, step_name, mf, description):
    """Print information about a step"""
    all_vals = []
    for mfi in mf:
        arr = mf.array(mfi).to_numpy()
        vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
        all_vals.append(vals.flatten())
    phi = np.concatenate(all_vals)
    
    print(f"\n  Step {step_num}: {step_name}")
    print(f"    {description}")
    print(f"    Min value: {phi.min():.6f}, Max value: {phi.max():.6f}")
    print(f"    Has inside (negative): {(phi < 0).any()}, Has outside (positive): {(phi > 0).any()}")


def main():
    amr.initialize([])
    try:
        print("=" * 70)
        print("COMPLEX EXAMPLE: Union + Elongation + Intersection + Subtraction")
        print("=" * 70)
        
        # Setup grid (higher resolution for better visualization)
        real_box = amr.RealBox([-1, -1, -1], [1, 1, 1])
        domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(127, 127, 127))  # 128^3
        geom = amr.Geometry(domain, real_box, 0, [0, 0, 0])
        ba = amr.BoxArray(domain)
        ba.max_size(32)
        dm = amr.DistributionMapping(ba)

        lib = SDFLibrary(geom, ba, dm)
        n = 128
        bounds = (-1, 1)

        # ============================================================
        # STEP 1: Base shape - A box
        # ============================================================
        print("\n" + "-" * 70)
        base = lib.box(center=(0.0, 0.0, 0.0), half_size=(0.3, 0.3, 0.3))
        print_step_info(1, "Base Box", base, "Starting with a cube at origin")
        if HAS_VIZ:
            full_array = gather_multifab_to_array(base, (n, n, n))
            save_3d_html(full_array, "complex_example", "step1", bounds)
        
        # ============================================================
        # STEP 2: Create elongated sphere (capsule) using MultiFab
        # ============================================================
        print("\n" + "-" * 70)
        # Create a capsule by union of two spheres (caps) and a box (cylinder)
        # This approximates an elongated sphere
        capsule_s1 = lib.sphere(center=(-0.3, 0.0, 0.0), radius=0.2)
        capsule_s2 = lib.sphere(center=(0.3, 0.0, 0.0), radius=0.2)
        capsule_box = lib.box(center=(0.0, 0.0, 0.0), half_size=(0.3, 0.2, 0.2))
        capsule_temp = lib.union(capsule_s1, capsule_s2)
        capsule = lib.union(capsule_temp, capsule_box)
        
        print_step_info(2, "Elongated Sphere (Capsule)", capsule,
                       "Capsule: union of 2 spheres + box (approximates elongated sphere)")
        if HAS_VIZ:
            full_array = gather_multifab_to_array(capsule, (n, n, n))
            save_3d_html(full_array, "complex_example", "step2", bounds)
        
        # ============================================================
        # STEP 3: Union - Combine base box with capsule
        # ============================================================
        print("\n" + "-" * 70)
        union_result = lib.union(base, capsule)
        print_step_info(3, "Union", union_result, 
                       "Union of base box and capsule (box ∪ capsule)")
        if HAS_VIZ:
            full_array = gather_multifab_to_array(union_result, (n, n, n))
            save_3d_html(full_array, "complex_example", "step3", bounds)
        
        # ============================================================
        # STEP 4: Intersection - Round the top with a large sphere
        # ============================================================
        print("\n" + "-" * 70)
        rounder = lib.sphere(center=(0.0, 0.2, 0.0), radius=0.6)
        rounded = lib.intersect(union_result, rounder)
        print_step_info(4, "Intersection (Rounding)", rounded,
                       "Intersect with large sphere(center=(0,0.2,0), r=0.6) to round top")
        if HAS_VIZ:
            full_array = gather_multifab_to_array(rounded, (n, n, n))
            save_3d_html(full_array, "complex_example", "step4", bounds)
        
        # ============================================================
        # STEP 5: Subtraction - Create a cavity/hole
        # ============================================================
        print("\n" + "-" * 70)
        cutter = lib.box(center=(0.0, -0.1, 0.0), half_size=(0.15, 0.15, 0.15))
        final = lib.subtract(rounded, cutter)
        print_step_info(5, "Subtraction (Cavity)", final,
                       "Subtract small box to create internal cavity")
        if HAS_VIZ:
            full_array = gather_multifab_to_array(final, (n, n, n))
            save_3d_html(full_array, "complex_example", "step5", bounds)
        
        # ============================================================
        # FINAL RESULT
        # ============================================================
        print("\n" + "=" * 70)
        print("FINAL RESULT: Complex Shape with All Operations")
        print("=" * 70)
        
        all_vals = []
        for mfi in final:
            arr = final.array(mfi).to_numpy()
            vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
            all_vals.append(vals.flatten())
        phi = np.concatenate(all_vals)
        
        print(f"Final shape statistics:")
        print(f"  Min value (deepest inside): {phi.min():.6f}")
        print(f"  Max value (furthest outside): {phi.max():.6f}")
        print(f"  Has inside regions: {(phi < 0).any()}")
        print(f"  Has outside regions: {(phi > 0).any()}")
        print(f"  Has surface (near zero): {(np.abs(phi) < 0.05).any()}")
        
        if HAS_VIZ:
            full_array = gather_multifab_to_array(final, (n, n, n))
            save_3d_html(full_array, "complex_example", "final", bounds)
        
        print("\n" + "=" * 70)
        print("✅ COMPLEX EXAMPLE COMPLETE")
        print("=" * 70)
        print("\nExpected Final Shape Description:")
        print("  - A rounded box-like structure (from base box)")
        print("  - With a cylindrical/capsule extension on the x-axis (from union)")
        print("  - Rounded top and upper edges (from intersection with large sphere)")
        print("  - A rectangular cavity/hole in the lower center (from subtraction)")
        print("  - Overall shape resembles a complex mechanical part or architectural element")
        print("\nVisualization files saved to: outputs/vis3d_plotly/complex_example_*.html")
        print("  - step1: Base box")
        print("  - step2: Elongated sphere (capsule)")
        print("  - step3: Union result (box + capsule)")
        print("  - step4: Intersection result (rounded)")
        print("  - step5: Subtraction result (with cavity)")
        print("  - final: Final complex shape")

    finally:
        amr.finalize()


if __name__ == "__main__":
    main()
