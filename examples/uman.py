"""
Phase 2: uSCMAN 2D Integration Test for pySdf (UPDATED)
-------------------------------------------------------
- Handles 1D flattened arrays from uSCMAN HDF5 files.
- Safely reads scalar dimensional data (I, J).
- Tests both Union and Subtraction operations.
- Exports to both PNG and Interactive Plotly HTML.
"""

import sys
import os
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
except ImportError:
    print("⚠️ Plotly not found. Run 'pip install plotly' for HTML outputs.")

# Connect to your pySdf library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sdf2d import Geometry2D, Circle2D, sample_levelset_2d

class ImageLeaf2D(Geometry2D):
    def __init__(self, hdf5_path: str, dataset_path: str, physical_size_xy: tuple) -> None:
        print(f"📥 Reading 2D SDF from {hdf5_path}...")
        
        with h5py.File(hdf5_path, 'r') as f:
            if dataset_path not in f:
                raise KeyError(f"Dataset '{dataset_path}' missing.")
            
            # 1. Grab the raw data
            raw_sdf = f[dataset_path][:]
            
            # 2. THE FIX: Re-inflate 1D arrays back to 2D!
            if len(raw_sdf.shape) == 1:
                print("⚠️ Detected 1D flattened array. Rebuilding 2D grid...")
                group_path = os.path.dirname(dataset_path) # e.g. MULTIPHASE_TEST/Segmentation/Image1
                
                # Safely read I and J (Handling the "scalar dataspace" issue)
                I_dataset = f[f"{group_path}/I"]
                J_dataset = f[f"{group_path}/J"]
                
                if I_dataset.ndim == 0:
                    # It's a single number (scalar)
                    res_x = int(I_dataset[()])
                    res_y = int(J_dataset[()])
                else:
                    # It's an array of coordinates
                    res_x = len(np.unique(I_dataset[:]))
                    res_y = len(np.unique(J_dataset[:]))
                
                # Reshape it! (Assuming standard Y, X ordering)
                self.sdf_2d = raw_sdf.reshape((res_y, res_x))
            else:
                self.sdf_2d = raw_sdf
            
        # 3. Map pixels to physical coordinates
        res_y, res_x = self.sdf_2d.shape
        width, height = physical_size_xy
        
        x_coords = np.linspace(-width/2, width/2, res_x)
        y_coords = np.linspace(-height/2, height/2, res_y)
        
        # 4. Create a continuous math function
        self.interpolator = RegularGridInterpolator(
            (x_coords, y_coords), 
            self.sdf_2d.T, 
            bounds_error=False, 
            fill_value=1.0 
        )

        def _sdf(p: np.ndarray) -> np.ndarray:
            return self.interpolator(p)
            
        super().__init__(_sdf)

# --- VISUALIZATION HELPER ---
def save_visuals(phi, bounds, name, title):
    os.makedirs("outputs", exist_ok=True)
    x0, x1 = bounds[0]; y0, y1 = bounds[1]
    
    # 1. SAVE PNG (Matplotlib)
    plt.figure(figsize=(6, 5))
    plt.title(title, fontweight='bold')
    plt.imshow(phi, extent=[x0, x1, y0, y1], origin='lower', cmap='RdBu_r')
    plt.colorbar(label='Signed Distance (φ < 0 is Inside)')
    plt.contour(phi, levels=[0.0], extent=[x0, x1, y0, y1], colors='black', linewidths=2.5)
    plt.savefig(f"outputs/{name}.png", dpi=200, bbox_inches='tight')
    plt.close()

    # 2. SAVE HTML (Plotly)
    if 'plotly' in sys.modules:
        fig = go.Figure(data=go.Contour(
            z=phi,
            x=np.linspace(x0, x1, phi.shape[1]),
            y=np.linspace(y0, y1, phi.shape[0]),
            colorscale='RdBu',
            reversescale=True,
            contours=dict(
                start=0, end=0, size=1, # Only draw the boundary line!
                showlines=True, coloring='heatmap'
            ),
            line=dict(color='black', width=4)
        ))
        fig.update_layout(title=title, xaxis_title="X (m)", yaxis_title="Y (m)", width=700, height=600)
        fig.write_html(f"outputs/{name}.html")
        print(f"✅ Saved outputs/{name}.png AND .html")
    else:
        print(f"✅ Saved outputs/{name}.png")


def main():
    print("="*60)
    print("🚀 2D INTEGRATION PROOF FOR PROFESSOR")
    print("="*60)

    # 1. PATH SETUP
    current_dir = os.path.dirname(__file__) 
    h5_file = os.path.abspath(os.path.join(current_dir, '../../uSCMAN/RESULTS/RESULTS.h5'))
    data_path = 'MULTIPHASE_TEST/Segmentation/Image1/Phi1'

    # 2. LOAD IMAGE GEOMETRY
    microstructure = ImageLeaf2D(h5_file, data_path, physical_size_xy=(0.10, 0.10))

    # 3. CREATE ANALYTICAL GEOMETRY
    # 2cm circle slightly offset
    math_circle = Circle2D(radius=0.02).translate(0.015, 0.015)

    # 4. BOOLEAN OPERATIONS!
    print("💥 Performing Boolean UNION...")
    scene_union = microstructure.union(math_circle)
    
    print("💥 Performing Boolean SUBTRACTION (Cutting circle out of image)...")
    scene_diff = microstructure.subtract(math_circle)

    # 5. SAMPLE AND VISUALIZE
    print("🎨 Sampling and rendering to PNG and HTML...")
    bounds = ((-0.05, 0.05), (-0.05, 0.05))
    
    # We use a 300x300 sampling grid. You can increase this for higher resolution!
    phi_union = sample_levelset_2d(scene_union, bounds=bounds, resolution=(300, 300))
    save_visuals(phi_union, bounds, "prof_proof_UNION", "2D Union: Image + Circle")
    
    phi_diff = sample_levelset_2d(scene_diff, bounds=bounds, resolution=(300, 300))
    save_visuals(phi_diff, bounds, "prof_proof_SUBTRACTION", "2D Subtraction: Image - Circle")

if __name__ == "__main__":
    main()
