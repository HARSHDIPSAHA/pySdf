import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

# Import your base class!
from sdf3d import Geometry3D

class ImageExtruded3D(Geometry3D):
    """
    Reads a 2D Chan-Vese SDF from uSCMAN and extrudes it into a 3D geometry node.
    Inherits from Geometry3D, so it automatically gets .translate(), .union(), 
    .save_plotly_html(), etc.
    """
    def __init__(self,hdf5_path: str,dataset_path: str,physical_size_xy: tuple,thickness_z: float) -> None:
        self.thickness_z = thickness_z

        # 1. Load and reshape the 2D SDF array from uSCMAN results
        with h5py.File(hdf5_path, 'r') as f:
            if dataset_path not in f:
                raise KeyError(f"Could not find {dataset_path} in {hdf5_path}")

            # Get grid dimensions stored by uSCMAN(1D SDF array into 2D sdf)
            seg = f[dataset_path]
            I = int(seg['I'][()]); J = int(seg['J'][()])
            self.sdf_2d = -seg['Phi1'][()].reshape(J, I).astype(np.float64)

        # Physical coordinate grid centred at (0, 0)
        width, height = physical_size_xy
        x_coords = np.linspace(-width/2,  width/2,  I)
        y_coords = np.linspace(-height/2, height/2, J)

        self.interpolator = RegularGridInterpolator(
            (x_coords, y_coords),  
            self.sdf_2d.T,             # Transpose to align with (x,y) axes
            bounds_error=False,
            fill_value=1.0,       
        )

        # 5. Define the 3D SDF evaluation function
        def _sdf(p: np.ndarray) -> np.ndarray:
            # Extract X, Y coordinates
            p_2d = p[..., :2] 
            # Get 2D distance
            d_2d = self.interpolator(p_2d)
            
            # Calculate Z distance (extrusion)
            d_z  = np.abs(p[..., 2]) - (self.thickness_z / 2.0)
            # The exact math for 3D extrusion
            return np.maximum(d_2d, d_z)

        # 6. Pass it to the Geometry3D base class
        super().__init__(_sdf)
        