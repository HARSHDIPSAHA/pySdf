import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

class ImageExtruded3D:
    """
    Reads a 2D Chan-Vese SDF from uSCMAN and extrudes it into a 3D geometry node.
    Compatible with sdf3d operations.
    """
    def __init__(self, hdf5_path, dataset_path, physical_size_xy, thickness_z):
        self.thickness_z = thickness_z
        
        # 1. Load the 2D SDF array from sir's uSCMAN results
        with h5py.File(hdf5_path, 'r') as f:
            if dataset_path not in f:
                raise KeyError(f"Could not find {dataset_path} in {hdf5_path}")
            self.sdf_2d = f[dataset_path][:]
            
        # 2. Create physical coordinate grids
        res_y, res_x = self.sdf_2d.shape
        width, height = physical_size_xy
        
        # Assume image is centered at (0,0)
        x_coords = np.linspace(-width/2, width/2, res_x)
        y_coords = np.linspace(-height/2, height/2, res_y)
        
        # 3. Build a continuous interpolator
        # fill_value=1.0 ensures points far outside the image are treated as "outside" space
        self.interpolator = RegularGridInterpolator(
            (x_coords, y_coords), 
            self.sdf_2d.T, # Transpose to align (x,y)
            bounds_error=False, 
            fill_value=1.0 
        )

    def __call__(self, p):
        """Evaluates the 3D SDF at points p (..., 3)."""
        # Extract X, Y coordinates
        p_2d = p[..., :2]
        
        # Get 2D distance
        d_2d = self.interpolator(p_2d)
        
        # Calculate Z distance (extrusion)
        d_z = np.abs(p[..., 2]) - (self.thickness_z / 2.0)
        
        # The exact math for 3D extrusion
        return np.maximum(d_2d, d_z)

    # Add standard pySdf chained transforms so it acts like a native shape
    def translate(self, dx, dy, dz):
        from sdf3d import Geometry # Assuming you have a base wrapper
        return Geometry(lambda p: self(p - np.array([dx, dy, dz])))
        
    def rotate_x(self, angle):
        from sdf3d import Geometry
        c, s = np.cos(-angle), np.sin(-angle)
        rot_mat = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        return Geometry(lambda p: self(p @ rot_mat.T))