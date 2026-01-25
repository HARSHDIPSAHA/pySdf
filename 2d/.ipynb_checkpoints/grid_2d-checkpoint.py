import os
import numpy as np

def sample_levelset_2d(geom, bounds, resolution):
    """Sample 2D SDF on a regular grid."""
    (x0, x1), (y0, y1) = bounds
    nx, ny = resolution
    xs = np.linspace(x0, x1, nx, endpoint=False) + (x1 - x0) / (2.0 * nx)
    ys = np.linspace(y0, y1, ny, endpoint=False) + (y1 - y0) / (2.0 * ny)

    Y, X = np.meshgrid(ys, xs, indexing="ij")
    p = np.stack([X, Y], axis=-1)
    phi = geom.sdf(p)
    return phi

def save_npy(path, phi):
    """Save numpy array to file."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(path, phi)
