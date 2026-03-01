import os
import numpy as np


def sample_levelset(geom, bounds, resolution):
    (x0, x1), (y0, y1), (z0, z1) = bounds
    nx, ny, nz = resolution

    xs = np.linspace(x0, x1, nx, endpoint=False) + (x1 - x0) / (2.0 * nx)
    ys = np.linspace(y0, y1, ny, endpoint=False) + (y1 - y0) / (2.0 * ny)
    zs = np.linspace(z0, z1, nz, endpoint=False) + (z1 - z0) / (2.0 * nz)

    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    p = np.stack([X, Y, Z], axis=-1)

    phi = geom.sdf(p)
    return phi


def save_npy(path, phi):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(path, phi)
