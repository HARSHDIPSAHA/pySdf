import mesh2sdf
import trimesh
import numpy as np

def read_stl(file_path) -> np.ndarray:
    size = 128
    level = 2 / size
    mesh = trimesh.load(file_path, force='mesh')
    sdf, mesh = mesh2sdf.compute(
        mesh.vertices, mesh.faces, size, fix=True, level=level, return_mesh=True
    )
    return sdf