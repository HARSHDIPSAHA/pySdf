import numpy as np
from .geometry import Box, Sphere, Union
from .grid import sample_levelset, save_npy


def main():
    # Build a simple union of a sphere and a box
    sphere = Sphere(radius=0.3)
    box = Box(half_size=(0.2, 0.2, 0.2)).translate(0.2, 0.0, 0.0)
    geom = Union(sphere, box).rotate_z(np.deg2rad(25.0))

    bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    resolution = (128, 128, 128)
    phi = sample_levelset(geom, bounds, resolution)
    save_npy("output/levelset.npy", phi)
    print("Saved output/levelset.npy")


if __name__ == "__main__":
    main()
