"""2D grid sampling utilities."""
from ._loader import load_module

# Load grid utilities from 2d folder
_grid = load_module("sdf2d._grid", "2d/grid_2d.py")

# Re-export functions
sample_levelset_2d = _grid.sample_levelset_2d
save_npy = _grid.save_npy

__all__ = [
    "sample_levelset_2d",
    "save_npy",
]
