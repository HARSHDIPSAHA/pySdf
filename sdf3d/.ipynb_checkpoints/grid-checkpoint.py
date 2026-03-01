from ._loader import load_module

_grid = load_module("sdf3d._grid", "3d/grid.py")

sample_levelset = _grid.sample_levelset
save_npy = _grid.save_npy

__all__ = ["sample_levelset", "save_npy"]
