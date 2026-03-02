# Installation Guide

## Core library

### pip

```bash
# Core library (numpy only)
pip install -e .

# With visualization support (plotly, matplotlib, scikit-image)
pip install -e ".[viz]"

# With development tools (adds pytest)
pip install -e ".[dev]"

# Everything except AMReX
pip install -e ".[viz,dev]"
```

### uv

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Core library (numpy only)
uv sync

# With visualization support (plotly, matplotlib, scikit-image)
uv sync --extra viz

# With development tools (adds pytest)
uv sync --extra dev

# Everything except AMReX
uv sync --extra viz --extra dev
```

Both install all three packages:

| Package | Description |
|---------|-------------|
| `sdf2d` | 2D geometry classes, grid sampling, optional AMReX output |
| `sdf3d` | 3D geometry classes, grid sampling, optional AMReX output |
| `stl2sdf` | STL mesh → SDF grid (pure NumPy, no extra deps) |

## Verification

```bash
uv run pytest tests/ -v   # test_amrex.py skips without pyAMReX

uv run python -c "from sdf3d import Sphere3D; print('sdf3d OK')"
uv run python -c "from sdf2d import Circle2D; print('sdf2d OK')"
uv run python -c "from stl2sdf import stl_to_geometry; print('stl2sdf OK')"
```

---

## pyAMReX (optional)

**pyAMReX is not on PyPI** — `pip`/`spack`/`brew` installs are listed as "coming soon"
in the official docs. There are three practical methods for using it with this project,
each with different trade-offs.

> **Dimension constraint:** Only one space dimension can be active per Python process.
> Import `amrex.space2d` *or* `amrex.space3d`, not both simultaneously.

---

### Method A — conda (CPU only, recommended for most users)

The official recommended path. Does **not** support GPU acceleration.

```bash
# Speed up conda's solver first (one-time, strongly recommended)
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# Create the environment and install pyAMReX
conda create -n pyamrex -c conda-forge pyamrex
conda activate pyamrex

# Install pySdf into the conda env (bypasses uv — uses pip directly)
pip install -e .
```

Run tests in the conda env:
```bash
pytest tests/   # test_amrex.py will now run instead of skipping
```

> **Note:** In the conda env you use `pip install -e .` directly, not `uv sync`.
> uv manages its own `.venv`; conda manages its own env. They are separate.

---

### Method B — build from source (GPU / MPI / custom dimensions)

Required for CUDA/HIP/SYCL GPU support or OpenMP. Builds pyAMReX into the
**active Python environment** (either a conda env or the uv `.venv`).

```bash
# 1. Clone the pyAMReX source
git clone https://github.com/AMReX-Codes/pyamrex.git $HOME/src/pyamrex
cd $HOME/src/pyamrex

# 2. Install build prerequisites into the target environment
python3 -m pip install -U pip build packaging "setuptools[core]" wheel pytest
python3 -m pip install -U -r requirements.txt

# 3. Configure — build all three space dimensions in one go
cmake -S . -B build -DAMReX_SPACEDIM="1;2;3"

# Optional flags:
#   -DAMReX_GPU_BACKEND=CUDA      (or HIP, SYCL)
#   -DAMReX_MPI=ON
#   -DAMReX_OMP=ON
#   -DCMAKE_BUILD_TYPE=Release

# 4. Build and install (pip_install target places it in the active env)
cmake --build build -j 4 --target pip_install
```

To set specific compilers:
```bash
export CC=$(which clang)
export CXX=$(which clang++)
export CUDACXX=$(which nvcc)       # CUDA only
export CUDAHOSTCXX=$(which clang++)
```

After the build, install pySdf into the same environment:
```bash
pip install -e .   # if in conda env
# — or —
uv pip install -e .   # if targeting the uv .venv
```

Full guide: [pyamrex.readthedocs.io — cmake build](https://pyamrex.readthedocs.io/en/latest/install/cmake.html)

---

### Method C — inject a pre-built wheel into the uv venv

If you have already built pyAMReX from source and have a `.whl` file (produced
by the cmake `pip_install` target or `python -m build`), you can add it to the
uv-managed environment without leaving the uv workflow:

```bash
uv pip install /path/to/pyamrex-*.whl
```

This keeps uv as the single environment manager.

> **Caveat:** uv will not track this wheel in `uv.lock` (it's not a PyPI package),
> so the installation is not reproducible from the lockfile alone. Document the
> wheel source and version in your project notes.

---

## Summary

| Method | GPU support | Env manager | Best for |
|--------|-------------|-------------|----------|
| A — conda | No | conda + pip | quickest AMReX setup, CPU-only |
| B — source build | Yes | conda or uv | GPU / MPI / custom dims |
| C — inject wheel | Yes (if built with GPU) | uv | staying in the uv workflow |
| (none) — uv only | N/A | uv | all pure-numpy work, CI, stl2sdf |

## Troubleshooting

### `ModuleNotFoundError` for `sdf2d` / `sdf3d` / `stl2sdf`
Run from the project root (where `pyproject.toml` lives) using `uv run`, or ensure
the correct environment is activated.

### `test_amrex.py` skips
Expected when pyAMReX is not installed. Install via Method A, B, or C above.

### `stl2sdf` wrong signs / holes in isosurface
Sign determination requires a **watertight** (closed, 2-manifold) mesh.
Check for open edges with Blender, MeshLab, or:
```python
from stl2sdf._math import _stl_to_triangles
import numpy as np
from collections import Counter
tris = _stl_to_triangles("mesh.stl")
verts = np.round(tris.reshape(-1, 3), 5)
_, inv = np.unique(verts, axis=0, return_inverse=True)
ids = inv.reshape(-1, 3)
edges = Counter(tuple(sorted((t[i], t[(i+1)%3]))) for t in ids for i in range(3))
print("Boundary edges:", sum(1 for c in edges.values() if c == 1))
```

### Uninstall

```bash
pip uninstall sdf-library   # in conda env
# — or —
uv pip uninstall sdf-library  # in uv venv
```
