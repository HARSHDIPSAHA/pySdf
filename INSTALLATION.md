# Installation Guide

## Quick Install

```bash
# Basic installation (numpy only)
pip install -e .

# With visualization support (plotly, matplotlib, scikit-image)
pip install -e .[viz]

# With AMReX support (if available)
pip install -e .[amrex]

# All features
pip install -e .[viz,amrex]
```

## What Gets Installed

The `pip install -e .` command installs:

- **`sdf2d`**: 2D signed distance function library
- **`sdf3d`**: 3D signed distance function library
- **`sdf2d/primitives.py`**, **`sdf3d/primitives.py`**: Core SDF primitive implementations

## Package Structure

After installation, you can import:

```python
# 2D API
from sdf2d import Circle, Box2D, Union2D, sample_levelset_2d, SDFLibrary2D

# 3D API
from sdf3d import Sphere, Box, Union, sample_levelset, SDFLibrary
```

## Dependencies

### Required
- `numpy>=1.20.0` (always installed)

### Optional
- **Visualization** (`[viz]`): `matplotlib`, `plotly`, `scikit-image`
- **AMReX** (`[amrex]`): `amrex>=23.0` (if available)
- **Development** (`[dev]`): `pytest`, `black`, `flake8`

## Verification

After installation, test that it works:

```bash
# Test 3D library
python -c "from sdf3d import Sphere; print('✅ sdf3d works')"

# Test 2D library
python -c "from sdf2d import Circle; print('✅ sdf2d works')"
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:
- Make sure you ran `pip install -e .` from the project root
- Check that you're using the correct Python environment

### AMReX Not Found

If AMReX features don't work:
- AMReX is optional - the library works without it (NumPy mode only)
- To use AMReX features, install pyAMReX separately and use `pip install -e .[amrex]`

### Visualization Not Working

If 3D HTML visualizations fail:
- Install visualization dependencies: `pip install -e .[viz]`
- This installs `plotly` and `scikit-image` required for interactive HTML

## Development Mode

The `-e` flag installs in "editable" mode, meaning:
- Changes to source code are immediately available
- No need to reinstall after code changes
- Perfect for development

## Uninstall

```bash
pip uninstall sdf-library
```
