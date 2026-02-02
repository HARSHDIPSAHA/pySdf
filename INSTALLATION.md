```bash
# Basic installation
pip install -e .

# With visualization support (plotly, matplotlib, scikit-image)
pip install -e .[viz]
```

Installing amrex may take more than two hours as it builds from source.

## What Gets Installed

The `pip install -e .` command installs:

- **`sdf2d`**: 2D signed distance function library
- **`sdf3d`**: 3D signed distance function library
- **`sdf_lib.py`**: Core SDF primitive implementations in numpy

## Package Structure

After installation, you can import:

```python
# 2D API
from sdf2d import Circle, Box2D, Union2D, sample_levelset_2d, SDFLibrary2D

# 3D API
from sdf3d import Sphere, Box, Union, sample_levelset, SDFLibrary
```

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
