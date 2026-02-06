# Visualization Layer Implementation Summary

## Feature Requested
A simplified visualization API for saving level set visualizations as interactive HTML files.

## Implementation

### New Functions

#### 3D Visualization: `save_levelset_html()`
Located in `3d/visualization.py`, exported from `sdf3d` package.

**Usage:**
```python
from sdf3d import Sphere, sample_levelset, save_levelset_html

sphere = Sphere(0.3)
phi = sample_levelset(sphere, ((-1, 1), (-1, 1), (-1, 1)), (64, 64, 64))
save_levelset_html(phi, bounds=(-1, 1), filename="sphere.html")
```

**Features:**
- Uses plotly for interactive 3D visualization
- Uses scikit-image marching cubes to extract isosurface at SDF=0
- Filters out small disconnected fragments for cleaner output
- Generates standalone HTML file that can be opened in any browser

#### 2D Visualization: `save_levelset_html_2d()`
Located in `2d/visualization_2d.py`, exported from `sdf2d` package.

**Usage:**
```python
from sdf2d import Circle, sample_levelset_2d, save_levelset_html_2d

circle = Circle(0.3)
phi = sample_levelset_2d(circle, ((-1, 1), (-1, 1)), (256, 256))
save_levelset_html_2d(phi, bounds=(-1, 1), filename="circle.html")
```

**Features:**
- Uses plotly heatmap with diverging colormap (red=positive, blue=negative)
- Shows contour lines and bold zero level set contour
- Interactive hover to inspect SDF values
- Equal aspect ratio for correct shape visualization

### Changes Made

1. **New Files:**
   - `3d/visualization.py` - 3D visualization implementation
   - `2d/visualization_2d.py` - 2D visualization implementation
   - `test_visualization.py` - Comprehensive test suite
   - `examples/simple_visualization_example.py` - 3D example
   - `examples/simple_visualization_2d_example.py` - 2D example
   - `.gitignore` - Exclude build artifacts and cache files

2. **Modified Files:**
   - `sdf3d/__init__.py` - Export `save_levelset_html` function
   - `sdf2d/__init__.py` - Export `save_levelset_html_2d` function
   - `sdf3d/geometry.py` - Made AMReX import optional
   - `sdf2d/geometry.py` - Made AMReX import optional
   - `API_DOCUMENTATION.md` - Added visualization function documentation
   - `README.md` - Added quick example with new functions

3. **Key Design Decisions:**
   - Made AMReX imports optional so visualization works without AMReX
   - Both functions accept flexible bounds format (uniform or per-axis)
   - Comprehensive error handling with informative messages
   - Automatic directory creation for output files
   - Named constants for magic numbers (code review feedback)

### Testing

Created comprehensive test suite (`test_visualization.py`) covering:
- Basic 3D visualization with uniform bounds
- 3D visualization with per-axis bounds
- Complex 3D geometry visualization
- Basic 2D visualization with uniform bounds
- 2D visualization with per-axis bounds
- Complex 2D geometry visualization
- Error handling for invalid inputs

**All tests pass successfully ✅**

### Documentation

- Updated `API_DOCUMENTATION.md` with detailed function reference
- Updated `README.md` with quick example
- Added comprehensive docstrings to all new functions
- Created example scripts demonstrating simple usage

### Security

- Ran CodeQL security scan: **0 alerts found** ✅
- No security vulnerabilities introduced

## Result

The feature is now available exactly as requested in the issue. Users can simply call:

```python
save_levelset_html(phi, bounds=(-1, 1), filename="shape.html")
```

to generate interactive HTML visualizations of their level set fields.

## Installation

Users need to install visualization dependencies:
```bash
pip install -e .[viz]
```

This installs plotly, scikit-image, and matplotlib.
