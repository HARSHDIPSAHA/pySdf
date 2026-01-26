# Complex Example - Expected Output Description

This document describes what the final shape should look like after all operations in `complex_example.py`.

## Step-by-Step Visual Description

### Step 1: Base Box
- **Shape**: A cube centered at origin (0, 0, 0)
- **Size**: 0.6×0.6×0.6 units (half_size = 0.3)
- **Appearance**: Sharp edges, rectangular faces
- **Color**: Light blue

### Step 2: Elongated Sphere (Capsule)
- **Shape**: A capsule (cylinder with spherical caps)
- **Construction**: Union of two spheres (at x=-0.3 and x=+0.3, radius 0.2) + a box (0.6×0.4×0.4)
- **Orientation**: Elongated along x-axis from -0.3 to +0.3
- **Appearance**: Smooth rounded ends, cylindrical middle
- **Color**: Light green

### Step 3: Union (Box + Capsule)
- **Shape**: Base box merged with capsule
- **Appearance**: 
  - The cube from step 1
  - With a capsule extension protruding along the x-axis
  - The capsule connects to the box, creating a single merged shape
  - Sharp transition where they meet
- **Color**: Gold

### Step 4: Intersection (Rounding)
- **Shape**: Union result intersected with a large sphere
- **Large Sphere**: Center at (0, 0.2, 0), radius 0.6
- **Effect**: 
  - The top and upper edges of the shape are rounded
  - The large sphere "cuts off" the top portion, creating a domed/rounded top
  - Lower portions that don't intersect with the sphere are removed
  - Creates a more organic, rounded appearance
- **Appearance**: 
  - Rounded top surface
  - Smooth curved edges on the upper portion
  - The capsule extension may be partially or fully removed if it extends beyond the sphere
- **Color**: Coral

### Step 5: Subtraction (Cavity)
- **Shape**: Rounded shape with a cavity
- **Cutter Box**: Center at (0, -0.1, 0), half_size = (0.15, 0.15, 0.15)
- **Effect**: 
  - A rectangular hole/cavity is cut out from the lower center
  - The cavity is 0.3×0.3×0.3 units
  - Creates an internal void in the shape
- **Appearance**: 
  - Same as step 4, but with a visible hole/cavity
  - The cavity should be clearly visible when viewing the shape
  - Internal surfaces of the cavity are visible
- **Color**: Plum

### Final Result
- **Overall Shape**: A complex, rounded mechanical/architectural element
- **Key Features**:
  1. **Base Structure**: A rounded box-like structure (from original box, rounded by intersection)
  2. **Extension**: A capsule/cylindrical extension on the x-axis (from union)
  3. **Rounded Top**: Domed/rounded upper surface (from intersection with large sphere)
  4. **Cavity**: A rectangular hole in the lower center (from subtraction)
- **Visual Characteristics**:
  - Smooth, rounded surfaces on top
  - Sharp edges where appropriate
  - Visible internal cavity
  - Asymmetric shape (extended on x-axis, rounded on top, hollow in center)
- **Color**: Medium purple

## Mathematical Verification Points

### Expected SDF Values at Key Points:

1. **Origin (0, 0, 0)**:
   - Should be inside the base box initially
   - After subtraction, may be inside or outside depending on cavity position
   - Expected: Negative value (inside) if not in cavity, positive if in cavity

2. **Top Center (0, 0.4, 0)**:
   - Should be outside (positive) - above the rounded top
   - Expected: Positive value

3. **Capsule End (0.3, 0, 0)**:
   - Should be inside the capsule extension
   - Expected: Negative value

4. **Cavity Center (0, -0.1, 0)**:
   - Should be outside (positive) - inside the cavity means outside the solid
   - Expected: Positive value

5. **Far Outside (0, 1, 0)**:
   - Should be far outside
   - Expected: Large positive value

## Visual Inspection Checklist

When viewing the final 3D HTML visualization, verify:

- [ ] The shape has a rounded top (domed appearance)
- [ ] There is a capsule/cylindrical extension along the x-axis
- [ ] A rectangular cavity/hole is visible in the lower center
- [ ] The shape is a single connected piece (no floating fragments)
- [ ] Surfaces are smooth (no excessive faceting artifacts)
- [ ] The cavity is clearly visible and properly formed
- [ ] The overall shape looks like a complex mechanical part

## Common Issues to Watch For

1. **Artifacts**: Small floating fragments (should be filtered out)
2. **Missing Cavity**: If cavity is not visible, check subtraction operation
3. **Missing Rounding**: If top is not rounded, check intersection operation
4. **Disconnected Parts**: If capsule is disconnected, check union operation
5. **Incorrect Proportions**: Verify all dimensions match expected values

## File Outputs

The example generates 6 HTML files:
1. `complex_example_step1_3d.html` - Base box
2. `complex_example_step2_3d.html` - Capsule
3. `complex_example_step3_3d.html` - Union result
4. `complex_example_step4_3d.html` - Intersection (rounded)
5. `complex_example_step5_3d.html` - Subtraction (with cavity)
6. `complex_example_final_3d.html` - Final complex shape

All files are saved to `outputs/vis3d_plotly/` and can be opened in any web browser.
