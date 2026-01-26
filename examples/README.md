# SDF Library Examples

This folder contains example scripts demonstrating various SDF operations with **mathematical verification**.

## Running Examples

All examples can be run from the project root:

```bash
python examples/union_example.py
python examples/intersection_example.py
python examples/subtraction_example.py
python examples/elongation_example.py
python examples/complex_example.py
```

## Example Descriptions

### 1. Union Example (`union_example.py`)

**Operation**: Union of two overlapping spheres

**Mathematical Formula**:
```
Union(S1, S2) = min(SDF_S1, SDF_S2)
```

**Expected Behavior**:
- At points inside either sphere: negative values
- At points outside both: positive values
- Surface (zero level set) exists where either sphere's surface is

**Test Setup**:
- Sphere 1: center (-0.2, 0, 0), radius 0.3
- Sphere 2: center (0.2, 0, 0), radius 0.3
- Distance between centers: 0.4, so spheres overlap (each radius 0.3)

**Verification**:
- Checks for negative (inside) and positive (outside) values
- Verifies surface exists (near-zero values)

---

### 2. Intersection Example (`intersection_example.py`)

**Operation**: Intersection of two overlapping spheres

**Mathematical Formula**:
```
Intersection(S1, S2) = max(SDF_S1, SDF_S2)
```

**Expected Behavior**:
- At points inside both spheres: negative (but less negative than individual)
- At points outside either: positive
- Surface exists only in the overlap region

**Test Setup**:
- Sphere 1: center (0, 0, 0), radius 0.35
- Sphere 2: center (0.2, 0, 0), radius 0.35

**Verification**:
- Computes `max(S1, S2)` and compares with intersection result
- Should match exactly (within numerical precision)

---

### 3. Subtraction Example (`subtraction_example.py`)

**Operation**: Subtract one sphere from another (cutting a hole)

**Mathematical Formula**:
```
Subtraction(base, cutter) = max(-SDF_base, SDF_cutter)
```

**Expected Behavior**:
- Creates a hole where the cutter overlaps the base
- Points inside the cutter become outside the result
- Points far from both remain outside

**Test Setup**:
- Base sphere: center (0, 0, 0), radius 0.4
- Cutter sphere: center (0.2, 0, 0), radius 0.25

**Verification**:
- Computes `max(-base, cutter)` and compares with subtraction result
- Should match exactly (within numerical precision)

---

### 4. Elongation Example (`elongation_example.py`)

**Operation**: Elongate a sphere along the x-axis

**Mathematical Formula**:
```
Elongation: q = p - clamp(p, -h, h)
Then: SDF_elongated = SDF_sphere(q)
```

**Expected Behavior**:
- Original sphere shape is preserved but stretched
- Elongation creates a capsule-like shape
- Distance field remains valid (signed distance preserved)

**Test Setup**:
- Base sphere: radius 0.25
- Elongation: (0.3, 0.0, 0.0) in x-direction

**Verification**:
- Checks values at origin (should be inside, ~-0.25)
- Checks values at elongation boundary (0.3, 0, 0)
- Checks values beyond elongation (0.5, 0, 0)

---

### 5. Complex Example (`complex_example.py`)

**Operation**: Combines all operations: Union, Elongation, Intersection, and Subtraction

**Sequence of Operations**:
1. **Base**: Create a box at origin
2. **Elongation**: Create an elongated sphere (capsule) using geometry API
3. **Union**: Combine base box with capsule
4. **Intersection**: Round the top with a large sphere
5. **Subtraction**: Create a cavity by subtracting a small box

**Mathematical Formula**:
```
Step 1: base = box(center=(0,0,0), half_size=(0.3,0.3,0.3))
Step 2: capsule = elongated_sphere(radius=0.2, elongation=(0.3,0,0))
Step 3: union_result = union(base, capsule)
Step 4: rounder = sphere(center=(0,0.2,0), radius=0.6)
Step 5: rounded = intersect(union_result, rounder)
Step 6: cutter = box(center=(0,-0.1,0), half_size=(0.15,0.15,0.15))
Step 7: final = subtract(rounded, cutter)
```

**Expected Final Shape**:
- A rounded box-like structure (from base box)
- With a cylindrical/capsule extension on the x-axis (from union with elongated sphere)
- Rounded top and upper edges (from intersection with large sphere)
- A rectangular cavity/hole in the lower center (from subtraction)
- Overall shape resembles a complex mechanical part or architectural element

**Verification**:
- Each step generates a separate 3D HTML visualization
- Final shape should have:
  - Negative values (inside the solid)
  - Positive values (outside)
  - Near-zero values (on the surface)
  - Visible cavity/hole in the structure

**Output Files**:
- `complex_example_step1_3d.html`: Base box
- `complex_example_step2_3d.html`: Elongated sphere (capsule)
- `complex_example_step3_3d.html`: Union result (box + capsule)
- `complex_example_step4_3d.html`: Intersection result (rounded)
- `complex_example_step5_3d.html`: Subtraction result (with cavity)
- `complex_example_final_3d.html`: Final complex shape

---

## Understanding the Output

Each example:
1. **Prints verification results**:
   - Min/Max values: Range of the SDF field
   - Has negative/positive: Confirms inside/outside regions exist
   - Mathematical verification: Compares computed result with expected formula
   - Pass/Fail status: ✅ or ❌ based on correctness checks

2. **Generates interactive 3D HTML visualization** (if plotly/scikit-image installed):
   - Saved to `outputs/vis3d_plotly/<example_name>_3d.html`
   - Open in a web browser to view and rotate the 3D isosurface
   - Shows the SDF=0 surface (geometry boundary)

## Expected Output Format

```
============================================================
OPERATION EXAMPLE: Description
============================================================
Min value (should be < 0, inside): -0.XXXXXX
Max value (should be > 0, outside): X.XXXXXX
Has negative values (inside): True
Has positive values (outside): True
[Additional mathematical checks...]

============================================================
✅ TEST PASSED: Description
============================================================
```

## Notes

- All examples use **AMReX MultiFab** for solver-native output
- Elongation example uses the **geometry API** (numpy-based) for comparison
- Numerical precision: differences < 1e-5 are considered exact matches
- Grid resolution: 128×128×128 cells in domain [-1, 1]³ (higher resolution for smoother visualizations)
- **3D visualizations require**: `pip install -e .[viz]` (installs plotly and scikit-image)
- HTML files are saved to `outputs/vis3d_plotly/` and can be opened in any web browser
