# SDF Library Examples

This folder contains example scripts demonstrating various SDF operations with **mathematical verification**.

## Running Examples

All examples can be run from the project root:

```bash
python examples/union_example.py
python examples/intersection_example.py
python examples/subtraction_example.py
python examples/elongation_example.py
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
- Sphere 1: center (-0.3, 0, 0), radius 0.25
- Sphere 2: center (0.3, 0, 0), radius 0.25

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

## Understanding the Output

Each example prints:
- **Min/Max values**: Range of the SDF field
- **Has negative/positive**: Confirms inside/outside regions exist
- **Mathematical verification**: Compares computed result with expected formula
- **Pass/Fail status**: ✅ or ❌ based on correctness checks

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
- Grid resolution: 64×64×64 cells in domain [-1, 1]³
