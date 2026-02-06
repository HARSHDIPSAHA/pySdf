#!/usr/bin/env python3
"""
Test script for save_levelset_html functionality.

This script tests both 3D and 2D visualization functions.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_3d_visualization():
    """Test 3D level set HTML visualization."""
    print("=" * 60)
    print("Testing 3D Level Set Visualization")
    print("=" * 60)

    try:
        from sdf3d import Sphere, sample_levelset, save_levelset_html

        # Create a simple sphere
        sphere = Sphere(0.3)

        # Sample on grid
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        res = (64, 64, 64)  # Use lower resolution for faster testing
        phi = sample_levelset(sphere, bounds, res)

        print(
            f"✅ Sampled sphere: shape={phi.shape}, min={phi.min():.3f}, max={phi.max():.3f}"
        )

        # Test basic usage with uniform bounds
        output_file = "outputs/test_sphere_simple.html"
        save_levelset_html(phi, bounds=(-1, 1), filename=output_file)
        assert os.path.exists(output_file), f"Output file not created: {output_file}"
        print("✅ Created visualization with uniform bounds")

        # Test with per-axis bounds
        output_file2 = "outputs/test_sphere_bounds.html"
        save_levelset_html(phi, bounds=bounds, filename=output_file2)
        assert os.path.exists(output_file2), f"Output file not created: {output_file2}"
        print("✅ Created visualization with per-axis bounds")

        # Test with complex geometry
        from sdf3d import Box, Union

        sphere2 = Sphere(0.25).translate(-0.3, 0, 0)
        box = Box((0.2, 0.2, 0.2)).translate(0.3, 0, 0)
        combined = Union(sphere2, box)

        phi_combined = sample_levelset(combined, bounds, res)
        output_file3 = "outputs/test_combined.html"
        save_levelset_html(phi_combined, bounds=(-1, 1), filename=output_file3)
        assert os.path.exists(output_file3), f"Output file not created: {output_file3}"
        print("✅ Created visualization for combined geometry")

        print("\n" + "=" * 60)
        print("✅ ALL 3D TESTS PASSED")
        print("=" * 60)
        return True

    except ImportError as e:
        print(f"⚠️  Skipping 3D visualization test: {e}")
        print("    Install visualization dependencies with: pip install -e .[viz]")
        return False
    except Exception as e:
        print(f"❌ 3D test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_2d_visualization():
    """Test 2D level set HTML visualization."""
    print("\n" + "=" * 60)
    print("Testing 2D Level Set Visualization")
    print("=" * 60)

    try:
        from sdf2d import Circle, sample_levelset_2d, save_levelset_html_2d

        # Create a simple circle
        circle = Circle(0.3)

        # Sample on grid
        bounds = ((-1.0, 1.0), (-1.0, 1.0))
        res = (128, 128)
        phi = sample_levelset_2d(circle, bounds, res)

        print(
            f"✅ Sampled circle: shape={phi.shape}, min={phi.min():.3f}, max={phi.max():.3f}"
        )

        # Test basic usage with uniform bounds
        output_file = "outputs/test_circle_simple.html"
        save_levelset_html_2d(phi, bounds=(-1, 1), filename=output_file)
        assert os.path.exists(output_file), f"Output file not created: {output_file}"
        print("✅ Created 2D visualization with uniform bounds")

        # Test with per-axis bounds
        output_file2 = "outputs/test_circle_bounds.html"
        save_levelset_html_2d(phi, bounds=bounds, filename=output_file2)
        assert os.path.exists(output_file2), f"Output file not created: {output_file2}"
        print("✅ Created 2D visualization with per-axis bounds")

        # Test with complex geometry
        from sdf2d import Box2D, Union2D

        circle2 = Circle(0.25).translate(-0.3, 0)
        box = Box2D((0.2, 0.2)).translate(0.3, 0)
        combined = Union2D(circle2, box)

        phi_combined = sample_levelset_2d(combined, bounds, res)
        output_file3 = "outputs/test_combined_2d.html"
        save_levelset_html_2d(phi_combined, bounds=(-1, 1), filename=output_file3)
        assert os.path.exists(output_file3), f"Output file not created: {output_file3}"
        print("✅ Created 2D visualization for combined geometry")

        print("\n" + "=" * 60)
        print("✅ ALL 2D TESTS PASSED")
        print("=" * 60)
        return True

    except ImportError as e:
        print(f"⚠️  Skipping 2D visualization test: {e}")
        print("    Install visualization dependencies with: pip install -e .[viz]")
        return False
    except Exception as e:
        print(f"❌ 2D test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)

    try:
        from sdf3d import save_levelset_html  # noqa: F401

        # Test with invalid phi (not an array)
        try:
            save_levelset_html("not an array", bounds=(-1, 1), filename="test.html")
            print("❌ Should have raised ValueError for non-array input")
            return False
        except ValueError as e:
            print(f"✅ Correctly raised ValueError for non-array input: {e}")

        # Test with wrong dimensions
        try:
            phi_wrong = np.zeros((10, 10))  # 2D array for 3D function
            save_levelset_html(phi_wrong, bounds=(-1, 1), filename="test.html")
            print("❌ Should have raised ValueError for wrong dimensions")
            return False
        except ValueError as e:
            print(f"✅ Correctly raised ValueError for wrong dimensions: {e}")

        # Test with invalid bounds
        try:
            phi = np.zeros((10, 10, 10))
            save_levelset_html(phi, bounds=(1, 2, 3), filename="test.html")
            print("❌ Should have raised ValueError for invalid bounds")
            return False
        except ValueError as e:
            print(f"✅ Correctly raised ValueError for invalid bounds: {e}")

        print("\n" + "=" * 60)
        print("✅ ALL ERROR HANDLING TESTS PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SAVE_LEVELSET_HTML TEST SUITE")
    print("=" * 60 + "\n")

    # Create output directory
    os.makedirs("outputs", exist_ok=True)

    # Run tests
    results = []
    results.append(("3D Visualization", test_3d_visualization()))
    results.append(("2D Visualization", test_2d_visualization()))
    results.append(("Error Handling", test_error_handling()))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")

    all_passed = all(passed for _, passed in results)
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
