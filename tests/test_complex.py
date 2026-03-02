"""Tests for sdf3d.examples geometries (NATOFragment, RocketAssembly).

These tests do NOT require AMReX — they test the geometry objects
directly using sample_levelset_3d.
"""

import numpy as np
import pytest

from sdf3d import sample_levelset_3d


# ---------------------------------------------------------------------------
# Helpers — we need a mock lib that just returns the geometry
# ---------------------------------------------------------------------------

class _MockLib:
    """Minimal stand-in for SDFLibrary3D that captures from_geometry() calls."""

    def from_geometry(self, geom):
        return geom  # return the geometry object instead of a MultiFab


# ===========================================================================
# NATOFragment
# ===========================================================================

class TestNATOFragment:
    def test_import(self):
        from sdf3d.examples import NATOFragment  # noqa

    def test_returns_tuple(self):
        from sdf3d.examples import NATOFragment
        lib = _MockLib()
        result = NATOFragment(lib, diameter=14.30e-3, L_over_D=1.09)
        assert len(result) == 2

    def test_geometry_has_sdf(self):
        from sdf3d.examples import NATOFragment
        lib = _MockLib()
        _, geom = NATOFragment(lib)
        # Evaluate on a small grid
        bounds = ((-0.02, 0.02), (-0.02, 0.02), (-0.005, 0.02))
        res = (8, 8, 8)
        phi = sample_levelset_3d(geom, bounds, res)
        assert phi.shape == (8, 8, 8)

    def test_has_interior(self):
        from sdf3d.examples import NATOFragment
        lib = _MockLib()
        _, geom = NATOFragment(lib, diameter=14.30e-3)
        bounds = ((-0.01, 0.01), (-0.01, 0.01), (0.001, 0.015))
        phi = sample_levelset_3d(geom, bounds, (16, 16, 16))
        # The fragment cylinder occupies this region
        assert (phi < 0).any()

    def test_parametric_different_diameters(self):
        from sdf3d.examples import NATOFragment
        lib = _MockLib()
        _, g1 = NATOFragment(lib, diameter=10e-3)   # r = 5 mm
        _, g2 = NATOFragment(lib, diameter=20e-3)   # r = 10 mm
        # Use an off-axis point so the lateral (radius) dimension governs the SDF.
        # At x=4mm (inside both cylinders) the larger cylinder is deeper inside.
        p = np.array([[[0.004, 0.0, 0.005]]])
        d1 = g1.sdf(p)[0, 0]
        d2 = g2.sdf(p)[0, 0]
        assert d2 < d1  # larger fragment → deeper inside at same lateral offset


# ===========================================================================
# RocketAssembly
# ===========================================================================

class TestRocketAssembly:
    def test_import(self):
        from sdf3d.examples import RocketAssembly  # noqa

    def test_returns_tuple(self):
        from sdf3d.examples import RocketAssembly
        lib = _MockLib()
        result = RocketAssembly(lib)
        assert len(result) == 2

    def test_geometry_has_sdf(self):
        from sdf3d.examples import RocketAssembly
        lib = _MockLib()
        _, geom = RocketAssembly(lib)
        bounds = ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 1.0))
        phi = sample_levelset_3d(geom, bounds, (8, 8, 8))
        assert phi.shape == (8, 8, 8)

    def test_has_interior(self):
        from sdf3d.examples import RocketAssembly
        lib = _MockLib()
        _, geom = RocketAssembly(lib, body_radius=0.15, L_extra=0.4)
        bounds = ((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))
        phi = sample_levelset_3d(geom, bounds, (16, 16, 16))
        assert (phi < 0).any()

    def test_four_fins(self):
        from sdf3d.examples import RocketAssembly
        lib = _MockLib()
        _, geom = RocketAssembly(lib, n_fins=4)
        # Fins at ±R+span along X and Y at z=-0.18
        R = 0.15
        fin_span = 0.12
        fin_dist = R + fin_span / 2.0
        p = np.array([[[fin_dist, 0.0, -0.18]]])
        assert geom.sdf(p)[0, 0] < 0  # inside fin region
