"""AMReX integration tests for SDFLibrary2D and SDFLibrary3D.

These tests require pyAMReX installed via conda::

    conda create -n pyamrex -c conda-forge pyamrex
    conda activate pyamrex

Without pyAMReX the entire module is skipped automatically.
Run the rest of the test suite (215 tests) without any AMReX dependency.
"""
import numpy as np
import pytest

# Skip the whole module when pyAMReX is not installed
amr2d = pytest.importorskip("amrex.space2d", reason="pyAMReX 2D not installed")
amr3d = pytest.importorskip("amrex.space3d", reason="pyAMReX 3D not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid_2d(n: int = 32, lo=(-1.0, -1.0), hi=(1.0, 1.0)):
    real_box = amr2d.RealBox(list(lo), list(hi))
    domain   = amr2d.Box(np.array([0, 0]), np.array([n - 1, n - 1]))
    geom     = amr2d.Geometry(domain, real_box, 0, [0, 0])
    ba       = amr2d.BoxArray(domain); ba.max_size(n // 2)
    dm       = amr2d.DistributionMapping(ba)
    return geom, ba, dm


def _make_grid_3d(n: int = 16, lo=(-1.0, -1.0, -1.0), hi=(1.0, 1.0, 1.0)):
    real_box = amr3d.RealBox(list(lo), list(hi))
    domain   = amr3d.Box(np.array([0, 0, 0]), np.array([n - 1, n - 1, n - 1]))
    geom     = amr3d.Geometry(domain, real_box, 0, [0, 0, 0])
    ba       = amr3d.BoxArray(domain); ba.max_size(n // 2)
    dm       = amr3d.DistributionMapping(ba)
    return geom, ba, dm


def _collect_2d(mf, n: int) -> np.ndarray:
    out = np.zeros((n, n))
    for mfi in mf:
        arr = mf.array(mfi).to_numpy()
        bx  = mfi.validbox()
        i0, j0 = bx.lo_vect; i1, j1 = bx.hi_vect
        out[j0:j1+1, i0:i1+1] = arr[:, :, 0, 0] if arr.ndim == 4 else arr[:, :, 0]
    return out


def _collect_3d(mf, n: int) -> np.ndarray:
    out = np.zeros((n, n, n))
    for mfi in mf:
        arr = mf.array(mfi).to_numpy()
        bx  = mfi.validbox()
        i0, j0, k0 = bx.lo_vect; i1, j1, k1 = bx.hi_vect
        vals = arr[:, :, :, 0] if arr.ndim == 4 else arr[:, :, :, 0, 0]
        out[k0:k1+1, j0:j1+1, i0:i1+1] = vals
    return out


# ---------------------------------------------------------------------------
# 2D tests
# ---------------------------------------------------------------------------

class TestSDFLibrary2D:
    @pytest.fixture(autouse=True)
    def init_amrex(self):
        amr2d.initialize([])
        yield
        amr2d.finalize()

    def test_circle_inside_origin(self):
        from sdf2d import SDFLibrary2D
        geom, ba, dm = _make_grid_2d(n=32)
        lib = SDFLibrary2D(geom, ba, dm)
        mf  = lib.circle(center=(0.0, 0.0), radius=0.3)
        phi = _collect_2d(mf, 32)
        # Centre cell should be inside (phi < 0)
        assert phi[16, 16] < 0

    def test_box_inside_origin(self):
        from sdf2d import SDFLibrary2D
        geom, ba, dm = _make_grid_2d(n=32)
        lib = SDFLibrary2D(geom, ba, dm)
        mf  = lib.box(center=(0.0, 0.0), half_size=(0.3, 0.3))
        phi = _collect_2d(mf, 32)
        assert phi[16, 16] < 0

    def test_returns_multifab(self):
        from sdf2d import SDFLibrary2D
        geom, ba, dm = _make_grid_2d(n=16)
        lib = SDFLibrary2D(geom, ba, dm)
        mf  = lib.circle(center=(0.0, 0.0), radius=0.5)
        assert hasattr(mf, "array")   # duck-type MultiFab check

    def test_union_contains_both(self):
        from sdf2d import SDFLibrary2D
        geom, ba, dm = _make_grid_2d(n=32)
        lib = SDFLibrary2D(geom, ba, dm)
        a = lib.circle(center=(-0.4, 0.0), radius=0.25)
        b = lib.circle(center=( 0.4, 0.0), radius=0.25)
        u = lib.union(a, b)
        phi = _collect_2d(u, 32)
        # Both centres should be inside the union
        assert phi[16, 8 ] < 0   # left circle centre
        assert phi[16, 24] < 0   # right circle centre


# ---------------------------------------------------------------------------
# 3D tests
# ---------------------------------------------------------------------------

class TestSDFLibrary3D:
    @pytest.fixture(autouse=True)
    def init_amrex(self):
        amr3d.initialize([])
        yield
        amr3d.finalize()

    def test_sphere_inside_origin(self):
        from sdf3d import SDFLibrary3D
        geom, ba, dm = _make_grid_3d(n=16)
        lib = SDFLibrary3D(geom, ba, dm)
        mf  = lib.sphere(center=(0.0, 0.0, 0.0), radius=0.3)
        phi = _collect_3d(mf, 16)
        assert phi[8, 8, 8] < 0

    def test_box_inside_origin(self):
        from sdf3d import SDFLibrary3D
        geom, ba, dm = _make_grid_3d(n=16)
        lib = SDFLibrary3D(geom, ba, dm)
        mf  = lib.box(center=(0.0, 0.0, 0.0), half_size=(0.4, 0.4, 0.4))
        phi = _collect_3d(mf, 16)
        assert phi[8, 8, 8] < 0

    def test_returns_multifab(self):
        from sdf3d import SDFLibrary3D
        geom, ba, dm = _make_grid_3d(n=16)
        lib = SDFLibrary3D(geom, ba, dm)
        mf  = lib.sphere(center=(0.0, 0.0, 0.0), radius=0.4)
        assert hasattr(mf, "array")

    def test_union_contains_both(self):
        from sdf3d import SDFLibrary3D
        geom, ba, dm = _make_grid_3d(n=16)
        lib = SDFLibrary3D(geom, ba, dm)
        a = lib.sphere(center=(-0.4, 0.0, 0.0), radius=0.2)
        b = lib.sphere(center=( 0.4, 0.0, 0.0), radius=0.2)
        u = lib.union(a, b)
        phi = _collect_3d(u, 16)
        assert phi[8, 4,  8] < 0   # left sphere
        assert phi[8, 12, 8] < 0   # right sphere
