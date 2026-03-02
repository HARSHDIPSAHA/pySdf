"""Tests for sdf3d grid utilities."""

import os
import tempfile

import numpy as np
import numpy.testing as npt

from sdf3d import Sphere3D, Box3D, sample_levelset_3d, save_npy


class TestSampleLevelset3D:
    def test_output_shape(self):
        g = Sphere3D(0.3)
        phi = sample_levelset_3d(g, ((-1, 1), (-1, 1), (-1, 1)), (16, 16, 16))
        assert phi.shape == (16, 16, 16)

    def test_non_cube(self):
        g = Sphere3D(0.3)
        phi = sample_levelset_3d(g, ((-1, 1), (-1, 1), (-1, 1)), (8, 16, 32))
        assert phi.shape == (32, 16, 8)

    def test_cell_centred_near_minus_r(self):
        n = 65
        g = Sphere3D(0.3)
        phi = sample_levelset_3d(g, ((-1, 1), (-1, 1), (-1, 1)), (n, n, n))
        centre = phi[32, 32, 32]
        npt.assert_allclose(centre, -0.3, atol=0.02)

    def test_inside_negative_outside_positive(self):
        g = Sphere3D(0.3)
        phi = sample_levelset_3d(g, ((-1, 1), (-1, 1), (-1, 1)), (16, 16, 16))
        assert (phi < 0).any()
        assert (phi > 0).any()


class TestSaveNpy3D:
    def test_round_trip(self):
        phi = np.random.rand(4, 4, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "phi3d.npy")
            save_npy(path, phi)
            loaded = np.load(path)
        npt.assert_array_equal(phi, loaded)

    def test_creates_nested_dirs(self):
        phi = np.zeros((4, 4, 4))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a", "b", "phi.npy")
            save_npy(path, phi)
            assert os.path.isfile(path)
