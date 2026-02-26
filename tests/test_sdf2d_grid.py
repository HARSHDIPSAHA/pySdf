"""Tests for sdf2d grid utilities."""

import os
import tempfile

import numpy as np
import numpy.testing as npt
import pytest

from sdf2d import Circle2D, Box2D, sample_levelset_2d, save_npy


class TestSampleLevelset2D:
    def test_output_shape(self):
        g = Circle2D(0.3)
        phi = sample_levelset_2d(g, ((-1, 1), (-1, 1)), (32, 32))
        assert phi.shape == (32, 32)

    def test_non_square(self):
        g = Circle2D(0.3)
        phi = sample_levelset_2d(g, ((-1, 1), (-1, 1)), (16, 32))
        assert phi.shape == (32, 16)

    def test_cell_centred_at_origin(self):
        # With odd resolution, the centre cell is very close to (0, 0)
        g = Circle2D(0.3)
        n = 65
        phi = sample_levelset_2d(g, ((-1, 1), (-1, 1)), (n, n))
        # Centre cell index is 32
        centre = phi[32, 32]
        npt.assert_allclose(centre, -0.3, atol=0.02)

    def test_inside_negative_outside_positive(self):
        g = Circle2D(0.3)
        phi = sample_levelset_2d(g, ((-1, 1), (-1, 1)), (64, 64))
        assert (phi < 0).any()
        assert (phi > 0).any()

    def test_dtype_is_float(self):
        g = Circle2D(0.3)
        phi = sample_levelset_2d(g, ((-1, 1), (-1, 1)), (16, 16))
        assert np.issubdtype(phi.dtype, np.floating)


class TestSaveNpy:
    def test_round_trip(self):
        phi = np.random.rand(8, 8).astype(np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "phi.npy")
            save_npy(path, phi)
            loaded = np.load(path)
        npt.assert_array_equal(phi, loaded)

    def test_creates_parent_dirs(self):
        phi = np.zeros((4, 4))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a", "b", "c.npy")
            save_npy(path, phi)
            assert os.path.isfile(path)
