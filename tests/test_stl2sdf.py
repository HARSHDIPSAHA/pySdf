"""Tests for stl2sdf.

STL loading tested via stl2sdf._math (private but tested directly).
Public API tested via stl2sdf.stl_to_geometry.
"""
from __future__ import annotations

import struct
import warnings
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from stl2sdf import stl_to_geometry
from stl2sdf._math import _stl_to_triangles
from sdf3d.geometry import Geometry3D
from sdf3d import Sphere3D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_box_triangles(hx: float = 0.5, hy: float = 0.5, hz: float = 0.5) -> np.ndarray:
    """12-triangle watertight box [-hx,hx]×[-hy,hy]×[-hz,hz]."""
    verts = np.array([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ], dtype=np.float64)
    face_indices = [
        (0, 2, 1), (0, 3, 2),   # -Z
        (4, 5, 6), (4, 6, 7),   # +Z
        (0, 4, 7), (0, 7, 3),   # -X
        (1, 2, 6), (1, 6, 5),   # +X
        (0, 1, 5), (0, 5, 4),   # -Y
        (3, 7, 6), (3, 6, 2),   # +Y
    ]
    return np.array([[verts[i], verts[j], verts[k]] for i, j, k in face_indices],
                    dtype=np.float64)


def _write_binary_stl(triangles: np.ndarray) -> bytes:
    header  = b"\x00" * 80
    count   = struct.pack("<I", len(triangles))
    records = bytearray()
    for tri in triangles:
        records += struct.pack("<fff", 0.0, 0.0, 0.0)
        for v in tri:
            records += struct.pack("<fff", float(v[0]), float(v[1]), float(v[2]))
        records += struct.pack("<H", 0)
    return header + count + bytes(records)


def _write_ascii_stl(triangles: np.ndarray) -> str:
    lines = ["solid test"]
    for tri in triangles:
        lines.append("  facet normal 0 0 0")
        lines.append("    outer loop")
        for v in tri:
            lines.append(f"      vertex {v[0]:.6g} {v[1]:.6g} {v[2]:.6g}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid test")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# _stl_to_triangles
# ---------------------------------------------------------------------------

class TestStlToTriangles:
    def test_binary_shape(self, tmp_path):
        stl = tmp_path / "box.stl"
        stl.write_bytes(_write_binary_stl(_make_box_triangles()))
        loaded = _stl_to_triangles(stl)
        assert loaded.shape == (12, 3, 3)
        assert loaded.dtype == np.float64

    def test_binary_values(self, tmp_path):
        tris = _make_box_triangles()
        stl  = tmp_path / "box.stl"
        stl.write_bytes(_write_binary_stl(tris))
        npt.assert_allclose(_stl_to_triangles(stl), tris, atol=1e-6)

    def test_ascii_shape(self, tmp_path):
        stl = tmp_path / "box.stl"
        stl.write_text(_write_ascii_stl(_make_box_triangles()))
        loaded = _stl_to_triangles(stl)
        assert loaded.shape == (12, 3, 3)
        assert loaded.dtype == np.float64

    def test_ascii_values(self, tmp_path):
        tris = _make_box_triangles()
        stl  = tmp_path / "box.stl"
        stl.write_text(_write_ascii_stl(tris))
        npt.assert_allclose(_stl_to_triangles(stl), tris, atol=1e-5)


# ---------------------------------------------------------------------------
# stl_to_geometry (public API)
# ---------------------------------------------------------------------------

class TestStlToGeometry:
    def test_returns_geometry3d(self, tmp_path):
        stl = tmp_path / "box.stl"
        stl.write_bytes(_write_binary_stl(_make_box_triangles()))
        assert isinstance(stl_to_geometry(stl), Geometry3D)

    def test_inside_negative(self, tmp_path):
        stl = tmp_path / "box.stl"
        stl.write_bytes(_write_binary_stl(_make_box_triangles()))
        assert stl_to_geometry(stl).sdf(np.array([[0.,0.,0.]]))[0] < 0

    def test_outside_positive(self, tmp_path):
        stl = tmp_path / "box.stl"
        stl.write_bytes(_write_binary_stl(_make_box_triangles()))
        assert stl_to_geometry(stl).sdf(np.array([[2.,0.,0.]]))[0] > 0

    def test_distance_inside(self, tmp_path):
        stl = tmp_path / "box.stl"
        stl.write_bytes(_write_binary_stl(_make_box_triangles()))
        phi = stl_to_geometry(stl).sdf(np.array([[0.3, 0., 0.]]))
        npt.assert_allclose(np.abs(phi), [0.2], atol=1e-5)
        assert phi[0] < 0

    def test_distance_outside(self, tmp_path):
        stl = tmp_path / "box.stl"
        stl.write_bytes(_write_binary_stl(_make_box_triangles()))
        npt.assert_allclose(stl_to_geometry(stl).sdf(np.array([[1., 0., 0.]])), [0.5], atol=1e-5)

    def test_batch_shape(self, tmp_path):
        stl = tmp_path / "box.stl"
        stl.write_bytes(_write_binary_stl(_make_box_triangles()))
        assert stl_to_geometry(stl).sdf(np.zeros((7, 3))).shape == (7,)

    def test_grid_shape(self, tmp_path):
        stl = tmp_path / "box.stl"
        stl.write_bytes(_write_binary_stl(_make_box_triangles()))
        lin = np.linspace(-1., 1., 5)
        Z, Y, X = np.meshgrid(lin, lin, lin, indexing="ij")
        P = np.stack([X, Y, Z], axis=-1)
        assert stl_to_geometry(stl).sdf(P).shape == (5, 5, 5)

    def test_translate(self, tmp_path):
        stl = tmp_path / "box.stl"
        stl.write_bytes(_write_binary_stl(_make_box_triangles()))
        shifted = stl_to_geometry(stl).translate(5., 0., 0.)
        assert shifted.sdf(np.array([[0.,0.,0.]]))[0] > 0
        assert shifted.sdf(np.array([[5.,0.,0.]]))[0] < 0

    def test_union_with_sphere(self, tmp_path):
        stl = tmp_path / "box.stl"
        stl.write_bytes(_write_binary_stl(_make_box_triangles()))
        combined = stl_to_geometry(stl).union(Sphere3D(0.3).translate(0.8, 0., 0.))
        assert combined.sdf(np.array([[0.8, 0., 0.]]))[0] < 0

    def test_subtract_sphere(self, tmp_path):
        stl = tmp_path / "box.stl"
        stl.write_bytes(_write_binary_stl(_make_box_triangles()))
        hollowed = stl_to_geometry(stl).subtract(Sphere3D(0.3))
        assert hollowed.sdf(np.array([[0., 0., 0.]]))[0] > 0

    def test_mesh_union_mesh(self, tmp_path):
        def _stl(name, **kw):
            p = tmp_path / name
            p.write_bytes(_write_binary_stl(_make_box_triangles(**kw)))
            return p
        a = stl_to_geometry(_stl("a.stl"))
        b = stl_to_geometry(_stl("b.stl")).translate(0.6, 0., 0.)
        assert a.union(b).sdf(np.array([[0.6, 0., 0.]]))[0] < 0

    def test_mesh_subtract_mesh(self, tmp_path):
        def _stl(name, **kw):
            p = tmp_path / name
            p.write_bytes(_write_binary_stl(_make_box_triangles(**kw)))
            return p
        base   = stl_to_geometry(_stl("base.stl"))
        cutter = stl_to_geometry(_stl("cutter.stl", hx=0.3, hy=0.3, hz=0.3))
        assert base.subtract(cutter).sdf(np.array([[0., 0., 0.]]))[0] > 0

    def test_ray_dir_warns(self, tmp_path):
        stl = tmp_path / "box.stl"
        stl.write_bytes(_write_binary_stl(_make_box_triangles()))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            stl_to_geometry(stl, ray_dir=np.array([0., 0., 1.]))
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "ray_dir" in str(w[0].message)

    @pytest.mark.parametrize("stem", ["orion_plug", "mars_wheel"])
    def test_real_stl(self, stem):
        stl_path = Path(__file__).parent.parent / "examples" / "stl2sdf" / f"{stem}.stl"
        if not stl_path.exists():
            pytest.skip(f"{stem}.stl not present")
        geom = stl_to_geometry(stl_path)
        assert isinstance(geom, Geometry3D)
        assert geom.sdf(np.array([[0., 0., 0.]])).shape == (1,)
