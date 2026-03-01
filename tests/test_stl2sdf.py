"""Tests for stl2sdf — STL mesh to Signed Distance Field.

All tests are synthetic (no external file downloads).
A closed unit box (12 triangles) is used as the reference mesh throughout.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from stl2sdf import load_stl, mesh_to_sdf, sample_sdf_from_stl
from stl2sdf.mesh_sdf import _mt_ray_hits, _sq_dist_to_tri, _RAY_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_box_triangles(hx: float = 0.5, hy: float = 0.5, hz: float = 0.5) -> np.ndarray:
    """Return 12 triangles forming a closed axis-aligned box [-hx,hx]×[-hy,hy]×[-hz,hz].

    Normals point outward by right-hand rule.  The mesh is watertight.
    """
    # 8 corner vertices
    verts = np.array([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ], dtype=np.float64)

    # Each face split into 2 triangles (CCW from outside)
    face_indices = [
        # -Z face
        (0, 2, 1), (0, 3, 2),
        # +Z face
        (4, 5, 6), (4, 6, 7),
        # -X face
        (0, 4, 7), (0, 7, 3),
        # +X face
        (1, 2, 6), (1, 6, 5),
        # -Y face
        (0, 1, 5), (0, 5, 4),
        # +Y face
        (3, 7, 6), (3, 6, 2),
    ]

    tris = np.array([[verts[i], verts[j], verts[k]] for i, j, k in face_indices],
                    dtype=np.float64)
    return tris  # (12, 3, 3)


def _write_binary_stl(triangles: np.ndarray) -> bytes:
    """Encode triangles as a valid binary STL bytestring."""
    header = b"\x00" * 80
    count  = struct.pack("<I", len(triangles))
    records = bytearray()
    for tri in triangles:
        # normal (placeholder zero)
        records += struct.pack("<fff", 0.0, 0.0, 0.0)
        for v in tri:
            records += struct.pack("<fff", float(v[0]), float(v[1]), float(v[2]))
        records += struct.pack("<H", 0)  # attr
    return header + count + bytes(records)


def _write_ascii_stl(triangles: np.ndarray) -> str:
    """Encode triangles as ASCII STL text."""
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
# TestLoadSTL
# ---------------------------------------------------------------------------

class TestLoadSTL:
    def test_binary_shape(self, tmp_path):
        tris = _make_box_triangles()
        raw  = _write_binary_stl(tris)
        stl_file = tmp_path / "box.stl"
        stl_file.write_bytes(raw)

        loaded = load_stl(stl_file)
        assert loaded.shape == (12, 3, 3)
        assert loaded.dtype == np.float64

    def test_binary_values(self, tmp_path):
        tris = _make_box_triangles()
        raw  = _write_binary_stl(tris)
        stl_file = tmp_path / "box.stl"
        stl_file.write_bytes(raw)

        loaded = load_stl(stl_file)
        # Values should round-trip to float32 precision
        np.testing.assert_allclose(loaded, tris, atol=1e-6)

    def test_ascii_shape(self, tmp_path):
        tris = _make_box_triangles()
        text = _write_ascii_stl(tris)
        stl_file = tmp_path / "box.stl"
        stl_file.write_text(text)

        loaded = load_stl(stl_file)
        assert loaded.shape == (12, 3, 3)
        assert loaded.dtype == np.float64

    def test_ascii_values(self, tmp_path):
        tris = _make_box_triangles()
        text = _write_ascii_stl(tris)
        stl_file = tmp_path / "box.stl"
        stl_file.write_text(text)

        loaded = load_stl(stl_file)
        np.testing.assert_allclose(loaded, tris, atol=1e-5)


# ---------------------------------------------------------------------------
# TestSqDistToTri — 7 Voronoi regions
# ---------------------------------------------------------------------------

class TestSqDistToTri:
    """Unit tests covering all 7 Voronoi regions of _sq_dist_to_tri."""

    def setup_method(self):
        # Right-angle triangle: A=(0,0,0), B=(1,0,0), C=(0,1,0)
        self.tri = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float64)

    def _sq(self, p):
        return _sq_dist_to_tri(np.atleast_2d(p).astype(np.float64), self.tri)

    # --- vertex regions ---
    def test_vertex_A(self):
        # Behind A along both AB and AC
        p = np.array([-1.0, -1.0, 0.0])
        sq = self._sq(p)
        expected = 1**2 + 1**2
        np.testing.assert_allclose(sq, [expected], atol=1e-10)

    def test_vertex_B(self):
        # Beyond B along AB, behind along AC
        p = np.array([2.0, -1.0, 0.0])
        sq = self._sq(p)
        expected = 1**2 + 1**2
        np.testing.assert_allclose(sq, [expected], atol=1e-10)

    def test_vertex_C(self):
        p = np.array([-1.0, 2.0, 0.0])
        sq = self._sq(p)
        expected = 1**2 + 1**2
        np.testing.assert_allclose(sq, [expected], atol=1e-10)

    # --- edge regions ---
    def test_edge_AB(self):
        # Directly above midpoint of AB
        p = np.array([0.5, -1.0, 0.0])
        sq = self._sq(p)
        np.testing.assert_allclose(sq, [1.0], atol=1e-10)

    def test_edge_AC(self):
        # Directly to the left of midpoint of AC
        p = np.array([-1.0, 0.5, 0.0])
        sq = self._sq(p)
        np.testing.assert_allclose(sq, [1.0], atol=1e-10)

    def test_edge_BC(self):
        # Point beyond edge BC midpoint (0.5, 0.5, 0)
        p = np.array([0.8, 0.8, 0.0])
        sq = self._sq(p)
        # Closest point on BC: parameterise B+(t)(C-B), project p
        BC = np.array([-1.0, 1.0, 0.0])
        t  = np.clip(np.dot(p - np.array([1,0,0]), BC) / np.dot(BC,BC), 0, 1)
        cp = np.array([1,0,0]) + t * BC
        expected = np.sum((p - cp)**2)
        np.testing.assert_allclose(sq, [expected], atol=1e-10)

    # --- interior ---
    def test_interior(self):
        # Centroid projected onto triangle plane should give distance = z²
        p = np.array([0.2, 0.2, 0.5])
        sq = self._sq(p)
        # Closest point is (0.2, 0.2, 0) — inside triangle
        np.testing.assert_allclose(sq, [0.25], atol=1e-10)

    def test_batch(self):
        # Batch mode: multiple points at once
        P = np.array([[-1,-1,0], [2,-1,0], [0.2,0.2,0.5]], dtype=np.float64)
        sq = _sq_dist_to_tri(P, self.tri)
        assert sq.shape == (3,)
        assert sq[0] == pytest.approx(2.0, abs=1e-10)
        assert sq[1] == pytest.approx(2.0, abs=1e-10)
        assert sq[2] == pytest.approx(0.25, abs=1e-10)


# ---------------------------------------------------------------------------
# TestMTRayHits
# ---------------------------------------------------------------------------

class TestMTRayHits:
    def setup_method(self):
        # Triangle in XY plane: A=(0,0,0), B=(2,0,0), C=(0,2,0)
        self.tri = np.array([[0,0,0],[2,0,0],[0,2,0]], dtype=np.float64)
        self.ray_dir = np.array([0.0, 0.0, 1.0])

    def test_hit_interior(self):
        # Ray origin below triangle interior, pointing +Z
        P = np.array([[0.5, 0.5, -1.0]])
        hits = _mt_ray_hits(P, self.ray_dir, self.tri)
        assert hits[0] == 1

    def test_miss_exterior(self):
        # Ray origin below but outside triangle footprint
        P = np.array([[3.0, 3.0, -1.0]])
        hits = _mt_ray_hits(P, self.ray_dir, self.tri)
        assert hits[0] == 0

    def test_behind_triangle(self):
        # Ray origin above triangle pointing +Z — t < 0, should miss
        P = np.array([[0.5, 0.5, 1.0]])
        hits = _mt_ray_hits(P, self.ray_dir, self.tri)
        assert hits[0] == 0

    def test_parallel_ray(self):
        # Ray parallel to triangle plane — det ≈ 0, no hit
        ray_parallel = np.array([1.0, 0.0, 0.0])
        P = np.array([[0.5, 0.5, -1.0]])
        hits = _mt_ray_hits(P, ray_parallel, self.tri)
        assert hits[0] == 0

    def test_batch(self):
        P = np.array([
            [0.5, 0.5, -1.0],   # inside
            [3.0, 3.0, -1.0],   # outside
            [0.5, 0.5,  1.0],   # behind
        ])
        hits = _mt_ray_hits(P, self.ray_dir, self.tri)
        assert list(hits) == [1, 0, 0]


# ---------------------------------------------------------------------------
# TestMeshToSDF
# ---------------------------------------------------------------------------

class TestMeshToSDF:
    def setup_method(self):
        self.tris = _make_box_triangles(0.5, 0.5, 0.5)

    def test_inside_negative(self):
        # Centroid
        P = np.array([[0.0, 0.0, 0.0]])
        phi = mesh_to_sdf(P, self.tris)
        assert phi[0] < 0.0, f"Expected phi<0 at origin, got {phi[0]}"

    def test_outside_positive(self):
        # Far outside
        P = np.array([[2.0, 2.0, 2.0]])
        phi = mesh_to_sdf(P, self.tris)
        assert phi[0] > 0.0, f"Expected phi>0 far outside, got {phi[0]}"

    def test_on_face_approx_zero(self):
        # Face centre of +X face: x=0.5, y=0, z=0
        P = np.array([[0.5, 0.0, 0.0]])
        phi = mesh_to_sdf(P, self.tris)
        np.testing.assert_allclose(phi, [0.0], atol=1e-6)

    def test_inside_distance_magnitude(self):
        # Point at (0.3, 0, 0) — distance to +X face = 0.2
        P = np.array([[0.3, 0.0, 0.0]])
        phi = mesh_to_sdf(P, self.tris)
        np.testing.assert_allclose(np.abs(phi), [0.2], atol=1e-5)
        assert phi[0] < 0.0

    def test_outside_distance_magnitude(self):
        # Point at (1.0, 0, 0) — distance to +X face = 0.5
        P = np.array([[1.0, 0.0, 0.0]])
        phi = mesh_to_sdf(P, self.tris)
        np.testing.assert_allclose(phi, [0.5], atol=1e-5)

    def test_custom_ray_dir(self):
        # Deliberately off-axis, off-diagonal — avoids hitting box edges/vertices from origin
        P = np.array([[0.0, 0.0, 0.0]])
        ray = np.array([0.1, 0.3, 0.7])
        phi = mesh_to_sdf(P, self.tris, ray_dir=ray)
        assert phi[0] < 0.0

    def test_batch_sign_pattern(self):
        # Grid of points: inner cube should all be < 0, outer should be > 0
        inner = np.array([[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [0.0, -0.2, 0.1]])
        outer = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        phi_in  = mesh_to_sdf(inner, self.tris)
        phi_out = mesh_to_sdf(outer, self.tris)
        assert np.all(phi_in  < 0), f"Expected all inner points negative: {phi_in}"
        assert np.all(phi_out > 0), f"Expected all outer points positive: {phi_out}"


# ---------------------------------------------------------------------------
# TestSampleSdfFromStl
# ---------------------------------------------------------------------------

class TestSampleSdfFromStl:
    def setup_method(self):
        self.tris = _make_box_triangles(0.5, 0.5, 0.5)

    def _write_stl(self, tmp_path) -> Path:
        raw = _write_binary_stl(self.tris)
        p   = tmp_path / "box.stl"
        p.write_bytes(raw)
        return p

    def test_shape_cubic(self, tmp_path):
        stl = self._write_stl(tmp_path)
        phi = sample_sdf_from_stl(stl, ((-1,1),(-1,1),(-1,1)), (4, 4, 4))
        assert phi.shape == (4, 4, 4)

    def test_shape_non_cubic(self, tmp_path):
        stl = self._write_stl(tmp_path)
        phi = sample_sdf_from_stl(stl, ((-1,1),(-1,1),(-1,1)), (3, 5, 7))
        assert phi.shape == (7, 5, 3)  # (nz, ny, nx)

    def test_matches_manual_call(self, tmp_path):
        stl = self._write_stl(tmp_path)
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        res    = (4, 4, 4)

        phi_func = sample_sdf_from_stl(stl, bounds, res)

        # Manually replicate the grid
        (x0,x1),(y0,y1),(z0,z1) = bounds
        nx, ny, nz = res
        xs = np.linspace(x0,x1,nx,endpoint=False) + (x1-x0)/(2.0*nx)
        ys = np.linspace(y0,y1,ny,endpoint=False) + (y1-y0)/(2.0*ny)
        zs = np.linspace(z0,z1,nz,endpoint=False) + (z1-z0)/(2.0*nz)
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
        P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        phi_manual = mesh_to_sdf(P, self.tris).reshape(nz, ny, nx)

        np.testing.assert_array_equal(phi_func, phi_manual)

    def test_interior_exterior_signs(self, tmp_path):
        stl = self._write_stl(tmp_path)
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        phi = sample_sdf_from_stl(stl, bounds, (8, 8, 8))
        # Cell centres of 8³ grid over [-1,1]³ with step 0.25; centres at ±0.125, ±0.375, ±0.625, ±0.875
        # Box occupies [-0.5, 0.5]³; cells with |x|,|y|,|z| ≤ 0.375 are interior
        # Just check some known-interior and known-exterior cells
        mid = phi[4, 4, 4]  # cell at (0.125, 0.125, 0.125) — inside
        corner = phi[0, 0, 0]  # near (-0.875, -0.875, -0.875) — outside
        assert mid < 0.0, f"Expected interior phi < 0, got {mid}"
        assert corner > 0.0, f"Expected exterior phi > 0, got {corner}"
