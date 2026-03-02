"""Internal SDF math for triangulated meshes.

All symbols here are private (underscore-prefixed).  Users should import
only from :mod:`stl2sdf.geometry`.

Algorithms
----------
Unsigned distance — Christer Ericson's Voronoi-region closest-point method
    (Real-Time Collision Detection §5.1.5).  Six dot products d1–d6 and three
    cross-term determinants va/vb/vc identify one of seven regions.
    ``np.select`` picks the formula; denominators are guarded with
    ``np.maximum(..., 1e-30)`` because np.select evaluates every branch.

Sign — Möller–Trumbore ray casting.
    A ray from each query point in a fixed irrational direction counts
    triangle crossings.  Odd count → inside (phi < 0).  Requires a
    watertight mesh; sign is undefined near gaps.

Complexity: O(F × N) where F = triangles, N = query points.
"""

from __future__ import annotations

import struct
from math import sqrt
from pathlib import Path
from typing import Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Fixed ray direction — irrational components avoid axis-aligned degeneracies
# ---------------------------------------------------------------------------
_RAY_DIR: np.ndarray = np.array(
    [sqrt(2) - 1.0, sqrt(3) - 1.0, 1.0 / sqrt(3)], dtype=np.float64
)
_RAY_DIR = _RAY_DIR / np.linalg.norm(_RAY_DIR)


# ---------------------------------------------------------------------------
# STL parsing
# ---------------------------------------------------------------------------

def _stl_to_triangles(path: Union[str, Path]) -> np.ndarray:
    """Read an STL file and return its triangles as a (F, 3, 3) float64 array.

    Supports binary and ASCII STL.  Normals are discarded.
    Detection uses the binary-size invariant (len == 84 + 50*F) rather than
    the "solid" keyword, which some CAD tools (e.g. SolidWorks) also write
    at the start of binary files.
    """
    path = Path(path)
    raw  = path.read_bytes()
    if len(raw) >= 84:
        count = struct.unpack_from("<I", raw, 80)[0]
        if len(raw) == 84 + 50 * count:
            return _binary_stl_to_triangles(raw)
    return _ascii_stl_to_triangles(raw.decode("ascii", errors="replace"))


def _binary_stl_to_triangles(raw: bytes) -> np.ndarray:
    count   = struct.unpack_from("<I", raw, 80)[0]
    dtype   = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    records = np.frombuffer(raw, dtype=dtype, count=count, offset=84)
    return records["vertices"].astype(np.float64)  # (F, 3, 3)


def _ascii_stl_to_triangles(text: str) -> np.ndarray:
    verts: list[list[float]] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("vertex"):
            parts = line.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float64).reshape(-1, 3, 3)


# ---------------------------------------------------------------------------
# Unsigned distance — Ericson Voronoi-region method
# ---------------------------------------------------------------------------

def _triangle_sq_dist(P: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Squared distance from each point in P (N, 3) to triangle tri (3, 3)."""
    A, B, C = tri[0], tri[1], tri[2]
    AB = B - A
    AC = C - A
    AP = P - A

    d1 = AP @ AB
    d2 = AP @ AC
    d3 = (P - B) @ AB
    d4 = (P - B) @ AC
    d5 = (P - C) @ AB
    d6 = (P - C) @ AC

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    denom_uv = np.maximum(va + vb + vc, 1e-30)
    denom_u  = np.maximum(d1 - d3, 1e-30)
    denom_v  = np.maximum((d4 - d3) + (d5 - d6), 1e-30)

    cond_A  = (d1 <= 0.0) & (d2 <= 0.0)
    cond_B  = (d3 >= 0.0) & (d4 <= d3)
    cond_C  = (d6 >= 0.0) & (d5 <= d6)
    cond_AB = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    cond_AC = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    cond_BC = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)

    def _sq(cp):
        diff = P - cp
        return (diff * diff).sum(axis=-1)

    t_AB  = np.clip(d1 / denom_u, 0.0, 1.0)
    cp_AB = A + t_AB[:, None] * AB

    t_AC  = np.clip(d2 / np.maximum(d2 - d6, 1e-30), 0.0, 1.0)
    cp_AC = A + t_AC[:, None] * AC

    t_BC  = np.clip((d4 - d3) / denom_v, 0.0, 1.0)
    cp_BC = B + t_BC[:, None] * (C - B)

    w_v    = vb / denom_uv
    w_w    = vc / denom_uv
    cp_int = A + np.clip(w_v, 0.0, 1.0)[:, None] * AC + np.clip(w_w, 0.0, 1.0)[:, None] * AB

    return np.select(
        [cond_A, cond_B, cond_C, cond_AB, cond_AC, cond_BC],
        [_sq(A), _sq(B), _sq(C), _sq(cp_AB), _sq(cp_AC), _sq(cp_BC)],
        default=_sq(cp_int),
    )


# ---------------------------------------------------------------------------
# Sign — Möller–Trumbore ray casting
# ---------------------------------------------------------------------------

def _ray_triangle_hits(P: np.ndarray, ray_dir: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Return (N,) int32: 1 where the ray from each point hits tri, 0 otherwise."""
    v0, v1, v2 = tri[0], tri[1], tri[2]
    e1  = v1 - v0
    e2  = v2 - v0
    h   = np.cross(ray_dir, e2)
    det = float(e1 @ h)

    if abs(det) < 1e-10:
        return np.zeros(len(P), dtype=np.int32)

    inv_det = 1.0 / det
    s = P - v0
    u = inv_det * (s @ h)
    q = np.cross(s, e1)
    v = inv_det * (q @ ray_dir)
    t = inv_det * (q @ e2)

    hit = (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t > 1e-10)
    return hit.astype(np.int32)


# ---------------------------------------------------------------------------
# Combined: unsigned distance + sign
# ---------------------------------------------------------------------------

def _triangles_to_sdf(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    ray_dir: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Signed distances from (N, 3) points to a triangle mesh (F, 3, 3).

    Returns (N,) phi: negative inside, positive outside.
    Requires a watertight mesh; O(F × N).
    """
    P    = np.asarray(points,    dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.float64)
    if ray_dir is None:
        ray_dir = _RAY_DIR
    else:
        ray_dir = np.asarray(ray_dir, dtype=np.float64)
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

    sq_min = np.full(len(P), np.inf)
    hits   = np.zeros(len(P), dtype=np.int32)
    for tri in tris:
        sq_min = np.minimum(sq_min, _triangle_sq_dist(P, tri))
        hits  += _ray_triangle_hits(P, ray_dir, tri)

    return np.where(hits % 2 == 1, -1.0, 1.0) * np.sqrt(np.maximum(sq_min, 0.0))
