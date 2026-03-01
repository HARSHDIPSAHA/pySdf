"""STL mesh → Signed Distance Field.

No pip dependencies beyond numpy itself.

Algorithm overview
------------------
Unsigned distance: Christer Ericson's Voronoi-region closest-point method
    (Real-Time Collision Detection §5.1.5).  For each triangle we compute
    6 dot products d1–d6 and 3 cross-term determinants va/vb/vc that
    identify one of 7 regions (3 vertex caps, 3 edge strips, 1 interior).
    ``np.select`` picks the correct formula; all branch denominators are
    guarded with ``np.maximum(..., 1e-30)`` because np.select evaluates
    every branch before selecting.

Sign determination: Möller–Trumbore ray casting.
    A ray is cast from each query point in a fixed irrational direction
    ``_RAY_DIR``.  Per triangle we solve for the intersection in O(1)
    scalar ops (vectorised over all N query points).  Odd hit count →
    inside (phi < 0).  Requires a **watertight** mesh; sign is undefined
    near gaps.

Complexity: O(F × N) where F = number of triangles, N = number of query
points.  Use ``--res 20`` for quick tests; production runs benefit from
BVH acceleration (not implemented here).
"""

from __future__ import annotations

import struct
from math import sqrt
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

_Bounds3D = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
_Resolution3D = Tuple[int, int, int]

# ---------------------------------------------------------------------------
# Fixed ray direction — irrational components avoid axis-aligned degeneracies
# ---------------------------------------------------------------------------
_RAY_DIR: np.ndarray = np.array(
    [sqrt(2) - 1.0, sqrt(3) - 1.0, 1.0 / sqrt(3)], dtype=np.float64
)
_RAY_DIR = _RAY_DIR / np.linalg.norm(_RAY_DIR)


# ---------------------------------------------------------------------------
# STL loading
# ---------------------------------------------------------------------------

def load_stl(path: Union[str, Path]) -> np.ndarray:
    """Load an STL file and return its triangles as a ``(F, 3, 3)`` float64 array.

    Supports both binary and ASCII STL.  Normals are discarded; only vertex
    coordinates are returned.

    Parameters
    ----------
    path:
        Path to the ``.stl`` file.

    Returns
    -------
    numpy.ndarray
        Shape ``(F, 3, 3)`` where ``triangles[i, j]`` is the j-th vertex of
        the i-th triangle as ``(x, y, z)``.
    """
    path = Path(path)
    raw = path.read_bytes()

    # Prefer the binary-size invariant: a valid binary STL satisfies
    # len(raw) == 84 + 50 * triangle_count.  This correctly handles binary
    # files whose 80-byte header happens to start with "solid" (produced by
    # some CAD tools such as SolidWorks), which would fool a keyword-only check.
    if len(raw) >= 84:
        count = struct.unpack_from("<I", raw, 80)[0]
        if len(raw) == 84 + 50 * count:
            return _load_binary_stl(raw)
    return _load_ascii_stl(raw.decode("ascii", errors="replace"))


def _load_binary_stl(raw: bytes) -> np.ndarray:
    """Parse a binary STL bytestring."""
    # Header: 80 bytes; triangle count: uint32 at offset 80
    count = struct.unpack_from("<I", raw, 80)[0]
    # Each record: 12 bytes normal + 36 bytes vertices + 2 bytes attr = 50 bytes
    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attr", "<u2"),
    ])
    records = np.frombuffer(raw, dtype=dtype, count=count, offset=84)
    return records["vertices"].astype(np.float64)  # (F, 3, 3)


def _load_ascii_stl(text: str) -> np.ndarray:
    """Parse an ASCII STL string."""
    verts: list[list[float]] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("vertex"):
            parts = line.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    arr = np.array(verts, dtype=np.float64)  # (F*3, 3)
    return arr.reshape(-1, 3, 3)


# ---------------------------------------------------------------------------
# Unsigned distance — Ericson Voronoi-region method
# ---------------------------------------------------------------------------

def _sq_dist_to_tri(P: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Squared distance from each point in *P* to triangle *tri*.

    Parameters
    ----------
    P:
        ``(N, 3)`` query points.
    tri:
        ``(3, 3)`` triangle vertices ``[A, B, C]``.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` squared distances.
    """
    A, B, C = tri[0], tri[1], tri[2]
    AB = B - A           # (3,)
    AC = C - A           # (3,)
    AP = P - A           # (N, 3)

    d1 = AP @ AB         # (N,)
    d2 = AP @ AC         # (N,)
    d3 = (P - B) @ AB    # (N,)
    d4 = (P - B) @ AC    # (N,)
    d5 = (P - C) @ AB    # (N,) — not AP@AB+... because we need (P-C)
    d6 = (P - C) @ AC    # (N,)

    # Cross-term determinants for Voronoi region test
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    denom_uv  = np.maximum(va + vb + vc, 1e-30)
    denom_u   = np.maximum(d1 - d3, 1e-30)
    denom_v   = np.maximum((d4 - d3) + (d5 - d6), 1e-30)  # for edge BC: Ericson §5.1.5

    # Region conditions (vertex regions tested first to avoid overlap)
    cond_A  = (d1 <= 0.0) & (d2 <= 0.0)
    cond_B  = (d3 >= 0.0) & (d4 <= d3)
    cond_C  = (d6 >= 0.0) & (d5 <= d6)
    cond_AB = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    cond_AC = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    cond_BC = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    # Interior: fallthrough

    # Closest-point per region
    def _sq(cp):
        diff = P - cp
        return (diff * diff).sum(axis=-1)

    # AB edge
    t_AB  = np.clip(d1 / denom_u, 0.0, 1.0)
    cp_AB = A + t_AB[:, None] * AB

    # AC edge
    t_AC  = np.clip(d2 / np.maximum(d2 - d6, 1e-30), 0.0, 1.0)
    cp_AC = A + t_AC[:, None] * AC

    # BC edge
    t_BC  = np.clip((d4 - d3) / denom_v, 0.0, 1.0)
    cp_BC = B + t_BC[:, None] * (C - B)

    # Interior (barycentric)
    w_v = vb / denom_uv
    w_w = vc / denom_uv
    cp_int = A + np.clip(w_v, 0.0, 1.0)[:, None] * AC + np.clip(w_w, 0.0, 1.0)[:, None] * AB

    sq = np.select(
        [cond_A, cond_B, cond_C, cond_AB, cond_AC, cond_BC],
        [_sq(A), _sq(B), _sq(C), _sq(cp_AB), _sq(cp_AC), _sq(cp_BC)],
        default=_sq(cp_int),
    )
    return sq


# ---------------------------------------------------------------------------
# Sign — Möller–Trumbore ray casting
# ---------------------------------------------------------------------------

def _mt_ray_hits(
    P: np.ndarray,
    ray_dir: np.ndarray,
    tri: np.ndarray,
) -> np.ndarray:
    """Count Möller–Trumbore ray-triangle intersections for each query point.

    Parameters
    ----------
    P:
        ``(N, 3)`` ray origins.
    ray_dir:
        ``(3,)`` unit ray direction (same for all origins).
    tri:
        ``(3, 3)`` triangle vertices ``[v0, v1, v2]``.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` int32 — 1 where the ray hits, 0 otherwise.
    """
    v0, v1, v2 = tri[0], tri[1], tri[2]
    e1 = v1 - v0   # (3,)
    e2 = v2 - v0   # (3,)

    h   = np.cross(ray_dir, e2)   # (3,)
    det = float(e1 @ h)           # scalar

    if abs(det) < 1e-10:
        return np.zeros(len(P), dtype=np.int32)

    inv_det = 1.0 / det
    s = P - v0                           # (N, 3)
    u = inv_det * (s @ h)                # (N,)

    q = np.cross(s, e1)                  # (N, 3)  — broadcasts (N,3) × (3,) → (N,3) ✓
    v = inv_det * (q @ ray_dir)          # (N,)
    t = inv_det * (q @ e2)               # (N,)

    hit = (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t > 1e-10)
    return hit.astype(np.int32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mesh_to_sdf(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    ray_dir: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the signed distance field at *points* for a triangulated mesh.

    Parameters
    ----------
    points:
        ``(N, 3)`` query point coordinates.
    triangles:
        ``(F, 3, 3)`` triangle vertex array from :func:`load_stl`.
    ray_dir:
        Optional ``(3,)`` unit ray direction for sign determination.
        Defaults to the built-in irrational direction ``_RAY_DIR``.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` signed distances.  Negative inside, positive outside.
        Requires a **watertight** mesh; sign is undefined near gaps.

    Notes
    -----
    Time complexity is O(F × N).  For a 14 K-triangle mesh and a 40³ grid
    (~64 K points) this is ~900 M ops and takes 2–5 min on a single CPU
    core.  Use a coarser resolution for quick tests.
    """
    P = np.asarray(points, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.float64)
    if ray_dir is None:
        ray_dir = _RAY_DIR
    else:
        ray_dir = np.asarray(ray_dir, dtype=np.float64)
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

    N = len(P)
    sq_min = np.full(N, np.inf)
    hits   = np.zeros(N, dtype=np.int32)

    for tri in tris:
        sq = _sq_dist_to_tri(P, tri)
        sq_min = np.minimum(sq_min, sq)
        hits  += _mt_ray_hits(P, ray_dir, tri)

    unsigned = np.sqrt(np.maximum(sq_min, 0.0))
    sign     = np.where(hits % 2 == 1, -1.0, 1.0)
    return sign * unsigned


def sample_sdf_from_stl(
    path: Union[str, Path],
    bounds: _Bounds3D,
    resolution: _Resolution3D,
    *,
    ray_dir: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Load an STL file and sample its SDF on a uniform cell-centred grid.

    The grid layout is **identical** to :func:`sdf3d.grid.sample_levelset_3d`:
    cell centres with z-first (nz, ny, nx) output shape.

    Parameters
    ----------
    path:
        Path to the ``.stl`` file.
    bounds:
        ``((x0, x1), (y0, y1), (z0, z1))`` physical extents of the domain.
    resolution:
        ``(nx, ny, nz)`` number of cells along each axis.
    ray_dir:
        Optional ray direction override; see :func:`mesh_to_sdf`.

    Returns
    -------
    numpy.ndarray
        Shape ``(nz, ny, nx)`` signed distance field.
    """
    triangles = load_stl(path)

    (x0, x1), (y0, y1), (z0, z1) = bounds
    nx, ny, nz = resolution

    xs = np.linspace(x0, x1, nx, endpoint=False) + (x1 - x0) / (2.0 * nx)
    ys = np.linspace(y0, y1, ny, endpoint=False) + (y1 - y0) / (2.0 * ny)
    zs = np.linspace(z0, z1, nz, endpoint=False) + (z1 - z0) / (2.0 * nz)

    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    phi = mesh_to_sdf(P, triangles, ray_dir=ray_dir)
    return phi.reshape(nz, ny, nx)
