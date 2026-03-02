"""Internal STL loading utilities for stl2sdf.

All symbols here are private (underscore-prefixed).  Users should import
only from :mod:`stl2sdf.geometry`.

STL loading supports both binary and ASCII formats.  Detection uses the
binary-size invariant ``len(raw) == 84 + 50*F`` rather than the ``"solid"``
keyword, which some CAD tools (e.g. SolidWorks) also write at the start of
binary files.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Union

import numpy as np


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
