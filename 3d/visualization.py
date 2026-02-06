"""Visualization utilities for 3D SDFs."""

import os
import numpy as np


def save_levelset_html(phi, bounds=(-1, 1), filename="levelset.html"):
    """
    Save a 3D level set field as an interactive HTML visualization.

    This function generates an interactive 3D mesh visualization of the zero
    level set (SDF=0 isosurface) using plotly and scikit-image marching cubes.

    Parameters
    ----------
    phi : ndarray
        3D numpy array containing the signed distance field values.
        Shape should be (nx, ny, nz).
    bounds : tuple or float, optional
        Physical domain bounds. Can be:
        - A single tuple (lo, hi) for uniform bounds in all dimensions
        - A tuple of tuples ((xlo, xhi), (ylo, yhi), (zlo, zhi)) for per-axis bounds
        Default is (-1, 1) for all axes.
    filename : str, optional
        Output HTML file path. Default is "levelset.html".
        Parent directories will be created if they don't exist.

    Returns
    -------
    None
        The function saves the HTML file to disk and prints the output path.

    Raises
    ------
    ImportError
        If plotly or scikit-image are not installed.
    ValueError
        If phi is not a 3D array or bounds format is invalid.

    Examples
    --------
    Basic usage with uniform bounds:

    >>> from sdf3d import Sphere, sample_levelset, save_levelset_html
    >>> sphere = Sphere(0.3)
    >>> phi = sample_levelset(sphere, ((-1, 1), (-1, 1), (-1, 1)), (64, 64, 64))
    >>> save_levelset_html(phi, bounds=(-1, 1), filename="sphere.html")

    With per-axis bounds:

    >>> phi = sample_levelset(sphere, ((-2, 2), (-1, 1), (-1, 1)), (128, 64, 64))
    >>> save_levelset_html(phi, bounds=((-2, 2), (-1, 1), (-1, 1)),
    ...                     filename="outputs/sphere_stretched.html")

    With complex geometry:

    >>> from sdf3d import Sphere, Box, Union, sample_levelset, save_levelset_html
    >>> a = Sphere(0.25).translate(-0.3, 0, 0)
    >>> b = Box((0.2, 0.2, 0.2)).translate(0.3, 0, 0)
    >>> combined = Union(a, b)
    >>> phi = sample_levelset(combined, ((-1, 1), (-1, 1), (-1, 1)), (128, 128, 128))
    >>> save_levelset_html(phi, bounds=(-1, 1), filename="combined.html")
    """
    try:
        from skimage import measure
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "save_levelset_html requires plotly and scikit-image. "
            "Install with: pip install plotly scikit-image\n"
            "Or install all visualization dependencies with: pip install -e .[viz]"
        ) from e

    # Validate input
    if not isinstance(phi, np.ndarray):
        raise ValueError("phi must be a numpy array")

    if phi.ndim != 3:
        raise ValueError(f"phi must be a 3D array, got shape {phi.shape}")

    # Parse bounds
    if isinstance(bounds, (tuple, list)):
        if len(bounds) == 2 and not isinstance(bounds[0], (tuple, list)):
            # Single tuple (lo, hi) - apply to all axes
            lo, hi = bounds
            bounds = ((lo, hi), (lo, hi), (lo, hi))
        elif len(bounds) == 3 and all(
            isinstance(b, (tuple, list)) and len(b) == 2 for b in bounds
        ):
            # Per-axis bounds ((xlo, xhi), (ylo, yhi), (zlo, zhi))
            pass
        else:
            raise ValueError(
                "bounds must be either (lo, hi) or ((xlo, xhi), (ylo, yhi), (zlo, zhi))"
            )
    else:
        raise ValueError("bounds must be a tuple")

    # Extract bounds for each axis
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds

    # Calculate spacing for marching cubes
    nx, ny, nz = phi.shape
    spacing = ((xhi - xlo) / nx, (yhi - ylo) / ny, (zhi - zlo) / nz)

    # Extract isosurface using marching cubes
    try:
        verts, faces, _, _ = measure.marching_cubes(phi, level=0.0, spacing=spacing)
    except (ValueError, RuntimeError) as e:
        # Handle case where no surface is found
        print(f"⚠️  Warning: Could not extract isosurface (level=0): {e}")
        print("    The field may not contain a zero crossing.")
        print("    Creating empty visualization.")
        verts = np.array([[0, 0, 0]])
        faces = np.array([[0, 0, 0]])

    # Adjust vertices to physical coordinates
    verts[:, 0] += xlo
    verts[:, 1] += ylo
    verts[:, 2] += zlo

    # Filter out small disconnected fragments (optional, for cleaner output)
    if len(verts) > 10 and len(faces) > 10:
        vertex_face_count = np.zeros(len(verts))
        for face in faces:
            vertex_face_count[face] += 1

        # Keep vertices that appear in at least 2 faces (removes tiny fragments)
        # Adaptive threshold based on the mesh density
        MIN_FACE_RATIO = 0.1  # Minimum ratio of faces per vertex
        min_faces = max(2, int(len(faces) / len(verts) * MIN_FACE_RATIO))
        valid_vertices = vertex_face_count >= min_faces

        # Only apply filtering if we retain a significant portion of vertices (>30%)
        MIN_VERTEX_RETENTION = 0.3  # Keep at least 30% of original vertices
        if valid_vertices.sum() > len(verts) * MIN_VERTEX_RETENTION:
            # Remap vertex indices
            vertex_map = np.full(len(verts), -1, dtype=int)
            new_idx = 0
            for i, valid in enumerate(valid_vertices):
                if valid:
                    vertex_map[i] = new_idx
                    new_idx += 1

            # Filter faces: keep only faces where all vertices are valid
            valid_faces = valid_vertices[faces].all(axis=1)
            if valid_faces.sum() > 0:  # Only apply if we have valid faces
                faces = faces[valid_faces]

                # Remap vertex indices in faces
                faces = np.array([[vertex_map[v] for v in face] for face in faces])

                # Filter vertices
                verts = verts[valid_vertices]

    # Create plotly mesh
    i, j, k = faces.T

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=i,
                j=j,
                k=k,
                opacity=1.0,
                color="limegreen",
                flatshading=True,
            )
        ]
    )

    # Update layout
    fig.update_layout(
        title="Level Set Visualization (SDF=0 isosurface)",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="cube"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # Create output directory if needed
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Save HTML file
    fig.write_html(filename)
    print(f"✅ Level set visualization saved to: {filename}")
