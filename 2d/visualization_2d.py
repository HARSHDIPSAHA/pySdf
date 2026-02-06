"""Visualization utilities for 2D SDFs."""

import os
import numpy as np


def save_levelset_html_2d(phi, bounds=(-1, 1), filename="levelset_2d.html"):
    """
    Save a 2D level set field as an interactive HTML visualization.

    This function generates an interactive 2D contour visualization of the
    signed distance field using plotly, showing the zero level set and
    the field values as a heatmap.

    Parameters
    ----------
    phi : ndarray
        2D numpy array containing the signed distance field values.
        Shape should be (nx, ny).
    bounds : tuple or float, optional
        Physical domain bounds. Can be:
        - A single tuple (lo, hi) for uniform bounds in both dimensions
        - A tuple of tuples ((xlo, xhi), (ylo, yhi)) for per-axis bounds
        Default is (-1, 1) for both axes.
    filename : str, optional
        Output HTML file path. Default is "levelset_2d.html".
        Parent directories will be created if they don't exist.

    Returns
    -------
    None
        The function saves the HTML file to disk and prints the output path.

    Raises
    ------
    ImportError
        If plotly is not installed.
    ValueError
        If phi is not a 2D array or bounds format is invalid.

    Examples
    --------
    Basic usage with uniform bounds:

    >>> from sdf2d import Circle, sample_levelset_2d, save_levelset_html_2d
    >>> circle = Circle(0.3)
    >>> phi = sample_levelset_2d(circle, ((-1, 1), (-1, 1)), (256, 256))
    >>> save_levelset_html_2d(phi, bounds=(-1, 1), filename="circle.html")

    With per-axis bounds:

    >>> phi = sample_levelset_2d(circle, ((-2, 2), (-1, 1)), (512, 256))
    >>> save_levelset_html_2d(phi, bounds=((-2, 2), (-1, 1)),
    ...                        filename="outputs/circle_stretched.html")

    With complex geometry:

    >>> from sdf2d import Circle, Box2D, Union2D, sample_levelset_2d, save_levelset_html_2d
    >>> a = Circle(0.25).translate(-0.3, 0)
    >>> b = Box2D((0.2, 0.2)).translate(0.3, 0)
    >>> combined = Union2D(a, b)
    >>> phi = sample_levelset_2d(combined, ((-1, 1), (-1, 1)), (512, 512))
    >>> save_levelset_html_2d(phi, bounds=(-1, 1), filename="combined_2d.html")
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "save_levelset_html_2d requires plotly. "
            "Install with: pip install plotly\n"
            "Or install all visualization dependencies with: pip install -e .[viz]"
        ) from e

    # Validate input
    if not isinstance(phi, np.ndarray):
        raise ValueError("phi must be a numpy array")

    if phi.ndim != 2:
        raise ValueError(f"phi must be a 2D array, got shape {phi.shape}")

    # Parse bounds
    if isinstance(bounds, (tuple, list)):
        if len(bounds) == 2 and not isinstance(bounds[0], (tuple, list)):
            # Single tuple (lo, hi) - apply to both axes
            lo, hi = bounds
            bounds = ((lo, hi), (lo, hi))
        elif len(bounds) == 2 and all(
            isinstance(b, (tuple, list)) and len(b) == 2 for b in bounds
        ):
            # Per-axis bounds ((xlo, xhi), (ylo, yhi))
            pass
        else:
            raise ValueError(
                "bounds must be either (lo, hi) or ((xlo, xhi), (ylo, yhi))"
            )
    else:
        raise ValueError("bounds must be a tuple")

    # Extract bounds for each axis
    (xlo, xhi), (ylo, yhi) = bounds

    # Create coordinate arrays
    nx, ny = phi.shape
    x = np.linspace(xlo, xhi, nx)
    y = np.linspace(ylo, yhi, ny)

    # Create figure with heatmap and contour
    fig = go.Figure()

    # Add heatmap of SDF values
    fig.add_trace(
        go.Heatmap(
            z=phi.T,  # Transpose for correct orientation
            x=x,
            y=y,
            colorscale="RdBu_r",  # Red-Blue diverging colormap
            zmid=0,  # Center colormap at zero
            colorbar=dict(
                title=dict(text="SDF Value", side="right"),
                tickmode="linear",
                tick0=-phi.max(),
                dtick=phi.max() / 5,
            ),
            hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<br>SDF: %{z:.3f}<extra></extra>",
        )
    )

    # Add contour lines, highlighting the zero level set
    fig.add_trace(
        go.Contour(
            z=phi.T,
            x=x,
            y=y,
            contours=dict(
                start=-max(abs(phi.min()), abs(phi.max())),
                end=max(abs(phi.min()), abs(phi.max())),
                size=max(abs(phi.min()), abs(phi.max())) / 10,
                showlabels=False,
            ),
            line=dict(width=1, color="rgba(0,0,0,0.3)"),
            showscale=False,
            hoverinfo="skip",
        )
    )

    # Add bold zero level set contour
    fig.add_trace(
        go.Contour(
            z=phi.T,
            x=x,
            y=y,
            contours=dict(
                start=0,
                end=0,
                size=1,
                showlabels=True,
                labelfont=dict(size=12, color="black"),
            ),
            line=dict(width=3, color="black"),
            showscale=False,
            name="Zero Level Set",
            hovertemplate="Zero level set<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title="2D Level Set Visualization",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis=dict(scaleanchor="y", scaleratio=1),  # Equal aspect ratio
        yaxis=dict(constrain="domain"),
        margin=dict(l=0, r=100, b=0, t=40),
        hovermode="closest",
    )

    # Create output directory if needed
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Save HTML file
    fig.write_html(filename)
    print(f"✅ 2D level set visualization saved to: {filename}")
