import argparse
import os

import numpy as np

try:
    import yt
except Exception as exc:  # pragma: no cover - runtime environment specific
    raise SystemExit("yt not available (pip install yt)") from exc


def _get_surface_api():
    try:
        from yt.visualization.volume_rendering.api import Scene, SurfaceSource
        return ("yt_surface", Scene, SurfaceSource)
    except Exception:
        try:
            from yt.visualization.volume_rendering.scene import Scene
            from yt.visualization.volume_rendering.render_source import SurfaceSource
            return ("yt_surface", Scene, SurfaceSource)
        except Exception:
            return ("marching_cubes", None, None)


def pick_field(ds, name="sdf"):
    if ("boxlib", name) in ds.field_list:
        return ("boxlib", name)
    return name


def render_surface(plotfile, out_path, field="sdf", color=(1.0, 0.85, 0.2, 1.0)):
    ds = yt.load(plotfile)
    field = pick_field(ds, field)

    mode, Scene, SurfaceSource = _get_surface_api()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if mode == "yt_surface":
        sc = Scene()
        sc.background = (0.0, 0.0, 0.0, 1.0)

        source = SurfaceSource(ds, field, 0.0)
        source.color = color
        sc.add_source(source)

        cam = sc.add_camera(ds, lens_type="perspective")
        cam.focus = ds.domain_center
        cam.width = ds.domain_width
        cam.position = ds.domain_center + ds.domain_width * np.array([1.5, 1.5, 1.5])
        cam.switch_orientation()
        cam.resolution = (1024, 1024)
        sc.save(out_path, sigma_clip=4.0)
        return

    try:
        from skimage import measure
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "SurfaceSource not available. Install scikit-image for marching cubes: "
            "pip install scikit-image"
        ) from exc

    ad = ds.covering_grid(
        level=0,
        left_edge=ds.domain_left_edge,
        dims=ds.domain_dimensions,
    )
    data = ad[field].to_ndarray()

    verts, faces, _, _ = measure.marching_cubes(data, level=0.0)
    # map voxel coords to physical coordinates
    dx = (ds.domain_right_edge - ds.domain_left_edge) / ds.domain_dimensions
    verts = verts * dx + ds.domain_left_edge

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts[faces], alpha=1.0)
    mesh.set_facecolor(color[:3])
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)

    ax.set_xlim(ds.domain_left_edge[0], ds.domain_right_edge[0])
    ax.set_ylim(ds.domain_left_edge[1], ds.domain_right_edge[1])
    ax.set_zlim(ds.domain_left_edge[2], ds.domain_right_edge[2])
    ax.set_axis_off()
    ax.view_init(elev=25, azim=35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="black", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _is_plotfile_dir(path):
    return os.path.isdir(path) and any(
        os.path.isfile(os.path.join(path, name)) for name in os.listdir(path)
    )


def _collect_plotfiles(root):
    plotfiles = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if _is_plotfile_dir(path):
            plotfiles.append(path)
    return plotfiles


def main():
    parser = argparse.ArgumentParser(
        description="Render SDF=0 surface from an AMReX plotfile."
    )
    parser.add_argument(
        "plotfile",
        help="Path to AMReX plotfile directory or a folder containing plotfiles",
    )
    parser.add_argument(
        "--field",
        default="sdf",
        help="Field name to render (default: sdf)",
    )
    parser.add_argument(
        "--out",
        default="zeroSDFvis",
        help="Output folder for PNGs (default: zeroSDFvis/)",
    )
    args = parser.parse_args()

    if os.path.isdir(args.plotfile) and not _is_plotfile_dir(args.plotfile):
        plotfiles = _collect_plotfiles(args.plotfile)
        if not plotfiles:
            raise SystemExit(f"No plotfiles found in {args.plotfile}")
        os.makedirs(args.out, exist_ok=True)
        for pf in plotfiles:
            name = os.path.basename(pf.rstrip("/\\"))
            out_path = os.path.join(args.out, f"{name}.png")
            render_surface(pf, out_path, field=args.field)
            print(f"Saved: {out_path}")
    else:
        os.makedirs(args.out, exist_ok=True)
        name = os.path.basename(args.plotfile.rstrip("/\\"))
        out_path = os.path.join(args.out, f"{name}.png")
        render_surface(args.plotfile, out_path, field=args.field)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
