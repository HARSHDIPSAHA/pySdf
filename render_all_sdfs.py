import os

import amrex.space2d as amr
import matplotlib.pyplot as plt
import numpy as np

import sdf_lib as sdf


def build_grid(n=192, max_grid_size=64):
    prob_lo = [0.0, 0.0]
    prob_hi = [1.0, 1.0]

    real_box = amr.RealBox(prob_lo, prob_hi)
    domain = amr.Box(np.array([0, 0]), np.array([n - 1, n - 1]))
    geom = amr.Geometry(domain, real_box, 0, [0, 0])

    ba = amr.BoxArray(domain)
    ba.max_size(max_grid_size)
    dm = amr.DistributionMapping(ba)
    return geom, ba, dm


def fill_multifab(sdf_mf, geom, sdf_func):
    dx = geom.data().CellSize()
    for mfi in sdf_mf:
        arr = sdf_mf.array(mfi).to_numpy()
        bx = mfi.validbox()
        i_lo, j_lo = bx.lo_vect
        i_hi, j_hi = bx.hi_vect

        i = np.arange(i_lo, i_hi + 1)
        j = np.arange(j_lo, j_hi + 1)

        x = (i + 0.5) * dx[0]
        y = (j + 0.5) * dx[1]

        Y, X = np.meshgrid(y, x, indexing="ij")
        p = sdf.vec3(X - 0.5, Y - 0.5, np.zeros_like(X))
        arr[:, :, 0, 0] = sdf_func(p)


def gather_full(sdf_mf, n):
    full = np.zeros((n, n))
    for mfi in sdf_mf:
        arr = sdf_mf.array(mfi).to_numpy()
        bx = mfi.validbox()
        i_lo, j_lo = bx.lo_vect
        i_hi, j_hi = bx.hi_vect
        full[j_lo : j_hi + 1, i_lo : i_hi + 1] = arr[:, :, 0, 0]
    return full


def plot_and_save(full, name, out_dir):
    lim = np.max(np.abs(full))
    if lim == 0:
        lim = 1e-6

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        full,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap="seismic",
        vmin=-lim,
        vmax=lim,
    )
    plt.colorbar(im, label="Signed Distance")
    plt.contour(full, levels=[0.0], colors="black", linewidths=1.5, extent=[0, 1, 0, 1])
    plt.title(name)
    plt.xlabel("X")
    plt.ylabel("Y")

    path = os.path.join(out_dir, f"{name}.png")
    plt.savefig(path, dpi=150)
    plt.close()


def make_cases():
    sphere_r = 0.3
    box_b = np.array([0.25, 0.2, 0.15])
    round_r = 0.05

    base_sphere = lambda p: sdf.sdSphere(p, sphere_r)
    base_box = lambda p: sdf.sdBox(p, box_b)
    base_round_box = lambda p: sdf.sdRoundBox(p, box_b, round_r)

    cases = [
        ("sdSphere", lambda p: sdf.sdSphere(p, sphere_r)),
        ("sdBox", lambda p: sdf.sdBox(p, box_b)),
        ("sdRoundBox", lambda p: sdf.sdRoundBox(p, box_b, round_r)),
        ("sdBoxFrame", lambda p: sdf.sdBoxFrame(p, np.array([0.25, 0.25, 0.25]), 0.05)),
        ("sdTorus", lambda p: sdf.sdTorus(p, np.array([0.25, 0.08]))),
        ("sdCappedTorus", lambda p: sdf.sdCappedTorus(p, np.array([0.5, 0.8660254]), 0.25, 0.06)),
        ("sdLink", lambda p: sdf.sdLink(p, 0.15, 0.2, 0.05)),
        ("sdCylinder", lambda p: sdf.sdCylinder(p, np.array([0.0, 0.0, 0.25]))),
        ("sdConeExact", lambda p: sdf.sdConeExact(p, np.array([0.6, 0.8]), 0.35)),
        ("sdConeBound", lambda p: sdf.sdConeBound(p, np.array([0.6, 0.8]), 0.35)),
        ("sdConeInfinite", lambda p: sdf.sdConeInfinite(p, np.array([0.6, 0.8]))),
        ("sdPlane", lambda p: sdf.sdPlane(p, np.array([0.0, 1.0, 0.0]), 0.0)),
        ("sdHexPrism", lambda p: sdf.sdHexPrism(p, np.array([0.25, 0.2]))),
        ("sdTriPrism", lambda p: sdf.sdTriPrism(p, np.array([0.3, 0.2]))),
        ("sdCapsule", lambda p: sdf.sdCapsule(p, np.array([-0.3, 0.0, 0.0]), np.array([0.3, 0.0, 0.0]), 0.15)),
        ("sdVerticalCapsule", lambda p: sdf.sdVerticalCapsule(p, 0.4, 0.15)),
        ("sdCappedCylinder", lambda p: sdf.sdCappedCylinder(p, 0.2, 0.3)),
        ("sdCappedCylinderSegment", lambda p: sdf.sdCappedCylinderSegment(p, np.array([0.0, -0.2, 0.0]), np.array([0.0, 0.2, 0.0]), 0.2)),
        ("sdRoundedCylinder", lambda p: sdf.sdRoundedCylinder(p, 0.2, 0.05, 0.25)),
        ("sdCappedCone", lambda p: sdf.sdCappedCone(p, 0.3, 0.2, 0.05)),
        ("sdCappedConeSegment", lambda p: sdf.sdCappedConeSegment(p, np.array([0.0, -0.25, 0.0]), np.array([0.0, 0.25, 0.0]), 0.2, 0.05)),
        ("sdSolidAngle", lambda p: sdf.sdSolidAngle(p, np.array([0.5, 0.8660254]), 0.4)),
        ("sdCutSphere", lambda p: sdf.sdCutSphere(p, 0.35, 0.15)),
        ("sdCutHollowSphere", lambda p: sdf.sdCutHollowSphere(p, 0.35, 0.15, 0.05)),
        ("sdDeathStar", lambda p: sdf.sdDeathStar(p, 0.35, 0.2, 0.3)),
        ("sdRoundCone", lambda p: sdf.sdRoundCone(p, 0.25, 0.05, 0.4)),
        ("sdRoundConeSegment", lambda p: sdf.sdRoundConeSegment(p, np.array([0.0, -0.25, 0.0]), np.array([0.0, 0.25, 0.0]), 0.2, 0.05)),
        ("sdEllipsoid", lambda p: sdf.sdEllipsoid(p, np.array([0.35, 0.25, 0.2]))),
        ("sdVesicaSegment", lambda p: sdf.sdVesicaSegment(p, np.array([-0.25, 0.0, 0.0]), np.array([0.25, 0.0, 0.0]), 0.2)),
        ("sdRhombus", lambda p: sdf.sdRhombus(p, 0.3, 0.2, 0.15, 0.05)),
        ("sdOctahedronExact", lambda p: sdf.sdOctahedronExact(p, 0.35)),
        ("sdOctahedronBound", lambda p: sdf.sdOctahedronBound(p, 0.35)),
        ("sdPyramid", lambda p: sdf.sdPyramid(p, 0.4)),
        ("udTriangle", lambda p: sdf.udTriangle(p, np.array([-0.3, -0.2, 0.0]), np.array([0.3, -0.2, 0.0]), np.array([0.0, 0.3, 0.0]))),
        ("udQuad", lambda p: sdf.udQuad(p, np.array([-0.3, -0.3, 0.0]), np.array([0.3, -0.3, 0.0]), np.array([0.3, 0.3, 0.0]), np.array([-0.3, 0.3, 0.0]))),
        ("opUnion", lambda p: sdf.opUnion(base_sphere(p), base_box(p))),
        ("opSubtraction", lambda p: sdf.opSubtraction(base_sphere(p), base_box(p))),
        ("opIntersection", lambda p: sdf.opIntersection(base_sphere(p), base_box(p))),
        ("opXor", lambda p: sdf.opXor(base_sphere(p), base_box(p))),
        ("opSmoothUnion", lambda p: sdf.opSmoothUnion(base_sphere(p), base_box(p), 0.1)),
        ("opSmoothSubtraction", lambda p: sdf.opSmoothSubtraction(base_sphere(p), base_box(p), 0.1)),
        ("opSmoothIntersection", lambda p: sdf.opSmoothIntersection(base_sphere(p), base_box(p), 0.1)),
        ("opRevolution", lambda p: sdf.opRevolution(p, lambda q: sdf.sdCircle2d(q, 0.2), 0.15)),
        ("opExtrusion", lambda p: sdf.opExtrusion(p, lambda q: sdf.sdBox2d(q, np.array([0.2, 0.2])), 0.1)),
        ("opElongate1", lambda p: sdf.opElongate1(p, base_sphere, np.array([0.2, 0.1, 0.0]))),
        ("opElongate2", lambda p: sdf.opElongate2(p, base_sphere, np.array([0.2, 0.1, 0.0]))),
        ("opRound", lambda p: sdf.opRound(p, base_box, 0.05)),
        ("opOnion", lambda p: sdf.opOnion(base_sphere(p), 0.05)),
        ("opScale", lambda p: sdf.opScale(p, 0.8, base_sphere)),
        ("opSymX", lambda p: sdf.opSymX(p, base_box)),
        ("opSymXZ", lambda p: sdf.opSymXZ(p, base_box)),
        ("opRepetition", lambda p: sdf.opRepetition(p, np.array([0.5, 0.5, 0.5]), base_sphere)),
        ("opLimitedRepetition", lambda p: sdf.opLimitedRepetition(p, np.array([0.5, 0.5, 0.5]), np.array([1.0, 1.0, 1.0]), base_sphere)),
        ("opDisplace", lambda p: sdf.opDisplace(p, base_sphere)),
        ("opTwist", lambda p: sdf.opTwist(p, base_round_box, 6.0)),
        ("opCheapBend", lambda p: sdf.opCheapBend(p, base_round_box, 6.0)),
        ("opTx", lambda p: sdf.opTx(p, rotation_z(np.deg2rad(30.0)), np.array([0.1, 0.1, 0.0]), base_box)),
    ]
    return cases


def rotation_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def main():
    amr.initialize([])
    try:
        if amr.Config.spacedim != 2:
            print("ERROR: pyAMReX not built in 2D")
            return

        out_dir = "vis"
        os.makedirs(out_dir, exist_ok=True)

        n = 192
        geom, ba, dm = build_grid(n=n)
        sdf_mf = amr.MultiFab(ba, dm, 1, 0)

        for name, sdf_func in make_cases():
            sdf_mf.set_val(0.0)
            fill_multifab(sdf_mf, geom, sdf_func)
            full = gather_full(sdf_mf, n)
            plot_and_save(full, name, out_dir)
            print(f"Saved: {name}.png")
    finally:
        amr.finalize()


if __name__ == "__main__":
    main()
