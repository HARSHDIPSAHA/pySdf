import os

import numpy as np
import yt

import sdf_lib as sdf


def build_volume(n=96, bounds=(-0.5, 0.5)):
    lo, hi = bounds
    coords = np.linspace(lo, hi, n, endpoint=False) + (hi - lo) / (2.0 * n)
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
    p = sdf.vec3(x, y, z)
    return p


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


def render_volume(values, name, out_dir, bounds=(-0.5, 0.5)):
    bbox = np.array([[bounds[0], bounds[1]], [bounds[0], bounds[1]], [bounds[0], bounds[1]]])
    ds = yt.load_uniform_grid({"sdf": values.astype(np.float64)}, values.shape, bbox=bbox)

    vmin = float(values.min())
    vmax = float(values.max())
    width = 0.03 * (vmax - vmin if vmax > vmin else 1.0)

    sc = yt.create_scene(ds, field="sdf")
    source = sc[0]
    source.set_log(False)

    tf = yt.ColorTransferFunction((vmin, vmax))
    tf.add_gaussian(0.0, width=width, height=[1.0, 1.0, 1.0, 0.7])
    source.transfer_function = tf

    cam = sc.camera
    cam.focus = ds.domain_center
    cam.width = ds.domain_width
    cam.position = ds.domain_center + ds.domain_width * np.array([1.5, 1.5, 1.5])
    cam.switch_orientation()

    path = os.path.join(out_dir, f"{name}.png")
    sc.save(path, sigma_clip=4.0)


def main():
    out_dir = "vis3d"
    os.makedirs(out_dir, exist_ok=True)

    n = 96
    p = build_volume(n=n)

    for name, sdf_func in make_cases():
        values = sdf_func(p)
        render_volume(values, name, out_dir)
        print(f"Saved: {name}.png")


if __name__ == "__main__":
    main()
