import os
import sys

import amrex.space3d as amr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sdf3d import SDFLibrary


def main():
    amr.initialize([])
    try:
        real_box = amr.RealBox([-1, -1, -1], [1, 1, 1])
        domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(63, 63, 63))
        geom = amr.Geometry(domain, real_box, 0, [0, 0, 0])
        ba = amr.BoxArray(domain)
        ba.max_size(32)
        dm = amr.DistributionMapping(ba)

        lib = SDFLibrary(geom, ba, dm)
        mf = lib.sphere(center=(0, 0, 0), radius=0.3)

        mins = []
        maxs = []
        for mfi in mf:
            arr = mf.array(mfi).to_numpy()
            vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
            mins.append(vals.min())
            maxs.append(vals.max())

        print("min/max:", min(mins), max(maxs))
    finally:
        amr.finalize()


if __name__ == "__main__":
    main()
