import numpy as np

import amrex.space3d as amr

import sdf_lib as sdf


def _get_component_view(arr):
    if arr.ndim == 4:
        return arr[:, :, :, 0]
    return arr[:, :, :, 0, 0]


def _copy_mf_like(src, ba, dm) -> amr.MultiFab:
    ncomp = src.n_comp() if callable(getattr(src, "n_comp", None)) else src.n_comp
    if callable(getattr(src, "n_grow", None)):
        ngrow = src.n_grow()
    elif hasattr(src, "nGrow"):
        ngrow = src.nGrow()
    else:
        ngrow = 0
    mf = amr.MultiFab(ba, dm, ncomp, ngrow)
    for mfi in src:
        arr_src = src.array(mfi).to_numpy()
        arr_dst = mf.array(mfi).to_numpy()
        _get_component_view(arr_dst)[...] = _get_component_view(arr_src)
    return mf


class SDFLibrary:
    """
    Build SDF fields directly in AMReX MultiFabs.
    """

    def __init__(self, geom, ba, dm):
        self.geom = geom
        self.ba = ba
        self.dm = dm

    def create_field(self):
        return amr.MultiFab(self.ba, self.dm, 1, 0)

    def from_geometry(self, geometry):
        mf = self.create_field()
        self._fill_multifab(mf, geometry.sdf)
        return mf

    def sphere(self, center, radius):
        def _sdf(p):
            return sdf.sdSphere(p - np.array(center, dtype=float), radius)
        mf = self.create_field()
        self._fill_multifab(mf, _sdf)
        return mf

    def box(self, center, half_size):
        half_size = np.array(half_size, dtype=float)
        center = np.array(center, dtype=float)
        def _sdf(p):
            return sdf.sdBox(p - center, half_size)
        mf = self.create_field()
        self._fill_multifab(mf, _sdf)
        return mf

    def round_box(self, center, half_size, radius):
        half_size = np.array(half_size, dtype=float)
        center = np.array(center, dtype=float)
        def _sdf(p):
            return sdf.sdRoundBox(p - center, half_size, radius)
        mf = self.create_field()
        self._fill_multifab(mf, _sdf)
        return mf

    def union(self, a, b):
        out = _copy_mf_like(a, self.ba, self.dm)
        for mfi in out:
            arr_out = out.array(mfi).to_numpy()
            arr_b = b.array(mfi).to_numpy()
            _get_component_view(arr_out)[...] = sdf.opUnion(
                _get_component_view(arr_out), _get_component_view(arr_b)
            )
        return out

    def subtract(self, a, b):
        out = _copy_mf_like(a, self.ba, self.dm)
        for mfi in out:
            arr_out = out.array(mfi).to_numpy()
            arr_b = b.array(mfi).to_numpy()
            _get_component_view(arr_out)[...] = sdf.opSubtraction(
                _get_component_view(arr_out), _get_component_view(arr_b)
            )
        return out

    def intersect(self, a, b):
        out = _copy_mf_like(a, self.ba, self.dm)
        for mfi in out:
            arr_out = out.array(mfi).to_numpy()
            arr_b = b.array(mfi).to_numpy()
            _get_component_view(arr_out)[...] = sdf.opIntersection(
                _get_component_view(arr_out), _get_component_view(arr_b)
            )
        return out

    def negate(self, a):
        out = _copy_mf_like(a, self.ba, self.dm)
        for mfi in out:
            arr_out = out.array(mfi).to_numpy()
            _get_component_view(arr_out)[...] *= -1.0
        return out

    def _fill_multifab(self, mf, sdf_func):
        dx = self.geom.data().CellSize()
        if hasattr(self.geom, "ProbLoArray"):
            prob_lo = self.geom.ProbLoArray()
        elif hasattr(self.geom, "ProbLo"):
            prob_lo = np.array(self.geom.ProbLo())
        else:
            raise AttributeError("Geometry has no ProbLoArray/ProbLo accessor")

        for mfi in mf:
            arr = mf.array(mfi).to_numpy()
            bx = mfi.validbox()

            i_lo, j_lo, k_lo = bx.lo_vect
            i_hi, j_hi, k_hi = bx.hi_vect

            i = np.arange(i_lo, i_hi + 1)
            j = np.arange(j_lo, j_hi + 1)
            k = np.arange(k_lo, k_hi + 1)

            x = (i + 0.5) * dx[0] + prob_lo[0]
            y = (j + 0.5) * dx[1] + prob_lo[1]
            z = (k + 0.5) * dx[2] + prob_lo[2]

            Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
            p = sdf.vec3(X, Y, Z)

            _get_component_view(arr)[...] = sdf_func(p)
