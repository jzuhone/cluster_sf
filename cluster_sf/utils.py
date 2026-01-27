import astropy.units as u
from numba import njit
import numpy as np
from tqdm.auto import tqdm
import h5py
from astropy import wcs
from astropy.constants import m_p
from cluster_sf.constants import angular_scale
from astropy.table import Table
from pathlib import Path


data_root = Path(__file__).parent / "../data"


def make_wcs(Lx, nx):
    dx = Lx * u.kpc / nx
    dtheta = (dx / angular_scale).to_value("deg")
    w = wcs.WCS(naxis=2)
    w.wcs.cunit = ["deg"] * 2
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [0.5 * (nx + 1)] * 2
    w.wcs.cdelt = [-dtheta, dtheta]
    w.wcs.crval = [194.929, 27.959]
    return w


def compute_sigma(mach):
    mu = 0.6
    gamma = 5.0 / 3.0
    kT = 8.0 * u.keV
    c_s = np.sqrt(gamma * kT / (mu * m_p)).to_value("km/s")
    return mach * c_s


@njit(parallel=True)
def make_em(Lx, Ly, Lz, nx, ny, nz):
    beta = 2.0 / 3.0
    rc2 = 300.0**2
    xx = np.linspace(-0.5 * Lx, 0.5 * Lx, nx + 1)
    yy = np.linspace(-0.5 * Ly, 0.5 * Ly, ny + 1)
    zz = np.linspace(-0.5 * Lz, 0.5 * Lz, nz + 1)
    x = 0.5 * (xx[1:] + xx[:-1])
    y = 0.5 * (yy[1:] + yy[:-1])
    z = 0.5 * (zz[1:] + zz[:-1])
    EM = np.zeros((nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                rr = x[i] ** 2 + y[j] ** 2 + z[k] ** 2
                EM[i, j, k] = 1.0 + rr / rc2
    EM **= -3.0 * beta
    return EM


def two_sided_std(data, axis=None):
    data = np.asarray(data)
    n = len(data)
    if n < 2:
        return np.nan, np.nan
    mean = np.mean(data, axis=axis, keepdims=True)
    diffs = data - mean
    diffs_neg = diffs[diffs < 0]
    diffs_pos = diffs[diffs > 0]
    return np.std(diffs_neg, axis=axis), np.std(diffs_pos, axis=axis)


def make_sf_err_func(fn):
    t = Table.read(fn, format="ascii.commented_header")
    A = t["amplitude"].data
    x0 = t["x_0"].data
    alpha = -t["alpha"].data
    def _sf_err_func(y):
        return A*(y / x0)**alpha
    return _sf_err_func
