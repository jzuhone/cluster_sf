import astropy.units as u
from numba import njit
import numpy as np
from tqdm.auto import tqdm
import h5py
from astropy import wcs
from astropy.constants import m_p
from .constants import angular_scale


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


def generate_realizations(vgen, num_tries, prefix, project_weight):
    sigma = vgen._compute_pspec()
    wx = np.sum(project_weight, axis=0)
    pbar = tqdm(leave=True, total=num_tries, desc="Generating field realizations ")
    for i in range(num_tries):
        vgen._generate_field(sigma=sigma)
        vgen._post_generate()
        gwx = vgen.gx * project_weight
        gwy = vgen.gy * project_weight
        gwz = vgen.gz * project_weight
        fx = np.sum(gwx, axis=0)
        fy = np.sum(gwy, axis=1)
        fz = np.sum(gwz, axis=2)
        fx /= wx
        fy /= wx
        fz /= wx
        f2x = np.sum(vgen.gx * gwx, axis=0)
        f2y = np.sum(vgen.gy * gwy, axis=1)
        f2z = np.sum(vgen.gz * gwz, axis=2)
        f2x /= wx
        f2y /= wx
        f2z /= wx
        units2 = f"{vgen.units}**2"
        with h5py.File(f"{prefix}_proj_field_{i}.h5", "w") as f:
            d = f.create_dataset("x", data=vgen.x)
            d.attrs["units"] = "kpc"
            d = f.create_dataset("y", data=vgen.y)
            d.attrs["units"] = "kpc"
            d = f.create_dataset("z", data=vgen.z)
            d.attrs["units"] = "kpc"
            d = f.create_dataset("fx", data=fx)
            d.attrs["units"] = vgen.units
            d = f.create_dataset("fy", data=fy)
            d.attrs["units"] = vgen.units
            d = f.create_dataset("fz", data=fz)
            d.attrs["units"] = vgen.units
            d = f.create_dataset("f2x", data=f2x)
            d.attrs["units"] = units2
            d = f.create_dataset("f2y", data=f2y)
            d.attrs["units"] = units2
            d = f.create_dataset("f2z", data=f2z)
            d.attrs["units"] = units2
        pbar.update()
    pbar.close()
