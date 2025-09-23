from astropy.convolution import convolve, Gaussian2DKernel
import numpy as np
import h5py
import astropy.units as u
from cluster_sf.constants import sigma_xrism
from cluster_sf.bins import make_bins
from cluster_sf.utils import (
    compute_sigma,
    make_em,
    make_wcs,
)
from astropy.table import Table
import argparse
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument("prefix", type=str)
parser.add_argument("l_min", type=int)
parser.add_argument("mach", type=float)
parser.add_argument("--alpha", type=float)
parser.add_argument("--edgesfile", type=str)
parser.add_argument("--noerr", action="store_true")
parser.add_argument("--nopsf", action="store_true")

args = parser.parse_args()

prefix = args.prefix
l_min = args.l_min
mach = args.mach

if args.alpha is None:
    alpha_str = ""
else:
    alpha_str = f"_a{-args.alpha}"
edgesfile = args.edgesfile

regfile = f"{prefix}.reg"

convolve_it = not args.nopsf
stat_err = not args.noerr

stat_str = "" if stat_err else "_noerr"

V_rms = compute_sigma(mach)
V_rms1D = V_rms / np.sqrt(3)

Lx, Ly, Lz = (2203.125,) * 3
nx, ny, nz = (282,) * 3
sig = sigma_xrism / (Lx / nx)
w = make_wcs(Lx, nx)

if edgesfile is None:
    edges = None
else:
    edges = np.loadtxt(edgesfile)

regs, seps, bins, bin_idxs, bins_used, edges = make_bins(regfile, edges=edges)

bin_edges = edges[bins_used]

bin_ctrs = np.mean(bin_edges, axis=1)

npts = len(regs)
nbins = len(bins)

reg_masks = [reg.to_pixel(w).to_mask() for reg in regs]

EM = make_em(Lx, Ly, Lz, nx, ny, nz)
EM_proj = EM.sum(axis=2)
if convolve_it:
    kernel = Gaussian2DKernel(sig)
    EM_proj = convolve(EM_proj, kernel)

EMr = [reg_m.cutout(EM_proj).sum() for reg_m in reg_masks]

if stat_err:
    mu_errs = np.random.normal(scale=42.72, size=(npts, 1500))
    sig_errs = np.random.normal(scale=25.0, size=(npts, 1500))
    if npts == 6:
        sig_errs[5, :] = np.random.normal(scale=39.0, size=1500)
else:
    mu_errs = np.zeros((npts, 1500))
    sig_errs = np.zeros((npts, 1500))


def make_obs(l_max, mach):
    shifts = defaultdict(list)
    sigmas = defaultdict(list)
    mratio = mach / 0.4
    SF_bins = [[] for _ in range(nbins)]
    k = 0
    for i in range(500):
        with h5py.File(
            f"/scratch2/jzuhone/data/coma_cubes/lmax_{int(l_max)}_lmin_{int(l_min)}_M0.4{alpha_str}_proj_field_{i}.h5"
        ) as f:
            for ax in "xyz":
                m = (f[f"f{ax}"][()] * u.kpc / u.Myr).to_value("km/s")
                m *= mratio
                v = (f[f"f2{ax}"][()] * (u.kpc / u.Myr) ** 2).to_value("km**2/s**2")
                v *= mratio * mratio
                mEM = m * EM_proj
                vEM = v * EM_proj
                if convolve_it:
                    mEM = convolve(mEM, kernel)
                    vEM = convolve(vEM, kernel)
                mus = np.array(
                    [
                        reg_m.cutout(mEM).sum() / emr
                        for reg_m, emr in zip(reg_masks, EMr)
                    ]
                )
                sigs = np.array(
                    [
                        reg_m.cutout(vEM).sum() / emr
                        for reg_m, emr in zip(reg_masks, EMr)
                    ]
                )
                sigs -= mus * mus
                sigs = np.sqrt(sigs)
                for j, (mu_err, sig_err) in enumerate(zip(mu_errs, sig_errs)):
                    mus[j] += mu_err[k]
                    sigs[j] += sig_err[k]
                for j, mu in enumerate(mus):
                    shifts[j].append(mu)
                    sigmas[j].append(sigs[j])
                for j in range(nbins):
                    bidxs = list(bin_idxs.values())[j]
                    SF_bins[j].append(
                        np.mean([(mus[bi[0]] - mus[bi[1]]) ** 2 for bi in bidxs])
                    )
                k += 1

    SF_bins = np.array(SF_bins)
    bin_means = np.mean(SF_bins, axis=1)
    bin_sigs = np.std(SF_bins, axis=1)
    d = SF_bins - bin_means[:, np.newaxis]
    idxs_lo = d < 0
    idxs_hi = ~idxs_lo
    bins_slo = []
    bins_shi = []
    for k in range(nbins):
        bins_slo.append(np.sum(d[k, idxs_lo[k]] ** 2) / (idxs_lo[k].sum() - 1.0))
        bins_shi.append(np.sum(d[k, idxs_hi[k]] ** 2) / (idxs_hi[k].sum() - 1.0))
    return SF_bins, bin_means, bin_sigs, np.sqrt(bins_slo), np.sqrt(bins_shi), shifts, sigmas


for l_max in [100, 300, 500, 1000]:
    SF_bins, bin_means, bin_sigs, bins_slo, bins_shi, shifts, sigmas = make_obs(l_max, mach)

    shifts = np.array([v for v in shifts.values()]).T
    sigmas = np.array([v for v in sigmas.values()]).T

    t = Table(shifts)
    t.write(
        f"shifts_lmax_{l_max}_lmin_{l_min}_M{mach}{alpha_str}{stat_str}_{prefix}.dat",
        format="ascii.commented_header",
        overwrite=True,
    )

    t = Table(sigmas)
    t.write(
        f"sigmas_lmax_{l_max}_lmin_{l_min}_M{mach}{alpha_str}{stat_str}_{prefix}.dat",
        format="ascii.commented_header",
        overwrite=True,
    )

    data = {
        "r": np.array([np.mean(bin) for bin in bins.values()]),
        "r_min": np.array([np.min(bin) for bin in bins.values()]),
        "r_max": np.array([np.max(bin) for bin in bins.values()]),
        "SF": bin_means,
        "SF_min": bins_slo,
        "SF_max": bins_shi,
        "SF_sig": bin_sigs,
    }

    t = Table(data)
    t.sort("r")
    t["r_ctr"] = bin_ctrs
    t["r_left"] = bin_edges[:, 0]
    t["r_right"] = bin_edges[:, 1]
    t["bin_num"] = bins_used

    t.write(
        f"SF_lmax_{l_max}_lmin_{l_min}_M{mach}{alpha_str}{stat_str}_{prefix}.dat",
         format="ascii.commented_header",
        overwrite=True,
    )

    sig_avg = np.mean(sigmas, axis=0)
    d = sigmas - sig_avg
    idxs_lo = d < 0
    idxs_hi = ~idxs_lo
    sig_min = []
    sig_max = []
    for k in range(npts):
        sig_min.append(np.sqrt(np.sum(d[idxs_lo[:,k],k] ** 2) / (idxs_lo[:,k].sum() - 1.0)))
        sig_max.append(np.sqrt(np.sum(d[idxs_hi[:,k],k] ** 2) / (idxs_hi[:,k].sum() - 1.0)))
    sig_std = np.std(sigmas, axis=0)

    data = {
        "sigma_avg": sig_avg,
        "sigma_min": sig_min,
        "sigma_max": sig_max,
        "sigma_std": sig_std,
    }

    t = Table(data)

    t.write(
        f"sig_lmax_{l_max}_lmin_{l_min}_M{mach}{alpha_str}{stat_str}_{prefix}.dat",
        format="ascii.commented_header",
        overwrite=True,
    )
