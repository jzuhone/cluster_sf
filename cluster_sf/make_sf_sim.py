from astropy.convolution import convolve, Gaussian2DKernel
import numpy as np
import h5py
import unyt as u
from constants import sigma_xrism
from sf_bins import make_bins
from utils import (
    compute_sigma,
    make_em,
    make_wcs,
)
from astropy.table import Table
import argparse
from collections import defaultdict
import json

parser = argparse.ArgumentParser()
parser.add_argument("prefix", type=str)
parser.add_argument("l_min", type=int)
parser.add_argument("mach", type=float)
parser.add_argument("--alpha", type=float)
parser.add_argument("--edgesfile", type=str)

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

convolve_it = True
stat_err = True

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

seps_names = {f"{k[0]}-{k[1]}": float(v) for k, v in seps.items()}

with open(f"separations_{prefix}.json", "w") as f:
    json.dump(seps_names, f, indent=4)

bin_edges = edges[bins_used]

bin_ctrs = np.mean(bin_edges, axis=1)

npts = len(regs)
nbins = len(bins)

print(npts, nbins)

reg_masks = [reg.to_pixel(w).to_mask() for reg in regs]

EM = make_em(Lx, Ly, Lz, nx, ny, nz)
EM_proj = EM.sum(axis=2)
if convolve_it:
    kernel = Gaussian2DKernel(sig)
    EM_proj = convolve(EM_proj, kernel)

EMr = [reg_m.cutout(EM_proj).sum() for reg_m in reg_masks]

if stat_err:
    errs = np.random.normal(scale=42.72, size=(npts, 1500))
else:
    errs = np.zeros(npts)


def make_sf(l_max, mach):
    shifts = defaultdict(list)
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
                mEM = m * EM_proj
                if convolve_it:
                    mEM = convolve(mEM, kernel)
                mus = [
                    reg_m.cutout(mEM).sum() / emr + err[k]
                    for err, reg_m, emr in zip(errs, reg_masks, EMr)
                ]
                for i, mu in enumerate(mus):
                    shifts[i].append(mu)
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
    return SF_bins, bin_means, bin_sigs, np.sqrt(bins_slo), np.sqrt(bins_shi), shifts


for l_max in [1000]:
    SF_bins, bin_means, bin_sigs, bins_slo, bins_shi, shifts = make_sf(l_max, mach)

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

    t = Table(shifts)
    t.write(
        f"shifts_lmax_{l_max}_lmin_{l_min}_M{mach}{alpha_str}{stat_str}_{prefix}.dat",
        format="ascii.commented_header",
        overwrite=True,
    )
