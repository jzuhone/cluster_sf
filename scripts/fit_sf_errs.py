from astropy.modeling.models import PowerLaw1D
from astropy.modeling.fitting import TRFLSQFitter
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("prefix", type=str)
args = parser.parse_args()


p = Path.cwd()
fns = list(p.glob(f"SF_lmax*{args.prefix}.dat"))

sf_mean = []
sf_min = []
sf_max = []
for fn in fns:
    t = Table.read(fn, format="ascii.commented_header")
    sf_mean.append(t["SF"].data)
    sf_min.append(t["SF_min"].data)
    sf_max.append(t["SF_max"].data)
sf_mean = np.array(sf_mean)
sf_min = np.array(sf_min)
sf_max = np.array(sf_max)

lines = []

print(sf_mean.shape)

fig, ax = plt.subplots()
for i in range(sf_mean.shape[1]):
    idxs = np.argsort(sf_mean[:,i])
    lfit = TRFLSQFitter()
    line_min = PowerLaw1D()
    line_max = PowerLaw1D()
    min_line = lfit(line_min, sf_mean[idxs,i], sf_min[idxs,i])
    max_line = lfit(line_max, sf_mean[idxs,i], sf_max[idxs,i])
    pmin = {
        k: np.atleast_1d(v)
        for k, v in zip(min_line.param_names, min_line.parameters)
    }
    pmax = {
        k: np.atleast_1d(v)
        for k, v in zip(max_line.param_names, max_line.parameters)
    }
    tmin = Table(pmin)
    tmax = Table(pmax)
    tmin.write(
        f"{args.prefix}_sf_err_fit_params_min_bin_{i}.dat",
        format="ascii.commented_header",
        overwrite=True,
    )
    tmax.write(
        f"{args.prefix}_sf_err_fit_params_max_bin_{i}.dat",
        format="ascii.commented_header",
        overwrite=True,
    )

    ax.plot(sf_mean[:,i], sf_min[:,i], "x", color=f"C{i}")
    ax.plot(sf_mean[:,i], sf_max[:,i], "+", color=f"C{i}")
    ax.plot(sf_mean[idxs,i], min_line(sf_mean[idxs,i]), color=f"C{i}")
    ax.plot(sf_mean[idxs,i], max_line(sf_mean[idxs,i]), color=f"C{i}")
ax.set_xscale("log")
ax.set_yscale("log")

fig.savefig(f"{args.prefix}_sf_fits.png")
