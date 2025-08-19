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

print(len(fns))

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

fig, ax = plt.subplots()
for i in range(sf_mean.shape[1]):
    print(i, fns[i])
    idxs = np.argsort(sf_mean[:,i])
    lfit = TRFLSQFitter()
    line_min = PowerLaw1D()
    line_max = PowerLaw1D()
    min_line = lfit(line_min, sf_mean[idxs,i], sf_min[idxs,i])
    max_line = lfit(line_max, sf_mean[idxs,i], sf_max[idxs,i])
    ax.plot(sf_mean[:,i], sf_min[:,i], "x", color=f"C{i}")
    ax.plot(sf_mean[:,i], sf_max[:,i], "+", color=f"C{i}")
    ax.plot(sf_mean[idxs,i], min_line(sf_mean[idxs,i]), color=f"C{i}")
    ax.plot(sf_mean[idxs,i], max_line(sf_mean[idxs,i]), color=f"C{i}")
ax.set_xscale("log")
ax.set_yscale("log")

fig.savefig(f"{args.prefix}_fits.png")
