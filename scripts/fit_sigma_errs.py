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
fns = list(p.glob(f"sig_lmax*noerr_{args.prefix}.dat"))

print(len(fns))

sig_mean = []
sig_min = []
sig_max = []
for fn in fns:
    print(fn)
    t = Table.read(fn, format="ascii.commented_header")
    sig_mean.append(t["sigma_avg"].data)
    sig_min.append(t["sigma_min"].data)
    sig_max.append(t["sigma_max"].data)
sig_mean = np.array(sig_mean)
sig_min = np.array(sig_min)
sig_max = np.array(sig_max)

lines = []

fig, ax = plt.subplots()
for i in range(sig_mean.shape[1]):
    idxs = np.argsort(sig_mean[:,i])
    #lfit = TRFLSQFitter()
    #line_min = PowerLaw1D()
    #line_max = PowerLaw1D()
    #min_line = lfit(line_min, sf_mean[idxs,i], sf_min[idxs,i])
    #max_line = lfit(line_max, sf_mean[idxs,i], sf_max[idxs,i])
    ax.plot(sig_mean[idxs,i], sig_min[idxs,i], ls="--", color=f"C{i}")
    ax.plot(sig_mean[idxs,i], sig_max[idxs,i], ls="-", color=f"C{i}")
    #ax.plot(sf_mean[idxs,i], min_line(sf_mean[idxs,i]), color=f"C{i}")
    #ax.plot(sf_mean[idxs,i], max_line(sf_mean[idxs,i]), color=f"C{i}")
#ax.set_xscale("log")
#ax.set_yscale("log")

fig.savefig(f"{args.prefix}_sig_fits.png")
