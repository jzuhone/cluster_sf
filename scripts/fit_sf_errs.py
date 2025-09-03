from astropy.modeling.models import PowerLaw1D
from astropy.modeling.fitting import TRFLSQFitter
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict


parser = ArgumentParser()
parser.add_argument("prefix", type=str)
args = parser.parse_args()


p = Path.cwd()
fns = list(p.glob(f"SF_lmax*{args.prefix}.dat"))

sf_mean = []
sf_min = []
sf_max = []
for fn in fns:
    if "lmin_50" in fn.name or "noerr" not in fn.name:
        continue
    t = Table.read(fn, format="ascii.commented_header")
    sf_mean.append(t["SF"].data)
    sf_min.append(t["SF_min"].data)
    sf_max.append(t["SF_max"].data)
sf_mean = np.array(sf_mean)
sf_min = np.array(sf_min)
sf_max = np.array(sf_max)

print(sf_mean.shape)

def make_err_model():
    err_model = PowerLaw1D()
    err_model.x_0.min = 1.0
    err_model.x_0.max = 1.0e6    
    err_model.amplitude.min = 0.0
    err_model.alpha.min = -6.0
    err_model.alpha.max = 6.0
    return err_model

min_models = defaultdict(list)
max_models = defaultdict(list)

fig, ax = plt.subplots()
for i in range(sf_mean.shape[1]):
    print(i)
    idxs = np.argsort(sf_mean[:,i])
    lfit = TRFLSQFitter()
    line_min = make_err_model()
    line_max = make_err_model()
    min_line = lfit(line_min, sf_mean[idxs,i], sf_min[idxs,i])
    max_line = lfit(line_max, sf_mean[idxs,i], sf_max[idxs,i])
    min_models["bin"].append(i)
    max_models["bin"].append(i)
    for k, v in zip(min_line.param_names, min_line.parameters):
        min_models[k].append(v)
    for k, v in zip(max_line.param_names, max_line.parameters):
        max_models[k].append(v)
    ax.plot(sf_mean[:,i], sf_min[:,i], "x", color=f"C{i}")
    ax.plot(sf_mean[:,i], sf_max[:,i], "+", color=f"C{i}")
    ax.plot(sf_mean[idxs,i], min_line(sf_mean[idxs,i]), color=f"C{i}")
    ax.plot(sf_mean[idxs,i], max_line(sf_mean[idxs,i]), color=f"C{i}")
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig(f"{args.prefix}_sf_fits.png")


tmin = Table(min_models)
tmax = Table(max_models)
tmin.write(
    f"{args.prefix}_sf_err_fit_params_min.dat",
    format="ascii.commented_header",
    overwrite=True,
)
tmax.write(
    f"{args.prefix}_sf_err_fit_params_max.dat",
    format="ascii.commented_header",
    overwrite=True,
)
