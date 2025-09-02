from astropy.modeling.models import custom_model
from astropy.modeling.fitting import TRFLSQFitter
import numpy as np
from astropy.table import Table
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict


@custom_model
def PowerLaw2D(
    x,
    y,
    A=1.0,
    x0=100.0,
    y0=4.0,
    x1=100.0,
    y1=4.0,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    delta=1.0,
):
    x_factor = (1.0 + (x / x0) ** 2) ** alpha
    y_factor = (1.0 + (y / y0) ** 2) ** beta
    x2_factor = (1.0 + (x / x1) ** 2) ** gamma
    y2_factor = (1.0 + (y / y1) ** 2) ** delta
    return A * x_factor * y_factor * x2_factor * y2_factor


def make_err_model():
    err_model = PowerLaw2D()
    err_model.x0.min = 10.0
    err_model.x0.max = 2000.0
    err_model.y0.min = 1.0
    err_model.y0.max = 10.0
    err_model.x1.min = 10.0
    err_model.x1.max = 2000.0
    err_model.y1.min = 1.0
    err_model.y1.max = 10.0
    err_model.alpha.min = -6.0
    err_model.alpha.max = 6.0
    err_model.beta.min = -6.0
    err_model.beta.max = 6.0
    err_model.gamma.min = -6.0
    err_model.gamma.max = 6.0
    err_model.delta.min = -6.0
    err_model.delta.max = 6.0
    return err_model


parser = ArgumentParser()
parser.add_argument("prefix", type=str)
args = parser.parse_args()


p = Path.cwd()
fns = list(p.glob(f"sig_lmax*noerr_{args.prefix}.dat"))

print(len(fns))

sig_min = defaultdict(list)
sig_max = defaultdict(list)
lmax = []
alpha = []
for i, fn in enumerate(fns):
    if "lmin_50" in fn.name:
        continue
    words = fn.name.split("_")
    lmax.append(float(words[2]))
    alpha.append(11.0 / 3.0 if len(words) == 9 else float(words[6][1:]))
    t = Table.read(fn, format="ascii.commented_header")
    sig_min[0].append(np.mean(t["sigma_min"].data[:4] / t["sigma_avg"].data[:4]))
    sig_max[0].append(np.mean(t["sigma_max"].data[:4] / t["sigma_avg"].data[:4]))
    sig_min[1].append(np.mean(t["sigma_min"].data[4:] / t["sigma_avg"].data[4:]))
    sig_max[1].append(np.mean(t["sigma_max"].data[4:] / t["sigma_avg"].data[4:]))
lmax = np.array(lmax)
alpha = np.array(alpha)
for key in sig_min:
    sig_min[key] = np.array(sig_min[key])
    sig_max[key] = np.array(sig_max[key])

lfit = TRFLSQFitter()
for key in sig_min:
    err_model_min = make_err_model()
    err_model_max = make_err_model()
    err_sheet_min = lfit(err_model_min, lmax, alpha, sig_min[key])
    err_sheet_max = lfit(err_model_max, lmax, alpha, sig_max[key])
    pmin = {
        k: np.atleast_1d(v)
        for k, v in zip(err_sheet_min.param_names, err_sheet_min.parameters)
    }
    pmax = {
        k: np.atleast_1d(v)
        for k, v in zip(err_sheet_max.param_names, err_sheet_max.parameters)
    }
    print(key, pmin)
    print(key, pmax)
    tmin = Table(pmin)
    tmax = Table(pmax)
    tmin.write(
        f"{args.prefix}_sig_err_fit_params_min_{key}.dat",
        format="ascii.commented_header",
        overwrite=True,
    )
    tmax.write(
        f"{args.prefix}_sig_err_fit_params_max_{key}.dat",
        format="ascii.commented_header",
        overwrite=True,
    )

"""
lmax_grid, alpha_grid = np.meshgrid(
    np.linspace(lmax.min(), lmax.max(), 100), np.linspace(alpha.min(), alpha.max(), 100)
)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(lmax, alpha, sig_max[0])
ax.plot_surface(lmax_grid, alpha_grid, err_sheet(lmax_grid, alpha_grid), alpha=0.5)
plt.show()
"""