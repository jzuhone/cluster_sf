from astropy.modeling.models import PowerLaw1D
from astropy.modeling.fitting import TRFLSQFitter
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from pathlib import Path

p = Path("three_pts")

mach = 0.4

sf_mean = []
sf_min = []
sf_mid = []
sf_max = []
for l_min in [1, 50]:
    for i, l_max in enumerate([100, 300, 500, 1000]):
        t = Table.read(
            p / f"SF_lmax_{l_max}_lmin_{l_min}_M{mach}.dat", format="ascii.commented_header"
        )
        sf_mean.append(t["SF"].data)
        sf_min.append(t["SF_min"].data)
        sf_mid.append(t["SF_sig"].data)
        sf_max.append(t["SF_max"].data)
for i, l_max in enumerate([100, 300, 500, 1000]):
    t = Table.read(
        p / f"SF_lmax_{l_max}_lmin_1_M{mach}_a5.0.dat", format="ascii.commented_header"
    )
    sf_mean.append(t["SF"].data)
    sf_min.append(t["SF_min"].data)
    sf_mid.append(t["SF_sig"].data)
    sf_max.append(t["SF_max"].data)
sf_mean = np.concatenate(sf_mean)
sf_min = np.concatenate(sf_min)
sf_max = np.concatenate(sf_max)
sf_mid = np.concatenate(sf_mid)

idxs = np.argsort(sf_mean)
sf_mean = sf_mean[idxs]
sf_min = sf_min[idxs]
sf_max = sf_max[idxs]
sf_mid = sf_mid[idxs]

lfit = TRFLSQFitter()
line_min = PowerLaw1D()
line_max = PowerLaw1D()
line_mid = PowerLaw1D()

# fit the data with the fitter
min_line = lfit(line_min, sf_mean, sf_min)
max_line = lfit(line_max, sf_mean, sf_max)
mid_line = lfit(line_max, sf_mean, sf_mid)


fig, ax = plt.subplots()
for l_min in [1,50]:
    for i, l_max in enumerate([100, 300, 500, 1000]):
        t = Table.read(
            p / f"SF_lmax_{l_max}_lmin_{l_min}_M{mach}.dat", format="ascii.commented_header"
        )
        ax.plot(t["SF"], t["SF_min"], 'x', color="C0")
        ax.plot(t["SF"], t["SF_max"], 'x', color="C1")
        ax.plot(t["SF"], t["SF_sig"], 'x', color="C2")
for i, l_max in enumerate([100, 300, 500, 1000]):
    t = Table.read(
        p / f"SF_lmax_{l_max}_lmin_1_M{mach}_a5.0.dat", format="ascii.commented_header"
    )
    ax.plot(t["SF"], t["SF_min"], 'x', color="C0")
    ax.plot(t["SF"], t["SF_max"], 'x', color="C1")
    ax.plot(t["SF"], t["SF_sig"], 'x', color="C2")
for i, l_max in enumerate([1000]):
    t = Table.read(
        p / f"SF_lmax_{l_max}_lmin_1_M0.55_a8.0.dat", format="ascii.commented_header"
    )
    ax.plot(t["SF"], t["SF_min"], 'x', color="C0")
    ax.plot(t["SF"], t["SF_max"], 'x', color="C1")
    ax.plot(t["SF"], t["SF_sig"], 'x', color="C2")

ax.plot(sf_mean, min_line(sf_mean), color="C0")
ax.plot(sf_mean, max_line(sf_mean), color="C1")
ax.plot(sf_mean, mid_line(sf_mean), color="C2")
ax.set_xscale("log")
ax.set_yscale("log")

