#!/usr/bin/env python
# coding: utf-8

# In[19]:


from cluster_sf.integrals import eps, W
from cluster_sf.constants import center_coord, angular_scale, kmin, kmax
from astropy.table import Table
import argparse
from cluster_sf.bins import make_bins
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from collections import defaultdict

# In[2]:


sys.argv = [""]


# In[3]:


parser = argparse.ArgumentParser()
parser.add_argument("prefix", type=str)
parser.add_argument("--edgesfile", type=str)

args = parser.parse_args(["two_pts"])

prefix = args.prefix
edgesfile = args.edgesfile
regfile = f"{prefix}.reg"
if edgesfile is None:
    edges = None
else:
    edges = np.loadtxt(edgesfile)

regs, seps, bins, bin_idxs, bins_used, edges = make_bins(regfile, edges=edges)

radii = np.array(
    [
        reg.center.separation(center_coord).to_value("arcmin") * angular_scale.value
        for reg in regs
    ]
)
widths = np.array([(reg.width*angular_scale).to_value("Mpc") for reg in regs])

averages = defaultdict(list)
for key, idxs in bin_idxs.items():
    uidxs = np.unique(idxs)
    averages["radii"].append(radii[uidxs].mean())
    averages["areas"].append((widths[uidxs]**2).mean())
    averages["nums"].append(uidxs.size)

t1 = Table({"radii": radii, "widths": widths})
t1.write(f"{prefix}_radii_widths.dat", format="ascii.commented_header", overwrite=True)

t2 = Table(averages)
t2.write(f"{prefix}_avg_rw.dat", format="ascii.commented_header", overwrite=True)

