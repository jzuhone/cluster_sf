import numpy as np
from regions import Regions
from collections import defaultdict
from astropy.stats import scott_bin_width
from tqdm.auto import tqdm

def get_seps(regionfile):
    pts = Regions.read(regionfile)

    n_pts = len(pts)

    seps = {}
    for i in tqdm(range(n_pts), desc="Getting separations:", leave=True):
        for j in range(n_pts):
            key = (i, j)
            if key in seps or key[::-1] in seps or i == j:
                continue
            seps[key] = pts[i].center.separation(pts[j].center).to_value("arcmin")

    return pts, seps


def find_bin_edges(seps):
    sep_values = np.array(list(seps.values()))
    edges = scott_bin_width(sep_values, return_bins=True)[1]
    edges[0] *= 0.99
    edges[-1] *= 1.01
    return np.array([edges[:-1], edges[1:]]).T


def bin_pts(seps, edges=None):
    if edges is None:
        edges = find_bin_edges(seps)
    bins = defaultdict(list)
    bin_idxs = defaultdict(list)
    bins_used = []
    for key, value in seps.items():
        for j, edge in enumerate(edges):
            if edge[0] <= value <= edge[1]:
                bins[j].append(value)
                bin_idxs[j].append(key)
                if j not in bins_used:
                    bins_used.append(j)

    bins_used.sort()

    return bins, bin_idxs, bins_used, edges


def make_bins(regfile, edges=None):
    pts, seps = get_seps(regfile)
    bins, bin_idxs, bins_used, edges = bin_pts(seps, edges=edges)
    print(
        "num_seps = ", len(seps), " len_bins = ", sum([len(v) for v in bins.values()])
    )
    return pts, seps, bins, bin_idxs, bins_used, edges
