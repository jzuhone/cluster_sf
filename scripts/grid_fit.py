from scipy.integrate import quad, quad_vec
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.optimize import minimize, least_squares
from utils import sigma_xrism, pixel_width, compute_sigma, angular_scale
from astropy.table import Table
import unyt as u
from observed_utils import *
from astropy.table import Table
from tqdm.auto import tqdm
import json
import sys
from IPython import embed
import argparse
from mpi4py import MPI
from tqdm.auto import tqdm

n = 2


def min_line(x):
    return 0.44256785 * (x / 1.83324546) ** 1.11136842


def mid_line(x):
    return 0.38639482 * (x / 1.83344618) ** 1.18724518


def max_line(x):
    return 0.35638803 * (x / 1.73599161) ** 1.23588083


def nll(params, res, x, y, y2, no_sig, l_inj_ini):
    if res in [1, 2]:
        mach, l_inj, alpha = params.copy()
        l_dis = l_min0
    elif res in [3, 4]:
        mach, l_inj = params.copy()
        alpha = alpha0
        l_dis = l_min0
    elif res in [5, 6, 14]:
        mach, l_inj = params.copy()
        l_dis = l_min0
        alpha = alpha0
    elif res in [7, 8]:
        mach, alpha = params.copy()
        l_dis = l_min0
        l_inj = l_inj_ini
    elif res in [9, 10]:
        mach, l_dis = params.copy()
        l_inj = l_inj_ini
        alpha = alpha0
    elif res in [11, 12, 13]:
        mach = params.copy()[0]
        l_inj = l_inj_ini
        alpha = alpha0
        l_dis = l_min0
    Cn = getC(mach, l_dis, l_inj, alpha, n)
    y_model1 = SF(x, Cn=Cn, l_dis=l_dis, l_inj=l_inj, alpha=alpha, n=n)
    y_model1 += 2.0 * sf_stat_err**2
    y_model2 = np.sqrt(sigma(Cn=Cn, l_dis=l_dis, l_inj=l_inj, alpha=alpha, n=n))
    dy = y - y_model1
    dy2 = y2 - np.array([y_model2] * 2)
    dy_neg = np.float64(dy > 0.0)
    # sig = y_min * dy_neg + (1.0 - dy_neg) * y_max
    sig = max_line(y_model1) * dy_neg + (1.0 - dy_neg) * min_line(y_model1)
    cv2 = sig_var(Cn, l_dis, l_inj, alpha, n) ** 0.5
    sig2 = np.array([np.sqrt(cv2 + 900)] * 2)
    # print(sig2)
    # sig = y_mid
    # print(dy**2, sig**2, dy2**2, y2, y_model2, y2err**2)
    # ret1 = np.sum(np.log(2.0 * np.pi * sig**2) + dy**2 / sig**2)
    # ret1 = np.sum(dy**2 / sig**2)
    ret = dy / sig
    # print(y2err)
    # print(dy2**2, y2err**2)
    if not no_sig:
        ret = np.concatenate([ret, dy2 / sig2])
    # print(dy / sig, dy2/y2err)
    # print(ret, np.sum(ret**2))
    # else:
    #    ret2 = 0.0
    # ret = ret1 + ret2
    # print(ret, ret1, ret2)  # , y_model1, y, sig)
    # print(ret)
    # print(y, y_model1, dy, sig)
    return ret


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("res", type=int)
    parser.add_argument("l_inj", type=float)
    parser.add_argument("--added", action="store_true")
    parser.add_argument("--no_sig", action="store_true")

    args = parser.parse_args()

    # res = int(sys.argv[1])
    l_inj_ini = args.l_inj
    res = args.res

    no_sig = args.no_sig

    """
    bounds = {
        "mach": [0.05, 2.0],
        "l_inj": [0.2, 10.0],
        "l_dis": [0.5e-3, 2.0],
        "alpha": [-8.0, -2.5],
    }
    """
    mach_min, mach_max = 0.05, 2.0
    l_inj_min, l_inj_max = 0.1, 5.0
    l_dis_min, l_dis_max = 0.5e-3, 0.3
    alpha_min, alpha_max = -8.0, -2.5

    np3 = 1

    if res in [1, 2]:
        initial = np.array([0.3, l_inj_ini, alpha0])
        bounds = ([mach_min, l_inj_min, alpha_min], [mach_max, l_inj_max, alpha_max])
        np2 = 200
        np3 = 200
    if res in [3, 4]:
        initial = np.array([0.3, l_inj_ini])
        bounds = ([mach_min, l_inj_min], [mach_max, l_inj_max])
        np2 = 200
    elif res in [5, 6, 14]:
        no_sig = res != 14
        initial = np.array([0.3, l_inj_ini])
        bounds = ([mach_min, l_inj_min], [mach_max, l_inj_max])
        np2 = 200
    elif res in [7, 8]:
        initial = np.array([0.3, alpha0])
        bounds = ([mach_min, alpha_min], [mach_max, alpha_max])
        np2 = 200
    elif res in [9, 10]:
        initial = np.array([0.3, 0.2])
        bounds = ([mach_min, l_dis_min], [mach_max, l_inj_ini])
        np2 = 200
    elif res in [11, 12, 13]:
        no_sig = res != 13
        initial = np.array([0.3])
        bounds = ([mach_min], [mach_max])
        np2 = 1
    if res in [2, 4, 6, 8, 10, 12, 13, 14]:
        added = True
    else:
        added = False

    np1 = 200
    p1_bins = np.linspace(bounds[0][0], bounds[1][0], np1 + 1)
    p1_mid = 0.5 * (p1_bins[1:] + p1_bins[:-1])

    if np2 != 1:
        p2_bins = np.linspace(bounds[0][1], bounds[1][1], np2 + 1)
        p2_mid = 0.5 * (p2_bins[1:] + p2_bins[:-1])
    else:
        p2_bins = p2_mid = None

    if np3 != 1:
        p3_bins = np.linspace(bounds[0][2], bounds[1][2], np3 + 1)
        p3_mid = 0.5 * (p3_bins[1:] + p3_bins[:-1])
    else:
        p3_bins = p3_mid = None

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Get the rank of the current process
    size = comm.Get_size()  # Get the total number of processes

    added_str = "_added" if added else ""
    t = Table.read(f"SF_observed{added_str}.dat", format="ascii.commented_header")
    # t = Table.read("SF_lmax_500_lmin_1_M0.4.dat", format="ascii.commented_header")
    sigma1, sigma2 = 200.0, 200.0
    # sigma1, sigma2 = 334.0, 340.0
    sigma_std = np.sqrt(60.0**2 + sigma_stat_err**2)
    x = t["r"].data * angular_scale.value
    y = t["SF"].data  # - 2.0 * sf_stat_err**2
    # y_min = min_line(t["SF"].data)
    # y_mid = mid_line(t["SF"].data)
    # y_max = max_line(t["SF"].data)
    # y2 = np.array([sigma1 * sigma1, sigma2 * sigma2])
    # y2err = 2.0 * np.sqrt(y2) * sigma_std
    y2 = np.array([sigma1, sigma2])

    def get_results(y_sf, y_sig, params):
        result = least_squares(
            nll,
            initial,
            bounds=bounds,
            args=(res, x, y_sf, y_sig, no_sig, l_inj_ini),
        )
        if res in [1, 2]:
            mach, l_inj, alpha = result.x
            l_dis = l_min0
        elif res in [3, 4]:
            mach, l_inj = result.x
            l_dis = l_min0
            alpha = alpha0
        elif res in [5, 6, 14]:
            mach, l_inj = result.x
            l_dis = l_min0
            alpha = alpha0
        elif res in [7, 8]:
            mach, alpha = result.x
            l_dis = l_min0
            l_inj = l_inj_ini
        elif res in [9, 10]:
            mach, l_dis = result.x
            alpha = alpha0
            l_inj = l_inj_ini
        elif res in [11, 12, 13]:
            mach = result.x[0]
            alpha = alpha0
            l_inj = l_inj_ini
            l_dis = l_min0
        Cn = getC(mach, l_dis, l_inj, alpha, n)
        y_model2 = np.sqrt(
            sigma(
                Cn=Cn,
                l_dis=l_dis,
                l_inj=l_inj,
                alpha=alpha,
                n=n,
            )
        )
        ym2 = np.array([y_model2] * 2)
        cv2 = sig_var(Cn, l_dis, l_inj, alpha, n) ** 0.5
        sig2 = np.array([np.sqrt(cv2 + 900)] * 2)
        params["mach"].append(mach)
        params["l_inj"].append(l_inj)
        params["alpha"].append(alpha)
        params["l_dis"].append(l_dis)
        params["cost"].append(2.0 * result.cost)
        params["cost_sigma"].append(np.sum((y_sig - ym2) ** 2 / sig2**2))
        params["sigma"].append(y_model2)
        params["sigma_err"].append(sig2[0])

    if rank == 0:
        p = defaultdict(list)
        get_results(y, y2, p)
        print(args.res, p)
    else:
        p = None
    p = comm.bcast(p, root=0)

    """
    loop_range = range(np1)

    # Divide the workload among processes
    local_start = rank * len(loop_range) // size
    local_end = (rank + 1) * len(loop_range) // size
    local_range = loop_range[local_start:local_end]

    lp = defaultdict(list)

    for k in tqdm(local_range, desc=f"Process {rank}", position=rank, leave=True):
        mach = p1_mid[k]
        for j in range(np2):
            if res in [1, 2]:
                l_inj = p2_mid[j]
                for i in range(np3):
                    alpha = p3_mid[i]
                    l_dis = l_min0
                    params = np.array([mach, l_inj, alpha])
                    resids = nll(params, res, x, y, y2, no_sig, l_inj_ini)
                    Cn = getC(mach, l_dis, l_inj, alpha, n)
                    y_model2 = np.sqrt(
                        sigma(
                            Cn=Cn,
                            l_dis=l_dis,
                            l_inj=l_inj,
                            alpha=alpha,
                            n=n,
                        )
                    )
                    cost = np.sum(resids**2)
                    lp["mach"].append(mach)
                    lp["l_inj"].append(l_inj)
                    lp["alpha"].append(alpha)
                    lp["l_dis"].append(l_dis)
                    lp["cost"].append(cost)
                    lp["cost_sigma"].append(np.sum(resids[-2:] ** 2))
                    lp["sigma"].append(y_model2)
            else:
                if res in [3, 4]:
                    l_inj = p2_mid[j]
                    l_dis = l_min0
                    alpha = alpha0
                    params = np.array([mach, l_inj])
                elif res in [5, 6, 14]:
                    l_inj = p2_mid[j]
                    l_dis = l_min0
                    alpha = alpha0
                    params = np.array([mach, l_inj])
                elif res in [7, 8]:
                    alpha = p2_mid[j]
                    l_dis = l_min0
                    l_inj = l_inj_ini
                    params = np.array([mach, alpha])
                elif res in [9, 10]:
                    l_dis = p2_mid[j]
                    alpha = alpha0
                    l_inj = l_inj_ini
                    params = np.array([mach, l_dis])
                elif res in [11, 12, 13]:
                    alpha = alpha0
                    l_inj = l_inj_ini
                    l_dis = l_min0
                    params = np.array([mach])
                resids = nll(params, res, x, y, y2, no_sig, l_inj_ini)
                Cn = getC(mach, l_dis, l_inj, alpha, n)
                y_model2 = np.sqrt(
                    sigma(
                        Cn=Cn,
                        l_dis=l_dis,
                        l_inj=l_inj,
                        alpha=alpha,
                        n=n,
                    )
                )
                cost = np.sum(resids**2)
                cv2 = sig_var(Cn, l_dis, l_inj, alpha, n) ** 0.5
                sig2 = np.sqrt(cv2 + 900)
                lp["mach"].append(mach)
                lp["l_inj"].append(l_inj)
                lp["alpha"].append(alpha)
                lp["l_dis"].append(l_dis)
                lp["cost"].append(cost)
                lp["cost_sigma"].append(np.sum(resids[-2:] ** 2))
                lp["sigma"].append(y_model2)
                lp["sigma_err"].append(sig2)

    all_lp = comm.gather(lp, root=0)
    
    if rank == 0:
        for key in lp.keys():
            for proc_lp in all_lp:
                p[key].extend(proc_lp[key])

        tout = Table(p)
        tout.write(
            f"params_res{res}_l_inj{l_inj_ini}_try.dat",
            format="ascii.commented_header",
            overwrite=True,
        )
    """


if __name__ == "__main__":
    main()
