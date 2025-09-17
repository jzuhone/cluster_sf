import numpy as np
from collections import defaultdict
from scipy.optimize import least_squares
from cluster_sf.constants import angular_scale, sf_stat_err, alpha0, l_min0
from astropy.table import Table
from cluster_sf.integrals import getC, SF, sigma, sig_var
from cluster_sf.utils import make_sf_err_func
import argparse
from mpi4py import MPI
from tqdm.auto import tqdm

stat_err_sig = np.array([25.0, 25.0, 25.0, 25.0, 25.0, 39.0])

n = 2


def modify_params(params, l_inj_ini, free_params):
    p = params.copy()
    i_free = 1
    if "l_inj" in free_params:
        l_inj = p[i_free]
        i_free += 1
    else:
        l_inj = l_inj_ini
    if "l_dis" in free_params:
        l_dis = p[i_free]
        i_free += 1 
    else:
        l_dis = l_min0
    if "alpha" in free_params:
        alpha = p[i_free]
    else:
        alpha = alpha0
    return [p[0], l_inj, l_dis, alpha]
    

def make_nll(prefix, l_inj_ini, free_params, no_sig):

    sf_err_min = make_sf_err_func(f"{prefix}_sf_err_fit_params_min.dat")
    sf_err_max = make_sf_err_func(f"{prefix}_sf_err_fit_params_max.dat")

    def _comp_models(params, x, y1, y2):
        n_pts = len(y2)
        mach, l_inj, l_dis, alpha = modify_params(params, l_inj_ini, free_params)
        p_out = np.array([mach, l_inj, l_dis, alpha])
        Cn = getC(mach, l_dis, l_inj, alpha, n)
        y_model1 = SF(x, Cn=Cn, l_dis=l_dis, l_inj=l_inj, alpha=alpha, n=n)
        y_model1 += 2.0 * sf_stat_err**2
        y_model2 = np.sqrt([sigma(Cn=Cn, l_dis=l_dis, l_inj=l_inj, alpha=alpha, n=n)]*n_pts)
        dy1_neg = np.float64(y1 - y_model1 > 0.0)
        sig1 = sf_err_max(y_model1) * dy1_neg + (1.0 - dy1_neg) * sf_err_min(y_model1)
        sig2 = sig_var(Cn, l_dis, l_inj, alpha, n, 0.0) ** 0.25
        sig2 = np.sqrt(sig2*sig2+stat_err_sig[:n_pts]**2)
        return p_out, y_model1, sig1, y_model2, sig2
    
    def _nll(params, x, y1, y2, cm_func):
        _, y_model1, sig1, y_model2, sig2 = cm_func(params, x, y1, y2)
        ret = (y1 - y_model1) / sig1
        if not no_sig:
            ret = np.concatenate([ret, (y2-y_model2) / sig2])
        #print((ret**2).sum())
        return ret

    return _nll, _comp_models, sf_err_min, sf_err_max


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str)
    parser.add_argument("free_params", type=str)
    parser.add_argument("l_inj", type=float)
    parser.add_argument("--added", action="store_true")
    parser.add_argument("--no_sig", action="store_true")

    args = parser.parse_args()

    prefix = args.prefix
    l_inj_ini = args.l_inj
    no_sig = args.no_sig
    added = args.added

    init_dict = {"mach": 0.3, "l_inj": l_inj_ini, "l_dis": l_min0, "alpha": alpha0}

    free_params = [k for k in init_dict if k in args.free_params.split(",")]

    bounds = {
        "mach": (0.05, 2.0),
        "l_inj": (0.1, 5.0),
        "l_dis": (0.5e-3, 0.3),
        "alpha": (-8.0, -2.5),
    }

    initial = np.array([init_dict[p] for p in free_params])

    lsq_bounds = ([bounds[p][0] for p in free_params],
                  [bounds[p][1] for p in free_params])
    
    np1 = 200
    p1_bins = np.linspace(bounds["mach"][0], bounds["mach"][1], np1 + 1)
    p1_mid = 0.5 * (p1_bins[1:] + p1_bins[:-1])

    np2 = 200 if len(free_params) > 1 else 1
    if np2 != 1:
        p2_bins = np.linspace(bounds[free_params[1]][0], bounds[free_params[1]][1], np2 + 1)
        p2_mid = 0.5 * (p2_bins[1:] + p2_bins[:-1])
    else:
        p2_mid = None
    
    np3 = 200 if len(free_params) > 2 else 1
    if np3 != 1:
        p3_bins = np.linspace(bounds[free_params[2]][0], bounds[free_params[2]][1], np3 + 1)
        p3_mid = 0.5 * (p3_bins[1:] + p3_bins[:-1])
    else:
        p3_mid = None

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Get the rank of the current process
    size = comm.Get_size()  # Get the total number of processes

    if prefix == "three_pts":
        n_pts = 6
    elif prefix == "two_pts":
        n_pts = 4
        
    added_str = "_added" if added else ""
    sig_str = "_nosig" if no_sig else ""
    orig_str = "_orig" if n_pts == 4 else ""
    t = Table.read(f"SF_observed{orig_str}{added_str}.dat", format="ascii.commented_header")
    x = t["r"].data * angular_scale.value
    y1 = t["SF"].data
    t2 = Table.read("sigma_observed.dat", format="ascii.commented_header")
    y2 = t2["sigma"].data[:n_pts]

    nll, comp_models, sf_err_min, sf_err_max = make_nll(prefix, l_inj_ini, free_params, no_sig)
    
    def get_results(result_out, params, y_sf, y_sig):
        p_out, y_model1, sig1, y_model2, sig2 = comp_models(result_out, x, y_sf, y_sig)
        cost1 = np.sum((y_sf - y_model1) ** 2 / sig1**2)
        cost2 = np.sum((y_sig - y_model2) ** 2 / sig2**2)
        params["mach"].append(p_out[0])
        params["l_inj"].append(p_out[1])
        params["l_dis"].append(p_out[2])
        params["alpha"].append(p_out[3])
        params["cost"].append(cost1+cost2)
        params["cost_sigma"].append(cost2)
        for i in range(len(y_model2)):
            params[f"sigma_{i}"].append(y_model2[i])
        return y_model1, y_model2, sig2
        
    if rank == 0:
        p = defaultdict(list)
        result = least_squares(
            nll,
            initial,
            bounds=lsq_bounds,
            args=(x, y1, y2, comp_models),
        )
        y_model1, y_model2, sig2 = get_results(result.x, p, y1, y2)
        print(free_params, p)
        tsf = Table({"sf_avg": y_model1, "sf_min": sf_err_min(y_model1), "sf_max": sf_err_max(y_model1)})
        tsf.write(
            f"{prefix}_{'_'.join(free_params)}_l_inj{l_inj_ini}{added_str}{sig_str}_sf_model.dat",
            format="ascii.commented_header",
            overwrite=True,
        )
        tsig = Table({"sig_avg": y_model2, "sig_err": sig2})
        tsig.write(
            f"{prefix}_{'_'.join(free_params)}_l_inj{l_inj_ini}{added_str}{sig_str}_sig_model.dat",
            format="ascii.commented_header",
            overwrite=True,
        )
    else:
        p = None

    p = comm.bcast(p, root=0)

    loop_range = range(np1)

    # Divide the workload among processes
    local_start = rank * len(loop_range) // size
    local_end = (rank + 1) * len(loop_range) // size
    local_range = loop_range[local_start:local_end]

    lp = defaultdict(list)

    for k in tqdm(local_range, desc=f"Process {rank}", position=rank, leave=True):
        for j in range(np2):
            for i in range(np3):
                p_in = [p1_mid[k]]
                if np2 > 1:
                    p_in.append(p2_mid[j])
                if np3 > 1:
                    p_in.append(p3_mid[i])
                _ = get_results(p_in, lp, y1, y2)

    all_lp = comm.gather(lp, root=0)
    
    if rank == 0:
        for key in lp.keys():
            for proc_lp in all_lp:
                p[key].extend(proc_lp[key])

        tp = Table(p)
        tp.write(
            f"{prefix}_{'_'.join(free_params)}_l_inj{l_inj_ini}{added_str}{sig_str}_params.dat",
            format="ascii.commented_header",
            overwrite=True,
        )


if __name__ == "__main__":
    main()
