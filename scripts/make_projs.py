import os
from field_kit import GaussianRandomField, plaw_with_cutoffs
import numpy as np
from cluster_sf.utils import compute_sigma, make_em
import sys
from tqdm.auto import tqdm
import h5py

make_fluct = True

l_max = float(sys.argv[1])
l_min = float(sys.argv[2])
mach = float(sys.argv[3])
if len(sys.argv) > 4:
    alpha = float(sys.argv[4])
    alpha_str = f"_a{-alpha}"
else:
    alpha = -11.0 / 3.0
    alpha_str = "_a3.67"

nx, ny, nz = (256,) * 3
Lx, Ly, Lz = (2000.0,) * 3
num_tries = 500

le = [-0.5 * Lx, -0.5 * Ly, -0.5 * Lz]
re = [0.5 * Lx, 0.5 * Ly, 0.5 * Lz]
ddims = (nx, ny, nz)
V_rms = compute_sigma(mach)
V_rms1D = V_rms / np.sqrt(3)
mach1D = mach / np.sqrt(3.0)
print(V_rms, V_rms1D)

power_spec_v = plaw_with_cutoffs(l_min, l_max, -11.0 / 3.0)
power_spec_v.renormalize(V_rms1D)

vgen = GaussianRandomField(le, re, ddims, power_spec_v)

if make_fluct:
    sigma_s = np.sqrt(np.log(1.0 + mach1D**2))
    power_spec_s = plaw_with_cutoffs(l_min, l_max, -6.0)
    power_spec_s.renormalize(sigma_s)
    dgen = GaussianRandomField(le, re, ddims, power_spec_s)
    fluct_str = "_fluct"
else:
    fluct_str = ""
    sigma_s = 0.0

EM = make_em(*(vgen.delta * vgen.ddims), *vgen.ddims)
w = np.sum(EM, axis=0)

prefix = (
    f"/scratch2/jzuhone/data/coma_cubes/lmin_{int(l_min)}_M{mach}{alpha_str}{fluct_str}"
)
os.makedirs(prefix, exist_ok=True)

pbar = tqdm(leave=True, total=num_tries, desc="Generating field realizations ")
for i in range(num_tries):
    v = vgen.generate_vector_field_realization()
    #print(np.std(vx), np.std(vy), np.std(vz))
    if make_fluct:
        s = dgen.generate_vector_field_realization()
        v *= np.exp(s)
        v *= V_rms1D/np.std(v, axis=(1,2,3))
    vEM = v * EM
    vx = np.sum(vEM, axis=0) / w
    vy = np.sum(vEM, axis=1) / w
    vz = np.sum(vEM, axis=2) / w
    v2x = np.sum(v * vEM, axis=0) / w
    v2y = np.sum(v * vEM, axis=1) / w
    v2z = np.sum(v * vEM, axis=2) / w
    with h5py.File(f"{prefix}/lmax_{int(l_max)}_proj_field_{i}.h5", "w") as f:
        d = f.create_dataset("x", data=vgen.x)
        d.attrs["units"] = "kpc"
        d = f.create_dataset("y", data=vgen.y)
        d.attrs["units"] = "kpc"
        d = f.create_dataset("z", data=vgen.z)
        d.attrs["units"] = "kpc"
        d = f.create_dataset("vx", data=vx)
        d.attrs["units"] = "km/s"
        d = f.create_dataset("vy", data=vy)
        d.attrs["units"] = "km/s"
        d = f.create_dataset("vz", data=vz)
        d.attrs["units"] = "km/s"
        d = f.create_dataset("v2x", data=v2x)
        d.attrs["units"] = "km**2/s**2"
        d = f.create_dataset("v2y", data=v2y)
        d.attrs["units"] = "km**2/s**2"
        d = f.create_dataset("v2z", data=v2z)
        d.attrs["units"] = "km**2/s**2"
    pbar.update()
pbar.close()
