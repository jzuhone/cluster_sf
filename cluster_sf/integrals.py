import numpy as np

from scipy.integrate import quad, quad_vec, dblquad
from scipy.special import jv
from cluster_sf.constants import c_s, sigma_xrism, pixel_width, r_c
from numba import njit

kmin = 0.0
kmax = 50.0


@njit
def P3D(k, C, l_dis, l_inj, alpha, n):
    ret = C * (1.0 + (k * l_inj) ** 2) ** (0.5 * alpha) * np.exp(-((k * l_dis) ** n))
    return ret


@njit
def eps(k, R):
    c = np.sqrt(r_c*r_c + R*R)
    return np.exp(-2.0 * np.pi * c * k) * (2.0 * np.pi * c * k + 1.0)


@njit
def P2D_int(k_z, k, C, l_dis, l_inj, alpha, n, R):
    kk = np.sqrt(k * k + k_z * k_z)
    return eps(k_z, R) ** 2 * P3D(kk, C, l_dis, l_inj, alpha, n)


def P2D(k, C, l_dis, l_inj, alpha, n, R):
    return (
        2.0
        * quad_vec(
            P2D_int,
            kmin,
            kmax,
            args=(k, C, l_dis, l_inj, alpha, n, R),
            points=(1.0, 30.0),
            epsrel=1.0e-5,
        )[0]
    )


@njit
def Ek(k, C, l_dis, l_inj, alpha, n):
    return 4.0 * np.pi * P3D(k, C, l_dis, l_inj, alpha, n) * k * k


def getC(mach, l_dis, l_inj, alpha, n=2):
    int_lower = quad(
        Ek,
        kmin,
        kmax,
        args=(1.0, l_dis, l_inj, alpha, n),
        points=(1.0, 30.0),
    )[0]
    Cn = (mach * c_s) ** 2 / int_lower / 3.0
    return Cn


def sigma_int(k, C, l_dis, l_inj, alpha, n, R):
    # Pg = (
    #    np.exp(-4.0 * np.pi**2 * k * k * sigma_xrism**2) * np.sinc(pixel_width * k) ** 2
    # )
    Pg = 1.0
    return 2.0 * np.pi * P2D(k, C, l_dis, l_inj, alpha, n, R) * Pg * k


def sigma(Cn=1.0, l_dis=1.0e-3, l_inj=1.0, alpha=-11.0 / 3.0, n=2.0, R=0.0):
    int1 = quad(
        Ek,
        kmin,
        kmax,
        args=(Cn, l_dis, l_inj, alpha, n),
        points=(1.0, 30.0),
        epsrel=1.0e-5,
    )[0]
    int2 = quad(
        sigma_int,
        kmin,
        kmax,
        args=(Cn, l_dis, l_inj, alpha, n, R),
        points=(1.0, 30.0),
        epsrel=1.0e-5,
    )[0]
    return int1 - int2


def SF_int(k, r, C, l_dis, l_inj, alpha, n, R):
    Pg = (
        np.exp(-4.0 * np.pi**2 * k * k * sigma_xrism**2) * np.sinc(pixel_width * k) ** 2
    )
    ret = (
        4.0
        * np.pi
        * (1.0 - jv(0, 2.0 * np.pi * k * r))
        * P2D(k, C, l_dis, l_inj, alpha, n, R)
        * k
        * Pg
    )
    return ret


def SF(x, Cn=1.0, l_dis=1.0e-3, l_inj=1.0, alpha=-11.0 / 3.0, n=2.0, R=0.0):
    int_upper = np.array(
        [
            quad(
                SF_int,
                kmin,
                kmax,
                args=(r, Cn, l_dis, l_inj, alpha, n, R),
                points=(1.0, 30.0),
                epsrel=1.0e-5,
            )[0]
            for r in x
        ]
    )
    return int_upper


r = np.linspace(30e-3, 2.0, 100)


def v_z(k, mach, l_dis, l_inj, alpha, n=2):
    C = getC(mach, l_dis, l_inj, alpha, n=n)
    return np.sqrt(4.0 * np.pi * P3D(k, C, l_dis, l_inj, alpha, n) * k**3)
