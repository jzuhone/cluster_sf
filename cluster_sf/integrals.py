import numpy as np

from scipy.integrate import quad, quad_vec, dblquad
from scipy.special import jv
from constants import c_s, sigma_xrism, pixel_width
from numba import njit

kmin = 0.0
kmax = 50.0

sf_stat_err = 30.0
sigma_stat_err = 30.0

r_c = 0.3


l_min0 = 0.001
l_max0 = 1.0
alpha0 = -11.0 / 3.0



@njit
def P3D(k, C, l_dis, l_inj, alpha, n):
    ret = C * (1.0 + (k * l_inj) ** 2) ** (0.5 * alpha) * np.exp(-((k * l_dis) ** n))
    return ret


@njit
def eps(k):
    return np.exp(-2.0 * np.pi * r_c * k) * (2.0 * np.pi * r_c * k + 1.0)


@njit
def P2D_int(k_z, k, C, l_dis, l_inj, alpha, n):
    kk = np.sqrt(k * k + k_z * k_z)
    return eps(k_z) ** 2 * P3D(kk, C, l_dis, l_inj, alpha, n)


@njit
def sig_var_int(k1, k2, C, l_dis, l_inj, alpha, n):
    P1 = P3D(k1, C, l_dis, l_inj, alpha, n)
    P2 = P3D(k2, C, l_dis, l_inj, alpha, n)
    epsp = eps(k1 + k2)
    epsm = eps(k1) * eps(k2)
    return 2.0 * P1 * P2 * (epsp - epsm) ** 2


def sig_var(C, l_dis, l_inj, alpha, n):
    return dblquad(
        sig_var_int,
        kmin,
        kmax,
        kmin,
        kmax,
        args=(C, l_dis, l_inj, alpha, n),
        epsrel=1.0e-5,
    )[0]


def sig_var_test(mach, l_dis, l_inj, alpha, n):
    C = getC(mach, l_dis, l_inj, alpha, n=n)
    return sig_var(C, l_dis, l_inj, alpha, n)


def sig_var_test2(mach, l_dis, l_inj, alpha, n):
    C = getC(mach, l_dis, l_inj, alpha, n=n)
    i1 = dblquad(
        sig_var_int,
        kmin,
        1.0,
        kmin,
        1.0,
        args=(C, l_dis, l_inj, alpha, n),
        epsrel=1.0e-5,
    )[0]
    i2 = dblquad(
        sig_var_int,
        kmin,
        1.0,
        1.0,
        30.0,
        args=(C, l_dis, l_inj, alpha, n),
        epsrel=1.0e-5,
    )[0]
    i3 = dblquad(
        sig_var_int,
        kmin,
        1.0,
        30.0,
        50.0,
        args=(C, l_dis, l_inj, alpha, n),
        epsrel=1.0e-5,
    )[0]
    i4 = dblquad(
        sig_var_int,
        1.0,
        30.0,
        kmin,
        1.0,
        args=(C, l_dis, l_inj, alpha, n),
        epsrel=1.0e-5,
    )[0]
    i5 = dblquad(
        sig_var_int,
        1.0,
        30.0,
        1.0,
        30.0,
        args=(C, l_dis, l_inj, alpha, n),
        epsrel=1.0e-5,
    )[0]
    i6 = dblquad(
        sig_var_int,
        1.0,
        30.0,
        30.0,
        50.0,
        args=(C, l_dis, l_inj, alpha, n),
        epsrel=1.0e-5,
    )[0]
    i7 = dblquad(
        sig_var_int,
        30.0,
        50.0,
        kmin,
        1.0,
        args=(C, l_dis, l_inj, alpha, n),
        epsrel=1.0e-5,
    )[0]
    i8 = dblquad(
        sig_var_int,
        30.0,
        50.0,
        1.0,
        30.0,
        args=(C, l_dis, l_inj, alpha, n),
        epsrel=1.0e-5,
    )[0]
    i9 = dblquad(
        sig_var_int,
        30.0,
        50.0,
        30.0,
        50.0,
        args=(C, l_dis, l_inj, alpha, n),
        epsrel=1.0e-5,
    )[0]
    return i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9


def P2D(k, C, l_dis, l_inj, alpha, n):
    return (
        2.0
        * quad_vec(
            P2D_int,
            kmin,
            kmax,
            args=(k, C, l_dis, l_inj, alpha, n),
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


def v_z(k, mach, l_dis, l_inj, alpha, n=2):
    C = getC(mach, l_dis, l_inj, alpha, n=n)
    return np.sqrt(4.0 * np.pi * P3D(k, C, l_dis, l_inj, alpha, n) * k**3)


def SF_int(k, r, C, l_dis, l_inj, alpha, n):
    Pg = (
        np.exp(-4.0 * np.pi**2 * k * k * sigma_xrism**2) * np.sinc(pixel_width * k) ** 2
    )
    ret = (
        4.0
        * np.pi
        * (1.0 - jv(0, 2.0 * np.pi * k * r))
        * P2D(k, C, l_dis, l_inj, alpha, n)
        * k
        * Pg
    )
    return ret


def sigma_int(k, C, l_dis, l_inj, alpha, n):
    # Pg = (
    #    np.exp(-4.0 * np.pi**2 * k * k * sigma_xrism**2) * np.sinc(pixel_width * k) ** 2
    # )
    Pg = 1.0
    return 2.0 * np.pi * P2D(k, C, l_dis, l_inj, alpha, n) * Pg * k


def sigma(Cn=1.0, l_dis=1.0e-3, l_inj=1.0, alpha=-11.0 / 3.0, n=2.0):
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
        args=(Cn, l_dis, l_inj, alpha, n),
        points=(1.0, 30.0),
        epsrel=1.0e-5,
    )[0]
    return int1 - int2


def sigma_test(mach, l_dis, l_inj, alpha, n=2):
    C = getC(mach, l_dis, l_inj, alpha, n=n)
    return sigma(C, l_dis, l_inj, alpha, n)


def all_int(k, C, l_dis, l_inj, alpha, n):
    return 2.0 * np.pi * P2D(k, C, l_dis, l_inj, alpha, n) * k


def all2D(Cn=1.0, l_dis=1.0e-3, l_inj=1.0, alpha=-11.0 / 3.0, n=2.0):
    return quad(
        all_int,
        kmin,
        kmax,
        args=(Cn, l_dis, l_inj, alpha, n),
        points=(1.0, 30.0),
        epsrel=1.0e-5,
    )[0]


def SF(x, Cn=1.0, l_dis=1.0e-3, l_inj=1.0, alpha=-11.0 / 3.0, n=2.0):
    int_upper = np.array(
        [
            quad(
                SF_int,
                kmin,
                kmax,
                args=(r, Cn, l_dis, l_inj, alpha, n),
                points=(1.0, 30.0),
                epsrel=1.0e-5,
            )[0]
            for r in x
        ]
    )
    return int_upper


r = np.linspace(30e-3, 2.0, 100)


def min_line(x):
    return 0.44256785 * (x / 1.83324546) ** 1.11136842


def mid_line(x):
    return 0.38639482 * (x / 1.83344618) ** 1.18724518


def max_line(x):
    return 0.35638803 * (x / 1.73599161) ** 1.23588083
