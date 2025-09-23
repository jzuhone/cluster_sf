import numpy as np

from scipy.integrate import quad, quad_vec, dblquad
from scipy.special import jv
from cluster_sf.constants import c_s, sigma_xrism, det_width, r_c, kmin, kmax
from numba import njit


@njit
def P3D(k, C, l_dis, l_inj, alpha, n):
    ret = C * (1.0 + (k * l_inj) ** 2) ** (0.5 * alpha) * np.exp(-((k * l_dis) ** n))
    return ret


@njit
def W(k, reg_width):
    return np.exp(-0.5* k * k * sigma_xrism ** 2) * np.sinc(reg_width * k) ** 2


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


def sigma_int(k, C, l_dis, l_inj, alpha, n, R, reg_width):
    Pg = W(k, reg_width)**2
    return 2.0 * np.pi * P2D(k, C, l_dis, l_inj, alpha, n, R) * Pg * k


def sigma(Cn=1.0, l_dis=1.0e-3, l_inj=1.0, alpha=-11.0 / 3.0, n=2.0, R=0.0, reg_width=det_width):
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
        args=(Cn, l_dis, l_inj, alpha, n, R, reg_width),
        points=(1.0, 30.0),
        epsrel=1.0e-5,
    )[0]
    return int1 - int2


def SF_int(k, r, C, l_dis, l_inj, alpha, n, R):
    Pg = W(k, det_width)**2
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


@njit
def sig_var_int(k1, k2, C, l_dis, l_inj, alpha, n, R, reg_width):
    P1 = P3D(k1, C, l_dis, l_inj, alpha, n)
    P2 = P3D(k2, C, l_dis, l_inj, alpha, n)
    epsp = eps(k1 + k2, R) * W(k1+k2, reg_width)
    epsm = eps(k1, R) * eps(k2, R) * W(k1, reg_width) * W(k2, reg_width)
    return 2.0 * P1 * P2 * (epsp - epsm) ** 2


def sig_var(C, l_dis, l_inj, alpha, n, R, reg_width=det_width):
    return dblquad(
        sig_var_int,
        kmin,
        kmax,
        kmin,
        kmax,
        args=(C, l_dis, l_inj, alpha, n, R, reg_width),
        epsrel=1.0e-5,
    )[0]


r = np.linspace(30e-3, 2.0, 100)


def v_z(k, mach, l_dis, l_inj, alpha, n=2):
    C = getC(mach, l_dis, l_inj, alpha, n=n)
    return np.sqrt(4.0 * np.pi * P3D(k, C, l_dis, l_inj, alpha, n) * k**3)
