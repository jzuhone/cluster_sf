"""


def make_sf_int(kk, sky_func):
    def _sf_int(k, C, l_dis, l_inj, alpha, n):
        p = P3D(k, C, l_dis, l_inj, alpha, n)
        emiss = np.interp(k, kk, sky_func)
        return p*emiss
    return _sf_int


def SF(sf_ints, Cn, l_dis, l_inj, alpha, n):
    int_upper = np.array(
        [
            quad(
                sf_int,
                0.0,
                10.0,
                args=(Cn, l_dis, l_inj, alpha, n),
                points=(0.1, 1.0, 5),
                epsrel=1.0e-5,
            )[0]
            for sf_int in sf_ints
        ]
    )
    return int_upper
"""
