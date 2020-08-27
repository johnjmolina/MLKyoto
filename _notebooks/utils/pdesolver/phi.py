import numpy as np
from mpmath import mp, mpf


def phi0(x0, *, dps=16):  # exp(x), |x| > 1e-32 -> dps > 16
    with mp.workdps(dps):
        y = mp.matrix([mp.exp(x) for x in x0])
        return np.array(y.tolist(), dtype=x0.dtype)[:, 0]


def phi1(x0, *, dps=16):  # (exp(x) - 1)/x, |x| > 1e-32 -> dps > 16
    with mp.workdps(dps):
        y = mp.matrix([mp.fdiv(mp.expm1(x), x) if x !=
                       0.0 else mpf('1') for x in x0])
        return np.array(y.tolist(), dtype=x0.dtype)[:, 0]


def phi2(x0, *, dps=50):  # (exp(x) - 1 - x) / (x * x), |x| > 1e-32 -> dps > 40
    with mp.workdps(dps):
        y = mp.matrix([mp.fdiv(mp.fsub(mp.expm1(x), x), mp.fmul(
            x, x)) if x != 0.0 else mpf(1)/mpf(2) for x in x0])
        return np.array(y.tolist(), dtype=x0.dtype)[:, 0]


def phi3(x0, *, dps=100):  # (epx(x)-1-x-0.5*x*x)/(x*x*x) , |x| > 1e-32 -> dps > 100
    with mp.workdps(dps):
        y = mp.matrix([mp.fdiv(mp.fsub(mp.fsub(mp.expm1(x), x), mp.fmul('0.5', mp.fmul(x, x))),
                               mp.power(x, '3')) if x != 0.0 else mpf(1)/mpf(6) for x in x0])
        return np.array(y.tolist(), dtype=x0.dtype)[:, 0]


def phin(n, z, *, dps=100):
    if n == 0:
        return phi0(z, dps=dps)
    elif n == 1:
        return phi1(z, dps=dps)
    elif n == 2:
        return phi2(z, dps=dps)
    elif n == 3:
        return phi3(z, dps=dps)
    else:
        print("Error!")
        return


def phi_a52(*, phi2_dt, phi3_dt, phi2_hdt, phi3_hdt):
    return 0.5*(phi2_hdt-phi3_hdt) + 0.25*phi2_dt - phi3_dt
