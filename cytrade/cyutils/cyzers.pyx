import cython
cimport numpy as np
import numpy as np
from scipy.integrate import quad
from libc.math cimport log, exp, sqrt, M_PI, abs, INFINITY


@cython.boundscheck(False)
@cython.wraparound(False)
def minimize_scalar_bounded(func, tuple args=(), double[:] bounds=np.array([0., 2.]),
                             double xatol=1e-5, int maxiter=500):
    x1, x2 = bounds

    cdef double sqrt_eps = sqrt(2.2e-16)
    cdef double golden_mean = 0.5 * (3.0 - sqrt(5.0))
    a, b = x1, x2
    cdef double fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    cdef double rat, e = 0.0
    x = xf
    cdef double fx = func(x, *args)
    cdef int num = 1
    cdef double fu = INFINITY

    cdef double ffulc, fnfc = fx
    cdef double xm = 0.5 * (a + b)
    cdef double tol1 = sqrt_eps * abs(xf) + xatol / 3.0
    cdef double tol2 = 2.0 * tol1

    while (abs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1
        # Check for parabolic fit
        if abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((abs(p) < abs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = np.sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # do a golden-section step
                golden = 1

        if golden:  # do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e

        si = np.sign(rat) + (rat == 0)
        x = xf + si * np.maximum(abs(rat), tol1)
        fu = func(x, *args)
        num += 1

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxiter:
            break

    return xf


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double ll_minimize_scalar_bounded(func, tuple args=(), double[:] bounds=np.array([0., 2.]),
                             double xatol=1e-5, int maxiter=500):
    x1, x2 = bounds

    cdef double sqrt_eps = sqrt(2.2e-16)
    cdef double golden_mean = 0.5 * (3.0 - sqrt(5.0))
    a, b = x1, x2
    cdef double fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    cdef double rat, e = 0.0
    x = xf
    cdef double fx = func(x, *args)
    cdef int num = 1
    cdef double fu = INFINITY

    cdef double ffulc, fnfc = fx
    cdef double xm = 0.5 * (a + b)
    cdef double tol1 = sqrt_eps * abs(xf) + xatol / 3.0
    cdef double tol2 = 2.0 * tol1

    while (abs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1
        # Check for parabolic fit
        if abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((abs(p) < abs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = np.sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # do a golden-section step
                golden = 1

        if golden:  # do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e

        si = np.sign(rat) + (rat == 0)
        x = xf + si * np.maximum(abs(rat), tol1)
        fu = func(x, *args)
        num += 1

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxiter:
            break

    return xf



@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double qdr(double s, double mean, double f, double var):
    return log(1 + f * s) * exp(-((s - mean) ** 2) / (2 * var)) / (sqrt(2 * M_PI * var))


@cython.boundscheck(False)
@cython.wraparound(False)
def ll_norm_integral(double f, double mean, double std):
    cdef double s, val, er
    cdef double var = std ** 2

    val, er = quad(lambda s: qdr(s, mean, f, var),
                    mean - 3 * std, 
                    mean + 3 * std)
    return -val


cpdef np.ndarray[double] apply_kelly1(np.ndarray mean_col, np.ndarray std_col, double[:] bounds=np.array([0., 2.])):
    assert (mean_col.dtype == np.float_
            and std_col.dtype == np.float_)
    cdef Py_ssize_t i, n = len(mean_col)
    assert (len(mean_col) == len(std_col) == n)
    cdef np.ndarray[double] res = np.empty(n)
    for i in range(len(mean_col)):
        res[i] = ll_minimize_scalar_bounded(ll_norm_integral, args=(mean_col[i], std_col[i]), bounds=bounds) 
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double] apply_kelly(np.ndarray[double] mean_col, np.ndarray[double] std_col, double[:] bounds=np.array([0., 2.])):

    cdef int i, n = len(mean_col)
    assert (len(mean_col) == len(std_col) == n)
    cdef np.ndarray[double] res = np.empty(n)
    for i in range(len(mean_col)):
        res[i] = ll_minimize_scalar_bounded(ll_norm_integral, args=(mean_col[i], std_col[i]), bounds=bounds) 
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double] apply_kelly_prog(pbar, np.ndarray[double] mean_col, np.ndarray[double] std_col, double[:] bounds=np.array([0., 2.])):

    cdef int i, n = len(mean_col)
    assert (len(mean_col) == len(std_col) == n)
    cdef np.ndarray[double] res = np.empty(n)
    for i in range(len(mean_col)):
        res[i] = minimize_scalar_bounded(ll_norm_integral, args=(mean_col[i], std_col[i]), bounds=bounds) 
        pbar.update(1)
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double ll_single_kelly(double m, double s, double[:] bounds=np.array([0., 2.])):
    x = ll_minimize_scalar_bounded(ll_norm_integral, args=(m, s), bounds=bounds)
    return round(x, 4)