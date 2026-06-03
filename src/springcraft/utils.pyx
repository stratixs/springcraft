"""
Utility functions providing low-level LAPACK wrappers.
"""
import numpy as np
cimport numpy as cnpy
cimport scipy.linalg.cython_lapack as cython_lapack

from libc.math cimport sqrt

cdef extern from "<algorithm>" namespace "std":
    void reverse[Iter](Iter first, Iter last)

def eigenvalue_update(double[::1] d, int n_triv, double[::1] z, double rho):
    """
    Computes the eigenvalues of a symmetric rank-one modified diagonal
    matrix using the LAPACK ``dlaed4`` routine.

    Solves for the updated eigenvalues of the system:

        diag(d) + rho * z * z^T

    where ``d(j) < d(j+1)`` for all ``j``.

    DLAED4 requires for z to be normed and for rho to be positive.
    If rho is negative we solve for

        -(-diag(d) + (-rho) * z * z^T)

    Parameters
    ----------
    d : ndarray, shape (n,), dtype=float64
        Original eigenvalues in strictly ascending order.
    n_triv : int
        The number of 0 eigenvalues.
    z : ndarray, shape (n,), dtype=float64
        Components of the rank-one updating vector. Assumed to have unit
        Euclidean norm.
    rho : float
        Scalar in the symmetric rank-one update. Must not be 0

    Returns
    -------
    dlam : float
        The eigenvalues of the permutation.

    Raises
    ------
    ValueError
        If `d` and `z` do not have the same size or `rho` is not positive.

    Notes
    -----
    This is a thin wrapper around the Fortran routine ``DLAED4`` accessed
    via ``scipy.linalg.cython_lapack``.  Input arrays must be
    C-contiguous, double-precision, and sorted in strictly ascending order.
    The routine performs no argument checking internally.
    """
    cdef int n = d.shape[0]
    cdef double[::1] d_val = d.copy()
    cdef double[::1] z_val = np.empty(n, dtype=np.float64)
    cdef double[::1] delta = np.empty(n, dtype=np.float64)
    cdef double[::1] res = np.empty(n, dtype=np.float64)
    cdef int i
    for i in range(n_triv + 1):
        res[i] = 0

    # norm z and adjust rho
    cdef double z_dot = 0
    for i in range(n):
        z_dot += z[i] * z[i]
    cdef double rho_val = rho * z_dot
    cdef double z_norm = sqrt(z_dot)
    for i in range(n):
        z_val[i] = z[i] / z_norm

    cdef int info
    if rho > 0:
        for i in range(1 + n_triv, n + 1):  # dlaed4 requires 1-based index
            cython_lapack.dlaed4(&n, &i, &d_val[0], &z_val[0], &delta[0], &rho_val, &res[i-1], &info)
            if info != 0:
                raise RuntimeError("LAPACK dlaed4 failed.")
    elif rho < 0:
        reverse(&d_val[0], &d_val[0] + n)
        reverse(&z_val[0], &z_val[0] + n)
        for i in range(n):
            d_val[i] = -d_val[i]

        rho_val = -rho_val
        for i in range(1, n + 1 - n_triv):  # dlaed4 requires 1-based index
            cython_lapack.dlaed4(&n, &i, &d_val[0], &z_val[0], &delta[0], &rho_val, &res[n-i], &info)
            if info != 0:
                raise RuntimeError("LAPACK dlaed4 failed.")

        for i in range(n):
            res[i] = -res[i]
    else:
        raise ValueError("Rho must not be 0.")


    return np.asarray(res)
