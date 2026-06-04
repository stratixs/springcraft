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

    This essentially means putting the eigenvalues d and z in reverse
    order and additionally negating the eigenvalues. For the final result
    the transformation must be put in reverse.

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
    res : ndarray, shape (n,), dtype=float64
        The eigenvalues of the permutation.

    Notes
    -----
    This is a thin wrapper around the Fortran routine ``DLAED4`` accessed
    via ``scipy.linalg.cython_lapack``.  Input arrays must be
    C-contiguous, double-precision, and sorted in strictly ascending order.
    The routine performs no argument checking internally.
    """
    cdef int n = d.shape[0]
    cdef double[::1] d_val = np.empty(n, dtype=np.float64)
    cdef double[::1] z_val = np.empty(n, dtype=np.float64)
    cdef double[::1] res = np.empty(n, dtype=np.float64)  # updated eigenvalues
    cdef double[::1] delta = np.empty(n, dtype=np.float64)  # buffer

    # norm z and adjust rho
    cdef double z_dot = 0
    cdef int i
    for i in range(n):
        z_dot += z[i] * z[i]
    cdef double rho_val = rho * z_dot
    cdef double z_norm = sqrt(z_dot)
    for i in range(n):
        z_val[i] = z[i] / z_norm

    cdef int info
    if rho > 0:
        for i in range(n):
            d_val[i] = d[i]

        for i in range(1 + n_triv, n + 1):  # dlaed4 requires 1-based index
            cython_lapack.dlaed4(&n, &i, &d_val[0], &z_val[0], &delta[0], &rho_val, &res[i-1], &info)
            if info != 0:
                raise RuntimeError("LAPACK dlaed4 failed.")
    elif rho < 0:
        for i in range(n):
            d_val[i] = -d[n - 1 - i]
        reverse(&z_val[0], &z_val[0] + n)

        rho_val = -rho_val
        for i in range(1, n + 1 - n_triv):  # dlaed4 requires 1-based index
            cython_lapack.dlaed4(&n, &i, &d_val[0], &z_val[0], &delta[0], &rho_val, &res[n-i], &info)
            if info != 0:
                raise RuntimeError("LAPACK dlaed4 failed.")

        for i in range(n):
            res[i] = -res[i]
    else:
        raise ValueError("Rho must not be 0.")


    for i in range(n_triv):
        res[i] = 0
    return np.asarray(res)

def eigen_update(double[::1] d, double[::1] z, double rho, int[::1] subset):
    """
    Computes the eigenvalues of a symmetric rank-one modified diagonal
    matrix using the LAPACK ``dlaed4`` routine for a subset of eigenvalues.
    Additionally returns values to construct the eigenvectors of the
    permutated system.

    Solves for the updated eigenvalues of the system:

        diag(d) + rho * z * z^T = w * w^T

    where ``d(j) < d(j+1)`` for all ``j``.

    DLAED4 requires for z to be normed and for rho to be positive.
    If rho is negative we solve for

        -(-diag(d) + (-rho) * z * z^T)

    This essentially means putting the eigenvalues d and z in reverse
    order and additionally negating the eigenvalues. For the final result
    the transformation must be put in reverse.

    The vectors `w` to the corresponding eigenvalue can be obtained by

        norm(z / delta[i])

    Parameters
    ----------
    d : ndarray, shape (n,), dtype=float64
        Original eigenvalues in strictly ascending order.
    z : ndarray, shape (n,), dtype=float64
        Components of the rank-one updating vector. Assumed to have unit
        Euclidean norm.
    rho : float
        Scalar in the symmetric rank-one update. Must not be 0
    mode_subset : ndarray, shape=(k,), dtype=int, optional
        Specifies the subset of eigenvalues/-vectors to calculate

    Returns
    -------
    res : ndarray, shape (k,), dtype=float64
        The eigenvalues of the permutation.
    delta : ndarray, shape (k,n), dtype=float64
        Vectors to calculate perturbated eigenvalues.

    Notes
    -----
    This is a thin wrapper around the Fortran routine ``DLAED4`` accessed
    via ``scipy.linalg.cython_lapack``.  Input arrays must be
    C-contiguous, double-precision, and sorted in strictly ascending order.
    The routine performs no argument checking internally.
    """
    cdef int n = d.shape[0]
    cdef int m = subset.shape[0]

    cdef double[::1] d_val = d.copy()
    cdef double[::1] z_val = np.empty(n, dtype=np.float64)
    cdef double[::1] res = np.empty(m, dtype=np.float64)  # updated eigenvalues
    cdef double[::1] delta = np.empty(m * n, dtype=np.float64)  # eigenvector changes

    # norm z and adjust rho
    cdef double z_dot = 0
    cdef int i
    for i in range(n):
        z_dot += z[i] * z[i]
    cdef double rho_val = rho * z_dot
    cdef double z_norm = sqrt(z_dot)
    for i in range(n):
        z_val[i] = z[i] / z_norm

    cdef int info
    cdef int idx
    if rho > 0:
        for i in range(n):
            d_val[i] = d[i]

        for i in range(m):  # dlaed4 requires 1-based index
            idx = subset[i] + 1
            cython_lapack.dlaed4(&n, &idx, &d_val[0], &z_val[0], &delta[i*n], &rho_val, &res[i], &info)
            if info != 0:
                raise RuntimeError("LAPACK dlaed4 failed.")
    elif rho < 0:
        for i in range(n):
            d_val[i] = -d[n - 1 - i]
        reverse(&z_val[0], &z_val[0] + n)

        rho_val = -rho_val
        for i in range(m):  # dlaed4 requires 1-based index
            idx = n - subset[i]
            cython_lapack.dlaed4(&n, &idx, &d_val[0], &z_val[0], &delta[(m-1-i)*n], &rho_val, &res[i], &info)
            if info != 0:
                raise RuntimeError("LAPACK dlaed4 failed.")

        for i in range(m):
            res[i] = -res[i]
        reverse(&delta[0], &delta[0] + m * n)
    else:
        raise ValueError("Rho must not be 0.")

    return np.asarray(res), np.asarray(delta).reshape((m, n))
