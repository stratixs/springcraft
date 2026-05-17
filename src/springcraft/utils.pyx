"""
Utility functions providing low-level LAPACK wrappers.
"""

import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_lapack as cython_lapack


def eigenvalue_update(int i, double[::1] d, double[::1] z, double rho):
    """
    Compute the i-th eigenvalue of a symmetric rank-one modified diagonal
    matrix using the LAPACK ``dlaed4`` routine.

    Solves for the i-th updated eigenvalue ``lambda_i`` of the system::

        diag(d) + rho * z * z^T

    where ``d(j) < d(j+1)`` for all j and ``rho > 0``.  The norm of ``z``
    is assumed to be 1.

    Parameters
    ----------
    i : int
        0-based index of the eigenvalue to compute (``0 <= i < n``).
    d : ndarray, shape (n,), dtype=float64
        Original eigenvalues in strictly ascending order.
    z : ndarray, shape (n,), dtype=float64
        Components of the rank-one updating vector.  Assumed to have unit
        Euclidean norm.
    rho : float
        Positive scalar in the symmetric rank-one update.

    Returns
    -------
    dlam : float
        The computed i-th updated eigenvalue ``lambda_i``.

    Raises
    ------
    ValueError
        If `d` and `z` do not have the same size or `rho` is not positive.
    IndexError
        If `i` is out of the valid range ``[0, n-1]``.

    Notes
    -----
    This is a thin wrapper around the Fortran routine ``DLAED4`` accessed
    via ``scipy.linalg.cython_lapack``.  Input arrays must be
    C-contiguous, double-precision, and sorted in strictly ascending order.
    The routine performs no argument checking internally.
    """
    cdef int n = d.shape[0]

    if n == 0:
        raise ValueError("The input diagonal array 'd' cannot be empty.")
    if z.shape[0] != n:
        raise ValueError("Arrays 'd' and 'z' must have identical lengths.")
    if i < 0 or i >= n:
        raise ValueError(f"The root index 'i' must be 0-based and in the range [0, {n-1}].")
    if rho <= 0.0:
        raise ValueError("The scalar parameter 'rho' must be strictly positive.")

    cdef double[::1] delta = np.zeros(n, dtype=np.float64)
    cdef double dlam = 0.0
    cdef int info = 0

    cdef int n_val = n
    cdef int i_val = i + 1
    cdef double rho_val = rho

    cython_lapack.dlaed4(&n_val, &i_val, &d[0], &z[0], &delta[0], &rho_val, &dlam, &info)
    if info != 0:
        raise RuntimeError("LAPACK dlaed4 failed.")

    return dlam
