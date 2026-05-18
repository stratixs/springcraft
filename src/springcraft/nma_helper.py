"""
This module contains the actual extended NMA logic.
It is decoupled from the `nma.py` module so that the functionality can
also be used by the ENM permutation logic.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann, Jan Krumbach, Faisal Islam, Raphael Sutter"

import numpy as np


def frequencies_helper(eigen_values: np.ndarray) -> np.ndarray:
    """
    Computes the frequency associated with each mode.

    The modes corresponding to rigid-body translations/rotations are
    omitted in the return value.
    The returned units are arbitrary and should only be compared
    relative to each other.

    Creates a new array. The input array is not changed.

    Parameters
    ----------
    eigen_values : ndarray, shape=(k,), dtype=float
        The eigenvalues of the ENM

    Returns
    -------
    freq : ndarray, shape=(k,), dtype=float
        The frequency of the associated eigenvalue.
        Has the same order as the input argument.
    """
    return 1 / (2 * np.pi) * np.sqrt(np.abs(eigen_values))
