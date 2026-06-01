"""
This module contains functionality for extended NMA as set of separate
functions for one-rank permutations.
"""

__name__ = "springcraft"
__author__ = "Raphael Sutter"
__all__ = ["mean_square_fluctuation_pert"]

import numpy as np

from springcraft.nma import K_B


def mean_square_fluctuation_pert(
    enm,
    atom_i: int,
    atom_j: int,
    delta: float | int | bool,
    tem: int | float | None = None,
    tem_factors: int | float = K_B,
) -> np.ndarray:
    """
    Compute the change in the *mean square fluctuation* for the atoms
    according to the ENM for a rank-one update to the model.

    Parameters
    ----------
    enm : ENM
        Elastic network model.
    atom_i, atom_j : int
        Atom indices with ``atom_i != atom_j``
    delta : bool or int or float
        A bool value gets interpreted as a turn on/off signal.
        Turning on resets the contact interaction strength to the initial value.
        Turning off sets the contact interaction strength to zero.
        A scalar value changes the contact interaction strength by the given amount.
    tem : int, float, None, optional
        Temperature in Kelvin to compute the temperature scaling
        factor by multiplying with the Boltzmann constant.
        If tem is None, no temperature scaling is conducted.
    tem_factors : int, float, optional
        Factors included in temperature weighting
        (with K_B as preset).

    Returns
    -------
    msqf : ndarray, shape=(n,), dtype=float
        The mean square fluctuations for each atom in the model.

    Raises
    ------
    AttributeError
        If the `interaction` or `covariance` matrix does not exist.
    IndexError
        If any index is out of bounds or the indices are the same
    ValueError
        If the resulting `delta` is (nearly) 0.
    """
    from springcraft.enm_pert import ENMPert

    if not isinstance(enm, ENMPert):
        raise ValueError("Instance of ENMPert class expected.")
    if not enm.has_covariance:
        raise ValueError("ENM does not have covariance.")

    msqf_pert = np.zeros(enm._natoms * enm.dof)

    def msqf_update(alpha, x, y):
        nonlocal msqf_pert
        msqf_pert += alpha * x * y

    slice_i, slice_j, slice_t, delta = enm.prepare_one_rank_update(
        atom_i, atom_j, delta
    )
    enm.covariance_rank_one_update(
        enm._interactions, enm.covariance, slice_i, slice_j, slice_t, delta, msqf_update
    )
    msqf_pert = msqf_pert.reshape((-1, enm.dof)).sum(axis=1)

    # Temperature weighting
    if tem is not None:
        msqf_pert *= tem * tem_factors

    return msqf_pert


def bfactor_pert(
    enm,
    atom_i: int,
    atom_j: int,
    delta: float | int | bool,
    tem: int | float | None = None,
    tem_factors: int | float = K_B,
) -> np.ndarray:
    """
    Computes the isotropic B-factors/temperature factors/
    Deby-Waller factors for atoms/coarse-grained nodes using
    the mean-square fluctuation for a rank-one update to the model.
    These can be used to relate results obtained from ENMs
    to experimental results.

    Parameters
    ----------
    enm : ENM
        Elastic network model.
    atom_i, atom_j : int
        Atom indices with ``atom_i != atom_j``
    delta : bool or int or float
        A bool value gets interpreted as a turn on/off signal.
        Turning on resets the contact interaction strength to the initial value.
        Turning off sets the contact interaction strength to zero.
        A scalar value changes the contact interaction strength by the given amount.
    tem : int, float, None, optional
        Temperature in Kelvin to compute the temperature scaling
        factor by multiplying with the Boltzmann constant.
        If tem is None, no temperature scaling is conducted.
    tem_factors : int, float, optional
        Factors included in temperature weighting
        (with K_B as preset).

    Returns
    -------
    bfac_values : ndarray, shape=(n,), dtype=float
        B-factors of C-alpha atoms.

    Raises
    ------
    AttributeError
        If the `interaction` or `covariance` matrix does not exist.
    IndexError
        If any index is out of bounds or the indices are the same
    ValueError
        If the resulting `delta` is (nearly) 0.
    """
    from springcraft.enm import ENM

    if not isinstance(enm, ENM):
        raise ValueError("Instance of ENM class expected.")

    b_factors_pert = mean_square_fluctuation_pert(
        enm, atom_i, atom_j, delta, tem, tem_factors
    )
    b_factors_pert = ((8 * np.pi**2) * b_factors_pert) / 3

    return b_factors_pert
