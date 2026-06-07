"""
This module contains functionality for extended NMA as set of separate
functions for one-rank permutations.
"""

__name__ = "springcraft"
__author__ = "Raphael Sutter"
__all__ = [
    "mean_square_fluctuation_update",
    "bfactor_update",
    "dcc_update",
]

import numpy as np
from scipy.linalg import blas

from springcraft.nma import K_B

ger = blas.get_blas_funcs("ger", dtype=np.float64)


def mean_square_fluctuation_update(
    enm,
    atom_i: int,
    atom_j: int,
    delta: float | int | bool,
    tem: int | float | None = None,
    tem_factors: int | float = K_B,
) -> np.ndarray:
    """
    Compute the *mean square fluctuation* for the atoms of a permutated model where
    the interaction strength between atoms `i` and `j` is changed by `delta`.

    Significantly faster than modifying the model and calculating from scratch.
    Calculates the update to the diagonal of the covariance matrix.

    Does not change any model attributes.

    Parameters
    ----------
    enm : ENMUpdate
        Elastic network model.
    atom_i, atom_j : int
        Atom indices with ``atom_i != atom_j``.
    delta : bool or int or float
        The change in interaction strength (``True``: reset, ``False``: set 0,
        scalar: change by value).
    tem : float or int or None, optional
        Temperature in Kelvin. If ``tem`` is ``None``, no temp scaling is conducted.
        The default is ``None``.
    tem_factors : float or int, optional
        Factors included in temperature weighting.
        The default is ``K_B``.

    Returns
    -------
    msqf : ndarray, shape=(n,), dtype=float
        The mean square fluctuations for each atom in the updated model.

    See Also
    --------
    springcraft.enm_update.ENMUpdate.prepare_update :
        More information about the update parameters.
    springcraft.nma.mean_square_fluctuation : Mean square fluctuation calculation.
    springcraft.enm_update.ENMUpdate.covariance_update :
        More information about the covariance update.

    Examples
    --------
    The following two snippets create the same result

    >>> msqf = nma_update.mean_square_fluctuation_update(enm, atom_i, atom_j, delta)

    >>> enm.modify_contact(atom_i, atom_j, delta)
    >>> msqf = nma.mean_square_fluctuation(enm)
    """
    from springcraft.enm_update import ENMUpdate

    if not isinstance(enm, ENMUpdate):
        raise ValueError("Instance of ENMUpdate class expected.")
    if not enm.has_covariance:
        raise ValueError("ENM needs to have covariance calculated.")

    msqf_update = np.diag(enm.covariance).copy()

    def msqf_update_fnc(alpha, x, y):
        nonlocal msqf_update
        msqf_update += alpha * x * y

    enm.covariance_update(
        enm._interactions,
        enm.covariance,
        *enm.prepare_update(atom_i, atom_j, delta),
        msqf_update_fnc,
    )
    msqf_update = msqf_update.reshape((-1, enm.dof)).sum(axis=1)
    # Temperature weighting
    if tem is not None:
        msqf_update *= tem * tem_factors

    return msqf_update


def bfactor_update(
    enm,
    atom_i: int,
    atom_j: int,
    delta: float | int | bool,
    tem: int | float | None = None,
    tem_factors: int | float = K_B,
) -> np.ndarray:
    """
    Compute the *mean square fluctuation* for the atoms of a permutated model where
    the interaction strength between atoms `i` and `j` is changed by `delta`.

    Parameters
    ----------
    enm : ENMUpdate
        Elastic network model.
    atom_i, atom_j : int
        Atom indices with ``atom_i != atom_j``.
    delta : bool or int or float
        The change in interaction strength (``True``: reset, ``False``: set 0,
        scalar: change by value).
    tem : float or int or None, optional
        Temperature in Kelvin. If ``tem`` is ``None``, no temp scaling is conducted.
        The default is ``None``.
    tem_factors : float or int, optional
        Factors included in temperature weighting.
        The default is ``K_B``.

    Returns
    -------
    b_factors : ndarray, shape=(n,), dtype=float
        B-factors of C-alpha atoms in the updated model.

    See Also
    --------
    springcraft.enm_update.ENMUpdate.prepare_update :
        More information about the update parameters.
    springcraft.nma.bfactor : The B-factor calculation.
    springcraft.nma_update.mean_square_fluctuation_update :
        The mean square fluctuation update.

    Examples
    --------
    The following two snippets create the same result

    >>> bfactors = nma_update.bfactor_update(enm, atom_i, atom_j, delta)

    >>> enm.modify_contact(atom_i, atom_j, delta)
    >>> bfactors = nma.bfactor(enm)
    """
    from springcraft.enm_update import ENMUpdate

    if not isinstance(enm, ENMUpdate):
        raise ValueError("Instance of ENM class expected.")

    b_factors_update = mean_square_fluctuation_update(
        enm, atom_i, atom_j, delta, tem, tem_factors
    )
    b_factors_update = ((8 * np.pi**2) * b_factors_update) / 3

    return b_factors_update


def dcc_update(
    enm,
    atom_i: int,
    atom_j: int,
    delta: float | int | bool,
    norm: bool = True,
    tem: int | float | None = None,
    tem_factors: int | float = K_B,
) -> np.ndarray:
    """
    Compute the *dynamic cross-correlation* between nodes of a permutated model
    where the interaction strength between atoms `i` and `j` is changed by `delta`.

    Significantly faster than modifying the model and calculating from scratch.
    Calculates the update to the covariance matrix.

    Parameters
    ----------
    enm : ENMUpdate
        Elastic network model.
    atom_i, atom_j : int
        Atom indices with ``atom_i != atom_j``.
    delta : bool or int or float
        The change in interaction strength (``True``: reset, ``False``: set 0,
        scalar: change by value).
    norm : bool
        Whether to normalize using the mean square fluctuations.
        The default is ``True``.
    tem : float or int or None, optional
        Temperature in Kelvin. If ``tem`` is ``None``, no temp scaling is conducted.
        The default is ``None``.
    tem_factors : float or int, optional
        Factors included in temperature weighting.
        The default is ``K_B``.

    Returns
    -------
    dcc : ndarray, shape=(n, n), dtype=float
        DCC values for the model nodes.

    See Also
    --------
    springcraft.enm_update.ENMUpdate.prepare_update :
        More information about the update parameters.
    springcraft.nma.dcc : The DCC calculation.
    springcraft.enm_update.ENMUpdate.covariance_update :
       More information about the covariance update.

    Examples
    --------
    The following two snippets create the same result

    >>> dcc = nma_update.dcc_update(enm, atom_i, atom_j, delta)

    >>> enm.modify_contact(atom_i, atom_j, delta)
    >>> dcc = nma.dcc(enm)
    """
    from springcraft.enm_update import ENMUpdate

    if not isinstance(enm, ENMUpdate):
        raise ValueError("Instance of ENMUpdate class expected.")
    if not enm.has_covariance:
        raise ValueError("ENM needs to have covariance calculated.")

    dcc_update = enm.covariance.copy()

    def dcc_update_fnc(alpha, x, y):
        nonlocal dcc_update
        ger(float(alpha), x, y, a=dcc_update.T, overwrite_a=True)

    enm.covariance_update(
        enm._interactions,
        enm.covariance,
        *enm.prepare_update(atom_i, atom_j, delta),
        dcc_update_fnc,
    )

    # calc mean over degrees of freedom
    dcc_update = (
        dcc_update.reshape(enm._natoms, enm.dof, enm._natoms, enm.dof)
        .swapaxes(1, 2)
        .trace(axis1=2, axis2=3)
    )

    # Compute the normalized DCC
    if norm:
        dcc_update_ii = np.sqrt(np.diagonal(dcc_update))
        dcc_update /= np.outer(dcc_update_ii, dcc_update_ii)

    # Temperature weighting
    if tem is not None:
        dcc_update = dcc_update * tem * tem_factors

    return dcc_update
