"""
This module contains functionality for extended NMA as set of separate
functions for one-rank permutations.
"""

__name__ = "springcraft"
__author__ = "Raphael Sutter"
__all__ = [
    "frequencies_update",
    "mean_square_fluctuation_update",
    "bfactor_update",
    "dcc_update",
]

import numpy as np
from scipy.linalg import blas

from springcraft.nma import K_B
from springcraft.utils import eigen_update, eigenvalue_update

ger = blas.get_blas_funcs("ger", dtype=np.float64)


def frequencies_update(
    enm,
    atom_i: int,
    atom_j: int,
    delta: float | int | bool,
) -> np.ndarray:
    """
    Compute the oscillation frequencies of a permutated model where the interaction
    strength between atoms `i` and `j` is changed by `delta`.

    Significantly faster than modifying the model and calculating from scratch.
    Calculates the eigenvalue of the perturbated system based on the known eigenvalues
    of the existing system using an algorithm from Gu and Eisenstat.

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

    Returns
    -------
    freq : ndarray, shape=(n,), dtype=float
        Oscillation frequencies of the updated model.

    See Also
    --------
    springcraft.enm_update.ENMUpdate.prepare_update :
        More information about the update parameters.
    springcraft.nma.frequencies : The frequency calculation.
    _calc_updated_eigen : More information about the eigenvalue update.

    Examples
    --------
    The following two snippets create the same result

    >>> freq = nma_update.frequencies_update(enm, atom_i, atom_j, delta)

    >>> enm.modify_contact(atom_i, atom_j, delta)
    >>> freq = nma.frequencies(enm)
    """
    from springcraft.enm_update import ENMUpdate

    if not isinstance(enm, ENMUpdate):
        raise ValueError("Instance of ENMUpdate class expected.")
    if not enm.has_eigen:
        raise AttributeError("The ENM's eigenvalues must exist.")

    eig_values, eig_vectors, eig_n_triv = enm.eigen(n_zero=True, copy=False)
    eig_vectors = eig_vectors.T

    slice_i, slice_j, slice_t, delta = enm.prepare_update(atom_i, atom_j, delta)
    z = slice_t @ eig_vectors[slice_i] - slice_t @ eig_vectors[slice_j]

    # check whether rank increases
    t = slice_t @ (
        eig_vectors[slice_i, :eig_n_triv] - eig_vectors[slice_j, :eig_n_triv]
    )
    if np.any(np.abs(t) > 1e-6):
        eig_n_triv -= 1

    eig_values_update = eigenvalue_update(
        eig_values, eig_n_triv, z, np.asarray(delta).item()
    )

    # rank decrease protection (near zero but negative)
    eig_values_update[eig_n_triv] = np.abs(eig_values_update[eig_n_triv])

    return 1 / (2 * np.pi) * np.sqrt(eig_values_update)


def mean_square_fluctuation_update(
    enm,
    atom_i: int,
    atom_j: int,
    delta: float | int | bool,
    mode_subset: np.ndarray | None = None,
    tem: int | float | None = None,
    tem_factors: int | float = K_B,
) -> np.ndarray:
    """
    Compute the *mean square fluctuation* for the atoms of a permutated model where
    the interaction strength between atoms `i` and `j` is changed by `delta`.

    Significantly faster than modifying the model and calculating from scratch.
    Either calculates the update to the diagonal of the covariance matrix or the
    eigenvalues of the perturbated system based on the known eigenvalues of the
    existing system using an algorithm from Gu and Eisenstat.

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
    mode_subset : ndarray, shape=(k,), dtype=int, optional
        Specifies the subset of modes considered in the computation.
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
    _calc_updated_eigen : More information about the eigenvalue update.

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

    if enm.has_covariance and mode_subset is None:
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
    else:
        eig_values_update, eig_vectors_update = _calc_updated_eigen(
            enm, atom_i, atom_j, delta, mode_subset
        )

        msqf_update = (eig_vectors_update.T**2) @ (1 / eig_values_update)
        msqf_update = msqf_update.reshape(-1, enm.dof).sum(axis=1)

    # Temperature weighting
    if tem is not None:
        msqf_update *= tem * tem_factors

    return msqf_update


def bfactor_update(
    enm,
    atom_i: int,
    atom_j: int,
    delta: float | int | bool,
    mode_subset: np.ndarray | None = None,
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
    mode_subset : ndarray, shape=(k,), dtype=int, optional
        Specifies the subset of modes considered in the computation.
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
        enm, atom_i, atom_j, delta, mode_subset, tem, tem_factors
    )
    b_factors_update = ((8 * np.pi**2) * b_factors_update) / 3

    return b_factors_update


def dcc_update(
    enm,
    atom_i: int,
    atom_j: int,
    delta: float | int | bool,
    mode_subset: np.ndarray | None = None,
    norm: bool = True,
    tem: int | float | None = None,
    tem_factors: int | float = K_B,
) -> np.ndarray:
    """
    Compute the *dynamic cross-correlation* between nodes of a permutated model
    where the interaction strength between atoms `i` and `j` is changed by `delta`.

    Significantly faster than modifying the model and calculating from scratch.
    Either calculates the update to the diagonal of the covariance matrix or the
    eigenvalues of the perturbated system based on the known eigenvalues of the
    existing system using an algorithm from Gu and Eisenstat.

    Parameters
    ----------
    enm : ENMUpdate
        Elastic network model.
    atom_i, atom_j : int
        Atom indices with ``atom_i != atom_j``.
    delta : bool or int or float
        The change in interaction strength (``True``: reset, ``False``: set 0,
        scalar: change by value).
    mode_subset : ndarray, shape=(k,), dtype=int or None, optional
        Specifies the subset of modes considered in the computation.
        The default is ``None``.
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
    _calc_updated_eigen : More information about the eigenvalue update.

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

    if mode_subset is None:
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

    else:
        eig_values_update, eig_vectors_update = _calc_updated_eigen(
            enm, atom_i, atom_j, delta, mode_subset
        )

        eig_vectors_update = np.reshape(
            eig_vectors_update, (len(mode_subset), -1, enm.dof)
        )
        eig_vectors_update_scal = eig_vectors_update / eig_values_update[:, None, None]
        dcc_update = np.einsum(
            "knd,kmd->nm", eig_vectors_update, eig_vectors_update_scal
        )

    # Compute the normalized DCC
    if norm:
        dcc_update_ii = np.sqrt(np.diagonal(dcc_update))
        dcc_update /= np.outer(dcc_update_ii, dcc_update_ii)

    # Temperature weighting
    if tem is not None:
        dcc_update = dcc_update * tem * tem_factors

    return dcc_update


def _calc_updated_eigen(
    enm,
    atom_i: int,
    atom_j,
    delta: float | int | bool,
    mode_subset: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the updated eigenvalues and -vectors for a subset of modes for a
    network model where the interaction strength between atoms `i` and `j` gets
    changed by `delta`. Uses the algorithm of Gu and Eisenstat which is implemented
    by LAPACK's ``dlaed4`` routine.

    The ENM attributes do not get changed.

    Parameters
    ----------
    enm : ENMUpdate
        Elastic network model.
    atom_i, atom_j : int
        Atom indices with ``atom_i != atom_j``.
    delta : bool or int or float
        The change in interaction strength (``True``: reset, ``False``: set 0,
        scalar: change by value).
    mode_subset : ndarray, shape=(k,), dtype=int, optional
        Specifies the subset of modes considered in the computation.

    Returns
    -------
    eig_values : ndarray, shape=(k,), dtype=float
        The updated subset of eigenvalues.
    eig_vectors : ndarray, shape=(k, n), dtype=float
        The updated subset of corresponding eigenvectors.

    Raises
    ------
    ValueError
        If any trivial (zero) eigenvalues are selected.

    See Also
    --------
    springcraft.enm_update.ENMUpdate.prepare_update :
        More information about the update parameters.

    Notes
    -----
    Changing the force constant between atoms `i` and `j` by an arbitrary amount
    :math:`\\delta` can be described by a rank-one update to the interaction matrix
    :math:`\\Gamma` with a vector :math:`\\vec{u}` of matching dimensions like

    .. math:: \\tilde{\\Gamma} = \\Gamma + \\delta \\vec{u} \\vec{u}^T

    Let :math:`\\Lambda` be the diagonal matrix of eigenvalues :math:`\\lambda` of
    :math:`\\Gamma` and :math:`V` be a matrix of the corresponding eigenvectors. Than
    the same update can be described as

    .. math:: \\tilde{\\Gamma} = V \\Lambda V^T + \\delta \\vec{c} \\vec{c}^T
                               = V (\\Lambda + \\rho \\vec{z} \\vec{z}^T) V^T

    with :math:`z = V^T c`. According to Gu and Eisenstathe eigenvalues
    :math:`\\tilde{\\lambda}` of the rank-one updated system
    :math:`\\Lambda + \\rho \\vec{z} \\vec{z}^T` are the roots of the secular equation

    .. math:: f(\\tilde{\\lambda})
              = 1 + \\sum_{j=1}^n \\frac{z_j^2}{\\lambda_j - \\tilde{\\lambda}} = 0

    One can easily convince itself that these updated eigenvalues
    :math:`\\tilde{\\lambda}` are the same for the perturbated system
    :math:`\\tilde{\\Gamma}`.

    This algorithm is implemented by LAPACK and is called dlaed4.

    Finally the updated eigenvectors :math:`w_i` of
    :math:`\\Lambda + \\rho \\vec{z} \\vec{z}^T = W \\tilde{\\Lambda} W^T` can be
    calculated using the delta :math:`\\epsilon_i` returned by DLAED4 for every
    eigenvalue :math:`\\tilde{\\lambda}` by elementwise divison and norming the
    resulting vector.

    The DLAED4 routine requires :math:`\\rho` to be positive and the the supplied
    eigenvalues to be in strictly ascending order. If the original :math:`\\delta` is
    negative we solve the equivalent system

    .. math:: \\Lambda + \\delta \\vec{z} \\vec{z}^T
              = -(-\\Lambda - (-\\delta) \\vec{z} \\vec{z}^T)

    where the elements of :math:`\\Lambda` and :math:`\\vec{z}` are in reversed order.

    References
    ----------
    .. [1] Ming Gu and Stanley C. Eisenstat, "A Stable and Efficient Algorithm for the
       Rank-One Modification of the Symmetric Eigenproblem", SIAM Journal on Matrix
       Analysis and Applications, vol. 15, p. 1266-1276, 1994, 10.1137/S089547989223924X
    """
    eig_values, eig_vectors, n_triv = enm.eigen(n_zero=True)
    eig_vectors = eig_vectors.T

    # Choose modes included in computation; raise error, if trivial
    # modes are included
    if mode_subset is None:
        mode_subset = np.arange(n_triv, len(eig_values))
    elif np.any(mode_subset < n_triv):
        raise ValueError(
            "Trivial modes are included in the current selection. "
            "Please check your input."
        )

    slice_i, slice_j, slice_t, delta = enm.prepare_update(atom_i, atom_j, delta)
    z = slice_t @ eig_vectors[slice_i] - slice_t @ eig_vectors[slice_j]

    rho = np.asarray(delta).item()
    mode_subset = mode_subset.astype(np.intc)
    eig_values_update, eig_vectors_delta = eigen_update(eig_values, z, rho, mode_subset)

    w = z / eig_vectors_delta
    w = w / np.linalg.norm(w, axis=1).reshape(-1, 1)
    eig_vectors_update = w @ eig_vectors.T

    return eig_values_update, eig_vectors_update
