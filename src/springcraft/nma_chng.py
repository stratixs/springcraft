"""
This module contains functionality for extended NMA as set of separate
functions for one-rank permutations.
"""

__name__ = "springcraft"
__author__ = "Raphael Sutter"
__all__ = ["frequencies_chng", "mean_square_fluctuation_chng", "bfactor_chng"]

import numpy as np
from scipy.linalg import blas

from springcraft.nma import K_B
from springcraft.utils import eigen_chng, eigenvalue_chng

ger = blas.get_blas_funcs("ger", dtype=np.float64)


def frequencies_chng(
    enm,
    atom_i: int,
    atom_j: int,
    delta: float | int | bool,
) -> np.ndarray:
    """
    Computes the frequency associated with each mode for the permutated
    ENM where the interaction strength between atoms `i` and `j` is
    changed by `delta`.

    The modes corresponding to rigid-body translations/rotations are
    omitted in the return value.
    The returned units are arbitrary and should only be compared
    relative to each other.

    Parameters
    ----------
    enm : ENM
        Elastic network model.
    atom_i, atom_j : int
        Atom index with ``atom_i != atom_j``
    delta : bool or int or float
        A bool value gets interpreted as a turn on/off signal.
        Turning on resets the contact interaction strength to the initial value.
        Turning off sets the contact interaction strength to zero.
        A scalar value changes the contact interaction strength by the given amount.

    Returns
    -------
    freq : ndarray, shape=(n,), dtype=float
        The frequency in ascending order of the associated modes'
        Eigenvalues.

    Raises
    ------
    AttributeError
        If the ENM's eigenvalues and -vectors do not exist.
    """
    from springcraft.enm_pert import ENMPert

    if not isinstance(enm, ENMPert):
        raise ValueError("Instance of ENMPert class expected.")
    if not enm.has_eigen:
        raise AttributeError("The ENM's eigenvalues must be exist.")

    eig_val, eig_vec, eig_n_triv = enm.eigen(n_zero=True, copy=False)
    eig_vec = eig_vec.T

    slice_i, slice_j, slice_t, delta = enm.prepare_one_rank_update(
        atom_i, atom_j, delta
    )
    z = slice_t @ eig_vec[slice_i] - slice_t @ eig_vec[slice_j]

    # check whether rank increases
    t = slice_t @ (eig_vec[slice_i, :eig_n_triv] - eig_vec[slice_j, :eig_n_triv])
    if np.any(np.abs(t) > 1e-6):
        eig_n_triv -= 1

    eig_val_chng = eigenvalue_chng(eig_val, eig_n_triv, z, np.asarray(delta).item())

    # rank decrease protection (near zero but negative)
    eig_val_chng[eig_n_triv] = np.abs(eig_val_chng[eig_n_triv])

    return 1 / (2 * np.pi) * np.sqrt(eig_val_chng)


def mean_square_fluctuation_chng(
    enm,
    atom_i: int,
    atom_j: int,
    delta: float | int | bool,
    mode_subset: np.ndarray | None = None,
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
    mode_subset : ndarray, shape=(n,) or (3n,), dtype=int, optional
        Specifies the subset of modes considered in the MSF computation.
        The first mode is counted as 0 in accordance with Python conventions.
        If mode_subset is None, all modes are included.
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
        If any atom index is out of bounds or the indices are the same
    ValueError
        If the resulting `delta` is (nearly) 0.
    """
    from springcraft.enm_pert import ENMPert

    if not isinstance(enm, ENMPert):
        raise ValueError("Instance of ENMPert class expected.")

    if enm.has_covariance and mode_subset is None:
        msqf_chng = np.diag(enm.covariance).copy()

        def msqf_update_fnc(alpha, x, y):
            nonlocal msqf_chng
            msqf_chng += alpha * x * y

        enm.covariance_rank_one_update(
            enm._interactions,
            enm.covariance,
            *enm.prepare_one_rank_update(atom_i, atom_j, delta),
            msqf_update_fnc,
        )
        msqf_chng = msqf_chng.reshape((-1, enm.dof)).sum(axis=1)
    else:
        eig_values_pert, eig_vectors_pert = _calc_updated_eigen(
            enm, atom_i, atom_j, delta, mode_subset
        )

        msqf_chng = (eig_vectors_pert.T**2) @ (1 / eig_values_pert)
        msqf_chng = msqf_chng.reshape(-1, enm.dof).sum(axis=1)

    # Temperature weighting
    if tem is not None:
        msqf_chng *= tem * tem_factors

    return msqf_chng


def bfactor_chng(
    enm,
    atom_i: int,
    atom_j: int,
    delta: float | int | bool,
    mode_subset: np.ndarray | None = None,
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
    mode_subset : ndarray, shape=(n,) or (3n,), dtype=int, optional
        Specifies the subset of modes considered in the MSF computation.
        The first mode is counted as 0 in accordance with Python conventions.
        If mode_subset is None, all modes are included.
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
    from springcraft.enm_pert import ENMPert

    if not isinstance(enm, ENMPert):
        raise ValueError("Instance of ENM class expected.")

    b_factors_chng = mean_square_fluctuation_chng(
        enm, atom_i, atom_j, delta, mode_subset, tem, tem_factors
    )
    b_factors_chng = ((8 * np.pi**2) * b_factors_chng) / 3

    return b_factors_chng


def dcc_chng(
    enm,
    atom_i: int,
    atom_j: int,
    delta: float | int | bool,
    mode_subset: np.ndarray | None = None,
    norm: bool = True,
    tem: int | float | None = None,
    tem_factors: int | float = K_B,
) -> np.ndarray:
    r"""
    Computes the normalized *dynamic cross-correlation* between
    nodes of the ENM for a rank-one updated model.

    The method does not change any attributes of the model class.

    Parameters
    ----------
    enm : ENM
        Elastic network model; an instance of either an GNM or ANM
        object.
    atom_i, atom_j : int
        Atom indices with ``atom_i != atom_j``
    delta : bool or int or float
        A bool value gets interpreted as a turn on/off signal.
        Turning on resets the contact interaction strength to the initial value.
        Turning off sets the contact interaction strength to zero.
        A scalar value changes the contact interaction strength by the given amount.
    mode_subset : ndarray, shape=(n,) or (3n,), dtype=int, optional
        Specifies the subset of modes considered in the MSF computation.
        The first mode is counted as 0 in accordance with Python conventions.
        If mode_subset is None, all modes are included.
    norm : bool, optional
        Normalize the DCC using the MSFs of interacting nodes.
    tem : int, float, None, optional
        Temperature in Kelvin to compute the temperature scaling
        factor by multiplying with the Boltzmann constant.
        If tem is None, no temperature scaling is conducted.
    tem_factors : int, float, optional
        Factors included in temperature weighting
        (with :math:`k_B` as preset).

    Returns
    -------
    dcc : ndarray, shape=(n, n), dtype=float
        DCC values for updated ENM nodes as NxN matrix.

    Notes
    -----

    The DCC for a nodepair :math:`ij` is computed as:

    .. math::

        DCC_{ij} = \frac{3 k_B T}{\gamma} \sum_k^L \left[ \frac{\vec{u}_k \cdot \vec{u}_k^T}{\lambda_k} \right]_{ij}

    with :math:`\lambda` and :math:`\vec{u}` as
    Eigenvalues and Eigenvectors corresponding to mode :math:`k` of
    the modeset :math:`L`.

    DCCs can be normalized to MSFs exhibited by two compared nodes
    following:

    .. math::

        nDCC_{ij} = \frac{DCC_{ij}}{[DCC_{ii} DCC_{jj}]^{1/2}}

    When all modes are considerered, the DCC is equal to the covariance matrix
    of GNMs or to the trace of all supermatrices (3x3) of the
    covariance matrix (3Nx3N) in the case of ANMs.
    Consequently, these are returned if standard parameters
    for 'mode_subset' and 'memory_efficient' are passed to the function.
    """
    from springcraft.enm import ENM

    if not isinstance(enm, ENM):
        raise ValueError("Instance of ENM class expected.")

    if mode_subset is None:
        dcc_update = enm.covariance.copy()

        def dcc_update_fnc(alpha, x, y):
            nonlocal dcc_update
            ger(alpha, x, y, a=dcc_update.T, overwrite_a=True)

        enm.covariance_rank_one_update(
            enm._interactions,
            enm.covariance,
            *enm.prepare_one_rank_update(atom_i, atom_j, delta),
            dcc_update_fnc,
        )

        # calc mean over degrees of freedom
        dcc_update = (
            dcc_update.reshape(enm._natoms, enm.dof, enm._natoms, enm.dof)
            .swapaxes(1, 2)
            .trace(axis1=2, axis2=3)
        )

    else:
        eig_val_update, eig_vec_update = _calc_updated_eigen(
            enm, atom_i, atom_j, delta, mode_subset
        )

        eig_vec_update = np.reshape(eig_vec_update, (len(mode_subset), -1, enm.dof))
        eig_vec_update_scal = eig_vec_update / eig_val_update[:, None, None]
        dcc_update = np.einsum("knd,kmd->nm", eig_vec_update, eig_vec_update_scal)

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
    Calculates the updated eigenvalues and vectors for a subset if modes.

    The ENM attributes do not get changed.

    Parameters
    ----------
    enm : ENM
        Elastic network model
    atom_i, atom_j : int
        Atom indices with ``atom_i != atom_j``
    delta : bool or int or float
        A bool value gets interpreted as a turn on/off signal.
        Turning on resets the contact interaction strength to the initial value.
        Turning off sets the contact interaction strength to zero.
        A scalar value changes the contact interaction strength by the given amount.
    mode_subset : ndarray, shape=(k,), dtype=int, optional
        Specifies the subset of modes considered in the update.
        The first mode is counted as 0 in accordance with Python conventions.

    Returns
    -------
    eigen_values : ndarray, shape=(n, n), dtype=float
        The updated subset of eigenvalues
    eigen_values : ndarray, shape=(n, n), dtype=float
        The updated subset of corresponding eigenvectors.
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

    slice_i, slice_j, slice_t, delta = enm.prepare_one_rank_update(
        atom_i, atom_j, delta
    )
    z = slice_t @ eig_vectors[slice_i] - slice_t @ eig_vectors[slice_j]

    rho = np.asarray(delta).item()
    mode_subset = mode_subset.astype(np.intc)
    eig_values_pert, eig_vectors_delta = eigen_chng(eig_values, z, rho, mode_subset)

    w = z / eig_vectors_delta
    w = w / np.linalg.norm(w, axis=1).reshape(-1, 1)
    eig_vectors_pert = w @ eig_vectors.T

    return eig_values_pert, eig_vectors_pert
