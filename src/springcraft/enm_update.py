"""
This module contains the :class:`ENM` class. An abstract base class for molecular
dynamics calculations Models. Extends the ENM base class with low rank perturbation
calculation.
"""

__name__ = "springcraft"
__author__ = "Raphael Sutter"
__all__ = ["ENMUpdate", "covariance_update"]

from abc import abstractmethod
from typing import Callable

import biotite.structure as struc
import numpy as np
from scipy.linalg import blas

from springcraft import nma_update
from springcraft.enm import ENM, K_B

ger = blas.get_blas_funcs("ger", dtype=np.float64)


class ENMUpdate(ENM):
    def modify_contact(self, atom_i: int, atom_j: int, delta: bool | int | float):
        """
        Changes the interaction strength between atoms `i` and `j` of this model by
        `delta`. Results in a change to the `interaction` matrix.

        If the `covariance` matrix exists, a low complexity algorithm is used to update
        the covariance matrix according to the small perturbation introduced to the
        `interaction` matrix.

        Parameters
        ----------
        atom_i, atom_j : int
            Atom indices with ``atom_i != atom_j``.
        delta : bool or int or float
            The change in interaction strength (``True``: reset, ``False``: set 0,
            scalar: change by value).

        Raises
        ------
        AttributeError
            If the `interaction` matrix does not exist.
        IndexError
            If any index is out of bounds or the indices are the same.
        ValueError
            If the resulting `delta` is (nearly) 0.

        See Also
        --------
        springcraft.enm_update.ENMUpdate.prepare_update :
            More Information about the update parameters.
        springcraft.enm_update.covariance_update :
            More information about the covariance update.
        """
        slice_i, slice_j, slice_t, delta = self.prepare_update(atom_i, atom_j, delta)

        if self._covariance is not None:
            self._modify_covariance(slice_i, slice_j, slice_t, delta)

        self._modify_interactions(slice_i, slice_j, slice_t, delta)

        # invalidate dependent values
        self._eig_values = None
        self._eig_vectors = None

    @abstractmethod
    def modify_atom(self, atom_i: int, new_atom: bool | struc.Atom):
        """
        Changes the interaction strength between atom `i` and every other connected atom
        of this model. Results in a change to the `interaction` matrix.

        If the `covariance` matrix exists, a low complexity algorithm is used to update
        the covariance matrix according to the small perturbation introduced to the
        `interaction` matrix.

        This is essentially a series applications of `modify_contact`.

        Parameters
        ----------
        atom_i : int
            The index of the atom to change.
        new_atom : bool or Atom
            A bool gets interpreted as a turn on/off signal.
            Turning on resets every contact interaction strength of this atom to the
            initial value. Turning off sets every contact interaction strength to zero.
            An Atom may result in a change to the `ForceField`.

        Raises
        ------
        AttributeError
            If the `interaction` matrix does not exist.
        IndexError
            If the `atom_i` index is out of bounds for the initialized structure.
        ValueError
            If new Atom does not change any force constants.

        See Also
        --------
        springcraft.enm_update.covariance_update :
            More information about the covariance update.
        """
        if self._interactions is None:
            raise AttributeError("Interaction matrix must exist.")
        if atom_i < 0 or atom_i >= self._natoms:
            raise IndexError(
                f"{atom_i} is out of bounds for structure of length {self._natoms}."
            )
        if isinstance(new_atom, struc.Atom) and not self._ff.update(atom_i, new_atom):
            raise ValueError("No change in atom detected.")

    @abstractmethod
    def prepare_update(
        self, atom_i: int, atom_j: int, delta: bool | int | float
    ) -> tuple[slice, slice, np.ndarray, float]:
        """
        This method checks arguments and provides values to describe an to the
        interaction matrix A where the interaction strength between atoms `i` and `j` is
        changed by `delta`.

        Does not change any attributes of the ENM class.

        Parameters
        ----------
        atom_i, atom_j : int
            Atom indices with ``atom_i != atom_j``.
        delta : bool or int or float
            A bool value gets interpreted as a turn on/off signal.
            Turning on resets the contact interaction strength to the initial value.
            Turning off sets the contact interaction strength to zero.
            A scalar value changes the contact interaction strength by the given amount.

        Returns
        -------
        slice_i, slice_j : slice
            Index ranges (size k).
        slice_t : ndarray, shape(k,), dtype=float
            Value(s) for the index range.
        delta : float
            Permutation factor.

        Raises
        ------
        AttributeError
            If the `interaction` matrix does not exist.
        IndexError
            If any index is out of bounds or the indices are the same.
        ValueError
            If the resulting `delta` is (nearly) 0.

        Notes
        -----
        The desired update to the `interaction` matrix :math:`\\Gamma` can be described
        as a rank-one update using a vector :math:`\\vec{c}` of appropriate dimensions
        like

        .. math:: \\Gamma + \\delta \\vec{c} \\vec{c}^T

        This update can be achieved using the return values like

        >>> c = np.zeros(n)
        ... c[slice_i] = slice_t
            c[slice_j] = -slice_t
            interactions + delta * np.outer(c, c)
        """
        if self._interactions is None:
            raise AttributeError("Interaction matrix must exist.")
        if atom_i < 0 or atom_i >= self._natoms:
            raise IndexError(
                f"{atom_i} is out of bounds for structure of length {self._natoms}."
            )
        if atom_j < 0 or atom_j >= self._natoms:
            raise IndexError(
                f"{atom_j} is out of bounds for structure of length {self._natoms}."
            )
        if atom_i == atom_j:
            raise IndexError("Cannot modify contact with itself.")

    def _modify_interactions(
        self,
        slice_i: int | np.intp | slice,
        slice_j: int | np.intp | slice,
        slice_t: None | np.ndarray,
        delta: float,
    ):
        """
        Performs a one-rank permutation to the given `interactions` matrix where the
        interaction strength between atoms `i` and `j` is changed by `delta`.

        This method does not perform any input checking.

        Parameters
        ----------
        interactions : np.ndarray, shape(n,n), dtype=float
            The interactions matrix to change.
        slice_i, slice_j : slice
            Index ranges (size k).
        slice_t : ndarray, shape(k,), dtype=float
            Value(s) for the index range.
        delta : float
            Permutation factor.

        See Also
        --------
        springcraft.enm_update.ENMUpdate.prepare_update :
           More information about the update parameters.
        """
        if slice_t is None:
            tensor = delta
        else:
            tensor = np.outer(delta * slice_t, slice_t)

        self._interactions[slice_i, slice_j] -= tensor
        self._interactions[slice_j, slice_i] -= tensor
        self._interactions[slice_i, slice_i] += tensor
        self._interactions[slice_j, slice_j] += tensor

    def _modify_covariance(
        self,
        slice_i: int | np.intp | slice,
        slice_j: int | np.intp | slice,
        slice_t: None | np.ndarray,
        delta: float,
    ):
        # numpydoc ignore=PR01
        """
        Application of the `covariance_update` method to this
        model's covariance matrix.
        """
        covariance_update(
            self._interactions,
            self._covariance,
            slice_i,
            slice_j,
            slice_t,
            delta,
            self._default_ger,
        )

    def _default_ger(self, alpha: float, x: np.ndarray, y: np.ndarray):
        ger(float(alpha), x, y, a=self._covariance.T, overwrite_a=True)

    def mean_square_fluctuation_update(
        self,
        atom_i: int,
        atom_j: int,
        delta: float | int | bool,
        tem: int | float | None = None,
        tem_factors: int | float = K_B,
    ) -> np.ndarray:
        """
        Compute the *mean square fluctuation* for the atoms of a permutated model where
        the interaction strength between atoms `i` and `j` is changed by `delta`.

        Significantly faster than creating a modified model and calculating from
        scratch. Does not change any model attributes.

        Parameters
        ----------
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
        springcraft.nma.mean_square_fluctuation :
            The mean square fluctuation calculation.
        springcraft.nma_update.mean_square_fluctuation_update :
            The mean square fluctuation update.

        Examples
        --------
        The following two snippets create the same result

        >>> msqf = enm.mean_square_fluctuation_update(atom_i, atom_j, delta)

        >>> enm.modify_contact(atom_i, atom_j, delta)
        >>> msqf = enm.mean_square_fluctuation()
        """
        return nma_update.mean_square_fluctuation_update(
            self, atom_i, atom_j, delta, tem, tem_factors
        )

    def bfactor_update(
        self,
        atom_i: int,
        atom_j: int,
        delta: float | int | bool,
        tem: float | int | None = None,
        tem_factors: float | int = K_B,
    ) -> np.ndarray:
        """
        Compute the *mean square fluctuation* for the atoms of a permutated model where
        the interaction strength between atoms `i` and `j` is changed by `delta`.

        Significantly faster than creating a modified model and calculating from
        scratch. Does not change any model attributes.

        Parameters
        ----------
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
        springcraft.nma_update.bfactor_update : The B-factor update.

        Examples
        --------
        The following two snippets create the same result

        >>> bfactors = enm.bfactor_update(atom_i, atom_j, delta)

        >>> enm.modify_contact(atom_i, atom_j, delta)
        >>> bfactors = enm.bfactor()
        """
        return nma_update.bfactor_update(self, atom_i, atom_j, delta, tem, tem_factors)

    def dcc_update(
        self,
        atom_i: int,
        atom_j: int,
        delta: float | int | bool,
        norm: bool = True,
        tem: float | int | None = None,
        tem_factors: float | int = K_B,
    ) -> np.ndarray:
        """
        Compute the *dynamic cross-correlation* between nodes of a permutated model
        where the interaction strength between atoms `i` and `j` is changed by `delta`.

        Significantly faster than creating a modified model and calculating from
        scratch. Does not change any model attributes.

        Parameters
        ----------
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
        springcraft.nma_update.dcc_update : The DCC update.

        Examples
        --------
        The following two snippets create the same result

        >>> dcc = enm.dcc_update(atom_i, atom_j, delta)

        >>> enm.modify_contact(atom_i, atom_j, delta)
        >>> dcc = enm.dcc()
        """
        return nma_update.dcc_update(
            self, atom_i, atom_j, delta, norm, tem, tem_factors
        )


def covariance_update(
    interactions: np.ndarray,
    covariance: np.ndarray,
    slice_i: int | np.intp | slice,
    slice_j: int | np.intp | slice,
    slice_t: None | np.ndarray,
    delta: float,
    update: Callable,
):
    """
    Performs a one-rank permutation on the given `interactions` matrix where the
    interaction strength between atoms `i` and `j` is changed by `delta`.

    The `update` Callable allows for different appliances of the update mechanism.
    It must have the following signature
    ``def update(alpha: float, x: np.ndarray, y: np.ndarray)`` and describe the
    following permutation to covariance matrix ``C``:

    >>> C + alpha * np.outer(x, y)

    This method does not perform any input checking.

    Parameters
    ----------
    interactions, covariance : np.ndarray, shape(n,n), dtype=float
        The `interactions` and `covariance` matrix to change.
    slice_i, slice_j : slice
        Index ranges (size k).
    slice_t : ndarray, shape(k,), dtype=float
        Value(s) for the index range.
    delta : float
        Permutation factor.
    update : Callable
        One-rank permutation to the covariance matrix.

    See Also
    --------
    springcraft.enm_update.ENMUpdate.prepare_update :
       More information about the update parameters.

    Notes
    -----
    Let the `covariance` matrix :math:`\\zeta` be the pseudo inverse of the
    `interactions` matrix :math:`\\Gamma`. Changing the force constant between atoms
    `i` and `j` by an arbitrary amount :math:`\\delta` can be described by a
    rank-one update to :math:`\\Gamma` with a vector :math:`\\vec{c}` of matching
    dimensions like

    .. math:: \\tilde{\\Gamma} = \\Gamma + \\delta \\vec{c} \\vec{c}^T

    This rank-one update can increase or decrease the rank or leave it unchanged.

    If the rank is unchanged than the updated covariance matrix can be described by

    .. math:: \\tilde{\\zeta} = \\zeta +
                                \\frac{\\zeta \\vec{c} \\delta \\vec{c}^T \\zeta}
                                      {1 + \\delta \\vec{c}^T \\zeta \\vec{c}}
                              = \\zeta + \\frac{\\vec{x} \\vec{x}^T}{\\beta}

    with :math:`\\vec{x} = \\zeta \\vec{c}` and
    :math:`\\beta = 1 + \\delta \\vec{c}^T \\zeta \\vec{c}`.
    """
    if slice_t is None:
        slice_t = 1  # pyright: ignore[reportAssignmentType]
        x = covariance[slice_i, :] - covariance[slice_j, :]
        beta = 1 + delta * (x[slice_i] - x[slice_j])
    else:
        x = slice_t @ covariance[slice_i, :] - slice_t @ covariance[slice_j, :]
        beta = 1 + delta * slice_t @ (x[slice_i] - x[slice_j])

    gamma = interactions[slice_i, slice_j]
    if len(gamma.shape) > 0:
        gamma = np.sum(np.diag(gamma))
    if np.abs(gamma) < 1e-6:
        # potential rank increase
        y = interactions @ x
        y[slice_i] -= slice_t
        y[slice_j] += slice_t
        y_dot = y @ y
        if y_dot > 1e-6:
            # rank increase
            update(alpha=1 / y_dot, x=x, y=y)
            update(alpha=1 / y_dot, x=y, y=x)
            update(alpha=beta / (delta * y_dot * y_dot), x=y, y=y)
            return
    elif np.abs(gamma - delta) < 1e-6:
        # potential rank decrease
        if np.abs(beta) < 1e-6:
            # rank decrease
            w = covariance @ x
            x_dot = x @ x
            alpha = (x @ w) / (x_dot**2)

            update(alpha=1 / -x_dot, x=x, y=w)
            update(alpha=1 / -x_dot, x=w, y=x)
            update(alpha=alpha, x=x, y=x)
            return
    # normal case: no rank change
    update(alpha=-delta / beta, x=x, y=x)
