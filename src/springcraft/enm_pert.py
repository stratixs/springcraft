"""
This module contains the :class:`ENM` class. An abstract base class for molecular dynamics calculations Models.
Extends the ENM base class with low rank perturbation calculation.
"""

__name__ = "springcraft"
__author__ = "Raphael Sutter"
__all__ = ["ENM"]

from abc import abstractmethod

import biotite.structure as struc
import numpy as np
from scipy.linalg import blas
from typing_extensions import Callable

from springcraft import nma_pert
from springcraft.enm import ENM, K_B

ger = blas.get_blas_funcs("ger", dtype=np.float64)


class ENMPert(ENM):
    def modify_contact(self, atom_i: int, atom_j: int, delta: bool | int | float):
        """
        Modifies the force constant in the `interaction` matrix between a
        pair of `atoms`. The interaction between the atoms can either be
        - turned off (the force constant is set to 0),
        - changed by an arbitrary amount `delta` or
        - turned on (reset to the value defined by the `force_field`).
        Turning on only works, if the contact is within cutoff distance.
        Otherwise nothing happens.

        If the `covariance` matrix exists, a low complexity algorithm is
        used to update the covariance matrix according to the small
        pertubation introduced to `interaction` matrix.

        The `interaction` matrix needs to be present.
        The `adjacency` matrix needs to be present, if you do not change
        the interactions strengths by an arbitrary amount. The
        `adjacency` matrix is present, if neither the `interaction`
        nor the `covariance` matrix were set manually.

        Parameters
        ----------
        atom_i, atom_j : int
            Index with ``atom_i != atom_j``
        delta : bool | int | float
            A bool value gets interpreted as a turn on/off signal.
            Turning on resets the contact interaction strength to the initial value.
            Turning off sets the contact interaction strength to zero.
            A scalar value changes the contact interaction strength by the given amount.

        Raises
        ------
        AttributeError
            If the `interaction` matrix does not exist.
        IndexError
            If any index is out of bounds or the indices are the same
        ValueError
            If the resulting `delta` is (nearly) 0.
        """
        slice_i, slice_j, slice_t, delta = self.prepare_one_rank_update(
            atom_i, atom_j, delta
        )

        if self._covariance is not None:
            self._modify_covariance(slice_i, slice_j, slice_t, delta)

        self._modify_interactions(slice_i, slice_j, slice_t, delta)

        # invalidate deoendant values
        self._eigen_values = None
        self._eigen_vectors = None

    @abstractmethod
    def modify_atom(self, atom_i: int, new_atom: bool | struc.Atom):
        """
        Modifies the force constants in the `interation` matrix between the
        `atom_i` and all its adjacent atoms. An atom is defined as adjacent
        if it is within cutoff distance. An atom can be either be
        - turned off (interactions to all atoms are turned off),
        - turned on (interactions to all adjacent atoms are turned on) or
        - modified in a way, that the interaction strengths to its
        adjacent atoms change. This results in a recalculation of all
        interactions of `atom_i`.

        If the `covariance` matrix exists, a low complexity algorithm is
        used to update the covariance matrix according to the small
        pertubation introduced to the `interaction` matrix.

        The `interaction` and the `adjacency` matrix need to be present.
        The `adjacency` matrix is present, if neither the `interaction`
        nor the `covariance` matrix were set manually.

        Parameters
        ----------
        atom_i : int
            The index of the atom to change.
        new_atom : bool or Atom
            A bool gets interpreted as a turn on/off signal.
            An Atom may result in a change to the `ForceField`.
        skip_checks : bool, optional
            Whether to skip argument checks, by default False

        Raises (if checks are enabled)
        ------
        AttributeError
            If the `interaction` matrix does not exist.
        IndexError
            If any indices are out of bounds for the initialized structure.
        ValueError
            If new Atom does not change any force constants.
        """
        if self._interactions is None:
            raise AttributeError("Interaction matrix must exist.")
        if atom_i < 0 or atom_i >= self._natoms:
            raise IndexError(
                f"atom_i={atom_i} is out of bounds for structure of length {self._natoms}."
            )
        if isinstance(new_atom, struc.Atom) and not self._ff.update(atom_i, new_atom):
            raise ValueError("No change in atom detected.")

    @abstractmethod
    def prepare_one_rank_update(
        self, atom_i: int, atom_j: int, delta: bool | int | float
    ) -> tuple[slice, slice, np.ndarray, float]:
        """
        This method checks arguments and provides values to describe a
        one-rank update to the interaction matrix A. The one-rank permutation
        :math:`A + \delta \vec{u} \vec{u}^T` can be described using the return
        values by

        >>> u = np.zeros(n)
        ... u[slice_i] = slice_t
            u[slice_j] = -slice_t
            A + delta * np.outer(u, u)

        Does not change any attributes of the ENM class.

        Parameters
        ----------
        atom_i, atom_j : int
            Atom indices with ``atom_i != atom_j``
        delta : bool or int or float
            A bool value gets interpreted as a turn on/off signal.
            Turning on resets the contact interaction strength to the initial value.
            Turning off sets the contact interaction strength to zero.
            A scalar value changes the contact interaction strength by the given amount.

        Returns
        -------
        slice_i, slice_j : slice
            Index ranges (size k)
        slice_t : ndarray, shape(k,), dtype=float
            Value(s) for the index range
        delta : float
            Permutation factor

        Raises
        ------
        AttributeError
            If the `interaction` matrix does not exist.
        IndexError
            If any index is out of bounds or the indices are the same
        ValueError
            If the resulting `delta` is (nearly) 0.
        """
        if self._interactions is None:
            raise AttributeError("Interaction matrix must exist.")
        if atom_i < 0 or atom_i >= self._natoms:
            raise IndexError(
                f"atom_i={atom_i} is out of bounds for structure of length {self._natoms}."
            )
        if atom_j < 0 or atom_j >= self._natoms:
            raise IndexError(
                f"atom_j={atom_j} is out of bounds for structure of length {self._natoms}."
            )
        if atom_i == atom_j:
            raise IndexError("Cannot modify contact with itself.")

    @staticmethod
    def interactions_rank_one_update(
        interactions: np.ndarray,
        slice_i: int | np.intp | slice,
        slice_j: int | np.intp | slice,
        slice_t: None | np.ndarray,
        delta: float,
    ):
        """
        Performs a one-rank permutation to the given `interactions`
        matrix A. The permutation :math:`A + \delta \vec{u} \vec{u}^T`
        can be described using the attributes by

        >>> u = np.zeros(n)
        ... u[slice_i] = slice_t
            u[slice_j] = -slice_t
            A + delta * np.outer(u, u)

        Parameters
        ----------
        interactions : np.ndarray, shape(n,n), dtype=float
            The interactions matrix to change
        slice_i, slice_j : slice
            Index ranges (size k)
        slice_t : ndarray, shape(k,), dtype=float
            Value(s) for the index range
        delta : float
            Permutation factor

        Note
        ----
        This method does not perform any input checking.
        """
        if slice_t is None:
            tensor = delta
        else:
            tensor = np.outer(delta * slice_t, slice_t)

        interactions[slice_i, slice_j] -= tensor
        interactions[slice_j, slice_i] -= tensor
        interactions[slice_i, slice_i] += tensor
        interactions[slice_j, slice_j] += tensor

    @staticmethod
    def covariance_rank_one_update(
        interactions: np.ndarray,
        covariance: np.ndarray,
        slice_i: int | np.intp | slice,
        slice_j: int | np.intp | slice,
        slice_t: None | np.ndarray,
        delta: float,
        update: Callable,
    ):
        """
        Performs a one-rank permutation on the given `covariance`
        matrix. The permutation to the corresponding interaction matrix
        :math:`A + \delta \vec{u} \vec{u}^T` can be described using the
        attributes by

        >>> u = np.zeros(n)
        ... u[slice_i] = slice_t
            u[slice_j] = -slice_t
            A + delta * np.outer(u, u)

        The `update` Callable allows for different appliances of the
        update mechanism. It must have the following signature
        `def update(alpha: float, x: np.ndarray, y: np.ndarray)` and
        describes the following permutation to covariance matrix C:
        :math:`C + \alpha \vec{x} \vec{y}^T`

        Parameters
        ----------
        interactions, covariance : np.ndarray, shape(n,n), dtype=float
            The `interactions` and `covariance` matrix to change
        slice_i, slice_j : slice
            Index ranges (size k)
        slice_t : ndarray, shape(k,), dtype=float
            Value(s) for the index range
        delta : float
            Permutation factor
        update : Callable
            One-rank permutation to the covariance matrix

        Note
        ----
        This method does not perform any input checking.
        """
        if slice_t is None:
            slice_t = 1  # pyright: ignore[reportAssignmentType]
            x = covariance[slice_i, :] - covariance[slice_j, :]
            beta = 1 + delta * (x[slice_i] - x[slice_j])
        else:
            x = slice_t @ covariance[slice_i, :] - slice_t @ covariance[slice_j, :]
            beta = 1 + delta * slice_t @ (x[slice_i] - x[slice_j])

        if np.abs(beta) < 1e-6:
            # rank decrease
            cov_mul_diff = covariance @ x
            x_dot = x @ x
            alpha = (x @ cov_mul_diff) / (x_dot**2)

            update(alpha=1 / -x_dot, x=x, y=cov_mul_diff)
            update(alpha=1 / -x_dot, x=cov_mul_diff, y=x)
            update(alpha=alpha, x=x, y=x)
            return

        t = interactions[slice_j] @ x + interactions[slice_i] @ x
        if np.max(np.abs(t)) < 1e-6:
            # normal case: no rank change
            update(alpha=-delta / beta, x=x, y=x)
            return

        y = -interactions @ x
        y[slice_i] += slice_t
        y[slice_j] -= slice_t
        y_dot = y @ y
        if y_dot < 1e-6:
            # still normal case but with more precision
            update(alpha=-delta / beta, x=x, y=x)
            return

        else:
            # rank increase
            update(alpha=1 / -y_dot, x=x, y=y)
            update(alpha=1 / -y_dot, x=y, y=x)
            update(alpha=beta / (delta * y_dot * y_dot), x=y, y=y)
            return

    def _modify_interactions(
        self,
        slice_i: int | np.intp | slice,
        slice_j: int | np.intp | slice,
        slice_t: None | np.ndarray,
        delta: float,
    ):
        """
        Application of the `covariance_rank_one_update` method to this
        model's covariance matrix.
        """
        self.interactions_rank_one_update(
            self._interactions,
            slice_i,
            slice_j,
            slice_t,
            delta,
        )

    def _modify_covariance(
        self,
        slice_i: int | np.intp | slice,
        slice_j: int | np.intp | slice,
        slice_t: None | np.ndarray,
        delta: float,
    ):
        """
        Application of the `interactions_rank_one_update` method to this
        model's interaction matrix.
        """
        self.covariance_rank_one_update(
            self._interactions,
            self._covariance,
            slice_i,
            slice_j,
            slice_t,
            delta,
            self._default_ger,
        )

    def _default_ger(self, alpha: float, x: np.ndarray, y: np.ndarray):
        ger(alpha, x, y, a=self._covariance.T, overwrite_a=True)  # pyright: ignore[reportCallIssue]

    def mean_square_fluctuation_pert(
        self,
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
        return nma_pert.mean_square_fluctuation_pert(
            self, atom_i, atom_j, delta, tem, tem_factors
        )

    def bfactor_pert(
        self,
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
        return nma_pert.bfactor_pert(self, atom_i, atom_j, delta, tem, tem_factors)
