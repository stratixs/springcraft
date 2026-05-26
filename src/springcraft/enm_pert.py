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
import numpy.typing as npt
from scipy.linalg import blas

from springcraft.enm import ENM

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
        atom_i : int
            First atom index
        atom_j : int
            Second atom index with ``atom_i != atom_j``
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
        slice_i, slice_j, slice_t, delta = self._prepare_one_rank_update(
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

    def _modify_interactions(
        self,
        slice_i: int | np.intp | slice,
        slice_j: int | np.intp | slice,
        slice_t: None | np.ndarray,
        delta: float,
    ):
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
        # fmt: off
        if slice_t is None:
            slice_t = 1
            x = self._covariance[slice_i, :] - self._covariance[slice_j, :]
            beta = 1 + delta * (x[slice_i] - x[slice_j])
        else:
            x = slice_t @ self._covariance[slice_i, :] - slice_t @ self._covariance[slice_j, :]
            beta = 1 + delta * slice_t @ (x[slice_i] - x[slice_j])

        if np.abs(beta) < 1e-6:
            # rank decrease
            cov_mul_diff = self._covariance @ x
            x_dot = x @ x
            alpha = (x @ cov_mul_diff) / (x_dot**2)

            ger(alpha=1 / -x_dot, x=x, y=cov_mul_diff, a=self._covariance.T, overwrite_a=True)  # pyright: ignore[reportCallIssue]
            ger(alpha=1 / -x_dot, x=cov_mul_diff, y=x, a=self._covariance.T, overwrite_a=True)  # pyright: ignore[reportCallIssue]
            ger(alpha=alpha, x=x, y=x, a=self._covariance.T, overwrite_a=True)  # pyright: ignore[reportCallIssue]
            return

        t = self._interactions[slice_j] @ x + self._interactions[slice_i] @ x
        if np.max(np.abs(t)) < 1e-6:
            # normal case: no rank change
            ger(alpha=-delta / beta, x=x, y=x, a=self._covariance.T, overwrite_a=True)  # pyright: ignore[reportCallIssue]
            return

        y = -self._interactions @ x
        y[slice_i] += slice_t
        y[slice_j] -= slice_t
        y_dot = y @ y
        if y_dot < 1e-6:
            # still normal case but with more precision
            ger(alpha=-delta / beta, x=x, y=x, a=self._covariance.T, overwrite_a=True)  # pyright: ignore[reportCallIssue]
            return

        else:
            # rank increase
            ger(alpha=1 / -y_dot, x=x, y=y, a=self._covariance.T, overwrite_a=True)  # pyright: ignore[reportCallIssue]
            ger(alpha=1 / -y_dot, x=y, y=x, a=self._covariance.T, overwrite_a=True)  # pyright: ignore[reportCallIssue]
            ger(alpha=beta / (delta * y_dot * y_dot), x=y, y=y, a=self._covariance.T, overwrite_a=True)  # pyright: ignore[reportCallIssue]
            return

    @abstractmethod
    def _prepare_one_rank_update(
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
        atom_i : int
            First atom index
        atom_j : int
            Second atom index with ``atom_i != atom_j``
        delta : bool | int | float
            A bool value gets interpreted as a turn on/off signal.
            Turning on resets the contact interaction strength to the initial value.
            Turning off sets the contact interaction strength to zero.
            A scalar value changes the contact interaction strength by the given amount.

        Returns
        -------
        slice_i : slice
            First index range (size k)
        slice_j : slice
            Second index range (size k)
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
