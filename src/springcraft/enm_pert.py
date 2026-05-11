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

from springcraft.enm import ENM, K_B


class ENMPert(ENM):
    def modify_contact(self, atom_i, atom_j, delta, skip_checks=False):
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
        atom_i : array_like of int, shape=(k,)
            First atom index
        atom_j : array_like of int, shape=(k,)
            Second atom index with ``atom_i[idx] != atom_j[idx]``
        delta : bool or float or array_like of bool or float, shape (1,) or (k,)
            A bool gets interpreted as a turn on/off signal.
            The amount by which the interaction strength between
        atom i and j gets changed in the `interaction` matrix.
            Must not be 0.
        skip_checks : bool, optional
            Whether to skip argument checks, by default False

        Raises (if checks are enabled)
        ------
        ValueError
            If the `interaction` matrix does not exist or
            If the index arrays cannot be converted to indices or
            If index arrays do not have the same size or
            If the `delta` value(s) are neither float nor bool.
        IndexError
            If any indices are out of bounds for the initialized structure or
            If the contact of an atom with itself shall be modified.
        TypeError
            If `delta` is neither a bool nor a float.
        """
        atom_i = np.atleast_1d(np.asarray(atom_i, dtype=int))
        atom_j = np.atleast_1d(np.asarray(atom_j, dtype=int))
        delta = np.asarray(delta)  # becomes 1d later

        if not skip_checks:
            if self._interactions is None:
                raise AttributeError("Interaction matrix must exist.")
            if atom_i.size != atom_j.size:
                raise ValueError(
                    f"Expected atom index arrays to have the same size "
                    f"but got {atom_i.size} and {atom_j.size}."
                )
            if (
                np.any(atom_i < 0)
                or np.any(atom_i >= self._coord.shape[0])
                or np.any(atom_j < 0)
                or np.any(atom_j >= self._coord.shape[0])
            ):
                raise IndexError(
                    f"Index out of bounds for a structure of length {self._coord.shape[0]}"
                )
            mask = atom_i == atom_j
            if np.any(mask):
                raise IndexError(
                    f"Expected array indices to be different "
                    f"but was {np.stack((atom_i, atom_j), axis=1)[mask, :]} "
                    f"at {np.nonzero(mask)[0]}"
                )
            if delta.size != 1 and delta.size != atom_i.size:
                raise ValueError(
                    f"There must be either 1 delta for all updates "
                    f"or as many as updates. "
                    f"Expected {atom_i.size} or 1 "
                    f"but was {delta.size}."
                )
            if (
                not np.issubdtype(delta.dtype, np.integer)
                and not np.issubdtype(delta.dtype, np.floating)
                and not np.issubdtype(delta.dtype, np.bool)
            ):
                raise TypeError(
                    f"Expected delta to be float or bool but was {delta.dtype}"
                )

        if delta.size == 1:
            delta = np.repeat(delta, atom_i.size)

        idx_i, idx_j, delta = self._modify_contact_values(atom_i, atom_j, delta)

        non_zero_mask = abs(delta) > 1e-8
        self._modify_contact_pair(
            idx_i[non_zero_mask], idx_j[non_zero_mask], delta[non_zero_mask]
        )

    def modify_atom(self, atom_i, new_atom, skip_checks=False):
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
        TypeError
            If `delta` is neither a bool nor an Atom.
        """
        if not skip_checks:
            if self._interactions is None:
                raise AttributeError("Interaction matrix must exist.")
            if atom_i < 0 or atom_i >= self._natoms:
                raise IndexError(
                    f"Index out of bounds for a structure of length {self._natoms}"
                )
            if not isinstance(new_atom, (bool, struc.Atom)):
                raise TypeError(f"Atom must be bool or Atom but was {type(new_atom)}")

        if isinstance(new_atom, bool):
            delta = new_atom
        else:  # new_atom is Atom
            if not self._ff.update(atom_i, new_atom, skip_checks=True):
                return  # ForceField did not change
            delta = True

        length = self._natoms
        atom_j = np.arange(length - 1)
        atom_j[atom_i:] = np.arange(atom_i + 1, length)
        self.modify_contact(
            np.repeat(atom_i, length - 1), atom_j, delta, skip_checks=True
        )

    @abstractmethod
    def _modify_contact_values(
        self, atom_i: np.ndarray, atom_j: np.ndarray, delta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper method to calculate the changes that need to be made to
        the `interaction` matrix when changing the strength of a contact
        between atoms.

        As this is a private function, no input checks are performed.

        Parameters
        ----------
        atom_i : ndarray, shape=(k,), dtype=int
            First atom index
        atom_j : ndarray, shape=(k,), dtype=int
            Second atom index with ``atom_i[idx] != atom_j[idx]``
        delta : ndarray, shape=(k,), dtype=(float or bool)
            The amount by which the interaction strength between
            atom i and j gets changed in the `interaction` matrix.
            True sets to initial value, False to 0. Must not be 0.

        Returns
        -------
        idx_i : ndarray, shape=(k * dof,), dtype=int
            First interaction index
        idx_j : ndarray, shape=(k * dof,), dtype=int
            Second interaction index
        delta : ndarray, shape=(k * dof,), dtype=float
            The amount to change.
        """
        pass  # pragma: no cover

    def _modify_contact_pair(self, atom_i, atom_j, deltas, tem=None, tem_factors=K_B):
        """
        Modifies the interaction strengths between the atoms i and j
        in the `interaction` matrix. Requires for the `interaction` matrix
        to exist and for the change delta to be not null. The interaction
        strength of an atom with itself shall not be changed.

        As this is a private method, the input arguments are not
        validated. Input argument validation happens in the user facing
        functions which will always supply semantically correct
        arguments.

        If the `covariance` matrix exists, this method performs a fast
        permutation to the `covariance` matrix based on the given
        permutation to the `interaction` matrix. This speeds up
        calculations as the `covariance` matrix does not need to be
        calculated by SVD again.

        TODO
        - formulas
        - speedup

        Parameters
        ----------
        atom_i : ndarray, shape=(k,), dtype=int
            First atom index
        atom_j : ndarray, shape=(k,), dtype=int
            Second atom index with ``atom_i[idx] != atom_j[idx]``
        delta : ndarray, shape=(k,), dtype=float
            The amount by which the interaction strength between
            atom i and j gets changed in the `interaction` matrix.
            Must not be 0.
        """
        for i, j, delta in np.nditer([atom_i, atom_j, deltas]):
            if self._covariance is not None:
                x = self._covariance[i, :] - self._covariance[j, :]
                beta = 1 + delta * (x[j] - x[i])
                print("beta: ", beta)

                if np.abs(beta) < 1e-10:  # TODO use relative instead of absolute diff?
                    self._modify_contact_pair_rank_decrease(x)
                elif np.abs(self._covariance[i, j]) < 1e-10:  # TODO mathematical proof
                    self._modify_contact_pair_rank_increase(i, j, delta, x, beta)
                else:
                    # default case
                    self._covariance += np.outer(x * delta / beta, x)

            self._interactions[i, j] += delta  # pyright: ignore[reportOptionalSubscript]
            self._interactions[j, i] += delta  # pyright: ignore[reportOptionalSubscript]
            self._interactions[i, i] -= delta  # pyright: ignore[reportOptionalSubscript]
            self._interactions[j, j] -= delta  # pyright: ignore[reportOptionalSubscript]

            print(
                "maxdiff: ",
                np.max(
                    np.abs(
                        self._interactions
                        - self._interactions @ self._covariance @ self._interactions
                    )
                ),
            )

            # TODO can we make it faster, if we only compute the upper triangle?

        self._eig_values = None
        self._eig_vectors = None

    def _modify_contact_pair_rank_decrease(self, x):
        cov_mul_diff = np.matvec(self._covariance, x)  # pyright: ignore[reportArgumentType]
        x_norm_sq = np.inner(x, x)

        dd_mul_cov = np.outer(x / -x_norm_sq, cov_mul_diff)
        alpha = np.inner(x, cov_mul_diff) / (x_norm_sq * x_norm_sq)
        k_cov_h_mul_kh = np.outer(alpha * x, x)

        self._covariance += dd_mul_cov + dd_mul_cov.T + k_cov_h_mul_kh

    def _modify_contact_pair_rank_increase(self, i, j, delta, x, beta):
        y = -np.matvec(self._interactions, x)  # pyright: ignore[reportArgumentType]
        y[i] += 1
        y[j] -= 1

        y_norm_sq = np.inner(y, y)
        x_y = np.outer(x / -y_norm_sq, y)
        beta_y_y = np.outer(y * beta / (-delta * y_norm_sq * y_norm_sq), y)

        self._covariance += x_y + x_y.T + beta_y_y
