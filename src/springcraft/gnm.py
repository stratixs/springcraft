"""
This module contains the :class:`GNM` class for molecular dynamics
calculations using *Gaussian Network Models*.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann, Faisal Islam, Raphael Sutter"
__all__ = ["GNM"]

import biotite.structure as struc
import numpy as np
from typing_extensions import Literal, Union, overload, override

from springcraft.enm import ENM
from springcraft.forcefield import ForceField


class GNM(ENM):
    """
    This class represents a *Gaussian Network Model*.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or ndarray, shape=(n,3), dtype=float
        The atoms or their coordinates that are part of the model.
        It usually contains only CA atoms.
    force_field : ForceField, natoms=n
        The :class:`ForceField` that defines the cutoff distance and
        pairwise interaction strengths between the given `atoms`.
    masses : bool or ndarray, shape=(n,), dtype=float, optional
        If an array is given, the Kirchhoff matrix is weighted with the
        inverse square root of the given masses.
        If set to true, these masses are automatically inferred from the
        ``res_name`` annotation of `atoms`, instead.
        This requires `atoms` to be an :class:`AtomArray`.
        By default no mass-weighting is applied.
    use_cell_list : bool, optional
        If true, a *cell list* is used to find atoms within cutoff
        distance instead of checking all pairwise atom distances.
        This significantly increases the performance for large number of
        atoms, but is slower for very small systems.
        If the `force_field` does not provide a cutoff, no cell list is
        used regardless.

    Attributes
    ----------
    kirchhoff : ndarray, shape=(n,n), dtype=float
        The *Kirchhoff* matrix for this model. Adjacency matrix of `atoms`
        that are within cutoff distance of another. The weights of this
        adjacency matrix are the force constants of the abstract springs
        between the atoms in the ENM.
        This is not a copy: Create a copy before modifying this matrix.
    covariance : ndarray, shape=(n,n), dtype=float
        The covariance matrix for this model, i.e. the inverted
        *Kirchhofff* matrix. The returned covariance matrix is not scaled
        correctly and does not have the correct unit. To obtain the true
        covariance matrix, you can calculate

        .. math::

            \\text{Cov}_\\text{true} = k_B T \\text{Cov}

        with Boltzman constant :math:`k_B` and absolut temperature
        :math:`[T] = K` in Kelvin.

        This is not a copy: Create a copy before modifying this matrix.
    masses : None or ndarray, shape=(n,), dtype=float
        The mass for each atom, `None` if no mass weighting is applied.
    """

    _kirchhoff: np.ndarray | None

    def __init__(
        self,
        atoms: struc.AtomArray | np.ndarray,
        force_field: ForceField,
        masses=None,
        use_cell_list=True,
    ):
        super().__init__(atoms, force_field, masses, use_cell_list)

        self._kirchhoff = None

    @property
    def kirchhoff(self) -> np.ndarray:
        if self._kirchhoff is None:
            if self._covariance is None:
                atom_i, atom_j, _, sq_dist = self._calc_adjacency()
                force_constants = self._ff.force_constant(atom_i, atom_j, sq_dist)

                self._kirchhoff = np.zeros((self._natoms, self._natoms))
                self._kirchhoff[atom_i, atom_j] = -force_constants

                # Set values for main diagonal
                np.fill_diagonal(self._kirchhoff, -np.sum(self._kirchhoff, axis=0))

                if self._mass_weight_matrix is not None:
                    self._kirchhoff *= self._mass_weight_matrix
            else:
                self._kirchhoff = np.linalg.pinv(
                    self._covariance, hermitian=True, rcond=1e-6
                )
        return self._kirchhoff

    @kirchhoff.setter
    def kirchhoff(self, value: np.ndarray):
        if value.shape != (self._natoms, self._natoms):
            raise ValueError(
                f"Expected shape {(self._natoms, self._natoms)}, got {value.shape}"
            )
        self._kirchhoff = value

        # Invalidate dependent values
        self._covariance = None
        self._eigen_values = None
        self._eigen_vectors = None

    @property
    @override
    def dof_per_node(self) -> int:
        """
        Returns
        -------
        dof_per_node : int
            Returns the Degree of Freedom per atom.
        """
        return 1

    @overload
    def eigen(
        self, zero_mask: Literal[False] = False, copy: bool = True
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def eigen(
        self, zero_mask: Literal[True], copy: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def eigen(
        self, zero_mask=False, copy=True
    ) -> Union[
        tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Compute or fetch the Eigenvalues and Eigenvectors of the
        *Kirchhoff* matrix.

        The laplacian *Kirchhoff* matrix is guaranteed to be
        rank-deficient. Numerical inconsistencies occur during
        eigenvalue calculation. All quasi-zero eigenvalues are set to 0.

        Parameters
        ----------
        zero_mask : bool, optional, default=False
            Whether to return a mask of non-zero eigenvalues.
        copy : bool, optional, default=True
            Whether to return the eigenvalues and eigenvectors as copies.
            If you choose not to return copies a modification to these
            values can reflect in incorrect behaviour of the class.

        Returns
        -------
        eig_values : ndarray, shape=(k,), dtype=float
            Eigenvalues of the *Kirchhoff* matrix in ascending order.
        eig_vectors : ndarray, shape=(k,n), dtype=float
            Eigenvectors of the *Kirchhoff* matrix.
            ``eig_values[i]`` corresponds to ``eigenvectors[i]``.
        zero_mask : ndarray, shape(k,), dtype=bool, optional
            The mask of non zero eigenvalues.
            Only returned if ``zero_mask`` is set.
        """
        self.kirchhoff
        return super().eigen(zero_mask, copy)

    @property
    @override
    def _interactions(self) -> np.ndarray | None:
        return self._kirchhoff

    @staticmethod
    @override
    def _calc_mass_weight_matrix(masses: np.ndarray) -> np.ndarray:
        mass_weights = 1 / np.sqrt(masses)
        return np.outer(mass_weights, mass_weights)

    @override
    def _on_covariance_set(self):
        self._kirchhoff = None
