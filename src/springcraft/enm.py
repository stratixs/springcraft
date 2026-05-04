"""
This module contains the :class:`ENM` class. An abstract base class for molecular dynamics calculations Models.
"""

__name__ = "springcraft"
__author__ = "Raphael Sutter"
__all__ = ["ENM"]

from abc import ABC, abstractmethod

import biotite.structure as struc
import biotite.structure.info as strucinfo
import numpy as np

from springcraft.forcefield import ForceField

K_B = 1.380649e-23
N_A = 6.02214076e23


class ENM(ABC):
    """
    Abstract base class for an *Elastic Network Model*.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or ndarray, shape=(n,3), dtype=float
        The atoms or their coordinates that are part of the model.
        It usually contains only CA atoms.
    force_field : ForceField, natoms=n
        The :class:`ForceField` that defines the force constants between
        the given `atoms`.
    masses : bool or ndarray, shape=(n,), dtype=float, optional
        If an array is given, the interaction matrix is weighted with the
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
    covariance : ndarray, shape=(n,n), dtype=float
        The covariance matrix for this model, i.e. the inverted
        interaction matrix. The returned covariance matrix is not scaled
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

    _coord: np.ndarray
    _covariance: np.ndarray | None
    _ff: ForceField
    _masses: np.ndarray | None
    _masses_weight_matrix: np.ndarray | None
    _natoms: int
    _use_cell_list: bool

    def __init__(
        self,
        atoms: struc.AtomArray | np.ndarray,
        force_field: ForceField,
        masses=None,
        use_cell_list=True,
    ):
        self._coord = np.asarray(struc.coord(atoms))
        self._natoms = len(self._coord)
        self._ff = force_field
        self._use_cell_list = use_cell_list

        if masses is None or masses is False:
            self._masses = None
        elif masses is True:
            if not isinstance(atoms, struc.AtomArray):
                raise TypeError(
                    "An AtomArray is required to automatically infer masses"
                )
            self._masses = np.array(
                [
                    strucinfo.mass(res_name, is_residue=True)
                    for res_name in atoms.res_name  # pyright: ignore[reportOptionalIterable]
                ]
            )
        else:
            if len(masses) != self._natoms:
                raise IndexError(f"{len(masses)} masses for {self._natoms} atoms given")
            if np.any(masses == 0):
                raise ValueError("Masses must not be 0")
            self._masses = np.array(masses, dtype=float)

        if self._masses is not None:
            self._mass_weight_matrix = self._calc_mass_weight_matrix(self._masses)
        else:
            self._mass_weight_matrix = None

        self._covariance = None

    @property
    def masses(self) -> np.ndarray | None:
        return self._masses

    @property
    def covariance(self) -> np.ndarray:
        if self._covariance is None:
            self._covariance = np.linalg.pinv(
                self._interactions, hermitian=True, rcond=1e-6
            )
        return self._covariance

    @covariance.setter
    def covariance(self, value: np.ndarray):
        length = self._natoms * self._dof_per_node
        if value.shape != (length, length):
            raise IndexError(f"Expected shape {(length, length)}, got {value.shape}")
        self._covariance = value

    @property
    @abstractmethod
    def _interactions(self) -> np.ndarray:
        """
        Returns the characteristic interaction matrix for this ENM e.g.
        - the *Kirchhoff* matrix in case of the GNM or
        - the *Hessian* matrix in case of the ANM.

        Primary goal is to create a private accessor for the interaction
        matrix as the characteristic name can differ but the same operations
        are performed on both matrices (like Eigenvalue calculation).

        This is not a copy: be careful when modifying the matrix or create
        a copy.

        Returns
        -------
        interactions : ndarray, dtype=float
            The characteristic interactions matrix.
        """
        pass

    @property
    @abstractmethod
    def _dof_per_node(self) -> int:
        """
        Returns
        -------
        dof_per_node : int
            Returns the Degree of Freedom per atom.
            1 for GNM and 3 for ANM.
        """
        pass

    @staticmethod
    @abstractmethod
    def _calc_mass_weight_matrix(masses: np.ndarray) -> np.ndarray:
        pass
