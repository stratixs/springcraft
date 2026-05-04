"""
This module contains the :class:`GNM` class for molecular dynamics
calculations using *Gaussian Network Models*.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann, Faisal Islam"
__all__ = ["GNM"]

import biotite.structure as struc
import numpy as np
from typing_extensions import override

from springcraft.enm import ENM, K_B
from springcraft.forcefield import ForceField

from . import nma
from .interaction import compute_kirchhoff


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
                self._kirchhoff, _ = compute_kirchhoff(
                    self._coord, self._ff, self._use_cell_list
                )
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

    @ENM.covariance.setter
    @override
    def covariance(self, value: np.ndarray):
        super().covariance = value
        # Invalidate dependent values
        self._kirchhoff = None

    def eigen(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the Eigenvalues and Eigenvectors of the
        *Kirchhoff* matrix.

        Returns
        -------
        eig_values : ndarray, shape=(k,), dtype=float
            Eigenvalues of the *Kirchhoff* matrix in ascending order.
        eig_vectors : ndarray, shape=(k,n), dtype=float
            Eigenvectors of the *Kirchhoff* matrix.
            ``eig_values[i]`` corresponds to ``eigenvectors[i]``.
        """
        return nma.eigen(self)

    def frequencies(self) -> np.ndarray:
        """
        Compute the oscillation frequencies of the model.

        The first mode corresponds to rigid-body translations/rotations
        and is omitted in the return value.
        The returned units are arbitrary and should only be compared
        relative to each other.

        Returns
        -------
        frequencies : ndarray, shape=(n,), dtype=float
            Oscillation frequencies of the model in in ascending order.
            *NaN* values mark frequencies corresponding to translations
            or rotations.
        """
        return nma.frequencies(self)

    def mean_square_fluctuation(
        self,
        mode_subset: np.ndarray | None = None,
        tem: float | None = None,
        tem_factors: float = K_B,
    ) -> np.ndarray:
        """
        Compute the *mean square fluctuation* for the atoms according to
        the GNM.
        This is equal to the sum of the diagonal of of the
        GNM covariance matrix, if all k-1 non-trivial
        modes are considered.

        Parameters
        ----------
        mode_subset : ndarray, shape=(n,), dtype=int, optional
            Specifies the subset of modes considered in the MSF
            computation.
            Only non-trivial modes can be selected.
            The first mode is counted as 0 in accordance with
            Python conventions.
            If mode_subset is None, all modes except the first
            trivial mode (0) are included.
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
        """
        return nma.mean_square_fluctuation(self, mode_subset, tem, tem_factors)

    def bfactor(
        self,
        mode_subset: np.ndarray | None = None,
        tem: float | None = None,
        tem_factors: float = K_B,
    ) -> np.ndarray:
        """
        Computes the isotropic B-factors/temperature factors/
        Deby-Waller factors for atoms/coarse-grained nodes using
        the mean-square fluctuation.

        These can be used to relate results obtained from ENMs
        to experimental results.

        Parameters
        ----------
        mode_subset : ndarray, shape=(n,), dtype=int, optional
            Specifies the subset of modes considered in the MSF
            computation.
            Only non-trivial modes can be selected.
            The first mode is counted as 0 in accordance with
            Python conventions.
            If mode_subset is None, all modes except the first
            trivial mode (0) are included.
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
        """
        return nma.bfactor(self, mode_subset, tem, tem_factors)

    def dcc(
        self,
        mode_subset: np.ndarray | None = None,
        norm: bool = True,
        tem: float | None = None,
        tem_factors: float = K_B,
    ) -> np.ndarray:
        r"""
        Computes the normalized *dynamic cross-correlation* between
        nodes of the GNM.

        The DCC is a measure for the correlation in fluctuations
        exhibited by a given pair of nodes. If normalized, pairs with
        correlated fluctuations (same phase and period),
        anticorrelated fluctuations (opposite phase, same period)
        and non-correlated fluctuations are assigned (normalized)
        DCC values of 1, -1 and 0 respectively.
        For results consistent with MSFs, temperature-weighted
        absolute values can be computed (only relevant if results
        are not normalized).

        Parameters
        ----------
        mode_subset : ndarray, shape=(n,), dtype=int, optional
            Specifies the subset of modes considered in the MSF
            computation.
            Only non-trivial modes can be selected.
            The first mode is counted as 0 in accordance with
            Python conventions.
            If mode_subset is None, all modes except the first
            trivial mode (0) are included.
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
            DCC values for ENM nodes.

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
        """
        return nma.dcc(self, mode_subset, norm, tem, tem_factors)

    @property
    @override
    def _interactions(self) -> np.ndarray:
        return self.kirchhoff

    @property
    @override
    def _dof_per_node(self) -> int:
        return 1

    @staticmethod
    @override
    def _calc_mass_weight_matrix(masses: np.ndarray) -> np.ndarray:
        mass_weights = 1 / np.sqrt(masses)
        return np.outer(mass_weights, mass_weights)
