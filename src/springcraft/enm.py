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
from typing_extensions import Literal, Union, overload

from springcraft import nma
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

    # euclidean coordinates of each atom
    _coord: np.ndarray
    # pseudo-inverse of the _interaction matrix
    _covariance: np.ndarray | None
    # eigenvalues/-vectors of the _interaction matrix
    _eig_values: np.ndarray | None
    _eig_vectors: np.ndarray | None
    # ForceField defining the atom interactions
    _ff: ForceField
    # atom masses
    _masses: np.ndarray | None
    # the mass weight matrix is used to weigh the _interactions matrix
    _masses_weight_matrix: np.ndarray | None
    # the number of atoms
    _natoms: int
    # whether to use an optimized algorithm to calculate the _interaction matrix
    _use_cell_list: bool

    def __init__(
        self,
        atoms: struc.AtomArray | np.ndarray,
        force_field: ForceField,
        masses=None,
        use_cell_list=True,
    ):
        self._coord = np.asarray(struc.coord(atoms)).astype(np.float64, copy=False)
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
        self._eig_values = None
        self._eig_vectors = None

    @property
    def masses(self) -> np.ndarray | None:
        return self._masses

    @property
    def covariance(self) -> np.ndarray:
        if self._covariance is None:
            # same algorithm as linalg.pinv
            # but we want to store calculates eigenvalues in the process
            s, u, n_zero = self.eigen(n_zero=True, copy=False)

            si = np.zeros_like(s)
            si[n_zero:] = 1 / s[n_zero:]

            self._covariance = u.T @ np.multiply(si[..., np.newaxis], u)

        return self._covariance

    @covariance.setter
    def covariance(self, value: np.ndarray):
        length = self._natoms * self.dof
        if value.shape != (length, length):
            raise IndexError(f"Expected shape {(length, length)}, got {value.shape}")
        self._covariance = value

        # invalidate dependant values
        self._eig_values = None
        self._eig_vectors = None

        self._on_covariance_set()

    @property
    def has_covariance(self) -> bool:
        """
        Returns
        -------
        has_covariance : bool
            Whether the covariance is already calculated.
        """
        return self._covariance is not None

    @property
    @abstractmethod
    def dof(self) -> int:
        pass  # pragma: no cover

    @overload
    def eigen(
        self, n_zero: Literal[False] = False, copy: bool = True
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def eigen(
        self, n_zero: Literal[True], copy: bool = True
    ) -> tuple[np.ndarray, np.ndarray, int]: ...

    def eigen(
        self, n_zero=False, copy=True
    ) -> Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, int]]:
        """
        Compute or fetch the Eigenvalues and Eigenvectors of the
        *interaction* matrix.

        The laplacian `interaction` matrix is guaranteed to be
        rank-deficient. Numerical inconsistencies occur during
        eigenvalue calculation. All quasi-zero eigenvalues are set to 0.

        Parameters
        ----------
        n_zero : bool, optional, default=False
            Whether to return number of zero eigenvalues.
            These are the first eigenvalues.
        copy : bool, optional, default=True
            Whether to return the eigenvalues and eigenvectors as copies.
            If you choose not to return copies a modification to these
            values can reflect in incorrect behaviour of the class.

        Returns
        -------
        eig_values : ndarray, shape=(k,), dtype=float
            Eigenvalues of the matrix in ascending order.
        eig_vectors : ndarray, shape=(k,n), dtype=float
            Eigenvectors of the matrix.
            ``eig_values[i]`` corresponds to ``eigenvectors[i]``.
        eigen_n_zero : int, optional
            The number of the (first) zero eigenvalues.
            Only returned if ``n_zero`` is set.
        """
        if self._eig_values is None or self._eig_vectors is None:
            assert self._interactions is not None  # should never happen

            self._eig_values, self._eig_vectors = np.linalg.eigh(self._interactions)

            threshold = self._eig_values[-1] * 1e-6  # max(eig_values) * 10^-6
            i = 0
            while self._eig_values[i] < -threshold:
                i = i + 1
            n_neg = i
            while self._eig_values[i] <= threshold:
                i = i + 1
            n_triv = i + n_neg

            if n_neg:
                # numerical error with some eigenvalues below 0
                v, V, n, m = self._eig_values, self._eig_vectors, n_neg, n_triv
                v[:m], v[m : m + n] = v[n : n + m].copy(), v[:n].copy()
                V[:, :m], V[:, m : m + n] = V[:, n : n + m].copy(), V[:, :n].copy()

                raise RuntimeWarning(
                    "Numerical error during EigenValue calculation. Some analysis might fail."
                )

            self._eigen_n_zero = n_triv

        val = self._eig_values
        vec = self._eig_vectors.T
        if copy:
            val = val.copy()
            vec = vec.copy()

        if n_zero:
            return val, vec, self._eigen_n_zero

        return val, vec

    @property
    def has_eigen(self) -> bool:
        """
        Returns
        -------
        has_eigen : bool
            Whether the eigenvalues and eigenvector are already calculated.
        """
        return self._eig_values is not None and self._eig_vectors is not None

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
        This is equal to the diagonal of the covariance matrix, if all
        k-1 non-trivial modes are considered (subset=None, default).

        Parameters
        ----------
        mode_subset : ndarray, shape=(n,), dtype=int, optional
            Specifies the subset of modes considered in the MSF
            computation.
            Only non-trivial modes can be selected.
            The first mode is counted as 0 in accordance with
            Python conventions.
            If mode_subset is None, all modes except the first
            trivial modes are included.
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
            trivial modes are included.
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
    @abstractmethod
    def _interactions(self) -> np.ndarray | None:
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
        interactions : ndarray, dtype=float or None
            The characteristic interactions matrix.
        """
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def _calc_mass_weight_matrix(masses: np.ndarray) -> np.ndarray:
        pass  # pragma: no cover

    @abstractmethod
    def _on_covariance_set(self):
        pass  # pragma: no cover
