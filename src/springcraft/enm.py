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

from . import nma
from .forcefield import ForceField

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

    # istores squared distances between atoms if below the ff cutoff distance, otherwise 0
    _adjacency: np.ndarray | None
    # euclidean coordinates of each atom
    _coord: np.ndarray
    # pseudo-inverse of the _interaction matrix
    _covariance: np.ndarray | None
    # eigenvalues/-vectors of the _interaction matrix
    _eigen_values: np.ndarray | None
    _eigen_vectors: np.ndarray | None
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
        self._eigen_values = None
        self._eigen_vectors = None

    @property
    def masses(self) -> np.ndarray | None:
        return self._masses

    @property
    def covariance(self) -> np.ndarray:
        if self._covariance is None:
            # same algorithm as linalg.pinv
            # but we want to store calculates eigenvalues in the process
            s, u, zero_mask = self.eigen(zero_mask=True, copy=False)

            si = np.divide(1, s, where=zero_mask, out=np.zeros_like(s))

            self._covariance = u.T @ np.multiply(si[..., np.newaxis], u)

        return self._covariance

    @covariance.setter
    def covariance(self, value: np.ndarray):
        length = self._natoms * self.dof_per_node
        if value.shape != (length, length):
            raise IndexError(f"Expected shape {(length, length)}, got {value.shape}")
        self._covariance = value

        # invalidate dependant values
        self._eigen_values = None
        self._eigen_vectors = None

        self._on_covariance_set()

    @property
    @abstractmethod
    def dof_per_node(self) -> int:
        pass  # pragma: no cover

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
        *interaction* matrix.

        The laplacian `interaction` matrix is guaranteed to be
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
            Eigenvalues of the matrix in ascending order.
        eig_vectors : ndarray, shape=(k,n), dtype=float
            Eigenvectors of the matrix.
            ``eig_values[i]`` corresponds to ``eigenvectors[i]``.
        zero_mask : ndarray, shape(k,), dtype=bool, optional
            The mask of non zero eigenvalues.
            Only returned if ``zero_mask`` is set.
        """
        if self._eigen_values is None or self._eigen_vectors is None:
            assert self._interactions is not None  # should never happen

            self._eigen_values, self._eigen_vectors = np.linalg.eigh(self._interactions)

        val = self._eigen_values
        vec = self._eigen_vectors.T
        if copy:
            val = val.copy()
            vec = vec.copy()

        if zero_mask:
            threshhold = 1e-12 * self._eigen_values[-1]  # max(eig_val) * 10^-12
            mask = np.abs(self._eigen_values) > threshhold
            return val, vec, mask

        return val, vec

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

    def _calc_adjacency(self):
        """
        Calculates the adjacency matrix and returns the values as lists.

        The adjacency matrix is a weigthed graph that spanns the
        elastic network. Atoms are considered as being in contact
        when their distance is below the `ForceField`s threshold.
        The contacts may be overridden by using a `PatchedForceField`.

        When `_use_cell_list` is set uses an optimized strategy
        working on point lists instead off matrizes which reduces
        base operations for sparse cases. Otherwise uses a
        computationally expensive brute-force approach.

        Sets the class attribute `_adjacency` and returns the indices,
        displacements and squared distances as list for faster
        processing in sparse matrices.

        Returns
        -------
        atom_i_list: ndarray, shape=(k,), dtype=list
            list of the first atom indices
        atom_j_list: ndarray, shape=(k,), dtype=list
            list of the sceond atom indices
        disp_list: ndarray, shape=(k,3), dtype=float
            displacement between atom_i and atom_j
        sq_dist_list: ndarray, shape=(k,), dtype=float
            squared distance between atom_i and atom_j
        """
        # Convert into higher precision to avert numerical issues in pseudoinverse calculation
        coord = self._coord.astype(np.float64, copy=False)
        # Find interacting atoms within cutoff distance
        cutoff_dist = self._ff.cutoff_distance

        # contacts to shut off
        turn_off = np.zeros((self._natoms, self._natoms), dtype=np.bool)
        if self._ff.contact_shutdown is not None:
            turn_off[self._ff.contact_shutdown, :] = True
            turn_off[:, self._ff.contact_shutdown] = True
        if self._ff.contact_pair_off is not None:
            contact_off = np.sort(self._ff.contact_pair_off, axis=1).T
            turn_off[contact_off[0, :], contact_off[1, :]] = True
            turn_off[contact_off[1, :], contact_off[0, :]] = True

        if cutoff_dist is not None and self._use_cell_list:
            # use optimized strategy
            sq_dist_matrix = np.zeros((self._natoms, self._natoms))
            cell_list = struc.CellList(coord, cutoff_dist)  # pyright: ignore[reportAttributeAccessIssue]
            adj_indices = cell_list.get_atoms(coord, cutoff_dist)

            # allocate return values
            init_len = adj_indices.size
            if self._ff.contact_pair_on is not None:
                init_len += np.size(self._ff.contact_pair_on, axis=0)
            atom_i_list = np.empty(init_len, dtype=np.int32)
            atom_j_list = np.empty(init_len, dtype=np.int32)
            disp_list = np.empty((init_len, 3))
            sq_dist_list = np.empty(init_len)
            idx = 0

            # iterate over atoms
            i_iter, j_iter = np.nested_iters(adj_indices, [[0], [1]], flags=["c_index"])
            for _ in i_iter:
                atom_i = i_iter.index
                atom_i_coord = coord[atom_i]

                # iterate over contacts
                for atom_j in j_iter:
                    if atom_j < 0:  # pyright: ignore[reportOperatorIssue]
                        # handled every contact for atom
                        break
                    if atom_j >= atom_i:  # pyright: ignore[reportOperatorIssue]
                        # only calc sq_distance for lower triangle
                        continue
                    if turn_off[atom_i, atom_j]:
                        continue

                    disp = struc.displacement(atom_i_coord, coord[atom_j])
                    sq_dist = np.dot(disp, disp)
                    sq_dist_matrix[atom_i, atom_j] = sq_dist
                    sq_dist_matrix[atom_j, atom_i] = sq_dist

                    atom_i_list[idx] = atom_i
                    atom_j_list[idx] = atom_j
                    disp_list[idx, :] = disp
                    sq_dist_list[idx] = sq_dist
                    idx += 1
                    atom_i_list[idx] = atom_j
                    atom_j_list[idx] = atom_i
                    disp_list[idx, :] = disp
                    sq_dist_list[idx] = sq_dist
                    idx += 1

            # manually switched on contacts
            if self._ff.contact_pair_on is not None:
                # TODO move input validation to ForceField
                contacts = np.sort(self._ff.contact_pair_on, axis=1)
                for atom_i, atom_j in np.nditer((contacts[:, 0], contacts[:, 1])):
                    if sq_dist_matrix[atom_i, atom_j] != 0:
                        # atom already on
                        continue

                    disp = struc.displacement(coord[atom_i], coord[atom_j])
                    sq_dist = np.dot(disp, disp)
                    sq_dist_matrix[atom_i, atom_j] = sq_dist
                    sq_dist_matrix[atom_j, atom_i] = sq_dist

                    atom_i_list[idx] = atom_i
                    atom_j_list[idx] = atom_j
                    disp_list[idx, :] = disp
                    sq_dist_list[idx] = sq_dist
                    idx += 1
                    atom_i_list[idx] = atom_j
                    atom_j_list[idx] = atom_i
                    disp_list[idx, :] = disp
                    sq_dist_list[idx] = sq_dist
                    idx += 1

            # trim output arrays
            atom_i_list = np.resize(atom_i_list, idx)
            atom_j_list = np.resize(atom_j_list, idx)
            disp_list = np.resize(disp_list, (idx, 3))
            sq_dist_list = np.resize(sq_dist_list, idx)

        else:
            # brute force
            disp_matrix = struc.displacement(
                coord[np.newaxis, :, :], coord[:, np.newaxis, :]
            )
            sq_dist_matrix = np.sum(disp_matrix * disp_matrix, axis=-1)

            # map which contacts to set zero
            if cutoff_dist is not None:
                map = sq_dist_matrix > cutoff_dist**2
            else:
                map = np.zeros(sq_dist_matrix.shape, dtype=np.bool)
            map |= turn_off
            if self._ff.contact_pair_on is not None:
                contacts_on = np.sort(self._ff.contact_pair_on, axis=1).T
                map[contacts_on[0, :], contacts_on[1, :]] = False
                map[contacts_on[1, :], contacts_on[0, :]] = False
            sq_dist_matrix[map] = 0

            # retrieve lists
            atom_i_list, atom_j_list = np.where(sq_dist_matrix)
            disp_list = disp_matrix[atom_i_list, atom_j_list, :]
            sq_dist_list = sq_dist_matrix[atom_i_list, atom_j_list]

        return atom_i_list, atom_j_list, disp_list, sq_dist_list

    @abstractmethod
    def _on_covariance_set(self):
        pass  # pragma: no cover
