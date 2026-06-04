"""
This module contains the :class:`ANM` class for molecular dynamics
calculations using *Anisotropic Network Models*.
"""

__name__ = "springcraft"
__author__ = "Patrick Kunzmann, Raphael Sutter"
__all__ = ["ANM"]

import biotite.structure as struc
import numpy as np
from typing_extensions import Literal, Union, overload, override

from springcraft import nma
from springcraft.enm_update import ENMUpdate
from springcraft.forcefield import ForceField
from springcraft.interaction import compute_hessian


class ANM(ENMUpdate):
    """
    This class represents an *Anisotropic Network Model*.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or ndarray, shape=(n,3), dtype=float
        The atoms or their coordinates that are part of the model.
        It usually contains only CA atoms.
    force_field : ForceField, natoms=n
        The :class:`ForceField` that defines the cutoff distance and
        pairwise interaction strengths between the given `atoms`.
    masses : bool or ndarray, shape=(n,), dtype=float, optional
        If an array is given, the Hessian is weighted with the inverse
        square root of the given masses.
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
    hessian : ndarray, shape=(n*3,n*3), dtype=float
        The *Hessian* matrix for this model.
        Each dimension is partitioned in the form
        ``[x1, y1, z1, ... xn, yn, zn]``.
        This is not a copy: Create a copy before modifying this matrix.
    covariance : ndarray, shape=(n,n), dtype=float
        The covariance matrix for this model, i.e. the inverted
        *Hessian* matrix. The returned covariance matrix is not scaled
        correctly and does not have the correct unit. To obtain the true
        covariance matrix, you can calculate

        .. math::

            \\text{Cov}_\\text{true} = k_B T \\text{Cov}

        with Boltzmann constant :math:`k_B` and absolute temperature
        :math:`[T] = K` in Kelvin.

        This is not a copy: Create a copy before modifying this matrix.
    masses : None or ndarray, shape=(n,), dtype=float
        The mass for each atom, `None` if no mass weighting is applied.
    """

    _hessian: np.ndarray | None

    def __init__(
        self,
        atoms: struc.AtomArray | np.ndarray,
        force_field: ForceField,
        masses: bool | np.ndarray | None = None,
        use_cell_list: bool = True,
    ):
        super().__init__(atoms, force_field, masses, use_cell_list)

        self._hessian = None

    @property
    def hessian(self) -> np.ndarray:
        if self._hessian is None:
            if self._covariance is None:
                self._hessian, _ = compute_hessian(
                    self._coord, self._ff, self._use_cell_list
                )
                if self._mass_weight_matrix is not None:
                    self._hessian *= self._mass_weight_matrix
            else:
                self._hessian = np.linalg.pinv(
                    self._covariance, hermitian=True, rcond=1e-6
                )
        return self._hessian

    @hessian.setter
    def hessian(self, value: np.ndarray):
        if value.shape != (self._natoms * self.dof, self._natoms * self.dof):
            raise IndexError(
                f"Expected shape "
                f"{(self._natoms * self.dof, self._natoms * self.dof)}, "
                f"got {value.shape}"
            )
        self._hessian = value

        # Invalidate dependent values
        self._covariance = None
        self._eig_values = None
        self._eig_vectors = None

    @property
    @override
    def dof(self) -> int:
        """
        Returns
        -------
        dof : int
            Returns the Degree of Freedom per atom.
        """
        return 3

    @override
    def modify_atom(self, atom_i: int, new_atom: bool | struc.Atom):
        super().modify_atom(atom_i, new_atom)

        disp = self._coord - self._coord[atom_i]
        sq_disp = disp * disp
        sq_dist = np.sum(sq_disp, axis=1)
        sq_dist[atom_i] = np.inf
        comp = sq_disp[:, 0] / sq_dist
        comp[atom_i] = np.inf

        tmp = np.arange(0, self._natoms * self.dof, self.dof)
        delta = self._hessian[atom_i * self.dof, tmp] / comp
        delta[atom_i] = 0

        if new_atom is not False:
            # TODO ff contact_pair_on
            if self._ff.cutoff_distance is None:
                if atom_i > 0:
                    delta[:atom_i] += self._ff.force_constant(
                        np.repeat(atom_i, atom_i), np.arange(atom_i), sq_dist[:atom_i]
                    )
                if atom_i < self._natoms - 1:
                    delta[atom_i + 1 :] += self._ff.force_constant(
                        np.repeat(atom_i, self._natoms - atom_i - 1),
                        np.arange(atom_i + 1, self._natoms),
                        sq_dist[atom_i + 1 :],
                    )
            else:
                idxs = np.argwhere(sq_dist <= self._ff.cutoff_distance**2).flatten()
                delta[idxs] += self._ff.force_constant(
                    np.repeat(atom_i, len(idxs)), idxs, sq_dist[idxs]
                )

        slice_i = slice(atom_i * self.dof, (atom_i + 1) * self.dof)
        atom_j_idxs = np.argwhere(np.abs(delta) > 1e-9).flatten()
        slice_t = disp[atom_j_idxs] / np.sqrt(sq_dist[atom_j_idxs]).reshape(
            (len(atom_j_idxs), 1)
        )
        for k in np.argsort(sq_dist[atom_j_idxs]):
            atom_j = atom_j_idxs[k]
            slice_j = slice(atom_j * self.dof, (atom_j + 1) * self.dof)
            if self._covariance is not None:
                self._modify_covariance(slice_i, slice_j, slice_t[k], delta[atom_j])
            self._modify_interactions(slice_i, slice_j, slice_t[k], delta[atom_j])

    @override
    def prepare_update(
        self, atom_i: int, atom_j: int, delta: bool | int | float
    ) -> tuple[slice, slice, np.ndarray, float]:
        super().prepare_update(atom_i, atom_j, delta)

        disp = self._coord[atom_j] - self._coord[atom_i]
        sq_dist = disp @ disp
        comp = disp[0] ** 2 / sq_dist
        if delta is False:
            # turn off contact
            delta = self._hessian[atom_i * self.dof, atom_j * self.dof] / comp
        elif delta is True:
            # turn on contact (reset to original value)
            if (
                self._ff.cutoff_distance is None
                or sq_dist <= self._ff.cutoff_distance**2
            ):
                # TODO ff contact_pair_on
                delta = self._hessian[atom_i * self.dof, atom_j * self.dof] / comp
                delta += self._ff.force_constant(  # pyright: ignore[reportAssignmentType]
                    np.atleast_1d(atom_i),
                    np.atleast_1d(atom_j),
                    np.atleast_1d(sq_dist),
                )

        if np.abs(delta) < 1e-6:
            raise ValueError("No change in interaction strength.")

        return (
            slice(atom_i * self.dof, (atom_i + 1) * self.dof),
            slice(atom_j * self.dof, (atom_j + 1) * self.dof),
            disp / np.sqrt(sq_dist),
            delta,
        )

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
        *Hessian* matrix.

        The laplacian *Hessian* matrix is guaranteed to be
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
            Eigenvalues of the *Hessian* matrix in ascending order.
        eig_vectors : ndarray, shape=(k,n), dtype=float
            Eigenvectors of the *Hessian* matrix.
            ``eig_values[i]`` corresponds to ``eig_vectors[i]``.
        eigen_n_zero : int, optional
            The number of the (first) zero eigenvalues.
            Only returned if ``n_zero`` is set.
        """
        self.hessian  # calc hessian if non-existant
        return super().eigen(n_zero, copy)

    def normal_mode(
        self,
        index: int,
        amplitude: int,
        frames: int,
        movement: Literal["sine", "triangle"] = "sine",
    ) -> np.ndarray:
        """
        Create displacements for a trajectory depicting the given normal
        mode.

        This is especially useful for molecular animations of the chosen
        oscillation mode.

        Note, that the first six modes correspond to rigid-body translations/
        rotations and are usually omitted in normal mode analysis.

        Parameters
        ----------
        index : int
            The index of the oscillation.
            The index refers to the Eigenvalues obtained from
            :meth:`eigen()`:
            Increasing indices refer to oscillations with increasing
            frequency.
            The first 6 modes represent rigid body movements
            (rotations and translations).
        amplitude : int
            The oscillation amplitude is scaled so that the maximum
            value for an atom is the given value.
        frames : int
            The number of frames (models) per oscillation.
        movement : {'sine', 'triangle'}
            Defines how to depict the oscillation.
            If set to ``'sine'`` the atom movement is sinusoidal.
            If set to ``'triangle'`` the atom movement is linear with
            *sharp* amplitude.

        Returns
        -------
        displacement : ndarray, shape=(m,n,3), dtype=float
            Atom displacements that depict a single oscillation.
            *m* is the number of frames.
        """
        return nma.normal_mode(self, index, amplitude, frames, movement)

    def linear_response(self, force: np.ndarray) -> np.ndarray:
        """
        Compute the atom displacement induced by the given force using
        *Linear Response Theory*. [1]_

        Parameters
        ----------
        force : ndarray, shape=(n,3) or shape=(n*3,), dtype=float
            The force that is applied to the atoms of the model.
            The first dimension gives the atom the force is applied on,
            the second dimension gives the three spatial dimensions.
            Alternatively, a flattened array in the form
            ``[x1, y1, z1, ... xn, yn, zn]`` can be given.

        Returns
        -------
        displacement : ndarray, shape=(n,3), dtype=float
            The vector of displacement induced by the given force.
            The first dimension represents the atom index,
            the second dimension represents spatial dimension.

        References
        ----------
        .. [1] M Ikeguchi, J Ueno, M Sato, A Kidera,
            "Protein Structural Change Upon Ligand Binding:
            Linear Response Theory."
            Phys Rev Lett. 94, 7, 078102 (2005).

        """
        return nma.linear_response(self, force)

    def prs_effector_sensor(
        self, norm: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the perturbation response scanning matrix following and
        the derived effector and sensor profiles after
        Atilgan et al. [1]_ and General et al. [2]_
        PRS matrices can be used to assess the relevance
        of amino acid residues/ANM nodes in transmitting allosteric
        mechanical information.

        The PRS matrix contains mechanical information of the response
        of every node in column index position j after perturbation of
        every amino acid with row index i.
        In the general case, these matrices are normalized by the diagonal
        values to compensate for the self perturbation-response of a
        given residue.

        The effector/sensor profiles are the row and column averages
        of a normalized PRS, respectively.
        These profiles allow an assessment, whether perturbations
        at a given residue position are effectively spread to
        the remaining residues (high effectivity) and how perturbations
        at other positions affect a residue (high sensitivity).

        Parameters
        ----------
        norm: bool, optional
            Normalize by the self perturbation-response of the perturbed
            ANM node.

        Returns
        -------
        prs_matrix : ndarray, shape=(n,n), dtype=float
            A 2D matrix with the perturbation response at each ENM node position.
            The row indices i correspond to the perturbed node with the same index,
            the responses of nodes j are stored at the respective columnar
            index positions.
            The whole matrix is normalized to the value of the self-perturbation
            response of node i stored in the diagonal i=j for 'norm=True'.
        effector_profile: ndarray, shape=(n), dtype=float
            Row averages of the non-diagonal row elements of the PRS.
            This profiles the effectiveness/influence of a given amino acid
            in relaying a mechanical signal to the whole structure
            after perturbation.
        sensor_profile: ndarray, shape=(n), dtype=float
            Column average of the non-diagonal row elements of the PRS.
            The resultant array is a measure for the sensitivity of
            the corresponding amino acid to perturbations in other positions.

        References
        ----------
        .. [1] C Atilgan, AR Atilgan
            "Perturbation-Response Scanning Reveals Ligand Entry-Exit
            Mechanisms of Ferric Binding Protein."
            PLoS Comput Biol 5(10) (2009).
        .. [2] IJ General, Y Liu, ME Blackburn, W Mao, LM Gierasch et al.
            "ATPase Subdomain IA Is a Mediator of Interdomain Allostery
            in Hsp70 Molecular Chaperones."
            PLOS Computational Biology 10(5) (2014).
        """
        prs_mat = nma.prs(self, norm)
        eff, sens = nma.effector_sensor(prs_mat)
        return prs_mat, eff, sens

    @property
    @override
    def _interactions(self) -> np.ndarray | None:
        return self._hessian

    @staticmethod
    @override
    def _calc_mass_weight_matrix(masses: np.ndarray) -> np.ndarray:
        mass_weights = 1 / np.sqrt(masses)
        # 3 repetitions,
        # as the Hessian has 3 entries (x, y, z) for each atom
        mass_weights = np.repeat(mass_weights, 3)
        return np.outer(mass_weights, mass_weights)

    @override
    def _on_covariance_set(self):
        self._hessian = None
