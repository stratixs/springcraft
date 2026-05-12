from os.path import dirname, join, realpath

import biotite.structure as struc
import numpy as np
from scipy.sparse import coo_matrix
from typing_extensions import override

from springcraft import ForceField
from springcraft.forcefield import PatchedForceField


def data_dir():
    return join(dirname(realpath(__file__)), "data")


class ModifiedForceField(ForceField):
    """
    Modifies force constant (`i`, `j`) of initial ForceField by `delta`.
    Does not work stacked with itself or a PatchedForceField.
    """

    def __init__(
        self,
        ff: ForceField,
        natoms: int,
        atom_i: np.ndarray,
        atom_j: np.ndarray,
        delta: np.ndarray,
    ):
        assert not isinstance(ff, (PatchedForceField, ModifiedForceField))
        self._ff = ff
        rows = np.concatenate([atom_i, atom_j])
        cols = np.concatenate([atom_j, atom_i])
        self._contact_pair_on = np.column_stack((rows, cols))
        vals = np.concatenate([delta, delta])
        self._modifications = coo_matrix(
            (vals, (rows, cols)), shape=(natoms, natoms)
        ).tocsr()

    @override
    def force_constant(
        self, atom_i: np.ndarray, atom_j: np.ndarray, sq_distance: np.ndarray
    ) -> np.ndarray:
        force_constants = self._ff.force_constant(atom_i, atom_j, sq_distance)
        if self._ff.cutoff_distance is not None:
            force_constants[sq_distance > self._ff.cutoff_distance**2] = 0
        return force_constants - np.array(self._modifications[atom_i, atom_j]).flatten()

    @override
    def update(self, atom_i: int, new_atom: struc.Atom, skip_checks=False) -> bool:
        return self._ff.update(atom_i, new_atom, skip_checks)

    @property
    @override
    def cutoff_distance(self) -> float | None:
        return self._ff.cutoff_distance

    @property
    @override
    def contact_shutdown(self) -> np.ndarray | None:
        return self._ff.contact_shutdown

    @property
    @override
    def contact_pair_off(self) -> np.ndarray | None:
        return self._ff.contact_pair_off

    @property
    @override
    def contact_pair_on(self) -> np.ndarray | None:
        if self._ff.contact_pair_on is not None:
            return np.concatenate([self._ff.contact_pair_on, self._contact_pair_on])
        return self._contact_pair_on

    @property
    @override
    def natoms(self) -> int | None:
        return self._ff.natoms

    @property
    @override
    def _force_constants(self) -> np.ndarray | None:
        return self._ff._force_constants
