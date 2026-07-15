from os.path import dirname, join, realpath

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix
from typing_extensions import override

import springcraft


def data_dir():
    return join(dirname(realpath(__file__)), "data")


def load_protein_structure(pdb_id: str) -> struc.AtomArray:
    file_path = join(dirname(realpath(__file__)), "data", pdb_id + ".cif")
    cif_file = pdbx.CIFFile.read(file_path)
    atoms = pdbx.get_structure(cif_file, model=1)
    return atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]  # pyright: ignore[reportReturnType, reportIndexIssue]


def prepare_gnm(pdb_id: str, cutoff: float | int) -> springcraft.GNM:
    ca = load_protein_structure(pdb_id)
    ff = springcraft.InvariantForceField(cutoff)
    return springcraft.GNM(ca, ff)


def prepare_anm(pdb_id: str, cutoff: float | int) -> springcraft.ANM:
    ca = load_protein_structure(pdb_id)
    ff = springcraft.InvariantForceField(cutoff)
    return springcraft.ANM(ca, ff)


class ModifiedForceField(springcraft.ForceField):
    """
    Modifies force constant (`i`, `j`) of initial ForceField by `delta`.
    Does not work stacked with itself or a PatchedForceField.
    """

    def __init__(
        self,
        ff: springcraft.ForceField,
        natoms: int,
        atom_i: npt.ArrayLike,
        atom_j: npt.ArrayLike,
        delta: npt.ArrayLike,
    ):
        assert not isinstance(ff, (springcraft.PatchedForceField, ModifiedForceField))
        self._ff = ff
        atom_i = np.atleast_1d(np.asarray(atom_i))
        atom_j = np.atleast_1d(np.asarray(atom_j))
        delta = np.atleast_1d(np.asarray(delta))
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
        return force_constants + np.array(self._modifications[atom_i, atom_j]).flatten()

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
