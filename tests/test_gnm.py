import itertools
from os.path import basename, join
from unittest.mock import patch

import biotite.structure.info as strucinfo
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest
from biotite.structure import AtomArray

import springcraft
from springcraft.nma import frequencies
from tests.util import ModifiedForceField, data_dir


def prepare_gnm(file_path, cutoff):
    if file_path.endswith("cif"):
        cif_file = pdbx.CIFFile.read(file_path)
        atoms = pdbx.get_structure(cif_file, model=1)
        assert isinstance(atoms, AtomArray)
    else:
        pdb_file = pdb.PDBFile.read(file_path)
        atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    assert isinstance(ca, AtomArray)

    ff = springcraft.InvariantForceField(cutoff)
    test_gnm = springcraft.GNM(ca, ff)

    return test_gnm


@pytest.mark.parametrize(
    "file_path, cutoff",
    itertools.product(
        [
            join(data_dir(), "1L2Y.cif"),
            join(data_dir(), "104L.cif"),
            join(data_dir(), "10NM.cif"),
        ],
        [4, 7, 13],
    ),
)
def test_adjacency(file_path, cutoff):
    """
    Tests that the cell list and brute force approaches produce
    the same result.
    Tests that PatchedForceFields are correctly handled. Activating
    a contact takes precedence over deactivation.
    """
    cif_file = pdbx.CIFFile.read(file_path)
    atoms = pdbx.get_structure(cif_file, model=1)
    assert isinstance(atoms, AtomArray)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    assert isinstance(ca, AtomArray)
    ff = springcraft.InvariantForceField(cutoff)
    ff = springcraft.PatchedForceField(
        ff,
        contact_shutdown=[2],
        contact_pair_off=[[3, 2], [3, 4], [3, 5]],
        contact_pair_on=[[2, 4], [3, 4], [3, 14]],
        force_constants=[2, 2, 2],
    )

    test_gnm_cell_list = springcraft.GNM(ca, ff, use_cell_list=True)
    test_gnm_brute_force = springcraft.GNM(ca, ff, use_cell_list=False)

    assert np.allclose(test_gnm_cell_list.kirchhoff, test_gnm_brute_force.kirchhoff)

    # contacts turned on
    kirchhoff = test_gnm_cell_list.kirchhoff
    assert kirchhoff[2, 4] == -2
    assert kirchhoff[4, 2] == -2
    assert kirchhoff[3, 4] == -2
    assert kirchhoff[4, 3] == -2
    assert kirchhoff[3, 14] == -2
    assert kirchhoff[14, 3] == -2

    # contacts turned off
    third = np.zeros(len(ca))
    third[2] = 2
    third[4] = -2
    assert np.array_equal(kirchhoff[2, :], third)
    assert np.array_equal(kirchhoff[:, 2], third)
    assert kirchhoff[3, 5] == 0
    assert kirchhoff[5, 3] == 0


@pytest.mark.parametrize(
    "file_path, cutoff",
    itertools.product([join(data_dir(), "1l2y.pdb")], [4, 7, 13]),
)
def test_kirchhoff(file_path, cutoff):
    """
    Compare computed Kirchhoff matrix with output from *ProDy* with
    test files.
    """
    test_gnm = prepare_gnm(file_path, cutoff)
    pdb_name = basename(file_path).split(".")[0]
    ref_kirchhoff = np.genfromtxt(
        join(data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_kirchhoff_{pdb_name}.csv.gz"),
        delimiter=",",
    )

    assert test_gnm.kirchhoff.flatten().tolist() == pytest.approx(
        ref_kirchhoff.flatten().tolist()
    )


def test_mass_weights_simple():
    """
    Expect that mass weighting with unit masses does not have any
    influence on an GNM, but different weights do.
    Expect that supplying TRUE infers residue weights.
    """
    pdb_file = pdb.PDBFile.read(join(data_dir(), "1l2y.pdb"))
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    ff = springcraft.InvariantForceField(7.9)

    # unit masses
    ref_gnm = springcraft.GNM(ca, ff)
    assert ref_gnm.masses is None
    identical_gnm = springcraft.GNM(ca, ff, masses=np.ones(ca.array_length()))
    assert identical_gnm.masses is not None
    assert np.allclose(identical_gnm.masses, np.ones(ca.array_length()))
    assert np.allclose(identical_gnm.kirchhoff, ref_gnm.kirchhoff)

    # arbitrary masses
    residue_weights = np.array(
        [
            strucinfo.mass(res_name, is_residue=True)
            for res_name in ca.res_name  # pyright: ignore[reportOptionalIterable]
        ]
    )
    with pytest.raises(IndexError, match="5 masses for 20 atoms given"):
        springcraft.GNM(ca, ff, masses=residue_weights[:5])
    with pytest.raises(ValueError, match="Masses must not be 0"):
        springcraft.GNM(ca, ff, masses=np.zeros_like(residue_weights))
    different_gnm = springcraft.GNM(ca, ff, masses=residue_weights)
    assert not np.allclose(different_gnm.kirchhoff, ref_gnm.kirchhoff)

    # infer residue weights
    with pytest.raises(
        TypeError, match="An AtomArray is required to automatically infer masses"
    ):
        springcraft.GNM(ca.coord, ff, masses=True)
    residue_weight_gnm = springcraft.GNM(ca, ff, masses=True)
    assert np.allclose(different_gnm.kirchhoff, residue_weight_gnm.kirchhoff)


def test_kirchhoff_covariance_setter():
    """
    Tests that the setter methods check for the correct matrix size and
    that dependend attributes are invalidated
    """
    cif_file = pdbx.CIFFile.read(join(data_dir(), "1L2Y.cif"))
    atoms = pdbx.get_structure(cif_file, model=1)
    assert isinstance(atoms, AtomArray)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    assert isinstance(ca, AtomArray)
    ff = springcraft.InvariantForceField(7)

    test_gnm = springcraft.GNM(ca, ff)
    test_kirchhoff1 = test_gnm.kirchhoff
    test_covariance1 = test_gnm.covariance
    test_eig_val1, _ = test_gnm.eigen()
    assert test_gnm._interactions is not None
    assert test_gnm._kirchhoff is not None
    assert test_gnm._covariance is not None
    assert test_gnm._eigen_values is not None
    assert test_gnm._eigen_vectors is not None

    with pytest.raises(ValueError, match="Expected shape \\(20, 20\\), got \\(5, 5\\)"):
        test_gnm.kirchhoff = np.ones((5, 5))
    test_gnm.kirchhoff = test_kirchhoff1
    assert test_gnm._interactions is not None
    assert test_gnm._kirchhoff is not None
    assert test_gnm._covariance is None
    assert test_gnm._eigen_values is None
    assert test_gnm._eigen_vectors is None

    test_kirchhoff2 = test_gnm.kirchhoff
    test_covariance2 = test_gnm.covariance
    test_eig_val2, _ = test_gnm.eigen()
    assert np.allclose(test_kirchhoff1, test_kirchhoff2)
    assert np.allclose(test_covariance1, test_covariance2)
    assert np.allclose(test_eig_val1, test_eig_val2)

    with pytest.raises(IndexError, match="Expected shape \\(20, 20\\), got \\(5, 5\\)"):
        test_gnm.covariance = np.ones((5, 5))
    test_gnm.covariance = test_covariance2
    assert test_gnm._interactions is None
    assert test_gnm._kirchhoff is None
    assert test_gnm._covariance is not None
    assert test_gnm._eigen_values is None
    assert test_gnm._eigen_vectors is None

    test_kirchhoff3 = test_gnm.kirchhoff
    test_covariance3 = test_gnm.covariance
    test_eig_val3, _ = test_gnm.eigen()
    assert np.allclose(test_kirchhoff2, test_kirchhoff3)
    assert np.allclose(test_covariance2, test_covariance3)
    assert np.allclose(test_eig_val2, test_eig_val3)


@pytest.mark.parametrize(
    "file_path, cutoff",
    itertools.product(
        [join(data_dir(), "1l2y.pdb")],
        # Cutoff must not be too large,
        # otherwise degenerate eigenvalues appear
        [4, 7],
    ),
)
def test_eigen(file_path, cutoff):
    """
    Compare computed eigenvalues and -vectors with output from *ProDy*
    with test files.
    """
    test_gnm = prepare_gnm(file_path, cutoff)

    test_eig_values, test_eig_vectors = test_gnm.eigen()

    pdb_name = basename(file_path).split(".")[0]

    ref_eig_values = np.genfromtxt(
        join(data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_evals_{pdb_name}.csv.gz"),
        delimiter=",",
    )
    ref_eig_vectors = np.genfromtxt(
        join(data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_evecs_{pdb_name}.csv.gz"),
        delimiter=",",
    )

    # Adapt sign of eigenvectors # TODO Is this correct?
    test_eig_vectors *= np.sign(test_eig_vectors[:, 0])[:, np.newaxis]
    ref_eig_vectors *= np.sign(ref_eig_vectors[:, 0])[:, np.newaxis]

    assert np.allclose(test_eig_values[1:], ref_eig_values[1:])
    assert test_eig_values[1:].tolist() == pytest.approx(ref_eig_values[1:].tolist())
    assert test_eig_vectors[1:].flatten().tolist() == pytest.approx(
        ref_eig_vectors[1:].flatten().tolist()
    )


def test_eigen_parameters():
    """
    Tests copies and the number of zero eigenvalues get returned
    depending on the input parameters.
    """
    file_path = join(data_dir(), "1L2Y.cif")
    cutoff = 7
    test_gnm = prepare_gnm(file_path, cutoff)

    eig_val1, eig_vec1 = test_gnm.eigen(copy=False, zero_mask=False)
    eig_val1[1] = 3
    eig_vec1[1, 1] = 3
    eig_val2, eig_vec2 = test_gnm.eigen(copy=False, zero_mask=False)
    assert np.array_equal(eig_val1, eig_val2)
    assert np.array_equal(eig_vec1, eig_vec2)

    test_gnm = prepare_gnm(file_path, cutoff)

    eig_val1, eig_vec1 = test_gnm.eigen(copy=True, zero_mask=False)
    eig_val1[1] = 3
    eig_vec1[1, 1] = 3
    eig_val2, eig_vec2 = test_gnm.eigen(copy=True, zero_mask=False)
    assert not np.array_equal(eig_val1, eig_val2)
    assert not np.array_equal(eig_vec1, eig_vec2)

    test_gnm = prepare_gnm(file_path, cutoff)

    eig_val1, eig_vec1, eig_zero_mask1 = test_gnm.eigen(copy=False, zero_mask=True)
    eig_val1[1] = 3
    eig_vec1[1, 1] = 3
    eig_val2, eig_vec2, eig_zero_mask2 = test_gnm.eigen(copy=False, zero_mask=True)
    assert np.array_equal(eig_val1, eig_val2)
    assert np.array_equal(eig_vec1, eig_vec2)
    assert np.array_equal(eig_zero_mask1, eig_zero_mask2)

    test_gnm = prepare_gnm(file_path, cutoff)

    eig_val1, eig_vec1, eig_zero_mask1 = test_gnm.eigen(copy=True, zero_mask=True)
    eig_val1[1] = 3
    eig_vec1[1, 1] = 3
    eig_val2, eig_vec2, eig_zero_mask2 = test_gnm.eigen(copy=True, zero_mask=True)
    assert not np.array_equal(eig_val1, eig_val2)
    assert not np.array_equal(eig_vec1, eig_vec2)
    assert np.array_equal(eig_zero_mask1, eig_zero_mask2)


@pytest.mark.parametrize(
    "file_path, cutoff",
    itertools.product(
        [
            join(data_dir(), "1L2Y.cif"),
            join(data_dir(), "104L.cif"),
            join(data_dir(), "10NM.cif"),
        ],
        [4, 7, 13],
    ),
)
def test_eigen_before_covariance(file_path, cutoff):
    """
    Tests that the `Kirchhoff` gets calculated if not present and no
    error is produced.
    Tests that covariance matrix calculation uses stored eigenvalues/-vector
    without calculating them all over again.
    """
    test_gnm = prepare_gnm(file_path, cutoff)

    eig_vals, eig_vecs = test_gnm.eigen()
    # eigen() should calc the kirchhoff if not present
    ref_kirchhoff = test_gnm.kirchhoff.copy()
    for eig_val, eig_vec in zip(eig_vals, eig_vecs):
        assert np.allclose(np.matvec(ref_kirchhoff, eig_vec), eig_val * eig_vec)

    with patch("numpy.linalg.eigh") as mock_eigh:
        test_covariance = test_gnm.covariance
        mock_eigh.assert_not_called()
    assert np.allclose(ref_kirchhoff, ref_kirchhoff @ test_covariance @ ref_kirchhoff)
    assert np.allclose(
        test_covariance, test_covariance @ ref_kirchhoff @ test_covariance
    )

    assert np.allclose(ref_kirchhoff, test_gnm.kirchhoff)


@pytest.mark.parametrize(
    "file_path, cutoff",
    itertools.product(
        [
            join(data_dir(), "1L2Y.cif"),
            join(data_dir(), "104L.cif"),
            join(data_dir(), "10NM.cif"),
        ],
        [4, 7, 13],
    ),
)
def test_eigen_after_covariance(file_path, cutoff):
    """
    Tests that calculating the covariance matrix works correctly
    and that in the process the eigenvalues/-vectors are stored
    so that they do not have to be recalculated again when accessing
    them afterwards.
    """
    test_gnm = prepare_gnm(file_path, cutoff)
    ref_kirchhoff = test_gnm.kirchhoff.copy()

    test_covariance = test_gnm.covariance
    assert np.allclose(ref_kirchhoff, ref_kirchhoff @ test_covariance @ ref_kirchhoff)
    assert np.allclose(
        test_covariance, test_covariance @ ref_kirchhoff @ test_covariance
    )

    with patch("numpy.linalg.eigh") as mock_eigh:
        eig_vals, eig_vecs = test_gnm.eigen()
        mock_eigh.assert_not_called()
    for eig_val, eig_vec in zip(eig_vals, eig_vecs):
        assert np.allclose(np.matvec(ref_kirchhoff, eig_vec), eig_val * eig_vec)

    assert np.allclose(ref_kirchhoff, test_gnm.kirchhoff)


@pytest.mark.parametrize(
    "file_path, cutoff", itertools.product([join(data_dir(), "1l2y.pdb")], [4, 7])
)
def test_fluctuation_dcc(file_path, cutoff):
    """
    Comparison of mean-square fluctuations and
    dynamic cross-correlations computed with Springcraft and Prody.
    """
    test_gnm = prepare_gnm(file_path, cutoff)
    test_fluc = test_gnm.mean_square_fluctuation()
    test_dcc = test_gnm.dcc()
    test_dcc_absolute = test_gnm.dcc(norm=False)
    test_dcc_subset = test_gnm.dcc(mode_subset=np.arange(1, 17))

    pdb_name = basename(file_path).split(".")[0]

    reference_fluc = np.genfromtxt(
        join(
            data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_fluctuations_{pdb_name}.csv.gz"
        ),
        delimiter=",",
    )
    reference_dcc = np.genfromtxt(
        join(data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_dcc_norm_{pdb_name}.csv.gz"),
        delimiter=",",
    )
    reference_dcc_norm_subset = np.genfromtxt(
        join(
            data_dir(),
            f"prody_gnm_{cutoff}_ang_cutoff_dcc_norm_subset_{pdb_name}.csv.gz",
        ),
        delimiter=",",
    )
    reference_dcc_absolute = np.genfromtxt(
        join(
            data_dir(), f"prody_gnm_{cutoff}_ang_cutoff_dcc_absolute_{pdb_name}.csv.gz"
        ),
        delimiter=",",
    )

    print(test_dcc_subset.shape)
    print(reference_dcc_norm_subset.shape)
    assert np.allclose(test_fluc, reference_fluc)
    assert np.allclose(test_dcc, reference_dcc)
    assert np.allclose(test_dcc_subset, reference_dcc_norm_subset)
    assert np.allclose(test_dcc_absolute, reference_dcc_absolute)


def test_modify_contact_pair():
    """
    Tests whether permutations to the `kirchhoff` matrix are
    performed correctly and the resulting permutations to the
    `covariance` matrix are correct.
    """
    pdb_file = pdb.PDBFile.read(join(data_dir(), "1l2y.pdb"))
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    ff = springcraft.InvariantForceField(7.0)
    test_gnm = springcraft.GNM(ca, ff)

    # error responses
    assert test_gnm._kirchhoff is None
    with pytest.raises(AttributeError, match="Interaction matrix must exist."):
        test_gnm.modify_contact(1, 2, 1)
    test_gnm.kirchhoff
    with pytest.raises(IndexError):
        test_gnm.modify_contact(-1, 2, 1)
    with pytest.raises(IndexError):
        test_gnm.modify_contact(20, 2, 1)
    with pytest.raises(IndexError):
        test_gnm.modify_contact(1, -2, 1)
    with pytest.raises(IndexError):
        test_gnm.modify_contact(1, 20, 1)
    with pytest.raises(IndexError):
        test_gnm.modify_contact(1, 1, 1)
    with pytest.raises(ValueError):
        test_gnm.modify_contact(1, 2, 0)  # zero delta
    with pytest.raises(ValueError):
        test_gnm.modify_contact(1, 2, True)  # turn on contact that is already on
    with pytest.raises(ValueError):
        test_gnm.modify_contact(1, 19, False)  # turn off contact that is already off

    test_gnm.covariance
    assert test_gnm._covariance is not None

    # arbitrary delta with rank unchanged
    test_gnm.modify_contact(4, 8, 2)
    ref_ff = ModifiedForceField(ff, len(ca), 4, 8, 2)
    ref_gnm = springcraft.GNM(ca, ref_ff)
    assert np.allclose(test_gnm.kirchhoff, ref_gnm.kirchhoff)
    assert np.allclose(test_gnm.covariance, ref_gnm.covariance)

    # rank unchanged
    test_gnm.modify_contact(4, 8, False)
    ref_ff = ModifiedForceField(ff, len(ca), 4, 8, -1)
    ref_gnm = springcraft.GNM(ca, ref_ff)
    assert np.allclose(test_gnm.kirchhoff, ref_gnm.kirchhoff)
    assert np.allclose(test_gnm.covariance, ref_gnm.covariance)

    # rank decrease
    test_gnm.modify_contact(5, 8, False)
    test_gnm.modify_contact(6, 8, False)
    test_gnm.modify_contact(7, 8, False)
    test_gnm.modify_contact(9, 8, False)
    test_gnm.modify_contact(10, 8, False)
    test_gnm.modify_contact(13, 8, False)
    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [4, 5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1, -1],
    )
    ref_gnm = springcraft.GNM(ca, ref_ff)
    assert np.allclose(test_gnm.kirchhoff, ref_gnm.kirchhoff)
    assert np.allclose(test_gnm.covariance, ref_gnm.covariance)

    # rank increase
    test_gnm.modify_contact(4, 8, True)
    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1],
    )
    ref_gnm = springcraft.GNM(ca, ref_ff)
    assert np.allclose(test_gnm.kirchhoff, ref_gnm.kirchhoff)
    assert np.allclose(test_gnm.covariance, ref_gnm.covariance)


def test_frequency_permutation():
    # TODO test negative
    pdb_file = pdb.PDBFile.read(join(data_dir(), "1l2y.pdb"))
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    ff = springcraft.InvariantForceField(7.9)

    test_gnm = springcraft.GNM(ca, ff)
    test_gnm.kirchhoff
    test_gnm.eigen()
    test_freq = test_gnm.frequencies_permutation(3, 5, 1)
    test_freq_subset = test_gnm.frequencies_permutation(3, 5, 1, [4, 5, 6])

    ref_gnm = springcraft.GNM(ca, ff)
    ref_gnm.kirchhoff
    ref_gnm.modify_contact(3, 5, 1)
    ref_freq = ref_gnm.frequencies()

    assert np.allclose(test_freq, ref_freq)
    assert np.allclose(test_freq_subset, ref_freq[4:7])
