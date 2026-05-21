import glob
import itertools
from os.path import basename, join
from unittest.mock import patch

import biotite.structure as struc
import biotite.structure.info as strucinfo
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest
from biotite.structure import AtomArray

import springcraft

from .util import ModifiedForceField, data_dir


def prepare_springcraft_anm(file_path, cutoff):
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
    test_anm = springcraft.ANM(ca, ff)

    return test_anm


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

    test_anm_cell_list = springcraft.ANM(ca, ff, use_cell_list=True)
    test_anm_brute_force = springcraft.ANM(ca, ff, use_cell_list=False)

    hessian = test_anm_cell_list.hessian
    assert np.allclose(hessian, test_anm_brute_force.hessian, atol=1e-7)

    # contacts turned on
    assert not np.array_equal(hessian[6:9, 12:15], np.zeros((3, 3)))  # (2, 4)
    assert not np.array_equal(hessian[12:15, 6:9], np.zeros((3, 3)))  # (4, 2)
    assert not np.array_equal(hessian[10:12, 12:15], np.zeros((3, 3)))  # (3, 4)
    assert not np.array_equal(hessian[12:15, 10:12], np.zeros((3, 3)))  # (4, 3)
    assert not np.array_equal(hessian[10:12, 42:45], np.zeros((3, 3)))  # (3, 14)
    assert not np.array_equal(hessian[42:45, 10:12], np.zeros((3, 3)))  # (14, 3)

    # contacts turned off
    third = np.zeros((3, np.size(hessian, axis=0)))
    third[:, 6:9] = hessian[6:9, 6:9]
    third[:, 12:15] = hessian[6:9, 12:15]
    assert np.array_equal(hessian[6:9, :], third)
    third = np.zeros((np.size(hessian, axis=0), 3))
    third[6:9, :] = hessian[6:9, 6:9]
    third[12:15, :] = hessian[6:9, 12:15]
    assert np.array_equal(hessian[:, 6:9], third)
    assert not np.array_equal(hessian[10:12, 15:18], np.zeros((3, 3)))  # (3, 5)
    assert not np.array_equal(hessian[15:18, 10:12], np.zeros((3, 3)))  # (5, 3)


@pytest.mark.parametrize("file_path", glob.glob(join(data_dir(), "*.pdb")))
def test_covariance(file_path):
    test_anm = prepare_springcraft_anm(file_path, cutoff=13)
    test_hessian = test_anm.hessian
    test_covariance = test_anm.covariance

    assert np.allclose(
        test_hessian, np.dot(test_hessian, np.dot(test_covariance, test_hessian))
    )
    assert np.allclose(
        test_covariance, np.dot(test_covariance, np.dot(test_hessian, test_covariance))
    )


def test_mass_weights_simple():
    """
    Expect that mass weighting with unit masses does not have any
    influence on an ANM, but different weights do.
    """
    pdb_file = pdb.PDBFile.read(join(data_dir(), "1l2y.pdb"))
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    ff = springcraft.InvariantForceField(7.9)

    ref_anm = springcraft.ANM(ca, ff)
    identical_anm = springcraft.ANM(ca, ff, masses=np.ones(ca.array_length()))
    different_anm = springcraft.ANM(
        ca, ff, masses=np.arange(1, ca.array_length() + 1, dtype=float)
    )

    assert np.allclose(identical_anm.hessian, ref_anm.hessian)
    assert not np.allclose(different_anm.hessian, ref_anm.hessian)

    # unit masses
    ref_anm = springcraft.ANM(ca, ff)
    assert ref_anm.masses is None
    identical_anm = springcraft.ANM(ca, ff, masses=np.ones(ca.array_length()))
    assert identical_anm.masses is not None
    assert np.allclose(identical_anm.masses, np.ones(ca.array_length()))
    assert np.allclose(identical_anm.hessian, ref_anm.hessian)

    # arbitrary masses
    residue_weights = np.array(
        [
            strucinfo.mass(res_name, is_residue=True)
            for res_name in ca.res_name  # pyright: ignore[reportOptionalIterable]
        ]
    )
    with pytest.raises(IndexError, match="5 masses for 20 atoms given"):
        springcraft.ANM(ca, ff, masses=residue_weights[:5])
    with pytest.raises(ValueError, match="Masses must not be 0"):
        springcraft.ANM(ca, ff, masses=np.zeros_like(residue_weights))
    different_anm = springcraft.ANM(ca, ff, masses=residue_weights)
    assert not np.allclose(different_anm.hessian, ref_anm.hessian)

    # infer residue weights
    with pytest.raises(
        TypeError, match="An AtomArray is required to automatically infer masses"
    ):
        springcraft.ANM(ca.coord, ff, masses=True)
    residue_weight_anm = springcraft.ANM(ca, ff, masses=True)
    assert np.allclose(different_anm.hessian, residue_weight_anm.hessian)


def test_hessian_covariance_setter():
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

    test_anm = springcraft.ANM(ca, ff)
    test_hessian1 = test_anm.hessian
    test_covariance1 = test_anm.covariance
    test_eig_val1, _ = test_anm.eigen()
    assert test_anm._interactions is not None
    assert test_anm._hessian is not None
    assert test_anm._covariance is not None
    assert test_anm._eigen_values is not None
    assert test_anm._eigen_vectors is not None

    with pytest.raises(IndexError, match="Expected shape \\(60, 60\\), got \\(5, 5\\)"):
        test_anm.hessian = np.ones((5, 5))
    test_anm.hessian = test_hessian1
    assert test_anm._interactions is not None
    assert test_anm._hessian is not None
    assert test_anm._covariance is None
    assert test_anm._eigen_values is None
    assert test_anm._eigen_vectors is None

    test_hessian2 = test_anm.hessian
    test_covariance2 = test_anm.covariance
    test_eig_val2, _ = test_anm.eigen()
    assert np.allclose(test_hessian1, test_hessian2)
    assert np.allclose(test_covariance1, test_covariance2)
    assert np.allclose(test_eig_val1, test_eig_val2)

    with pytest.raises(IndexError, match="Expected shape \\(60, 60\\), got \\(5, 5\\)"):
        test_anm.covariance = np.ones((5, 5))
    test_anm.covariance = test_covariance2
    assert test_anm._interactions is None
    assert test_anm._hessian is None
    assert test_anm._covariance is not None
    assert test_anm._eigen_values is None
    assert test_anm._eigen_vectors is None

    test_hessian3 = test_anm.hessian
    test_covariance3 = test_anm.covariance
    test_eig_val3, _ = test_anm.eigen()
    assert np.allclose(test_hessian2, test_hessian3)
    assert np.allclose(test_covariance2, test_covariance3)
    assert np.allclose(test_eig_val2, test_eig_val3)


@pytest.mark.parametrize("file_path", glob.glob(join(data_dir(), "*.pdb")))
def test_compare_eigenvals_BiophysConnectoR(file_path):
    """
    Compare non-mass-weighted eigenvalues with those computed with
    BiophysConnectoR for eANMs.
    """
    pdb_file = pdb.PDBFile.read(file_path)
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    ff = springcraft.TabulatedForceField.e_anm(ca)
    eanm = springcraft.ANM(ca, ff)

    ref_name = basename(file_path).split(".")[0]
    ref_file = f"biophysconnector_anm_eanm_evals_{ref_name}.csv"

    test_eigenval, _ = eanm.eigen()

    # Load .csv.gz file data from BiophysConnectoR
    ref_eigenval = np.genfromtxt(
        join(data_dir(), ref_file), skip_header=1, delimiter=","
    )

    # Omit trivial modes
    assert np.allclose(test_eigenval[6:], ref_eigenval[6:])


@pytest.mark.parametrize(
    "file_path, ff_name",
    itertools.product(
        glob.glob(join(data_dir(), "*.pdb")), ["Hinsen", "sdENM", "pfENM"]
    ),
)
def test_mass_weights_eigenvals(file_path, ff_name):
    """
    Compare mass-weighted eigenvalues with reference values obtained
    with bio3d to test the correctness of the mass-weighting procedure
    and the validity of results obtained with SVD.
    To this end, bio3d-assigned masses are used.
    """
    pdb_file = pdb.PDBFile.read(file_path)
    pdb_name = basename(file_path).split(".")[0]
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    if ff_name == "Hinsen":
        ff = springcraft.HinsenForceField()
        ff_bio3d_str = "calpha"
    if ff_name == "sdENM":
        ff = springcraft.TabulatedForceField.sd_enm(ca)
        ff_bio3d_str = "sdenm"

        # NOTE: Different chains are not correctly identified in bio3d
        # -> Connect single chains with modified covalent contacts
        #    in springcraft
        if struc.get_chain_count(ca) > 1:
            after_chainbreak = struc.check_res_id_continuity(ca)
            prior_chainbreak = after_chainbreak - 1
            contact_mod_pairs = np.array([prior_chainbreak, after_chainbreak]).T
            bonded_force_constant = 43.52 * 0.0083144621 * 300 * 10
            ff = springcraft.PatchedForceField(
                ff,
                contact_pair_off=contact_mod_pairs,
                contact_pair_on=contact_mod_pairs,
                force_constants=np.full(len(contact_mod_pairs), bonded_force_constant),
            )
    if ff_name == "pfENM":
        ff = springcraft.ParameterFreeForceField()
        ff_bio3d_str = "pfanm"

    # ENM-NMA -> Reference
    bio3d_masses_file = f"bio3d_mass_{pdb_name}.csv.gz"
    bio3d_eigvals_file = f"bio3d_anm_{ff_bio3d_str}_ff_evals_mw_{pdb_name}.csv.gz"
    reference_masses = np.genfromtxt(join(data_dir(), bio3d_masses_file), delimiter=",")
    reference_eigenval = np.genfromtxt(
        join(data_dir(), bio3d_eigvals_file), delimiter=","
    )

    test = springcraft.ANM(ca, ff, masses=reference_masses)
    test_eigenval, _ = test.eigen()
    assert np.allclose(
        test_eigenval[6:], reference_eigenval[6:], rtol=5e-03, atol=2e-03
    )


def test_eigen_parameters():
    """
    Tests copies and the number of zero eigenvalues get returned
    depending on the input parameters.
    """
    file_path = join(data_dir(), "1L2Y.cif")
    cutoff = 7
    test_anm = prepare_springcraft_anm(file_path, cutoff)

    eig_val1, eig_vec1 = test_anm.eigen(copy=False, zero_mask=False)
    eig_val1[1] = 3
    eig_vec1[1, 1] = 3
    eig_val2, eig_vec2 = test_anm.eigen(copy=False, zero_mask=False)
    assert np.array_equal(eig_val1, eig_val2)
    assert np.array_equal(eig_vec1, eig_vec2)

    test_anm = prepare_springcraft_anm(file_path, cutoff)

    eig_val1, eig_vec1 = test_anm.eigen(copy=True, zero_mask=False)
    eig_val1[1] = 3
    eig_vec1[1, 1] = 3
    eig_val2, eig_vec2 = test_anm.eigen(copy=True, zero_mask=False)
    assert not np.array_equal(eig_val1, eig_val2)
    assert not np.array_equal(eig_vec1, eig_vec2)

    test_anm = prepare_springcraft_anm(file_path, cutoff)

    eig_val1, eig_vec1, eig_zero_mask1 = test_anm.eigen(copy=False, zero_mask=True)
    eig_val1[1] = 3
    eig_vec1[1, 1] = 3
    eig_val2, eig_vec2, eig_zero_mask2 = test_anm.eigen(copy=False, zero_mask=True)
    assert np.array_equal(eig_val1, eig_val2)
    assert np.array_equal(eig_vec1, eig_vec2)

    test_anm = prepare_springcraft_anm(file_path, cutoff)

    eig_val1, eig_vec1, eig_zero_mask1 = test_anm.eigen(copy=True, zero_mask=True)
    eig_val1[1] = 3
    eig_vec1[1, 1] = 3
    eig_val2, eig_vec2, eig_zero_mask2 = test_anm.eigen(copy=True, zero_mask=True)
    assert not np.array_equal(eig_val1, eig_val2)
    assert not np.array_equal(eig_vec1, eig_vec2)


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
    Tests that the `Hessian` gets calculated if not present and no
    error is produced.
    Tests that covariance matrix calculation uses stored eigenvalues/-vector
    without calculating them all over again.
    """
    test_anm = prepare_springcraft_anm(file_path, cutoff)

    eig_vals, eig_vecs = test_anm.eigen()
    # eigen() should calc the hessian if not present
    ref_hessian = test_anm.hessian.copy()
    for eig_val, eig_vec in zip(eig_vals, eig_vecs):
        assert np.allclose(np.matvec(ref_hessian, eig_vec), eig_val * eig_vec)

    with patch("numpy.linalg.eigh") as mock_eigh:
        test_covariance = test_anm.covariance
        mock_eigh.assert_not_called()
    assert np.allclose(ref_hessian, ref_hessian @ test_covariance @ ref_hessian)
    assert np.allclose(test_covariance, test_covariance @ ref_hessian @ test_covariance)

    assert np.allclose(ref_hessian, test_anm.hessian)


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
    test_anm = prepare_springcraft_anm(file_path, cutoff)
    ref_hessian = test_anm.hessian.copy()

    test_covariance = test_anm.covariance
    assert np.allclose(ref_hessian, ref_hessian @ test_covariance @ ref_hessian)
    assert np.allclose(test_covariance, test_covariance @ ref_hessian @ test_covariance)

    with patch("numpy.linalg.eigh") as mock_eigh:
        eig_vals, eig_vecs = test_anm.eigen()
        mock_eigh.assert_not_called()
    for eig_val, eig_vec in zip(eig_vals, eig_vecs):
        assert np.allclose(np.matvec(ref_hessian, eig_vec), eig_val * eig_vec)

    assert np.allclose(ref_hessian, test_anm.hessian)


@pytest.mark.parametrize(
    "ff_name", ["ANM_standard", "Hinsen", "eANM", "sdENM", "pfENM"]
)
def test_frequency_fluctuation_dcc(ff_name):
    """
    Compare quantities commonly computed as part of NMA.
    Prody/BioPhysConnectoR/Bio3d are used as references.
    """
    K_B = 1.380649e-23
    N_A = 6.02214076e23
    tem = 300

    pdb_file = pdb.PDBFile.read(join(data_dir(), "1l2y.pdb"))
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    # Prody
    if ff_name == "ANM_standard":
        # No mass or temperature weighting for comparison with Prody
        ff = springcraft.InvariantForceField(13)
        test_anm = springcraft.ANM(ca, ff)
        test_freq_no_mw = test_anm.frequencies()
        test_fluc_nomw = test_anm.mean_square_fluctuation(tem=None)

        test_dcc = test_anm.dcc()
        test_dcc_absolute = test_anm.dcc(norm=False)
        test_dcc_subset = test_anm.dcc(mode_subset=np.arange(6, 36))

        ## Read in Prody reference csv files
        # Evals and fluctuations
        prody_ff_cutoff_name = "anm_13_ang_cutoff"
        prody_evals = np.genfromtxt(
            join(data_dir(), f"prody_{prody_ff_cutoff_name}_evals_1l2y.csv.gz"),
            delimiter=",",
        )
        reference_freq = 1 / (2 * np.pi) * np.sqrt(prody_evals)
        reference_fluc = np.genfromtxt(
            join(data_dir(), f"prody_{prody_ff_cutoff_name}_fluctuations_1l2y.csv.gz"),
            delimiter=",",
        )

        # DCC
        ref_dcc = np.genfromtxt(
            join(data_dir(), f"prody_{prody_ff_cutoff_name}_dcc_norm_1l2y.csv.gz"),
            delimiter=",",
        )

        # Subset: First 30 non-triv. modes
        ref_dcc_norm_subset = np.genfromtxt(
            join(
                data_dir(), f"prody_{prody_ff_cutoff_name}_dcc_norm_subset_1l2y.csv.gz"
            ),
            delimiter=",",
        )
        # Absolute values
        ref_dcc_absolute = np.genfromtxt(
            join(data_dir(), f"prody_{prody_ff_cutoff_name}_dcc_absolute_1l2y.csv.gz"),
            delimiter=",",
        )

        assert np.allclose(test_freq_no_mw[6:], reference_freq[6:])
        assert np.allclose(test_fluc_nomw, reference_fluc)
        assert np.allclose(test_dcc, ref_dcc)
        assert np.allclose(test_dcc_absolute, ref_dcc_absolute)
        assert np.allclose(test_dcc_subset, ref_dcc_norm_subset)

    # References computed with R packages -> read .csv.gz files in "data"
    else:
        # BioPhysConnectoR -> no internal computation of DCCs available;
        #                     no mass-/temperature weighting
        if ff_name == "eANM":
            ff = springcraft.TabulatedForceField.e_anm(ca)
            test_nomw = springcraft.ANM(ca, ff)
            ref_fluc = "biophysconnector_anm_eanm_bfacs_1l2y.csv"
            test_nomw = springcraft.ANM(ca, ff)
            test_fluc_nomw = test_nomw.mean_square_fluctuation()
            # -> For alternative MSF computation method;
            # no temperature weighting
            tem_scaling = 1
            tem = 1

            ## Read in reference file
            reference_fluc = np.genfromtxt(
                join(data_dir(), ref_fluc), skip_header=1, delimiter=","
            )

        # Bio3d -> Mass- and temperature weighting
        else:
            if ff_name == "Hinsen":
                ff = springcraft.HinsenForceField()
                ff_bio3d_str = "calpha"
            elif ff_name == "sdENM":
                ff = springcraft.TabulatedForceField.sd_enm(ca)
                ff_bio3d_str = "sdenm"
            elif ff_name == "pfENM":
                ff = springcraft.ParameterFreeForceField()
                ff_bio3d_str = "pfanm"

            ## Read in reference files
            # Frequencies and fluctuations
            bio3d_masses_file = "bio3d_mass_1l2y.csv.gz"
            reference_masses = np.genfromtxt(
                join(data_dir(), bio3d_masses_file), delimiter=","
            )
            reference_freq = np.genfromtxt(
                join(
                    data_dir(), f"bio3d_anm_{ff_bio3d_str}_ff_frequencies_mw_1l2y.csv"
                ),
                delimiter=",",
            )
            reference_fluc = np.genfromtxt(
                join(
                    data_dir(),
                    f"bio3d_anm_{ff_bio3d_str}_ff_fluctuations_non_mw_1l2y.csv",
                ),
                delimiter=",",
            )
            reference_fluc_subset = np.genfromtxt(
                join(
                    data_dir(),
                    f"bio3d_anm_{ff_bio3d_str}_ff_fluctuations_subset_mw_1l2y.csv",
                ),
                delimiter=",",
            )

            # DCC and DCC subset (first 30 nontriv. modes)
            reference_dcc = np.genfromtxt(
                join(data_dir(), f"bio3d_anm_{ff_bio3d_str}_ff_dcc_mw_1l2y.csv"),
                delimiter=",",
            )
            reference_dcc_subset = np.genfromtxt(
                join(data_dir(), f"bio3d_anm_{ff_bio3d_str}_ff_dcc_subset_mw_1l2y.csv"),
                delimiter=",",
            )

            tem_scaling = K_B * N_A
            test_nomw = springcraft.ANM(ca, ff)
            test_fluc_nomw = test_nomw.mean_square_fluctuation(
                tem=tem, tem_factors=tem_scaling
            )

            test = springcraft.ANM(ca, ff, masses=reference_masses)
            test_freq = test.frequencies()

            ## Scale for consistency with bio3d; T=300 K; no mass weighting
            # Start with mass_weighted eigenvals
            test_fluc = test.mean_square_fluctuation(
                tem=tem, tem_factors=tem_scaling
            ) / (1000 * reference_masses)

            # Select a subset of modes: 12-33
            test_fluc_subset = test.mean_square_fluctuation(
                tem=tem, tem_factors=tem_scaling, mode_subset=np.arange(11, 33)
            )
            test_fluc_subset /= 1000 * reference_masses

            # DCCs
            test_dcc = test.dcc()
            # Only consider the first 30 non-triv. modes
            # Mode 6-36 (conventional enumeration)
            test_dcc_subset = test.dcc(mode_subset=np.arange(6, 36))

        ## No mass-weighting
        # Alternative Method for MSF computation considering all modes
        diag = test_nomw.covariance.diagonal()
        reshape_diag = np.reshape(diag, (len(test_nomw._coord), -1))

        # Compute MSF directly from covariance matrix
        msqf_alternative = np.sum(reshape_diag, axis=1) * tem_scaling * tem

        if ff_name == "eANM":
            assert np.allclose(test_fluc_nomw, reference_fluc)
        # Bio3d-FFs
        else:
            assert np.allclose(
                test_freq[6:], reference_freq[6:], rtol=5e-03, atol=2e-03
            )
            assert np.allclose(test_fluc, reference_fluc, rtol=5e-03, atol=2e-03)
            assert np.allclose(
                test_fluc_subset, reference_fluc_subset, rtol=5e-03, atol=2e-03
            )
            print(test_dcc)
            print(reference_dcc)
            print(np.max(np.abs(test_dcc - reference_dcc)))
            assert np.allclose(test_dcc, reference_dcc, rtol=5e-03, atol=2e-03)
            assert np.allclose(
                test_dcc_subset, reference_dcc_subset, rtol=5e-03, atol=2e-03
            )

        # Compare with alternative method of MSF computation
        assert np.allclose(test_fluc_nomw, msqf_alternative)


@pytest.mark.parametrize("file_path", glob.glob(join(data_dir(), "*.pdb")))
def test_prs(file_path):
    """
    Compare perturbation response scanning (PRS)
    results with those obtained with ProDy.
    """
    test_anm = prepare_springcraft_anm(file_path, cutoff=13)

    strucname = basename(file_path).split(".")[0]

    test_prs, test_eff, test_sens = test_anm.prs_effector_sensor()
    ref_prs, ref_eff, ref_sens = [
        np.genfromtxt(
            join(data_dir(), f"prody_anm_13_ang_cutoff_{prs_type}_{strucname}.csv.gz"),
            delimiter=",",
        )
        for prs_type in ["prs_mat", "prs_eff", "prs_sens"]
    ]

    assert np.allclose(test_prs, ref_prs)
    assert np.allclose(test_eff, ref_eff)
    assert np.allclose(test_sens, ref_sens)


def test_modify_contact_pair():
    """
    Tests whether permutations to the `hessian` matrix are
    performed correctly and the resulting permutations to the
    `covariance` matrix are correct.
    """
    pdb_file = pdb.PDBFile.read(join(data_dir(), "1l2y.pdb"))
    atoms = pdb.get_structure(pdb_file, model=1)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
    ff = springcraft.InvariantForceField(7.9)
    test_anm = springcraft.ANM(ca, ff)

    # error responses
    assert test_anm._hessian is None
    with pytest.raises(AttributeError, match="Interaction matrix must exist."):
        test_anm.modify_contact(1, 2, 1)
    test_anm.hessian
    with pytest.raises(IndexError):
        test_anm.modify_contact(-1, 2, 1)
    with pytest.raises(IndexError):
        test_anm.modify_contact(20, 2, 1)
    with pytest.raises(IndexError):
        test_anm.modify_contact(1, -2, 1)
    with pytest.raises(IndexError):
        test_anm.modify_contact(1, 20, 1)
    with pytest.raises(IndexError):
        test_anm.modify_contact(1, 1, 1)
    with pytest.raises(ValueError):
        test_anm.modify_contact(1, 2, 0)  # zero delta
    with pytest.raises(ValueError):
        test_anm.modify_contact(1, 2, True)  # turn on contact that is already on
    with pytest.raises(ValueError):
        test_anm.modify_contact(1, 19, False)  # turn off contact that is already off

    test_anm.covariance
    assert test_anm._covariance is not None

    # arbitrary delta with rank unchanged
    test_anm.modify_contact(4, 8, 2)
    ref_ff = ModifiedForceField(ff, len(ca), 4, 8, 2)
    ref_anm = springcraft.ANM(ca, ref_ff)
    assert np.allclose(test_anm.hessian, ref_anm.hessian)
    assert np.allclose(test_anm.covariance, ref_anm.covariance)

    # rank unchanged
    test_anm.modify_contact(4, 8, False)
    ref_ff = ModifiedForceField(ff, len(ca), 4, 8, -1)
    ref_anm = springcraft.ANM(ca, ref_ff)
    assert np.allclose(test_anm.hessian, ref_anm.hessian)
    assert np.allclose(test_anm.covariance, ref_anm.covariance)

    # rank decrease
    test_anm.modify_contact(5, 8, False)
    test_anm.modify_contact(6, 8, False)
    test_anm.modify_contact(7, 8, False)
    test_anm.modify_contact(9, 8, False)
    test_anm.modify_contact(10, 8, False)
    test_anm.modify_contact(13, 8, False)
    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [4, 5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1, -1],
    )
    ref_anm = springcraft.ANM(ca, ref_ff)
    assert np.allclose(test_anm.hessian, ref_anm.hessian)
    assert np.allclose(test_anm.covariance, ref_anm.covariance)

    # rank increase
    test_anm.modify_contact(4, 8, True)
    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1],
    )
    ref_anm = springcraft.ANM(ca, ref_ff)
    assert np.allclose(test_anm.hessian, ref_anm.hessian)
    assert np.allclose(test_anm.covariance, ref_anm.covariance)
