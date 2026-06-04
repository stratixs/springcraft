import numpy as np
import pytest

import springcraft
from tests.util import ModifiedForceField, load_protein_structure


def test_modify_contact():
    """
    Tests whether permutations to the `kirchhoff` matrix are
    performed correctly and the resulting permutations to the
    `covariance` matrix are correct.
    """
    ca = load_protein_structure("1l2y")
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


def test_modify_atom():
    ca = load_protein_structure("1l2y")
    ff = springcraft.TabulatedForceField.d_enm(ca)
    test_gnm = springcraft.GNM(ca, ff)

    # error responses
    assert test_gnm._kirchhoff is None
    with pytest.raises(AttributeError, match="Interaction matrix must exist."):
        test_gnm.modify_atom(8, False)
    test_gnm.kirchhoff
    with pytest.raises(IndexError):
        test_gnm.modify_atom(-1, False)
    with pytest.raises(IndexError):
        test_gnm.modify_atom(20, False)
    with pytest.raises(ValueError):
        test_gnm.modify_atom(8, ca[8])  # no change in atom

    test_gnm.covariance
    assert test_gnm._covariance is not None

    # turn off
    test_gnm.modify_atom(8, False)
    print(test_gnm.kirchhoff)
    ref_ff = springcraft.PatchedForceField(ff, contact_shutdown=[8])
    ref_gnm = springcraft.GNM(ca, ref_ff)
    assert np.allclose(test_gnm.kirchhoff, ref_gnm.kirchhoff)
    assert np.allclose(test_gnm.covariance, ref_gnm.covariance)

    # turn on
    test_gnm.modify_atom(8, True)
    ref_gnm = springcraft.GNM(ca, ff)
    assert np.allclose(test_gnm.kirchhoff, ref_gnm.kirchhoff)
    assert np.allclose(test_gnm.covariance, ref_gnm.covariance)

    # change amino acid type
    ca.res_name[8] = "LEU"
    test_gnm.modify_atom(8, ca[8])
    ref_ff = springcraft.TabulatedForceField.d_enm(ca)
    ref_gnm = springcraft.GNM(ca, ref_ff)
    assert np.allclose(test_gnm.kirchhoff, ref_gnm.kirchhoff)
    assert np.allclose(test_gnm.covariance, ref_gnm.covariance)


def test_freq_chng():
    ca = load_protein_structure("1l2y")
    ff = springcraft.InvariantForceField(7.0)

    # positive delta
    test_gnm = springcraft.GNM(ca, ff)
    test_gnm.eigen()
    freq = test_gnm.frequencies_chng(6, 8, 3)

    ref_ff = ModifiedForceField(ff, len(ca), [6], [8], [3])
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_freq = ref_gnm.frequencies()
    assert np.allclose(freq, ref_freq)

    # negative delta
    test_gnm = springcraft.GNM(ca, ff)
    test_gnm.eigen()
    freq = test_gnm.frequencies_chng(6, 8, -3)

    ref_ff = ModifiedForceField(ff, len(ca), [6], [8], [-3])
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_freq = ref_gnm.frequencies()
    assert np.allclose(freq, ref_freq)

    # rank decrease
    test_gnm = springcraft.GNM(ca, ff)
    test_gnm.kirchhoff
    for i in [5, 6, 7, 9, 10, 13]:
        test_gnm.modify_contact(i, 8, False)
    test_gnm.eigen()
    freq = test_gnm.frequencies_chng(4, 8, False)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [4, 5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1, -1],
    )
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_freq = ref_gnm.frequencies()
    assert np.allclose(freq, ref_freq)

    # rank increase
    test_gnm.modify_contact(4, 8, False)
    test_gnm.eigen()
    freq = test_gnm.frequencies_chng(4, 8, True)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1],
    )
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_freq = ref_gnm.frequencies()
    assert np.allclose(freq, ref_freq)


def test_msqf_chng():
    ca = load_protein_structure("1l2y")
    ff = springcraft.InvariantForceField(7.0)

    # no rank change
    test_gnm = springcraft.GNM(ca, ff)
    test_gnm.kirchhoff
    test_gnm.covariance
    msqf = test_gnm.mean_square_fluctuation_chng(6, 8, 2)

    ref_ff = ModifiedForceField(ff, len(ca), [6], [8], [2])
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_msqf = ref_gnm.mean_square_fluctuation()
    assert np.allclose(msqf, ref_msqf)

    # rank decrease
    for i in [5, 6, 7, 9, 10, 13]:
        test_gnm.modify_contact(i, 8, False)
    msqf = test_gnm.mean_square_fluctuation_chng(4, 8, False)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [4, 5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1, -1],
    )
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_msqf = ref_gnm.mean_square_fluctuation()
    assert np.allclose(msqf, ref_msqf)

    # rank increase
    test_gnm.modify_contact(4, 8, False)
    msqf = test_gnm.mean_square_fluctuation_chng(4, 8, True)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1],
    )
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_msqf = ref_gnm.mean_square_fluctuation()
    assert np.allclose(msqf, ref_msqf)

    # temp scaling
    msqf = test_gnm.mean_square_fluctuation_chng(4, 8, True, tem=300)
    ref_msqf = ref_gnm.mean_square_fluctuation(tem=300)
    assert np.allclose(msqf, ref_msqf)


def test_msqf_subset_chng():
    ca = load_protein_structure("1l2y")
    ff = springcraft.InvariantForceField(7.0)

    # no rank change
    subset = np.array([5, 14, 18])
    test_gnm = springcraft.GNM(ca, ff)
    test_gnm.kirchhoff
    msqf = test_gnm.mean_square_fluctuation_chng(6, 8, 2, subset)

    ref_ff = ModifiedForceField(ff, len(ca), [6], [8], [2])
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_msqf = ref_gnm.mean_square_fluctuation(subset)
    assert np.allclose(msqf, ref_msqf)

    # rank decrease
    for i in [5, 6, 7, 9, 10, 13]:
        test_gnm.modify_contact(i, 8, False)
    msqf = test_gnm.mean_square_fluctuation_chng(4, 8, False, subset)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [4, 5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1, -1],
    )
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_msqf = ref_gnm.mean_square_fluctuation(subset)
    assert np.allclose(msqf, ref_msqf)

    # rank increase
    test_gnm.modify_contact(4, 8, False)
    msqf = test_gnm.mean_square_fluctuation_chng(4, 8, True, subset)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1],
    )
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_msqf = ref_gnm.mean_square_fluctuation(subset)
    assert np.allclose(msqf, ref_msqf)

    # temp scaling
    msqf = test_gnm.mean_square_fluctuation_chng(4, 8, True, subset, tem=300)
    ref_msqf = ref_gnm.mean_square_fluctuation(subset, tem=300)
    assert np.allclose(msqf, ref_msqf)


def test_bfactor_chng():
    ca = load_protein_structure("1l2y")
    ff = springcraft.InvariantForceField(7.0)

    # no rank change
    test_gnm = springcraft.GNM(ca, ff)
    test_gnm.kirchhoff
    test_gnm.covariance
    bfactor = test_gnm.bfactor_chng(6, 8, 2)

    ref_ff = ModifiedForceField(ff, len(ca), [6], [8], [2])
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_bfactor = ref_gnm.bfactor()
    assert np.allclose(bfactor, ref_bfactor)

    # temp scaling
    test_gnm = springcraft.GNM(ca, ff)
    test_gnm.kirchhoff
    test_gnm.covariance
    bfactor = test_gnm.bfactor_chng(6, 8, 2, tem=300)

    ref_bfactor = ref_gnm.bfactor(tem=300)
    assert np.allclose(bfactor, ref_bfactor)


def test_bfactor_subset_chng():
    ca = load_protein_structure("1l2y")
    ff = springcraft.InvariantForceField(7.0)

    # no rank change
    subset = np.array([5, 14, 18])
    test_gnm = springcraft.GNM(ca, ff)
    test_gnm.kirchhoff
    bfactor = test_gnm.bfactor_chng(6, 8, 2, subset)

    ref_ff = ModifiedForceField(ff, len(ca), [6], [8], [2])
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_bfactor = ref_gnm.bfactor(subset)
    assert np.allclose(bfactor, ref_bfactor)


def test_dcc_chng():
    ca = load_protein_structure("1l2y")
    ff = springcraft.InvariantForceField(7.0)

    # no rank change
    test_gnm = springcraft.GNM(ca, ff)
    test_gnm.kirchhoff
    test_gnm.covariance
    dcc = test_gnm.dcc_chng(6, 8, 2)

    ref_ff = ModifiedForceField(ff, len(ca), [6], [8], [2])
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_dcc = ref_gnm.dcc()
    assert np.allclose(dcc, ref_dcc)

    # rank decrease
    for i in [5, 6, 7, 9, 10, 13]:
        test_gnm.modify_contact(i, 8, False)
    dcc = test_gnm.dcc_chng(4, 8, False, norm=False)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [4, 5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1, -1],
    )
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_dcc = ref_gnm.dcc(norm=False)
    assert np.allclose(dcc, ref_dcc)

    # rank increase
    test_gnm.modify_contact(4, 8, False)
    dcc = test_gnm.dcc_chng(4, 8, True)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1],
    )
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_dcc = ref_gnm.dcc()
    assert np.allclose(dcc, ref_dcc)

    # temp scaling
    dcc = test_gnm.dcc_chng(4, 8, True, tem=300)
    ref_dcc = ref_gnm.dcc(tem=300)
    assert np.allclose(dcc, ref_dcc)


def test_dcc_subset_chng():
    ca = load_protein_structure("1l2y")
    ff = springcraft.InvariantForceField(7.0)

    # no rank change
    subset = np.array([5, 14, 18])
    test_gnm = springcraft.GNM(ca, ff)
    test_gnm.kirchhoff
    dcc = test_gnm.dcc_chng(6, 8, 2, subset)

    ref_ff = ModifiedForceField(ff, len(ca), [6], [8], [2])
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_dcc = ref_gnm.dcc(subset)
    assert np.allclose(dcc, ref_dcc)

    # rank decrease
    for i in [5, 6, 7, 9, 10, 13]:
        test_gnm.modify_contact(i, 8, False)
    dcc = test_gnm.dcc_chng(4, 8, False, subset, norm=False)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [4, 5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1, -1],
    )
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_dcc = ref_gnm.dcc(subset, norm=False)
    assert np.allclose(dcc, ref_dcc)

    # rank increase
    test_gnm.modify_contact(4, 8, False)
    dcc = test_gnm.dcc_chng(4, 8, True, subset)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1],
    )
    ref_gnm = springcraft.GNM(ca, ref_ff)
    ref_gnm.kirchhoff
    ref_dcc = ref_gnm.dcc(subset)
    assert np.allclose(dcc, ref_dcc)

    # temp scaling
    dcc = test_gnm.dcc_chng(4, 8, True, subset, tem=300)
    ref_dcc = ref_gnm.dcc(subset, tem=300)
    assert np.allclose(dcc, ref_dcc)
