import numpy as np
import pytest

import springcraft
from tests.util import ModifiedForceField, load_protein_structure


def test_modify_contact():
    """
    Tests whether permutations to the `hessian` matrix are
    performed correctly and the resulting permutations to the
    `covariance` matrix are correct.
    """
    ca = load_protein_structure("1l2y")
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


def test_modify_atom():
    ca = load_protein_structure("1l2y")
    ff = springcraft.TabulatedForceField.d_enm(ca)
    test_anm = springcraft.ANM(ca, ff)

    # error responses
    assert test_anm._hessian is None
    with pytest.raises(AttributeError, match="Interaction matrix must exist."):
        test_anm.modify_atom(8, False)
    test_anm.hessian
    with pytest.raises(IndexError):
        test_anm.modify_atom(-1, False)
    with pytest.raises(IndexError):
        test_anm.modify_atom(20, False)
    with pytest.raises(ValueError):
        test_anm.modify_atom(8, ca[8])  # no change in atom

    test_anm.covariance
    assert test_anm._covariance is not None

    # turn off
    test_anm.modify_atom(8, False)
    ref_ff = springcraft.PatchedForceField(ff, contact_shutdown=[8])
    ref_anm = springcraft.ANM(ca, ref_ff)
    assert np.allclose(test_anm.hessian, ref_anm.hessian)
    assert np.allclose(test_anm.covariance, ref_anm.covariance, atol=1e-7)

    # turn on
    test_anm.modify_atom(8, True)
    ref_anm = springcraft.ANM(ca, ff)
    assert np.allclose(test_anm.hessian, ref_anm.hessian)
    assert np.allclose(test_anm.covariance, ref_anm.covariance, atol=1e-6)

    # change amino acid type
    ff = springcraft.TabulatedForceField.d_enm(ca)
    test_anm = springcraft.ANM(ca, ff)
    test_anm.hessian
    test_anm.covariance

    ca.res_name[8] = "LEU"
    test_anm.modify_atom(8, ca[8])
    ref_ff = springcraft.TabulatedForceField.d_enm(ca)
    ref_anm = springcraft.ANM(ca, ref_ff)
    assert np.allclose(test_anm.hessian, ref_anm.hessian)
    assert np.allclose(test_anm.covariance, ref_anm.covariance, atol=1e-7)


def test_freq_update():
    ca = load_protein_structure("1l2y")
    ff = springcraft.InvariantForceField(7.0)

    # positive delta
    test_anm = springcraft.ANM(ca, ff)
    test_anm.eigen()
    freq = test_anm.frequencies_update(6, 8, 3)

    ref_ff = ModifiedForceField(ff, len(ca), [6], [8], [3])
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_freq = ref_anm.frequencies()
    assert np.allclose(freq, ref_freq)

    # negative delta
    test_anm = springcraft.ANM(ca, ff)
    test_anm.eigen()
    freq = test_anm.frequencies_update(6, 8, -0.5)

    ref_ff = ModifiedForceField(ff, len(ca), [6], [8], [-0.5])
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_freq = ref_anm.frequencies()
    assert np.allclose(freq, ref_freq)

    # rank decrease
    test_anm = springcraft.ANM(ca, ff)
    test_anm.hessian
    for i in [5, 6, 7, 9, 10, 13]:
        test_anm.modify_contact(i, 8, False)
    test_anm.eigen()
    freq = test_anm.frequencies_update(4, 8, False)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [4, 5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1, -1],
    )
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_freq = ref_anm.frequencies()
    assert np.allclose(freq, ref_freq)

    # rank increase
    test_anm.modify_contact(4, 8, False)
    test_anm.eigen()
    freq = test_anm.frequencies_update(4, 8, True)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1],
    )
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_freq = ref_anm.frequencies()
    assert np.allclose(freq, ref_freq)


def test_msqf_update():
    ca = load_protein_structure("1l2y")
    ff = springcraft.InvariantForceField(7.0)

    # no rank change
    test_anm = springcraft.ANM(ca, ff)
    test_anm.hessian
    test_anm.covariance
    msqf = test_anm.mean_square_fluctuation_update(6, 8, 2)

    ref_ff = ModifiedForceField(ff, len(ca), [6], [8], [2])
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_msqf = ref_anm.mean_square_fluctuation()
    assert np.allclose(msqf, ref_msqf)

    # rank decrease
    for i in [5, 6, 7, 9, 10, 13]:
        test_anm.modify_contact(i, 8, False)
    msqf = test_anm.mean_square_fluctuation_update(4, 8, False)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [4, 5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1, -1],
    )
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_msqf = ref_anm.mean_square_fluctuation()
    assert np.allclose(msqf, ref_msqf)

    # rank increase
    test_anm.modify_contact(4, 8, False)
    msqf = test_anm.mean_square_fluctuation_update(4, 8, True)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1],
    )
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_msqf = ref_anm.mean_square_fluctuation()
    assert np.allclose(msqf, ref_msqf)

    # temp scaling
    msqf = test_anm.mean_square_fluctuation_update(4, 8, True, tem=300)
    ref_msqf = ref_anm.mean_square_fluctuation(tem=300)
    assert np.allclose(msqf, ref_msqf)


def test_msqf_subset_update():
    ca = load_protein_structure("1l2y")
    ff = springcraft.InvariantForceField(7.0)

    # no rank change
    subset = np.array([11, 43, 58])
    test_anm = springcraft.ANM(ca, ff)
    test_anm.hessian
    test_anm.covariance
    msqf = test_anm.mean_square_fluctuation_update(6, 8, 2, subset)

    ref_ff = ModifiedForceField(ff, len(ca), [6], [8], [2])
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_msqf = ref_anm.mean_square_fluctuation(subset)
    assert np.allclose(msqf, ref_msqf)

    # rank decrease
    for i in [5, 6, 7, 9, 10, 13]:
        test_anm.modify_contact(i, 8, False)
    msqf = test_anm.mean_square_fluctuation_update(4, 8, False, subset)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [4, 5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1, -1],
    )
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_msqf = ref_anm.mean_square_fluctuation(subset)
    assert np.allclose(msqf, ref_msqf)

    # rank increase
    test_anm.modify_contact(4, 8, False)
    msqf = test_anm.mean_square_fluctuation_update(4, 8, True, subset)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1],
    )
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_msqf = ref_anm.mean_square_fluctuation(subset)
    assert np.allclose(msqf, ref_msqf)

    # temp scaling
    msqf = test_anm.mean_square_fluctuation_update(4, 8, True, subset, tem=300)
    ref_msqf = ref_anm.mean_square_fluctuation(subset, tem=300)
    assert np.allclose(msqf, ref_msqf)


def test_bfactor_update():
    ca = load_protein_structure("1l2y")
    ff = springcraft.InvariantForceField(7.0)

    # no rank change
    test_anm = springcraft.ANM(ca, ff)
    test_anm.hessian
    test_anm.covariance
    bfactor = test_anm.bfactor_update(6, 8, 2)

    ref_ff = ModifiedForceField(ff, len(ca), [6], [8], [2])
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_bfactor = ref_anm.bfactor()
    assert np.allclose(bfactor, ref_bfactor)

    # temp scaling
    test_anm = springcraft.ANM(ca, ff)
    test_anm.hessian
    test_anm.covariance
    bfactor = test_anm.bfactor_update(6, 8, 2, tem=300)

    ref_bfactor = ref_anm.bfactor(tem=300)
    assert np.allclose(bfactor, ref_bfactor)


def test_bfactor_subset_update():
    ca = load_protein_structure("1l2y")
    ff = springcraft.InvariantForceField(7.0)

    # no rank change
    subset = np.array([11, 43, 58])
    test_anm = springcraft.ANM(ca, ff)
    test_anm.hessian
    test_anm.covariance
    bfactor = test_anm.bfactor_update(6, 8, 2, subset)

    ref_ff = ModifiedForceField(ff, len(ca), [6], [8], [2])
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_bfactor = ref_anm.bfactor(subset)
    assert np.allclose(bfactor, ref_bfactor)


def test_dcc_update():
    ca = load_protein_structure("1l2y")
    ff = springcraft.InvariantForceField(7.0)

    # no rank change
    test_anm = springcraft.ANM(ca, ff)
    test_anm.hessian
    test_anm.covariance
    dcc = test_anm.dcc_update(6, 8, 2)

    ref_ff = ModifiedForceField(ff, len(ca), [6], [8], [2])
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_dcc = ref_anm.dcc()
    assert np.allclose(dcc, ref_dcc)

    # rank decrease
    for i in [5, 6, 7, 9, 10, 13]:
        test_anm.modify_contact(i, 8, False)
    dcc = test_anm.dcc_update(4, 8, False, norm=False)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [4, 5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1, -1],
    )
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_dcc = ref_anm.dcc(norm=False)
    assert np.allclose(dcc, ref_dcc)

    # rank increase
    test_anm.modify_contact(4, 8, False)
    dcc = test_anm.dcc_update(4, 8, True)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1],
    )
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_dcc = ref_anm.dcc()
    assert np.allclose(dcc, ref_dcc)

    # temp scaling
    dcc = test_anm.dcc_update(4, 8, True, tem=300)
    ref_dcc = ref_anm.dcc(tem=300)
    assert np.allclose(dcc, ref_dcc)


def test_dcc_subset_update():
    ca = load_protein_structure("1l2y")
    ff = springcraft.InvariantForceField(7.0)

    # no rank change
    subset = np.array([11, 43, 58])
    test_anm = springcraft.ANM(ca, ff)
    test_anm.hessian
    test_anm.covariance
    dcc = test_anm.dcc_update(6, 8, 2, subset)

    ref_ff = ModifiedForceField(ff, len(ca), [6], [8], [2])
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_dcc = ref_anm.dcc(subset)
    assert np.allclose(dcc, ref_dcc)

    # rank decrease
    for i in [5, 6, 7, 9, 10, 13]:
        test_anm.modify_contact(i, 8, False)
    dcc = test_anm.dcc_update(4, 8, False, subset, norm=False)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [4, 5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1, -1],
    )
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_dcc = ref_anm.dcc(subset, norm=False)
    assert np.allclose(dcc, ref_dcc)

    # rank increase
    test_anm.modify_contact(4, 8, False)
    dcc = test_anm.dcc_update(4, 8, True, subset)

    ref_ff = ModifiedForceField(
        ff,
        len(ca),
        [5, 6, 7, 9, 10, 13],
        [8, 8, 8, 8, 8, 8],
        [-1, -1, -1, -1, -1, -1],
    )
    ref_anm = springcraft.ANM(ca, ref_ff)
    ref_anm.hessian
    ref_dcc = ref_anm.dcc(subset)
    assert np.allclose(dcc, ref_dcc)

    # temp scaling
    dcc = test_anm.dcc_update(4, 8, True, subset, tem=300)
    ref_dcc = ref_anm.dcc(subset, tem=300)
    assert np.allclose(dcc, ref_dcc)
