Springcraft
===========

*Springcraft* is a *Biotite* extension package, that allows the analysis
of `AtomArray` objects via *Elastic Network Models* (ENMs).
An ENM can be thought of as a system that connects residues via springs:
Interaction of nearby residues is governed by a harmonic potential, with the
native (input) conformation representing the energy minimum.
Normal mode analysis allows the researcher to investigate global
functional movements of a protein in a fast coarse-grained manner.

.. note::

  *Springcraft* is still in alpha stage.
  Although most implemented functionalities should already work as
  expected, some features are not well tested, yet.

Installation
------------

*Springcraft* can be installed via

.. code-block:: console

   $ pip install springcraft

or

.. code-block:: console

   $ conda install -c conda-forge springcraft

You can also install *Springcraft* from source.
The package uses `Poetry <https://python-poetry.org/>`_ for building
distributions.
Via :pep:`517` it is possible to install the package from local source code
via *pip*:

.. code-block:: console

   $ git clone https://github.com/biotite-dev/springcraft.git
   $ pip install ./springcraft

Development
-----------

For development and testing first create a conda enviroment from the
`environment.yml` and than let poetry install the required dependencies.

.. code-block:: console

   $ conda env create -f environment.yml
   $ conda activate springcraft-dev
   $ poetry config virtualenvs.create false
   $ poetry install

Run tests with `pytest` like this

.. code-block:: console

   $ pytest tests/
   $ pytest tests/ --cov=src --cov-report=html --cov-branch  # to run with coverage

If you additionally want to render the documentation use the environment in
`environment-docs.yml` and follow the same steps as above. To generate the
documentation html with sphinx run

.. code-block:: console

   $ sphinx-build -E -a -b html doc doc/build/html

Scripts to generate reference files for tests are stored in tests/data;
a separate environment to rerun these locally can be found in `environment-test-data.yml`.
BioPhysConnectoR has to be installed separately.

Example
=======

.. code-block:: python

   import numpy as np
   import biotite.structure.io.pdbx as pdbx
   import springcraft


   pdbx_file = pdbx.PDBxFile.read("path/to/1l2y.cif")
   atoms = pdbx.get_structure(pdbx_file, model=1)
   ca = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]
   ff = springcraft.InvariantForceField(cutoff_distance=7.0)
   gnm = springcraft.GNM(ca, ff)
   kirchhoff = gnm.kirchhoff

   np.set_printoptions(linewidth=100)
   print(kirchhoff)

Output:

.. code-block:: none

   [[ 4. -1. -1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [-1.  6. -1. -1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1.  0.]
    [-1. -1.  7. -1. -1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1.  0.]
    [-1. -1. -1.  7. -1. -1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [-1. -1. -1. -1.  8. -1. -1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 0. -1. -1. -1. -1.  9. -1. -1. -1.  0. -1.  0.  0.  0.  0.  0.  0. -1.  0.  0.]
    [ 0.  0. -1. -1. -1. -1.  8. -1. -1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 0.  0.  0. -1. -1. -1. -1.  7. -1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 0.  0.  0.  0. -1. -1. -1. -1.  7. -1. -1.  0.  0. -1.  0.  0.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0. -1. -1. -1.  7. -1. -1. -1. -1.  0.  0.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0. -1. -1. -1. -1. -1.  8. -1. -1. -1.  0.  0.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0. -1. -1.  7. -1. -1. -1. -1. -1.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0. -1. -1. -1.  5. -1. -1.  0.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0. -1. -1. -1. -1. -1.  7. -1. -1.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1. -1. -1.  4. -1.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1.  0. -1. -1.  5. -1. -1.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1.  0.  0.  0. -1.  4. -1. -1.  0.]
    [ 0.  0.  0.  0.  0. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1. -1.  5. -1. -1.]
    [ 0. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1. -1.  5. -1.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1. -1.  2.]]
