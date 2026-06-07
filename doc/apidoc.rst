API Reference
=============

.. currentmodule:: springcraft


.. contents::
   :depth: 3


Network Models
--------------

.. autoclass:: GNM

   Normal Mode Analysis
   ~~~~~~~~~~~~~~~~~~~~
   .. automethod:: frequencies
   .. automethod:: mean_square_fluctuation
   .. automethod:: bfactor
   .. automethod:: dcc

   Model Modifications
   ~~~~~~~~~~~~~~~~~~~
   .. automethod:: modify_contact
   .. automethod:: modify_atom

   Normal Mode Anaysis Update
   ~~~~~~~~~~~~~~~~~~~~~~~~~~
   .. automethod:: mean_square_fluctuation_update
   .. automethod:: bfactor_update
   .. automethod:: dcc_update

|

.. autoclass:: ANM

   Normal Mode Analysis
   ~~~~~~~~~~~~~~~~~~~~
   .. automethod:: frequencies
   .. automethod:: mean_square_fluctuation
   .. automethod:: bfactor
   .. automethod:: dcc

   Model Modifications
   ~~~~~~~~~~~~~~~~~~~
   .. automethod:: modify_contact
   .. automethod:: modify_atom

   Normal Mode Anaysis Update
   ~~~~~~~~~~~~~~~~~~~~~~~~~~
   .. automethod:: mean_square_fluctuation_update
   .. automethod:: bfactor_update
   .. automethod:: dcc_update

|

.. autoclass:: springcraft.enm.ENM

   .. automethod:: eigen

|

.. autoclass:: springcraft.enm_update.ENMUpdate

   .. automethod:: prepare_update
   .. automethod:: interactions_update
   .. automethod:: covariance_update


Normal Mode Analysis
--------------------

.. autofunction:: springcraft.nma.frequencies

.. autofunction:: springcraft.nma.mean_square_fluctuation

.. autofunction:: springcraft.nma.bfactor

.. autofunction:: springcraft.nma.dcc


Normal Mode Analysis Update
---------------------------

.. autofunction:: springcraft.nma_update.mean_square_fluctuation_update

.. autofunction:: springcraft.nma_update.bfactor_update

.. autofunction:: springcraft.nma_update.dcc_update


Force Fields
------------

.. autoclass:: ForceField
   :members:

|

.. autoclass:: PatchedForceField

|

.. autoclass:: InvariantForceField

|

.. autoclass:: HinsenForceField

|

.. autoclass:: ParameterFreeForceField

|

.. autoclass:: TabulatedForceField
   :members: s_enm_10, s_enm_13, d_enm, sd_enm, e_anm, e_anm_mj, e_anm_ke

|

Miscellaneous
-------------

.. autofunction:: compute_kirchhoff

.. autofunction:: compute_hessian
