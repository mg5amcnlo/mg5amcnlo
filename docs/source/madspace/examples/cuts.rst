Defining fiducial cuts
========================

Phase-space mappings can be restricted to a fiducial region before any sampling happens. This
example builds the default generation cuts MadGraph7 applies to jets and hands them to a
:py:class:`PhaseSpaceMapping <madspace.PhaseSpaceMapping>` for the ``g g > t t~ g`` process
from the previous example. Cutting the sampled region, rather than discarding events after the
fact, is also what makes the integrator examples below converge.

Building the cuts
-------------------

A cut is a :py:class:`CutItem <madspace.CutItem>`: an :py:class:`Observable
<madspace.Observable>` together with the range it must fall in. ``pids`` lists the PDG id of
every external particle, beams first, matching the order ``masses`` uses in
:py:class:`PhaseSpaceMapping <madspace.PhaseSpaceMapping>`. ``[O.jet_pids]`` selects the
outgoing particles that count as jets, here the single final-state gluon:

.. code-block:: python

    import numpy as np
    import madspace as ms

    E_CM = 13000.0
    masses = [0.0, 0.0, 173.0, 173.0, 0.0]   # g g -> t t~ g
    pids = [21, 21, 6, -6, 21]               # beams first, then outgoing

    O = ms.Observable
    cuts = ms.Cuts([
        ms.CutItem(O(pids, O.obs_pt, [O.jet_pids]), min=20.0),
        ms.CutItem(O(pids, O.obs_eta_abs, [O.jet_pids]), max=5.0),
        ms.CutItem(O(pids, O.obs_delta_r, [O.jet_pids]), min=0.4),
    ])

These three bounds, a minimum transverse momentum, a maximum pseudorapidity and a minimum
pairwise separation, are MadGraph7's default jet cuts. With a single jet in the final state,
the separation cut has nothing to act on. It still does no harm to include it, and the same
:py:class:`Cuts <madspace.Cuts>` object works unchanged for a process with more jets.

Building the mapping with cuts
---------------------------------

:py:class:`PhaseSpaceMapping <madspace.PhaseSpaceMapping>` takes the cuts through its ``cuts``
argument:

.. code-block:: python

    mapping = ms.PhaseSpaceMapping(masses, E_CM, mode="rambo", cuts=cuts)
    rng = np.random.default_rng(7)
    r = rng.random((20000, mapping.random_dim()))
    momenta, x1, x2, det = mapping.map_forward([r])
    print(f"physical fraction: {(det != 0).mean():.4f}")

A sampled point outside the cuts comes back with ``det == 0``, the same convention Chili uses
for points outside its support. The fraction of points that pass depends on the mapping.
``"rambo"`` samples the full phase space and lets these cuts reject part of it afterwards::

    physical fraction: 0.9743
