Constructing and calling a phase-space mapping
================================================

This example builds a phase-space mapping for the process ``g g > t t~ g`` and uses it to
draw a batch of momenta from random numbers. It uses :py:class:`PhaseSpaceMapping
<madspace.PhaseSpaceMapping>` with a flat list of external masses. This builds a mapping
without needing a :py:class:`Topology <madspace.Topology>` first; the next example covers
that case. Everything here is plain NumPy. MadSpace does not require NumPy specifically. It
accepts whichever array library the input arrays belong to.

Building the mapping
---------------------

We pick a collision energy and the external masses, beams first, then the outgoing particles
in the order ``t``, ``t~``, ``g``:

.. code-block:: python

    import numpy as np
    import madspace as ms

    E_CM = 13000.0
    masses = [0.0, 0.0, 173.0, 173.0, 0.0]  # g g -> t t~ g

    mapping = ms.PhaseSpaceMapping(masses, E_CM, mode="rambo")
    print(f"random numbers per event: {mapping.random_dim()}")

``mode`` selects how the mapping handles the two incoming (t-channel) legs. ``"rambo"`` uses
the :py:class:`FastRamboMapping <madspace.FastRamboMapping>` building block. It does not
follow any particular Feynman diagram. The output is::

    random numbers per event: 7

Sampling momenta
-----------------

:py:meth:`map_forward <madspace.Mapping.map_forward>` turns a batch of uniform random numbers
into momenta. It returns a namedtuple. For this mapping its fields are ``momenta``, ``x1``,
``x2`` (the parton momentum fractions) and ``det``, the Jacobian of the transformation:

.. code-block:: python

    rng = np.random.default_rng(1234)
    r = rng.random((10000, mapping.random_dim()))
    momenta, x1, x2, det = mapping.map_forward([r])
    print(f"momenta: {momenta.shape}, x1: {x1.shape}, det: {det.shape}")
    np.set_printoptions(precision=1, suppress=True)
    print("first event, (E, px, py, pz) per particle:")
    print(momenta[0])

``momenta`` holds one four-vector per external particle for every event in the batch. The
order matches ``masses``: beams first, then the outgoing particles::

    momenta: (10000, 5, 4), x1: (10000,), det: (10000,)
    first event, (E, px, py, pz) per particle:
    [[ 6281.4     0.      0.   6281.4]
     [ 6147.4     0.      0.  -6147.4]
     [ 5163.9   199.8 -2715.4  4384.4]
     [ 4392.7  -199.6  -153.8 -4382. ]
     [ 2872.2    -0.3  2869.1   131.6]]

Checking the output
---------------------

The sampled momenta reproduce the requested external masses. They also conserve momentum, up
to floating-point precision:

.. code-block:: python

    m = np.sqrt(np.abs(momenta[..., 0] ** 2 - np.sum(momenta[..., 1:] ** 2, axis=-1)))
    print("masses of the first event:", m[0])

    p_in = momenta[:, :2].sum(axis=1)
    p_out = momenta[:, 2:].sum(axis=1)
    print("largest momentum-conservation violation:", np.abs(p_out - p_in).max())

::

    masses of the first event: [  0.   0. 173. 173.   0.]
    largest momentum-conservation violation: 9.458744898438454e-11

Inverting the mapping
-----------------------

Every :py:class:`Mapping <madspace.Mapping>` is invertible. :py:meth:`map_inverse
<madspace.Mapping.map_inverse>` recovers the random numbers from the momenta. The two
Jacobians are reciprocal:

.. code-block:: python

    r_inv, det_inv = mapping.map_inverse([momenta, x1, x2])
    print("largest deviation of the recovered random numbers:", np.abs(r_inv - r).max())
    print("largest deviation of det * det_inv from 1:", np.abs(det * det_inv - 1).max())

::

    largest deviation of the recovered random numbers: 3.0175417720101905e-11
    largest deviation of det * det_inv from 1: 1.37008404621497e-10

Switching to Chili
--------------------

Passing ``mode="chili"`` instead builds a :py:class:`ChiliMapping <madspace.ChiliMapping>`.
It samples directly in observable variables such as transverse momenta and rapidities,
rather than through recursive two-body decays. This suits processes with many jets. The cost
is a mapping that misses part of phase space: points outside its support come back with
``det == 0`` and must be discarded.

.. code-block:: python

    chili = ms.PhaseSpaceMapping(masses, E_CM, mode="chili")
    r_chili = rng.random((10000, chili.random_dim()))
    momenta_chili, _, _, det_chili = chili.map_forward([r_chili])
    physical = det_chili != 0.0
    print(f"fraction of physical Chili points: {physical.mean():.3f}")

    pt_rambo = np.hypot(momenta[:, 2, 1], momenta[:, 2, 2])
    pt_chili = np.hypot(momenta_chili[physical, 2, 1], momenta_chili[physical, 2, 2])
    print(f"mean top pT: rambo = {pt_rambo.mean():.0f} GeV, chili = {pt_chili.mean():.0f} GeV")

Chili concentrates its samples at lower transverse momentum, closer to where a physical cross
section peaks::

    fraction of physical Chili points: 0.881
    mean top pT: rambo = 1272 GeV, chili = 634 GeV
