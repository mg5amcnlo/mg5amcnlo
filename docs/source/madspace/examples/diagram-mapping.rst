Building a mapping for a Feynman diagram
==========================================

The previous examples built a mapping from a flat list of masses. This example instead
follows one Feynman diagram, using :py:class:`Diagram <madspace.Diagram>` and
:py:class:`Topology <madspace.Topology>`. The diagram is a single-quark t-channel exchange
feeding a resonant W boson, ``u d~ > mu+ vm g``. It has one t-channel propagator and one
resonant s-channel propagator, so the mapping can sample the W resonance directly instead of
missing it.

Building the diagram
-----------------------

:py:class:`Diagram <madspace.Diagram>` takes the incoming and outgoing masses, a list of
:py:class:`Propagator <madspace.Propagator>` and the vertices connecting them. A vertex is a
triple of line labels: ``i`` for an incoming line, ``o`` for an outgoing one and ``p`` for a
propagator, each followed by its 0-based index:

.. code-block:: python

    import numpy as np
    import madspace as ms

    M_W, W_W = 80.379, 2.085  # W mass and width, in GeV

    diagram = ms.Diagram(
        [0.0, 0.0],            # incoming u, d~
        [0.0, 0.0, 0.0],       # outgoing g, mu+, vm
        [
            ms.Propagator(mass=0.0, width=0.0, pdg_id=2),   # t-channel u-quark
            ms.Propagator(mass=M_W, width=W_W, pdg_id=24),  # s-channel W+
        ],
        [["i0", "o0", "p0"], ["p0", "i1", "p1"], ["p1", "o1", "o2"]],
    )

The first vertex is the incoming ``u`` quark radiating the gluon and turning into the
t-channel propagator. The second vertex absorbs the ``d~`` and produces the W. The third
vertex is the W decay.

Inspecting the topology
--------------------------

:py:class:`Topology <madspace.Topology>` turns a diagram into one integration channel. It can
be printed directly, which shows the t-channel propagator and the s-channel decay tree it
builds:

.. code-block:: python

    topology = ms.Topology(diagram)
    print(topology)

::

    ├── incoming: index=0, mass=0
    ├── incoming: index=1, mass=0
    └── t-channel: (order=0, mass=0, width=0)
        ├── outgoing: index=0, mass=0
        └── decay: order=0, mass=80.379, width=2.085, e_min=0, e_max=0, pdg_id=24, on_shell=true, on_shell_boundary=false
            ├── outgoing: index=1, mass=0
            └── outgoing: index=2, mass=0

Sampling the resonance
-------------------------

Building a :py:class:`PhaseSpaceMapping <madspace.PhaseSpaceMapping>` from the topology
samples the W invariant mass around its pole, using the propagator's mass and width:

.. code-block:: python

    mapping = ms.PhaseSpaceMapping(topology, 13000.0)
    rng = np.random.default_rng(0)
    r = rng.random((100000, mapping.random_dim()))
    momenta, x1, x2, det = mapping.map_forward([r])

    p_mu, p_vm = momenta[:, 3], momenta[:, 4]
    p_sum = p_mu + p_vm
    m_inv = np.sqrt(np.abs(p_sum[:, 0] ** 2 - np.sum(p_sum[:, 1:] ** 2, axis=1)))
    print(f"fraction within 5 GeV of the W mass: {(np.abs(m_inv - M_W) < 5).mean():.3f}")

Most of the sample lands close to the pole, even though the window is narrow compared to the
full available energy::

    fraction within 5 GeV of the W mass: 0.877

For comparison, a flat mapping built from the same external masses, with ``mode="rambo"``,
has no notion of this resonance and only finds a tiny fraction of its events there:

.. code-block:: python

    flat = ms.PhaseSpaceMapping([0.0, 0.0, 0.0, 0.0, 0.0], 13000.0, mode="rambo")
    momenta_flat, _, _, _ = flat.map_forward([r])
    p_sum_flat = momenta_flat[:, 3] + momenta_flat[:, 4]
    m_inv_flat = np.sqrt(np.abs(p_sum_flat[:, 0] ** 2 - np.sum(p_sum_flat[:, 1:] ** 2, axis=1)))
    print(f"fraction within 5 GeV of the W mass: {(np.abs(m_inv_flat - M_W) < 5).mean():.3f}")

::

    fraction within 5 GeV of the W mass: 0.009
