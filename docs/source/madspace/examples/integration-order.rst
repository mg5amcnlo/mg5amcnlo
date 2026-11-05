Multiple on-shell configurations and integration order
==========================================================

A diagram can have more than one resonance where not all of them can be on shell at once.
The process ``g g > h > Z Z > e+ e- mu+ mu-`` is a standard example: the Higgs mass, 125 GeV,
is below twice the Z mass, so the two Z propagators cannot both sit on their pole while the
Higgs also sits on its pole. :py:meth:`Topology.topologies <madspace.Topology.topologies>`
enumerates the on-shell configurations this forces, each one a separate integration channel.

Building the diagram
-----------------------

The Higgs propagator ``p0`` splits into the two Z propagators ``p1`` and ``p2``, which each
decay to a lepton pair:

.. code-block:: python

    import madspace as ms

    M_H, W_H = 125.0, 0.00407
    M_Z, W_Z = 91.188, 2.4952

    props = [
        ms.Propagator(mass=M_H, width=W_H, pdg_id=25),
        ms.Propagator(mass=M_Z, width=W_Z, pdg_id=23),
        ms.Propagator(mass=M_Z, width=W_Z, pdg_id=23),
    ]
    diagram = ms.Diagram(
        [0.0, 0.0], [0.0, 0.0, 0.0, 0.0], props,
        [["i0", "i1", "p0"], ["p0", "p1", "p2"], ["p1", "o0", "o1"], ["p2", "o2", "o3"]],
    )

Enumerating the channels
---------------------------

:py:meth:`Topology.topologies <madspace.Topology.topologies>` returns one channel per
combination of on-shell flags that is both kinematically possible and not already covered by
another channel:

.. code-block:: python

    topologies = ms.Topology.topologies(diagram)
    print(f"number of channels: {len(topologies)}")
    for t in topologies:
        on_shell = [d.on_shell for d in t.decays if d.mass != 0]
        print("on-shell flags (H, Z, Z):", on_shell)

Three channels come out: the Higgs on its pole with either Z on shell and the other one
floating, and the Higgs off its pole with both Z bosons on shell::

    number of channels: 3
    on-shell flags (H, Z, Z): [True, True, False]
    on-shell flags (H, Z, Z): [True, False, True]
    on-shell flags (H, Z, Z): [False, True, True]

Inspecting a channel
-----------------------

Printing a :py:class:`Topology <madspace.Topology>` shows its decay tree, with the
``on_shell`` flag and the ``order`` in which each invariant is sampled:

.. code-block:: python

    print(topologies[0])

::

    ├── incoming: index=0, mass=0
    ├── incoming: index=1, mass=0
    └── decay: order=0, mass=125, width=0.00407, e_min=0, e_max=0, pdg_id=25, on_shell=true, on_shell_boundary=false
        ├── decay: order=1, mass=91.188, width=2.4952, e_min=0, e_max=0, pdg_id=23, on_shell=true, on_shell_boundary=false
        │   ├── outgoing: index=0, mass=0
        │   └── outgoing: index=1, mass=0
        └── decay: order=2, mass=91.188, width=2.4952, e_min=0, e_max=0, pdg_id=23, on_shell=false, on_shell_boundary=false
            ├── outgoing: index=2, mass=0
            └── outgoing: index=3, mass=0

The on-shell Higgs is sampled first, then the on-shell Z, then the off-shell Z last, since its
window depends on what energy the first two leave behind.

Breaking a tie with integration_order
----------------------------------------

:py:class:`Propagator <madspace.Propagator>` also takes an ``integration_order`` argument. A
lower value is sampled first. It only matters between propagators the on-shell status alone
does not already order, such as the two Z bosons in the channel where both are on shell:

.. code-block:: python

    props[1] = ms.Propagator(mass=M_Z, width=W_Z, pdg_id=23, integration_order=1)
    props[2] = ms.Propagator(mass=M_Z, width=W_Z, pdg_id=23, integration_order=0)
    diagram_reordered = ms.Diagram(
        [0.0, 0.0], [0.0, 0.0, 0.0, 0.0], props,
        [["i0", "i1", "p0"], ["p0", "p1", "p2"], ["p1", "o0", "o1"], ["p2", "o2", "o3"]],
    )
    both_on_shell = [
        t for t in ms.Topology.topologies(diagram_reordered) if not t.decays[0].on_shell
    ][0]
    print(both_on_shell)

The second Z, now with the lower ``integration_order``, is sampled first::

    ├── incoming: index=0, mass=0
    ├── incoming: index=1, mass=0
    └── decay: order=1, mass=125, width=0.00407, e_min=0, e_max=0, pdg_id=25, on_shell=false, on_shell_boundary=false
        ├── decay: order=2, mass=91.188, width=2.4952, e_min=0, e_max=0, pdg_id=23, on_shell=true, on_shell_boundary=false
        │   ├── outgoing: index=0, mass=0
        │   └── outgoing: index=1, mass=0
        └── decay: order=0, mass=91.188, width=2.4952, e_min=0, e_max=0, pdg_id=23, on_shell=true, on_shell_boundary=false
            ├── outgoing: index=2, mass=0
            └── outgoing: index=3, mass=0
