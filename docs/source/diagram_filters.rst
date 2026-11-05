Diagram filters
===============

A diagram filter removes Feynman diagrams during the generation of a process. It is a
Python function that MadGraph7 calls on every generated diagram:

.. code-block:: python

    def remove_diag(diag, model):
        ...

It returns ``True`` to remove the diagram and ``False`` to keep it. ``model`` is the
model of the process.

.. warning::

    Removing diagrams usually breaks gauge invariance. The result is then not a physical
    prediction. Use filters to study subsets of diagrams.

Writing a filter plugin
-----------------------

A filter comes from a plugin. A plugin is a Python package in the ``PLUGIN`` directory
of MadGraph7, or in a ``MG5aMC_PLUGIN`` directory on the ``PYTHONPATH``. It provides
one filter, assigned to ``new_diagram_filter`` in its ``__init__.py``. The filter is
selected by the name of the plugin.

For example, ``PLUGIN/no_photon/__init__.py`` removes the diagrams with a photon
propagator:

.. code-block:: python

    # versions of MadGraph7 this plugin works with
    minimal_mg7_version = (0, 2, 0)
    maximal_mg7_version = (1000, 0, 0)
    latest_validated_mg7_version = (0, 2, 1)

    def remove_diag(diag, model):
        """Remove the diagrams with a photon propagator."""
        for vertex in diag['vertices'][:-1]:
            # The last leg of a vertex is the propagator to the next vertex.
            # The last vertex closes the diagram and has no propagator.
            if vertex['legs'][-1]['id'] == 22:
                return True
        return False

    new_diagram_filter = remove_diag

Every plugin gives the range of MadGraph7 versions it supports with
``minimal_mg7_version`` and ``maximal_mg7_version``. ``latest_validated_mg7_version``
is optional: MadGraph7 warns when it is more recent than that version. A plugin without
the version range is ignored. The ``minimal_mg5amcnlo_version`` and
``maximal_mg5amcnlo_version`` of MG5_aMC@NLO plugins refer to the MG5_aMC@NLO
numbering. A plugin with only those is used without version check, with a warning.

The filter is then used with:

.. code-block:: text

    import model sm
    generate u u~ > d d~ QED=2 --diagram_filter=no_photon

MadGraph7 reports the number of removed diagrams for each subprocess. ``display
diagrams`` shows the remaining ones.

The diagram object
~~~~~~~~~~~~~~~~~~

A tree-level diagram is a list of vertices, ``diag['vertices']``. Each vertex has an
interaction ``id`` (the id of the model interaction, ``0`` for the special vertex that
closes some diagrams) and a list of ``legs``. A leg has the PDG code ``id``, the
``number`` of the external particle (or the smallest external number it combines) and
the ``state`` (``False`` for an incoming particle). The last leg of a vertex is the
propagator that leaves it, except for the last vertex of the diagram.

``model.get_particle(pdg)`` and ``model.get_interaction(id)`` give the particle and the
interaction behind a PDG code or an interaction id.

Loop diagrams (see below) are ``LoopDiagram`` objects. They have the same
``vertices`` and also helpers such as ``diag.get_loop_lines_pdgs()``, the PDG codes of
the particles in the loop. A filter used at loop level must handle both kinds of
diagram.

Selecting the filter
--------------------

The option depends on the kind of generation:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Generation
     - Option
     - Filtered diagrams
   * - Tree level, also decay chains and ``[treextree]``
     - ``--diagram_filter=NAME``
     - All the diagrams, both sides of ``[treextree]``.
   * - MadLoop, ``[virt=QCD]`` and loop-induced ``[noborn=QCD]``
     - ``--diagram_filter=NAME``
     - The Born diagrams and the loop diagrams.
   * - NLO with FKS subtraction, ``[QCD]``
     - ``--tree_diagram_filter=NAME``
     - The Born diagrams, the real-emission diagrams, the diagrams of the counterterms
       and of the electroweak Sudakov amplitudes. They also set the automatic coupling
       orders.
   * - NLO with FKS subtraction, ``[QCD]``
     - ``--loop_diagram_filter=NAME``
     - The loop diagrams of the virtual corrections.

``--diagram_filter`` is refused for the NLO generation with FKS subtraction. The two
options there can be used together, with the same plugin or with two different ones.

``NAME`` is the name of the plugin. Without ``=NAME``, the options use the function
``remove_diag`` of ``PLUGIN/user_filter.py``, the older way to define a filter.
The options also work with ``add process``.

Things to know
--------------

- **NLO real emissions.** The FKS subtraction pairs every real-emission diagram in a
  soft or collinear limit with a Born diagram. A tree filter must keep the
  real-emission diagrams that match the kept Born diagrams. Otherwise the generation
  stops with ``Cannot map born/real configurations``.

- **No Born diagram left.** If ``--tree_diagram_filter`` removes all the Born diagrams,
  the generation stops. A ``[QCD]`` process that has no Born diagram at all, such as
  ``g g > h QED=1 [QCD]``, is generated in the loop-induced mode. There, use
  ``--loop_diagram_filter`` (or ``--diagram_filter``).

- **Multicore NLO generation.** With ``low_mem_multicore_nlo_generation``, the filter is
  sent to the worker processes. It must be a function defined at the top level of a
  module, not a lambda or a nested function.

- **Coupling orders.** At tree level, the automatic coupling orders are determined on
  the filtered diagrams. Give the orders explicitly if a filter removes the lowest-order
  diagrams.

- **Crossing.** Processes generated with a filter do not reuse the diagrams of their
  crossed processes, since a filter does not need to be crossing symmetric.
