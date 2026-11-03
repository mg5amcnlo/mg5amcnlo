Streaming events from Python
============================

A gridpack can generate events directly into a Python program, without writing any files.
This is useful when the events are consumed right away, for example by an analysis written
in Python or as training data for a neural network. The ``EventStream`` class in the
gridpack's ``bin/event_stream.py`` returns the events either as LHE event objects or as
batches of NumPy arrays or PyTorch tensors.

Event streaming needs a gridpack created with ``run_mode = "fix_max_weight"`` (see
:ref:`gridpack-run-mode`). In this mode, the maximum weight of every channel is fixed when
the gridpack is created. An event is final as soon as it is unweighted, so it can be returned
right away without a survey or a combination step.

Example: the e+ e- invariant mass
---------------------------------

This example fills a histogram of the invariant mass of the electron pair in
``p p > e+ e-``. First, create the gridpack at the MadGraph7 prompt:

.. code-block:: text
    :name: event-stream-mg5-commands

    generate p p > e+ e-
    output mg7 PROC_ppee
    launch PROC_ppee
    set save_gridpack true
    set run_mode fix_max_weight
    done

The gridpack is written to ``PROC_ppee/Events/run_01/gridpack``. Its ``bin`` directory
contains ``event_stream.py``, so add it to the Python path and import ``EventStream``. The
script can run from any directory.

.. code-block:: python

    import sys

    import numpy as np

    sys.path.insert(0, "PROC_ppee/Events/run_01/gridpack/bin")
    from event_stream import EventStream

Creating an ``EventStream`` starts the event generation in background threads. Its
arguments, like the device, the seed and the batch sizes, default to the values in the
gridpack's ``Cards/grid_run_card.toml``. The stream is endless, so ``max_events`` limits
the number of events. Each batch is an :py:class:`EventBatch <event_stream.EventBatch>`
with one array per LHE field. The particle fields have the shape
``(batch_size, max_particle_count)``.

.. code-block:: python

    bins = np.linspace(60.0, 120.0, 31)
    hist = np.zeros(len(bins) - 1)
    weight_sum = 0.0

    with EventStream(seed=42) as stream:
        for batch in stream.batches(10000, max_events=100000):
            # four-momentum of the e+ e- pair: sum over the outgoing electrons
            is_electron = (np.abs(batch.pdg_id) == 11) & (batch.status_code == 1)
            e, px, py, pz = (
                np.sum(np.where(is_electron, p, 0.0), axis=1)
                for p in (batch.energy, batch.px, batch.py, batch.pz)
            )
            m_ee = np.sqrt(np.maximum(e**2 - px**2 - py**2 - pz**2, 0.0))
            hist += np.histogram(m_ee, bins, weights=batch.weight)[0]
            weight_sum += np.sum(batch.weight)
        cross_section = stream.integral
        print(f"cross section: {cross_section:.1f} +- {stream.integral_error:.1f} pb")

The event weights are not normalized to the cross section (see :ref:`event-stream-weights`).
Scaling them to the cross section gives the differential cross section in pb/GeV:

.. code-block:: python

    dsigma = hist * cross_section / weight_sum / np.diff(bins)
    for low, value in zip(bins[:-1], dsigma):
        print(f"{low:5.0f} GeV {value:8.3f} pb/GeV " + "#" * int(50 * value / dsigma.max()))

    peak = np.argmax(dsigma)
    assert bins[peak] < 91.2 < bins[peak + 1], "the Z peak is missing"

The output shows the Z peak (shortened)::

    cross section: 880.2 +- 0.6 pb
    ...
       84 GeV    7.569 pb/GeV ##
       86 GeV   16.543 pb/GeV #####
       88 GeV   51.287 pb/GeV ################
       90 GeV  158.376 pb/GeV ##################################################
       92 GeV   66.743 pb/GeV #####################
       94 GeV   18.457 pb/GeV #####
    ...

Output formats
--------------

There are three ways to get events from a stream:

- Iterating over the stream yields one ``madspace.LHEEvent`` at a time. The events are
  completed in chunks of 1000 internally. If the loop is left early, the rest of the chunk
  is returned first by the next call, so no event is skipped.
- :py:meth:`next_events <event_stream.EventStream.next_events>` returns a list of a given
  number of ``LHEEvent`` objects.
- :py:meth:`batches <event_stream.EventStream.batches>` and
  :py:meth:`next_batch <event_stream.EventStream.next_batch>` return
  :py:class:`EventBatch <event_stream.EventBatch>` objects. With ``output_type="torch"``,
  their fields are PyTorch tensors on the CPU instead of NumPy arrays.

Events have different numbers of particles if the process has several subprocesses or
includes decays. The particle fields are then padded with zeros. ``particle_count`` holds
the number of particles of every event and ``particle_mask`` marks the valid entries.

.. _event-stream-weights:

Weights and cross section
-------------------------

After the unweighting, every event weight is ``1``, or ``-1`` for an event with a negative
weight. The exception are overweight events, whose weight exceeded the fixed maximum weight
of their channel. They keep the ratio of their weight to the maximum weight, so their weight
is larger than ``1``. Treating them as ``1`` would bias the distributions in the phase-space
regions where they occur.

The events of ``[amplicol]`` subprocesses are then reweighted to full colour in the default
``mode = "reweight"``: their weight is multiplied by the ratio of the full-colour to the
leading-colour matrix element, so it is not ``1``. With ``mode = "unweight"``, they are
unweighted a second time and have unit weights apart from overweights.

In all cases, to normalize a sample to the cross section, scale its weights by
``stream.integral / sum(weights)``. The cross section estimate improves while events are
generated: it combines the integrals stored in the gridpack with the weighted events
generated by the stream so far. Read it after generating the sample, as in the example
above.

How it works
------------

Each channel of the gridpack keeps a buffer of unweighted events, which background threads
refill whenever it runs low. The channel of every event is drawn at random, with
probabilities given by the current cross section estimates of the channels. Generating
the phase-space points and evaluating the matrix elements happen in separate steps, as in a
normal gridpack run.

The events are reproducible: for a given seed, the stream always returns the same sequence of
events. It does not depend on the number of threads or on how the events are split into
batches.

API reference
-------------

.. autoclass:: event_stream.EventStream
    :members:
    :special-members: __iter__
    :member-order: bysource

.. autoclass:: event_stream.EventBatch
    :members:
    :member-order: bysource
