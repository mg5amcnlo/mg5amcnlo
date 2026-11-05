Gridpacks
=========

A gridpack is a snapshot of a finished run. The survey and the training are done, so only
the event generation is left. It is a good fit for producing many independent samples,
for example one job per seed on a cluster, without repeating the expensive setup.

Creating a gridpack
-------------------

Set ``save_gridpack`` in the ``[gridpack]`` section of the :doc:`run card <run_card>` and
start a run:

.. code-block:: text

    set save_gridpack true
    launch

The gridpack is written to ``Events/<run_name>_NN/gridpack``, or to ``gridpack`` in the
``output_dir`` of the ``[gridpack]`` section if it is set. It is a plain directory. With
``compress = true``, it is packed into ``gridpack.tar.gz`` instead, ready to be moved to
another machine.

By default, the run stops once the gridpack is ready and does not generate events. See
:ref:`gridpack-run-mode` for the other options.

A gridpack contains:

``bin/generate_events``
    The script that generates events.

``Cards/``
    ``grid_run_card.toml`` with the settings that matter for event generation, the
    ``param_card.dat``, and a copy of the full ``run_card.toml`` for reference. Changing
    the copy has no effect.

``data/``
    The trained phase-space channels and the information needed to write LHE files and
    the scale and PDF weights.

``lib/`` or ``src/``, ``SubProcesses/`` and ``backend/``
    The matrix elements, see :ref:`gridpack-matrix-elements`.

``madspace/``
    The ``madspace`` package, see :ref:`gridpack-madspace`.

``Events/``
    Empty. The output of each gridpack run goes here.

.. _gridpack-run-mode:

Ending the run
^^^^^^^^^^^^^^

``run_mode`` in the ``[gridpack]`` section sets how the run that creates the gridpack
ends:

``minimal`` (default)
    No events are generated. The gridpack is stored right after the MadNIS training or,
    with VEGAS, once the grid of every channel has converged. This is the fastest way
    to create a gridpack. Each gridpack run then estimates the maximum weights and the
    channel integrals on its own.

``fix_max_weight``
    No events are generated. Each channel runs until ``freeze_max_weight_after``
    unweighted events are reached. Its maximum weight is stored in the gridpack and kept
    fixed in every gridpack run, so all runs unweight against the same maximum. The
    channel integrals are stored as well and combined with the ones of the gridpack run
    to share the events between the channels from the start. Creating the gridpack takes
    longer, especially for channels with a low unweighting efficiency, but the gridpack
    runs do not have to estimate the maximum weights first. This helps if each run only
    generates a few events.
    The gridpack also gets ``bin/event_stream.py``, see :ref:`gridpack-event-stream`.

``regular``
    The events of the run are generated as usual, then the gridpack is stored.

Generating events
-----------------

Run the script from anywhere:

.. code-block:: bash

    ./bin/generate_events --events 10000 --seed 42

Each call creates a new directory ``Events/<run_name>_NN``. Use ``--output_dir`` to write
the output to a given directory instead, and ``--temp_output_dir`` to put the large
temporary npy files of the channels somewhere else, for example on a fast local disk of a
cluster node. With ``--temp_output_dir auto``, a new temporary directory is created for
each run (in ``$TMPDIR`` if set), so that runs sharing a gridpack do not collide. Relative paths on the command line refer to the current directory. A gridpack generates LHE files
by default, since it usually feeds a shower or detector simulation. The file is
compressed to ``events.lhe.gz``. Use ``--output_format`` to get one of the npy formats
instead.

Make sure every job uses a different seed. The default seed ``-1`` draws a fresh random
seed for each call, which is fine for independent jobs. The seed that was used is recorded
in the output.

.. _gridpack-event-stream:

Streaming events from Python
----------------------------

A gridpack created with ``run_mode = "fix_max_weight"`` also contains
``bin/event_stream.py``. Its ``EventStream`` class generates events in memory and returns
them to a Python program, without writing any files. See :doc:`event_stream`.

Run card settings
-----------------

The settings of ``grid_run_card.toml`` are the ones marked below. They are read from the
card when the script starts. Edit the card to change the defaults of a gridpack. All other
settings, such as the cuts, the beams or the integration parameters, are fixed when the
gridpack is created.

.. list-table::
    :header-rows: 1
    :widths: 22 33 45

    * - Section
      - Setting
      - Command-line option
    * - ``[run]``
      - ``run_name``
      - ``--run_name``
    * -
      - ``seed``
      - ``--seed``
    * -
      - ``device``
      - ``--device`` (several entries are allowed)
    * -
      - ``madspace_cpu_mode``
      - ``--madspace_cpu_mode``
    * -
      - ``cpu_thread_pool_size``, ``gpu_thread_pool_size``
      - ``--cpu_thread_pool_size``, ``--gpu_thread_pool_size``
    * -
      - ``verbosity``
      - ``--verbosity``
    * -
      - ``output_format``
      - ``--output_format``
    * -
      - ``cpu_mode``, ``precision``, ``combine_thread_pool_size``
      - Card only.
    * - ``[generation]``
      - ``events``
      - ``--events``
    * -
      - ``max_overweight_truncation``, ``freeze_max_weight_after``
      - ``--max_overweight_truncation``, ``--freeze_max_weight_after``
    * -
      - ``cpu_batch_size``, ``gpu_batch_size``
      - ``--cpu_batch_size``, ``--gpu_batch_size``
    * -
      - ``cut_efficiency_threshold``, ``max_cut_repetitions``
      - Card only.
    * - ``[amplicol]``
      - ``first_pass_max_overweight_truncation``, ``first_pass_freeze_max_weight_after``
      - ``--first_pass_max_overweight_truncation``, ``--first_pass_freeze_max_weight_after``
    * - ``[gridpack]``
      - ``output_dir``, ``temp_output_dir``
      - ``--output_dir``, ``--temp_output_dir``
    * - ``[systematics]``
      - ``enable``, ``mur``, ``muf``, ``together``, ``dynamical_scale``, ``pdf``,
        ``write_inputs``
      - Card only.

An option on the command line overrides the card. The settings have the same meaning as in
the main run card, see :doc:`run_card`. ``./bin/generate_events --help`` lists all options.
Relative ``output_dir`` and ``temp_output_dir`` paths in the card refer to the gridpack
directory. Both are reset to ``""`` when the gridpack is created.

With ``[amplicol]`` processes unweighted to full colour, ``max_overweight_truncation`` and
``freeze_max_weight_after`` set the second pass, which gives the events, and the
``first_pass_`` options the leading-colour unweighting of these processes, as in the main
run. The other processes are unweighted once, as set by ``[generation]``. The colour
accuracy and the mode are fixed when the gridpack is created. The pyamplicol artifacts
the libraries read are copied to ``SubProcesses/``, and ``include_source = true`` is not
supported for these processes.

The card names the ``cpu_mode`` that the gridpack was built with, which can differ from the
``"auto"`` of the original run card. The matrix element libraries carry this name. If you
change ``cpu_mode`` or ``precision``, you have to rebuild them.

The scale and PDF weights need the LHAPDF grids. They are searched in
``$LHAPDF_DATA_PATH``, in the paths known to the ``lhapdf`` module and, finally, in the
directory used when the gridpack was created. Set ``LHAPDF_DATA_PATH`` on machines where
the grids are somewhere else.

Including the matrix elements and madspace
------------------------------------------

By default, a gridpack contains compiled code only. This makes it small and ready to run,
but ties it to the operating system and the CPU architecture it was built on. The options
in the ``[gridpack]`` section allow you to ship source code instead.

.. _gridpack-matrix-elements:

Matrix elements
^^^^^^^^^^^^^^^

``include_source = false`` (default)
    The compiled matrix element libraries are copied to ``lib/``. They are the ones that
    were built for the run, which means the backends of the devices and the ``cpu_mode``
    that you used. A gridpack can only run on devices for which a library exists.

``include_source = true``
    The source code is copied to ``src/``, ``SubProcesses/`` and ``backend/`` and ``lib/``
    stays empty.
    This is useful if the gridpack has to run on a different platform. The gridpack does
    not compile anything by itself, so build the libraries once before the first run:

    .. code-block:: bash

        cd SubProcesses
        make -j8 BACKEND=simd_128 FPTYPE=m USEBUILDDIR=1

    ``BACKEND`` is the ``cpu_mode`` in ``Cards/grid_run_card.toml`` for a ``cpu`` device, and
    ``cuda`` or ``hip`` for a GPU. ``FPTYPE`` follows the ``precision`` of the card:
    ``d`` for ``all64``, ``f`` for ``all32``, ``m`` for ``color32`` and ``v`` for
    ``denom64``. A C++ compiler is required, and a CUDA or HIP toolchain for the GPU
    backends. Without the libraries, ``generate_events`` stops with an error that it could
    not load a shared object from ``lib/``.

.. _gridpack-madspace:

madspace
^^^^^^^^

The script looks for ``madspace`` in this order:

1. ``madspace/install/madspace``, a compiled copy. It is included with
   ``include_madspace = true`` (default).
2. ``madspace/install.py``, the source code. It is included with
   ``include_madspace_source = true``. On the first call, the script starts the
   interactive installation of ``madspace``, which requires a compiler. After this,
   the compiled copy is used.
3. The ``madspace`` that Python can import from the environment, for example from
   ``pip`` or ``PYTHONPATH``. This is used if the gridpack contains neither option.

If both options are on, the compiled copy is used and the source is only a fallback for
rebuilding it. The compiled copy has to match the platform. The source code makes the
gridpack portable.

The script stops with an error if the ``madspace`` it uses differs from the one that
created the gridpack, since the results can be wrong in this case. Use
``--ignore_source_hash`` to run anyway, which only prints a warning.

The options can be combined. The smallest gridpack sets ``include_madspace = false`` and
relies on an installed ``madspace``. A fully portable gridpack sets ``include_source = true``
and ``include_madspace_source = true``. The default is the best choice if all machines
are the same.
