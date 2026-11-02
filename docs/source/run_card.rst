Run card
========

The run card ``Cards/run_card.toml`` holds the settings of an MG7 run. It is a
`TOML <https://toml.io>`_ file with one section per topic. This page describes every
section. If you are coming from MadEvent, see :doc:`madevent_to_mg7`.

Editing the run card
--------------------

You can edit the file directly or change values at the launch question with ``set``:

.. code-block:: text

    set events 50000
    set beam.e_cm 14 TeV
    set jet-pt.min 30
    set seed 42

A key name without its section works as long as it is unique. Otherwise write
``section.key``. Energies accept units (``eV`` to ``PeV``) and are stored in GeV. Numeric
values also accept arithmetic expressions with the masses ``mz``, ``mw``, ``mh``, ``mt``,
``mb``, ``mc``, ``mtau`` and ``mmu``, for example ``set fixed_scale mz/2``. ``help <name>`` prints
a short description of a parameter.

Some parameters can be set to ``"auto"``. MG7 then chooses the value during the run. A
number you set explicitly always wins.

The sections ``[multiparticles]``, ``[cuts]`` and ``[histograms]`` are lists rather than
fixed parameters. Their entries are described below.

[run]
-----

``run_name`` (``"run"``)
    Name of the run. Events are written to ``Events/<run_name>_NN``, where ``NN`` counts up
    from 01.

``seed`` (``-1``)
    Random seed. The same seed reproduces a run exactly. ``-1`` draws a new seed for each
    run. The seed that was used is recorded in the output.

``device`` (``["cpu"]``)
    List of devices to run on. Each entry is ``cpu``, ``cuda`` or ``hip``, optionally
    followed by a device index such as ``"cuda:1"``. Several devices share the work.

``cpu_mode`` (``"auto"``)
    SIMD width used on ``cpu`` devices. Choose from ``auto``, ``scalar``, ``simd_128``,
    ``simd_256``, ``simd_512`` and ``avx512y``. ``auto`` picks the widest width the host
    supports.

``madspace_cpu_mode`` (``"auto"``)
    SIMD width of the phase-space sampling on ``cpu`` devices. Choose from ``auto``,
    ``scalar``, ``simd_128`` (ARM only), ``simd_256``, ``avx512y`` and ``simd_512``.
    ``auto`` picks ``simd_512``, then ``simd_256`` on x86. On ARM it picks ``scalar``,
    because the vectorized mode is currently slower there.

``precision`` (``"color32"``)
    Floating-point precision of the matrix elements. ``all64`` and ``all32`` use double and
    single precision throughout. ``color32`` evaluates the color algebra in single precision
    and everything else in double precision. ``denom64`` keeps the momenta and propagator
    denominators in double precision and uses single precision elsewhere.


``cpu_thread_pool_size`` (``-1``), ``gpu_thread_pool_size`` (``1``), ``combine_thread_pool_size`` (``-1``)
    Number of threads for each ``cpu`` device, each GPU device and for combining the
    channel results. ``-1`` chooses based on the number of CPUs.

``output_format`` (``"lhe_npy"``)
    Format of the event file. ``compact_npy`` and ``lhe_npy`` write ``events.npy`` together
    with a ``header.lhe`` that carries the run and param card and the cross-section
    information. ``lhe`` writes a compressed LHE file directly. Tools that read LHE files,
    such as the shower programs, need ``lhe``. The launch question switches to it when you
    select one of them.

``weighted_histograms`` (``false``)
    Fill the :ref:`histograms <run-card-histograms>` with weighted events during the
    integration. This costs one observable evaluation per phase-space point.

``postprocessing_histograms`` (``true``)
    Fill the histograms with the final events, including their scale and PDF weights.
    Plots and the HwU file are made from these.

``make_plots`` (``true``)
    Plot the histograms into ``Events/<run>/plots``. This needs ``matplotlib``. Without it
    the numbers are still written to ``info.json``.

``write_hwu`` (``false``)
    Also write the histograms as ``MADatLO.HwU``. This is the format MG5_aMC writes as
    ``MADatNLO.HwU`` for NLO runs, with the same conventions: the cross section per bin
    (not divided by the bin width, unlike the plots), and the same scale-variation
    columns. ``madgraph/various/histograms.py`` reads both files together.

``verbosity`` (``"auto"``)
    Amount of console output: ``silent``, ``pretty`` or ``log``. ``auto`` is ``pretty`` in
    a terminal and ``log`` otherwise.

``dummy_matrix_element`` (``false``)
    Skip the matrix element evaluation. This is meant for testing the phase-space sampling.

[gridpack]
----------

Gridpacks are self-contained event generators created after the survey and training.
See :doc:`gridpacks` for details.

``save_gridpack`` (``false``)
    Create a gridpack at the end of the run.

``include_source`` (``false``)
    Ship the process source code instead of the compiled libraries.

``include_madspace`` (``true``)
    Include the installed ``madspace`` package, so the gridpack runs without a separate
    installation.

``include_madspace_source`` (``false``)
    Include the ``madspace`` source code.

``compress`` (``false``)
    Pack the gridpack into ``gridpack.tar.gz`` instead of leaving it as a directory.

``run_mode`` (``"minimal"``)
    How the run that creates the gridpack ends. Only used if ``save_gridpack`` is set.

    - ``regular``: generate the events of the run, then store the gridpack.
    - ``minimal``: no events. The gridpack is stored right after the MadNIS training or,
      with VEGAS, once the grid of every channel has converged.
    - ``fix_max_weight``: no events. Each channel runs until ``freeze_max_weight_after``
      unweighted events are reached. Its maximum weight is then fixed in the gridpack,
      and its integral is used to share the events between the channels. This can be
      slow for channels with a low unweighting efficiency.

    See :ref:`gridpack-run-mode`.

``temp_output_dir`` (``""``)
    Directory for the temporary npy files of the channels. ``""`` uses the output
    directory. ``"auto"`` creates a new temporary directory for each run, see
    :py:func:`tempfile.mkdtemp`. Its location can be set with the ``TMPDIR``
    environment variable.

``output_dir`` (``""``)
    Directory for the output files. ``""`` creates a new run folder
    ``Events/<run_name>_NN``. If set, no run folder is created.


[beam]
------

``ebeam1`` (``6500.0``), ``ebeam2`` (``6500.0``)
    Energy of each beam in GeV, beam 1 moving along :math:`+z`. Units are accepted, for
    example ``"7 TeV"``. The collision energy is :math:`2\sqrt{E_1 E_2}`. With different
    energies the events are written, and the :math:`\eta` cuts applied, in this lab frame.
    ``e_cm`` can still be read and set: ``set e_cm 13 TeV`` sets both beams to 6.5 TeV.

``leptonic`` (``false``)
    Treat the beams as leptons: no PDFs are used and the partonic energy equals the
    collision energy. This is set automatically for lepton-collider processes, together
    with 500 GeV beams and no jet cuts.

``pdf1`` (``"NNPDF40_lo_as_01180"``), ``pdf2`` (``"NNPDF40_lo_as_01180"``)
    Name of the LHAPDF set of each beam. ``set beam.pdf X`` sets both. MG7 tries to
    download a set that is not installed. The default is the 5-flavor NNPDF 4.0 LO set,
    which has 100 error members for the PDF variations. With different sets on the two
    beams the PDF member variations of ``[systematics]`` are dropped.

``fixed_ren_scale`` (``false``), ``fixed_fact_scale`` (``false``)
    Use the fixed values below instead of the dynamical scale choice. The two switches act
    on the renormalization and the factorization scale separately.

``ren_scale`` (``91.188``), ``fact_scale1`` (``91.188``), ``fact_scale2`` (``91.188``)
    Fixed scales in GeV. They are only used when the matching switch is on.
    ``fact_scale1`` and ``fact_scale2`` are the factorization scales of the two beams.

``dynamical_scale_choice`` (``"half_transverse_mass"``)
    Scale used when it is not fixed. ``transverse_energy`` is the sum of the transverse
    energies of the final-state particles. ``transverse_mass`` is the sum of their
    transverse masses, :math:`H_T`. ``half_transverse_mass`` is :math:`H_T/2`.
    ``partonic_energy`` is the partonic center-of-mass energy. ``set dynamical_scale_choice
    HT/4`` is a shortcut that sets this parameter and ``scale_factor`` to give
    :math:`H_T/4`.

``scale_factor`` (``1.0``)
    Factor applied to the dynamical scale. Fixed scales are not affected. With the default
    choice, ``scale_factor = 0.5`` gives :math:`H_T/4`.

[generation]
------------

``events`` (``100000``)
    Number of unweighted events to generate.

``survey_min_iters`` (``3``), ``survey_max_iters`` (``3``), ``survey_target_precision`` (``0.1``)
    The survey adapts the sampling before events are generated. These set the minimum and
    maximum number of iterations and the relative precision the survey aims for.

``max_overweight_truncation`` (``0.001``)
    Fraction of the weight distribution's tail that is ignored when the maximum weight for
    unweighting is estimated. Larger values raise the unweighting efficiency at the price of
    more events with a weight above one.

``freeze_max_weight_after`` (``100000``)
    Number of events after which the maximum weight estimate is no longer updated.

``cpu_batch_size`` (``1000``), ``gpu_batch_size`` (``64000``)
    Number of phase-space points per batch on CPU and GPU devices.

``cut_efficiency_threshold`` (``0.7``), ``max_cut_repetitions`` (``1000``)
    When fewer than this fraction of the points in a batch pass the cuts, the batch is
    sampled again, up to the given number of times.

``interference_helicity`` (``"exact"``)
    How the helicity of an interference matrix element is handled, that is of a process
    whose squared split orders keep a cross term (e.g. ``QCD^2==2``) and so can give
    negative weights. Such a cross term can be negative for some helicities and positive
    for others. ``"exact"`` does as madevent: the helicity of each event is drawn with
    probability :math:`|T_i|/\sum_j |T_j|` and the event gets the weight
    :math:`\mathrm{sign}(T_i)\sum_j |T_j|`, so the events carry their exact helicities
    at the price of a larger variance (more negative weights). ``"summed"`` gives each
    event the helicity sum as its weight (smaller variance) and writes its helicities as
    9 (unknown) in the LHE file. Processes without such a cross term are not affected.

[systematics]
-------------

Scale and PDF variation weights are computed while the events are written. They are stored
as ``<rwgt>`` blocks in LHE files and as ``rwgt_<id>`` columns in the npy files.

``enable`` (``true``)
    Compute the variation weights.

``mur`` (``[0.5, 1.0, 2.0]``), ``muf`` (``[0.5, 1.0, 2.0]``)
    Factors applied to the renormalization and the factorization scale. ``1.0`` is the
    nominal value.

``together`` (``true``)
    If true, all combinations of ``mur`` and ``muf`` are evaluated. If false, one scale is
    varied at a time.

``dynamical_scale`` (``[]``)
    Alternative dynamical scale choices, using the same names as ``dynamical_scale_choice``.
    Each is combined with the factors above.

``pdf`` (``["errorset"]``)
    PDF variations. ``"errorset"`` evaluates all members of the nominal set and
    ``"central"`` its central member. LHAPDF set names or ids evaluate all members of that
    set, and ``"<set>@<member>"`` a single one.

``write_inputs`` (``false``)
    Also store :math:`x_1`, :math:`x_2` and the scales of each event, so the weights can be
    recomputed later. In LHE files they go into an ``<mgrwt>`` block.

[postprocessing]
----------------

This section drives the older post-processing steps of MadEvent. They work on the LHE
file, so enabling one switches ``output_format`` to ``"lhe"``.

``time_of_flight`` (``-1.0``)
    Threshold in mm below which the invariant livetime is not written. ``-1`` disables the
    step.

``systematics`` (``false``)
    Recompute the uncertainties with ``systematics.py`` after the run. This only runs if
    ``enable`` in ``[systematics]`` is false, since the native computation replaces it.

``systematics_mur`` (``[0.5, 1.0, 2.0]``), ``systematics_muf`` (``[0.5, 1.0, 2.0]``), ``systematics_pdf`` (``["errorset"]``)
    Scale factors and PDF variations for ``systematics.py``.

``systematics_str_options`` (``""``)
    Extra command-line options passed to ``systematics.py``, for example
    ``"--together=mur,muf"``.

[vegas]
-------

VEGAS grids adapt the phase-space sampling during the survey.

``enable`` (``true``)
    Use VEGAS grids. MadNIS flows start from the VEGAS grid when both are active.

``bins`` (``64``)
    Number of grid bins per dimension.

``damping`` (``0.4``)
    Damping of the grid updates. Raise it if the grid oscillates instead of settling.

``optimization_patience`` (``5``), ``optimization_threshold`` (``0.9``)
    The grid optimization stops when this many iterations in a row fail to improve the
    result by the given relative margin.

``start_batch_size`` (``1000``), ``max_batch_size`` (``32000``)
    Batch size of the first survey iteration and the largest batch size used afterwards.
    Larger batches give a better grid at a proportionally higher cost.

[phasespace]
------------

``mode`` (``"multichannel"``)
    Strategy for the phase space. ``multichannel`` uses one channel per diagram. ``flat``
    uses a single channel described by ``flat_mode``. ``both`` surveys the multichannel
    phase space and then replaces the least important channels with one flat channel, see
    ``combine_channel_threshold``. ``auto`` behaves like ``both`` if MadNIS is enabled and
    like ``multichannel`` otherwise. Decays always use ``multichannel``.

``merge_subprocesses`` (``false``)
    Merge subprocesses with the same diagram topologies into shared channels.

``sde_strategy`` (``"diagrams"``)
    How the multichannel weights are computed. ``diagrams`` uses the squared single
    diagrams, ``denominators`` the product of the propagator denominators.

``t_channel`` (``"propagator"``)
    Parametrization of the t-channel part of the multichannel phase space. ``propagator``
    follows the diagram as a chain of :math:`2 \to 2` steps. ``rambo`` applies RAMBO to the
    t-channel particles and ``chili`` uses collider coordinates.

``flat_mode`` (``"rambo"``)
    The same choice for the flat channel.

``combine_channel_threshold`` (``0.01``)
    In ``both`` mode, the channels that together contribute this fraction of the cross
    section are replaced by the flat channel.

``drop_qcd_s_channel`` (``20``)
    If a subprocess has more channels than this, channels without a QCD s-channel resonance
    are dropped. ``-1`` keeps all channels. This helps with processes that have too many
    channels.

``invariant_power`` (``0.7``)
    Exponent :math:`p` of the :math:`1/s^p` sampling of time-like invariants.

``bw_cutoff`` (``15``)
    Breit-Wigner propagators are sampled within this many widths of the mass. The same
    window applies to propagators excluded with ``$``.

``cut_decays`` (``false``)
    Apply the cuts to the decay products of on-shell particles. As in MadEvent, they are
    left uncut by default.

``adaptive_symmetry_sampling`` (``true``)
    Learn the probabilities of the permutations of identical particles instead of using
    equal ones.

[custom_functions]
------------------

User-written functions, created with ``bin/init_custom_func``. Each one is a shared
library (``.so``) or a Python file (``.py``). Paths are relative to the process directory.
An empty string disables the function. See :doc:`custom_functions` for details.

``cut`` (``""``)
    Custom cut, for example ``"lib/custom_cut.so"`` or ``"custom_cut/custom_cut.py"``. It
    is applied on top of the ``[cuts]``.

``scale`` (``""``)
    Custom renormalization and factorization scales. They replace
    ``dynamical_scale_choice``.

``pdf`` (``""``)
    Custom parton density. It replaces the PDF set, but :math:`\alpha_s` still comes from
    ``[beam] pdf``. The systematics are disabled.

``bias`` (``""``)
    Custom event bias. Events are generated with the weight times the bias and written with
    the weight divided by the bias.

``<function>_parameters.<name>``
    A parameter of a custom function, for example ``cut_parameters.pt_min = 30.0``. It replaces
    the default value declared by the function. See :ref:`custom-function-parameters`.

.. _run-card-multiparticles:

[multiparticles]
----------------

Named groups of particles, given as lists of PDG ids. The names are used in the cuts and
histograms. The defaults are ``jet``, ``bottom``, ``lepton``, ``missing`` and ``photon``:

.. code-block:: toml

    [multiparticles]
    jet = [1, 2, 3, 4, -1, -2, -3, -4, 21]
    bottom = [-5, 5]
    lepton = [11, 13, 15, -11, -13, -15]
    missing = [12, 14, 16, -12, -14, -16]
    photon = [22]

You can add your own groups, for example ``top = [6, -6]``.

.. _run-card-cuts:

[cuts]
------

Each entry has the form ``<selection>-<observable>`` with the bounds ``min`` and ``max``:

.. code-block:: toml

    [cuts]
    jet-pt.min = 20.0
    jet-eta_abs.max = 5.0
    jet-delta_r.min = 0.4
    lepton-pt.min = 10.0
    lepton-eta_abs.max = 2.5
    lepton-delta_r.min = 0.4
    jet-lepton-delta_r.min = 0.4
    sqrt_s.min = 0.0

The selection is one or more groups from ``[multiparticles]``. The observable is one of

* ``pt``, ``mass``, ``e``, ``px``, ``py``, ``pz``, ``p_mag``, ``phi``, ``theta``, ``y``,
  ``y_abs``, ``eta`` and ``eta_abs`` for single particles,
* ``delta_r``, ``delta_eta`` and ``delta_phi`` for pairs of particles,
* ``pair_mass`` and ``sfos_pair_mass`` for the invariant mass of pairs,
* ``sqrt_s`` for the partonic center-of-mass energy. It takes no selection.

A single group applies the observable to each of its particles. ``jet-pt.min = 20``
requires every jet to pass. Two groups, as in ``jet-lepton-delta_r``, form all pairs with
one particle from each group. The pairs of a single group, as in ``jet-delta_r``, are all
distinct pairs within it.

``pair_mass`` is the invariant mass of every pair the groups can form, so
``lepton-pair_mass`` is the mass of every lepton pair. ``sfos_pair_mass`` only keeps
same-flavor opposite-sign pairs such as :math:`e^+ e^-`. It corresponds to MadEvent's
``mmll``.

Add ``-sum-`` before the observable to add up the momenta first. ``lepton-lepton-sum-mass``
is the mass of two leptons taken together, and ``lepton-jet-top-sum-mass`` the mass of a
lepton, a jet and a top. Add ``-sum`` after the observable to add up the values instead, so
``jet-pt-sum`` is the scalar sum of the jet transverse momenta.

An index selects particles by rank. ``jet_1-pt`` is the hardest jet, ``jet_2-pt`` the second
hardest, and so on. The key ``order_by = "pt"`` sets the observable that defines the rank.

By default a bound must hold for every selected particle. Setting ``mode = "any"`` on an
entry relaxes this to at least one.

The selections are flexible because you define the groups. Some examples:

.. code-block:: toml

    [multiparticles]
    alllepton = [11, 13, 15, -11, -13, -15, 12, 14, 16, -12, -14, -16]
    top = [6, -6]

    [cuts]
    jet-pt-sum.min = 200.0             # H_T of the jets
    jet_1-jet_2-pt-sum.min = 150.0     # H_T of the two hardest jets
    jet_2-pt.min = 40.0                # second hardest jet
    alllepton-sum-pt.min = 30.0        # pT of all leptons and neutrinos together
    top-pair_mass.min = 250.0          # mass of every pair of tops
    jet-eta_abs.min = 1.0              # bounds work in both directions

Cuts also restrict the sampling. ``pair_mass`` and ``sqrt_s`` minima bound the
integration region, as do ``pt`` and ``delta_r`` minima through the masses they imply.
``eta_abs`` maxima bound the rapidity of the partonic system and, together with the ``pt``
minima, the scattering angle. This only improves efficiency. The result is the same.

Use ``set no_parton_cut`` to remove all cuts. ``set <selection>-<observable>.min <value>``
changes a single bound. Processes without initial-state particles, such as decays, start
without cuts.

.. _run-card-histograms:

[histograms]
------------

Histograms of observables, filled during the run and written to ``Events/<run>/info.json``.
They use the same observable names as the cuts. Each entry needs a range and a number of
bins:

.. code-block:: toml

    [histograms]
    jet_1-pt.min = 0.0
    jet_1-pt.max = 500.0
    jet_1-pt.bin_count = 50

The entry ``weight`` is not an observable of the momenta. It is the distribution of the
event weight in units of the unit weight :math:`\sigma_\mathrm{abs}`, the integral of
the absolute value of the weights. A fully unweighted sample is a spike at 1, and its
negative weights (from an interference) a spike at -1.

``output`` writes a default set of histograms for the final state of your process. ``set
histograms OFF`` at the launch question removes them and ``set histograms default``
restores them. The parameters ``weighted_histograms``, ``postprocessing_histograms``,
``make_plots`` and ``write_hwu`` in ``[run]`` control what is done with them.

[madnis]
--------

`MadNIS <https://arxiv.org/abs/2212.06172>`_ trains neural networks to improve the phase-space
sampling. It needs PyTorch. ``enable`` can be ``true``, ``false`` or ``"auto"``. In auto
mode, MG7 surveys first and then decides for each subprocess. Processes with two outgoing
particles never use MadNIS. Three outgoing particles use it if the survey is noisy, if more
than a million events are requested or if a gridpack is built. Four or more outgoing
particles always use it. Auto mode also sizes the networks and picks the learning rate.
The parameters marked ``"auto"`` below are chosen this way, and setting one yourself
overrides this.

Start tuning with ``train_batches`` and ``batch_size_per_channel``. Next try ``lr`` and the
loss. Change the network shapes last.

Training
^^^^^^^^

``enable`` (``"auto"``)
    Train MadNIS networks.

``train_batches`` (``1000``)
    Number of training batches.

``batch_size_per_channel`` (``"auto"``)
    Minimum number of training events per active channel.

``batch_size_offset`` (``512``)
    Number of events added to the training batch size of every channel.

``log_interval`` (``100``)
    Batches between two entries in the training log.

``uniform_channel_ratio`` (``0.5``)
    Fraction of the training batch spread uniformly over the channels. The rest is
    distributed according to the channel integrals.

``integration_history_length`` (``100``)
    Number of batches for which the mean and variance of each channel are kept.

``drop_zero_integrands`` (``true``)
    Ignore points with a vanishing integrand in the training.

``fixed_cwnet_fraction`` (``"auto"``)
    Fraction of the training during which the channel weight network is frozen. It only
    starts training afterwards.

``softclip_threshold`` (``30.0``)
    Soft clipping threshold of the channel weights. ``0`` disables it.

``compressed_channel_weight_count`` (``50``)
    Number of channel weights kept per event in the multichannel weight loss.

Optimizer
^^^^^^^^^

``lr`` (``"auto"``)
    Learning rate of the Adam optimizer.

``lr_scheduler`` (``"cosine"``)
    ``cosine`` decays the learning rate to zero over the training. ``none`` keeps it
    constant.

``adam_beta1`` (``0.9``), ``adam_beta2`` (``0.999``), ``adam_eps`` (``1e-8``), ``adam_weight_decay`` (``1e-4``)
    Adam parameters.

``grad_clip_threshold`` (``0.003``)
    Maximum gradient norm. ``0`` disables clipping.

Replay buffer
^^^^^^^^^^^^^

Training can reuse earlier samples. This saves matrix element evaluations.

``buffer_capacity`` (``60000``)
    Number of samples kept per channel. ``0`` disables the buffer.

``minimum_buffer_size`` (``10000``)
    Number of buffered samples needed before training on the buffer starts.

``buffered_steps_fraction`` (``0.8``)
    Fraction of the training steps done on buffered samples.

``buffer_skip_batches`` (``1000``)
    Number of initial batches that are not stored.

``buffer_unweighting_quantile`` (``0.95``)
    Quantile of the weights used as the maximum when buffered samples are unweighted.

Sample generation
^^^^^^^^^^^^^^^^^

``generator_target_size_factor`` (``32``)
    Number of training batches the sample generator keeps buffered for each channel.

``gpu_generator_batch_granularity`` (``1000``)
    Batch sizes for sample generation on GPUs are rounded to a multiple of this.

Channel dropping
^^^^^^^^^^^^^^^^

``channel_dropping_threshold`` (``0.01``), ``channel_dropping_interval`` (``100``)
    Every ``channel_dropping_interval`` batches, channels that contribute less than the
    threshold fraction of the integral are dropped.

Networks
^^^^^^^^

Three networks are trained. The flow maps the phase-space sampling, the discrete network
learns the probabilities of discrete choices such as permutations and flavors, and the
channel weight network learns the multichannel weights. Each is built from fully connected
subnetworks and has settings for the number of hidden units and layers and the activation
function. The activations are ``relu``, ``leaky_relu``, ``elu``, ``gelu``, ``sigmoid`` and
``softplus``.

``flow_hidden_dim`` (``"auto"``), ``flow_layers`` (``"auto"``), ``flow_activation`` (``"leaky_relu"``)
    Size and activation of the flow subnetworks.

``flow_spline_bins`` (``10``)
    Number of spline bins per flow transformation.

``flow_invert_spline`` (``false``)
    Apply the splines in the inverse direction.

``discrete_hidden_dim`` (``"auto"``), ``discrete_layers`` (``3``), ``discrete_activation`` (``"leaky_relu"``)
    Size and activation of the discrete network.

``cwnet_hidden_dim`` (``"auto"``), ``cwnet_layers`` (``3``), ``cwnet_activation`` (``"leaky_relu"``)
    Size and activation of the channel weight network.

``nn_float_precision`` (``"float32"``)
    Floating-point precision of the weights and the evaluation of all networks,
    ``float32`` or ``float64``. The rest of the computation, including the network
    outputs, softmaxes and Jacobians, stays in double precision.
