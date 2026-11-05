Coming from MadEvent
====================

MadEvent is configured with ``Cards/run_card.dat``, a list of ``value = name`` lines. MG7 uses
``Cards/run_card.toml`` instead, which is described in :doc:`run_card`. This page shows where
the MadEvent settings went. The ``param_card.dat`` is unchanged.

Not every MadEvent setting has a counterpart yet. The last section lists what is missing.

Setting values from old scripts
-------------------------------

At the launch question, a few MadEvent names are translated to their MG7 equivalent, so
existing scripts keep working. These are ``nevents``, ``gridpack``, ``fixed_ren_scale``,
``scale``, ``dsqrt_q2fact1``, ``dsqrt_q2fact2``, ``scalefact``, ``bwcutoff``, ``cut_decays``
and ``use_syst``. The shortcuts ``set lhc``, ``set lep``, ``set fixed_scale``,
``set no_parton_cut`` and ``set dynamical_scale_choice HT/n`` work as before, and so does
``set iseed``. The beam settings ``ebeam``, ``ebeam1``, ``ebeam2``, ``lpp``, ``lpp1``,
``lpp2``, ``pdlabel``, ``pdlabel1``, ``pdlabel2`` and ``lhaid`` are translated as in the
table below. MadEvent draws a random seed for ``iseed = 0``. MG7 does the same for
``seed = -1`` and treats ``0`` as an ordinary fixed seed, so ``set iseed 0`` is translated.

The cut names of the `Cuts`_ table are translated as well, so ``set mmll 200`` sets
``lepton-sfos_pair_mass.min`` and ``set ptj1min 50`` sets ``jet_1-pt.min``. As in
``run_card.dat``, a minimum of zero or a negative maximum removes that bound. A cut on the
``parton`` or ``alllepton`` group also adds the group to ``[multiparticles]`` if it is not
there yet. The per-pdg cuts are written as in ``run_card.dat``, for example
``set pt_min_pdg {6: 100}``, and put on a group ``pdg6 = [6, -6]``. ``ptheavy`` is not
translated, because it needs a group of your own.

``fixed_fac_scale``, ``fixed_fac_scale1``, ``fixed_fac_scale2`` and ``maxjetflavor`` are
translated as in the tables below, and so are the numbers of ``dynamical_scale_choice``
(``set dynamical_scale_choice 3``) and ``sde_strategy``. MG7 has one
``beam.fixed_fact_scale`` for both beams, so different ``fixed_fac_scale1`` and
``fixed_fac_scale2`` are reported once the launch question closes.

All other settings must use their new names.

Beams and PDFs
--------------

.. list-table::
    :header-rows: 1
    :widths: 30 30 40

    * - ``run_card.dat``
      - ``run_card.toml``
      - Notes
    * - ``ebeam1``, ``ebeam2``
      - ``beam.ebeam1``, ``beam.ebeam2``
      - One energy per beam. With different energies the events are written, and the
        :math:`\eta` cuts applied, in the lab frame.
    * - ``lpp1``, ``lpp2``
      - ``beam.leptonic``
      - ``false`` for proton beams and ``true`` for lepton beams without PDFs. Other beam
        types, such as elastic photons or antiprotons, are not supported.
    * - ``pdlabel``, ``lhaid``
      - ``beam.pdf1``, ``beam.pdf2``
      - The name of the LHAPDF set instead of its id. For example, ``lhaid = 331900`` becomes
        ``"NNPDF40_lo_as_01180"``. ``set beam.pdf`` sets both beams.
    * - ``pdlabel1``, ``pdlabel2``
      - ``beam.pdf1``, ``beam.pdf2``
      - One set per beam. A beam without PDF exists only for a lepton collider, as
        ``beam.leptonic``.

Scales
------

.. list-table::
    :header-rows: 1
    :widths: 30 30 40

    * - ``run_card.dat``
      - ``run_card.toml``
      - Notes
    * - ``fixed_ren_scale``
      - ``beam.fixed_ren_scale``
      -
    * - ``fixed_fac_scale``
      - ``beam.fixed_fact_scale``
      - A single switch for both beams.
    * - ``scale``
      - ``beam.ren_scale``
      -
    * - ``dsqrt_q2fact1``, ``dsqrt_q2fact2``
      - ``beam.fact_scale1``, ``beam.fact_scale2``
      -
    * - ``dynamical_scale_choice``
      - ``beam.dynamical_scale_choice``
      - A name instead of a number. ``1`` is ``"transverse_energy"``, ``2`` is
        ``"transverse_mass"``, ``3`` is ``"half_transverse_mass"`` and ``4`` is
        ``"partonic_energy"``.
    * - ``scalefact``
      - ``beam.scale_factor``
      - Applies to the dynamical scale only.

The MadEvent default ``dynamical_scale_choice = -1`` follows the clustering of the diagram,
as needed for CKKW merging. MG7 does not have it. Its default is ``half_transverse_mass``.

Generation and phase space
--------------------------

.. list-table::
    :header-rows: 1
    :widths: 30 30 40

    * - ``run_card.dat``
      - ``run_card.toml``
      - Notes
    * - ``nevents``
      - ``generation.events``
      -
    * - ``iseed``
      - ``run.seed``
      - ``-1`` instead of ``0`` for a random seed.
    * - ``run_tag``
      - ``run.run_name``
      - Similar, but the run directory is ``Events/<run_name>_NN``.
    * - ``gridpack``
      - ``gridpack.save_gridpack``
      - See ``[gridpack]`` for further options.
    * - ``bwcutoff``
      - ``phasespace.bw_cutoff``
      -
    * - ``cut_decays``
      - ``phasespace.cut_decays``
      - ``false`` by default in both.
    * - ``SDE_strategy``
      - ``phasespace.sde_strategy``
      - ``1`` is ``"diagrams"`` and ``2`` is ``"denominators"``.
    * - ``maxjetflavor``
      - ``[multiparticles]`` ``jet``
      - List the PDG ids of the quarks that count as jets. The default includes the quarks
        up to the charm quark and the gluon. Add ``5`` and ``-5`` for ``maxjetflavor = 5``.
    * - ``use_syst``
      - ``systematics.enable``
      - The scale and PDF variations are computed natively. Choose them with
        ``systematics.mur``, ``muf`` and ``pdf``.
    * - ``time_of_flight``
      - ``postprocessing.time_of_flight``
      -

Technical settings for the MadEvent run engine, such as ``vector_size``, ``nb_warp``,
``job_strategy``, ``survey_splitting``, ``refine_evt_by_job`` and the ``hel_*`` options, have
no counterpart. The integration and the hardware are configured in ``[run]``,
``[generation]``, ``[vegas]``, ``[phasespace]`` and ``[madnis]`` instead.

Cuts
----

MadEvent has one parameter per cut. MG7 has one entry per observable in ``[cuts]``, named
after the particle groups and the observable, as explained in :ref:`the run card page
<run-card-multiparticles>`. A bound that is switched off in MadEvent, such as a minimum of
zero, is simply left out.

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - ``run_card.dat``
      - ``run_card.toml``
    * - ``ptj``, ``ptb``, ``pta``, ``ptl`` and the ``max`` versions
      - ``jet-pt``, ``bottom-pt``, ``photon-pt``, ``lepton-pt`` with ``.min`` and ``.max``
    * - ``misset``, ``missetmax``
      - ``missing-pt.min``, ``missing-pt.max``
    * - ``etaj``, ``etab``, ``etaa``, ``etal``
      - ``jet-eta_abs.max``, ``bottom-eta_abs.max``, ``photon-eta_abs.max``,
        ``lepton-eta_abs.max``
    * - ``drjj``, ``drbb``, ``drll``, ``draa`` and the ``max`` versions
      - ``jet-delta_r``, ``bottom-delta_r``, ``lepton-delta_r``, ``photon-delta_r``
    * - ``drbj``, ``draj``, ``drab``, ``drbl``, ``drjl``, ``dral`` and the ``max`` versions
      - ``bottom-jet-delta_r``, ``photon-jet-delta_r``, ``photon-bottom-delta_r``,
        ``bottom-lepton-delta_r``, ``jet-lepton-delta_r``, ``photon-lepton-delta_r``
    * - ``mmjj``, ``mmbb``, ``mmaa`` and the ``max`` versions
      - ``jet-pair_mass``, ``bottom-pair_mass``, ``photon-pair_mass``
    * - ``mmll``, ``mmllmax``
      - ``lepton-sfos_pair_mass.min``, ``lepton-sfos_pair_mass.max``
    * - ``dsqrt_shat``, ``dsqrt_shatmax``
      - ``sqrt_s.min``, ``sqrt_s.max``
    * - ``etajmin``, ``etabmin``, ``etaamin``, ``etalmin``
      - ``jet-eta_abs.min``, ``bottom-eta_abs.min``, ``photon-eta_abs.min``,
        ``lepton-eta_abs.min``
    * - ``ej``, ``eb``, ``ea``, ``el`` and the ``max`` versions
      - ``jet-e``, ``bottom-e``, ``photon-e``, ``lepton-e`` with ``.min`` and ``.max``
    * - ``ptj1min`` to ``ptj4max``, ``ptl1min`` to ``ptl4max``
      - ``jet_1-pt`` to ``jet_4-pt`` and ``lepton_1-pt`` to ``lepton_4-pt`` with ``.min``
        and ``.max``
    * - ``xptj``, ``xptl``
      - ``jet_1-pt.min``, ``lepton_1-pt.min``. At least one particle passes if the hardest one
        does.
    * - ``htjmin``, ``htjmax``
      - ``jet-pt-sum.min``, ``jet-pt-sum.max``
    * - ``ihtmin``, ``ihtmax``
      - ``parton-pt-sum.min``, ``parton-pt-sum.max``, with a group ``parton`` that holds all
        quarks including the b quark and the gluon
    * - ``ht2min``, ``ht3min``, ``ht4min`` and the ``max`` versions
      - ``jet_1-jet_2-pt-sum``, ``jet_1-jet_2-jet_3-pt-sum``,
        ``jet_1-jet_2-jet_3-jet_4-pt-sum``
    * - ``ptllmin``, ``ptllmax``
      - ``alllepton-sum-pt.min``, ``alllepton-sum-pt.max``, with a group ``alllepton`` that
        holds the charged leptons and the neutrinos
    * - ``mmnl``, ``mmnlmax``
      - ``alllepton-sum-mass.min``, ``alllepton-sum-mass.max``
    * - ``ptheavy``
      - ``heavy_1-pt.min``, with a group ``heavy`` that holds the heavy particles of your
        process
    * - ``pt_min_pdg``, ``pt_max_pdg``, ``e_min_pdg``, ``e_max_pdg``, ``eta_min_pdg``,
        ``eta_max_pdg``
      - A group for the particle, for example ``top = [6, -6]``, and ``top-pt``,
        ``top-e`` or ``top-eta_abs`` with ``.min`` and ``.max``
    * - ``mxx_min_pdg``
      - ``top-pair_mass.min`` for the group ``top = [6, -6]``. Use ``top-sfos_pair_mass``
        instead if ``mxx_only_part_antipart`` is true.

The cut ``mmll`` only applies to pairs of same-flavor and opposite-sign leptons. The
observable ``sfos_pair_mass`` does the same.

The groups in the last rows are your own entries in ``[multiparticles]``. The page about the
:ref:`cut syntax <run-card-cuts>` explains how the selections, ``-sum-`` and ``-sum`` are
combined.

Decay widths
------------

``compute_widths`` and the ``auto`` widths of a ``param_card`` integrate the decays into
three bodies or more with madspace, like the rest of MG7, where MadEvent used to be run. The
two-body widths still come from the formulas of the model when it has them. Each decaying
particle is written as a decay output (``output mg7``) in a temporary directory and surveyed:
the integration takes milliseconds and the build is what there is to wait for. A build of at
least four subprocess directories (this holds for any mg7 run) precompiles the standard
headers first (``make PCH=1``, clang), which halves it. For the Higgs of the ``heft`` model up to three bodies, the whole
command takes 6 s, against 26 s with MadEvent.

``alpha_s`` of a decay is run from the ``alpha_s(m_Z)`` of the card up to the mass of the
decaying particle, with the same two-loop running as MadEvent. The widths of the two
integrators agree within the integration error (the ``--precision_channel`` of
``compute_widths``, 1% by default, is the relative precision aimed at for each channel).

The integrator is madspace. The ``--engine=madevent`` flag of ``compute_widths`` (or of an
``auto`` width of the card, ``DECAY 6 auto --engine=madevent``) selects the Fortran MadEvent
integration of the former releases instead::

    compute_widths t --body_decay=3 --engine=madevent

MG7 goes back to MadEvent on its own, with a warning, if madspace cannot compute the widths
and no engine was asked for with ``--engine``. The complex mass scheme is not available in
madspace: the widths are always computed with MadEvent then.

NLO widths
^^^^^^^^^^

``compute_widths --nlo`` computes the width at NLO. A model that comes with a dedicated code for
it (``SMWidth``, for the Standard Model: NLO QCD and EW for t, W and Z, HDECAY for the Higgs)
uses it. For the others the two-body decays that have a colour charge, in the decaying particle
or in a decay product, are run at NLO in QCD with MC@NLO (aMC@NLO at fixed order, one run per
kind of decay, up to four at a time). ``--nlo=smwidth`` and ``--nlo=mcatnlo`` choose the code.
``--corrections`` chooses the corrections of the MC@NLO runs, ``QCD`` (the default), ``QED`` (the
electroweak ones, ``EW`` is accepted too) or ``QCD+QED``; the electroweak corrections are for
every two-body decay, not only the coloured ones, and ask for a model with loop information
(``loop_qcd_qed_sm``). Asking for corrections is asking for MC@NLO, as SMWidth has both.
The model needs loop information: ``sm`` is replaced by ``loop_sm``, a model that cannot make
loops stops the computation of that decay, which keeps its LO width with a warning. The decays
into three bodies or more are always at LO (madspace, or MadEvent), so that
``--nlo --body_decay=3`` is the two-body decays at NLO and the three-body ones at LO, and
``--nlo --body_decay=2`` the two-body decays alone::

    compute_widths t w+ z h --nlo --body_decay=3

The result depends on the renormalisation scheme of the model: with the pole-mass Yukawa of
``loop_sm``, ``h > b b~`` gets a correction of -35%, which is the known
``1 + 4/3 alpha_s/pi (9/4 + 3/2 log(m_b^2/m_H^2))`` and not what a physical width with a running
Yukawa has.

The options also go in the param_card, after the ``auto`` of a width::

    DECAY  6 auto --nlo --body_decay=3
    DECAY 25 auto --body_decay=2
    DECAY 23 auto@NLO

(``auto@NLO`` is the older spelling of ``auto --nlo``.) The options are those of
``compute_widths``: ``--nlo``, ``--corrections``, ``--body_decay``, ``--precision_channel``,
``--min_br`` and ``--engine``. The widths with the same options are computed together, and
``set decay 6 auto --nlo --body_decay=3`` writes the same line from the card editor.

Validation against SMWidth
^^^^^^^^^^^^^^^^^^^^^^^^^^

The two NLO codes can be run on the same model, the Standard Model with SMWidth
(``loop_qcd_qed_sm``, alpha(MZ) scheme, with its default card: ``m_t`` = 173.3 GeV,
``m_H`` = 125 GeV, ``alpha_s(m_Z)`` = 0.118)::

    import model loop_qcd_qed_sm
    compute_widths t w+ z --nlo=smwidth      # NLO QCD and EW, a few seconds
    compute_widths t w+ z --nlo=mcatnlo      # NLO QCD, 4 runs of MC@NLO, about a minute

SMWidth adds the EW corrections to the QCD ones, which the default runs of MC@NLO (QCD) do not
have, so this first comparison is made on the QCD part (the electroweak one is below): the QCD-only column comes from the width functions of SMWidth called
with the orders ``(qcd, qed) = (1, 0)``, the LO column with ``(0, 0)``. The errors of the
MC@NLO column are the statistical ones of the runs (0.1%); the LO widths of the two codes agree
to 1e-4. Widths in GeV:

.. list-table::
    :header-rows: 1
    :widths: 10 14 18 22 16 20

    * - particle
      - LO
      - SMWidth, NLO QCD
      - MC@NLO, NLO QCD
      - MC@NLO / SMWidth
      - SMWidth, NLO QCD+EW (``--nlo``)
    * - ``t``
      - 1.5462
      - 1.4136
      - 1.4130 ± 0.0020
      - 0.9996 ± 0.0014
      - 1.3673
    * - ``w+``
      - 2.1049
      - 2.1586
      - 2.1576 ± 0.0022
      - 0.9995 ± 0.0010
      - 2.0847
    * - ``z``
      - 2.5138
      - 2.5793
      - 2.5783 ± 0.0020
      - 0.9996 ± 0.0008
      - 2.4888

The QCD corrections agree to 0.05%, which is within the statistical error. The EW corrections
that SMWidth also includes are -3.3% for the top, -3.4% for the W and -3.5% for the Z, relative
to the NLO QCD width. In MC@NLO the corrections are the ``NLO/LO`` of each two-body decay:
0.914 for ``t > w+ b``, 1.038 for ``w+ > u d~`` (``1 + alpha_s(m_W)/pi`` is 1.0383 in the
dedicated code) and 1.037 for ``z > q q~``.

The Higgs is not in the table: SMWidth returns the HDECAY total (running quark masses,
and the loop-induced ``gg``, ``a a`` and ``z a``), where MC@NLO has the tree-level decays only
and the Yukawa couplings of the model.

The electroweak corrections of the two-body decays can be run with MC@NLO too::

    compute_widths t w+ z --nlo --corrections=QED       # NLO EW, 8 runs, about two minutes
    compute_widths t w+ z --nlo --corrections=QCD+QED   # NLO QCD and EW

(the 17 two-body decays of these three particles are run in 8 distinct runs: the b is not a d,
its isospin partner being the top quark, but the light quarks of a type and the leptons
and the neutrinos of the families are alike). The same model and card give, in GeV (SMWidth
called with the orders ``(0, 1)`` for the electroweak corrections alone and ``(1, 1)`` for
both):

.. list-table::
    :header-rows: 1
    :widths: 8 12 16 20 20 18 24

    * - particle
      - LO
      - SMWidth, NLO EW
      - MC@NLO, NLO EW
      - MC@NLO / SMWidth
      - SMWidth, NLO QCD+EW
      - MC@NLO, NLO QCD+EW
    * - ``t``
      - 1.5462
      - 1.5245
      - 1.5260 ± 0.0034
      - 1.0010 ± 0.0022
      - 1.3918
      - 1.3900 ± 0.0021
    * - ``w+``
      - 2.1049
      - 2.0309
      - 2.0307 ± 0.0021
      - 0.9999 ± 0.0010
      - 2.0847
      - 2.0839 ± 0.0024
    * - ``z``
      - 2.5138
      - 2.4234
      - 2.4232 ± 0.0017
      - 0.9999 ± 0.0007
      - 2.4888
      - 2.4878 ± 0.0019

The two codes agree to 0.1% on the widths with the electroweak corrections, and to 0.13% (the
top quark) or 0.04% (W and Z) with the QCD and the electroweak ones. The top quark of SMWidth
is taken without the finite width of the W that ``compute_widths --nlo`` adds to it (1.3673
GeV): it is a three-body effect, not in a two-body decay. Channel by channel the electroweak
correction ``Gamma_NLO - Gamma_LO`` of MC@NLO is, in GeV (SMWidth in parentheses),
-0.02485 (-0.02477) for ``w+ > u d~``, -0.00818 (-0.00814) for ``w+ > l+ nu``, -0.01256
(-0.01230) for ``z > u u~``, -0.01337 (-0.01335) for ``z > d d~``, -0.01977 (-0.01973) for
``z > b b~``, -0.00273 (-0.00272) for ``z > l+ l-`` and -0.00359 (-0.00374) for
``z > nu nu~``, the last two within the statistical error.

Not supported yet
-----------------

The following MadEvent features have no MG7 equivalent. Settings for them are ignored.

* Matching and merging: ``ickkw``, ``xqcut``, ``ktdurham``, ``ptlund``, ``dparameter``,
  ``highestmult``, ``ktscheme``, ``alpsfact``, ``chcluster``, ``pdfwgt``,
  ``asrwgtflavor``, ``clusinfo``, ``auto_ptj_mjj``, ``pdgs_for_merging_cut``.
* Biasing: ``bias_module`` and ``bias_parameters``.
* Beam polarization and heavy-ion or equivalent-photon beams: ``polbeam1``, ``polbeam2``,
  ``nb_proton1``, ``nb_proton2``, ``nb_neutron1``, ``nb_neutron2``, ``mass_ion1``,
  ``mass_ion2``, ``ievo_eva``, ``evaorder``, ``eva_xcut``.
* Fixed factorization scales for the two beams, and the scale parameters
  ``mue_over_ref``, ``mue_ref_fixed`` and ``fixed_extra_scale``.
* Cuts that are not a bound on a single observable: ``cutuse = 1``, which accepts an event if
  any of the ordered jet cuts passes, and the vector-boson-fusion cuts ``xetamin`` and
  ``deltaeta``, which require the two hardest jets to lie in opposite hemispheres.
* Photon isolation: ``ptgmin``, ``r0gamma``, ``xn``, ``epsgamma``, ``isoem``.
* Helicity sampling: ``nhel``, ``limhel``, ``hel_recycling``, ``hel_filtering``,
  ``hel_splitamp``, ``hel_zeroamp``.
* Event output options: ``event_norm``, ``lhe_version``, ``boost_event``, ``me_frame``,
  ``frame_id``, ``systematics_program`` and ``systematics_arguments``.
