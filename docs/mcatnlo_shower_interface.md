# MC@NLO shower interface

This note describes how events generated in aMC@NLO mode are showered, and
how the shower is kept consistent with the MC counterterms of the
short-distance cross section. User documentation is in
`Template/NLO/MCatNLO/README.shower`.

## Flow

1. Event generation (`reweight_and_collect_events`): before the events are
   collected, `mcatnlo_shower.contract_from_generation` records the matching
   choices in an `<MCatNLOMatching>` block of the run banner, which
   `collect_events.py` copies into the header of `events.lhe.gz`
   (`MCatNLOMatching` is in `template_header_keep_tags`; `banner.py` knows the
   tag, so MadSpin and reweighting keep it).
2. Showering (`run_mcatnlo`, also reached by `./bin/shower` and
   `./bin/generate_events` with the shower on):
   * `get_matching_contract` reads the block. For older event files,
     `legacy_contract` rebuilds it from the banner, the current cards and a
     scan of the event file, with a warning.
   * PYTHIA8: `prepare_pythia8_shower` builds the command file with
     `mcatnlo_shower.pythia8_settings`, and the driver with
     `build_pythia8_driver` (cached in `MCatNLO/build/PYTHIA8_<hash>`).
     The run directory receives `Pythia8.exe`, `Pythia8.cmd` and
     `mcatnlo_locked.dat`.
   * HERWIG6, PYTHIA6Q, PYTHIA6PT, HERWIG7: `banner_to_mcatnlo` writes
     `MCatNLO/banner.dat`, and `MCatNLO_MadFKS.inputs` sources the shell
     script `Scripts/MCatNLO_MadFKS_<shower>.Script`.
   * `shower_template.sh` runs each (split) job; `check_pythia8_logs`
     reports what the Pythia8 driver did.

## The matching contract

| Field | Meaning |
|---|---|
| `parton_shower` | shower of the MC counterterms; must match the run card |
| `mcatnlo_delta` | MC@NLO-Delta: per-dipole starting scales in the LHE file |
| `ickkw`, `ptj` | FxFx merging and its matrix-element cut |
| `qed_matching` | QED splittings are matched (`splitting_types` contains QED) |
| `pythia8_mmaxgamma` | `#Pythia8MMaxGamma` of `FKS_params.dat` |
| `fxfx_pythia8_finite_terms` | FxFx Sudakov finite terms (forced for PYTHIA8) |
| `born_coloured_partons` | coloured final-state particles of the Born production system (`TimeShower:nPartonsInBorn`); -1 if mixed |
| `nlo_resonance_decays` | resonances written in the events whose decay is part of the NLO calculation (coloured daughters, or charged ones with QED matching) |
| `npnlo_attribute` | the events carry the `npNLO` attribute |
| `fks_params_md5` | to notice cards changed after the generation |
| `legacy` | rebuilt for an old event file |

The last three event-dependent entries come from `scan_events`, which reads
the beginning of every event file produced by the integration channels.

## Pythia8 settings

`Pythia8Settings` holds three tiers:

* `INTERNAL`: run control and event input, never changed by the user;
* `LOCKED`: required by the PYTHIA8 MC counterterms (starting scales,
  global-recoil scheme, recoil dead cone, `dipoleRecoil`, `rapidityOrder`,
  MEC policy, Delta, FxFx, QED matching, masses, beams). They are written
  last in the command file and listed in `mcatnlo_locked.dat`; the driver
  checks them after `Pythia::init()` and exits with status 3 on a mismatch.
  `pythia8_options` may override them only with
  `allow_unsafe_shower_options = T`;
* `FREE`: defaults (couplings, PDF, MPI, hadronisation, decays, seeds).

Required settings unknown to the installed Pythia8 version stop the run
(per-feature gating, from the `xmldoc` of the installation); unknown
defaults are dropped with a note. Matrix-element corrections follow
arXiv:2308.06389.

To add a requirement of the counterterms, lock it in `pythia8_settings`
with a `reason`, and add a test in
`tests/unit_tests/various/test_mcatnlo_shower.py`.

## Pythia8 driver

`Template/NLO/MCatNLO/pythia8/mcatnlo_pythia8.cc`, built with
`pythia8/Makefile` from `pythia8-config`:

* `MG5_ANALYSIS`: the Fortran analysis (`pyabeg`/`pyanal`/`pyaend`) reads
  `/HEPEVT/`, filled directly from the Pythia8 event record (no HepMC);
* `MG5_HEPMC=3|2`: HepMC2 ascii events through HepMC3 or HepMC2. All the
  weights (nominal and variations) follow `event_norm` of the run card:
  unchanged for `average` and `bias` (they average to the cross section),
  rescaled by nTotal/nShower for `sum` (they sum to it). Each event also
  carries the running estimate of the total cross section (pb);
* it reads `Main:spareMode1` events of the LHE file (vetoed and failed
  events included, so that the normalisation is that of the events read)
  and prints `MG5_MCATNLO_CONTRACT` and `MG5_MCATNLO_SUMMARY` lines.

Exit status: 0 success, 1 set-up error, 2 too many failed events,
3 required settings changed.
