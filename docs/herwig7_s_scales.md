# Herwig7 S-event scales

Ordinary MC@NLO with `parton_shower = HERWIG7` now draws one damped
hard-process scale for S-event `SCALUP`. The hard reference, damping
function, scale factor and infrared floor are the same as in the Pythia8
prescription. `Template/NLO/SubProcesses/herwig7_scales.f90` calculates
the corresponding directed transverse-momentum envelopes from Herwig's
angular starting scales. It has no dependency on Herwig or ThePEG.
The detailed formulas and interface are in
[herwig7_scales.README](../Template/NLO/SubProcesses/herwig7_scales.README).

The shower selector is `HERWIG7` and its installation setting is
`herwig7_path`. Legacy run cards selecting `HERWIGPP` or `HERWIG++`
(case insensitive) are automatically converted to `HERWIG7` when read.
The same conversion applies when editing `parton_shower` with `set`.

Herwig evolves in an angular variable, while `SCALUP` sets a transverse-
momentum veto. The existing MG5 steering already configures this with
`MaxPtIsMuF Yes` and `RestrictPhasespace Yes` (or the Herwig 7.0
`HardVetoMode`/`HardVetoScaleSource` switches). The angular limits remain
independent of the scalar veto. For example, a massless outgoing emitter
has an envelope `pT_max = qtilde_max/4`. The module also handles massive
outgoing emitters and II, IF and FI connections. It follows the symmetric
angular partition in [Gieseke, Stephens and Webber, JHEP 12 (2003) 045](https://arxiv.org/abs/hep-ph/0310083).

The subtraction damping interval belongs to the hard-scale draw. It is
not rescaled separately for each dipole; the existing angular dead-zone
test remains separate. The saved scalar follows the selected sector and
fold and is written without per-dipole tags. Born-only events use the
undamped hard reference. The prescription applies to two incoming legs
and excludes FxFx and Delta. Delta remains a Pythia8-only mode.

H-event formulas are unchanged, but their values can change indirectly:
the ordinary H prescription takes the maximum of its real-event scale
and an underlying S-event dipole scale. That latter input now comes from
the Herwig module, just as it does from the Pythia8 module for Pythia8.

## Validation on 28 September 2026

* All 94 focused tests passed: scale reconstruction, sampling, subtraction,
  angular support, native histories, colour assignment, momentum maps,
  Born support, helicity selection, PDF cache, clustering and Pythia8
  interfaces. The new tests check that a soft wide-angle emission with
  `qtilde > SCALUP` and `pT < SCALUP` remains allowed.
* A direct comparison linked against the installed Herwig 7.3.0
  `PartnerFinder` and `KinematicHelpers` implementations. It evaluated
  3,000 matrices / 36,000 directed connections, including massless and
  unequal massive emitters, all connection types and longitudinal boosts.
  The reference pT maxima were obtained by numerically maximizing
  Herwig's branching formula. The largest discrepancy was `6.38e-13`
  using `abs(Fortran-Herwig)/max(1,abs(Herwig))`, for scales in GeV.
* A fresh `p p > e+ e- [QCD]` export generated 40 events: 31 S and 9 H.
  All subprocess matrix-element/MC and pole checks passed. The events
  have finite positive `SCALUP`, conserved colour and momentum, 27 finite
  scale reweights each, and no dipole tags. S-event scales span
  10.62--92.51 GeV. Herwig 7.3.0 showered and hadronized all 40 events;
  its log reports no exceptions, and the HepMC file contains 40 events.
* Broader process generation exposed existing limitations in this branch.
  `p p > t t~ [QCD]` stops at the unconditional `FIX AP REDUCED MASSIVE`
  placeholder in `AP_reduced_massive` in `fks_singular.f`.
  `u u~ > g g [QCD]` stops at `Invalid native MC H radiation projection`;
  the same failure was reproduced after restoring the pre-change scale
  and event-writing code from commit `01182c1c58`. These paths were not
  altered by this change. Massive and final-state scale reconstruction
  are covered by the standalone and direct Herwig comparisons above.

These are numerical and execution checks, not a precision cross-section
comparison. Cards, logs, comparison wrappers, JSON results, LHE and
HepMC files are retained in
`/export/tmp/rikkert/mg5_herwig_scales_s3ej165h`.

## Interface rename validation

After renaming the selectors, configuration, scripts, Fortran routines
and analysis files to Herwig7:

* All 139 focused unit tests and the configuration acceptance test passed.
  The compatibility test covers legacy run-card values, including mixed
  case and quotes, and checks both card and Fortran output.
* A fresh `p p > e+ e- [QCD]` export with `herwigpp` in the run card
  generated `SHOWER_MC = 'HERWIG7'` and 40 events (31 S, 9 H). All
  matrix-element, matching and pole checks passed. Herwig 7.3.0 showered
  and hadronized all 40 events with no exceptions, producing
  `events_HERWIG7_0.hepmc.gz` with 40 entries.
* A Fortran counting analysis loaded through `Herwig7Analyzer` processed
  all 40 events and produced a nonzero HwU histogram. The C++ interface
  now explicitly includes ThePEG's `HepMCConverter.h`, and the Makefile
  links libraries after the objects so `--as-needed` retains HepMCfio.
* The existing rates example assumes the first two HEPEVT entries are
  the incoming partons. That assumption fails for this Herwig7 sample,
  causing it to reject all events with `WARNING 111 IN HWANAL`. Its
  analysis logic was not changed as part of the rename; the counting
  analysis used for interface validation does not make that assumption.

The cards, logs and validation analysis are retained in
`/export/tmp/rikkert/mg5_herwig7_rename_lsdjmd8_`.

## Local shower installation

Herwig 7.3.0 and ThePEG 2.3.0 are installed together under
`/export/tmp/rikkert/Herwig`, using the existing HepMC, LHAPDF and FastJet
libraries. The installation includes an environment script:

```sh
source /export/tmp/rikkert/Herwig/activate.sh
Herwig --version
```

MG5 configuration paths for this installation are:

```text
herwig7_path = /export/tmp/rikkert/Herwig
thepeg_path = /export/tmp/rikkert/Herwig
hepmc_path = /export/tmp/rikkert/HepMC
```

They were supplied to the validation run; no global MG5 configuration was
changed. The installation's `INSTALLATION.md` records its dependencies.
