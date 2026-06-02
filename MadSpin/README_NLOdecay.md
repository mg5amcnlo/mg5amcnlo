# MadSpin NLO decay (`decay_precision = NLO`) — WIP branch notes

This branch (`madspin_nlodecay`, based on `madspin_density`) is a **work in
progress checkpoint**. It adds the plumbing for generating the decay events of
a MadSpin run at NLO accuracy, but it is **not functional end to end**: NLO+PS
unweighting of a 1 → N decay hits a structural wall in the MC@NLO/FKS machinery
(see "Known blocker" below). It is pushed as a record of what was done so the
NLO+PS-for-decay question can be solved on a separate branch and then merged
back here.

## Goal

Add a MadSpin option `decay_precision` (`LO` default, `NLO`). When set to `NLO`
and a density-based spin-correlation mode is active, the decay-event sample
used by the density method is generated at NLO accuracy (`[QCD]`) instead of
LO. Because an NLO+PS event may carry one extra parton, each decay channel
needs *two* density matrices (e.g. `Z > q q~` **and** `Z > q q~ g`); the right
multiplicity is selected automatically per event via the event tag.

## What was implemented in this branch

### MadSpin option and validation
- `MadSpin/interface_madspin.py`
  - New option `decay_precision` (`LO`/`NLO`) in `MadSpinOptions`.
  - `do_set` validation accepts only `LO`/`NLO` (upper-cased).
  - In `check_param_consistency`, `decay_precision=NLO` is rejected unless a
    density-capable spin mode is active (spinmode `PA`; `full` with
    `ME_mode` in {auto, density}; `onshell` with `ME_mode` = density).
    These are the modes that route through `run_onshell(density_method=True)`.
  - `generate_events`: when `decay_precision=NLO`, the decay process is
    generated with a ` [QCD]` tag and the generation commands are recorded in
    the command history (`precmd=True`) because the FKS exporter reconstructs
    the perturbation order from `history.get('generate')`.
  - New `_generate_nlo_decay_events()` drives the aMC@NLO machinery
    (`aMCatNLOCmdShell`) and runs `generate_events -p -n run_01 -f`, i.e. the
    *noshower* configuration (order = NLO, fixed_order = OFF, shower = OFF), to
    produce parton-level unweighted `events.lhe.gz` ready for showering.
  - `decay_precision=NLO` is refused in gridpack (`ms_dir`) mode.

### Real-emission decay amplitudes
- `MadSpin/decay.py`
  - `get_decay_command` additionally generates the real-emission decay matrix
    elements (e.g. `Z > q q~ g`) via
    `fks_common.find_pert_particles_interactions(model, pert_order='QCD')`
    (`soft_particles`, label `pert_QCD`).
  - New static helper `add_radiation_to_decay(proc, pert_label)` appends the
    perturbation label and bumps the coupling order (`QCD=` / `QCD<=`) so the
    real-emission amplitude is not killed by an inherited Born `QCD=0`.

### Restriction bypasses (experimental, loud warnings)
These were added **purely to let the NLO+PS path run for testing**; they make
no physics claim and are gated to fire only for the 1 → N case:
- `madgraph/interface/amcatnlo_run_interface.py` (`run`, ~L1982): the
  `Decay processes can only be run at fixed order` error for
  `aMC@NLO/aMC@LO/noshower/noshowerLO` is downgraded to a loud warning.
- `Template/NLO/SubProcesses/driver_mintMC.f` (~L101): the `nincoming.ne.2`
  `stop 1` (*"Decay processes not supported for event generation"*) is
  downgraded to a loud warning.
- `Template/NLO/SubProcesses/montecarlocounter.f` (`assign_emsca`, ~L4540): the
  `shat == 2·p1·p2` consistency check (false for a decay, where
  `shat = m²` of the decaying particle) is skipped when `nincoming != 2`.

### Event parsing
- `madgraph/various/lhe_parser.py` (`Event.add_decay_to_particle`, ~L2347):
  tolerate a decay record with no explicit mother (flat 1 → N record); the
  product is attached directly to the decaying particle. (Kept from the
  earlier fixed-order LHE experiment; may be unnecessary for the noshower
  path — revisit when the path is made to work.)

### Tests / CI
- `tests/acceptance_tests/test_cmd_amcatnlo.py::test_madspin_density_decay_atNLO`:
  `p p > z j`, shower off, MadSpin `decay z > j j` with
  `set decay_precision=NLO`; checks the decayed LHE exists and per-event the Z
  decayed into 2–3 coloured daughters with mass in the BW window.
- `.github/workflows/acceptancetest.yml`: CI job running the above (installs
  meson/ninja).

## Known blocker (why this is WIP)

Bypassing the *guards* above is not enough: the MC@NLO matching/subtraction is
structurally a 2 → N construction. Driving `z > u u~ [QCD]` through the
noshower generator, after the three bypasses above the run stops in
`montecarlocounter.f` `check_invariants` (ileg = 4, "imprecision 6"):

```
|xq1q + 2·p1·k1 − xm12| / sh  ≈  1.99      (tolerance 1e-5)
Error: 10 imprecisions. Stopping...
```

This is **not** a rounding guard. The FKS invariants (`xtk, xuk, xq1q, xq2q`,
…) are *defined* relative to the incoming partons `p1, p2`; for a 1 → N decay
that parametrization does not map and the reconstructed invariants are wrong by
order unity. The MC@NLO subtraction counterterms consume exactly these
invariants, so making NLO+PS unweighting work for a decay requires reworking
the FKS counterterm kinematics in `montecarlocounter.f`, not adding more
guards. That work is deferred to a dedicated branch.

## How to resume

1. On a separate branch, define the FKS/MC-counterterm kinematic mapping for
   1 → N decays (the `ileg`, `xtk/xuk/xq1q/xq2q`, `assign_emsca` scales).
2. Replace the experimental bypasses here with the proper treatment.
3. Re-enable / tighten `test_madspin_density_decay_atNLO`.

## Reproducing the blocker

The acceptance test deletes its temp dir; to inspect the fortran logs, generate
a standalone decay process `z > u u~ QED=2 QCD=0 [QCD]`, `output` it, then run
`generate_events -p -n run_01 -f` through `aMCatNLOCmdShell` and read
`SubProcesses/P0_z_uux/GF1.0/log.txt`.
