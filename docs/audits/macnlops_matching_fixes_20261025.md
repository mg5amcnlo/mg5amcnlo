# Matching failures reproduced on 25 October 2026

These fixes use the sampled kl/ij, fixed-colour-flow campaign in
`production_1M_sampled`. They change the branch implementation; the completed
or partial production samples and their checkpoints have not been overwritten
or certified as rematched samples. No new events or shower histories were
generated for these checks. Installed Pythia libraries were not modified.

## Massive-FSR Born validation

All 65 ttbar, 2 Wbb and 40 single-top native status-38 failures compared a
nonzero source Born momentum fraction against `xbjrk_cnt(:,0)=0`. The massive
FSR second solution has a valid Born state but deliberately has no soft
counterevent. Its ordered Born Gram-invariant checks passed; the counterevent
fraction was the wrong field to validate.

The density driver now requests the existing native phase-space snapshot
only when a component Born check is needed, and compares against its immutable
`born%xbjrk`. Colour, flavour, on-shell and ordered-invariant checks remain
active. A compiled fixture covers missing counterevents and rejects a changed
Born geometry or fraction.

## Signed PDF luminosities

The DY Delta failure is saved event 1028 of
`P0_dxd_emep/GF2.0_24`. Its incoming anti-strange has x=0.8134114253.
MSTW2008nlo90cl/0 gives x*sbar about -2.22e-7 at Q=50.18 GeV there.
The native density response is

```
R = -180137.16691678748
D = -186101.79956101454
K = -186101.79956101452
```

The former unconditional positivity check rejected this signed luminosity.
The signed-residual prescription now uses

```
u = max(D - R, 0)
transfer = min(u, max(K, 0))
S acceptance = 1 - transfer/K  if K > 0, otherwise 1
negative-H acceptance = 1 - transfer/u  if u > 0
```

Nonpositive K therefore leaves S and any negative-H residual unchanged. This
is a conservative zero transfer, not an absolute-valued veto. Negative R with
positive K is permitted when the bounded transfer can compensate it. All
inputs and the reconstructed hard remainder must be finite. An input negative
H whose reconstructed remainder is not negative still fails. The strict API
mode retains its positivity/envelope requirements.

For multiple equivalent daughter orientations, the common luminosity sign
cancels in their auxiliary kernel ratios. The physical density remains signed;
negative kernels cannot transfer negative H. Incompatible signs are rejected
because they do not define this certified common-sign marking law. This rule
does not classify an S event by its nominal weight: NLO virtual contributions
can make that weight negative even when K is positive.

## Single-top H-sign reconstruction

Two further negative H records, events 5240 and 4240 in `P0_ub_td/GF1.0_6`
and `GF1.0_39`, reconstructed a positive H. In both, the one-particle recoil is
a massless down quark. Subtracting boosted incoming/emitter momenta and then
squaring produced fictitious spectator masses squared of -2.97e-9 and
-3.29e-9 GeV^2. They exceed the support helper's 64-epsilon*shat roundoff guard
(about 82 and 92 epsilon*shat respectively). The MC kernel incorrectly became
zero, losing the source subtraction.

For exactly one physical spectator, the MC kinematics now use its explicit
four-vector and its on-shell model mass. Composite recoil and the strict
spacelike support guard remain as before; empty legs outside a selected
resonance subsystem do not count as spectators. The exact-mass state is reset
for every point, including ISR. No source event is modified.

| Source H record | Reconstructed H before | Reconstructed H after | K after |
|---|---:|---:|---:|
| GF1.0_6 / 5240 | +5.4475700e6 | -3.5739534e7 | 6.6948733e8 |
| GF1.0_39 / 4240 | +8.1532668e7 | -7.0448827e7 | 1.5243296e8 |

These are native density units, not integrated cross sections. Both records
now have the correct negative sign and fit inside the available positive K.

## W+jet Delta checks

The W+jet Delta run completed its generation while these fixes were being
checked. It exposed 23 status-11 failures and two hard-input preflight failures.
For the former, the source MC routine returns without reading Delta when both
FKS partitions vanish. The local evaluator had nevertheless invented a unit
replay row and then rejected it as unused. It now creates that row under the
same nonzero-partition condition as the source. Missing/repeated keys and
unconsumed genuine replay records remain errors.

The two boosted hard inputs, events 1438 and 4270 of `P0_ug_wpd/GF2.0_5` and
`GF2.0_13`, exceeded the annotated shell roundoff budget by factors 1.78 and
1.27. Their largest shell residuals were +3.75e-5 and -1.32e-5 GeV^2 on
multi-TeV nominally massless legs. Annotated pre-shuffle inputs now receive a
16384-epsilon*eventScale^2 budget for accumulated boost roundoff. Plain LHE
and shower endpoints keep the 4096 budget. No momenta or declared masses are
changed. A standalone compiled preflight test accepts both source points and
still rejects plain inputs at that precision, larger off-shell errors, and
physical charm-mass shifts.

## Verification and its limits

`check_macnlops_matching_failures.py` replays one exact saved density failure
per channel. It compiles only temporary current driver/counterterm objects,
links existing generated objects, and leaves source cards/objects/samples
untouched. Every numerical Delta call is replaced by an abort, checking that
the local unit-Delta construction remains free of numerical Delta evaluation.
All **133** cases passed: DY 1, ttbar 65, Wbb 2, single top 42, W+jet 23.
The report is `macnlops_matching_failures_20261025.json`. The two W+jet
preflight points are in `tests/input_files/macnlops_hard_preflight_20261025.json`.

Focused Python and compiled Fortran/C++ tests cover signed transfers and
unchanged positive-kernel formulas; common-sign orientation probabilities;
Born identity and missing counterevents; single/massive/composite recoils;
and existing shower support and mass-shell controls. No full campaign,
distribution comparison or full-shower rerun is claimed by these checks.
Subsequent points in a formerly aborted channel still require normal matching
validation. Existing generated outputs retain their original implementation;
they need an explicitly qualified rematching build before resuming production.

Example bounded replay command (serial, using existing process exports):

```bash
taskset -c 6 python docs/audits/check_macnlops_matching_failures.py \
  --output production_1M_sampled/diagnostics_matching/fixed_failure_replay.json
```
