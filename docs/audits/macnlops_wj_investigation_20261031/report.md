# Why the W+jet transfer shifts

The investigation establishes two failures of the approximations used in the
S-veto normalization: a channel-average positive S/B factor fails the sign and
shape of the signed S density near the generation cut, and the gluon-FSR
Sudakov denominator differs from the actual global-recoil first shower.
Continuation then transports the already unequal differential transfers,
mainly through ISR. The written starting-scale checks pass.

This uses the corrected 1M-event W+jet campaign at 13 TeV, NNPDF40 NLO,
generation anti-kT R=0.5 jets above 20 GeV, and stable W+. Source events,
matching decisions and original shower outputs are unchanged. Pythia source
and libraries were not modified. New controls are kept separately.

Throughout, S denotes signed weight removed by S veto and H denotes signed
weight added by negative-H removal. Cancellation requires H/S=1 and H−S=0.
The raw variable is pT(W+leading inclusive anti-kT jet), in 5–20 GeV unless
stated otherwise. The 14–20% ratios are relative to the transfers: the realized change
of the whole raw-bin cross section is +1.51% initially and +2.01% after the
original full shower. Its inclusive cross-section shift is +0.3602%.

## 1. The first-emission excess comes from the jet-cut boundary

These are expected transfers, using recorded acceptance probabilities on all
1M source events; this removes Bernoulli rejection fluctuations.

| First-emission selection | S [pb] | H [pb] | H/S | H−S [pb] |
|---|---:|---:|---:|---:|
| All generated events | 1541.556 | 1773.327 | 1.15035 | +231.772 ± 13.927 |
| Leading jet pT <20 GeV | 326.551 | 654.552 | 2.00444 | +328.000 ± 9.450 |
| Leading jet pT 20–25 GeV | 137.498 | 230.615 | 1.67722 | +93.117 ± 4.823 |
| Leading jet pT ≥30 GeV | 862.021 | 664.855 | 0.77127 | −197.166 ± 7.727 |
| Pass jet pT>25 GeV, \|y\|<4.5 | 1070.931 | 882.836 | 0.82436 | −188.096 ± 8.990 |

All original S Born jets pass pT≥20 GeV. Emission can move the endpoint below
that cut. The recorded R=0 transfer contributions exactly reproduce the
endpoint-leading-jet-pT<20 contributions: the real density fails its cut,
while the Born-supported MC subtraction remains nonzero. In that region
negative H is almost completely removed. The S transfer is much smaller.
The R>0 remainder instead has a raw-bin deficit of −96.229 pb.

At the first step, raw versus fiducial recoil definitions cause zero crossings
of this particular 5–20 bin in this sample. Its raw/fiducial reversal is due
to selection. Rapidity effects are small; the pT boundary dominates. This
first-stage fact does not identify raw/fiducial recoil definitions in general.
The nonclosure also exists in native matching-scale 5–20 GeV: H/S=1.20776.

## 2. The local S≈beta B approximation fails

The S-veto denominator uses the integrated channel ratio beta=sigma_S/sigma_B
as a local density normalization. Every recorded beta is positive, in
[0.786972,1.034160]. Reconstructing beta from the endpoint kernel/coupling/PDF
factors agrees with recorded channel values to 4.44e-16.

However, the underlying Born-jet-pT interval 20–21 GeV contains signed S
weight **−441.287 ± 13.110 pb**, from +1425.255−1866.543 pb. Its expected
signed veto transfer is **−114.866 ± 4.809 pb**. Removing this negative S
weight adds positive matched weight, in the same direction as H removal.
A positive beta B model cannot reproduce even the integrated sign there.

A separate 200k-event positive Born control, with the same model, PDF, jet
cut and H_T/2 hard-scale definition, gives the following shape comparison:

| Underlying Born jet pT [GeV] | Signed S [pb] | Positive B control [pb] | S/B |
|---|---:|---:|---:|
| 20–21 | −441.287 | 1191.450 | −0.3704 |
| 21–22 | 352.006 | 1103.244 | 0.3191 |
| 25–30 | 2899.746 | 3582.622 | 0.8094 |
| 30–40 | 4714.182 | 4487.222 | 1.0506 |
| 50–75 | 3620.518 | 2974.798 | 1.2171 |
| 100–150 | 831.878 | 638.122 | 1.3036 |

This explains the direction of the observed hard-pT distortion: the veto is
underfed near the cut and overfed in harder regions when normalized with a
channel constant. Full local densities can also depend on colour, flavour,
starting scale and source chart, beyond the one-dimensional projection here.
The comparison demonstrates failure of the approximation; it does not assign
every residual pb uniquely to it or prove loss of fixed-order NLO accuracy.

The inclusive expected mismatch +106.604 ±16.086 pb partitions into
ISR −7.197 ±13.151 pb and FSR +113.801 ±9.263 pb. Final-quark FSR contributes
+78.774 ±6.108 pb. Thus the discrepancy is broader than a Born-gluon or ISR
warning effect alone. Classification is certified by packet registry SHA,
native sector alias, and actual Born/FKS PDGs, rather than sector numbers.

The smooth source/shower Sudakov-ratio correction also matters: on identical
endpoints, omitting that correction changes raw expected S from 1541.556 to
1811.881 pb, while H changes by only +0.320 pb. Its raw H/S would be 0.97888.
This is a diagnostic, not a validated alternative matching law. The omitted
factor is needed by the intended construction; its apparent agreement can
hide the other density approximations. S-veto saturation changes less than
0.4 pb signed and cannot explain hundreds of pb.

## 3. The gluon-FSR denominator is independently inaccurate

For 96 selected S contexts, the evaluator's exact conditional starting-scale
quadrature nodes and weights were extracted. Before empirical testing,
R/D/K/source probability/shower mean were replayed. 95 contexts pass 1e-8;
one 5.05% D/K replay mismatch is excluded and retained as an open diagnostic
case. The final native test uses distinct per-context/provider/mode Pythia and
Q-sampling streams and samples that same conditional-Q mixture, retaining
Born momenta, colours and line caps. A fixed-Q comparison would be invalid.
A separate import check reproduces original source leg/dipole caps to
2.81e-13 relative. Using exact production min-leg scales instead of the
diagnostic Q-leg assignment gives bit-identical shower outputs over 1.44M
paired full/component trials. Direct native initial-dipole listings for
95 contexts at three Q nodes each (285 imports) also give zero global,
production and dipole-scale errors. The measured discrepancy survives
these import checks.

There are 5,000 unmodified stock first-shower trials per valid context:
475,000 trials in total. Empirical/predicted survival is approximately unity
for the 60 quark-Born contexts, but has substantial local deviations for the
35 Born-gluon contexts. Their selected-sample ranges are 0.927–1.054 and
0.773–1.381, respectively. These are selected-context diagnostics, not
unbiased population residual estimates.

A verified example, context 47 (Born u dbar -> W+ g, Born jet pT=20.908 GeV,
t=5.687605 GeV), predicts full survival **0.406006**. Native survival is
**2377/5000=0.475400**, Wilson 95% CI [0.46158,0.48926]. Component controls give:

| Shower component | Predicted survival | Native survival |
|---|---:|---:|
| FSR only | 0.71728 | 0.87960 ±0.00460 |
| ISR only | 0.55456 | 0.54900 ±0.00704 |

The FSR-only test has no ISR warnings. Across the 35 gluon contexts,
FSR-only controls reproduce the dominant pattern of deviations, while ISR-only survival agrees at the sampled
precision. The source code explains the approximation: the gluon
part of `macnlops_delta_profile` multiplies `macnlops_fsr_sudakov` factors
for separate connected Born dipole masses; quarks instead have the explicit
global-recoil correction `quarkFsrRate`. This demonstrates an inaccurate
gluon-FSR denominator. The tests do not isolate a unique missing boundary,
finite-kernel or table correction, and changing a ceiling alone is not a
validated fix. Conditional-Q ISR/FSR averages must not be multiplied: they
share the same Q. The full mean averages the conditional product.

## 4. Continuation transports the existing differential mismatch

All **155,756** original events with nonzero realized signed transfer were
continued independently in four controls, with common per-event random
seeds. The 844,244 omitted events have exactly zero transfer and therefore
cannot contribute to any transfer ratio or its migration. No warning-based
selection or retry is made. All modes respect every event's global cap.

| Continuation | First raw H/S | Full raw H/S | Paired change |
|---|---:|---:|---:|
| Original 1M production | 1.14344 | 1.20408 | +0.06064 ±0.00810 |
| Independent ISR+FSR replay | 1.14344 | 1.20723 | +0.06380 ±0.00845 |
| ISR only | 1.14344 | 1.19401 | +0.05057 ±0.00780 |
| FSR only | 1.14344 | 1.16157 | +0.01814 ±0.00529 |
| ISR+FSR with alpha_s(MZ)=0.1365 | 1.14344 | 1.19920 | +0.05576 ±0.00879 |

Controls are nonadditive. Matching the continuation coupling to the first
step does not remove the drift. The dominant continuation effect is ISR.
The independent full replay changes S from 1550.979 to1296.242 pb and H from
1773.450 to1564.868 pb: the S transfer contracts more strongly.

The H−S excess increases by **46.155 ±10.338 pb**. Splitting by the jet pT
BEFORE continuation, events below25 contribute −3.759 ±6.825 pb to this
change, while events above25 contribute **+49.914 ±7.765 pb**. The shower
erodes the initially negative hard-jet contribution; the low-jet-pT excess
persists. A first-emission bin average that partly cancels these populations
does not retain that cancellation after shower migration.

Direct flow accounting gives outgoing S/H=427.451/414.138 pb and incoming
S/H=172.714/205.555 pb. Their net difference is the same +46.155 pb. The ratio
change decomposes exactly into +0.03561 from net migration and +0.02819 from
the smaller S denominator. There is no new scale-import error in these tests.

## 5. Native warnings and remaining cross-checks

All 104 first-emission channels were replayed at original seeds/settings.
All 1M endpoints and scales match production bit-for-bit, reproducing 151
warnings in151 S events. Their expected S-veto transfer is just0.10587 pb;
in raw5–20 it is0.05240 pb versus231.772 pb mismatch. Directly warned records
cannot account for the first-step effect. Excluding their recorded transfer
changes realized H/S from1.143439 to1.143632; this is bookkeeping, not a
warning-free physical control.

The stock warning means an ISR trial acceptance exceeds one and is accepted;
warnings at pT≤1 GeV are suppressed. The preceding small tagged contribution
is not a bound on the entire proposal bias or changed survival/random paths.
Continuation has the same caveat. Sampling uncertainties do not include it.

A separate Born-normalization cross-check remains open: the positive LO
control totals20681.30±8.99 pb versus native res_1 Born integrals20845.70±12.53
pb. Correct 104-to-12 channel deduplication was verified. The discrepancy is
localized mainly to P0_dxg_wpux/GF2; other physical Born classes agree with
the control. The corresponding S/B ratios nearly cancel this normalization
shift (mirrored GF2 beta differs only0.242%), so it does not explain the
15–30% transfer-shape discrepancy alone. Full per-directory NLO rates are
not physical mirror tests after H-owner repartition. No deterministic
source/PDF bug was established. The LO shape plot retains its actual
normalization; it is not silently rescaled to the native integral.

## Consequences for the next implementation

The matching density needs attention. A negative local S/B value cannot be
inserted into the current positive bounded-veto denominator. A consistent
construction must represent the local signed S density, or its positive and
negative components, and use the same transfer density on H and S. It should
condition on the relevant Born state, source starting scale and flow rather
than a single integrated channel beta. Any revised source H/S split belongs
before unweighting.

The gluon shower denominator also needs the actual global-recoil first-step
law, checked against native conditional-Q survival. Its existing tables can
remain the definition of the source damping; their use for the shower
normalization needs a correction or replacement. No Pythia modification is
required to measure or correct this interface-level approximation.

A useful fresh-source control is to lower the generation cut well below the
fixed25/40 GeV analysis cuts, then inspect transfer closure differentially
in Born jet pT and emission scale. That tests boundary sensitivity, but by
itself does not fix the local-density or gluon-Sudakov approximation. This
investigation has made no production matching/source change.

## Evidence

`first/` retains the all-event signed, kinematic, registry, beta, R=0,
clipping and Born-normalization audits. `kernel/` retains the independent
Born control and provenance. `warnings/` retains exact endpoint replays,
native-source warning semantics, conditional-Q tests and provider bindings.
`continuation/` retains four replay controls, full migration accounting and
input coverage. `plots/wj_investigation.pdf` summarizes the three main
comparisons; `warnings/conditional_sudakov.pdf` shows the empirical CDF tests.

Quoted errors are conditional finite-event/shower estimates. They exclude
hard-integration, channel-beta, context-selection and native-envelope
uncertainties where relevant. Original raw data and all new diagnostic
scripts/outputs remain under:
`/export/slow1/rikkert/macnlops_native_ir_jets_1M_20261031/investigation_wj`.
