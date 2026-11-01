# W+jet analysis cut: 25 versus 40 GeV

Raising the analysis jet threshold to 40 GeV **does not improve the signed
H-removal/S-veto balance** in the original W+jet sample. The absolute
cross-section deficit becomes smaller, while its size relative to the
selected source cross section increases. Most of the imbalance is already
present after the first emission.

## Scope

This is an analysis-only comparison on the same **1,000,000** W+jet inputs
used in the [original multijet campaign](../macnlops_native_ir_jets_1M_20261031/report.txt).
Generation remains standalone `ickkw=0`, pp at 13 TeV with a stable W+,
anti-kT R=0.5 generation jets above 20 GeV and no generation rapidity cut.
The fiducial analysis requires at least one anti-kT R=0.5 jet with
`pT > 25` or `pT > 40 GeV` and `|y| < 4.5`. The cut is reapplied at each
stage. Recoil is `pT(W + leading jet passing the fiducial cuts)`.
The highlighted 5–20 GeV interval is recoil, not jet transverse momentum.

Source events, signed weights, matching decisions, continuation scales,
shower settings and original random streams are fixed. **This retains the
original matching epoch, before the global-gluon denominator correction.**
It measures the analysis-cut effect and does not measure the effect of that
later correction. Pythia source and installed libraries are unchanged.

## Recoil 5–20 GeV

Unity of H-removal/net S-veto indicates transfer cancellation. The
cross-section shifts are `(matched/source - 1)` on the same selected events.

| Analysis jet cut | First H/S | Full H/S | First cross-section shift | Full cross-section shift |
|---|---:|---:|---:|---:|
| 25 GeV | 0.82810 ± 0.00918 | 0.83002 ± 0.00913 | -1.656% | -1.533% |
| 40 GeV | 0.73359 ± 0.01003 | 0.74056 ± 0.01253 | -2.972% | -2.580% |

All table uncertainties use the same 50 paired contiguous 20k-event
source/shower chunks. They are conditional sampling estimates, excluding
hard-integration, channel-beta and native ISR-envelope uncertainties.
The historical eventwise first-emission errors, also retained in the
numerical records, are 0.00997 and 0.01252 for 25 and 40 GeV respectively.
No surviving-event, fiducial or displayed-range normalization is applied.

| Stage | Jet cut | Source [pb] | Matched [pb] | Signed S veto [pb] | H removal [pb] | H−S [pb] |
|---|---:|---:|---:|---:|---:|---:|
| First emission | 25 GeV | 11065.387 | 10882.127 | 1066.096 | 882.836 | -183.261 ± 10.842 |
| Full shower | 25 GeV | 8873.883 | 8737.874 | 800.133 | 664.124 | -136.010 ± 8.120 |
| First emission | 40 GeV | 4513.483 | 4379.353 | 503.471 | 369.341 | -134.130 ± 5.819 |
| Full shower | 40 GeV | 3668.709 | 3574.051 | 364.850 | 270.192 | -94.659 ± 5.400 |

The paired change of H/S on raising the cut is
**−0.09451 ± 0.00924** initially and **−0.08946 ± 0.01017** after showering.
At 40 GeV, the full-minus-first ratio change is **+0.00697 ± 0.00976**.
There is no resolved additional ratio drift there. The absolute net transfer
still changes by **+39.47 ± 4.91 pb**, reflecting shower migration and
contraction of both transfers; a stable ratio does not imply no migration.
Conditional expected-transfer controls, averaging over the independent
matching-decision draw, confirm the same pattern: at 40 GeV H/S is 0.73306
initially and 0.74004 after showering.

## Entire fiducial selection and negative S events

Without restricting recoil, the matched/source shifts at 25→40 GeV are
**−1.844%→−2.721%** initially and **−1.954%→−2.724%** after showering.
Selected event counts fall **668,457→315,349** initially and
**598,980→285,372** after showering.

Among all selected S events, the negative-weight event fraction falls
**18.85%→12.61%** initially and **18.16%→12.32%** after showering. Counts
and signed/absolute weights are retained separately in `comparison.json`.
Reducing that population therefore does not, by itself, repair the transfer
imbalance. This cut study does not uniquely attribute the remaining deficit
to one approximation, and does not test the later gluon correction.

## Verification and reproduction

- All 1M first-emission events were reclustered independently. Original
  weights, event kinds, raw observables, 25 GeV selection and recoil are
  reproduced bit-for-bit. A separate two-parton kinematic reconstruction
  agrees with the new 40 GeV selection and selected recoils.
- Complete full-shower momenta were not retained originally. All 50 original
  20k-event jobs were replayed in isolated directories, preserving their
  settings and random streams. **Every original 25 GeV HwU file is
  byte-identical**, including every bin, error and all four weights.
- Both thresholds are evaluated on exactly the same final particles. The
  40 GeV selection is nested within the 25 GeV selection; surviving events
  have identical fiducial recoil. All source indices and four weights match.
- The four weights satisfy `veto + removal = source + matched` event by
  event. Inclusive weight sums are preserved; all cuts use absolute signed
  weights. The independent review retains cut and first/full covariance.

The [two-page PDF](plots/wj_analysis_pt40.pdf) contains the four-weight recoil
distributions, matched/source ratios and signed transfer comparisons.
`comparison.json` contains the scalar results and their paired errors.
Scripts and build/input provenance are retained alongside this report;
`snapshot.json` pins the compact files and external per-event records.

Complete inputs, isolated builds, 1M-event arrays, original-job comparison
files and analysis outputs remain at
`/export/slow1/rikkert/macnlops_native_ir_jets_1M_20261031/wj_analysis_pt40_20261031`.
