# General native Born-scale correction

Implemented the missing Born-reference correction in the **default v4 matcher**. The prior Wbb investigation had only evaluated this change diagnostically. The production implementation now restores the canonical Born coupling and hard-PDF reference in the common S/H transfer denominator, without assuming a process-wide alpha_s power or a particular scale formula.

## Implementation

The native evaluator computes the same selected-flow scalar Born contribution twice: once using the canonical Born reference scales and once using the real-point reference scales. Both evaluations use native model coupling updates, the native split Born amplitudes and each order's custom scale prefactor. This covers Born QCD order zero, ordinary QCD Born processes, effective couplings such as HEFT, and mixed Born coupling orders.

The resulting coupling ratio is multiplied by the ratio of the two canonical Born-flavour hard PDFs. The native Born momentum fractions and flavours are used, including the spectator and flavour-changing ISR histories. Bare fixed lepton beams use unit PDF factors without a PDF lookup. Fixed and dynamical scales go through the existing native scale functions; the implementation does not assume H_T/2.

The raw kernel and the source use the existing weight-line convention: the first stored factorization scale is applied to both beams. The new ratio and default-v4 kernel-PDF metadata consistently follow that actual convention, including asymmetric requested beam-scale multipliers. The source's PDF convention is not changed. Specialized UPC/dressed-lepton beam modes remain explicitly unsupported by the new native reference.

If G is the product of these two native ratios, both event classes use the same corrected density `F = beta_c * K_sh * G * P`. The source-limited transfer and its saturation bound are unchanged. Finite nonpositive Born-reference support retains both original contributions; missing or nonfinite reference data fails explicitly. The native real CMS, scales, Bjorken fractions, coupling and Born cache are restored after the ratio evaluation.

Decision XML exposes `born_scale_factor`, `born_coupling_factor` and `born_pdf_factor`. Corrected headers and summaries identify `born_scale_correction="native-born-v1"`; channel checking prevents collection of mixed corrected/uncorrected channels. The handoff contract exposes that marker. Explicit v2/v3 modes retain their historical computation.

## Verification

**138 distinct relevant tests pass, with no skips**, including 19 new tests for the native reference, common S/H bound and collection compatibility. Coverage includes native Born powers 0/1/2, effective and mixed couplings, custom running prefactors, canonical boost-dependent scales, both PDF beams, fixed scales, no-PDF lepton beams, zero/invalid support, native-state restoration and legacy independence. An active-bound fixture verifies that changing G changes H removal when the common density bound requires it.

The initial regression run passed137 checks and exposed a missing declaration in an older compiled test fixture after adding the new request-reset state. The fixture was updated and now also checks that the Born reference resets; all four tests in that module pass on the unchanged production sources. Both logs and the final composite verdict are retained rather than discarding the initial result.

Native first-step replays compare the pre-correction and corrected matcher on identical source inputs and seeds:

| Process | Replayed events |
|---|---:|
| e+e -> 2 jets | 1,163 |
| e+e -> 3 jets | 853 |
| e+e -> 4 jets | 2,223 |
| pp -> W+ jet | 3,600 |
| pp -> W+ bbbar, 4FS | 11,973 |
| Total | 19,812 |

Every physical/source event payload, nominal weight and continuation cap is byte-identical. The three fixed-scale e+e controls have G=1 exactly and identical decisions. Wbb includes the complete formerly failing9,973-event shard: protected point5294 remains accepted with its original PDF-support rule. The recorded new kernel scale equals the old scale times G. Wbb and e+e R/D/K/source factors agree exactly; W+jet has only native cache roundoff, at most6.43e−16 relative in the checked densities and markings.

Additional direct native evaluations cover **1,700 eligible frozen endpoints**: DY441, HEFT Higgs321 and ttbar938. These use1,500 retained archived input records per process; their raw pre-shower channels were removed by earlier campaign cleanup, so these are endpoint/provider checks rather than fresh shower runs. First R/D/K, kernel coupling and source Sudakov values reproduce archived values exactly. Repeated-point sequences show pre-existing cache roundoff of at most3.23e−15 relative; no-G controls establish that classification.

The implementation also reproduces 5,616 independently calculated Wbb S factors within 3.54e−10 relative, limited by the earlier diagnostic hard-scale reconstruction. HEFT independently verifies the native effective-coupling ratio to6.7e−16 relative and its canonical Born hard-PDF ratio to3.4e−13. DY has Born coupling ratio exactly one and nontrivial PDF ratios. The native tests span eight processes in total. Isolated clone dependency updates needed for older exports are recorded; original production trees and archived results were not modified.

## Scope and provenance

This fixes the missing Born-scale term. The channel-average approximation `S ≈ beta_c * B` remains; no local-density fit from the Wbb investigation is introduced. These are implementation and controlled replay checks, not a new1M-event production or full-shower validation campaign.

Pythia source/library and the original generated source weights are unchanged. Applying the new matching to an existing source requires rebuilding the native matcher and rerunning its matching probabilities; the archived1M validation output is not retroactively replaced.

The compact record includes source snapshots/patch, focused tests, regression logs, native replay summaries, direct-point controls and provenance. Full isolated inputs, binaries and per-event records remain at `/export/slow1/rikkert/macnlops_born_scale_fix_20261101`; `snapshot.json` pins the compact files and retained external evidence.
