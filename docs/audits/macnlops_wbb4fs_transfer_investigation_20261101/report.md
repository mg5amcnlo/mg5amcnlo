# Why Wbb H-removal and S-veto differ

The no-cut Wbb mismatch is predominantly a **first-emission density-normalization problem**. A concrete missing Born-scale factor explains part of it; the use of one channel-average S/B factor explains a substantial additional part. Native shower-scale import, the tested Sudakov denominator, massive-bottom splitting shape and inverse FSR Jacobian do not show an error large enough to explain the observed deficit.

These conclusions use the frozen 1M-event four-flavour sample, a new independent positive 200k-event Born control, native shower probes and diagnostic probability reweightings. No production source, matching code, event, seed, Pythia library or acceptance cut was changed during this investigation. The production probabilities still have the mismatch reported in the [validation](../macnlops_wbb4fs_nocuts_1M_20261101/report.md).

## Quantitative diagnosis

The table uses **expected transfers**, averaging over the independent matching-decision draw on the same original first-emission endpoints. It therefore starts at H/S=0.91114 rather than the realized 0.91464. H is fixed at 9.097114 pb throughout the 5–20 GeV recoil interval; the fully paired H-bound check below proves that the same density corrections leave H removal unchanged in this sample.

| Density used in S veto | S transfer [pb] | H/S | H−S [pb] |
|---|---:|---:|---:|
| Current channel-average model | 9.984321 | 0.91114 ± 0.00768 | −0.887207 ± 0.078221 |
| Restore Born-scale coupling and PDFs | 9.751672 | 0.93288 ± 0.00786 | −0.654559 |
| Also use held-out local S/B in two Born variables | 9.239724 | 0.98457 ± 0.01297 | −0.142611 ± 0.120192 |

The last control also reduces the inclusive expected residual from **−1.766852 pb to −0.141977 ± 0.129648 pb**, with H/S **0.99087 ± 0.00831**. It accounts for about 84% of the primary-range deficit and 92% inclusively. Errors for the first two ratios use 50 paired source chunks; the fitted-density control uses a 20-block delete-one jackknife including fit statistics. They exclude hard-integration and remaining model/systematic errors.

The residual is statistically compatible with zero in these aggregate observables. **This is not pointwise or per-radiator closure.** In the final 5–20 GeV control, ISR1 still contributes −0.17618 pb, while antibottom FSR contributes +0.08053 pb; their H/S values are 0.9572 and 1.1841. Unresolved Born coordinates or component-dependent density effects can remain. The controls diagnose the dominant bias; they do not constitute a fully validated replacement matching prescription.

## 1. A missing Born-scale factor

The current matcher forms the candidate shower density using the raw counterterm K at the **real-event** scale, then replaces one emission coupling and, for ISR, its emission PDF ratio. However, the actual showered S event carries its underlying Born cross section at the **Born-event** scale. In this setup both scales are H_T/2 evaluated on their respective kinematics.

Wbb Born production contains two powers of alpha_s. The missing factor is

`G = [alpha_s(mu_B) / alpha_s(mu_R)]^2 × L_B(mu_B) / L_B(mu_R)`.

Here `mu_R` denotes the real-configuration reference scale, `mu_B` the canonical projected Born scale, and `L_B` the product of the **two Born-flavour PDFs** at their fixed Born momentum fractions. This includes the spectator beam. It is separate from the existing ISR mother/daughter emission-PDF correction. In code, the corrected denominator is `F_new = F_old × G`; the same F must enter the common bound on both S and H. Source Delta factors and the original source weights are unaffected.

The raw-kernel driver explicitly calls `set_alphaS(p_real)` and records that coupling and both factorization scales. Across all 471,549 active S contexts, native hard-PDF alpha_s at H_T(real)/2 reproduces the recorded kernel alpha_s within 4.2e−10 relative. An independent Fortran Born getter agrees with the original S Born H_T/2 within 4.3e−10. The common factorization-scale assumption is checked against the getter too.

Restoring G reduces the primary-range deficit by **0.232649 ± 0.000905 pb**, or **26.2%**. The paired H/S increase is **0.021737 ± 0.000188**. Inclusively it removes 0.595409 pb, or 33.7% of the deficit. Coupling-only and PDF-only controls show a partial cancellation: the coupling correction improves H/S more strongly, while the Born luminosity correction offsets part of it. Both are required.

All **24,003 negative H events** were independently replayed with their correct provider/GF normalization and canonical inverse Born state. Original keep probabilities are bit-identical. There are 23,980 transfer-capable H contexts and 23 retained without transfer. The H G range is 0.93826–1.34984 and **no H probability changes**. The minimum corrected density bound relative to its source-limited transfer is 2.62770. Any further local-beta multiplier above 0.380562 therefore also preserves all H decisions; every density fit used here satisfies that bound.

## 2. A channel-average S/B factor does not describe the source locally

The code uses `S(Phi_B) ≈ beta_c × B(Phi_B)`, with beta_c between 1.8493 and 1.8896 for the four integration groups. Correct total channel normalization does not imply correct local density. The veto selects particular Born configurations, so the covariance between local S/B and emission/rejection probability matters.

The independent Born control uses the exact same loop_sm parameters, massive mb=4.7 GeV, four-flavour PDF334700, H_T/2 scales and **no cuts**. Density estimates use signed original S weights and positive Born weights. For each of two seeded held-out source folds, beta(Phi_B) is fitted from the opposite source fold and the independent Born control. Fits use only Born coordinates and source densities, never H/S transfer targets. Coordinate/model comparisons were exploratory, so they are not independent confirmation of a chosen final model.

A one-variable Born invariant-mass fit improves H/S only to about 0.956. A joint fit in **Born m(Wbb) and H_T(Born)/2** is much more effective. With a 12×12 positive-Born quantile grid it gives the final table above, with **zero fallback cells for active events and zero saturated rejections**. Grid refinement is stable: 6×6, 8×8, 10×10, 12×12 and 16×16 give primary H/S 0.9815, 0.9835, 0.9841, 0.9846 and 0.9847. The 16×16 fit has eight fallback active events; 12×12 has none.

This uses `F_new = F_old × G × beta(Phi_B)/beta_c`. The minimum 12×12 local multiplier is 0.68235, safely above the all-H bound. All four original channel groups have a deficit; channel integration errors can alter the primary residual by only about 0.039 pb in a conservative coherent one-sigma excursion. Unit flow probabilities, diagram/statistical symmetries and the GF/config mapping also pass the audit.

The LO control totals 183.704 pb versus auxiliary native Born184.209 pb, a small 0.275% normalization difference that remains documented. Rescaling the control to that auxiliary integral changes the final primary H/S to 0.98187 and residual to −0.16802 pb; it does not remove the density-shape conclusion.

Negative S events are included throughout. Their expected inclusive veto contribution is **−3.216772 pb**, offsetting +20.400076 pb from positive S. Thus their presence alone does not explain the excess net S veto; removing their contribution would increase the deficit.

## Independent exclusions and limits

- Previous all-event scale checks and 1,754 native continuation probes pass. The original full-minus-first H/S drift is −0.00154 ± 0.00754, so continuation does not create the main deficit.
- **936,000 native survival trials** test 26 representative Wbb Born contexts, with ALL/ISR/FSR isolation and both saved quadrature and continuous starting-Q sampling. Continuous-Q native/analytic survival is 1.00011 ± 0.00346 overall, 1.00040 ± 0.00271 for ISR and 1.00139 ± 0.00096 for b FSR. These are selected-context checks, not an all-event certificate.
- A separate 156,000-trial FSR control gives 138,252 emissions. The exact native conditional z-to-CDF transformation has zero support violations and uniformity KS p=0.620, with both b and bbar controls passing.
- Independent differentiation of the FSR inverse map at 966 actual endpoints reproduces its determinant within 2.23e−8 relative; native z and evolution t agree to about 1e−12.
- Wbb Born has no final gluon: all 368 inspected profiles have zero Born-gluon radiators, so the earlier global-gluon correction is inactive here.

These checks exclude the tested large Sudakov, massive-FSR shape and mapping defects. They do not establish every ISR branch law, all diagram/flow measures, full high-dimensional signed S/B, or the effect of rare native PDF/trial-envelope warnings. The original 4FS setup retains stock Pythia nfmax=5 coupling running, as recorded in the validation. No Pythia modification is needed for the Born-scale density correction.

## Consequences

The concrete correction is to restore G in the common S/H transfer denominator, using each history's canonical Born state. The remaining dominant approximation is local S/B. A constant beta retune cannot remove its kinematic dependence; a local-density construction needs independent training/validation and per-radiator checks. The two-variable fit is a useful Wbb diagnosis, not a claim that two variables suffice generally or that negative local signed densities can be inserted into a positive veto without further treatment.

[Density-control plot](plots/density_controls.pdf), [radiator decomposition](plots/transfer_decomposition.pdf) and [local-density per-radiator controls](plots/local_density_control.pdf) accompany the numerical records. Scripts, native checks, fitted-density tables and all-H bound proofs are retained. Large event/cache/prediction files remain pinned externally under `/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101/diagnosis_transfer_20261101`. `snapshot.json` records compact hashes and external bindings.
