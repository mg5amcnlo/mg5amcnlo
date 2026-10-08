# Internal stream proposal investigation

The current worker has one physical integration channel and two possible internal production streams: the nonvirtual contribution (including the frozen virtual approximation), and the virtual residual. It draws the virtual stream with a fixed probability obtained from the survey's absolute rates. It then divides both the absolute and signed sampled contribution by that stream's probability before calling the numerical integrator. Virtual fits, discrete auxiliary proposals and folding are unchanged during production.

## Minimal experiment

A fixed, survey-selected mixture can be adjusted without changing event history or the pool format. The probability stays the same throughout the worker, so its factor cancels from every historical proposal ratio. It also cancels separately within every folded observation; no folding capability is lost.

For surveyed conditional maxima `M_n` and `M_v`, the probability minimizing the initial common maximum is

```
p_max = M_v / (M_n + M_v)
max(M_n/(1-p_max), M_v/p_max) = M_n + M_v.
```

This is only an empirical initial-envelope statement. Maxima are noisy, and the sampling density subsequently changes. It does not establish better production throughput.

The experiment uses

```
p_rate = clamp(A_v/(A_n+A_v), 0.001, 0.999)
p = max(p_rate, min(p_max, 4*p_rate, 0.1))
```

The one-sided correction avoids decreasing the existing virtual sampling rate because of a nonvirtual survey outlier. The factor-four and ten-percent caps are conservative heuristics, not derived optimal values. They limit oversampling due to a noisy virtual maximum. For baseline probabilities below ten percent, the relative loss of nonvirtual draws is at most eleven percent. Baseline probabilities already above ten percent are unchanged. Born-only, virtual-only and zero-quota behavior remain governed by the existing adapter paths.

The final prototype is in `../experiments/stream_mixture/ampli_mint_adapter.f90`. It factors this calculation into `ampli_stream_probability`, adds finite/nonnegative validation and overflow-safe ratios, and logs the rate and chosen probabilities. Ordinary arithmetic is identical to the measured first prototype. It does not add a third stage, change the rate target, update the virtual fit, change coordinate grids, or weaken the three final overweight checks. **It is not enabled in the production adapter:** the measured extra gain on top of the adopted relative-envelope change was inconsistent.

## What the saved surveys say

The five common ttbar surveys imply the following changes. These numbers are predictions from survey data, not measured generation speedups.

| Channel | Existing virtual probability | Proposed probability | Initial envelope ratio | Estimated absolute variance ratio |
|---|---:|---:|---:|---:|
| gg/GF1 | 0.006735 | 0.021473 | 0.314 | 0.980 |
| gg/GF2 | 0.005677 | 0.010778 | 0.527 | 0.998 |
| gg/GF3 | 0.006325 | 0.006325 | 1.000 | 1.000 |
| uux/GF1 | 0.007001 | 0.013856 | 0.505 | 0.993 |
| uxu/GF1 | 0.008253 | 0.010292 | 0.802 | 0.998 |

The variance estimate is based on the final survey iteration's absolute-stream moments. If its virtual evaluation probability is `r`, the survey's virtual estimator is `I(virtual draw) * v/r`. Therefore the conditional second moments under the shared coordinate proposal are

```
S_n = (N-1)*error_n**2 + A_n**2
S_v = r * ((N-1)*error_v**2 + A_v**2)
Var[production weight] = S_n/(1-p) + S_v/p - (A_n+A_v)**2.
```

The absolute-variance optimum is `sqrt(S_v)/(sqrt(S_n)+sqrt(S_v))`; it differs from the maximum optimum. For example gg/GF1 gives about 1.56%, compared with 2.15% from its maxima. The final survey has only about 128–162 expected virtual evaluations per channel, so these conditional-tail and second-moment estimates are uncertain. They cannot replace generation measurements or independent-seed checks. The saved signed total does not retain the separate second moments needed to infer the signed-mixture variance by this calculation.

The complete values and reproducer are [survey_choices.json](survey_choices.json) and [survey_choices.py](survey_choices.py).

## Requirements for more intrusive alternatives

If a candidate from stream `s` was born under joint proposal `p_j(s) g_j(x|s)`, its weight under proposal `k` must be

```
w_k = w_j * p_j(s)/p_k(s) * g_j(x|s)/g_k(x|s).
```

Equivalently the coordinate factor is the ratio of the inverse-density Jacobians `J_k/J_j`. The current coordinate-only history can evaluate that coordinate factor because every proposal uses one shared grid and fixed stream probabilities. It cannot supply a varying stream-probability ratio. Correct adaptive mixtures require storing the stream identity and birth probability on every candidate, and the probability vector with every historical proposal. An actual probability change must consume a proposal-history entry even when the coordinate grids are identical.

Separate stream grids additionally need a stream-indexed mapping in sampling, training and historical reweighting. Folded coordinates must still be held fixed; historical folded sums cannot generally be transformed using a single point's Jacobian. Each stream's training threshold must use its relevant attempted draws. A virtual sample containing only a few hundred observations is not automatically sufficient for a reliable high-dimensional separate grid. Independent stream envelopes also need stream-aware storage cutoffs and proof that every contribution above a final accepted cutoff was retained.

Updating the virtual fit is a separate problem. Although the signed sum of the approximation and residual is invariant, their separate absolute targets change when the fit changes. Combining their old event pools without accounting for that change is not justified by a coordinate or stream-probability ratio.

## Measurement

The diagnostic instrumentation in `../physics/instrument_streams.py` records every attempted draw's epoch, stream, fixed probability, absolute and signed probability-adjusted weight, and candidate identity after `native_consider`. It introduces no RNG calls. The paired gg/GF1 production replays attribute the observed final tail and candidate envelopes to streams, rather than relying solely on the initial survey maximum. Instrumented CPU timings contain diagnostic I/O overhead and should be identified as such.

The baseline, fixed-mixture-only and quantile-only experiments used the same saved eight-bin survey, seed 19727, final quota 25,985 and reserve 28,584. Their final pool metadata and trace candidate identities agree exactly; all three overweight validators pass in every run.

| Experiment | Trials | Virtual draws | Virtual absolute-rate share | Virtual share of full overweight mass | Virtual overweight draws |
|---|---:|---:|---:|---:|---:|
| Baseline | 189,058 | 1,301 | 0.5884% | 10.435% | 14 |
| Fixed mixture only | 173,173 | 3,676 | 0.5974% | 0.389% | 1 |
| Quantile only | 151,142 | 1,058 | 0.6201% | 16.525% | 26 |

The fixed-mixture-only run uses 8.40% fewer trials and reduces the largest sampled virtual weight from 2172.73 to 570.21. It shifts the remaining observed tail toward the nonvirtual stream. The quantile-only run gains more overall on this seed despite a larger virtual share of its residual tail. Therefore stream-tail improvement is not synonymous with an equal improvement in total event throughput. The combined result is measured separately by the benchmark runner; gains must not be added.

| Seed | Quantile only trials | Quantile plus fixed mixture trials | Additional mixture change |
|---|---:|---:|---:|
| 19727 | 151,142 | 153,352 | +1.46% |
| 59727 | 145,189 | 143,829 | −0.94% |

The combined mixture reduces the virtual fraction of the final full tail to 0.835% and 1.395%, respectively, compared with 16.525% and 17.442% for quantile-only runs. Nevertheless the extra total trial change is inconsistent and small. The production decision is therefore to adopt the relative-envelope quantile and retain the original rate-based stream mixture. The fixed-mixture helper and tests remain an isolated reproducible experiment. Separate internal grids or adaptive stream probabilities are not justified by these measurements.

At fixed event quota, the baseline generation rate estimates are `ABS=103.1696 ± 0.2531`, `signed=62.9882 ± 0.2136`; fixed mixture gives `ABS=102.8852 ± 0.2712`, `signed=62.8313 ± 0.2282`. The quoted uncertainties increase because the worker finishes sooner and the subsequent proposals differ. These runs share survey/seed inputs and are not an independent uncertainty-coverage test. No full-sample speedup follows from this single worker comparison.

Stream attribution and exact trace/pool consistency checks are reproduced by [analyze_trace.py](analyze_trace.py). Data are [baseline](gf1_baseline.json), [fixed mixture](gf1_stream_mixture.json), and [quantile](gf1_quantile.json).

## Analytic checks

The prototype adapter suite passes 24 tests, including explicit probability-cap, zero-rate, zero-maximum, extreme-finite, invalid-input, Born-only and virtual-only cases. New signed narrow-virtual fixtures force the chosen probability above the rate fraction, then independently reconstruct every generated absolute/signed moment using the fixed selected-stream probability. Both fully folded grids and a mixture of folded and adaptive coordinates are exercised. Mature relative scales are checked independently through order-statistic counts; final overweight checks remain unchanged. The adopted production adapter suite passes its 20 tests, including the updated relative-envelope assertions; no unused stream-mixture helper or configuration knob remains in production.

[analytic_seeds.py](analytic_seeds.py) explicitly loads and compiles the archived prototype adapter and its fixtures. It runs sixteen fresh survey-plus-generation seeds for each folding setting, with exact analytic absolute and signed integrals and three bins in the first coordinate. The corrected reserve histogram uses the conditional expectation over the folded points, so diagnostic histogramming introduces no extra RNG calls. All 32 runs pass the three strict one-percent checks. Ensemble rate means differ from exact targets by at most 1.70 empirical standard errors of the mean; absolute and signed histogram means differ by at most 1.63. The absolute-rate pull standard deviations are 0.987 and 0.892; signed-rate pull deviations are 0.776 and 0.664. These small seed ensembles are consistent with the analytic means but cannot establish precise uncertainty calibration. Details, seeds and source hashes are in [analytic_seeds.json](analytic_seeds.json).

Reproduce with:

```
python -m unittest tests.unit_tests.fks.test_ampli_adapter
python -m unittest discover -s validation/ampli_relative_envelopes_20261008/experiments/stream_mixture -p test_ampli_adapter_prototype.py
python validation/ampli_relative_envelopes_20261008/streams/analytic_seeds.py
```

Separate grids or adaptive stream probabilities should be considered only after the less intrusive fixed-mixture and relative-envelope experiments identify a remaining stream-specific loss in production.
