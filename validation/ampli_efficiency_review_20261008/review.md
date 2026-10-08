# AmpliCol MC@NLO efficiency review

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


Date: 2026-10-08

There are credible opportunities for significant improvement. The strongest concern how many physics evaluations we perform, how we retain earlier events, and how we train the sampling grids. This review combines representative-worker profiling with analysis of the saved pools from the two 300K ttbar runs.

The proposals preserve two integration stages, one physics channel per executable, the 3% absolute-rate survey with at least four iterations, the 10% event reserve, the folding constraints, native correction weights, and the existing strict 1% overweight checks.

No production algorithm was changed during this investigation. The potential gains below require implementation and measurement.

## Where the time goes

Representative-worker profiling gives:

| Activity | Share of CPU |
|---|---:|
| Complete integrand evaluation | **86–88%** |
| Complete event-candidate preparation and writing | **About 11%** |
| Sampler bookkeeping | **Around 1%** |

The integrand includes phase space, PDFs, subtraction and physics bookkeeping as well as matrix elements. Candidate preparation was measured in a separate replay of the small worker. Historical-envelope calculations are included in sampler bookkeeping and account for less than 0.5% of total CPU.

The instrumented replays preserved candidate LHE and POOL4 files byte-for-byte, and all non-timing integration results were identical. These are within-run CPU fractions, not measured optimization speedups. See [profiling findings](profile/findings.txt).

Significant gains therefore require reducing expensive physics evaluations or improving their event yield. Optimizing historical-grid arithmetic alone would have little impact.

## 1. Predict remaining work using the actual overweight distribution

Currently, when a job has enough candidate events but fails the overweight check, its estimate of effective completion is reduced using a heuristic involving `0.8 × quota` and `0.01 / overweight_fraction`. The relationship between threshold, overweight mass and surviving events is nonlinear; this heuristic can request considerably more work than necessary.

We already retain enough information to estimate how many events survive a threshold satisfying the observed 1% tail criterion. This gives a more direct estimate of remaining work. For fixed iteration envelopes, all contributions above the storage cutoffs are retained, so the observed full-weight tail can be evaluated for admissible thresholds.

There is measurable evidence of excess capacity:

| Events per job | Existing reserve | Capacity allowed by all current 1% checks |
|---|---:|---:|
| 2,500 | 330,092 | 376,934 — **14.2% more** |
| 30,000 | 330,006 | 389,857 — **18.1% more** |

All 148 adjusted worker pools passed the production validator, preserving the full-trial, reserve and worst-final-subset checks. See [capacity results](statistics/surplus.json) and [reproducer](statistics/surplus.py).

**These percentages are unused final-state capacity, not demonstrated CPU savings.** The final envelopes contain information unavailable earlier. Nevertheless, they strongly motivate improving the forecast.

Initially, change only the forecast and retain the existing final acceptance checks. The discrete reserve/subset checks must not all be assumed monotone in the threshold.

## 2. Separate completion checks from grid updates

Completion is currently checked at the end of an entire integration batch. A job can consequently continue evaluating expensive points after it could potentially have finished. Approximately one quarter of all trials occur in the final batches, although only part of that work might be avoidable.

Near predicted completion, check progress in shorter chunks while keeping the current proposal fixed. Adapt the grid only when enough new information warrants it. This avoids making every completion check a new, potentially noisy grid update.

Keep the requested 10% reserve. Shortened iterations must record consistent counters, moments and pool metadata. Changing the stopping rule requires checking rate estimates and uncertainties across independent seeds.

## 3. Stabilize adaptation and replace the fixed eight-iteration event history

This appears particularly relevant to larger jobs:

| Diagnostic | 2,500/job | 30,000/job |
|---|---:|---:|
| Trials no longer eligible to supply final events | 3.1% | **13.7%** |
| Reserve events / still-eligible trials | 11.1% | **12.5%** |
| Final events / all trials | 9.76% | 9.79% |

Larger jobs obtain better event yield from their eligible trials, but much more earlier generation work has fallen outside the eight-iteration window.

Retain history according to its useful event contribution and compatibility with current thresholds. Consecutive batches with identical proposals could share one history entry. Short completion checks should not automatically consume history slots.

The current generation code updates grids after every unfinished iteration, including small finishing batches. Original AmpliCol requires a substantial contribution of new statistics before adapting. Restoring a stability criterion or damping could prevent noisy updates from degrading an already useful proposal. Coordinates with `ifold=1` remain adaptive; folded coordinates retain their current constraints.

Retaining everything indefinitely is not automatically optimal: poor old proposals or high storage cutoffs can constrain subsequent selection. Expired trials still contribute to integration and training, so the percentages above are not directly recoverable savings. They must not be added to the surplus percentages in the previous section.

See [epoch diagnostics](pool_diagnostics.json) and [reproducer](pool_diagnostics.py).

## 4. Train a better shared survey grid

Every saved ttbar survey grid has only **eight learned bins per coordinate**. The 2,048 sampling cells interpolate those bins; they are not independently learned bins.

Reaching 3% uncertainty on the integral does not necessarily produce a good proposal for unweighting. Test:

- 16 or 32 learned bins, with sufficient statistics and smoothing.
- More stable grid accumulation. The current one-sided soft-maximum update depends on sample order and can follow rare outliers.
- Modest additional proposal training within the survey.

The shared survey costs approximately **60 CPU seconds**, compared with **3,093 seconds for generation**. Doubling survey effort would pay for itself with roughly a 2% generation improvement. This is a break-even calculation, not a measured gain.

This preserves the two-stage setup. More robust training must never mean clipping physics contributions. Resolution, training rule and survey effort should first be tested separately.

See [saved survey-grid and stream diagnostics](native/survey_sampling_diagnostics.json).

## 5. Revisit relative iteration envelopes, then internal stream sampling

Original AmpliCol uses a candidate-weight quantile for provisional event counts and storage, but its final selection recomputes maxima over historically reweighted candidates. The implementation reviewed here also used these maxima for final relative envelopes. Rare outliers can therefore change how efficiently different iterations contribute events. This distinction was established during the step-5 investigation; see the [source assessment](../ampli_relative_envelopes_20261008/design/findings.md).

A robust relative scale deserves testing while retaining the actual 1% cross-section checks. Lowering every envelope by the same factor achieves nothing: the common factor cancels in selection. The original quantile is not itself a substitute for the required cross-section tail criterion.

A more substantial extension is separate virtual/nonvirtual proposals or envelopes inside each executable. The virtual residual supplies only about 0.6–0.8% of the absolute rate, yet its probability-adjusted maximum dominates the initial envelope in four of five channels. Its direct CPU cost is tiny, but its effect on sampling may matter.

This still means one physical channel per executable. It requires recording stream identity and birth probabilities in event history so historical reweighting remains correct. Coordinate-only history cannot support arbitrary stream-probability adaptation. Virtual fits themselves should remain frozen unless their changing absolute target is also handled.

## Other opportunities and limitations

Approximately 63% of stored candidates in the small-job run are unused in the final reserve. Avoiding their preparation at negligible cost would have an idealized ceiling around 7% overall, given the measured candidate-preparation cost. Future envelope changes require retaining sufficient information, however, and later reconstruction could erase the saving.

Increasing job size alone barely improved trial efficiency. The 30,000/job run used 4.9% less recorded generation CPU but took 5.6% longer in generation wall time. This is a single comparison, not a repeated timing experiment; load balance also matters.

MINT's product bounding envelope is a useful proposal-training reference. AmpliCol's factorized maps can in principle represent such a product proposal. More general correlated or cell-based proposals could help, but are substantially more intrusive and have no quantified benefit from these diagnostics.

## Recommended implementation and validation order

1. Implement a better tail-aware forecast, retaining all existing final checks.
2. Add completion checks independent of adaptation.
3. Test stable adaptation and useful-history retention.
4. Test improved shared survey grids.
5. Investigate relative-envelope choices and separate internal stream proposals if needed.

Benchmark each change separately before combining them. Use common inputs and the saved survey for initial computational comparisons, then fresh surveys and independent seeds for statistical validation. Include analytic signed, sparse, narrow-tail and folded targets, followed by multiple moderate physics samples for the leading candidate. Compare rates, empirical fluctuations, quoted uncertainties, weighted distributions, trials, CPU and wall time. A fixed-trial diagnostic can help isolate stopping effects without adding a third production stage.

Matching MINT's observed trial count would require approximately **21% fewer AmpliCol evaluations**. This is a useful target; the evidence does not yet establish that gain or a factor-of-two improvement. The MINT and AmpliCol uncertainty and per-worker overweight contracts also differ.

The two job-size runs differ by about **2.9 nominal quadrature errors in signed rate**. No obvious missing-trial or probability-factor error was found, but shared survey/seeds and adaptive stopping require care when interpreting this difference. One pair of runs neither establishes bias nor validates uncertainty calibration. See [the statistical review](statistics/review.txt).

The empirical 1% checks measure observed tail mass; they do not guarantee the absence of unseen rare phase space. Preserve native correction factors and retain every attempted contribution in the integration estimate.

## Supporting material

- [Detailed technical notes](review.txt)
- [Profiling and numerical-invariance findings](profile/findings.txt)
- [Epoch and history diagnostics](pool_diagnostics.json)
- [Retrospective event-capacity analysis](statistics/surplus.json)
- [Survey-grid and stream diagnostics](native/survey_sampling_diagnostics.json)
- [Statistical review](statistics/review.txt)
- [Source provenance](source_provenance.json): all eight reviewed numerical/coordinator sources match the completed baseline benchmark.
