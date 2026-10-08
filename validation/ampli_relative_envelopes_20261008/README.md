# Relative envelopes and internal stream proposals

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


This implements item 5 of the [efficiency review](../ampli_efficiency_review_20261008/review.md). The selected change uses a robust relative scale for each sampling proposal. The existing stream mixture is retained: changing it did not consistently improve efficiency once the new relative envelopes were enabled.

## Production change

With more than 200 stored candidates, each proposal's relative envelope is the `max(floor(0.05*N),1)`-th largest historically reweighted candidate weight. With 200 or fewer candidates, the existing maximum calculation and survey-maximum floor remain in use. Batches with the same proposal share the scale. The initial proposal also switches to the candidate quantile once the pool is sufficiently populated.

This is a relative scale of the **stored candidate pool**, not a 95% percentile of the cross section and not a bound that clips event weights. Actual thresholds still come from the common rank selection and cannot fall below any original storage cutoff. Outliers remain in the pool, in full-weight tail sums, and in native correction factors. All-trial rates, folding constraints, the two-stage workflow, quota reserve, stable adaptation and history retention are unchanged.

The full-trial, reserve, worst-final-subset and actual collection checks all remain strictly below 1%. POOL5 needs no format change: it already stores the numerical scales, thresholds, birth information and corrections that the collector independently verifies. There is no new user setting or integration stage.

The upstream source inspection corrected one premise of the original review: AmpliCol uses a quantile for provisional counts and storage, but reconstructs maxima for its final selection. The policy implemented here is therefore an optimization of final relative scales, not a restoration of upstream finalization. See the [design assessment](design/findings.md) for the source references and conditional density argument.

## Physics comparisons

The baseline is the implementation after stable adaptation/history retention, including the tail-aware forecast and independent completion checks. Saved survey grids, cards and worker quotas are common between variants. These replays use the earlier eight-bin survey and no folding, deliberately isolating generation changes from survey refinement.

Each executable still handles one physical channel. The table reports individual `gg -> ttbar` channel workers, not complete ttbar samples. Large workers correspond to the `nevt_job=30000` allocation; small to `nevt_job=2500`.

| Channel / worker / seed | Final events | Baseline trials | Quantile trials | Change |
|---|---:|---:|---:|---:|
| GF3 small / 19727 | 2,242 | 18,294 | 17,270 | −5.60% |
| GF3 large / 19727 | 24,655 | 299,340 | 187,206 | −37.46% |
| GF3 large / 39727 | 24,655 | 326,096 | 192,279 | −41.04% |
| GF1 large / 19727 | 25,985 | 189,058 | 151,142 | −20.06% |
| GF1 large / 59727 | 25,985 | 176,559 | 145,189 | −17.77% |

The four large-worker comparisons total 991,053 baseline versus 675,816 quantile trials, **31.81% fewer**. This is a measured reduction in those workers, not a prediction for an entire 300K-event run or every process. Some baseline results are reused only after exact source-hash agreement; instrumentation was separately verified to preserve the pool and candidate LHE byte for byte. CPU timings are retained in the raw results, but different host load and diagnostic I/O make trial counts the primary comparison.

The GF3 seed-39727 replay used the experimental combined binary. Its stream probability is unchanged for this channel, so its numerical algorithm is the quantile-only policy; provenance distinguishes that executable from the final source snapshot. This was verified bit for bit from every logged draw in [the probability comparison](physics/gf3_mixture_invariance.json). A fresh replay using the final adopted sources reproduces the quantile prototype's small-worker pool and candidate LHE byte for byte; see [final-source invariance](physics/adopted_invariance.json).

Every native tail check passed. Actual LHE collection was also performed for all 18 experimental worker outputs, with exact quotas and full-weight selected tails below 1%. In the adopted large-worker comparisons, the largest native bound was 0.99405% and the largest collected tail 0.92806%. Collection uses diagnostic unit normalization, so these are channel payload/selection checks rather than a newly normalized complete ttbar sample. Corrected-weight top-pT and ttbar-mass histograms retain negative events and native residual weights; their raw bin counts and sums are in [shape diagnostics](physics/weighted_shape_diagnostics.json). Shared seeds correlate the samples, so these are not independent uncertainty-coverage tests.

A separate isolation experiment removes the first-proposal survey floor after 200 candidates while retaining historical maxima. It gives the same small-worker gain, but only **0.50%** fewer trials for large GF3 and **0.38%** for large GF1 at seed 19727. The substantial large-worker improvement therefore comes from the relative quantiles, not just removal of that floor.

Fewer physics evaluations also mean less precise integration estimates at the same final event quota. For example, the seed-19727 GF3 worker's signed estimate changes from `269.1837 ± 0.8271 pb` to `268.3723 ± 1.0311 pb`, and its absolute estimate from `468.8376 ± 1.0248 pb` to `469.0895 ± 1.2703 pb`. These are correlated generation-only channel estimates, not the total ttbar cross section. The optimization preserves the requested event and tail contract; it does not impose an additional production rate-accuracy target.

## Stream investigation and decision

The survey-selected virtual stream typically accounts for only 0.6–0.8% of the absolute rate, but its probability-adjusted maximum can dominate initialization. We tested the fixed probability

```text
p_rate = clamp(A_virtual / (A_nonvirtual + A_virtual), 0.001, 0.999)
p_max  = M_virtual / (M_nonvirtual + M_virtual)
p_test = max(p_rate, min(p_max, 4*p_rate, 0.1))
```

This raises an undersampled virtual component without changing probabilities during a worker. The caps limit the influence of noisy survey maxima; they are a heuristic, not a proven optimum. The prototype preserves inverse probability factors, virtual fits and the absolute target. It passed deterministic edge cases and signed/folded analytic diagnostics.

On GF1 seed 19727, the mixture alone reduced trials from 189,058 to 173,173 (8.40%). Once quantile envelopes were enabled, however, the comparison was:

| GF1 large seed | Quantile only | Quantile + altered mixture | Additional change |
|---|---:|---:|---:|
| 19727 | 151,142 | 153,352 | +1.46% |
| 59727 | 145,189 | 143,829 | −0.94% |

There is no consistent additional gain, so **the altered mixture is not enabled in production**. Its implementation, diagnostics and tests remain as an experiment. Stream attribution also shows that nonvirtual contributions dominate the final overweight mass; the virtual dominance inferred from the initial survey maximum does not persist to the same degree during generation.

Separate stream envelopes are not a drop-in improvement: a different normalization per physical stream would alter their relative contribution unless weights or stream allocations compensate. Adapting stream probabilities also requires recording stream identity and birth probabilities in each candidate's history. Separate coordinate maps require corresponding map provenance. Given the measured envelope gains and weak incremental mixture result, that larger change is not justified here. See [stream findings](streams/findings.md) and the [design assessment](design/findings.md).

## Analytic validation and limitations

The standalone study uses a signed target, 60% zero trials, a narrow high-weight region, actual two-image folding, and 30,000 final events per replica. A fixed diagnostic first batch of 128 nonzero points exercises longer histories. Sixteen independent seeds per version check rates, quoted errors and corrected event shapes against analytic values. All native/collected tail checks and folded-map checks pass. Quantile rate means are within 1.86 empirical standard errors of their exact values and corrected shape means within 0.96 errors; this does not prove unbiasedness or calibrate uncertainty coverage precisely.

The matched-seed efficiency follow-up is an explicit counterexample to universal improvement: quantiles use **3.30% more trials** on this analytic target. The paired difference is `10,263 ± 4,372` trials (empirical standard error): slower in nine seeds, unchanged in five and faster in two. Releasing only the survey floor leaves every seed's trials, rates and corrected shapes unchanged, so this regression is due to the relative quantiles. The tradeoff is retained alongside the larger gains in the tested MC@NLO workers; no claim of universal speedup is made.

The [analytic findings](analytic/findings.md) include independent and paired studies, compressed evidence, exact sources and reproduction commands. The quantile test fixtures independently reconstruct historical scales, exercise the 200/201-candidate boundary and common-scale cancellation, and verify that a retained outlier can still block completion or receive its full correction without changing rates or RNG state.

All **153 focused production tests pass**: 59 standalone integrator, 20 adapter, 41 pool/collector, 27 orchestration and six LHE tests. `git diff --check` passes. Commands are in [test results](test_results.txt). The withdrawn stream-mixture prototype separately retains 24 passing tests and 32 analytic replicas; these are experimental validation, not additional production features.

## Evidence and reproduction

- [Physics findings](physics/findings.txt) and [full numerical comparison](physics/comparison.json).
- [Actual collection checks](physics/collection_validation.json), [baseline instrumentation invariance](physics/instrumentation_invariance.json), and [source match](physics/adopted_sources_match_production.json).
- [Source snapshots](physics/adopted_sources/), source hashes, saved inputs and individual worker records in `physics/`.
- [Analytic study](analytic/findings.md), including all independent and paired outcomes.
- [Stream study](streams/findings.md), fixed-mixture prototype and signed/folded analytic results.

For a fresh physics replay, copy the scripts and desired `*_sources/`
directories from `physics/` to a clean scratch directory. Leave the existing
result directories and result JSON files out of this copy. Run its
`setup.py`, then `run_workers.py` with the desired source variant and workload.
The original process exports and installed libraries in `paths.json` must
remain available; adjust `setup.py` if they move. Run
`validate_collection.py baseline adopted`, `analyze_collected_shapes.py` and
`summarize.py` after running matching baseline/adopted workloads. Pass other
completed variant names to the collector for their comparisons. Set
`AMPLI_STREAM_DIAGNOSTICS=1` for both variants to reproduce diagnostic I/O;
set `AMPLI_SEED=39727` or `59727` for the extra-seed workloads, using the
matching source/output variant names to keep runs separate. Preserve recorded results by using fresh
scratch output, rather than overwriting this evidence. Source snapshots
identify each experimental variant; `adopted_sources` is the deployed policy.
Large candidate/collected LHE files remain in the isolated `/tmp` export with
recorded paths and hashes. Archived stream traces are gzip-compressed with
[uncompressed and compressed hashes](physics/trace_manifest.json).
