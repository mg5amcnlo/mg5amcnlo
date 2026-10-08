# NLO ttbar: 10,000 events without folding

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


Completed 2026-10-07. Both generators produced 10,000 events with folding
`(1,1,1)`. MINT was faster, while AmpliCol obtained a smaller integration
uncertainty and passed its 1% overweight checks. MINT's measured full overweight
cross-section fraction exceeded 1%.

## Configuration and scope

Process `p p > t t~ [QCD]`, `loop_sm`, 13 TeV, stable tops with mass 173 GeV,
`nn23nlo`, dynamic scale choice `-1`, PYTHIA8 MC@NLO matching without showering.
Five cores, seed 19727, `nevt_job=2500`, `req_acc=0.01`, `event_norm=average`,
`UsePolyVirtual=True`, Born spreading off, scale/PDF reweighting and internal
reweight information off.

Relative to the preceding [folded benchmark](../ttbar_native_10k_20261007/README.md),
the only card change for each backend was folding `(2,1,1)` to `(1,1,1)`.
MINT and AmpliCol cards agree except for `NLOPSIntegrator`.
All eight exported production source hashes match the preceding benchmark.
The matrix-element source export was reused; each run performed fresh grid
adaptation, integration and generation without reusing saved grids or results.

Runs were sequential: pristine MINT, AmpliCol, then instrumented MINT.
The third run measures MINT's overweight tail and reproduces every worker and
final event sequence exactly. The timings below use pristine MINT.
Instrumentation was confined to the diagnostic export; production source files
were not changed for this benchmark.

## Rates and efficiency

| Quantity | MINT | AmpliCol |
|---|---:|---:|
| Final events | 10,000 | 10,000 |
| Generation trials | 57,851 | 93,528 |
| Final events / generation trials | 17.286% | 10.692% |
| Grid adaptation + survey CPU | 95.633 s | 117.003 s |
| Generation CPU | 54.769 s | 93.512 s |
| Total worker CPU | 150.402 s | 210.515 s |
| Whole-launch wall time | 76.633 s | 102.389 s |
| Signed cross section | 680.565 ± 4.277 pb | 676.562 ± 2.634 pb |
| Absolute cross section | 1146.825 ± 4.903 pb | 1150.202 ± 3.131 pb |
| Relative signed integration uncertainty | 0.6285% | 0.3894% |
| Negative-event fraction | 20.47% | 21.11% |
| Signed effective events, `(sum w)^2 / sum w^2` | 3488.08 | 3323.26 |
| Signed effective events / generation CPU second | 63.69 | 35.54 |
| Absolute-weight effective events | 10,000.00 | 9982.71 |

AmpliCol uses 70.7% more generation CPU and 40.0% more total worker CPU, with a
38.4% smaller quoted signed-rate uncertainty. The signed rates differ by 4.00 pb,
or 0.80 times their errors combined in quadrature.

These are the implemented generation workflows, not runs adjusted to equal
achieved precision. MINT uses its normal absolute-rate accuracy target and
reports the stage-1 integration estimate. AmpliCol surveys each channel to 3%,
then continues estimating the rate during event generation. Its final rate
includes the survey once and all production iterations. Errors are Monte Carlo
integration uncertainties, not theory uncertainties. Native adaptive stopping
gives conventional estimated errors, not exact finite-sample coverage.

CPU is summed over workers and excludes compilation and Python coordination;
whole-launch wall time includes them. Trial efficiency includes AmpliCol's
reserve events and any rejected candidates. Timing results are single runs.
AmpliCol retains native overweight correction factors, so its final event
magnitudes are not exactly equal.

## Overweight measurements

The full tail means the **entire absolute cross section** associated with points
above the applicable unweighting envelope. It differs from both the fraction of
trials above the envelope and the excess weight above that envelope.

MINT's diagnostic records every production trial before acceptance, with no
additional random-number calls. Its results are:

| MINT diagnostic | Result |
|---|---:|
| Full tail, combined using survey stream rates | 1.3906% ± 0.1437 percentage points |
| Full tail, using production stream means | 1.4363% ± 0.1655 percentage points |
| Largest parent-channel survey-weighted full tail | 1.5502% (`P0_gg_ttx/GF2.0`) |
| Largest split-worker survey-weighted full tail | 2.3449% |
| Excess-only fraction, using survey rates | 0.3591% |
| Above-envelope trials / accepted events | 105 / 10,000 |
| Above-envelope share of nominal final LHE absolute weight | 1.0500% |
| Largest weight / envelope ratio | 10.8150 |

MINT does not enforce a 1% full-tail limit and keeps equal nominal event
magnitudes for these points. Its 1.05% final-LHE tail share therefore differs
from the estimated 1.39–1.44% full cross-section tail.

For nonvirtual trials, MINT samples the envelope proposal `q=H/Z`; the correct
absolute observation for tail integration is `a=fABS*Z/H`, where `Z` is the
product of the mean envelope heights in each coordinate. Virtual trials have
`Z=H`. The full-tail observation is `a * I[fABS>H]`. Split workers are pooled
within each parent channel and stream, with distinct streams combined using
their absolute rates. Raw `fABS` sums would give the wrong tail estimate.
The quoted tail errors use a delta-method approximation conditional on grids,
trial counts and survey rates, excluding survey-rate errors and finite-sample
effects of stopping at a fixed number of accepted events. The excess-only
fraction measures omitted correction mass, not a net total-rate bias; the LHE
retains the surveyed normalization.

AmpliCol passed every native full-trial, reserve-sample, possible-subset and
collection tail check. The largest check was **0.879790%**. Its final collected
sample contains 44 tail-flagged events carrying **0.614978%** of the corrected
absolute event weight. The largest normalized correction is 3.21253.
These are empirical checks on the observed pools, not a guarantee about unseen
phase space. The largest worker bound and MINT's global tail have different
aggregation; the actual collected-weight fractions are 0.615% and 1.05%,
respectively.

## Native adaptation and collection checks

All seven coordinates had adaptive mask `1111111`. Seven production workers
completed 20 native iterations and 13 grid updates, scheduled using nonzero
points. There were 93,526 nonzero trials and two zero trials.
The pool contained 29,166 stored candidates, yielding 11,003 reserve events
and finally 10,000 collected events. The extra three relative to 11,000 come
from integer allocation across workers. Collection finished in one round,
without top-ups or rethresholding. The largest updated parent-channel quota
change was 36 events.

The audit reproduced rates from 76,800 survey trials plus all 93,528 production
trials, both initial and updated quotas, native thresholds and correction
factors, deterministic collection, and all final weight magnitudes.
An independent audit recomputed these quantities from raw files and also
verified MINT's proposal correction and exact event replay.

## Effect of removing folding

Compared with the preceding `(2,1,1)` runs:

| Quantity | MINT | AmpliCol |
|---|---:|---:|
| Generation CPU, folded → unfolded | 91.20 → 54.77 s | 146.09 → 93.51 s |
| Generation CPU reduction | 39.9% | 36.0% |
| Total worker CPU reduction | 40.3% | 33.9% |
| Generation trial-count increase | 16.6% | 24.0% |
| Negative-event fraction, folded → unfolded | 18.64% → 20.47% | 18.92% → 21.11% |
| Signed effective events / generation CPU improvement | 47.6% | 34.7% |

Removing folding costs more trials and increases cancellations, but reduces the
cost per trial enough to improve effective-event throughput in both runs.
The folded and unfolded signed rates were 684.401 ± 3.464 → 680.565 ± 4.277 pb
for MINT and 684.080 ± 2.358 → 676.562 ± 2.634 pb for AmpliCol.
Their nominal changes are 0.70 and 2.13 quadrature errors; the same seed was
used, so these are not strictly independent comparisons. One pair of runs does
not establish a systematic folding dependence.

## Saved evidence and reproduction

`metrics.json` and `mint_tail_metrics.json` contain the original analyses.
`archived_metrics.json` and `archived_mint_tail_metrics.json` reproduce their
numerical results using only the saved archive. `folding_comparison.json`
records comparisons with the preceding run; `final_checks.json` records final
consistency checks. Cards, logs, results, pool metadata, trial sidecars, LHEs,
source snapshots, hashes and exact diagnostic patch are included.

From the repository root, reproduce the archived analyses with:

```bash
python validation/ttbar_unfolded_10k_20261007/analyze.py \
  --work validation/ttbar_unfolded_10k_20261007 \
  --run-name benchmark_10k --output /tmp/ttbar-unfolded-metrics.json
python validation/ttbar_unfolded_10k_20261007/analyze_mint_tail.py \
  --process validation/ttbar_unfolded_10k_20261007/mint_tail \
  --reference validation/ttbar_unfolded_10k_20261007/mint \
  --output /tmp/ttbar-unfolded-tail.json
```

`run_benchmark.py` and `run_sequence.py` preserve the launch procedure. To rerun
generation, place them and the diagnostic module in a fresh work directory with
an unrun matrix-element export named `base`, then execute `run_sequence.py`.
The archived `generate.cmd`/`generate.log` describe the original reused source
export; their output path belongs to the preceding benchmark.
