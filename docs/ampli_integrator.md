# AmpliCol integration for MC@NLO

The branch `MCcntRefactor_Sfun_Granny_AmpliColIntegrator` adds an optional
AmpliCol numerical backend to `madevent_mintMC`. Generate the process output
from this branch and set the following in `Cards/FKS_params.dat`:

```text
#NLOPSIntegrator
1
```

`0` selects MINT and remains the default, including for older cards without
the setting. Fixed-order runs always use MINT. Switching backends requires
fresh integration. Existing exported processes need to be regenerated to
include the new sources and build rules.

The normal `launch aMC@NLO` workflow, run-card folding, `nevents`, `nevt_job`,
and event normalization settings apply. The AmpliCol survey uses 3% accuracy
per channel with at least four iterations; `req_acc` does not set a production
trial budget. `nevents=0` runs only this survey, without event generation.
The implementation is process
independent; the validation below samples massless, massive, initial-state,
and final-state processes rather than establishing every process/shower
combination.

## Coordination and stages

The Python NLO run manager retains one subprocess/integration channel per
worker. Workers communicate through files at stage boundaries; they do not
merge grids or communicate directly. Fixed-order runs and MINT retain their
existing scheduling.

| Stage | AmpliCol behavior |
| --- | --- |
| 1: survey | Start fresh and adapt grids, MC integer sampling and the virtual approximation with the requested folding. Determine each channel's absolute integral to 3% relative statistical accuracy, requiring at least four iterations. Save the signed rate, grids, maxima and auxiliary state. Write no events. A nonconverged survey cannot launch generation. |
| Allocation | Gather all survey results in Python. Draw channel quotas proportional to their absolute integrals, with the quotas summing to exactly `nevents`. Request a 10% reserve for generation. |
| 2: event generation | Start directly from the saved survey grids, maxima and auxiliary state. Run iterations sized by nonzero points and expected event demand; adapt only `ifold=1` maps and update relative proposal envelopes. Continue estimating signed and absolute rates while accumulating events and checking the overweight bound. |
| Collection | Combine survey and production estimates, update channel quotas, and use the reserve or request extra workers where needed. Uniformly select each channel's events once, retain residual corrections, and collect exactly `nevents`. |

These are two executable integration stages. Allocation and collection are
Python coordination; there is no separate stage-0 adaptation run for AmpliCol.
The existing stage-1/2 filenames are retained for job management and restarts.

Updating the virtual approximation changes the absolute integral used for
allocation. Each survey iteration therefore tests its own absolute-rate
estimate. Earlier iterations train the grids and auxiliary state; the final
successful iteration supplies the saved rates and statistical sample count.
No further grid or auxiliary update follows that iteration, so its maxima and
rates describe exactly the saved generation state. Survey CPU accounting still
includes all iterations. Folded observations train the survey maps at every
fold image but count as one statistical observation.

If Born spreading is enabled, its calibration runs within the survey. Since
calibration changes the absolute target, integration then restarts the
four-iteration minimum and accuracy test using the calibrated target.

A channel with final quota `n` requests `ceil(1.1*n)` generated events.
`nevt_job` can divide a channel into independent workers with explicit final
quotas and rounded reserves. Each worker still handles only one channel.
If set to a positive limit, `nevt_job` must be at least two to allow the
rounded reserve even for a one-event quota.
There is no minimum 1,000-event allocation and no Python loop that doubles
fixed evaluation budgets. A channel assigned no events still contributes its
survey rate to the reported cross section.

The survey gives the initial allocation. Production estimates continue evolving,
using every attempted point, including zero and rejected points. Python combines
all completed production iterations with each parent channel's survey exactly
once, even when that channel has several workers. Updated absolute rates set the
final quotas and LHE normalization; updated signed rates and uncertainties are
reported in the results. Channels without production retain their survey estimate.

Rate merging follows AmpliCol's trial-count weighting, including the variance
from differences between iteration means. Production errors use its population
second-moment convention. These are native Monte Carlo estimates under adaptive,
nonzero-count stopping, not an independent fixed-size estimator or a claim of
exact finite-sample unbiasedness.

Each production iteration plans a requested number of **nonzero folded
observations**. The final iteration may finish early once the event reserve
and all three overweight checks pass. The initial request is the worker's generated-event quota
clamped to 1,024–8,192 nonzero points. It is independent of the saved maximum,
so a rare survey weight cannot postpone the first grid update with an enormous
point request. This iteration produces event candidates immediately.

The saved maximum includes the generation probabilities of the virtual and
nonvirtual streams. With at most 200 stored candidates, proposal envelopes use
historical maxima and the surveyed proposal retains its saved-maximum floor.
Larger pools use the relative scales described below, including for the initial
proposal. Candidate storage is separate: its initial cutoff is the smaller of
the saved maximum and survey absolute rate. Retaining additional candidates
allows subsequent selection to adjust the rejection threshold; every final
threshold still respects its recorded storage cutoff.

Subsequent point requests forecast usable events from the observed weight
distribution. For the currently eligible iterations, the sampler finds a
threshold above every storage cutoff whose full overweight mass is below 1%.
The denominator includes all eligible trials, including rejected and zero
observations. The forecast also bounds the worst final subset using the
smallest observed LHE factors, so a collection-dominated tail is
not mistaken for sufficient event yield. These monotone bounds are evaluated
from sorted weights and cumulative sums, with LHE masses evaluated in logs.

The sampler counts saved-priority survivors at that forecast threshold and
estimates acceptance from the latest iteration's survivors per nonzero point.
If the pool is still short of the final quota, its observed factors provide
only a work forecast. The forecast never alters the actual selection threshold,
event corrections, reported tail checks or completion decision. All existing
full-trial, reserve and worst-collected-subset checks remain authoritative.

Point requests use the remaining event demand and forecast acceptance with
10% forecasting headroom. They can shrink near completion
and can grow by at most a factor of two per iteration. When an actual grid
change occurs with eight event-producing proposals already retained, the forecast includes
replacement demand for every batch belonging to the oldest proposal.
There is no permanent upper cap that would prevent large unsplit quotas from
being filled. Zero points contribute to integration statistics but do not fill
this nonzero quota. Trial and iteration limits are failure guards, not
successful completion criteria.

Completion checks are independent of adaptation. Within a long iteration,
the sampler schedules a check near the forecast finish, with at least 1024
new nonzero points between interim checks. If the stored pool is too small,
it skips the expensive envelope/selection work. Every actual check uses the
existing selection and all three strict overweight criteria, after the
previous candidate's actual LHE factor has been recorded. A failed check
continues the same iteration and can update the forecast for its next check;
it does not redraw events, change the sampling map, reset accumulators, or
consume another history entry. Grid adaptation is considered only at full
iteration boundaries, subject to the accumulated-statistics requirement below.

Successful checks stop before another physics evaluation. The final epoch
retains its originally planned nonzero target in the pool, alongside the actual
nonzero and total trial counts. Only the final epoch may be shorter than its
target; rate combination always uses actual observations. At the trial safety
ceiling, the last candidate's LHE factor is recorded and one final completion
check is allowed. Failure then aborts without taking another trial. More
frequent stopping remains subject to the finite-sample statistical limitations
described above; the checks do not impose a separate rate-accuracy target.

During the first three adapting survey updates, the learned bin count doubles
from 8 to 16, 32 and 64. Higher survey statistics can select finer grids through
the existing statistics-based sizing rule, up to 2048 bins; later survey
updates never reduce an existing resolution. Empty/all-zero iterations and
the final converged survey iteration leave the grid unchanged. The 2048-cell
sampling map interpolates these learned bins. Bin counts and boundaries are
preserved by the existing checkpoint format.

During generation, only sampling maps for dimensions with `ifold=1` can adapt
between iterations. An update requires accumulated training draws to exceed
20% of all generation trials so far. Both counts include zero and rejected
draws, and exclude the survey. Small batches retain their training histograms
and pool them with later batches until this strict requirement is met. A
consumed histogram is reset; skipping an update preserves it. Only an actual
change of the sampling map advances the proposal ID and adaptation counter.
Adaptation uses the original AmpliCol grid kernel to move bin boundaries while
keeping each coordinate's saved survey bin count fixed. Folded-direction maps,
folding factors, virtual fits, stream probabilities, MC-integer sampling maps
and Born-spreading tables stay fixed.
An unfolded coordinate's Jacobian multiplies every member of a folded group
equally, so changing that map preserves the fold partners and cancellations.

Production retains each iteration's proposal grids and each candidate's birth
iteration, coordinates, draw-time weight and random priority. Counterfactual
weights under the saved unfolded maps estimate an envelope `M_k` for each
proposal, shared by its consecutive batches. They do not replace draw-time
weights in the rate estimates and do not require reevaluating matrix elements
or rebuilding stored event payloads. With more than 200 stored candidates,
`M_k` is the `max(floor(0.05*N),1)`-th largest historically reweighted candidate
weight for proposal `k`. This is an approximately 95th-percentile scale of the
stored pool, not a percentile of all physics draws or of the cross section.
All candidates, including outliers and event-ineligible history, enter this
calculation. The scale does not clip weights or replace a tail check. Only
ratios between proposal scales matter; a common rescaling cancels against `z`.
Candidates from the last eight distinct event-producing proposals compete
using `log(a/u) - log(M_birth)`. All batches with the same proposal share a
history slot, including batches separated by a skipped adaptation. Their
trial moments and storage cutoffs remain separate. History selection depends
on proposal provenance, without searching for a subset with a favorable
observed tail. Older iterations still contribute to the rates and their
candidates still inform the envelope estimates.
A common normalized threshold `z` gives each eligible iteration its own
acceptance threshold `T_k = M_k*z`. No threshold may fall below the storage
cutoff used when its candidates were written.

## Overweight fraction and event collection

For nonnegative sampling weights `a = abs(w)` and a current maximum weight
`T`, the controlled quantity is the **full absolute cross-section fraction**

```text
f_tail = sum(a for a > T_birth) / sum(a over trials in eligible iterations) < 0.01
```

This includes the entire weight of an overweight point. It is different from
the previous average excess `mean(max(1, a/T)-1)` and from a fraction of event
counts. Signed cancellations do not enter the denominator, weights exactly
at `T` are not overweight, and equality with 1% does not pass.

Workers retain AmpliCol's weight/random candidate priorities and re-evaluate
their acceptance threshold. All points above their birth iteration's final threshold are retained, so their full-tail numerator can be reconstructed
from candidate metadata; the denominator includes rejected and zero-weight
trials from the same eligible iterations. Retained events carry residual
corrections `max(1, a/T_birth)`, normalized over the generated reserve.
The corrected reserve must also pass the 1% tail check.

Uniformly discarding the 10% reserve can increase the realized tail fraction.
Workers therefore require that even the worst possible subset of their final
quota passes the limit. With unit nominal magnitudes, if `m` reserve events
are overweight and their raw corrections sum to `U`, that bound is
`U/(n-m+U)` for `m<n`, and one otherwise. Bias inverse factors are included
when evaluating the corresponding bound on actual event magnitudes.
Python recomputes this guarantee for the updated channel quota before any
random selection. If needed it raises the normalized threshold consistently
across a worker's eligible iterations, recomputes acceptance and corrections,
and requests fresh workers to cover a resulting event shortage. Original pool
metadata and every completed iteration's rate contribution are retained. The
10% reserve usually absorbs quota changes but is not a guarantee of sufficiency.
After all channels pass preflight, collection pools their reserves and trims
once, uniformly, then checks the actual absolute-weight tail fraction. It never
redraws subsets to find a favorable one or clips quotas to available supply.

The raw overweight flags are preserved through normalization; comparing the
final normalized magnitude with the nominal magnitude cannot reconstruct
those flags reliably. The constraints are empirical checks on sampled
weights, not a guarantee about an unseen tail of the integrand.

These are native-style finite-pool weighted samples. Residual corrections
survive in `XWGTUP`, and output uses **`IDWTUP=-4`**. The nominal global
magnitude is `sum(A_channel)/nevents` for `sum`, `sum(A_channel)` for
`average` and bias mode, and one for `unity`. The absolute sum can fluctuate
around its nominal normalization. Bias mode retains its inverse bias factor.
Shower scales, event attributes and internal reweighting records are preserved;
scale/PDF reweighting follows finalization.

The version-2 `ampli_job.dat` specifies generated and provisional final event
counts. Version-5 `ampli_pool.dat` records aggregate and per-iteration moments,
nonzero budgets, storage cutoffs, envelopes, thresholds, event eligibility,
candidate birth iterations/corrections/flags, proposal IDs, and adaptation
metadata alongside `ampli_candidates.lhe`. Version-2/3 pools remain readable
for older workflows. Native version-4 pools retain their original eight-batch
history semantics and can be collected alongside version-5 workers.
New native production writes version 5; the production JSON manifest keeps
its independent version-4 schema.
`ampli_production.json` records the survey and updated rates, revised quotas,
extra workers, collection threshold changes, generation trials, CPU time and
tail diagnostics. Old fixed-budget pools cannot be silently interpreted as
native iteration pools.

## Checkpoints and generation-only runs

Each channel keeps `ampli_grids`, containing a versioned sampler, rates,
folding, channel identity, and virtual approximation state. Its companion
files are `grid.MC_integer`, `res_1.dat`, and, when enabled,
`born_spreading.dat`. Preserve these together with the saved Python jobs.
Split workers start from the same trained files and retain MG5's independent
random streams. Each adapts its own unfolded maps in memory; the saved survey
grids remain unchanged. The same files are included in the existing cluster
transfer protocol. New surveys write checkpoint version 3; checkpoints from
the preceding three-stage implementation require a fresh survey. Native production restarts from
the survey using a fresh worker; serializing or resuming an active native
event pool is explicitly unsupported. The older staged sampler's checkpoint
format remains available for its legacy API.

For example, from the exported process's `bin/aMCatNLO` interface:

```text
launch aMC@NLO -f -p --only_generation --name=additional_events
```

Here `-p` stops after producing the MC@NLO LHE events. Generation-only runs
can change the event count, seed, job-size limit, output/reweight controls,
or choose among `sum`, `average`, and `unity`. They reject a different
backend, central physics settings, folding, Born spreading, bias mode, FKS
settings, run mode, or parameter card. The parameter card is compared by
checksum. These are stage-boundary restarts, not recovery of a partially
written stage-2 LHE file.

## Relative proposal envelopes validated on 2026-10-08

The current implementation passes 153 focused integrator, adapter, pool,
orchestration and LHE tests. New checks independently reconstruct the
relative candidate quantile, exercise sparse-pool bootstrap, and verify that
outliers retain their full tail mass and correction and can still prevent
completion. The pool protocol and all strict 1% checks are unchanged.

Representative unfolded ttbar workers at two seeds use 37–41% fewer trials
for gg/GF3 and 18–20% fewer for gg/GF1 at the large job allocation. The small
GF3 worker uses 5.6% fewer. All actual collections retain the exact requested
counts and pass the selected-tail check. These are individual channel-worker
comparisons using a common older survey, not a full 300K benchmark. At the
same event quota, fewer trials also increase the quoted rate uncertainties.

There is a measured tradeoff: a signed, sparse, folded analytic target uses
3.3% more trials in a 16-seed paired comparison. Rate and corrected-event
shape diagnostics pass, with the usual finite-sample limitations. The
survey-frozen virtual-probability experiment showed no consistent added gain
on top of quantile envelopes, so production retains the existing stream
mixture and common coordinate proposals. See the
[validation report](../validation/ampli_relative_envelopes_20261008/README.md)
for all variants, source provenance, statistics and reproduction instructions.

## Stable adaptation and proposal history validated on 2026-10-08

The focused integrator, adapter, pool, orchestration and LHE suites pass 151
tests. New regressions exercise the strict 20% training requirement, histogram
retention across skipped updates, folded-map invariance, retention beyond
eight batches, complete proposal-group expiry and replacement forecasts,
and mixed POOL4/POOL5 collection with unchanged tail checks.

An isolated ttbar worker comparison holds the saved survey and seed fixed.
The small worker is numerically unchanged at 18,294 trials. The large worker
uses 299,340 instead of 307,498 trials (2.65% fewer), with six instead of eight
grid updates. It collects its exact 24,655-event quota with a 0.5020%
full-weight tail. These are generation-only worker diagnostics with the
older eight-bin survey, not a full 300K-event result or a precision timing
comparison. See the [validation report](../validation/ampli_stable_history_20261008/README.md)
for the policy, source provenance, rates, limitations and additional checks.

A separate analytic experiment uses 16 independent seeds per version on a
signed, sparse, narrow-tail target with one folded coordinate. All 32 pools
pass the reader and tail checks, and four candidate replicas exercise skipped
updates and retention of nine batches. Rates, quoted errors and corrected
event shapes are compared in the report, including a candidate absolute mean
2.19 empirical standard errors below the analytic value. This limited experiment
does not establish unbiasedness or precise uncertainty coverage.

## Independent completion checks validated on 2026-10-08

The standalone integrator, adapter and pool suites pass 111 tests (55, 20 and
36 respectively). Checks cover early finalization, unchanged random draws and
subsequent adaptation after failed probes, actual candidate factors, folded
coordinates, event-history retention, and successful or failed last-trial
checks. The pool reader accepts shortened final epochs while retaining their
planned targets and independently reconstructing every tail check.

An analytic adapter case ends its last epoch at 1024 of 1523 planned nonzero
points, with no additional grid update. A separate regression compares 16
independent seeds per mode for signed rates and corrected event distributions
with and without interim probes. This is an analytic regression check, not a
physics efficiency measurement or a full uncertainty-coverage study.

## Bounded generation iterations validated on 2026-10-07

The revised scheduler passes 128 focused tests. Regression checks cover a
survey maximum many orders above the absolute integral, shrinking batches,
sparse candidate recovery, trial ceilings and large quotas with expiring
event history.

A fresh unfolded `p p > t t~ [QCD]` run collected 10,000 events with automatic
accuracy and seed 19727. Its survey grids and maxima exactly reproduce those
of the stopped 300,000-event run. Generation used 108,662 trials and 106.79 CPU
seconds, with 40 grid updates. The largest native full-tail bound was 0.8729%;
the collected overweight fraction was 0.5943%. The updated signed rate was
`680.6122 ± 2.8253 pb`. A folded Drell–Yan restart collected 2,000 events in
12,851 trials and 37.56 CPU seconds; all folded maps remained fixed and every
tail check passed. These validate the scheduler.

See [the scheduler validation report](../validation/ampli_bounded_iterations_20261007/README.md)
for the changes, exact settings, numerical evidence and reproduction commands.

A subsequent fresh unfolded 300,000-event ttbar run with automatic accuracy
completed in 3,074,406 generation trials and 3,093.50 generation CPU seconds.
Survey plus generation cost 3,153.43 CPU seconds, within 0.3% of the previous
three-stage AmpliCol run. The rate is `680.8823 ± 0.5713 pb`; the collected full
overweight fraction is 0.5765%, with every native and collection bound below
1%. Independent live and archived audits passed. See the
[300K benchmark report](../validation/ttbar_bounded_300k_20261007/README.md)
for the comparison with saved MINT and three-stage AmpliCol results.

A generation-only 300K ttbar test increasing `nevt_job` from 2,500 to 30,000
reduces the split count from 135 to 13 and generation CPU by 4.9%, with nearly
unchanged final-event efficiency (9.758% to 9.791%). The collected overweight
fraction is 0.4001%; every native and collection bound passes 1%. Its signed
rate is `683.2122 ± 0.5736 pb`, a shift of 2.88 nominal combined errors from
the smaller-job run. Multiple seeds are needed to interpret this fluctuation.
See the [job-size comparison](../validation/ttbar_30k_jobs_300k_20261007/README.md)
for CPU accounting, restart normalization and the independent audit.

## Earlier two-stage workflow validation on 2026-10-07

The initial two-stage implementation passed 123 focused tests covering the sampler,
adapter, collection, orchestration, LHE handling and backend selection. New
checks include the four-iteration minimum, folded survey adaptation, retaining
the successful iteration's auxiliary target and maxima, Born calibration inside
the survey, and rejecting checkpoints from the preceding implementation.

A fresh unfolded `p p > t t~ [QCD]` run collected 2,000 events. All five channels
completed four survey iterations below 3% absolute-rate uncertainty, and
generation started directly from their saved maxima. Generation performed three
grid updates in eight iterations. The largest native tail check was 0.5819%;
the collected full overweight fraction was 0.1265%. The updated signed rate was
`683.1833 ± 3.5274 pb`. This validates behavior, not comparative performance.

A fresh `p p > e+ e- [QCD]` run with folding `2,2,2` and a generation-only
restart each collected 2,000 events. The four unfolded coordinates adapted
during generation; the three folded coordinates remained fixed. Fresh
generation performed four grid updates, and the restart performed two. All
survey checkpoints remained byte-identical through both runs, and every
native and collection tail check passed.

See [the two-stage validation report](../validation/ampli_two_stage_20261007/README.md)
for commands, source snapshots, numerical artifacts and independent checks.

## Earlier native iteration workflow validation on 2026-10-07

The preceding three-stage implementation passed 116 focused tests: 34 sampler, 16 MC@NLO
adapter, 33 pool/collector, 23 orchestration, six LHE and four backend-selection
tests. These include nonzero stopping with zero and rejected trials retained
in rate moments, historical Jacobian/envelope reconstruction, nine-iteration
history truncation, folded-grid preservation, evolving signed rates, strict
full-tail checks, changing channel quotas, deterministic threshold increases
and targeted additional generation. Fortran-generated POOL4 files also pass
the Python reader and collector checks.

A `p p > e+ e- [QCD]` survey with folding `2,2,2` was followed by a
6,000-event generation-only run using the final sources. Its eight workers
completed 15 production iterations and seven grid updates. The signed rate
evolved from the survey's `2091.0744 ± 4.2586 pb` to
`2091.3149 ± 3.7355 pb`. Final quotas changed by at most five events per channel;
the reserves covered these changes without additional workers.

Generation used 31,683 attempted points and 85.21 CPU seconds. The largest
final channel tail bound was 0.5214%; the collected sample's full-weight tail
fraction was 0.1662%. All 6,000 events were collected, including 177 negative
weights, with `IDWTUP=-4`. Survey and MC-integer grid files remained unchanged.
This validates the workflow and is not a performance comparison with MINT.

See [the native workflow validation report](../validation/ampli_native_iterations_20261007/README.md)
for source snapshots, commands, numerical records and an independent verifier.

## Earlier fixed-batch adaptation validation on 2026-10-07

The following results describe the preceding POOL3 implementation, which kept
survey-only published rates and used attempted-point batches. They do not
validate the newer native iteration workflow described above.

The adaptation implementation passed 100 focused tests: 28 sampler, 16
adapter, 27 collection, 19 orchestration, six LHE and four backend-selection
tests. New numerical checks cover analytic signed rates and event density
across grid updates, immutable candidate weights, bitwise preservation of
folded maps, training from rejected and zero-weight points, preservation of
survey grid resolution, and exact continuation from an adaptive checkpoint.

A fresh `p p > e+ e- [QCD]` run generated 6,000 events with folding `2,2,2`.
A generation-only restart using the final implementation generated another
6,000 events from the same survey. Each run performed 11 production grid
updates across eight workers, adapting four coordinates and freezing the
three folded coordinates. All saved survey and MC-integer grids remained
unchanged. Both runs reported `2092.9619 ± 4.2579 pb` from the survey.

The final restart used 21,524 generation trials and 61.43 CPU seconds.
Its largest worker tail bound was 0.9883%; the collected sample's full
overweight fraction was 0.5477%. The verifier reconstructs the deterministic
selection, checks corrections against the actual LHE weight magnitudes,
and checks all worker tail fractions. Both outputs contain negative weights
and retain `IDWTUP=-4`. These runs test behavior, not a performance comparison.

See [the adaptation validation report](../validation/ampli_unfolded_adaptation_20261007/README.md)
for commands, source snapshots, metadata and reproducible checks.

## Earlier survey and full-tail validation on 2026-10-07

Before generation-time adaptation was added, 117 focused tests passed covering the sampler,
adapter, LHE writing and collection, coordinator, backend selection, Born
spreading and momentum maps. Tail tests include full-weight versus excess
definitions, nonzero corrections, a strict 1% boundary, worst-case trimming,
bias inverse factors, finite-precision LHE factors and extreme thresholds.

A freshly exported `p p > e+ e- [QCD]` process completed the full workflow
with folding `2,2,2`, PYTHIA8 matching and eight surveyed channels. All channel
absolute errors were below 0.674%; an observation during the survey confirmed
that no candidate event files or production jobs existed yet. Although the
run card specified `req_acc=0.15`, the survey enforced the 3% channel target.

| Check | Full run | Generation-only restart |
| --- | ---: | ---: |
| Collected events | 120 | 60 |
| Production workers | 6 | 11 |
| Generated reserve events | 135 | 71 |
| Production trials | 316 | 165 |
| Maximum reported full-tail fraction | 0 | 0 |
| Normalization | `sum` | `unity` |

The restart split individual channels across workers, preserved all eight
survey grids byte for byte, retained negative weights and scale-reweighting
records, and reported the same survey rate, `2097.7554 ± 4.2192 pb`. Both
outputs have `IDWTUP=-4`. The restart's raw worker metadata and both final
LHE samples were independently checked. These small tests validate the
workflow, not large-sample performance or shower execution. Nonzero tail
behavior is covered by the focused tests.

Commands, logs, events, source fingerprints and the verification script are
in [the validation report](../validation/ampli_survey_3pct_20261007/README.md).

## Historical candidate-pool validation on 2026-10-04

These benchmarks used the previous fixed-budget workflow and mean-excess
criterion. They do not measure the revised 3% survey and full-tail workflow.

That workflow passed 98 automated tests, including fixed-budget
moments with zero contributions, split-worker covariance, large-weight
retention, candidate rethresholding, forced top-ups with frozen rates,
restart checks, compact LHE event counts, and empty worker pools.

A 100,000-event `p p > t t~ [QCD]` run completed with PYTHIA8 matching,
13 TeV beams, folding `2,1,1`, polynomial virtual approximation,
`req_acc=0.003`, and five cores. The comparison uses the earlier MINT run
with the same physical settings and seed:

| Quantity | MINT | AmpliCol pools |
| --- | ---: | ---: |
| Signed cross section [pb] | 683.704 ± 2.713 | 682.535 ± 1.297 |
| Final events / production trials | 18.58% | 22.73% |
| Generation CPU time [minutes] | 18.63 | 17.01 |
| Integration + generation CPU time [minutes] | 23.39 | 19.88 |
| Signed-weight effective event count | 38,405 | 38,584 |

AmpliCol used 440,012 production trials across 46 workers, retained
287,136 raw candidates, and finalized exactly 100,000 events without a
top-up. The signed rates differ by 0.39 combined standard deviations. The
output has `IDWTUP=-4`, and the residual weight factors are retained. These
are parton-level matched LHE tests; the shower itself was not run.

A Drell–Yan generation-only restart also exercises weighted `unity`
normalization and scale reweighting, checking the final event count,
updated init rates, and preservation of internal reweighting data.
Commands, source fingerprints, statistics, and detailed caveats are in
[`validation/ttbar_pool_100k_20261004`](../validation/ttbar_pool_100k_20261004/).

The subsequent rerun with the **1% overweight limit enforced** again produced
100,000 events from 440,012 trials, with no top-ups. All channel mean excesses
were below the limit (maximum 0.638%); the final event records and physical
rate were identical to the earlier pool run. The earlier pool implementation
reported overweight but did not yet enforce the native tolerance. The new
checks cover both excessive-overweight top-ups and acceptance below 1%; 69
focused tests passed. See the
[1% rerun report](../validation/ttbar_pool_overweight_1pct_20261004/README.md).

## Historical initial backend validation on 2026-10-04

Before the candidate-pool revision, the combined automated suite passed 73 tests covering the staged sampler,
folding, small-quota sampling, rejected envelope violations, checkpoints,
adapter statistics/virtual streams, LHE preservation, backend selection,
job allocation/restarts, Born spreading, and momentum maps:

```sh
python -m unittest \
  tests.unit_tests.fks.test_ampli_integrator \
  tests.unit_tests.fks.test_ampli_adapter \
  tests.unit_tests.fks.test_ampli_lhe \
  tests.unit_tests.fks.test_nlops_integrator_selection \
  tests.unit_tests.interface.test_ampli_orchestration \
  tests.unit_tests.fks.test_born_spreading \
  tests.unit_tests.fks.test_momentum_maps
```

| Generated-process check | Result |
| --- | --- |
| `e+ e- > u u~ [QCD]`, folding `2,2,2`, native matching, averaged virtual optimization | All three stages; exactly 10 channel events. Signed channel integral `0.129203 ± 0.000522 pb`, compared with MINT `0.129877 ± 0.000547 pb`. |
| Same process, Born spreading and native matching | Full 800,000-point training plus 200,000-point validation; readaptation, table reload, folded integration, and exactly 10 channel events. |
| Same process, fixed order with switch `1` | Two grouped channels still run MINT and produce `mint_grids`. |
| `p p > e+ e- [QCD]`, folding `2,2,2` | Four subprocesses/eight channel jobs; exactly 100 events, including negative weights, zero/one-event quotas, and split workers. All events retain scale and internal reweight records. |
| Drell–Yan normalization/restart | Initial `sum` weights have magnitude `19.486296`; generation-only runs produce 60 `unity` events and 40 `average` events with magnitude `1948.6296`. |
| Drell–Yan, `nevents=0` | Both integration stages complete, with no event-generation stage. |
| `p p > t t~ [QCD]`, folding `2,1,1`, polynomial virtual approximation | Exactly 20 signed events across five production jobs; signed integral `676.4 ± 4.4 pb`. |

These checks produce MC@NLO LHE samples with PYTHIA8 matching; they do not
include running the parton shower. Standalone electron-positron channel
tests supply a synthetic global normalization to exercise the event path;
the hadronic runs test the actual Python coordinator. Those initial checks did not constitute a performance comparison; the
candidate-pool ttbar comparison above was added subsequently. Machine-readable results, commands, logs, and
source provenance are in
[`validation/ampli_integrator_20261004`](../validation/ampli_integrator_20261004/).

Born calibration exposed a precision failure in the native massless FSR
inverse map for nearly antiparallel daughters and soft recoil. The branch
uses a stable angular coordinate and factored endpoint expressions there.
The original failing point passes reconstruction in both daughter orders,
and the full native calibration passes without loosening tolerances or
discarding contributions. This map correction also applies to MINT users
of native matching.

## Source provenance

`simple_integrator.f90` and `integrator_helpers.f90` originate from
`~/space/git/AmpliCol/master/SimpleIntegrator` at commit
`61b9cd52f44b4cdfc4773daa197e94eeb482b560`. The additional `staged_integrator`
API reuses its adaptive grids. MG5 supplies `ran2`; there is no second RNG.
The AmpliCol objects are linked into `madevent_mintMC` only. Their compile
flags omit the legacy `-fno-automatic` because imported `PURE` procedures
cannot have implicit `SAVE` variables.
