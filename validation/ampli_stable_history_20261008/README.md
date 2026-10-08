# Stable adaptation and proposal history

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


Date: 2026-10-08

This implements item 3 of the [efficiency review](../ampli_efficiency_review_20261008/review.md). Generation now pools training information across small batches and retains event history by sampling proposal. The representative large ttbar worker used **2.65% fewer physics trials**; the small worker was numerically unchanged. These are worker diagnostics, not a full 300K-event performance result.

## Implemented policy

- At a full generation-batch boundary, update unfolded grids only when pending training draws **strictly exceed 20% of all generation trials so far**. Both counts include zero and rejected draws, and exclude the survey.
- Preserve the training histograms across skipped updates. Reset them once consumed. Use the existing AmpliCol adaptation kernel and the saved survey bin counts. Folded coordinates remain fixed.
- Advance the proposal ID only when an actual sampling map changes. A consumed histogram that produces the same map does not create a new proposal.
- Retain all event-bearing batches belonging to the **last eight distinct event-bearing proposals**. Consecutive batches with unchanged maps share a history slot. Keep each batch's moments, storage cutoff and candidate birth records separate.
- Share historical envelope calculations across batches with the same proposal. Every batch using the initial survey proposal inherits the survey maximum floor. Reweighting between identical proposals preserves the original candidate weight exactly.
- When a new proposal will expire the oldest retained group, forecast replacement of the survivors from **all batches** of that group. A held adaptation does not expire history.

The survey, two-stage workflow, allocation reserve, completion probes, native correction weights, and final selection formulas are unchanged. Every attempted generation contribution still enters the evolving rate estimate. Full-trial, reserve and worst-final-subset tail fractions must each remain strictly below 1%, and collection verifies the written sample.

New pools use `MG5_AMPLI_POOL 5`, adding a proposal ID to each batch record. The Python reader validates complete group retention, sequential proposal IDs and consistency with the actual grid-update count. Legacy POOL4 keeps its original eight-batch semantics; mixed POOL4/POOL5 workers and POOL5 top-ups are supported. The independent production-manifest schema remains version 4.

## Why this retention rule

The [offline experiment](offline/findings.txt) replays all 148 archived pools from the earlier 300K ttbar comparisons. It enumerates all native rank thresholds and validates the resulting corrections and all three tail bounds with the production validator.

| Archived job-size setting | Original last-eight-batch capacity | Capacity retaining all history | Difference |
|---|---:|---:|---:|
| 2,500 | 376,934 | 382,733 | +1.54% |
| 30,000 | 389,857 | 428,354 | +9.87% |

These capacities use final-state information. They do not demonstrate earlier stopping or CPU savings. All-history retention also need not help for every possible set of storage cutoffs. In these archives, a seemingly conservative alternative that retained only older batches compatible with the existing storage floor actually reduced one worker's safe capacity by 1,616 events.

The implemented rule groups batches by proposal provenance. It does not search observed event yields or tails for a favorable history subset. This keeps the retention decision simple while removing expiry caused solely by short batches using an unchanged map. General quality-based history selection and unlimited retention remain outside this change.

## Representative ttbar replay

Both variants include the preceding tail-aware forecast and independent completion checks. The only numerical difference is the policy implemented here. They start from the same saved eight-bin survey, cards, seed 19727 and worker quotas, with no folding. Holding this older survey fixed isolates generation changes; the recently implemented survey-bin refinement is present in both source variants but is not exercised.

The worker is `P0_gg_ttx/GF3.0`. The job-size settings determine its allocated quotas; these are not complete ttbar samples.

| Metric | 2,500/job baseline | 2,500/job new | 30,000/job baseline | 30,000/job new |
|---|---:|---:|---:|---:|
| Final worker events | 2,242 | 2,242 | 24,655 | 24,655 |
| Generated reserve | 2,467 | 2,467 | 27,121 | 27,121 |
| Physics trials | 18,294 | 18,294 | 307,498 | 299,340 |
| Generation CPU, seconds | 18.15 | 17.39 | 289.36 | 277.24 |
| Completed batches | 4 | 4 | 9 | 8 |
| Grid updates | 3 | 3 | 8 | 6 |
| Trials outside event history | 0 | 0 | 8,192 | 0 |
| Largest of three native tail bounds | 0.8537% | 0.8537% | 0.9987% | 0.5459% |
| Actually collected full-weight tail | 0.6856% | 0.6856% | 0.9474% | 0.5020% |

The small worker's candidate LHE file is byte-identical and its rates are unchanged. Its apparent 4.2% timing improvement is noise, so the similar 4.2% CPU decrease of the large worker is not a precise speedup estimate. The deterministic large-worker trial reduction is 8,158, or 2.65%.

The larger runs agree through batch 6. The new policy holds adaptation after that batch, pools its training with batch 7, and finishes in batch 8. The baseline finishes in batch 9 and expires its first batch. This comparison does not separate the adaptation and retention contributions. In particular, the candidate has only eight batches, so retention beyond eight batches is exercised by the regressions rather than this physics replay.

Generation-only channel rates, read from `res.dat`, are:

| Large worker rate | Baseline | New |
|---|---:|---:|
| Absolute, pb | 468.8906 ± 0.9986 | 468.8376 ± 1.0248 |
| Signed, pb | 269.5851 ± 0.8085 | 269.1837 ± 0.8271 |

These are correlated single-seed worker estimates, not full ttbar cross sections or a coverage test. The somewhat larger quoted errors accompany the reduced trial count. No rate or shape accuracy improvement is claimed from this pair.

Actual collection was run for all four worker outputs with fixed collection RNG and diagnostic unit normalization. It wrote exactly the requested worker event counts and passed the strict tail checks. This exercises real candidate payloads and the production collector; it is not a newly normalized full physics sample.

See [physics findings](physics/findings.txt), [machine-readable comparison](physics/comparison.json), [collection validation](physics/collection_validation.json), [prefix invariance](physics/prefix_invariance.json), and the [input provenance](physics/input_provenance.json). Full [baseline](physics/baseline_source_hashes.json) and [candidate](physics/candidate_source_hashes.json) source hashes and snapshots are retained alongside the replay scripts.

To replay, first copy `physics/` into fresh scratch space so the recorded
results remain intact. In that copy, run `setup.py`, `run.py baseline`,
`run.py candidate`, `validate_collection.py`, and `summarize.py`, in that
order. Setup creates a fresh isolated process export. The two original
benchmark exports listed in `paths.json`, and their installed libraries,
must still be available; their absolute paths can be adjusted in `setup.py`
if moved. Large candidate and collected LHE files remain in the isolated
temporary export rather than this report directory.

## Regression validation

The focused checks cover:

- The strict 20% boundary, counting zeros and rejected trials, and preservation of histogram contents across skipped updates. Splitting training across small batches reproduces the combined-batch maps and all-trial moments exactly.
- Folded-map invariance and correct proposal counters across skipped and actual updates.
- Retention beyond eight batches using one proposal, preservation of known old overweight points, and the survey maximum floor across unchanged proposals.
- Expiry and forecast replacement of a complete oldest proposal containing multiple batches.
- Independent reconstruction of final tail checks, corrections and moments, corrupted proposal metadata, legacy POOL4 semantics, mixed-worker collection and top-ups.

The controlled Fortran expiry fixture changes public batch budgets to force otherwise rare transitions. Its serialized eligibility is inspected, but its artificial intermediate targets are not presented as a valid production schedule. Normal adapter outputs independently exercise the complete POOL5 reader and collector.

All **151 focused tests pass**: 57 integrator, 20 adapter, 41 pool/collector,
27 orchestration and 6 LHE tests. Commands and results are in
[test results](test_results.txt). `git diff --check` also passes.

## Independent-seed analytic comparison

The [analytic experiment](analytic/findings.md) compares 16 fresh seeds per
version on a signed target with 60% zero trials, a localized high-weight
region, and actual two-image folding of the second coordinate. Each replica
requests 30,000 final events plus the 10% reserve. The first batch requests
128 nonzero points to exercise long proposal histories; this is a diagnostic
API setting, not a change to the production default. Pilot runs selected
schedule coverage only and are excluded from these results.

All 32 pools pass the production reader and all three strict tail checks.
Uniform final selection also passes its tail check, and the folded map stays
bitwise fixed. Four of the 16 candidate replicas skip an adaptation and retain
nine batches in their eight-proposal history. Candidate native tail bounds
reach at most 0.98174%; collected selections reach at most 0.85565%.

| Rate | Exact integral | Baseline mean ± empirical standard error | New mean ± empirical standard error |
|---|---:|---:|---:|
| Absolute | 0.08313333 | 0.08309311 ± 0.00005051 | 0.08304561 ± 0.00004008 |
| Signed | 0.00358333 | 0.00359410 ± 0.00001239 | 0.00357752 ± 0.00001168 |

The candidate absolute mean is 2.19 empirical standard errors below the
analytic answer, or 1.47 standard errors using the reported integration
uncertainties. The baseline-to-candidate difference is 0.74 combined empirical
standard errors. Signed-rate scatter divided by the RMS quoted error is
1.029 for the baseline and 0.973 for the candidate; the corresponding absolute
ratios are 0.845 and 0.672. These 16-replica diagnostics do not establish
unbiasedness or precise uncertainty coverage.

The three corrected final-event observables measure the low-coordinate
region, the narrow tail, and the signed event fraction. Candidate means are
all within 0.71 empirical standard errors of their analytic values. The
largest baseline deviation is 1.99 standard errors in the tail fraction.
Trial totals are 5,258,919 and 5,236,559 respectively. Since the variants use
independent seed sets, the 0.43% difference is not performance evidence.

The [per-seed records](analytic/replicas/records.json),
[summary and source provenance](analytic/replicas/summary.json), driver and
runner preserve the complete setup. Raw pools and candidate observables are
compressed alongside their diagnostics.
