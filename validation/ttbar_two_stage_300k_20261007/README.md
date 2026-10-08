# Stopped 300K ttbar two-stage benchmark

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


Stopped at the user's request on 2026-10-07 because generation was too slow.
All benchmark worker processes were terminated. This is an incomplete run;
there is no final 300,000-event sample or validated final rate.

The run used a fresh export and fresh survey, with the previous benchmark's
physics settings: `p p > t t~ [QCD]`, 13 TeV, `nn23nlo`, stable tops, PYTHIA8
matching without showering, polynomial virtual approximation, folding `(1,1,1)`,
`req_acc=-1`, seed 19727, `nevt_job=2500`, five cores, average event normalization.
The first attempt was stopped during the survey after noticing that an archived
post-launch run card had its seed reset to zero. Its metadata is isolated under
`superseded_seed_reset/` and none of its work enters the results below. The
actual run's banner, random state and input cards confirm seed 19727.

## Diagnosis

The numerical/orchestration change passed the saved stream maximum into the
existing native generator, where that argument controls both candidate storage
and the first iteration's nonzero-point budget:

```
N_first = max(1024, ceil(Q * max(1, M / A)))
```

`Q` includes the event reserve, `A` is the surveyed absolute integral, and `M`
is the maximum corrected for the virtual/nonvirtual selection probabilities.
Previously the adapter passed `A` in place of `M`, giving an initial budget
approximately equal to `Q`. It also retained more candidates initially.

For `gg/GF3.0`, the new survey gives `A=476.4682 pb` and `M=35859.6140 pb`:
`M/A=75.2613`. Each worker consequently requests about 185,600 nonzero points
before its first completion/envelope/adaptation check, rather than about 2,400.
Across all 135 initial workers, the initial iterations alone request
**13,871,396 nonzero points**, versus **3,021,464 total generation trials** in
the completed previous three-stage benchmark. This is a factor of 4.59 before
any continuation iterations or zero points.

The existing continuation rule aggravates that oversized start:

```
N_next >= 2 * N_current
```

For example, `gg/GF2.0_1` had generated 2,426 of its required 2,458 reserve
events after 41,583 nonzero points. The remaining 32 events would forecast
about 549 points at its measured acceptance, but the scheduler requested
**83,166** more points. Grid and envelope updates only occur at iteration
boundaries. The storage cutoff also prevents recovering candidates rejected
earlier under the high initial threshold.

The two-stage survey itself was faster: **59.624 CPU seconds**, compared with
**117.517 CPU seconds** for adaptation plus survey previously. The observed
regression is therefore in production initialization and iteration scheduling.

## Partial observations

At termination, 66 of 135 initial workers had completed: all 12 `gg/GF1.0`
workers and all 54 `gg/GF2.0` workers. Their counters were:

| Quantity | Completed workers only |
| --- | ---: |
| Generation trials | 5,876,966 |
| Generation CPU seconds | 5,494.183 |
| Iterations / grid updates | 95 / 29 |
| Largest final worker tail check | 0.5548% |

These exclude all unfinished workers and are not extrapolated final results.
The previous complete three-stage generation used 3,028.574 CPU seconds.

Of 29 continuation boundaries in completed-worker logs, 28 report the
`1,1,1` diagnostic sentinel used before a sufficient candidate quota is
available; these are not measurements of a 100% overweight fraction. One
continuation has a genuine measured tail failure. Every completed worker's
final tail checks pass the strict 1% condition.

The appropriate next change is to retain the two executable stages and saved
grid/maximum, but give event generation a separate, bounded iteration schedule.
Short event-producing iterations can refine the grids and envelope promptly;
later budgets should follow the remaining event demand and measured efficiency
without compulsory doubling near completion. This need not add a survey or
relax the final 1% overweight checks. No such production changes were made in
this benchmark task.

## Evidence

`initial_epoch_forecast.json` records the exact per-channel forecasts.
`partial_diagnostics.json` records completed-worker counters and continuation
logs. `physics_source_comparison.json` confirms the same physics as the saved
MINT reference: 833 shared files are byte-identical, 48 generated HELAS files
only reorder equivalent declarations/independent assignments, and the four
substantive differences are the expected two-stage implementation changes.

`partial/` preserves completed logs, results and survey state; `sources/`
preserves the exact implementation. The full temporary export, including
unfinished worker files, is identified by `work_directory.txt`. `stopped.json`
records the termination. Analysis/archive scripts requiring a complete run
were prepared but deliberately not run on this incomplete sample.
