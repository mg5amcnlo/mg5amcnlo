# NLO ttbar: 300,000 events with 30,000-event jobs

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


Completed 2026-10-07. Increasing `nevt_job` from 2,500 to 30,000 reduces
generation CPU by 4.92% in this run, while final-event efficiency changes only
from 9.758% to 9.791%. The estimated parallel generation phase takes longer.
All native and collection overweight checks pass, and exactly 300,000 events
are collected. The rate shifts by 2.88 nominal combined quoted errors; this
single comparison does not establish stable precision across job sizes.

## Controlled inputs and restart setup

The test uses an isolated copy of the
[previous 300K ttbar run](../ttbar_bounded_300k_20261007/README.md), including
its compiled process, saved survey grids, stream maxima and auxiliary state.
All numerical source fingerprints are identical. Settings remain
`p p > t t~ [QCD]`, `loop_sm`, 13 TeV, stable tops, `nn23nlo`, dynamic scales,
PYTHIA8 MC@NLO matching without showering, folding `(1,1,1)`, automatic
`req_acc=-1`, seed 19727, five cores, average event normalization, polynomial
virtual approximation, Born spreading and reweighting off.

Only `nevt_job` changes, to 30,000. The survey is reused, not rerun.
All 20 survey grid, MC-integer grid, result and log files remain byte-identical
to the reference, with unchanged timestamps during generation. Both the actual
banner and random state confirm seed 19727.

The restart interface reenumerates the channel list read from `job_status.pkl`,
while the saved list was sorted by cross section during reporting. An initial
attempt therefore changed the allocation CDF despite using the same seed.
It was stopped and isolated under `superseded_allocation_order/`; its work and
timing are excluded. For the measured run, the launcher sorts the **copied**
pickle by its already recorded `ampli_allocation_order` before restarting.
This restores exactly the previous initial channel quotas. The original run
and production source code are unchanged.

`nevt_job` includes the 10% reserve and is an upper cap. With the restored
quotas, there are five jobs each for the two leading gluon channels and one
job each for the other three channels: 13 jobs in total. The reserve is
330,006 events, compared with 330,092 previously; the small difference is
integer rounding at split boundaries.

## Results

| Quantity | Saved 2,500/job | New 30,000/job |
| --- | ---: | ---: |
| Final events | 300,000 | 300,000 |
| Generation workers | 135 | 13 |
| Generation trials | 3,074,406 | 3,064,052 |
| Final events / all generation trials | 9.75798% | 9.79096% |
| Generation CPU [s] | 3093.4968 | 2941.3711 |
| Equivalent survey + generation CPU [s] | 3153.4324 | 3001.3067 |
| Estimated parallel generation span [s] | 622.426 | 657.409 |
| Signed cross section [pb] | 680.88229 ± 0.57133 | 683.21217 ± 0.57355 |
| Absolute cross section [pb] | 1156.00864 ± 0.67857 | 1159.01023 ± 0.67914 |
| Native iterations / grid updates | 965 / 830 | 117 / 104 |
| Stored candidates | 892,555 | 804,297 |
| Collection rounds | 1 | 1 |
| Negative events | 61,508 | 61,353 |
| Signed effective events | 103,912.26 | 104,264.79 |
| Absolute-weight effective events | 299,518.46 | 299,395.92 |

The newly executed worker CPU is **2,941.37 s**. The equivalent full-workflow
number adds the inherited 59.93562422 s survey cost explicitly; no new survey
CPU was spent. Generation CPU falls by 4.92%, equivalent total CPU by 4.82%,
and trial count by only 0.337%. Thus the measured benefit is modest, with little
change in unweighting efficiency. Fewer iterations and worker starts may reduce
overhead, but this test does not profile its sources or isolate host variation.

The generation-only invocation takes 684.969 wall seconds. The previous
696.771-second invocation also included compilation and the survey, so these
whole-launch times do not define a valid speedup. From preserved worker log
timestamps and integer wall counters, the parallel generation span increases
by about 5.6%. Fewer larger jobs leave less scope to balance the work across
five cores; the progress history records the final two/three-worker tail.
These are single-run CPU measurements and inferred phase wall spans.

## Rates and overweight checks

The signed-rate difference is **+2.32988 pb**, with a quadrature error scale of
0.80955 pb: **2.878 nominal combined errors**. The absolute rate differs by
3.00159 pb, or 3.1265 nominal combined errors. The signed shift is dominated
by the two leading gluon channels (+1.03034 and +1.21407 pb).

The runs share a survey and seed but have different job sizes, sampling
histories and stopping times. Their correlation is not known. These error
ratios are descriptive, not calibrated significances, and the observed shift
should not be described as demonstrated rate agreement. Multiple independent
seeds would be needed to distinguish an ordinary fluctuation from sensitivity
of adaptive stopping or the quoted uncertainty to job size. The audit finds
no discrepancy in rate reconstruction or event collection.

| Full-weight overweight diagnostic | 2,500/job | 30,000/job |
| --- | ---: | ---: |
| Largest native worker bound | 0.980748% | 0.791048% |
| Largest final channel collection bound | 0.851108% | 0.719811% |
| Collected corrected absolute-weight tail share | 0.576486% | 0.400089% |

Every native full-trial, reserve, worst-subset and collection check is strictly
below 1%. The definition counts the full weight belonging to events above the
iteration threshold, rather than just excess weight. Native correction factors
are preserved in the `IDWTUP=-4` LHE; the largest normalized factor is 5.99131.
The 1% criterion constrains the tail's cross-section share, not each factor.

Collection requires neither top-ups nor threshold increases. The largest
updated channel-quota change is 1,624 events, covered by the reserve. The
weighted event-sample rate is 683.9574 pb, with about 1.7109 pb sampling error
conditional on the integrated normalization, consistent with its own
integration estimate. The rate estimate uses the final survey iteration once
per channel and all production iterations, totaling 3,105,012 observations.

## Validation and evidence

The main verifier and independent standard-library-only audit pass. They
reconstruct signed/absolute rates and errors, allocation, thresholds, native
correction factors, deterministic collection, final LHE weight magnitudes and
all tail checks. The comparison passes all 19 checks, including exact initial
quotas, seed, physics settings, source equality and survey identity.

`metrics.json`, `comparison.json` and `independent_audit.json` hold the results;
their adjacent logs record successful execution. The compact archive contains
190 checksummed files, including the
[final LHE sample](ampli/Events/jobs30k_300k/events.lhe.gz). Build files and raw
candidate LHE spools are omitted. `work_directory.txt` identifies the full
isolated process directory. `allocation_order.json` records the restart
normalization, and `ampli_started.json`/`ampli_finished.json` record provenance.

From the repository root:

```sh
python validation/ttbar_30k_jobs_300k_20261007/verify.py
python validation/ttbar_30k_jobs_300k_20261007/compare.py
python validation/ttbar_30k_jobs_300k_20261007/independent_audit.py \
  validation/ttbar_30k_jobs_300k_20261007
```

`run_benchmark.py` reproduces the setup using the preserved previous process;
use a fresh validation directory to preserve this evidence. No production
code was changed for this test.
