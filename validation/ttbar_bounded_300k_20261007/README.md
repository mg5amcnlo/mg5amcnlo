# NLO ttbar: corrected two-stage AmpliCol, 300,000 events

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


Completed 2026-10-07. A fresh export and fresh survey produced exactly
300,000 MC@NLO hard events. The corrected scheduler restores the total cost
to the previous three-stage AmpliCol level: 3,153.43 versus 3,146.09 worker
CPU seconds (+0.23%). All native and collection overweight checks pass 1%.

## Settings and provenance

`p p > t t~ [QCD]`, `loop_sm`, 13 TeV, stable tops of mass 173 GeV,
`nn23nlo`, dynamic scale choice `-1`, PYTHIA8 matching without showering,
folding `(1,1,1)`, automatic `req_acc=-1`, seed 19727, `nevt_job=2500`,
five cores, average event normalization, polynomial virtual approximation,
Born spreading and scale/PDF/internal reweighting off.

All three input cards are byte-identical to the stopped 300K attempt. The
only numerical source change since that attempt is `simple_integrator.f90`.
Exact exported sources, input cards, hashes and launch metadata are saved.
The actual banner and random state confirm the seed; the persistent run card
resets `iseed` to zero after launch, as usual.

Five survey channels each complete four iterations with absolute-rate errors
of 0.884–1.456%, below the 3% requirement. Their saved grids, MC-integer grids
and maxima exactly reproduce the stopped run. There is no stage 0. Generation
starts directly from that state, with all seven coordinates adaptive.

## Rates and cost

Only the new AmpliCol column was rerun. The references are the previously
audited [300K MINT and three-stage AmpliCol runs](../ttbar_unfolded_300k_20261007/README.md)
on the same machine and inputs. MINT timing comes from its pristine run;
its separate diagnostic replay is used only for overweight measurements.

| Quantity | Saved MINT | Saved three-stage AmpliCol | New two-stage AmpliCol |
| --- | ---: | ---: | ---: |
| Final events | 300,000 | 300,000 | 300,000 |
| Generation workers | 122 | 134 | 135 |
| Generation trials | 2,419,293 | 3,021,464 | 3,074,406 |
| Final events / all generation trials | 12.4003% | 9.9290% | 9.7580% |
| Survey and grid CPU [s] | 360.425 | 117.517 | 59.936 |
| Generation CPU [s] | 2206.902 | 3028.574 | 3093.497 |
| Total worker CPU [s] | 2567.328 | 3146.091 | 3153.432 |
| Whole-launch wall time [s] | 623.902 | 700.584 | 696.771 |
| Signed cross section [pb] | 681.48155 ± 1.68937 | 681.89983 ± 0.57503 | 680.88229 ± 0.57133 |
| Absolute cross section [pb] | 1157.11858 ± 1.93088 | 1156.52020 ± 0.67980 | 1156.00864 ± 0.67857 |
| Signed effective events | 103,777.30 | 104,368.20 | 103,912.26 |
| Absolute-weight effective events | 300,000.00 | 299,371.30 | 299,518.46 |

Relative to three-stage AmpliCol, generation takes 2.14% more CPU, while
survey/grid CPU falls by 49.0%. Total CPU and wall time are essentially the
same in these single runs. Relative to MINT, new AmpliCol uses 40.2% more
generation CPU and 22.8% more total worker CPU. Its quoted signed integration
error is 66.2% smaller. The algorithms achieve different precision, so this
compares their native workflows rather than equal-precision timings.

The signed rate differs from MINT by -0.5993 pb and from old AmpliCol by
-1.0175 pb. The corresponding quadrature error scales are 1.7834 and 0.8106 pb.
These are descriptive comparisons: shared seeds and integrands do not establish
independent errors. Errors are conventional Monte Carlo integration estimates
under adaptive, nonzero-count stopping, not theory uncertainties.

Worker CPU excludes compilation and Python coordination; wall time includes
them. Efficiencies count every generation trial, including rejected and zero
observations, and include reserve and collection losses. Timing is from one
run of each implementation and remains subject to host variation.

## Scheduler, collection and overweight checks

Generation completes 965 iterations and 830 grid updates across 135 workers.
There are 55 zero observations and 892,555 stored candidates. First-iteration
budgets follow the quota clamp; 342 subsequent iterations shrink their budgets.
The independent checks reconstruct all forecasts, maximum growth bounds,
initial storage cutoffs and first-epoch envelope floors.

At the same 66 completed workers reached in the stopped attempt, this run uses
1,523,762 trials and 1,578.11 CPU seconds, versus 5,876,966 trials and 5,494.18
CPU seconds previously. This isolates the scheduler improvement for matching
channel jobs; those stopped partial totals are not a completed sample.

The 330,092 reserve events cover the evolving rates: final channel quotas
change by at most 1,620 events. Collection completes in one round without
top-ups or threshold increases, yielding exactly 300,000 events.

| Full-weight overweight diagnostic | New AmpliCol |
| --- | ---: |
| Largest native worker bound | 0.980748% |
| Largest final parent-channel collection bound | 0.851108% |
| Collected corrected absolute-weight tail share | 0.576486% |

Every native full-trial, reserve, worst-subset and collection check is strictly
below 1%. This definition counts the full absolute weight associated with
events above their iteration threshold, rather than only their excess weight.
Native correction factors remain in the `IDWTUP=-4` LHE; the largest normalized
correction is 4.45866. The 1% limit constrains the tail fraction, not each
individual correction factor. The checks describe sampled phase space.

For context, saved MINT's full cross-section tail estimate is 0.341973% ±
0.013628 percentage points; its largest parent-channel estimate is 0.821874%
and largest split-worker estimate is 1.308424%. MINT does not enforce the 1%
criterion. Those estimates and AmpliCol's collected corrected-weight fraction
have different aggregation and weight treatment.

The sample contains 61,508 negative events (20.5027%); the negative share of
absolute weight is 20.5496%. The weighted sample rate is 680.9048 pb, with
about 1.7077 pb event sampling error conditional on the integrated normalization,
consistent with the integration estimate. The rate estimate includes the final
survey iteration once per channel and all production iterations: 3,115,366
statistical observations in total. Earlier survey iterations train the grids.

## Verification and artifacts

The main verifier and separate standard-library-only audit both pass. They
reconstruct rates and uncertainties, initial and updated channel allocations,
native thresholds/corrections, deterministic selection, final LHE weights and
all overweight checks from raw records. The independent audit passes against
both the live output and compact archive. Source fingerprints, physics/cards,
seed, survey checkpoint identity and two-stage scheduling are also checked.

`metrics.json` and `comparison.json` contain the full results;
`independent_audit.json` and `archived_independent_audit.json` contain independent
reconstructions. The archive has 1,166 checksummed files, including the
[final LHE sample](ampli/Events/bounded_300k/events.lhe.gz). It omits build files
and candidate LHE spools. The full temporary export is recorded in
`work_directory.txt`.

From the repository root:

```sh
python validation/ttbar_bounded_300k_20261007/verify.py
python validation/ttbar_bounded_300k_20261007/compare.py
python validation/ttbar_bounded_300k_20261007/independent_audit.py \
  validation/ttbar_bounded_300k_20261007
```

`run_benchmark.py` records the fresh export, cards and launch commands.
Use a fresh validation directory if rerunning, to preserve the evidence here.
No production code was changed during this benchmark.
