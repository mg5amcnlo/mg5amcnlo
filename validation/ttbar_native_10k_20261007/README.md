# NLO ttbar: 10,000-event MINT versus native AmpliCol comparison

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


The current AmpliCol implementation completed the requested sample but was
slower for event generation: 60.2% more generation CPU time and 26.4% more
total worker CPU time than MINT. Its quoted total-rate uncertainty was 31.9%
smaller. Both rates agree within 0.077 quadrature-combined Monte Carlo errors.

## Settings and scope

Fresh `p p > t t~ [QCD]` export from the current working tree, 2026-10-07:
13 TeV protons, stable tops with mass 173 GeV, built-in `nn23nlo`, dynamic
scale choice -1, PYTHIA8 MC@NLO matching without showering, folding `(2,1,1)`,
polynomial virtual approximation, native MC counterterm evaluation, and Born
spreading disabled. Both runs use seed 19727, 10,000 requested events,
`nevt_job=2500`, `event_norm=average`, and five cores. Scale/PDF reweighting
and internal reweight records are disabled.

The run and parameter cards are byte-identical. FKS cards differ only in
`NLOPSIntegrator`: 0 for MINT and 1 for AmpliCol. Source hashes also match.
Runs executed sequentially, MINT then AmpliCol, from separate copies of one
fresh matrix-element export. No integration grids or prior results were reused.
Production source files were not changed for this benchmark.

Both run cards specify `req_acc=0.01`, the natural MINT target for 10,000
events. MINT translates this to per-channel absolute-rate accuracy targets.
The new AmpliCol path instead surveys every channel to 3% and continues
improving rate estimates during production. The achieved precisions below
therefore differ; these measurements compare the configured workflows.

## Results

| Measurement | MINT | AmpliCol |
| --- | ---: | ---: |
| Final LHE events | 10,000 | 10,000 |
| Generation trials | 49,596 | 75,402 |
| Final events / generation trials | 20.163% | 13.262% |
| Grid-adaptation CPU time | 31.871 s | 60.141 s |
| Survey/envelope CPU time | 128.839 s | 112.242 s |
| Adaptation + survey CPU time | 160.711 s | 172.383 s |
| Generation CPU time | 91.195 s | 146.087 s |
| Total worker CPU time | 251.906 s | 318.471 s |
| Generation events / CPU second | 109.655 | 68.452 |
| Whole-launch wall time | 101.266 s | 114.014 s |
| Signed cross section | 684.4008 ± 3.4636 pb | 684.0795 ± 2.3580 pb |
| Relative signed integration error | 0.5061% | 0.3447% |
| Absolute rate | 1100.9666 ± 3.8889 pb | 1101.5121 ± 2.6867 pb |
| Negative event fraction | 18.640% | 18.920% |
| Signed-weight effective events | 3933.80 | 3852.90 |
| Signed effective events / generation CPU second | 43.136 | 26.374 |
| Absolute-weight effective events | 10,000.00 | 9986.84 |

A generation trial is one complete folded observation in both codes, including
rejected and zero-weight points. Efficiency uses the final 10,000 events.
The AmpliCol spool's candidate retention rate is not used as unweighting
efficiency. CPU times sum the Fortran worker counters and exclude compilation,
preliminary checks and Python coordination; whole-launch wall time includes
those costs. These are single runs without timing error bars or CPU pinning.

The quoted cross-section errors are Monte Carlo integration errors. They do
not include scale/PDF theory uncertainty or the finite final-event sample's
sign fluctuations. AmpliCol uses the native conventional uncertainty estimator
under adaptive, nonzero-count stopping. Absolute rates and negative-weight
fractions can also depend on the independently learned virtual approximations.

## Native production and tail checks

AmpliCol used seven workers, 19 completed production iterations and 12 grid
updates. Each worker adapted the six coordinates with `ifold=1`; the folded
coordinate remained fixed (mask `1111011`). All 75,402 production observations
were nonzero in this sample. Every production iteration and each parent survey
exactly once contributed to the final rates: 152,202 total rate observations.

It wrote 30,204 candidate records, finalized an 11,004-event reserve, and
uniformly selected the final 10,000 events. The updated channel quotas changed
by at most 28 events, within the reserve. There was one collection round,
with no extra workers or collection threshold tightening.

All native and final-channel full-weight tail checks passed 1%. The largest
native worker check was 0.6105%; the largest final-channel bound was 0.5542%.
The collected sample contains 24 tail-flagged events, carrying **0.3486%**
of its absolute weight. Residual corrections remain in the LHE weights as
requested. The largest corrected magnitude is 3.1239 times the nominal
normalization; the absolute-weight effective count is 9986.84.

MINT's 72 logged envelope exceedances are counts, corresponding to 0.1452%
of generation trials. MINT accepts such points at nominal event magnitude;
it does not apply the AmpliCol residual correction or full-weight tail cap.
The weight mass of those exceedances is not logged and cannot be inferred
from their count. The two generators therefore have different tail safeguards.
Both LHE files advertise `IDWTUP=-4`; MINT's event magnitudes are nevertheless
constant in this run.

## Evidence and reproduction

`metrics.json` contains the complete analysis of the original exports.
`archived_metrics.json` reproduces the analysis from this evidence directory.
`final_checks.json` records completion, input/source comparisons, LHE hashes
and native validation. Commands, input cards, logs, result files, pool metadata,
final LHE samples and source snapshots are archived here. Large candidate LHE
spools and compiled matrix-element outputs remain in the original work directory,
recorded in `work_directory.txt`.

```sh
python validation/ttbar_native_10k_20261007/analyze.py \
  --work validation/ttbar_native_10k_20261007 \
  --run-name benchmark_10k \
  --output /tmp/ttbar_native_10k_metrics.json
```

The analysis independently recomputes native survey-plus-production means and
errors, reproduces initial/final channel allocations and seeded collection,
checks every worker tail fraction, and compares selected event magnitudes with
the final LHE. A separate read-only audit reproduced the trial/CPU counts,
rate means/errors, event counts, effective counts and collected tail fraction.

## Follow-up MINT full-tail measurement

An instrumented repeat reproduced the same event records and measured the
previously unrecorded full-weight tail: 0.933% using survey stream rates,
or 1.073% from production-rate estimates alone. The worst channel was about
1.3%; the nominal LHE tail-event share was 0.72%. See [the diagnostic report](../ttbar_mint_tail_10k_20261007/README.md) for uncertainties and sampling corrections.
