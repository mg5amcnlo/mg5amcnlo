# NLO ttbar: 300,000 events, automatic accuracy, no folding

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


Completed 2026-10-07. Both generators produced 300,000 events with
`req_acc=-1` and folding `(1,1,1)`. AmpliCol used 37.2% more generation CPU and
22.5% more total worker CPU, with a 66.0% smaller quoted integration uncertainty.
The rates agree within 0.234 errors combined in quadrature.

MINT's measured global full overweight fraction is 0.342%, with all parent
channels below 1%; the largest individual split-worker estimate is 1.31%.
AmpliCol passed every native and collection 1% check, and its collected
overweight events carry 0.487% of the corrected absolute weight.

## Configuration and automatic accuracy

`p p > t t~ [QCD]`, `loop_sm`, 13 TeV, stable tops of mass 173 GeV,
`nn23nlo`, dynamic scale choice `-1`, PYTHIA8 MC@NLO matching without showering.
Five cores, seed 19727, `nevt_job=2500`, `event_norm=average`,
`UsePolyVirtual=True`, Born spreading off, scale/PDF reweighting and internal
reweight information off.

Relative to the preceding unfolded 10K ttbar check, the only run-card changes
are `nevents=300000` and **`req_acc=-1`**. Parameter and FKS cards are unchanged
for each backend. MINT and AmpliCol differ only in `NLOPSIntegrator`.
All eight exported numerical source hashes are unchanged from that benchmark.
The unrun matrix-element export was reused; grids and integration results were
created afresh for each run.

Automatic MINT accuracy uses `1/sqrt(nevents)` as the global absolute-rate target:
0.182574% for 300,000 events. Per-channel targets follow the code's allocation
formula. The achieved relative absolute-rate error is 0.166870%.
AmpliCol retains its prescribed 3% per-channel survey and refines estimates
during quota-driven adaptive production. Its final error uses the survey once
and every production epoch. Thus the backends use their implemented workflows,
with different achieved precision; this is not an equal-precision timing test.

An initial attempt with `req_acc=0.01` was stopped when the user corrected the
setting. Its metadata is isolated under `superseded_req_acc_001/`; none of its
results or timing contributes to this report. Both completed runs and the MINT
diagnostic replay use `req_acc=-1`.

## Rates and efficiency

| Quantity | MINT | AmpliCol |
|---|---:|---:|
| Final events | 300,000 | 300,000 |
| Production workers | 122 | 134 |
| Generation trials | 2,419,293 | 3,021,464 |
| Final events / all generation trials | 12.4003% | 9.9290% |
| Zero generation trials | 165 | 57 |
| Grid adaptation CPU | 31.768 s | 60.355 s |
| Survey / envelope CPU | 328.657 s | 57.162 s |
| Generation CPU | 2206.902 s | 3028.574 s |
| Total worker CPU | 2567.328 s | 3146.091 s |
| Whole-launch wall time | 623.902 s | 700.584 s |
| Signed cross section | 681.48155 ± 1.68937 pb | 681.89983 ± 0.57503 pb |
| Absolute cross section | 1157.11858 ± 1.93088 pb | 1156.52020 ± 0.67980 pb |
| Relative signed integration uncertainty | 0.24790% | 0.08433% |
| Negative events | 61,777 (20.5923%) | 61,313 (20.4377%) |
| Negative absolute-weight fraction | 20.5923% | 20.4778% |
| Signed effective events, `(sum w)^2 / sum w^2` | 103,777.30 | 104,368.20 |
| Absolute-weight effective events | 300,000.00 | 299,371.30 |
| Signed effective events / generation CPU second | 47.02 | 34.46 |
| Signed effective events / total worker CPU second | 40.42 | 33.17 |

The rate difference is 0.41828 pb, with a combined error of 1.78455 pb.
MINT's reported estimate uses 197,120 stage-1 integration points. AmpliCol's
estimate includes 76,800 survey points and 3,021,464 production trials.
Its use of production observations contributes to the smaller rate uncertainty.

Errors are Monte Carlo integration uncertainties, not theory uncertainties.
Adaptive and nonzero-count stopping give conventional estimated errors rather
than exact finite-sample coverage. Timing results are single runs. Worker CPU
excludes compilation and Python coordination; whole-launch wall time includes
them. All backends ran sequentially on the same machine.

The generation efficiency includes all rejection and reserve costs. MINT
retains equal absolute LHE event weights. AmpliCol preserves native correction
factors and marks its LHE as weighted; its largest normalized correction is
12.14287. The correction's magnitude is not limited to 1%; the limit applies
to the full overweight cross-section fraction.

## Full overweight measurements

The full tail is the entire absolute cross section associated with points above
the applicable unweighting envelope, divided by the total absolute cross
section. It differs from both the fraction of trials above the envelope and the
excess-only weight above that envelope.

| MINT diagnostic | Result |
|---|---:|
| Full tail combined using survey stream rates | 0.341973% ± 0.013628 percentage points |
| Full tail using production stream means | 0.350102% ± 0.014195 percentage points |
| Largest parent-channel full tail | 0.821874% (`P0_uxu_ttx/GF1.0`) |
| Largest split-worker full-tail estimate | 1.308424% ± 0.271582 percentage points (`P0_uxu_ttx/GF1.0_4`) |
| Excess-only cross-section fraction using survey rates | 0.092960% |
| Above-envelope trials / accepted events | 755 / 300,000 |
| Above-envelope share of nominal final LHE absolute weight | 0.251667% |
| Largest weight / envelope ratio | 10.19243 |

The per-parent full-tail estimates are:

| Parent channel | Survey-weighted full tail |
|---|---:|
| `P0_gg_ttx/GF1.0` | 0.431171% |
| `P0_gg_ttx/GF2.0` | 0.288036% |
| `P0_gg_ttx/GF3.0` | 0.271503% |
| `P0_uux_ttx/GF1.0` | 0.706414% |
| `P0_uxu_ttx/GF1.0` | 0.821874% |

All five parent channels and ten positive virtual/nonvirtual streams were
sampled. The diagnostic records the pre-acceptance envelope and makes no extra
random-number calls. Every worker and final event sequence exactly reproduces
the pristine MINT run. MINT timing above comes from the pristine run, excluding
diagnostic I/O.

For MINT's envelope proposal `q=H/Z`, the absolute observation is `a=fABS*Z/H`;
virtual trials have `Z=H`. The full-tail observation is `a * I[fABS>H]`.
Split workers are pooled within each parent channel and stream before combining
rates; each survey stream rate is used once. The production cross-check sums
separately estimated stream means. Both methods include rejected and zero trials.

Tail errors are approximate delta-method errors conditional on grids, trial
counts and survey rates; they exclude survey errors and finite accepted-count
stopping effects. MINT does not enforce a 1% tail limit. The excess-only number
is omitted correction mass, not a net total-rate bias: the sample retains its
surveyed normalization and equal nominal weight magnitudes.

AmpliCol passed every final full-trial, reserve-sample, possible-subset and
collection check. The largest native worker check is **0.988734%**; the largest
final parent-channel collection bound is **0.888354%**. The selected sample has
**1,055 tail-flagged events** carrying **0.487096%** of the corrected absolute
weight. These empirical checks do not guarantee a bound on unseen phase space.
The maximum worker check and MINT's global estimate have different aggregation.
Actual collected-weight tail shares are 0.487% for AmpliCol and 0.252% for MINT;
the latter differs from its full cross-section tail because MINT omits native
overweight corrections.

## Adaptation, collection and validation

AmpliCol kept all seven coordinates adaptive. Its 134 workers completed 438
native epochs and 304 grid updates, scheduled using nonzero observations.
There were 916,048 stored candidates, 330,065 reserve events, and 300,000 final
events. The reserve includes integer rounding across split workers. Collection
finished in one round without top-ups or rethresholding. The largest updated
parent-channel quota change was 1,051 events.

The main analyzers and independent standard-library-only audits reproduce the
rates, uncertainties, CPU sums, native thresholds and correction factors, initial
and updated allocations, seeded collection, final weight magnitudes, MINT stream
coverage and exact event replay. Analyses from the archive reproduce the original
numerical results. `verify.py` also checks automatic accuracy, card differences,
source hashes and all completed event counts.

Four MINT trials have tiny signed/absolute cancellation residuals. The largest
is 6.11e-10 in raw observation units, or 2.36e-12 of the local proposal scale;
none is overweight. The diagnostic consistency tolerance was changed from
1e-12 to 1e-11 of that scale and every residual is recorded. No observation,
tail decision, integration or event-generation code was changed.
See `mint_roundoff_diagnostics.json` for the complete scan.

The preceding 10K ttbar test used fixed `req_acc=0.01`; this run uses automatic
accuracy. Their timing and efficiency differences therefore reflect both event
count and integration/envelope accuracy, rather than event-count scaling alone.

## Evidence and reproduction

Detailed results are in `metrics.json`, `mint_tail_metrics.json`, their
`archived_*.json` counterparts, `independent_*_audit.json`, and
`final_checks.json`. Cards, logs, grids, pool metadata, complete trial sidecars,
final and worker MINT LHEs, source snapshots and the diagnostic patch are saved.
`generate.cmd` and `generate.log` describe the original reused matrix-element
export, whose original output path belongs to the earlier ttbar benchmark.

From the repository root:

```bash
python validation/ttbar_unfolded_300k_20261007/analyze.py \
  --work validation/ttbar_unfolded_300k_20261007 \
  --run-name benchmark_300k --expected-events 300000 \
  --output /tmp/ttbar-300k-metrics.json
python validation/ttbar_unfolded_300k_20261007/analyze_mint_tail.py \
  --process validation/ttbar_unfolded_300k_20261007/mint_tail \
  --reference validation/ttbar_unfolded_300k_20261007/mint \
  --run-name benchmark_300k --reference-run-name benchmark_300k \
  --expected-events 300000 --output /tmp/ttbar-300k-tail.json
python validation/ttbar_unfolded_300k_20261007/independent_native_audit.py \
  validation/ttbar_unfolded_300k_20261007
python validation/ttbar_unfolded_300k_20261007/independent_mint_audit.py \
  validation/ttbar_unfolded_300k_20261007
python validation/ttbar_unfolded_300k_20261007/verify.py
```

The verification also refers to the preceding 10K archive and current workspace
source hashes. To rerun generation, place `run_benchmark.py`, `run_sequence.py`
and the diagnostic module in a fresh work directory containing an unrun export
named `base`, then execute `run_sequence.py`. Exact settings and paths are saved
in `experiment.json` and wrapper metadata.
