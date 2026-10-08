# NLO e+ ve j: 10,000 events without folding

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


Completed 2026-10-07. Both MINT and AmpliCol produced 10,000 events for
`p p > e+ ve j [QCD]`, with folding `(1,1,1)`. MINT was faster, but its measured
full overweight cross-section fraction was 4.28%. AmpliCol passed all its 1%
tail checks and achieved a smaller integration uncertainty.

## Configuration

Fresh `loop_sm` matrix-element export, 13 TeV proton collisions, `nn23nlo`,
dynamic scale choice `-1`, PYTHIA8 MC@NLO matching without showering. One
positron/neutrino flavor, finite W width 2.0476 GeV. The default proton and jet
definitions contain gluons, u/d/s/c and their antiparticles, excluding b.
Jets use kT with R=0.7 and pT>10 GeV; no jet rapidity or lepton cuts.

As in the preceding unfolded ttbar check: 10,000 events, seed 19727, five cores,
`nevt_job=2500`, `req_acc=0.01`, `event_norm=average`, `UsePolyVirtual=True`,
Born spreading off, scale/PDF reweighting and internal reweight information off.
The ttbar-specific zero-top-width override was removed; there are no external
tops in this process. Full cards and source hashes are archived.

Six subprocess groups contain 12 integration channels, each kept separate.
All runs started from the same unrun export, with fresh grids and integration.
They ran sequentially: pristine MINT, AmpliCol, then diagnostic MINT.
The diagnostic changes only the exported MINT module to record every production
trial; it reproduces all worker and final event sequences exactly.
Production source files were not changed for this benchmark.

## Rates, timing and efficiency

| Quantity | MINT | AmpliCol |
|---|---:|---:|
| Final events | 10,000 | 10,000 |
| Generation trials | 279,632 | 1,674,228 |
| Final events / all generation trials | 3.5761% | 0.5973% |
| Zero generation trials | 40,422 | 339,764 |
| Grid adaptation CPU | 286.639 s | 204.859 s |
| Survey / envelope CPU | 484.986 s | 179.587 s |
| Generation CPU | 223.973 s | 1322.816 s |
| Total worker CPU | 995.598 s | 1707.262 s |
| Whole-launch wall time | 311.583 s | 506.092 s |
| Signed cross section | 5660.228 ± 55.513 pb | 5709.932 ± 25.109 pb |
| Absolute cross section | 13156.465 ± 64.844 pb | 13246.242 ± 28.088 pb |
| Relative signed integration uncertainty | 0.9807% | 0.4397% |
| Negative-event fraction | 28.03% | 29.02% |
| Negative absolute-weight fraction | 28.03% | 29.101% |
| Signed effective events, `(sum w)^2 / sum w^2` | 1930.72 | 1743.51 |
| Absolute-weight effective events | 10,000.00 | 9979.77 |

AmpliCol uses **5.91 times the generation CPU** and **1.71 times the total
worker CPU**, while quoting a **54.8% smaller signed-rate error**. Its adaptation
and survey cost is about half MINT's; production accounts for the slowdown.
Signed effective-event throughput per generation CPU is 15.3% of MINT's.
The rates differ by 49.70 pb, or 0.816 errors combined in quadrature.

These compare the implemented workflows, not equal achieved precision or equal
overweight control. MINT uses its normal absolute-rate accuracy target and
reports stage-1 integration. AmpliCol surveys each channel to 3% absolute-rate
accuracy, then continues estimating the integral during production. Its final
estimate includes each survey once and all subsequent iterations. Integration
errors are conventional Monte Carlo estimates, not theory uncertainties or
exact finite-sample coverage under adaptive stopping.

Worker CPU excludes compilation and Python coordination; launch wall time
includes them. Timings are single runs. Efficiency counts final retained events
over every attempted generation point, including the cost of reserves and
rejections. AmpliCol preserves native correction factors in weighted LHE output,
so its event magnitudes are not exactly equal.

## Full overweight check

The full tail is the entire absolute cross section carried by points above the
applicable unweighting envelope, divided by the total absolute cross section.
It is different from the fraction of trials above the bound or just their excess
weight above that bound.

| MINT diagnostic | Result |
|---|---:|
| Full tail combined using survey stream rates | 4.28057% ± 0.33139 percentage points |
| Full tail using production stream means | 4.32002% ± 0.33581 percentage points |
| Largest parent-channel full tail | 5.57792% (`P0_dxu_veepg/GF1.0`) |
| Excess-only cross-section fraction using survey rates | 1.88972% |
| Above-envelope points / accepted events | 243 / 10,000 |
| Above-envelope share of nominal final LHE absolute weight | 2.43000% |
| Largest weight / envelope ratio | 13.57803 |

All 12 parent channels and all 24 positive virtual/nonvirtual streams were
sampled. The proposal-corrected absolute observation is `a=fABS*Z/H`, with
envelope `H` and proposal normalization `Z`; virtual trials have `Z=H`.
The full-tail observation is `a * I[fABS>H]`. Stream ratios are combined using
each survey absolute rate once. The alternative estimate sums independently
estimated production stream means. This accounts for MINT's nonuniform envelope
proposal, including rejected and zero trials.

Tail errors are approximate delta-method errors conditional on the observed
trial counts, grids and survey rates; they exclude survey errors and finite
accepted-event stopping effects. The excess-only number is omitted correction
mass, not a net total-rate bias. MINT retains the survey normalization and equal
nominal LHE magnitudes, giving a 2.43% collected tail share despite a measured
full cross-section tail above 4%. MINT does not enforce the 1% full-tail limit.

AmpliCol passed every final native full-trial, reserve, possible-subset and
collection check. Its largest check was **0.874903%**. The collected sample has
24 tail-flagged events carrying **0.399029%** of the corrected absolute weight.
These are empirical checks on observed points, not guarantees about unseen
phase space. The largest worker check has a different aggregation from MINT's
global tail; the actual collected-weight tail shares are 0.399% and 2.43%,
respectively.

## Adaptation, collection and independent audit

All ten coordinates adapted. The 12 workers completed 71 native iterations and
59 grid updates, scheduled by nonzero points. The largest survey relative
absolute-rate error was 2.633%. Production sampled 1,334,464 nonzero points and
339,764 zeros; there were 114,984 stored candidates and 11,005 reserve events.
The reserve differs from 11,000 because of integer rounding across workers.
Collection completed in one round without top-ups or rethresholding; the largest
updated channel quota change was 15 events.

Both the regular analyzer and a separate standard-library-only audit reproduce
rates, native thresholds, tail fractions, correction factors, seeded collection
and all final weight magnitudes. The MINT audit independently verifies its tail,
complete stream coverage, counters and exact event replay. Reanalyzing archived
files reproduces the original numerical results. All final consistency checks
passed; that does not mean MINT passes the overweight limit.

The only analysis-script adjustment from ttbar allows and records tiny signed/
absolute cancellation residuals in 23 MINT rows. Their maximum excess is
1.77e-8 in raw observation units, or 2.22e-13 of the local proposal scale. None
is overweight. The permitted residual is at most 1e-12 of that scale, and the
raw observations and tail decisions are unchanged. The independent audit
confirmed the residuals arise from different summation orders in the signed
and absolute subtraction contributions. No integration or generation code was
modified to accommodate them.

## Evidence and reproduction

`metrics.json`, `mint_tail_metrics.json`, the corresponding `archived_*.json`,
`independent_*_audit.json`, and `final_checks.json` hold detailed results.
Cards, logs, results, grids, pool metadata, all-trial sidecars, final LHEs,
worker MINT LHEs, source snapshots and diagnostic patch are included.

From the repository root:

```bash
python validation/epvej_unfolded_10k_20261007/analyze.py \
  --work validation/epvej_unfolded_10k_20261007 \
  --run-name benchmark_10k --output /tmp/epvej-metrics.json
python validation/epvej_unfolded_10k_20261007/analyze_mint_tail.py \
  --process validation/epvej_unfolded_10k_20261007/mint_tail \
  --reference validation/epvej_unfolded_10k_20261007/mint \
  --output /tmp/epvej-tail.json
python validation/epvej_unfolded_10k_20261007/independent_native_audit.py \
  validation/epvej_unfolded_10k_20261007
python validation/epvej_unfolded_10k_20261007/independent_mint_audit.py \
  validation/epvej_unfolded_10k_20261007
python validation/epvej_unfolded_10k_20261007/verify.py
```

`verify.py` also checks the current workspace against saved source hashes.
To rerun generation, use `generate.cmd` with an output named `base` in a fresh
work directory. Put `run_benchmark.py`, `run_sequence.py` and the diagnostic
module there, then execute `run_sequence.py`. The original work path and exact
launch settings are preserved in `experiment.json` and wrapper metadata.
