# MINT overweight-tail measurement: NLO ttbar, 10,000 events

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


The diagnostic rerun reproduces all 10,000 event records byte for byte, in
both worker and final-file order, and again uses 49,596 generation trials.
The full overweight cross-section fraction is approximately 1% globally;
the largest channel estimate is approximately 1.3%. These fractions count
the full weight of every above-envelope observation, not just its excess.

## Measured fractions

| Quantity | Result |
| --- | ---: |
| Full tail, survey-rate-weighted stream fractions | 0.93277% |
| Approximate conditional statistical error | 0.11000 percentage points |
| Full tail, direct production-rate estimate | 1.07300% |
| Approximate conditional statistical error | 0.14500 percentage points |
| Excess-only mass, survey-rate-weighted | 0.23841% |
| Excess-only mass, direct production estimate | 0.30829% |
| Nominal final-LHE weight fraction from flagged events | 0.72000% |
| Above-envelope observations / final events | 72 / 10,000 |
| Largest observation / local envelope | 3.93091 |
| Largest channel full tail, survey-rate-weighted | 1.26714% |
| Largest channel full tail, direct production estimate | 1.31059% |
| Largest worker full tail, survey-rate-weighted | 1.32428% |

MINT has no enforced 1% full-tail limit. The observed global estimate lies
close to 1%, with finite statistical uncertainty, while the largest channel's
empirical estimate exceeds 1% under either combination. The rerun does not
establish that the true global tail lies on a particular side of 1%.

The two global estimators differ mostly because virtual-stream generation has
low statistics: only 15, 154, 176, 60 and 15 attempted virtual points across
the five channels. Their direct absolute-rate estimate totals approximately
11.29 pb, versus approximately 5.70 pb in the survey. Nonvirtual rates agree
much more closely. The quoted errors use a delta-method approximation,
conditional on frozen grids and observed sample counts; survey-rate errors
and finite-sample effects from accepted-event stopping are not included.

The nominal final-LHE fraction is smaller because MINT writes equal absolute
event weights even when the raw observation exceeds its envelope. All 72
above-envelope observations are accepted. The excess-only fractions quantify
omitted correction mass; they are not predictions of a total-rate bias, since
MINT retains the surveyed overall normalization and normalizes its accepted
stream distributions.

| Parent channel | Above-envelope events | Survey-weighted full tail | Direct full tail |
| --- | ---: | ---: | ---: |
| `P0_gg_ttx/GF1.0` | 4 | 0.6169% | 0.7002% |
| `P0_gg_ttx/GF2.0` | 25 | 0.7604% | 0.9903% |
| `P0_gg_ttx/GF3.0` | 38 | 1.2671% | 1.3106% |
| `P0_uux_ttx/GF1.0` | 3 | 0.7244% | 0.7242% |
| `P0_uxu_ttx/GF1.0` | 2 | 0.4706% | 0.8484% |

## Sampling correction and aggregation

The diagnostic captures the original local envelope `H` before MINT multiplies
it by the acceptance random number. The folded `fABS` already contains the
Vegas mapping Jacobian. Nonvirtual generation additionally samples cells in
proportion to their envelope height, so its proposal density is `q=H/Z`, where

```text
Z = product over dimensions of mean(ymax over that dimension's folded cells)
a = fABS * Z/H
full-tail observation = a * indicator(fABS > H)
excess observation = max(fABS-H, 0) * Z/H
```

Virtual proposals are flat and have `Z=H`, so `a=fABS`. A naive sum of the raw
`fABS` values would not estimate the cross-section tail under the nonuniform
proposal.

Split workers are pooled within their parent channel and stream before forming
means or ratios. The survey-weighted estimator combines each stream's measured
full-tail fraction using its surveyed absolute rate exactly once. The independent
direct estimator sums the per-stream tail means and divides by the sum of the
per-stream absolute means. It gives 11.883989 pb / 1107.551822 pb = 1.072996%.

A deterministic nonuniform two-cell test verifies these distinctions. With
`H=(1,9)` and `fABS=(2,3)`, the proposal probabilities are `(0.1,0.9)`.
The correct full tail is 40%, the excess-only fraction is 20%, while a raw
weight sum under that proposal would incorrectly give 6.897%.

## Rerun setup and checks

The settings match the preceding comparison exactly: `p p > t t~ [QCD]`,
13 TeV, stable tops, `nn23nlo`, PYTHIA8 matching without showering, folding
`(2,1,1)`, polynomial virtual approximation, Born spreading off, seed 19727,
`req_acc=0.01`, `nevt_job=2500`, average event normalization, and five cores.
All run, parameter and FKS cards are byte-identical to the baseline.

Instrumentation exists only in a fresh isolated process export. The workspace
MINT source is unchanged. It records every attempted folded observation and
its acceptance, stream, event index, original envelope, proposal normalization
and corrected absolute integration weight. It consumes no random numbers and
does not alter acceptance, grids or event weights. Exact event-record identity
confirms that it preserved the original sampled run.

The signed integration result remains 684.400832 ± 3.463631 pb and generation
efficiency remains 20.1629%. Generation used 91.395 CPU seconds, including
per-trial diagnostic output. The previous uninstrumented run used 91.195 CPU
seconds. This small timing difference is not interpreted as a performance change.

`metrics.json` contains the original analysis; `archived_metrics.json` repeats
it using only this archive. `mint/` contains all trial sidecars, worker LHE files,
final events, logs, results and cards; `reference/` contains the baseline event
records for identity checks. The exact module sources and diagnostic patch are
included alongside launch metadata. The original work directory is recorded in
`work_directory.txt`.

```sh
python validation/ttbar_mint_tail_10k_20261007/analyze.py --self-test
python validation/ttbar_mint_tail_10k_20261007/analyze.py \
  --process validation/ttbar_mint_tail_10k_20261007/mint \
  --reference validation/ttbar_mint_tail_10k_20261007/reference \
  --output /tmp/ttbar_mint_tail_metrics.json
```

The analysis verifies proposal corrections, all footer/log counters, per-event
mapping, exact baseline identity, and full-tail/excess definitions. A separate
read-only audit independently reproduced the raw-trial fractions, stream means,
worst channel and final-event identity.
