# AmpliCol 1% overweight limit: 100,000-event ttbar rerun

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


The fresh AmpliCol run completed successfully with **100,000 finite signed events**, no bound failures, and no additional generation batches. The allowed mean overweight excess is now **0.01**, and the coordinator checks it for every parent channel before finalizing events. The final LHE file retains correction factors and advertises `IDWTUP=-4`.

The earlier candidate-pool path did not enforce the native numerical overweight tolerance. This rerun enforces the new 1% limit. Every channel's earlier excess was already below 1%, so the fixed budgets and event selection remain unchanged. The complete, ordered event blocks are **byte-identical** to the previous pool run, excluding the run banner; their SHA-256 is `579a72a072285cde3d946fe84f120ce7f2a7291b6a9ea2e7906f34af228b5355`. Rates, uncertainties, trial counts, candidate counts, and run-card hashes also match exactly.

Settings match the preceding comparison: `p p > t t~ [QCD]`, 13 TeV protons, `nn23nlo`, PYTHIA8 matching, folding `(2,1,1)`, polynomial virtual optimization, native MC counterterm sum, Born spreading disabled, dynamic scales, seed 19721, requested accuracy 0.3%, `event_norm=average`, five cores, and 2,500 provisional events per worker. Showering and scale/PDF reweighting were disabled. The MINT baseline was reused; only AmpliCol was rerun.

| Measurement | MINT baseline | Previous AmpliCol pools | AmpliCol with 1% limit |
|---|---:|---:|---:|
| Signed cross section [pb] | 683.704 ± 2.713 | 682.535 ± 1.297 | 682.535 ± 1.297 |
| Relative Monte Carlo uncertainty | 0.397% | 0.190% | 0.190% |
| Final events | 100,000 | 100,000 | 100,000 |
| Production trials | 538,210 | 440,012 | 440,012 |
| Final events / production trials | 18.58% | 22.73% | 22.73% |
| Adaptation + survey CPU minutes | 4.766 | 2.876 | 2.866 |
| Production CPU minutes | 18.628 | 17.006 | 15.108 |
| Total worker CPU minutes | 23.394 | 19.882 | 17.975 |
| Final events / production CPU second | 89.47 | 98.00 | 110.31 |
| Negative event fraction | 19.014% | 18.804% | 18.804% |
| Signed-weight effective events | 38,405 | 38,584 | 38,584 |

The fresh wall time was 274.849 seconds, including compilation and coordination. Worker CPU sums exclude those costs. The lower CPU time than the previous pool run is timing variation, not evidence of an improvement from the new limit: both runs executed exactly the same production trials and produced identical events. AmpliCol's signed rate differs from the MINT baseline by 0.39 combined Monte Carlo standard errors. These errors do not include scale or PDF theory uncertainty.

The mean excess below is the reserve average of `max(1, weight / threshold) - 1`, before normalizing correction factors. It is not the fraction of individual events above the threshold. The 1% limit applies separately to each channel after merging its workers.

| Parent channel | Final quota | Retained reserve | Mean overweight excess |
|---|---:|---:|---:|
| `P0_gg_ttx/GF3.0` | 40,235 | 44,259 | 0.637725% |
| `P0_gg_ttx/GF2.0` | 40,297 | 44,327 | 0.481810% |
| `P0_gg_ttx/GF1.0` | 8,855 | 9,741 | 0.069559% |
| `P0_uux_ttx/GF1.0` | 5,331 | 5,865 | 0.199743% |
| `P0_uxu_ttx/GF1.0` | 5,282 | 5,811 | 0.123390% |

All 46 initial production workers finished. They retained 287,136 provisional candidates from 440,012 trials; no top-up trials were needed. The published cross section uses all initial production moments. The absolute rate is 1,097.543 ± 1.471 pb. Event magnitudes range from 1,090.588 to 7,786.648 pb, with absolute-weight effective count 99,477.8 and weighted negative fraction 18.8604%. Their mean signed weight is 683.523 ± 2.727 pb; this event-sampling error is conditional on the absolute normalization and omits finite-pool correlations. The LHE `<init>` rate and error match the production manifest.

Validation includes 69 focused tests: 35 numerical/adapter/LHE tests, 16 pool tests, and 18 coordinator tests. Captured logs are [numeric_tests.log](numeric_tests.log) and [pool_tests.log](pool_tests.log); coordinator tests were also run successfully. [Implementation changes and source hashes](implementation_changes.json), [metrics](metrics.json), [completion checks](final_checks.json), cards, worker logs/results, launch metadata, and production manifests are retained here.

The large final LHE file remains at `/tmp/mg5-ttbar-overweight-1pct-u7q06hxa/ampli/Events/benchmark_100k/events.lhe.gz`. Analysis can be repeated with:

```sh
python3 validation/ttbar_pool_overweight_1pct_20261004/analyze_ttbar_pools.py \
  --mint /tmp/mg5-ttbar-100k-rd5zxvbr/mint \
  --ampli /tmp/mg5-ttbar-overweight-1pct-u7q06hxa/ampli \
  --output validation/ttbar_pool_overweight_1pct_20261004/metrics.json
python3 validation/ttbar_pool_overweight_1pct_20261004/verify_results.py \
  --metrics validation/ttbar_pool_overweight_1pct_20261004/metrics.json \
  --previous-metrics validation/ttbar_pool_100k_20261004/metrics.json \
  --output validation/ttbar_pool_overweight_1pct_20261004/final_checks.json
```
