# 100,000-event ttbar comparison after candidate-pool implementation

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


Both backends produced exactly **100,000 finite signed LHE events** for `p p > t t~ [QCD]`. The new AmpliCol path completed without a bound failure or additional generation batches. It updated the physical cross section from all 440,012 production trials, then assigned final channel quotas and selected events from provisional pools.

Identical physics/run settings: 13 TeV proton collisions, internal `nn23nlo` PDF, PYTHIA8 matching, folding `(2,1,1)`, polynomial virtual optimization enabled, native MC counterterm sum enabled, Born spreading disabled, dynamic scales, seed 19721, requested accuracy 0.3%, `event_norm=average`, five cores, 2,500 provisional events per worker. Showering and scale/PDF reweighting were disabled for this timing comparison. Cards differ only in `#NLOPSIntegrator`.

| Measurement | MINT | AmpliCol pools |
|---|---:|---:|
| Physical signed cross section [pb] | 683.704 ± 2.713 | 682.535 ± 1.297 |
| Relative Monte Carlo uncertainty | 0.397% | 0.190% |
| Final events | 100,000 | 100,000 |
| Production trials, including additional batches | 538,210 | 440,012 |
| Final events / production trials | 18.58% | 22.73% |
| Adaptation + survey CPU minutes | 4.766 | 2.876 |
| Production CPU minutes | 18.628 | 17.006 |
| Total worker CPU minutes | 23.394 | 19.882 |
| Final events / production CPU second | 89.47 | 98.00 |
| Negative event fraction | 19.014% | 18.804% |
| Signed-weight effective events | 38,405 | 38,584 |

The physical rates differ by **0.39 combined statistical standard errors**. AmpliCol uses 15.0% less total worker CPU and yields 9.5% more finalized events per production CPU second. Its quoted rate uncertainty is 52.2% smaller; this includes the benefit of estimating the rate from generation trials, whereas the MINT estimate remains its stage-1 result. CPU figures sum each completed worker's own timer, include all production workers, and exclude compilation and Python coordination. Wall times are not compared because these runs were not simultaneous controlled timing repetitions.

AmpliCol retained 287,136 provisional candidates (65.26% retention), from 46 workers, and selected the required 100,000 events after combining workers by parent channel. Final quotas needed no additional batches. Candidate retention and final-event efficiency are different quantities; the table uses final events throughout.

Residual AmpliCol weight corrections are preserved. Both final files advertise `IDWTUP=-4`. MINT event magnitudes are 1,108.4612 pb; AmpliCol magnitudes range from 1,090.588 to 7,786.648 pb. AmpliCol's absolute-weight effective count is 99,478, a 0.52% reduction relative to 100,000 equal-magnitude events. Its weighted negative fraction is 18.860%. These are weighted finite-pool events, not a claim of exact unit-weight rejection sampling. MINT logs 499 upper-bound exceedances in this baseline; this comparison does not quantify their distributional effect.

The absolute generation targets are MINT **1,108.461 ± 2.810 pb** and AmpliCol **1,097.543 ± 1.471 pb** (3.44 combined errors apart). They depend on each run's independently trained virtual approximation and stream decomposition and are not invariant physical cross sections. The physical comparison is the signed rate above.

As a normalization check, the mean finalized event weights give **686.936 ± 2.751 pb** for MINT and **683.523 ± 2.727 pb** for AmpliCol. These event-sampling errors use the sample variance conditional on the integrated absolute normalization; they exclude its uncertainty and finite-pool correlations. Both agree with their integration estimates. AmpliCol's LHE `<init>` signed rate and uncertainty exactly match its frozen production manifest. Reweighting and generation-only restart checks are recorded separately in [the Drell–Yan validation](drell_yan/README.md).

The benchmark used an exported snapshot of the candidate-pool implementation, identified by [source hashes](source_hashes.json), on top of commit `7fa5120a4`. It predates the subsequent fixes for an empty first worker's header, the compact per-channel header event count, and coordinator CPU summaries. This ttbar run had no empty workers, requested no reweighting, and the final collector replaced the compact header, so these later fixes do not alter its final sample. CPU measurements here come directly from worker logs, independently of the coordinator summary. The final automated suite passed 98 tests; see [the captured log](automated_tests.log).

[Machine-readable metrics](metrics.json), [completion checks](final_checks.json), cards, launch metadata, channel logs/results, and production manifests are retained here; large LHE payloads remain in `/tmp/mg5-ttbar-pools-2kmutl1p/ampli` and `/tmp/mg5-ttbar-100k-rd5zxvbr/mint`. The read-only analysis is reproducible with:

```sh
python validation/ttbar_pool_100k_20261004/analyze_ttbar_pools.py \
  --mint /tmp/mg5-ttbar-100k-rd5zxvbr/mint \
  --ampli /tmp/mg5-ttbar-pools-2kmutl1p/ampli \
  --output validation/ttbar_pool_100k_20261004/metrics.json
```

The final implementation is recorded separately in [implementation.patch](implementation.patch) and [implementation_source.json](implementation_source.json), including the later header fixes validated by Drell–Yan.
