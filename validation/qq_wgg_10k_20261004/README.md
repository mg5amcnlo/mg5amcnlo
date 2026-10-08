# 10,000-event quark-antiquark W+gg benchmark — AmpliCol incomplete

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


MINT completed with **10,000 finite events**. The AmpliCol run is incomplete: no generator/coordinator processes were active at the final observation, and `GF3.0_5` had no completed pool after top-up round 4. There is no final AmpliCol LHE sample or completion manifest. Consequently, no final AmpliCol event efficiency or completed-sample comparison is reported. The interruption's cause is not inferred here.

The requested Born component was selected from a complete `p p > w+ j j [QCD]` export by retaining only `P0_udx_wpgg` in `subproc.mg`, while keeping the complete foreign Born support. That subprocess groups `u d~ > W+ g g` and `c s~ > W+ g g` in one beam ordering. Every rate below is for this **selected Born component**, not the inclusive physical W+2-jet cross section. [Selection](selected_component.json) and [FKS-completeness verification](verification.json) record the precise scope. The initial isolated-process export was unsuitable because it omitted a required singular limit; its stopped-run evidence is kept separately in [isolated_export_failure](isolated_export_failure/README.md) and excluded from this comparison.

Both runs use 13 TeV protons, a stable W, `nn23nlo`, PYTHIA8 MC@NLO matching without showering, folding `(2,1,1)`, seed 19721, anti-kT jets with R=0.4, pT>30 GeV and |eta|<4.5, dynamic scale choice -1, polynomial virtual optimization, native explicit MC-history summation, Born spreading disabled, average event normalization, and five cores. Scale/PDF reweighting is disabled.

The nominal absolute-rate accuracy target was matched: MINT used `req_acc=0.01`; AmpliCol used `0.03642407168790584`, obtained by multiplying 0.01 by the MINT stage-1 ABS/signed ratio. The reference target is 2.99020 pb. AmpliCol plans its fixed production budgets from its own rough survey, so this conversion matches the nominal target rather than guaranteeing identical achieved errors. See [accuracy_conversion.json](accuracy_conversion.json).

| Measurement | MINT completed sample | AmpliCol frozen initial production estimate |
|---|---:|---:|
| Selected-component signed rate [pb] | 82.093969 ± 2.137686 | 81.464732 ± 2.688404 |
| Absolute rate [pb] | 299.019662 ± 2.233252 | 307.992068 ± 2.952624 |
| Achieved relative ABS error | 0.7469% | 0.9587% |
| Final events | 10,000 | Not finalized |
| Production trials | 500,621 | At least 2,038,109; incomplete |
| Final events / production trials | 1.9975% | Not available |
| Adaptation + survey CPU minutes | 54.946 | 35.397 |
| Production CPU minutes | 42.730 | At least 166.326; incomplete |
| Total worker CPU minutes | 97.676 | At least 201.722; incomplete |
| Negative event fraction | 34.820% | Not available |
| Signed-weight effective events | 921.7 | Not available |

The signed estimates differ by 0.18 combined reported Monte Carlo standard errors. The errors are integration uncertainties, not scale/PDF theory uncertainties. The AmpliCol rate and error are frozen from 278,008 initial fixed-budget trials, costing 1,348.022 worker CPU seconds. Subsequent top-up moments are excluded from that rate estimator. The 11,782 initial provisional targets exceed 10,000 by 17.82%, because the per-channel minimum reserve is 1,000; this is not uniformly 10% oversampling.

AmpliCol's initial channel pools exceeded its enforced 1% **mean overweight excess** limit, despite containing enough candidates. Extra sampling was therefore needed for pool quality, not merely event counts. The snapshot counts all 36 completed production workers, including completed workers in the interrupted fourth top-up round. It excludes the unknown trial count and CPU already spent in the unfinished worker, so AmpliCol costs in the table are strict lower bounds, not completed-run timings. Five-core scheduling also leaves a long single-channel tail because workers are split by provisional event quota rather than by trial budget.

The stopping policies differ. MINT recorded 388 upper-bound exceedances (359 nonvirtual and 29 virtual); it accepts these at nominal event magnitude rather than carrying an AmpliCol-style residual correction. That count cannot be equated to a 1% mean excess or used to infer a bias size. AmpliCol enforces its mean-excess cap and preserves residual factors. MINT also draws from its trained local envelope, whereas the current AmpliCol bridge uses a frozen sampling grid and a global parent-channel priority threshold. These configured-workflow differences mean the timing comparison does not isolate intrinsic integrator speed.

MINT's completed LHE has `IDWTUP=-4`; its mean signed event weight is 90.782369 ± 2.849201 pb. This is an event-sampling diagnostic conditional on the ABS normalization, not an independent integration uncertainty. MINT launch wall time was 1,571.632 seconds, including compilation and coordination. Worker CPU sums exclude those costs; no final AmpliCol launch wall time is available.

[Complete MINT metrics](mint_metrics.json), [AmpliCol frozen initial moments](ampli_initial_production.json), [interrupted-run inventory and cost lower bounds](ampli_interrupted_status.json), cards, commands, source hashes, worker logs/results, and the existing implementation [patch](implementation.patch) are retained here. No core source changes were made for this benchmark. The large event/pool files remain under `/tmp/mg5-qq-wgg-complete-10k-kmsl0rms`.

The prepared analysis and verification scripts should be run only after AmpliCol completes. They count every production/top-up worker, check both 10,000-event samples, verify the intended card differences, recompute every channel's overweight excess directly from raw pool sidecars, and compare the final rates against the frozen initial snapshot:

```sh
python3 validation/qq_wgg_10k_20261004/analyze_pools.py \
  --mint /tmp/mg5-qq-wgg-complete-10k-kmsl0rms/mint \
  --ampli /tmp/mg5-qq-wgg-complete-10k-kmsl0rms/ampli \
  --run-name benchmark_10k --expected-events 10000 \
  --output validation/qq_wgg_10k_20261004/metrics.json
python3 validation/qq_wgg_10k_20261004/verify_results.py \
  --metrics validation/qq_wgg_10k_20261004/metrics.json \
  --output validation/qq_wgg_10k_20261004/final_checks.json
```
