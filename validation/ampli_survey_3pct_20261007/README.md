# AmpliCol survey and full-tail validation, 2026-10-07

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


The revised workflow surveys each channel to 3% relative accuracy on its
absolute integral, allocates event quotas in Python, generates a rounded
10% reserve from the saved grids, and collects the exact requested count.
The full absolute cross section belonging to overweight points must stay
strictly below 1%, including the retained reserve and final LHE selection.

The published rate and uncertainty come from the independent survey.
Event-count and tail-stopped generation moments are diagnostics only.

## Exported-process checks

Process: `p p > e+ e- [QCD]`, loop SM, 13 TeV, `nn23nlo`, PYTHIA8 matching,
folding `2,2,2`, polynomial virtual approximation, Born spreading off,
five cores. These are matched parton-level LHE tests; no shower was run.

| Quantity | Full workflow | Generation-only restart |
| --- | ---: | ---: |
| Run name | `survey3pct` | `restart60` |
| Requested and collected events | 120 | 60 |
| Seed | 19721 | 19722 |
| `nevt_job` | 45 | 10 |
| Event normalization | `sum` | `unity` |
| Workers | 6 | 11 |
| Generated reserve | 135 | 71 |
| Generation trials | 316 | 165 |
| Generation CPU seconds | 1.619878 | 1.758168 |
| Signed survey rate [pb] | 2097.755368 | 2097.755368 |
| Survey uncertainty [pb] | 4.219168 | 4.219168 |
| Absolute survey rate [pb] | 2215.154934 | 2215.154934 |
| Maximum recorded full-tail fraction | 0 | 0 |
| Negative final events | 0 | 5 |

All eight survey channels reached an absolute relative error below 0.674%.
The input `req_acc=0.15` deliberately differs from the enforced 3% survey
target. A live observation after five completed surveys found no candidate
spools and no production jobs. Production starts after all surveys finish.

The restart forces multiple workers within channels and changes to `unity`
normalization with scale reweighting and internal weight records enabled.
All eight saved survey grid hashes remained unchanged. Both LHE files have
`IDWTUP=-4`, the correct event count and normalization, and the survey rate
in their init blocks. All restart events retain scale-reweighting records.

Small quotas in these runs led to zero overweight events. The tests below
exercise nonzero overweight tails. These checks are not an efficiency
comparison with MINT or validation of every process/shower combination.

## Recheck the evidence

From the repository root:

```sh
python validation/ampli_survey_3pct_20261007/verify_smoke.py
```

The script checks the persisted manifests and both final LHE samples. It
also independently reads and validates every restart worker's raw pool,
quota, reserve and tail metadata using the current collector. The initial
run's worker directories were reused by the generation-only restart, so
its raw pools are no longer available; its manifest and final LHE remain.
`validation.json` records that distinction.

Run banners preserve each run's cards; `drell_yan/Cards` contains the final
restart settings. `run_smoke.py` and `run_restart.py` record the exact setup,
commands and source synchronization. `started.json`, `finished.json` and
`restart_finished.json` record source hashes, timings and grid hashes.

The first run predates the collector fix for thresholds saturating at the
largest representable float. Its source hash is retained in `finished.json`.
The restart and independent evidence check use the final collector. Neither
physical smoke run reaches that numerical corner case; focused tests cover it.

`source_provenance.json` records the final source hashes and base commit.
`implementation.patch` includes tracked implementation changes and the new
collector/tests relative to that commit. It excludes this report and docs.

## Automated checks

117 focused tests passed during implementation:

| Suite | Tests |
| --- | ---: |
| `test_ampli_integrator` | 22 |
| `test_ampli_adapter` | 14 |
| `test_ampli_lhe` | 6 |
| `test_nlops_integrator_selection` | 4 |
| `test_ampli_orchestration` | 19 |
| `test_ampli_pool` | 22 |
| `test_born_spreading` | 3 |
| `test_momentum_maps` | 27 |

They were run in focused groups as the corresponding changes completed.
The combined equivalent command is:

```sh
python -m unittest \
  tests.unit_tests.fks.test_ampli_integrator \
  tests.unit_tests.fks.test_ampli_adapter \
  tests.unit_tests.fks.test_ampli_lhe \
  tests.unit_tests.fks.test_nlops_integrator_selection \
  tests.unit_tests.interface.test_ampli_orchestration \
  tests.unit_tests.various.test_ampli_pool \
  tests.unit_tests.fks.test_born_spreading \
  tests.unit_tests.fks.test_momentum_maps
```

Coverage includes analytic signed integrals, survey nonconvergence, no-event
surveys, frozen-grid production, quota allocation and splitting, generation
restarts, native corrections, the full-weight tail definition, strict 1%
rejection, uniform trimming bounds, bias inverse factors, LHE rounding,
extreme thresholds and preservation of reweighting payloads. A real compiled
native pool was also read successfully through the Python version-2 reader.
