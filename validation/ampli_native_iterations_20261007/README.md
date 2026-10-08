# Native AmpliCol production workflow validation, 2026-10-07

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


The revised MC@NLO backend retains one channel per executable and a separate
3% absolute-rate survey. Production now uses nonzero-point iterations, adapts
only unfolded coordinates, reconstructs historical-grid envelopes and selects
from the most recent eight event-producing iterations. Every production
iteration contributes to evolving integration estimates. Python combines the
survey once, reallocates channel quotas and checks the full-weight 1% tail
bound before uniform collection.

## Final-source generated-process check

Process: `p p > e+ e- [QCD]`, loop SM, 13 TeV, `nn23nlo`, PYTHIA8 matching,
folding `2,2,2`, polynomial virtual approximation, Born spreading off,
`event_norm=sum`, `nevt_job=2000`, five cores. No shower was run.

A fresh full workflow created the survey with seed 19725. Review subsequently
fixed channel ordering in the allocation CDF and refined the forecast of
remaining generation work. The recorded validation run uses the final sources
and seed 19726 in a generation-only restart from that survey.

| Quantity | Final-source run |
| --- | ---: |
| Requested and collected events | 6,000 |
| Worker jobs | 8 |
| Completed production iterations | 15 |
| Production grid updates | 7 |
| Generation trials, including zero points | 31,683 |
| Generation CPU seconds | 85.2145 |
| Negative events | 177 |
| Largest final channel tail bound | 0.5214% |
| Collected full-weight overweight fraction | 0.1662% |
| Survey signed rate | 2091.074397 ± 4.258640 pb |
| Updated signed rate | 2091.314930 ± 3.735540 pb |
| Survey absolute rate | 2212.457862 pb |
| Updated absolute rate | 2214.534565 pb |

All workers have adaptation mask `[1,1,1,1,0,0,0]`. Four unfolded coordinates
can adapt and the three folded coordinates remain fixed. All eight saved survey
grids and eight companion MC-integer grid files stayed byte-identical.
Channel quotas changed by at most five events; the initial 10% reserves covered
all changes. This run needed neither collection threshold increases nor extra
workers. Those paths are covered by orchestration tests, including a 400-event
case where a reduced quota requires threshold tightening and additional workers.

The LHE file contains residual corrections in `XWGTUP` and has `IDWTUP=-4`.
Reported errors follow the native adaptive Monte Carlo convention; nonzero-count
stopping is not claimed to give an exactly unbiased finite-sample estimator.
This is a functional check, not a MINT performance comparison.

## Independent verification

From the repository root:

```sh
python validation/ampli_native_iterations_20261007/run/verify.py
```

The verifier independently combines the survey and production moments, checks
both initial and updated quota allocations using the recorded channel order,
replays seeded candidate selection, and compares selected magnitudes with the
final LHE file. It verifies rate/error normalization in the LHE init block,
all tail diagnostics, event count, negative weights, folded-coordinate masks,
unchanged saved grids and source fingerprints. Results are written to
`run/validation.json`.

`run/` contains the final run commands, logs, cards, POOL4 sidecars, survey
checkpoints, final LHE sample and exact source snapshots. The candidate LHE
spools remain in the original temporary export; the numerical pool metadata
needed for verification is archived here. `survey_setup/` records the preceding
fresh export and full workflow with its earlier source snapshots. Its production
results are not the final-source validation result above.
`source_provenance.json` identifies the branch, base commit and original
working directory; `implementation.patch` captures the implementation and tests
relative to that base, including preceding uncommitted backend changes.

## Focused automated checks

116 tests passed against the final sources:

| Suite | Tests |
| --- | ---: |
| Sampler | 34 |
| MC@NLO adapter | 16 |
| Pool reader and collector | 33 |
| Python orchestration | 23 |
| LHE payload handling | 6 |
| Backend selection | 4 |

```sh
python -m unittest \
  tests.unit_tests.fks.test_ampli_integrator \
  tests.unit_tests.fks.test_ampli_adapter \
  tests.unit_tests.various.test_ampli_pool \
  tests.unit_tests.interface.test_ampli_orchestration \
  tests.unit_tests.fks.test_ampli_lhe \
  tests.unit_tests.fks.test_nlops_integrator_selection -q
```

Tests include analytic signed integration and event density across grid updates,
independent historical Jacobian/envelope reconstruction, rejection and zero-point
accounting, nine-iteration history with expired events retained in the rate,
unchanged folded maps, storage-threshold floors, full-tail versus excess-weight
definitions, bias inverse factors, exactly-once survey accounting, quota
redistribution, threshold tightening, targeted top-ups and stable channel CDF
ordering across reports and collection rounds.
