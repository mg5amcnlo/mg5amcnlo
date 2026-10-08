# AmpliCol generation scheduler correction

> Artifact availability: see [the validation archive policy](../ampli_artifacts.md). Large raw samples, pools and grids are retained locally and are not included in Git.


The two-stage workflow is preserved: each channel surveys its absolute rate
to 3% with at least four iterations, then generates from its saved grids and
maxima. The correction prevents a large surveyed maximum from delaying
generation-time adaptation and removes compulsory doubling near completion.

## Implementation

Only `Template/NLO/SubProcesses/simple_integrator.f90` changes production
behavior in this correction. The adapter's survey, Python allocation and
collection, MINT and fixed-order code are unchanged. `scheduler.patch` records
the numerical source changes relative to the stopped 300K run.

- First iteration: request the worker's reserve quota clamped to 1,024–8,192
  nonzero points, also respecting the emergency trial limit. The saved maximum
  does not determine this budget.
- Preserve the saved maximum as an envelope floor for the first proposal,
  which uses the survey grids. Later proposals estimate their own envelopes.
- Store candidates initially using `min(saved_maximum, survey_absolute_rate)`.
  This separates candidate retention from the rejection envelope. Subsequent
  thresholds cannot fall below the storage cutoff recorded for each iteration.
- Forecast later point requests from remaining demand and measured surviving
  event efficiency, with 10% headroom. Doubling is an upper growth limit;
  requests can shrink to the 1,024-point floor near completion.
- Include events about to expire from the eight-iteration selection window
  in the demand forecast. There is no permanent upper batch-size cap.
- Recover sparse candidate storage by reducing its cutoff using the observed
  conditional nonzero absolute mean when at most 200 candidates are retained.

All attempted points still enter the integration estimates. Full nonzero
iterations complete before adaptation. Coordinates with `ifold=1` can adapt;
folded maps and auxiliary physics state remain fixed. The 1% full-weight tail
checks for trials, reserves, possible subsets and final collection are
unchanged. Native correction factors remain in the `IDWTUP=-4` LHE weights.

## Physical checks

| Measurement | Unfolded ttbar | Folded Drell–Yan |
| --- | ---: | ---: |
| Final events | 10,000 | 2,000 |
| Generation trials, including zeros | 108,662 | 12,851 |
| Production iterations / grid updates | 47 / 40 | 12 / 4 |
| Generation worker CPU | 106.794 s | 37.555 s |
| Signed cross section [pb] | 680.6122 ± 2.8253 | 2096.3661 ± 5.1003 |
| Absolute cross section [pb] | 1158.9065 ± 3.3327 | 2211.8595 |
| Largest native tail bound | 0.872923% | 0.338634% |
| Collected full overweight fraction | 0.594284% | 0.068409% |
| Collection rounds | 1 | 1 |

Both runs use MC@NLO hard events with PYTHIA8 matching; the shower is not run.
The overweight definition is the full absolute weight belonging to events
above their iteration threshold, rather than just their excess weight.

### Fresh ttbar check

`p p > t t~ [QCD]`, 13 TeV, `nn23nlo`, folding `(1,1,1)`, automatic
`req_acc=-1`, `nevt_job=2500`, seed 19727, polynomial virtual approximation,
three cores. Survey CPU: 60.048 s; total survey plus generation CPU: 166.842 s.
The launch took 110.551 wall seconds, including compilation and coordination.
Exactly five survey channels and seven production workers ran, with no stage 0.

Every saved survey grid, MC-integer grid and maximum is byte-identical to
the corresponding state in the stopped 300K benchmark. The most problematic
channel has saved maximum / absolute integral = 75.2613. Its first worker now
requests 2,271 nonzero points; the previous formula would request 170,919
points for the same reserve quota. It completes generation in 19,399 total
trials. Sixteen later iterations across the run shrink their point request.

The reserve contains 11,005 events; updated rates change channel quotas by at
most 36 events, covered without extra generation. The final sample contains
2,075 negative events. Its weighted rate, 678.1836 pb, is consistent with the
integration estimate given the event sampling fluctuation (about 9.41 pb).

The verifier reproduces rates from all iterations, initial and final quotas,
seeded selection, final LHE weight magnitudes, every tail bound and the point
forecasts. It verifies source fingerprints and the unchanged survey state.
`ttbar/ampli/` contains cards, logs, checkpoints, pool metadata and the final
compressed LHE; `archive_sha256.json` checksums all 142 archived files.

The stopped 300K run remains stopped. This smaller test establishes that the
pathological survey state now works, but does not supply a new 300K runtime
or a controlled MINT comparison.

A subsequent user-requested fresh 300K rerun completed successfully; see the
[full benchmark report](../ttbar_bounded_300k_20261007/README.md).

### Folded Drell–Yan restart

The isolated generation-only restart uses the previous validated survey,
folding `(2,2,2)`, automatic accuracy, seed 19731 and two cores. All 16 survey
checkpoint files remain unchanged. Only the four unfolded coordinates adapt.
The previous scheduler used 39,871 trials and 93.5398 CPU seconds for the same
survey, seed and requested sample. Different core counts and concurrent
workloads limit the timing comparison. See [the DY report](dy/README.md).

## Automated checks and reproduction

128 focused tests passed: 40 integrator, 19 adapter, 33 pool/collection,
26 orchestration, six LHE and four backend-selection tests.
The standalone integrator compiles with `-O1 -g -fcheck=all,no-recursion
-ffpe-trap=invalid,zero,overflow -fbacktrace`. Its final run completed 40 tests
in 6.722 seconds; output is retained in the tool transcript. Adapter and
collection/interface output is saved in the adjacent test logs.

```sh
python -m unittest tests.unit_tests.fks.test_ampli_integrator -v
python -m unittest tests.unit_tests.fks.test_ampli_adapter
python -m unittest tests.unit_tests.fks.test_ampli_lhe \
  tests.unit_tests.various.test_ampli_pool \
  tests.unit_tests.interface.test_ampli_orchestration \
  tests.unit_tests.fks.test_nlops_integrator_selection
python validation/ampli_bounded_iterations_20261007/ttbar/verify.py
python validation/ampli_bounded_iterations_20261007/dy/verify.py
```

`ttbar/run_smoke.py` can launch a fresh 10K check; it consumes the preserved
prelaunch cards from the stopped benchmark. Both check directories include
exact source snapshots and machine-readable metrics. The compact archives
omit build files and candidate LHE spools. Full temporary build paths are
recorded in their `work_directory.txt` files.
