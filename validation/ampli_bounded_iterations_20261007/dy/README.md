# Folded DY validation of bounded native iterations

A generation-only restart from an isolated copy of the validated two-stage
Drell–Yan survey completed 2,000 MC@NLO hard events with the revised scheduler.
Settings: `p p > e+ e- [QCD]`, 13 TeV, `nn23nlo`, PYTHIA8 matching without
showering, folding `(2,2,2)`, `UsePolyVirtual=True`, Born spreading off,
`req_acc=-1`, `event_norm=sum`, seed 19731, `nevt_job=2500`, two cores.

| Measurement | Result |
| --- | ---: |
| Final events | 2,000 |
| Generation trials, including zeros | 12,851 |
| Production iterations / grid updates | 12 / 4 |
| Generation worker CPU | 37.5551 s |
| Negative events | 44 |
| Signed rate | 2096.3661 ± 5.1003 pb |
| Absolute rate | 2211.8595 pb |
| Largest native worker / final channel tail bound | 0.338634% |
| Collected full-weight overweight fraction | 0.068409% |

All eight workers use mask `[1,1,1,1,0,0,0]`; only the four unfolded coordinates
can adapt. All first batches contain 1,024 nonzero observations, matching
`max(1024, min(8192, generated_target))`. Four workers need a second batch;
their measured remaining-event forecasts also request 1,024 nonzero points.
The verifier reproduces those forecasts and the maximum doubling rule from
the logged remaining demand and acceptance.

Each initial storage cutoff equals `min(saved_envelope, survey_ABS)`. The
saved envelope is independently reconstructed from the survey's virtual and
nonvirtual maxima and stream probabilities. Every first-epoch envelope retains
that value as a lower bound after production. All 16 saved survey and
MC-integer checkpoint files remain byte-identical. Exported numerical and
Python source hashes match the archived snapshots before and after the run.
Only generation was executed; no survey or stage-0 jobs were launched.

Collection completed in one round, with no additional workers or threshold
increases. The archived checks reproduce survey-plus-production rates,
initial and updated allocations, seeded candidate selection, final LHE weight
magnitudes, and all strict full-trial, reserve, subset and collection tail
bounds. Native correction factors remain in the final LHE weights.

For context, the previous scheduler's generation-only restart used the same
survey, seed and physics settings and required 39,871 trials and 93.5398 worker
CPU seconds. This run used fewer trials and less worker CPU, although the
launches used different core counts and concurrent workloads. This is a
functional scheduler check, not a controlled timing benchmark.

Reproduce the archived validation from the repository root:

```sh
python validation/ampli_bounded_iterations_20261007/dy/verify.py
```

`validation.json` contains the concise results and per-worker iteration
budgets; `metrics.json` includes full rate, pool and LHE diagnostics. `process/`
contains cards, logs, checkpoints, POOL4 sidecars and the final compressed LHE.
`sources/` contains the exact implementation used. The archive omits build
products and raw candidate LHE spools; worker checkpoint symlinks refer to
archived parent channels. The isolated build remains at the path in
`work_directory.txt`.
