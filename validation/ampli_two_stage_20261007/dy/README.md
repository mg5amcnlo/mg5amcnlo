# Folded Drell–Yan two-stage validation, 2026-10-07

A fresh `p p > e+ e- [QCD]` export completed 2,000 MC@NLO hard events,
followed by a separate 2,000-event `--only_generation` restart from its saved
survey. Both runs use folding `(2,2,2)`, `UsePolyVirtual=True`, `req_acc=-1`,
13 TeV, the built-in `nn23nlo` PDF, PYTHIA8 matching, Born spreading off,
`event_norm=sum`, `nevt_job=2500`, and three cores. Showers and decays were
not run. Seeds were 19730 and 19731.

The fresh launch executes survey stage 1 and generation stage 2; there is no
stage 0. Every one of the eight channels completes four folded survey
iterations, with 15,360 evaluated points and 8,192 points retained in the
survey estimate. Their absolute-rate relative errors range from 0.5604%
to 0.9824%, below the requested 3% channel limit.

| Measurement | Fresh run | Generation-only restart |
| --- | ---: | ---: |
| Final events | 2,000 | 2,000 |
| Generation trials, including zeros | 53,084 | 39,871 |
| Production iterations | 12 | 10 |
| Production grid updates | 4 | 2 |
| Generation worker CPU, s | 128.292 | 93.540 |
| Negative events | 51 | 58 |
| Signed rate, pb | 2094.5488 ± 4.0376 | 2090.2353 ± 4.3432 |
| Absolute rate, pb | 2210.4699 | 2205.4399 |
| Largest native worker tail bound | 0% | 0.3182% |
| Largest final channel tail bound | 0% | 0.3174% |
| Collected full-weight overweight fraction | 0% | 0% |

Both runs completed collection in one round, with the initial 10% reserves
covering the updated channel allocations. Neither required additional workers
or collection threshold increases. All production masks are `[1,1,1,1,0,0,0]`:
the four unfolded coordinates can adapt, while the three folded coordinates
remain fixed. A nonzero reserve tail need not appear in the randomly trimmed
final sample, as illustrated by the restart.

The initial production envelope was independently reconstructed from the two
saved survey stream maxima and the fixed virtual-stream selection probability.
Every worker's first epoch cutoff matches that envelope, and its nonzero-point
target matches `max(1024, ceil(generated_target * max(1, envelope / survey_ABS)))`.

A watcher recorded each survey checkpoint when it first appeared. All eight
`ampli_grids` and eight `grid.MC_integer` files stayed byte-identical through
fresh production and the generation-only restart. The restart also preserved
all survey logs and rate files. Exported source hashes match the archived
source snapshots before and after both runs.

Run the archived verification from the repository root:

```sh
python validation/ampli_two_stage_20261007/dy/verify.py
```

The verifier checks the stage sequence, survey accuracy and iteration counts,
initial envelope and point forecast, folding masks, source and checkpoint
checksums, evolving rates, seeded initial/final allocation, deterministic pool
selection, final LHE magnitudes and rate normalization, and all 1% tail limits.
Results are in `validation.json`; each run also has a detailed `metrics.json`.
The exact pool reader/collector is archived alongside the Fortran and Python
sources; the rate-combination replay is independent of production code.

The compact archive includes cards, launch commands/logs, channel results,
POOL4 metadata, survey checkpoints, source snapshots and final compressed LHE
files. Worker checkpoint links point to the archived parent channel. It omits
build products and raw candidate LHE spools. The original export is
`/tmp/mg5-ampli-two-stage-dy-ovcj8baq/drell_yan`.

These runs validate the two-stage implementation and restart behavior. They
are not a performance comparison or a high-statistics physics validation.
