---
title: "Profiling and Benchmarking"
nav_order: 5
has_children: true
---
<!-- SPDX-License-Identifier: 0BSD -->

# Profiling and Benchmarking

`pyamplicol profile` measures one generated artifact through the same
optimized total path as `Runtime.evaluate()`.

## First profile

From the copied example workspace, generate the primary artifact and profile
one public process ordering:

```console
pyamplicol generate_pp_zjj_from_ufo_sm.toml

pyamplicol profile artifacts/pp_zjj \
  --process 'd d~ > g z g' \
  --momenta data/pp_zjj_momenta.json \
  --target-runtime 1.0 \
  --batch-size 128 \
  --color-flow 1 \
  --precision 16
```

The shipped shorthand card runs the same one-second workload:

```console
pyamplicol benchmark.toml
```

`benchmark` is a compatibility alias for `profile`; both direct commands
accept the same options. TOML retains `action = "benchmark"`—there is no
separate `action = "profile"` card value.

## What is timed

At precision 16, the headline wall timer starts inside Rusticol after Python
has converted the input to the native momentum buffer. It covers the ordinary
optimized core evaluation, including the selected engine's scheduling and
reduction. This is the runtime shared by Python, C11, C++17, Fortran 2008, and
Rust 2021 callers; it deliberately excludes Python/NumPy conversion overhead.

The profiler separately runs a paired native-attribution pass over the same
batch and repetition count. That pass explains where time was spent but never
replaces the ordinary headline wall sample.

The main rows mean:

| Metric | Interpretation |
| --- | --- |
| `wall time` | Repeated ordinary native runtime calls, normalized per point |
| `evaluator total` | Separately measured, minimally instrumented complete evaluator call |
| `Profile wall (paired profiled pass)` | Wall time of the attribution-enabled paired pass |
| `Native input pack` | Native-side input preparation after caller conversion |
| engine-specific runtime rows | Exclusive or explicitly labeled lane attribution |
| `Other Rusticol core` | Residual work closing the exclusive component sum against profile wall |

For compiled Direct-Arena, the row otherwise represented by an orchestration
counter is labeled **Direct-Arena runtime envelope**. It includes arena setup,
generated kernel execution, reduction, normalization, and orchestration. It is
not a kernel-only time. The current uninstrumented Direct-Arena hot path cannot
derive a pure kernel duration from that bucket.

Rows whose measured mean is exactly zero are omitted from the human timing
breakdown, which keeps the table focused on work actually observed for the
selected engine.

OTF has one additional, separately reported preparation observation. Before
configured warmups, the profiler snapshots the compact native runtime census,
times exactly one requested-selector evaluation over the full benchmark batch,
and authenticates that the family is retained. It labels whether the starting
handle was cold or already retained. That observation is not a headline wall
sample and is ineligible for ratios or acceptance. It is also distinct from an
explicit public `warm_up(...)`, which always takes exactly one binary64 point,
never a profiling batch.

## Sampling and uncertainty

The profiler performs configured warmups, estimates one evaluation cost, and
calibrates:

- the number of independent timed blocks;
- repetitions within each block;
- the batch of phase-space points used by every call.

Fast evaluators receive more repetitions per block. Slow evaluators use fewer
blocks while never dropping below `--minimum-samples`. A suspiciously slow
first one-repetition calibration observation is confirmed once at the same
repetition count so a transient busy-host outlier does not force an unusable
sampling plan.

The human report includes:

- mean time per point;
- sample standard deviation;
- standard error of the mean;
- relative standard error (`standard error / mean`);
- sample and repetition counts;
- target and measured cumulative timing;
- process, mode, precision, batch, and selector workload.

The uncertainty describes repeatability of the measured blocks on that host;
it is not a physics uncertainty.

Interrupting with `Ctrl-C` stops after the current complete block and still
prints the usual tables from completed samples. The result is explicitly
marked partial.

## Selectors change the workload

Profile selectors are runtime choices. They do not require regenerating an
artifact that retained complete coverage:

```console
pyamplicol profile artifacts/qq_z6g_recurrence_jit_o2 \
  --process 'u u~ > Z g g g g g g' \
  --color-flow 2 \
  --target-runtime 5
```

or select one helicity:

```console
pyamplicol profile artifacts/qq_z6g_recurrence_jit_o2 \
  --helicity 'h:-1,+1,-1,+1,+1,-1,+1,-1,+1'
```

For LC, match the flow layout to the workload:

| Workload | Preferred generated layout |
| --- | --- |
| One selected flow, all helicities | Default `topology-replay` |
| All flows, one selected helicity | `all-flow-union` |

When both profiling selector axes are omitted, pyAmpliCol chooses a
deterministic hot workload for the stored layout: one computed flow for
`topology-replay`, or one computed nonzero helicity for `all-flow-union`.
Explicit subsets and selected-axis lists are preserved; a complete summed-axis
list is normalized to equivalent omission. Any valid explicit shape outside
the stored layout's optimized workload emits at most one warning per loaded
process, before timing begins. This is a performance warning, not a correctness
failure; both layouts preserve complete physical flow and helicity coverage.
Ordinary unqualified evaluation still returns the complete matrix element.

There is no `all` selector shorthand. A complete list on the layout's summed
axis is equivalent to omission and still permits hot-selector inference; a
complete list on its selected axis explicitly requests the broader all-entry
workload. Repeat `--color-flow` or `--helicity` for the complete stable-ID list
shown by `inspect`; in Python, pass `runtime.physics.color_ids` or
`runtime.physics.helicity_ids`. See
[Runtime and Selectors](runtime-and-selectors.md#omitted-selectors-and-requesting-every-entry)
for complete examples.

NLC/full artifacts have contracted color and reject flow selectors.

## Native timing attribution

When the selected f64 runtime exposes native profiling, the report can include:

- input packing and crossing;
- state preparation and clearing;
- source and momentum setup;
- model-parameter setup;
- stage or amplitude input/evaluator/output work;
- reduction, total materialization, and final output copies;
- selector planning, gather, and scatter;
- recurrence source, contribution, finalization, closure, and replay work;
- OTF query-family recurrence-schedule work, separately labelled from both the
  complete evaluator envelope and recurrence mode's static schedule;
- eager initialization, gather, kernel calls, scatter, finalization, closure,
  and copy-out;
- per-stage timing where applicable.

Some internal rows are **attribution**, not additional top-level time. For
example, a full-stage evaluator envelope may already own its leaf-input gather.
The report labels these paths so they are not added twice.

`Other Rusticol core` is the residual between the paired profile wall and the
exclusive component sum. A small nonzero value is ordinary unsplit bookkeeping;
it should not be interpreted as hidden Python overhead because the timer is
already inside Rusticol.

## Native work counters

The optional counter table makes a profile easier to interpret by reporting
work volume as well as duration. Depending on the engine it can include:

- input bytes and native container counts;
- state, source, momentum, and model-parameter components;
- stage/amplitude movement and backend calls;
- reduction and materialized output values;
- selector gather/scatter volume;
- observed reusable-scratch capacity changes;
- explicit native output allocations;
- recurrence call and row counts;
- Direct-Arena engine/call and boundary-byte counts.

Movement and materialization are normalized per profiled point. Backend-call
and allocation activity are normalized per runtime call.

For the fused compiled Arena path, boundary input/current-output/amplitude-
output bytes must be zero and Arena calls must cover evaluator backend calls.
An invalid counter relationship is rejected rather than printed as a valid
optimized profile.

## Progress and machine-readable output

On a terminal, `--progress auto` selects a colored live progress display with
elapsed time, uncertainty, repetitions, and batch size. A redirected or
non-interactive run uses rate-limited log messages instead.

Choose explicitly when scripting:

```console
pyamplicol profile artifacts/pp_zjj --progress log
pyamplicol profile artifacts/pp_zjj --progress off --json > profile.json
```

JSON mode reserves stdout for the machine-readable result; diagnostics stay on
stderr.

## Python API

The typed API accepts either a loaded runtime or an artifact path:

```python
import json
from pathlib import Path

from pyamplicol import BenchmarkConfig, BenchmarkRunner, Runtime

momenta = json.loads(Path("data/pp_zjj_momenta.json").read_text())
runtime = Runtime.load("artifacts/pp_zjj", process="d d~ > g z g")
runner = BenchmarkRunner(
    BenchmarkConfig(
        target_runtime=1.0,
        batch_size=128,
        precision=16,
        warmup_runs=2,
        minimum_samples=5,
        color_flow_ids=("1",),
    )
)
result = runner.run(runtime, points=momenta)

print(result.wall_time_per_point)
print(result.uncertainty.standard_error)
print(result.repetitions_per_sample)
```

`BenchmarkResult.timing_breakdown` exposes typed component timings, stage
timings, and counter summaries when the backend supplies them.

Python may profile a higher-precision evaluator when the artifact retains one,
but native component attribution is f64-only. A non-f64 run therefore reports
wall timing rather than the Rusticol f64 breakdown. OTF rejects non-f64
precision altogether. See [Python API](python-api.md) and
[Symbolica and Licensing](symbolica-and-licensing.md).

## Published reports

The upstream repository publishes two rendered manual snapshots:

- [MacBook M3 report](https://github.com/MadGraphTeam/pyamplicol/blob/main/docs/performance_reports/macbook_M3_pyAmpliCol.pdf)
- [AMD EPYC report](https://github.com/MadGraphTeam/pyamplicol/blob/main/docs/performance_reports/EPYC_pyAmpliCol.pdf)

They are measurement-host results, not release-CI benchmarks.

## Comparing results responsibly

- Keep process, model, parameter card, momenta, color accuracy, flow layout,
  mode, backend, precision, selectors, and batch size fixed.
- Compare ordinary wall means with their standard errors; do not compare one
  engine's internal component against another engine's complete wall time.
- Treat very small relative standard errors on a busy host cautiously; repeat
  the full profile if system conditions materially changed.
- Use a fresh runtime after rebuilding an artifact.
- Record the exact artifact and package version, but do not infer physics
  independence from agreement among APIs that all call Rusticol.

## See also

- [Runtime and Selectors](runtime-and-selectors.md)
- [Generation Modes and Evaluators](generation-modes-and-evaluators.md)
- [LC workloads and execution modes](lc-workloads-and-execution-modes.md)
- [Examples Gallery](examples-gallery.md)

<!-- Page hierarchy for the MadGraph7 (Sphinx) documentation. -->
```{toctree}
:hidden:

troubleshooting
release-and-support
```
