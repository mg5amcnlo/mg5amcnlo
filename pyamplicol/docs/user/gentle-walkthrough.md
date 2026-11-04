---
title: "Get started: a gentle walkthrough"
nav_order: 2.5
has_children: true
---
<!-- SPDX-License-Identifier: 0BSD -->

# Get started: a gentle walkthrough

This page is for physicists who are comfortable writing a MadGraph process such
as `p p > Z j j`, but who may not want to learn software-packaging vocabulary
before obtaining a matrix element. It follows one example from the model and
process request to numerical evaluation, timing, Python, and the native
languages used in event generators.

pyAmpliCol generates and evaluates tree-level matrix elements. It does not
replace the PDF convolution, phase-space integration, event generation,
showering, or analysis parts of a Monte Carlo workflow. A useful first mental
picture is:

| In a MadGraph-style workflow | In pyAmpliCol |
| --- | --- |
| Import or choose a model | Select `built-in-sm`, a serialized JSON model, a trusted UFO directory, or a prepared model |
| Define `p`, `j`, and other multiparticles | Fill `[process.multiparticles]` or pass `--multiparticle` |
| `generate p p > z j j` | Put `p p > Z j j` in a run card or pass it to `pyamplicol generate` |
| `output` | Generate a self-contained **process output directory** |
| Choose one subprocess | Select a readable expression or its stable process ID |
| Evaluate or time the matrix element | Use `evaluate`, `profile`, the Python API, or a native API |

The process output directory is sometimes called an *artifact* in error
messages and in the more technical documentation. On this page, “process
output” means the same thing.

## A five-minute `p p > Z j j` tour

First activate the Python environment in which pyAmpliCol is installed. Copy
the packaged examples into a directory you can edit:

```console
pyamplicol examples copy ./pyamplicol-examples
cd pyamplicol-examples
```

The copy includes cards, momenta, Python examples, native examples, and the
packaged model files. It is independent of the package installation, so it is
safe to edit.

Generate a leading-colour recurrence output for a small two-flavour definition
of `p` and `j`:

```console
pyamplicol generate "p p > Z j j" artifacts/pp_zjj_recurrence \
  --model built-in-sm \
  --multiparticle 'p=d,d~,g' \
  --multiparticle 'j=d,d~,g' \
  --flavor-scheme 2 \
  --max-quark-lines 2 \
  --color-accuracy lc \
  --execution-mode recurrence
```

Generation uses the protective `error` policy by default: if that output
directory already exists, the command stops instead of overwriting it. Choose a
new output name when repeating the tour, or use `--force` only when replacing
the complete existing output is intentional.

The wheel contains prepared Standard Model building blocks for this recurrence
example. With these definitions, pyAmpliCol finds the ordered partonic channels,
identifies channels related by incoming or outgoing permutations, and stores
one reusable representative for each class. The loop-induced
`g g > Z g g` channel has no tree-level Standard Model amplitude and is
reported rather than generated.

Here `p` and `j` are simply names for the particle lists supplied on the
command line. They expand the inclusive request into separate partonic matrix
elements; they do not apply PDFs or sum over incoming parton luminosities.
`--color-accuracy lc` keeps the leading-colour result resolved into its physical
flows. With no runtime selector, evaluation sums all those flows and all
helicities.

Now ask what was produced:

```console
pyamplicol inspect artifacts/pp_zjj_recurrence
```

The coloured tables list the model, runtime, concrete representatives, aliases,
helicity and colour coverage, and the recurrence execution mode. Pick the
representative with stable ID `p_p_to_z_j_j_3`, or use its readable requested
ordering `d d~ > g z g`:

```console
pyamplicol inspect artifacts/pp_zjj_recurrence \
  --process 'd d~ > g z g'
```

The documentation uses *selector* for one of these explicit choices: a process
ID or expression, a helicity ID, or an LC flow ID. It is a choice, not a new
physics object.

Evaluate the supplied phase-space point. Its external legs are in exactly the
order written in `--process`, and every four-vector is `[E, px, py, pz]`:

```console
pyamplicol evaluate artifacts/pp_zjj_recurrence \
  --process 'd d~ > g z g' \
  --momenta data/pp_zjj_momenta.json \
  --precision 16
```

This returns the helicity- and colour-summed result. To see the components and
their explicit sum, add `--resolved`:

```console
pyamplicol evaluate artifacts/pp_zjj_recurrence \
  --process 'd d~ > g z g' \
  --momenta data/pp_zjj_momenta.json \
  --precision 16 \
  --resolved
```

Finally, time the optimized summed path:

```console
pyamplicol profile artifacts/pp_zjj_recurrence \
  --process 'd d~ > g z g' \
  --momenta data/pp_zjj_momenta.json \
  --batch-size 128 \
  --target-runtime 1.0 \
  --precision 16
```

The profile table records the selected process and selectors, batch size,
independent timing blocks, wall time per point, and the native phase breakdown
available for that execution mode. A 128-point profile batch is a throughput
measurement; it does not mean that an on-the-fly warm-up should use 128 points.

At this stage you have exercised the usual loop:

```text
choose model and process -> generate process output -> inspect -> evaluate -> profile
```

## The run card is a reusable set of choices

Long commands are useful while exploring. A TOML run card is better when the
same physics choices must be recorded and repeated. The recurrence command
above corresponds to the following card, which you could save as
`my_pp_zjj.toml`:

```toml
schema_version = 1
action = "generate"

[model]
source = "built-in-sm"

[process]
entries = [{ expression = "p p > Z j j" }]
flavor_scheme = 2
max_quark_lines = 2

[process.multiparticles]
p = ["d", "d~", "g"]
j = ["d", "d~", "g"]

[color]
accuracy = "lc"

[generation]
output = "artifacts/pp_zjj_recurrence_card"
emit_api_bundle = true

[evaluator]
execution_mode = "recurrence"
```

Run a card directly:

```console
pyamplicol my_pp_zjj.toml
```

or name the action explicitly:

```console
pyamplicol generate --card my_pp_zjj.toml
```

These spellings use the same configuration. The first reads `action` from the
card; the second says that this invocation is a generation action.

The packaged `generate_pp_zjj_from_ufo_sm.toml` card demonstrates a related
choice: it reads the serialized form of the UFO Standard Model and selects
`compiled` execution. Raw JSON or UFO model data contains the physics rules but
not the prepared recurrence building blocks, so that card deliberately builds
process-wide compiled evaluators. See [Models and Processes](models-and-processes.md) when
preparing recurrence, eager, or on-the-fly building blocks for another model.

### Cards, named options, and `--set`

Cards and command-line options use the same list of named settings (called the
configuration schema). Values are applied in this order:

1. pyAmpliCol defaults;
2. values in the card;
3. named command-line options such as `--workers 4`;
4. repeated `--set dotted.path=value` options, from left to right;
5. any unavoidable license or resource adjustment.

Later entries win. For example, this uses two workers and writes to a fresh
output without editing the packaged card:

```console
pyamplicol generate --card generate_pp_zjj_from_ufo_sm.toml \
  --workers 4 \
  --set generation.workers=2 \
  --set generation.output=artifacts/pp_zjj_trial
```

The named option first requests four workers; the later `--set` changes that
request to two. `--set` values follow TOML syntax. Numbers and booleans are
written naturally; quote a string when it contains spaces.

To see the complete interpretation without generating a process output, use:

```console
pyamplicol config resolve generate_pp_zjj_from_ufo_sm.toml \
  --set generation.workers=2
```

The `requested` section shows the card after defaults and overrides. The
`effective` section is the schema-valid configuration at this resolution step,
and `adjustments` records any clamps supplied to that resolver. Generation may
later apply constraints from a prepared model or the active Symbolica license;
those requested/effective differences and their reasons are retained in the
generated output and reported during the command. Use `config resolve` first
when a card or override does not appear to select the value you expected, then
read any adjustment reported by generation itself.

Paths written in a card are interpreted relative to that card, not relative to
the shell from which it is run. That is why the packaged cards continue to find
their `models/`, `data/`, and `artifacts/` paths after the example directory is
moved.

The exhaustive field reference is `all_options.toml` in the copied workspace.
The shorter explanation of colour, execution modes, and selectors is in
[Configuration](configuration.md).

## What is inside a process output?

Think of `artifacts/pp_zjj_recurrence/` as a generated, self-contained matrix-
element library for one model and a set of partonic processes. It contains:

- the model identity, restriction, parameter catalogue, and configuration used
  for generation;
- a stable list of concrete process representatives and their aliases;
- the helicity and colour information needed to interpret results;
- executable evaluator data for the selected execution mode;
- deterministic validation momenta; and
- when `emit_api_bundle = true`, small Python, C, C++, Fortran, and Rust
  standalone drivers under `API/`.

It is a directory rather than one shared library because the metadata and
physics axes are as important as the numerical code. You can move the whole
directory and load it from another working directory. Do not move individual
files out of it or edit them by hand.

Process outputs are executable inputs, much like a compiled library. Only load
one that you generated yourself or obtained from a source you trust. Normal
loading validates its declared structure and compatibility. It is not a proof
of who published it; the trust rules and optional all-file hash check are
described in [Artifacts and Portability](artifacts-and-portability.md).

### Representatives and aliases avoid duplicate work

For the request `p p > Z j j`, several ordered channels differ only by a
permutation of incoming legs, outgoing legs, or both. pyAmpliCol stores one
representative evaluator where that equivalence is established. The other
orders appear as aliases.

You may therefore select either a stable ID:

```console
pyamplicol inspect artifacts/pp_zjj_recurrence \
  --process p_p_to_z_j_j_3
```

or a readable ordering:

```console
pyamplicol inspect artifacts/pp_zjj_recurrence \
  --process 'd d~ > g z g'
```

The runtime remaps momenta and result labels to the order you requested. An
incoming particle never crosses the `>` sign to become outgoing, or vice versa.
If a readable expression could refer to more than one representative,
pyAmpliCol stops and prints the possible stable IDs instead of guessing.

Stable IDs are convenient in production scripts because they do not depend on
case or whitespace. Readable expressions are often clearer in a notebook.

### What is shared between processes?

A *current* here is an intermediate off-shell object built from a subset of the
external legs—the same physical idea that appears in Berends–Giele-style
recursion. Sharing is conservative:

- Multiparticle aliases and side-preserving permutations use the same concrete
  representative rather than duplicating its calculation.
- In eager, recurrence, and on-the-fly outputs, the model's prepared numerical
  building blocks are stored once and referenced by the processes that need
  them. These are the reusable rules for constructing currents from vertices.
- In recurrence mode, two process schedules are stored once only when an exact
  process-to-process mapping proves that the complete current construction is
  the same. A small per-process binding restores the external order, helicities,
  and colour labels.
- The model provenance, common configuration, dependencies, and parameter
  definitions belong to the output as a whole.

What is not shared by assumption is just as important. Unrelated subprocesses
do not have their currents merged merely because some particle names coincide.
Each genuine process retains its own external-state metadata, selector
coverage, validation point, and process binding. Compiled mode has
process-specific stage evaluators; eager mode has process-specific invocation
tables; recurrence has a process binding to its recorded and checked schedule;
and on-the-fly mode has a compact seed for each process.

This is why adding several processes to one output can save model and schedule
storage without changing the meaning of any individual matrix element.

### Adding another process later

The safest starting point is to put the complete process set in one
`process.entries` list and generate it together. pyAmpliCol can also append a
new process without exposing a half-written intermediate output when the
existing mode permits it and the model, target, pyAmpliCol build, and generation
settings are identical.

The typed Python API makes the unchanging configuration explicit. This compact
example creates a compiled output, then appends a second process:

```python
from pyamplicol import Generator
from pyamplicol.config import (
    Action,
    EvaluatorConfig,
    GenerationConfig,
    RunConfig,
)

config = RunConfig(
    action=Action.GENERATE,
    generation=GenerationConfig(emit_api_bundle=True),
    evaluator=EvaluatorConfig(execution_mode="compiled"),
)
generator = Generator(config)
output = "artifacts/z_plus_jet"

generator.generate("u u~ > Z g", output)
generator.generate("d d~ > Z g", output, mode="append")
```

Append rejects a duplicate stable ID or a change of model, target, prepared
model building blocks (called a kernel pack in technical messages),
configuration, or producer identity before modifying the existing output.
Current recurrence root schedules are immutable, so generate the full
recurrence process set together instead of appending to it. If settings must
change, write a new output or regenerate the complete set with explicit
`replace` mode. The default `error` mode protects an existing directory from
accidental replacement.

Inspect the output after an append and keep the output directory together. The
root `API/` bundle is regenerated to include validation points for the full
process set.

## Reading `inspect` as a physicist

Start with the compact view:

```console
pyamplicol inspect artifacts/pp_zjj_recurrence
```

It reads metadata and does not instantiate every helicity/colour evaluator.
The tables are intentionally more detailed than a MadGraph process list, but a
first-time user only needs a few fields.

### The `Artifact` table

| Field | What to check |
| --- | --- |
| `path` | The process output you meant to open. |
| `artifact type` | Usually a generated process set. “Artifact” means process output here. |
| `artifact ID` | A content label for the files that affect evaluation; useful when recording exactly what was benchmarked. |
| `producer` | The pyAmpliCol version that generated the output. |
| `target` | `portable-64le` can be loaded on supported 64-bit little-endian hosts; an explicit machine triple or CPU feature list indicates a target-specific output. |
| `model` | Model source and restriction. Check this before comparing physics results. |
| `default process` | The process selected when a multiprocess output is loaded without an explicit selector. Production code should normally select one explicitly. |
| `contents` | Number of stored representatives and aliases. |
| `runtime` / `capabilities` | The native engine and features required from the installed runtime. |
| `payloads` | Physical size and, for packed JIT data, the number and unpacked size of logical evaluators. |
| `integrity` | The top-level file inventory (the manifest) and declared structure were accepted. This is not publisher authentication. |

### The `Processes` and `Aliases` tables

| Field | Physical meaning |
| --- | --- |
| `stable ID` | Durable selector for CLI, Python, and native calls. |
| `concrete process` | External particles in the representative ordering. |
| `color` | `lc`, contracted `nlc`, or contracted `full`. |
| `helicities` | Physical configurations, with a smaller evaluated count in parentheses when certified reuse is active. |
| `color outputs` | Physical LC flows; NLC/full expose one contracted colour result per helicity. |
| `coverage (hel./color)` | `complete / complete` means selectors were not fixed during generation. A specialized output says which axis was restricted. |
| `aliases` | Number of requested orderings mapped to this representative. |

The separate `Aliases` table names each mapped ordering and the representative
that evaluates it. This is the closest analogue to seeing the expanded
subprocess list while also learning which entries reuse one evaluator.

### The `Execution` table

The most useful lines are:

- `mode / backend`: recurrence, compiled, eager, or on-the-fly and the prepared
  numerical backend, where applicable;
- `LC flow layout`: whether a materialized LC output is arranged for one-flow
  helicity sums (`topology-replay`) or all-flow single-helicity work
  (`all-flow-union`);
- `runtime selectors`: whether helicity and colour flow remain selectable;
- `generation specialization` and `generation selection`: whether generation
  fixed a particular flow, helicity assignment, or both;
- `native profile phases`: which timing pieces can appear in `profile`; and
- the requested/effective point tile and workspace, which matter for memory at
  large batches.

Recurrence rows and eager invocation counts are engineering descriptions of
the same current construction; they are mainly useful when comparing output
size or diagnosing performance. For an on-the-fly output, look instead for the
physical helicity/flow census, query-construction threads, and `f64 only`
(standard double precision). These describe what can be selected without
forcing the complete high-multiplicity family to be built during inspection.

For a physicist-oriented comparison of both LC layouts, all four execution
modes, and the OTF warm-state lifecycle, continue with [LC workloads and
execution modes](lc-workloads-and-execution-modes.md).

Use a process selector to shorten the table:

```console
pyamplicol inspect artifacts/pp_zjj_recurrence \
  --process p_p_to_z_j_j_3
```

The explicit full-physics view materializes complete helicity and colour
metadata and may be very large:

```console
pyamplicol inspect artifacts/pp_zjj_recurrence \
  --process p_p_to_z_j_j_3 \
  --full-physics
```

Reserve it for a selected low-multiplicity process when the compact table is
not enough.

## Choose the interface that fits your calculation

All interfaces load the same process output and use the same stable process and
selector IDs. You do not generate a separate “Python version” and “Fortran
version.”

| Interface | A natural use |
| --- | --- |
| CLI | Generate, inspect, make one-off evaluations, compare settings, and profile. |
| Typed Python API | Notebooks, scans, tests, parameter updates, and integration with Python phase-space code. |
| C11 API | A small set of stable C functions for an existing C or mixed-language program. |
| C++17 wrapper | Automatic lifetime management and `std::vector`-based evaluation. |
| Fortran 2008 module | Monte Carlo and phenomenology codes written in Fortran. |
| Rust 2021 wrapper | Ownership-checked access without an external Rust crate dependency. |

### Typed Python

The numerical part of the five-minute tour is only a few lines:

```python
import json
from pathlib import Path

from pyamplicol import Runtime

points = json.loads(Path("data/pp_zjj_momenta.json").read_text())
runtime = Runtime.load(
    "artifacts/pp_zjj_recurrence",
    process="d d~ > g z g",
)

print(runtime.physics.process)
print(runtime.physics.helicity_ids[:2])
print(runtime.physics.color_flow_ids[:2])
print(runtime.evaluate(points, precision=16))
```

`runtime.physics` describes the selected process. Pass IDs from that object to
`evaluate(..., helicities=..., color_flows=...)` to select components. LC
accepts both kinds of selector; NLC and full colour are already contracted and
therefore accept helicity selectors but not an LC flow selector.

The packaged scripts show complete typed generation, parameter updates,
resolved evaluation, and benchmarking:

```console
python python/typed_generation.py artifacts/pp_zjj_typed --plan-only
python python/runtime_evaluation.py \
  artifacts/pp_zjj_recurrence data/pp_zjj_momenta.json \
  --process 'd d~ > g z g'
```

See [Runtime and Selectors](runtime-and-selectors.md) for point-wise selectors, exact precision, model
parameters, resolved output shapes, and explicit on-the-fly `warm_up(...)`.

### Generated Python, C, C++, Fortran, and Rust drivers

Because the tour kept the default `emit_api_bundle = true`, the process output
contains standalone examples for every supported language. Run them from the
copied example workspace:

```console
python artifacts/pp_zjj_recurrence/API/python/check_standalone.py \
  --process 'd d~ > g z g' --json

make -C artifacts/pp_zjj_recurrence/API/c run \
  ARGS='--process "d d~ > g z g" --json'

make -C artifacts/pp_zjj_recurrence/API/cpp run \
  ARGS='--process "d d~ > g z g" --json'

make -C artifacts/pp_zjj_recurrence/API/fortran run \
  ARGS='--process "d d~ > g z g" --json'

make -C artifacts/pp_zjj_recurrence/API/rust run \
  ARGS='--process "d d~ > g z g" --precision 16 --json'
```

With no `--kinematics` option, each driver uses the validation point stored in
the output and remaps it to the requested leg order. The drivers compare the
optimized total with the explicit sum of resolved components, so they are also
a useful first check of a compiler and linker setup.

The installed `rusticol-config` command supplies the include paths, static
library, Fortran module source, and linker flags. Native calls use standard
double precision, also called binary64 (`precision = 16`). Python can also use
a retained higher-precision evaluator when that execution mode provides one;
on-the-fly execution is double precision in all languages.

For embedding rather than running the example drivers, start with
[Native APIs](native-apis.md). It contains complete loading/evaluation examples,
memory layout, compiler commands, model-parameter updates, selector arrays, and
the cross-language on-the-fly warm-up callback.

## Where on-the-fly mode fits

Recurrence generated all reusable schedules before the first load. On-the-fly
(OTF) instead stores a compact process description and constructs a selected
family when it is first requested. Its practical focus is a leading-colour,
single-flow calculation summed over helicities, especially as multiplicity
grows.

The packaged OTF walkthrough uses the same `p p > Z j j` process:

```console
pyamplicol generate --card otf_pp_zjj.toml
python python/otf_pp_zjj_warm_up.py
pyamplicol profile --card otf_pp_zjj.toml
```

The Python step calls `warm_up(...)` with exactly one double-precision
phase-space point and one LC flow. Omitting the helicity selector requests the
helicity sum. Its progress display reports query construction and live memory;
the following table reports the selected flow, warm-up time, query counts,
memory, and matrix element. The independent `profile` command then measures
128-point throughput.

You can keep completed preparation between programs: save it with
`runtime.save("process.otf-cache")`, then load the original process output in
the next program and call `runtime.load_cache("process.otf-cache")`. Continue
with the same helicity/flow selectors to reuse the saved work. This is explicit
API functionality, not an automatic cache shared by separate CLI commands.
See the [small Python example](runtime-and-selectors.md#saving-an-otf-warm-cache)
and [native SDK examples](native-apis.md#saving-and-restoring-an-otf-cache).

OTF NLC and full-colour calculation are available for low multiplicity, but
their contracted family grows rapidly and is not intended here as a
high-multiplicity performance route. The detailed differences among recurrence,
compiled, eager, and OTF are summarized in
[Generation Modes and Evaluators](generation-modes-and-evaluators.md).

## A practical checklist for a new calculation

1. Copy the examples and make sure the five-minute tour works.
2. Choose a trusted model. Prefer serialized JSON for reproducibility; treat a
   UFO directory as executable Python input.
3. Write the process and multiparticle definitions in a card. Start with a
   narrow flavour scheme while testing.
4. Choose colour accuracy and an execution mode. Recurrence is the general
   prepared-model default; compiled accepts a model that has not had its local
   numerical building blocks prepared; eager stores compact invocation tables
   around those prepared building blocks; OTF is aimed at selected
   double-precision workloads.
5. Generate into a new output directory. Keep the default `error` policy until
   replacement is intentional.
6. Run compact `inspect`. Check the model, process, colour accuracy, coverage,
   execution mode, and target before evaluating.
7. Evaluate a known point and compare total with resolved output.
8. Profile the exact helicity/flow workload and batch size your application
   will use.
9. Integrate the same output through Python or one native API. Select a process
   explicitly in production code.
10. Record the card, process selector, model parameters, and process-output ID
    with physics results.

From here, continue with [Configuration](configuration.md) for all run-card
choices, [Models and Processes](models-and-processes.md) for UFO/JSON and multiprocess details,
[Runtime and Selectors](runtime-and-selectors.md) for evaluation, and [Native APIs](native-apis.md) for
language integration. Every
maintained packaged card and script is indexed in the
[examples guide](https://github.com/MadGraphTeam/MadGraph7/blob/main/pyamplicol/examples/README.md).

<!-- Page hierarchy for the MadGraph7 (Sphinx) documentation. -->
```{toctree}
:hidden:

lc-workloads-and-execution-modes
```
