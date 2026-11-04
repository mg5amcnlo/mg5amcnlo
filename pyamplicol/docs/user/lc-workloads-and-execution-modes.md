---
title: "LC workloads and execution modes"
nav_order: 1
parent: "Get started: a gentle walkthrough"
---
<!-- SPDX-License-Identifier: 0BSD -->

# LC workloads and execution modes

This page helps with two choices that are easy to mix up:

1. **Which part of the leading-colour result will be evaluated most often?**
   This determines the LC flow layout for a materialized process output.
2. **When should pyAmpliCol construct the numerical representation?** This
   determines the execution mode.

Neither choice changes the requested physics. The LC flow layout reorganizes
the same leading-colour components, while the execution mode reorganizes when
the same current construction is prepared. Use [the gentle
walkthrough](gentle-walkthrough.md) first if “process output,” “flow,” or
“selector” is unfamiliar.

Here are the few computing terms needed on this page:

- A **process output** is the directory produced by `pyamplicol generate` for
  one or more requested reactions. It plays roughly the role of a generated
  standalone process directory in MadGraph: a later program loads it and
  supplies phase-space points.
- A **current** is an intermediate off-shell object in the recursive amplitude
  construction. Reusing a current avoids recomputing the same substructure in
  many diagrams.
- A **materialized** output has already written the reusable current-building
  plan to disk during generation. Recurrence, eager, and compiled outputs are
  materialized in different forms. OTF instead writes a compact recipe and
  constructs the requested plan after the output is loaded.
- The **runtime** is the loaded numerical calculator, and a **selector** is a
  request for particular helicity or colour-flow components. “Cache” below
  simply means state retained in the runtime's memory for reuse. An OTF cache
  can also be saved to a file, but that file supplements the original process
  output rather than replacing it.

## Two useful slices through an LC result

At one phase-space point, imagine the leading-colour result as a rectangular
table. Its rows are the physical helicity configurations \(h\), and its columns
are the physical LC flows \(f\). A runtime request may sum either axis:

| Workload | Quantity requested repeatedly | Typical physics use |
| --- | --- | --- |
| One flow, helicity sum | Fix one \(f\), sum every retained \(h\) | A flow-resolved calculation or a colour-flow sampling/integration step |
| All flows, one helicity | Fix one \(h\), sum every retained \(f\) | A helicity-resolved calculation or repeated evaluations for sampled helicities |

Asking for neither selector sums both axes. Asking for a set of IDs sums the
selected subset. The two layouts below are performance arrangements for these
requests; they do not add or remove LC terms when their coverage is complete.
They optimize different physical outer loops: colour-flow sampling repeatedly
needs a vertical slice of the table, whereas helicity sampling repeatedly needs
a horizontal slice. Neither layout is universally faster.

### `topology-replay`: one flow, summed helicities

`topology-replay` is the default. It retains complete physical flow and
helicity coverage, but arranges the generated construction so that the compact
topology can be replayed for the requested flow. It is therefore the natural
layout when each evaluation normally selects one flow and omits the helicity
selector, meaning “sum all helicities.”

The profiling campaign calls this same workload `non-union-flow`: currents for
different physical flows are not first merged into one cross-flow plan. The
public process-output setting remains `topology-replay`.

Use it when:

- a downstream colour sampler asks for one LC flow at a time;
- the matrix element should remain reusable for another flow later; or
- the main observable is a flow-resolved contribution summed over spin.

The explicit card setting is:

```toml
[color]
accuracy = "lc"
lc_flow_layout = "topology-replay"
```

Because it is the default, omitting `lc_flow_layout` has the same effect. A
direct generation override is:

```console
pyamplicol generate --card run.toml \
  --color-accuracy lc \
  --lc-flow-layout topology-replay
```

Once generated, select the first physical flow shown by `inspect` and sum all
helicities by omitting `--helicity`:

```console
pyamplicol evaluate --card run.toml --color-flow 1
pyamplicol profile --card run.toml --color-flow 1
```

CLI flow numbers are **one-based**. A stable flow ID printed by `inspect`, such
as `flow:...`, may be used in place of `1`.

### `all-flow-union`: all flows, one helicity

`all-flow-union` retains the same complete axes, but builds one shared
cross-flow recurrence. It is the complementary layout for repeatedly selecting
one helicity and omitting the flow selector, meaning “sum all physical LC
flows.”

The profiling campaign calls this workload `union-flow`: it measures the
cross-flow plan for one selected helicity. The public process-output setting is
`all-flow-union`.

Use it when:

- a helicity sampler asks for one physical helicity at a time;
- every LC flow is needed for each selected helicity; or
- the main observable is a helicity-resolved LC result with colour summed.

Set it in the card:

```toml
[color]
accuracy = "lc"
lc_flow_layout = "all-flow-union"
```

or override the card during generation:

```console
pyamplicol generate --card run.toml \
  --color-accuracy lc \
  --lc-flow-layout all-flow-union
```

Then select one stable helicity ID from `inspect` and omit `--color-flow`:

```console
pyamplicol evaluate --card run.toml \
  --helicity 'h:-1,+1,-1,+1,-1'
pyamplicol profile --card run.toml \
  --helicity 'h:-1,+1,-1,+1,-1'
```

The displayed helicity length depends on the number of external particles, so
copy an ID from the process output rather than copying the illustrative ID
above into an unrelated process.

`all-flow-union` is LC-only. It is available for recurrence, eager, and
compiled outputs. OTF does not materialize either LC layout: the helicity and
flow arguments in the first runtime request determine which family of currents
is constructed in memory.

### Pick the layout from the repeated workload

For a complete materialized output, either layout can still evaluate all
flows, all helicities, or a supported selected subset. “Optimized for” does not
mean “only capable of.” It means that the current-building plan written during
generation is organized around one common loop direction.

| If most calls request... | Start with... |
| --- | --- |
| One flow and every helicity | `topology-replay` |
| Every flow and one helicity | `all-flow-union` |
| The completely summed LC result | Benchmark both layouts for the real process and batch size |
| A mixture that changes during a run | Keep complete coverage, then profile the dominant request |

The packaged cards provide a larger-multiplicity comparison without hiding
the settings:

```console
pyamplicol examples copy ./pyamplicol-examples
cd pyamplicol-examples
pyamplicol generate --card benchmark_z6g_single_flow_helicity_sum.toml
pyamplicol profile --card benchmark_z6g_single_flow_helicity_sum.toml
pyamplicol generate --card benchmark_z6g_all_flows_single_helicity.toml
pyamplicol profile --card benchmark_z6g_all_flows_single_helicity.toml
```

These are performance examples, so generation can take appreciably longer
than the small `p p > Z j j` tour.

## Runtime selection in Python and native programs

For recurrence, eager, and compiled modes, the layout is fixed when the process
output is generated. Python and native programs load that output and select
physical IDs while evaluating points; they cannot rearrange its stored plan.
OTF is the exception: its first selected request constructs the corresponding
family after loading.

### Typed Python

The typed objects in `runtime.physics` avoid copying IDs by hand:

```python
from pyamplicol import Runtime

runtime = Runtime.load("artifacts/pp_zjj", process="d d~ > g z g")
flow = runtime.physics.color_flows[0]
helicity = runtime.physics.helicities[0]

# One physical flow, summed over every helicity.
flow_values = runtime.evaluate(points, color_flows=(flow,), precision=16)

# One physical helicity, summed over every LC flow.
helicity_values = runtime.evaluate(points, helicities=(helicity,), precision=16)
```

Here `points` is a batch with shape
`[point][external particle][E, px, py, pz]`. The same calls accept stable string
IDs instead of the typed objects. The global selector arguments are sequences:
passing one entry selects one component, passing several selects and sums that
subset, and passing `None` sums the complete retained axis.

Typed generation uses the same schema as a TOML card. For example, this is the
complete choice of LC layout and execution mode in Python:

```python
from pyamplicol import Generator, ModelSource
from pyamplicol.config import resolve_config

settings = resolve_config(
    {
        "schema_version": 1,
        "action": "generate",
        "color": {
            "accuracy": "lc",
            "lc_flow_layout": "all-flow-union",
        },
        "evaluator": {"execution_mode": "recurrence"},
    }
)
Generator(settings).generate(
    "d d~ > g z g",
    "artifacts/ddbar_to_zgg_union",
    model=ModelSource.built_in_sm(),
)
```

Replace `all-flow-union` by `topology-replay`, or `recurrence` by `compiled`,
`eager`, or `on-the-fly` as described below. For OTF, omit
`lc_flow_layout`: no materialized flow layout is used.

### C11

The C ABI uses a null pointer with a zero count for an unselected axis. Given
stable `flow_id` and `helicity_id` strings, the two LC calls are:

```c
const char *one_flow[] = {flow_id};
int status = rusticol_runtime_evaluate_selected_f64(
    handle, momenta, momentum_count, point_count,
    NULL, 0, one_flow, 1,
    NULL, 0, NULL, 0,
    values, point_count);

const char *one_helicity[] = {helicity_id};
status = rusticol_runtime_evaluate_selected_f64(
    handle, momenta, momentum_count, point_count,
    one_helicity, 1, NULL, 0,
    NULL, 0, NULL, 0,
    values, point_count);
```

The arguments after the global string IDs are optional per-point **zero-based**
selector-index arrays. Do not supply both the global and per-point form for the
same axis. End the handle lifetime with `rusticol_runtime_free(handle)`.

### C++17

The C++ wrapper uses empty vectors for an unselected axis:

```cpp
auto flow_values = runtime.evaluate_selected(
    momenta, point_count, {}, {flow_id});
auto helicity_values = runtime.evaluate_selected(
    momenta, point_count, {helicity_id}, {});
```

The final two optional arguments, omitted here, are the per-point helicity and
flow index vectors. The `rusticol::Runtime` destructor releases the handle.

### Fortran 2008

In the Fortran module, omit the optional axis that should be summed:

```fortran
character(len=128) :: flow_ids(1), helicity_ids(1)
real(c_double), allocatable :: values(:)

flow_ids(1) = flow_id
call runtime%evaluate_selected(momenta, point_count, values, &
    color_ids=flow_ids, ierr=ierr)

helicity_ids(1) = helicity_id
call runtime%evaluate_selected(momenta, point_count, values, &
    helicity_ids=helicity_ids, ierr=ierr)
```

Use `call runtime%close()` when the loaded process is no longer needed.

### Rust 2021

The installed safe wrapper represents omitted axes with `Selectors::all()`:

```rust
let one_flow = Selectors::all().with_colors([flow_id]);
let flow_values = runtime.evaluate_selected_f64(
    &momenta, point_count, &one_flow, None, None,
)?;

let one_helicity = Selectors::all().with_helicities([helicity_id]);
let helicity_values = runtime.evaluate_selected_f64(
    &momenta, point_count, &one_helicity, None, None,
)?;
```

The two `None` arguments are the optional per-point zero-based helicity and
flow index slices. Dropping `runtime` releases the native handle.

Complete load, error-handling, metadata, memory-layout, and compiler examples
are in the [Native APIs guide](native-apis.md). Stable IDs can be read from that
metadata or copied from `pyamplicol inspect`; do not infer IDs from their
position except where the CLI explicitly accepts a one-based ordinal.

## Generation specialization is not runtime selection

Complete coverage gives one process output with reusable runtime selectors. A
runtime selection chooses among components that are already represented by
that output and can change from call to call. This is analogous to generating a
general standalone process once and choosing the desired component during the
event loop.

Generation specialization instead permanently omits unneeded work while
constructing a recurrence, eager, or compiled output. It can be valuable when
a production run will use exactly one known flow or helicity, but that smaller
output cannot later answer a different selection. The relevant card fields are:

```toml
[process]
# Flow-specialized packaged example: u u~ > Z g g g g g g.
reference_color_order = [2, 4, 5, 6, 7, 8, 9, 1, 3]
selected_color_sector_ids = [0]
```

or, in the separate helicity-specialized card:

```toml
[process]
# Helicity-specialized packaged example: u u~ > Z g g g g g g.
selected_source_helicities = { "1" = -1, "2" = 1, "3" = -1, "4" = 1, "5" = -1, "6" = 1, "7" = -1, "8" = 1, "9" = -1 }
```

These are low-level generation inputs. `selected_color_sector_ids` are
zero-based internal construction-sector IDs, not the one-based physical flow
numbers accepted by `--color-flow`. A specialized-away flow or helicity cannot
be restored by a later runtime call. Use the maintained
`benchmark_z6g_generation_specialized_*.toml` cards as worked examples instead
of guessing sector IDs or a reference order.

For reusable compiled and eager outputs, do not combine generation
specialization with `all-flow-union`; generation rejects that combination.
Recurrence can encode a valid specialized union construction, but its excluded
selectors are still absent, so it should be treated as an advanced,
non-reusable output. `process.max_color_sectors` is incompatible with
`all-flow-union` in every materialized mode.

OTF takes the opposite approach: it rejects these generation-specialization
fields and keeps a compact list of all physical selectors. The first warm-up or
evaluation then constructs only the requested family in memory.

## Four ways to organize the same calculation

The execution mode is selected in `[evaluator]`:

```toml
[evaluator]
execution_mode = "recurrence" # or compiled, eager, on-the-fly
```

or on the generation command line:

```console
pyamplicol generate --card run.toml --execution-mode recurrence
```

In the descriptions below, a **kernel** is a small compiled numerical routine
for a model vertex or current operation. A **dependency graph** records which
currents must be computed before others; the common abbreviation DAG means a
directed acyclic graph. A **prepared model** is a reusable package of these
model-specific kernels, prepared once and shared by several process outputs.

| Mode | What generation stores | What happens after loading | Natural starting point |
| --- | --- | --- | --- |
| `recurrence` | An ordered recipe for building and combining currents | The loaded calculator follows that stored recipe for each batch | General-purpose default with a prepared model |
| `compiled` | Machine-code stages compiled for the complete process graph | Those process-specific stages evaluate each batch | More generation work in exchange for the broadest process-wide optimization |
| `eager` | A compact list of calls to prepared model kernels | The loaded calculator makes those calls directly | Usually the quickest materialized output to generate and load |
| `on-the-fly` (OTF) | A compact process recipe, not a ready-to-run selected family | The first selected request constructs one family and retains it in memory | Compact LC output when one selected family dominates many later calls |

The choice changes where work and storage occur, not the meaning of LC, NLC,
or full colour. Results for the same supported physics selection should agree
across modes within the stated numerical tolerance.

### Recurrence

Recurrence is the default and the closest stored representation to the
physical current recursion. Generation writes a compact, checked schedule of
which currents to build and in what order; numerical evaluation follows it
through the model's prepared kernels. It supports LC, NLC, and full colour,
reusable runtime selectors, and optional generation specialization. Large
point batches are divided into smaller tiles so temporary memory stays bounded.

Choose recurrence first when a prepared model is available and the process
output should remain broadly reusable.

### Compiled

Compiled mode turns the complete process dependency graph into process-wide
machine-code stages during generation. That costs more generation work and
lets the compiler optimize across the broadest view of one process. Unlike
recurrence, eager, and OTF, it can start directly from a machine-readable
JSON/UFO model description without a separately prepared kernel package. It
supports LC, NLC, full colour, reusable selectors when coverage is complete,
and generation-specialized outputs.

Choose it when process-local compilation is desirable, when comparing a
specialized baseline, or when no prepared model bundle exists.

### Eager

Eager mode stores compact tables saying which prepared model kernel to call,
with which inputs, and how to combine its result. It does less process-wide
compilation and is usually the quickest materialized mode to generate and
load. Like recurrence and compiled mode, it supports LC, NLC, full colour,
complete runtime selection, and optional generation specialization.

Choose it when fast materialized generation/load matters and a prepared model
is available.

### On-the-fly

OTF stores a compact process recipe rather than a ready-to-run schedule for
every selector. It still needs a prepared model. For LC it can express both
physical request families: one flow summed over helicities, or all flows summed
at one helicity. It does not use the `topology-replay`/`all-flow-union`
generation setting because the first runtime request itself defines the family
to build.

OTF is native `f64` only (`precision=16`), meaning ordinary IEEE double
precision. Its practical performance focus is LC with one selected flow and a
helicity sum. Contracted NLC and full-colour families are available for
low-multiplicity correctness work, but the initial construction grows rapidly
and is not intended as a practical high-multiplicity route.

The release acceptance suite exercises catalogued contracted NLC and
full-colour OTF processes through `n <= 4` against recurrence or MadGraph as
appropriate. That establishes the low-multiplicity correctness surface; it is
not a claim that their cold construction is affordable at larger `n`.

## The OTF warm-state lifecycle

OTF moves a potentially expensive operation from generation to the first use
of a selected family. Treat warm-up as part of job planning, not as a small
extra numerical sample.

A **family** here means one physical selection together with all currents
needed to evaluate it repeatedly. For example, “flow 1, summed over all
helicities” is one family, while “flow 2, summed over all helicities” is
another. A family is structural: it says what to calculate, not what numerical
value the matrix element takes at one phase-space point.

### Cold, warm, replaced, and cleared

One loaded OTF runtime follows this lifecycle:

| Action | State afterwards |
| --- | --- |
| Load an OTF process | Cold: compact seed loaded, no selected family retained |
| `warm_up` selection A | A is retained, and exactly one `f64` point has also been evaluated |
| Evaluate selection A again | Reuses A's structural family; ordinary numerical execution remains |
| Successfully warm or evaluate selection B | B replaces A; only the most recent selected family is retained |
| Change model parameters, then evaluate the retained selection | The same structure is retained, and its numerical workspace is refreshed with the latest parameter values before execution |
| `runtime.save(path)` | The completed retained cache is written to disk; the current handle is unchanged |
| Load a matching process, then `runtime.load_cache(path)` | The saved family is restored without repeating current construction; numeric workspaces are initialized for the receiving handle |
| Python `runtime.clear()` | Cold again while the process output and current model parameters remain loaded |
| Close/free/drop the native handle | In-memory state is released; a later load starts cold unless a saved cache is explicitly restored |

The in-memory family cache is **handle-local and last-family-only**. A handle
is simply one loaded `Runtime` object (or its C/Fortran/Rust/C++ equivalent).
A second `pyamplicol` CLI command starts a new process and loads a new handle,
so it does not inherit warm state from the first command. Keep one Python or
native runtime alive when same-selection reuse matters, or explicitly
[save and restore its cache](runtime-and-selectors.md#saving-an-otf-warm-cache)
between runs. A snapshot retains the current family, not previously evicted
selections, and does not checkpoint an unfinished cold construction.

For a small complete Python example, including generation and a check that no
queries are rebuilt, see
[Saving an OTF warm cache](runtime-and-selectors.md#saving-an-otf-warm-cache).
The [native examples](native-apis.md#saving-and-restoring-an-otf-cache) show
`save` and `load_cache` in C, C++, Fortran and Rust. Restore the cache before
the first evaluation and use the same selectors as when it was saved.

There is no public setting for retaining several OTF families: the most recent
successful family replaces the previous one. `evaluator.optimization.cores`
controls how many CPU workers may help with the initial construction; it does
not change this one-family retention rule or promise that later numerical
evaluation uses that many threads. `warm_up(..., n_cores=4)` overrides that
construction setting for one call, without discarding an already-warm family
or persisting the override in a saved cache. Query traces can be built in
parallel, with a bounded number of temporary traces; their shared-cache merge
and family finalization remain serial.

### Explicit warm-up means exactly one binary64 point

Python exposes the structural operation only on OTF runtimes:

```python
flow = runtime.physics.color_flows[0]
one_point = (point,)

result = runtime.warm_up(
    one_point,
    precision=16,
    color_flows=(flow,),
    n_cores=4,
    progress=progress,
)
```

The outer batch must contain exactly one phase-space point, and the precision
must be `16` (native binary64). This point is genuinely evaluated after the
family is prepared. Passing a 128-point profiling batch, or asking for higher
precision, is rejected. A repeated `warm_up` for the same family reuses its
structure but still performs the required one-point evaluation;
`result.already_warm` and `result.warmed_query_count` distinguish reuse from a
cold construction.

The optional progress callback (a function called whenever progress changes)
receives four stages:

1. process preparation;
2. query-family construction;
3. family finalization; and
4. the first one-point evaluation.

Progress includes completed/total work, elapsed time, worker count, and current
and peak resident memory (RAM physically held by the process) when the platform
supplies them. Without a callback or Python progress display, progress
reporting and its memory sampling are not enabled. The final first-evaluation
notification occurs after the warmed state has been committed and is not
cancellable.

The packaged example supplies a coloured terminal progress display and result
table:

```console
pyamplicol examples copy ./pyamplicol-examples
cd ./pyamplicol-examples
pyamplicol generate --card otf_pp_zjj.toml
python python/otf_pp_zjj_warm_up.py --n-cores 4
pyamplicol profile --card otf_pp_zjj.toml
```

The Python program keeps one runtime loaded across `warm_up` and the timed
evaluations, so the expensive family construction is not accidentally repeated
by a second process. The final CLI `profile` command is a separate benchmark
run and therefore loads its own runtime; it is included to demonstrate the
standard profiling interface, not to reuse the Python program's in-memory
family.

### The same warm-up in native languages

All native interfaces use one flattened point in
`[external particle][E, px, py, pz]` order and stable selector IDs. Omitting an
axis sums it.

In C, warm one flow and all helicities:

```c
const char *one_flow[] = {flow_id};
RusticolWarmUpResult result = {0};
int status = rusticol_runtime_warm_up_f64_with_cores(
    handle, point, momentum_count,
    NULL, 0, one_flow, 1,
    2, report_progress, user_data, &result);
```

The callback has this signature and returns nonzero to continue:

```c
int report_progress(
    const RusticolWarmUpProgressEvent *event,
    void *user_data);
```

The wrappers preserve the same selector order:

```cpp
auto result = runtime.warm_up(point, {}, {flow_id}, report_progress, 2);
```

```fortran
call runtime%warm_up(point, result, color_ids=flow_ids, &
    progress_callback=report_progress, progress_user_data=user_data, ierr=ierr, n_cores=2)
```

```rust
let selectors = Selectors::all().with_colors([flow_id]);
let result = runtime.warm_up_f64(
    &point, &selectors, Some(2), Some(&mut report_progress),
)?;
```

These examples request up to two construction workers for this call only.
Use zero in C/C++, omit `n_cores` in Fortran, or pass `None` in Rust to retain
the process-output default. The original C `rusticol_runtime_warm_up_f64`
entry point keeps its argument list without `n_cores` and selects that default.
Explicit Fortran values below one and Rust
`Some(0)` are rejected. The override neither changes evaluation defaults nor
invalidates an existing warm family or saved structural cache.

For all flows at one helicity, put one helicity ID in the helicity argument and
leave the colour argument empty. Passing no callback is valid. C++ releases at
destruction, Fortran exposes `runtime%close()`, Rust releases on `drop`, and C
uses `rusticol_runtime_free(handle)`. The native wrappers do not expose
Python's clear-without-unload convenience; close and reload to return to a
fully cold native handle.

### Model-parameter updates do not keep stale numbers

`runtime.set_model_parameters(...)` updates the loaded runtime atomically. On
the next warm-up or evaluation, an already retained OTF family refreshes its
parameter-dependent workspace before executing. Merely changing a numerical
model parameter does not require rebuilding the same structural selector
family, and the old parameter values are not silently reused.

Python `runtime.clear()` discards the OTF family, prepared process state, and
temporary workspace while keeping the process output and the latest parameter
values loaded. In recurrence, compiled, and eager modes, `clear()` is a no-op
because they do not have the corresponding OTF family cache.

## A practical decision sequence

1. Decide the colour accuracy from the physics question: `lc`, `nlc`, or
   `full`.
2. For recurrence, eager, or compiled LC output, decide which slice is repeated
   most often:
   one-flow/helicity-sum suggests `topology-replay`; all-flow/one-helicity
   suggests `all-flow-union`.
3. Start with recurrence for a reusable prepared-model output. Use eager when
   the time to generate and load a ready-to-run output is the priority, and
   compiled when process-wide compilation or a raw JSON/UFO model is needed.
4. Consider OTF when a compact process output and one dominant LC selection
   outweigh an expensive first construction. Keep the same loaded runtime
   alive so later calls reuse it.
5. Keep generation coverage complete until a measured production workload
   justifies a non-reusable specialized output.
6. Profile the exact process, selector, parameters, batch size, and target
   machine used by the application.

See [Configuration](configuration.md#color-accuracy-and-lc-layout) for every card
field, [Runtime and Selectors](runtime-and-selectors.md) for resolved output and
per-point selectors, and the [Native APIs](native-apis.md) for complete
buildable language examples.
