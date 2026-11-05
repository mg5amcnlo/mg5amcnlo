---
title: "Runtime and Selectors"
nav_order: 1
parent: "Python API"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Runtime and Selectors

pyAmpliCol process artifacts are designed to be evaluated after generation,
without rebuilding the model or process DAG. The Python `Runtime` and all
native APIs use the same RustiCol execution layer, process resolver, selector
planner, model-parameter state, and numerical conventions for ordinary
evaluations.

[Born Correlations](../correlators.md) use an opt-in, separate Python evaluator.
They accept generation-time colour-operator IDs and runtime spin vectors, not
the ordinary LC flow selector. This path currently supports full-colour SU(3)
only, and requires the original generated particle ordering (no nonidentity
aliases or automatic permutations). A process-group member is accepted only
with that ordering and without charge conjugation; its factor is applied.

> **Start here:** follow [Quick Start](quick-start.md) to create the example workspace and
> generate `artifacts/pp_zjj`. See [Generation Modes and Evaluators](generation-modes-and-evaluators.md) for the generation
> side of the workflow and [Artifacts and Portability](artifacts-and-portability.md) before moving an
> artifact between machines.

## Inspect before loading

Inspection reads the artifact inventory without loading executable evaluator
state:

```console
pyamplicol inspect artifacts/pp_zjj
```

The human view is a colored set of tables covering the artifact, model,
runtime target, concrete processes, crossing aliases, particle order,
helicities, color coverage, payload sizes, and dependencies. Use JSON for a
machine-readable inventory:

```console
pyamplicol inspect artifacts/pp_zjj --json > pp_zjj-inventory.json
```

For one process, either a stable ID or a process expression is accepted:

```console
pyamplicol inspect artifacts/pp_zjj --process p_p_to_z_j_j_3
pyamplicol inspect artifacts/pp_zjj --process 'd d~ > g z g'
```

For an on-the-fly artifact, this compact inventory reports the physical
selector census, requested query-construction workers, and binary64-only
runtime without instantiating the runtime or enumerating a dense selector
axis. That metadata-only view is the safe default at high multiplicity. Ask
for the legacy fully materialized physics view only when its complete selector
catalog is genuinely needed:

```console
pyamplicol inspect artifacts/otf_pp_zjj \
  --process 'd d~ > g z g' --full-physics
```

`--full-physics` loads executable evaluator state and can be factorially large
in the number of colored legs.

## Load a process from Python

```python
from pyamplicol import Runtime

runtime = Runtime.load(
    "artifacts/pp_zjj",
    process="d d~ > g z g",
)

print(runtime.execution_mode)  # compiled, eager, recurrence, or on-the-fly
print(runtime.artifact_id)             # 64 lowercase hexadecimal characters
print(runtime.physics.process)         # requested public particle order
print(runtime.physics.external_particles)
print(runtime.physics.helicity_ids)
print(runtime.physics.color_flow_ids)
```

If an artifact contains only one process, `process=` may be omitted. For a
multiprocess artifact, make the selection explicit so the chosen public
particle order is obvious at the call site.

## Process expressions and automatic permutations

Selection is resolved in this order:

1. stable process ID;
2. explicit alias ID;
3. exact stored process expression;
4. exact stored alias expression;
5. a unique permutation-equivalent expression.

Incoming and outgoing particles are matched independently. A particle never
crosses the `>` boundary. Repeated identical particles use deterministic
first-unused matching. If several stored representatives would fit, loading
fails and reports their stable IDs instead of guessing.

For example, the representative `p_p_to_z_j_j_3` is stored for one ordering,
but this is a valid public selection:

```python
runtime = Runtime.load(
    "artifacts/pp_zjj",
    process="d d~ > g z g",
)
```

Rusticol then applies one central side-preserving permutation to all public
axes:

- input four-momenta and external PDGs;
- external-particle metadata;
- helicity IDs, vectors, representatives, and selectors;
- leading-color flow words, IDs, replay labels, and selectors;
- reduction groups and resolved output metadata;
- compiled, eager, recurrence, on-the-fly, scalar/SIMD, and Python exact
  execution.

Consequently, momenta supplied to this runtime must follow `d, d~, g, z, g`,
not the representative's internal order.

A sixth step resolves processes that a `--flavor-blind` generation did not
generate but listed in the artifact's process-group map. Such a member is
evaluated by its representative at the composed permutation, with every
matrix element multiplied by the member's exact factor. A charge-conjugated
member reverses each LC flow word, so its flow IDs and selectors are the
member's own:

```python
runtime = Runtime.load("artifacts/qg", process="u~ g > u~ g")
print(runtime.representative_process_key)
print(runtime.process_group_factor, runtime.charge_conjugated)
```

Every other selection reports `process_group_factor == 1` and
`charge_conjugated is False`. See
[Process-group members](process-selection-and-permutations.md#process-group-members).

## Momenta

Python accepts a batch with shape
`(point, external particle, [E, px, py, pz])`. A single five-particle point is
therefore wrapped in one outer batch dimension:

```python
momenta = [[
    [500.0, 0.0, 0.0, 500.0],
    [500.0, 0.0, 0.0, -500.0],
    [462.6501613061637, 14.340107538562991,
     155.76435943335707, -425.7484539710246],
    [369.7738416261408, -17.479290785282917,
     2.0064955613504103, 369.3550355960509],
    [167.57599706769557, 3.1391832467199254,
     -157.77085499470743, 56.3934183749737],
]]
```

The fourth-vector convention is `[E, px, py, pz]`. The particle axis follows
the expression passed to `Runtime.load()`.

The CLI reads the same shape from JSON:

```console
pyamplicol evaluate artifacts/pp_zjj \
  --process 'd d~ > g z g' \
  --momenta data/pp_zjj_momenta.json
```

## Model parameters

Inspect mutable external parameters and their defaults before applying an
update:

```python
for parameter in runtime.physics.model_parameters:
    if parameter.mutable:
        print(parameter.name, parameter.default_real)
```

Mutable external model parameters can be set while loading:

```python
runtime = Runtime.load(
    "artifacts/pp_zjj",
    process="d d~ > g z g",
    model_parameters={"aS": 0.117, "MZ": 91.188},
)
```

They can also be updated atomically later:

```python
runtime.set_model_parameters({"aS": 0.1165, "MZ": 91.1876})
runtime.set_model_parameter("MT", 172.5)
```

Derived couplings and dependent parameters are refreshed before the update is
committed. If any name is unknown or immutable, or any value is invalid or
non-finite, the complete update is rejected.

A parameter card is a flat JSON object matching a serialized UFO restriction:

```json
{
  "aS": 0.117,
  "MZ": 91.188,
  "complex_external_parameter": [1.0, -0.25]
}
```

The generated standalone API drivers apply card values first, then let repeated
`--set-parameter NAME REAL IMAG` options override them. The main CLI accepts
the card through `--model-parameters`; Python callers pass a mapping while
loading or use the setters shown above. See
[Models and Processes](models-and-processes.md) for model compilation and card
production.

## Total and resolved evaluation

The optimized total returns one value per input point:

```python
total = runtime.evaluate(momenta)
```

The resolved path exposes the physical axes retained in the artifact:

```python
resolved = runtime.evaluate_resolved(momenta)

for optimized, explicit in zip(total, resolved.total(), strict=True):
    scale = max(1.0, abs(optimized))
    assert abs(optimized - explicit) <= 1.0e-12 * scale

print(resolved.shape)
print(resolved.helicity_ids)
print(resolved.color_flow_ids)
```

At leading color (LC), the resolved shape is `(point, helicity, physical
color flow)`. At NLC and full color, color is contracted and the final axis has
length one. That contracted singleton is output metadata, not a selectable LC
flow.

Human CLI output uses colored tables and scientific notation; add `--json` only
when a machine-readable representation is wanted.

## Batch-global selectors

Stable IDs come from `runtime.physics`:

```python
helicity = runtime.physics.helicity_ids[0]
flow = runtime.physics.color_flow_ids[0]

selected_total = runtime.evaluate(
    momenta,
    helicities=[helicity],
    color_flows=[flow],
)

selected_resolved = runtime.evaluate_resolved(
    momenta,
    helicities=[helicity],
    color_flows=[flow],
)
```

`--helicity` accepts a stable helicity ID. `--color-flow` accepts either a
stable flow ID or a one-based advertised position; for example,
`--color-flow 1` selects the first physical LC flow.

Color-flow selection is LC-only. NLC/full accept helicity selectors and reject
color-flow selectors.

## Stay on the generated LC layout's optimized workload

LC artifacts retain the layout chosen during generation. Runtime selectors do
not switch an artifact from one layout to the other. Inspect the complete
artifact first; its process-execution table includes an `LC flow layout` row:

```console
pyamplicol inspect artifacts/my_process
```

Then invoke `Runtime.evaluate()` with the selector pattern that matches that
layout:

| Generated LC layout | Supply | Deliberately omit | Optimized total computed per point |
| --- | --- | --- | --- |
| `topology-replay` (non-union) | exactly one color-flow ID | helicity selector | selected flow, summed over all physical helicities |
| `all-flow-union` | exactly one helicity ID | color-flow selector | selected helicity, summed over all physical flows |

For the default `topology-replay` layout:

```python
flow = runtime.physics.color_flow_ids[0]

single_flow_helicity_sum = runtime.evaluate(
    momenta,
    color_flows=(flow,),
    precision=16,
)
```

The equivalent CLI call uses one stable flow ID, or the one-based position
shown by `inspect`:

```console
pyamplicol evaluate artifacts/my_process \
  --process 'd d~ > g z g' \
  --momenta data/point.json \
  --precision 16 \
  --no-resolved \
  --color-flow 1
```

For an artifact generated with `lc_flow_layout = "all-flow-union"`:

```python
helicity = next(
    item
    for item in runtime.physics.helicities
    if item.computed and not item.structural_zero
)

all_flows_single_helicity = runtime.evaluate(
    momenta,
    helicities=(helicity,),
    precision=16,
)
```

The CLI form uses the stable helicity ID printed by `inspect`:

```console
pyamplicol evaluate artifacts/my_union_process \
  --process 'd d~ > g z g' \
  --momenta data/point.json \
  --precision 16 \
  --no-resolved \
  --helicity 'h:-1,+1,-1,+1,+1'
```

For a mixed batch, `color_flow_by_point=(...)` with no helicity selector is the
per-point form of the topology-replay workload;
`helicity_by_point=(...)` with no color-flow selector is the per-point form of
the all-flow-union workload.

To sample one helicity and one LC flow from a helicity-generic output, select
both axes at runtime:

```python
single_flow_single_helicity = runtime.evaluate(
    momenta,
    helicities=(helicity,),
    color_flows=(flow,),
    precision=16,
)
```

With `topology-replay`, recurrence prepares and caches the current dependencies
needed by the selected helicity and flow. Compiled evaluation selects the
corresponding colour and helicity sub-schedules instead of computing an
all-colour companion. The output still supports every generated helicity:
changing the runtime selectors requires no regeneration. Shared currents and
physical helicity/flow relabellings are preserved. Omitting the helicity
selector retains the existing helicity-summed evaluation path.

Use the same selectors with the public profiler to time the warmed, batched
single-flow/single-helicity workload:

```console
pyamplicol profile artifacts/my_process \
  --color-flow 1 --helicity 'h:-1,+1,-1,+1,+1' \
  --batch-size 128 --precision 16
```

Other selector combinations remain valid when the output has the required
coverage. Omitting the layout's required selector asks for a broader sum. Use
`Runtime.evaluate()` at `precision=16` for the optimized native total;
`evaluate_resolved()` and higher precision are diagnostic/exact paths rather
than that benchmarked execution unit.

These two stored layouts apply to recurrence, compiled, and eager artifacts.
Those modes can also fix flow IDs, helicity IDs, or both during generation when
a deliberately specialized artifact is preferable. OTF does not materialize a
layout or use generation-time specialization: one compact artifact builds the
family implied by the runtime selectors and retains only the last selected
family. Its practical high-multiplicity target is one selected LC flow summed
over all helicities. See
[LC workloads and execution modes](lc-workloads-and-execution-modes.md) for the
physical distinction and call signatures.

### Omitted selectors and requesting every entry

The meaning of omission depends on the operation:

- `Runtime.evaluate(momenta)` omits both axes and therefore sums every retained
  physical helicity and color flow: it is the complete matrix element.
- Profiling with both selector lists empty asks pyAmpliCol to choose the stored
  layout's deterministic hot workload described above.
- A complete explicit list on the layout's summed axis is equivalent to
  omission and still permits the optimized selector to be inferred. A subset
  on that axis is preserved and prevents inference.
- An explicit list on the layout's selected axis is always intentional. A
  complete list there requests the broader all-entry workload rather than
  being replaced by one inferred selector.

There is deliberately no magic `all` selector token. To profile every member
of an axis explicitly, pass the complete stable-ID tuple exposed by process
metadata. For example, either of these requests the all-helicity/all-flow LC
total while making the intent unambiguous to the profiler:

```python
from pyamplicol import BenchmarkConfig, BenchmarkRunner

# Natural explicit-all spelling for a topology-replay artifact.
topology_all = BenchmarkRunner(
    BenchmarkConfig(color_flow_ids=runtime.physics.color_ids)
).run(runtime, points=momenta)

# Natural explicit-all spelling for an all-flow-union artifact.
union_all = BenchmarkRunner(
    BenchmarkConfig(helicity_ids=runtime.physics.helicity_ids)
).run(runtime, points=momenta)
```

Those are valid broader workloads and consequently emit the non-hot warning.
For CLI profiling, read the stable IDs from `pyamplicol inspect` and repeat the
option once per entry, for example
`--color-flow FLOW_1 --color-flow FLOW_2 ...` or
`--helicity HELICITY_1 --helicity HELICITY_2 ...`. Color-flow one-based
ordinals may be repeated instead of stable flow IDs. Supplying complete lists
on both axes is also valid, but redundant. The complete list on the selected
axis is the part that makes the all-entry profiling intent explicit.

## One selector per point

Complete-coverage artifacts may mix selectors inside one batch:

```python
mixed = runtime.evaluate(
    momenta_batch,
    color_flow_by_point=[
        runtime.physics.color_flow_ids[i % 2]
        for i in range(len(momenta_batch))
    ],
    helicity_by_point=[
        runtime.physics.helicity_ids[(i // 2) % 2]
        for i in range(len(momenta_batch))
    ],
)
```

Rusticol stably groups equal selectors to preserve contiguous SIMD work, then
restores caller order. A batch-global selector and a per-point selector cannot
be supplied on the same axis. Per-point selection is available for optimized
totals; rectangular `evaluate_resolved()` uses batch-global axes.

## Explicit OTF warm-up

OTF exposes its cold structural work through `warm_up(...)`. The call accepts
exactly one phase-space point at native binary64 precision and the same
batch-global selectors as `evaluate(...)`:

```python
import sys

from pyamplicol import Runtime
from pyamplicol.reporting import close_progress_sink, progress_sink

runtime = Runtime.load("artifacts/otf_pp_zjj", process="d d~ > g z g")
point = points[0]  # one [external particle][E, px, py, pz] point
flow = runtime.physics.color_flows[0]
progress = progress_sink("auto", stream=sys.stderr, color=True)
try:
    result = runtime.warm_up(
        (point,),
        precision=16,
        color_flows=(flow,),
        n_cores=4,
        progress=progress,
    )
finally:
    close_progress_sink(progress)
```

Warm-up never uses the later profiling batch: passing 128 points, or any
precision other than `16`, is rejected. It builds the selected query family,
retains it on this runtime handle, and genuinely evaluates the supplied point.
Repeating the same request reuses the family but still performs that required
one-point evaluation. The returned `WarmUpResult` records elapsed time, total
and newly built query counts, cache reuse, and sampled memory.

The optional positive integer `n_cores` bounds the workers constructing query
traces in parallel. Completed traces are merged into the shared cache serially;
family finalization and the first evaluation also remain serial. More workers
use additional temporary memory, and the speed-up depends on the selected
family. Omit `n_cores` (or pass `None`) to use the process output's
`evaluator.optimization.cores` setting. This override applies only to this
warm-up call: it is not saved in the cache, does not change subsequent
evaluation threading, and does not rebuild an already-warm family.

The optional progress stream separates process preparation, query-family
construction, family finalization, and first evaluation. It includes completed
query counts, construction workers, and current/peak resident memory when the
platform exposes them. With no observer, warm-up performs no progress callback
or memory-sampling work.

The cache is handle-local and last-family-only: successfully selecting another
flow or helicity replaces the previously warmed family. LC accepts flow and
helicity selectors; contracted NLC/full accepts helicity selectors only.
`runtime.clear()` returns a Python OTF handle to cold state while keeping its
artifact and current model parameters loaded. Closing/freeing a native handle
releases all of its warm state. C, C++, Fortran, and Rust expose the same four
stages through a fixed-layout optional callback; see
[Native APIs](native-apis.md) and the complete
[OTF lifecycle walkthrough](lc-workloads-and-execution-modes.md#the-otf-warm-state-lifecycle).

## Saving an OTF warm cache

`save(path)` writes the completed OTF cache; `load_cache(path)` restores it
into a runtime loaded from the matching process output. Both return `None`.
For a small example, generate a four-gluon process with generic helicity and
flow coverage:

```console
pyamplicol generate "g g > g g" artifacts/otf_gg_gg \
  --model built-in-sm --color-accuracy lc --execution-mode on-the-fly
```

Then warm one LC flow, summed over all helicities, and save it:

```python
from pyamplicol import Runtime

point = (
    (500.0, 0.0, 0.0, 500.0),
    (500.0, 0.0, 0.0, -500.0),
    (500.0, 300.0, 0.0, 400.0),
    (500.0, -300.0, 0.0, -400.0),
)
runtime = Runtime.load("artifacts/otf_gg_gg")
flow = runtime.physics.color_flows[0].id
selection = {"color_flows": (flow,)}  # no helicity selector: sum helicities
runtime.warm_up((point,), **selection)
runtime.save("gg.otf-cache")

# This can instead be done in a later Python session.
restored = Runtime.load("artifacts/otf_gg_gg")
restored.load_cache("gg.otf-cache")
report = restored.warm_up((point,), **selection)
assert report.already_warm and report.warmed_query_count == 0
values = restored.evaluate((point, point), **selection)
print(values)
```

The example repeats its point only to keep the two-point batch short; a later
batch may contain different momenta. A successful ordinary `evaluate(...)`
also leaves saveable state, so explicit `warm_up(...)` is optional. The extra
warm-up after restoration above demonstrates cache reuse and can be omitted.
Keep the same selectors when reusing the saved family: `load_cache` restores
cached work, not new defaults for subsequent `evaluate` calls.

Both methods are OTF-only. The same operations are available through the
[native APIs](native-apis.md#saving-and-restoring-an-otf-cache).
The snapshot contains the retained structural calculation, including its
helicity/colour selection and reduction mappings. It supports selected LC
requests, helicity/flow sums, and contracted NLC/full-colour families, with
either direct colour contraction or FFT where the original process supports it.

Restoration skips current construction and binds the saved schedule to the
prepared kernels in the matching process output. It does not save evaluated
currents, native pointers or numeric workspaces. New momenta and batch sizes
are supported, and the receiving runtime keeps its current model parameters.
The original process output is still required; the cache is not a standalone
process export. Incompatible process outputs or snapshot formats are rejected
without replacing the receiving runtime's existing cache.

The cache path names a file (its parent directory must already exist). Saving
again replaces that file; a failed save leaves any previous complete file
intact. Empty caches and completed zero-valued selections are also saveable.

Saving does not change the running handle or its warmed evaluation path.
It saves the state currently retained, not a history of evicted selections.
A different selection after restoration follows the usual OTF construction
and replacement rules. This is a completed-state snapshot, not a checkpoint
of an unfinished warm-up.

## Precision

For ordinary evaluation, `precision=16` uses the native f64 RustiCol runtime. Direct JIT artifacts load
the separate MIT-licensed SymJIT runtime; compatible C++ and ASM artifacts load
their native evaluator libraries. None of these f64 paths imports Symbolica or
checks a Symbolica runtime license.

The separate `evaluate_correlated` method uses retained Symbolica evaluator
states at every requested precision, including 16; see
[Born Correlations](../correlators.md).

Python may request another positive decimal precision:

```python
high_precision = runtime.evaluate(momenta, precision=80)
```

This higher-precision path exists only when the artifact retains an exact
evaluator. OTF artifacts are native-f64-only and reject every precision other
than `16` in Python as well as in the native APIs.

That path lazily loads retained Symbolica evaluator state and is therefore
subject to Symbolica availability and licensing. Decimal inputs retain their
digits. Values first supplied as binary64 do not acquire new physical
information simply because a larger arithmetic precision is requested.
Results are rounded to the requested decimal precision after guard-digit
evaluation. Derived model parameters and couplings are reevaluated from their
retained symbolic definitions at this precision, including constants such as
pi. Coupling powers and combinatorial normalization factors are also evaluated
in the requested arithmetic, rather than copied from the native f64 result.
Generated full-colour and NLC contraction weights also retain their rational
coefficients for higher-precision evaluation; native evaluation continues to
use their binary64 values.
This matters when subtracting matrix elements at different multiplicities.

After updating from a version that rounded these definitions during generation,
regenerate the prepared model and process output: increasing `precision` cannot
repair coefficients already rounded in an older output. The prepared-model
compiler version rejects incompatible bundled kernels. The public
`precision=32` call uses guarded arbitrary precision, not a dedicated
double-double runtime. See
[Symbolica and Licensing](symbolica-and-licensing.md).

Native C, C++, Fortran, and Rust APIs expose f64 only. Their shared SDK is
covered in [Native APIs](native-apis.md).

## Profiling one runtime

Profile the same optimized total path used by `Runtime.evaluate()`:

```console
pyamplicol profile artifacts/pp_zjj \
  --process 'd d~ > g z g' \
  --momenta data/pp_zjj_momenta.json \
  --target-runtime 1.0 \
  --batch-size 128 \
  --color-flow 1 \
  --precision 16
```

If both profiling selector axes are omitted, pyAmpliCol derives one
deterministic selector matching the stored hot layout. Explicit subsets and
selected-axis lists are preserved; a complete summed-axis list is normalized
to equivalent omission as described above. A valid non-hot profile instead
emits at most one pre-loop warning per loaded process. This profiling
convenience does not change the complete-matrix-element default of
`Runtime.evaluate()`.

For OTF, the profiler first snapshots the compact native runtime census, times
one requested-selector evaluation over the configured benchmark batch, and
then authenticates that the family is retained before ordinary configured
warmups begin. The result records whether the starting handle was cold or
already warm. This implicit profiling preparation is distinct from the public
one-point `warm_up(...)` API and is reported separately; neither preparation
time nor configured warmups are folded into the steady-state wall samples.

A retained structural-zero request intentionally has no amplitude destination,
executor handle, semantic binding, or active family-union census. The profiler
recognizes that as a valid retained state instead of inventing executable work.
OTF native attribution labels the narrower recurrence-schedule core separately
from the complete evaluator envelope and from recurrence mode's statically
generated core.

The headline wall time is sampled independently of the paired native
attribution pass. `evaluator total` is a minimally instrumented complete
evaluator call. Native component rows describe a separate profiled pass and
must not be subtracted from that headline as though they were measured in one
identical call. `pyamplicol benchmark` is a compatibility alias for `profile`.

## Handle ownership and warnings

Parameter and warning state belongs to one runtime handle. Do not invoke the
same mutable handle concurrently. Independent handles may execute on separate
threads.

Artifacts are trusted executable inputs. Generate them yourself or obtain
them through a trusted channel; content hashes establish internal consistency,
not publisher identity. See [Artifacts and Portability](artifacts-and-portability.md) for the fast load
boundary and optional explicit checksum audit.

## Related pages

- [Quick Start](quick-start.md) — generate and evaluate the primary example.
- [Python API](python-api.md) — typed generation, model, and benchmark interfaces.
- [Native APIs](native-apis.md) — C11, C++17, Fortran 2008, and Rust 2021.
- [LC workloads and execution modes](lc-workloads-and-execution-modes.md) — LC
  call shapes and the OTF warm-state lifecycle.
- [Troubleshooting](troubleshooting.md) — missing artifacts, target mismatches, and stale source
  environments.
