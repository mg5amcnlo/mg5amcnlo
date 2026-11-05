---
title: "Python API"
nav_order: 4
has_children: true
---
<!-- SPDX-License-Identifier: 0BSD -->
# Python API

The public Python interface offers typed services for model compilation,
process planning/generation, runtime evaluation, Born correlations, selectors,
and profiling.
Importing `pyamplicol` is lightweight: Symbolica and model tooling are loaded
only when the requested operation needs them.

```python
import pyamplicol

print(pyamplicol.__version__)
```

## Public service map

| Service | Purpose |
| --- | --- |
| `ModelSource` | Resolve and compile built-in, JSON, UFO, compiled, or prepared models. |
| `Generator` | Plan or generate schema-v3 process artifacts. |
| `Runtime` | Load one concrete process and evaluate totals/resolved components. |
| `CorrelatorConfig`, `ColorCorrelator` | Declare allowed spin replacements and named colour operators at generation. |
| `CorrelatedValue` | A correlated result with `Decimal` real and imaginary parts. |
| `BenchmarkRunner` | Profile an artifact or loaded runtime. |
| `group_processes` | Group the subprocesses of an artifact with equal matrix elements. |
| `generate`, `load`, `benchmark` | One-shot convenience functions. |

The shipped scripts under
[`examples/python/`](https://github.com/MadGraphTeam/pyamplicol/tree/main/examples/python)
are complete runnable programs.

## Plan and generate with the built-in model

```python
from pyamplicol import GenerationConfig, Generator, ModelSource

generator = Generator(GenerationConfig(workers=2))
model = ModelSource.built_in_sm()

plan = generator.plan("d d~ > z g", model=model)
for process in plan.concrete_processes:
    print(process.name, process.expression)

result = generator.generate(
    "d d~ > z g",
    "artifacts/builtin_ddbar_to_zg",
    model=model,
    mode="replace",
)
print(result.output)
```

`plan()` resolves the model/process/color/evaluator contract without writing a
process artifact. `generate()` returns an immutable `GenerationResult` with the
absolute output, stored process set, schema version, and file inventory.

The packaged scalar HEFT model uses the same public selector:

```python
from pyamplicol import Generator, ModelSource

heft = ModelSource.built_in_sm_heft()
Generator().plan("g g > H g g", model=heft)
```

Use an explicit `HIG = 1` coupling-order limit in a resolved run configuration
when generating the process; see [Models and Processes](models-and-processes.md#built-in-scalar-heft-model)
and the packaged `builtin_sm_heft.toml` card.

## Generate with adjoint FFT contraction

The Python API uses the same `RunConfig` and `ColorConfig` fields as TOML and
the CLI. Defaults `contraction="auto"` and `fft_basis="adjoint"` try adaptive
exact FFT for NLC/full recurrence and on-the-fly execution, with direct fallback
for unsupported FFT plans. This certified pure Yang–Mills example uses DDM:

```python
from pyamplicol import ColorConfig, Generator, ModelSource, RunConfig
from pyamplicol.config import EvaluatorConfig

config = RunConfig(
    action="generate",
    color=ColorConfig(accuracy="full"),
    evaluator=EvaluatorConfig(execution_mode="recurrence"),
)
result = Generator(config).generate(
    "g g > g g g",
    "artifacts/ggg_adjoint_fft",
    model=ModelSource.built_in_sm(),
)
```

Select `accuracy="nlc"` for NLC contraction, or `execution_mode="on-the-fly"`
for a compact OTF artifact. The effective configuration can retain `"auto"`
until per-process selection. LC, compiled/eager execution, and `correlators=`
use direct contraction and the trace basis under the automatic policy.

Use `ColorConfig(accuracy="full", contraction="direct")` to opt out of FFT;
direct contraction ignores `fft_basis`. To force FFT and retain errors for
unsupported plans, set `contraction="symmetric-group-fft"`, optionally with
`fft_basis="trace"` for the original representation. The string `"adjoint"`
is equivalent to `ColorFFTBasis.ADJOINT` (imported from `pyamplicol`). Explicit
FFT does not support compiled/eager modes or LC accuracy.

For certified trees with `n` external gluons, DDM fixes two anchors and reduces
the trace set from `(n-1)!` orderings to `(n-2)!`, representing the same amplitude.
Certified single-insertion HEFT also uses DDM; specify explicit coupling orders
with `HIG = 1`, as in `examples/builtin_sm_heft.toml`. Quark processes keep exact
fundamental chains or their products, not a newly assumed JO primitive basis;
uncertified adjoint interactions retain trace tensors. The saved
`fft_basis_selection` reports `requested`, `actual_basis`, `name`, `reason`,
`permutation_degree`, and `tensor_count`, making that choice explicit.
Forced adjoint FFT cannot be combined with `correlators=`. Runtime totals,
resolved helicities and native SDK evaluation calls are unchanged. No universal
speedup is implied.
In the benchmarked pure-gluon family the adjoint basis was faster than trace
for six or more external gluons (3.9x per sample at ten gluons); benchmark
other multiplicities and selector workloads.

## Generate a named process set

```python
from pyamplicol import Generator, ModelSource, ProcessRequest, ProcessSet

processes = ProcessSet(
    requests=(
        ProcessRequest.parse("u u~ > Z g", name="uubar_Zg"),
        ProcessRequest.parse("u u~ > Z g g", name="uubar_Zgg"),
    )
)

result = Generator().generate(
    processes,
    "artifacts/z_ladder",
    model=ModelSource.built_in_sm(),
    mode="replace",
)
```

Process names must be unique. For custom aliases and crossing metadata, use
`ProcessAlias`; runtime expression matching already handles unique
side-preserving permutations automatically.
The correlated reference path currently requires identity ordering; these
permutation facilities apply to ordinary evaluation.

## Compile and reuse an external model

```python
from pyamplicol import Generator, ModelSource

source = ModelSource.from_path(
    "models/json/sm/sm.json",
    restriction="default",
)
model = source.compile()

print(model.info.name)
print(model.capabilities.supported_color_accuracies)
print(model.supported)

Generator().plan("d d~ > z g", model=model)
```

For a trusted UFO directory:

```python
source = ModelSource.from_path(
    "/path/to/MyUFO",
    restriction="restrict_default.dat",
)
model = source.compile()
```

UFO modules execute Python while loading. See
[Models and Processes](models-and-processes.md) for trust, serialization, and
prepared bundles.

## Use a fully resolved run card

For complex configuration, resolve the same schema used by the CLI and pass the
result to `Generator`:

```python
from pyamplicol import Generator, ModelSource, ProcessRequest, ProcessSet
from pyamplicol.config import resolve_config

card = {
    "schema_version": 1,
    "action": "generate",
    "model": {"source": "models/json/sm/sm.json", "restriction": "default"},
    "process": {
        "entries": [{"expression": "p p > Z j j"}],
        "multiparticles": {"p": ["d", "d~", "g"], "j": ["d", "d~", "g"]},
        "flavor_scheme": 2,
        "max_quark_lines": 2,
    },
    "color": {"accuracy": "lc"},
    "generation": {"output": "artifacts/pp_zjj", "mode": "replace"},
    "evaluator": {"execution_mode": "compiled", "backend": "jit"},
}

resolution = resolve_config(card)
model = ModelSource.from_config(resolution.effective.model).compile()
processes = ProcessSet(
    tuple(
        ProcessRequest.parse(entry.expression, name=entry.name)
        for entry in resolution.effective.process.entries
    )
)
result = Generator(resolution).generate(
    processes,
    resolution.effective.generation.output,
    model=model,
    mode=resolution.effective.generation.mode,
)
```

The complete maintained implementation is
[`examples/python/typed_generation.py`](https://github.com/MadGraphTeam/pyamplicol/blob/main/examples/python/typed_generation.py).

## Load a runtime

```python
from pyamplicol import Runtime

runtime = Runtime.load(
    "artifacts/pp_zjj",
    process="d d~ > g z g",
    model_parameters={"aS": 0.117, "MZ": 91.1876},
)

print(runtime.artifact_id)
print(runtime.execution_mode)
print(runtime.physics.process_id)
print(runtime.physics.process)
print(runtime.representative_process_key)
print(runtime.external_permutation)
```

`process` may be a stable process ID, explicit alias ID, exact stored expression,
or unique permutation-equivalent expression within each side. The loaded
runtime exposes metadata in the requested public order.
For correlated evaluation, load the stored process in its generated ordering;
nonidentity aliases/permutations are not supported by that path yet.

## Evaluate totals and resolved components

```python
import json
import math
from pathlib import Path

points = json.loads(Path("data/pp_zjj_momenta.json").read_text())

totals = runtime.evaluate(points)
resolved = runtime.evaluate_resolved(points)

print(resolved.helicity_ids)
print(resolved.color_ids)
for optimized, explicit in zip(totals, resolved.total(), strict=True):
    assert math.isclose(optimized.real, explicit.real, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(optimized.imag, explicit.imag, rel_tol=1e-12, abs_tol=1e-15)
```

Input shape is:

```text
(point, external particle, [E, px, py, pz])
```

At LC, resolved shape is `(point, helicity, color_flow)`. At NLC/full,
the color dimension has length one because color is contracted.

## Retain completed OTF preparation

For an on-the-fly runtime, `runtime.save(path)` writes the completed cache
left by evaluation or explicit warm-up. In a later program, load the original
process output with `Runtime.load(...)`, then call `runtime.load_cache(path)`
before evaluating with the same selectors. Both methods return `None`; the
restored calculation accepts new momenta and batch sizes while retaining the
receiving runtime's model parameters.

See the [complete four-gluon example](runtime-and-selectors.md#saving-an-otf-warm-cache)
for generation, saving and restoring, and a check that no queries are rebuilt.
The cache supplements the original process output; it does not replace it or
checkpoint an unfinished cold construction.

## Evaluate Born correlations

Correlations require an explicit `Generator.generate(..., correlators=...)`
declaration. They support **LC, NLC and full colour**, through a separate Python
direct, generic compiled exact executor. `color.accuracy` selects the correlated
approximation. Supplying `correlators` retains complete underlying amplitudes
and selects direct contraction and compiled generation. Ordinary generation
and evaluation remain unchanged.

```python
from pyamplicol import (
    ColorCorrelator, CorrelatorConfig, Generator, ModelSource, Runtime,
)
from pyamplicol.config import ColorConfig, RunConfig

declarations = CorrelatorConfig(
    color_correlations=(ColorCorrelator.dipole("T13", 1, 3),),
    spin_correlations=((3,), (3, 4)),
)
generated = Generator(RunConfig(color=ColorConfig(accuracy="full"))).generate(
    "g g > g g", "artifacts/gg_correlated",
    model=ModelSource.built_in_sm(), correlators=declarations,
)
correlated_runtime = Runtime.load(generated.output)
point = (
    (500, 0, 0, 500), (500, 0, 0, -500),
    (500, 300, 0, 400), (500, -300, 0, -400),
)
colour = correlated_runtime.evaluate_correlated(
    (point,), color_correlation="T13", precision=40,
)[0]
correlated_runtime.set_spin_correlation_vectors({3: (0, 0, 1, 0)})
spin_colour = correlated_runtime.evaluate_correlated(
    (point,), color_correlation="T13", precision=40,
)[0]
print(spin_colour.real, spin_colour.imag)  # Decimal values, not a forced real part.
correlated_runtime.set_spin_correlation_vectors(None)
```

Instead of explicit colour requests, use
`CorrelatorConfig.all_color(through_order=k, spin_correlations=...)` before
generation to prepare every compatible directed pair of tree-soft connections
through any positive order `k`. The nonminimal catalogue includes emissions from
emitted partons and splitting of auxiliary gluons, and grows rapidly with order.
It adds no kinematic kernels, flavour sums or multiplicity weights. Inspect the
operators actually stored for a selected process with:

```python
for entry in correlated_runtime.available_color_correlations():
    print(entry.id, entry.order, entry.bra, entry.ket)
```

This includes `"born"` and reads the catalogue without initializing amplitude
evaluation. See the [automatic catalogue guide](../correlators.md)
for generation and grouped evaluation.

`evaluate_correlated(momenta, *, color_correlation="born", helicities=None,
precision=16, arithmetic="arbitrary")` returns one `CorrelatedValue` per point. The reserved ID
`"born"` uses the Born metric at the requested colour accuracy; other IDs are the declared
operators. Optional batch-global helicity selectors accept stable IDs or typed
`HelicityConfiguration` objects. There is no colour-flow or per-point selector
argument on this method. Normalization, initial-state averages, and
identical-particle factors are retained, and runtime model-parameter updates
also apply to correlations.

For multiple combinations, `evaluate_correlated_many(points, requests)` takes
a mapping from your result labels to `CorrelatedRequest(color_correlation=...,
spin_vectors=...)`. Its result is a dictionary of tuples:
`result[request_label][point_index].real` or `.imag` selects one contraction
at one phase-space point. Label and point order are preserved. Identical spin
assignments share their amplitudes across colour operators, and unchanged
stages can be reused across different assignments. See the
[two-axis example](../correlators.md#several-requests-and-several-phase-space-points)
for request-centric and point-centric access. Ordinary `evaluate()` on the
underlying correlated output remains full-colour; use the correlated `"born"`
request when comparing its selected LC/NLC approximation.

The spin-vector mapping uses one-based public leg labels. Each value is a
literal real or complex `(v0, vx, vy, vz)` vector, broadcast across the batch,
or one such vector per point. The nonempty key set must exactly match a
declared spin class: this example permits `{3}` and `{3, 4}`, but not `{4}`.
Vectors are neither normalized nor projected. They are copied into
instance-local state; a failed setter preserves the previous state. `None` or
`{}` resets to ordinary helicity sources; `clear()` releases warmed evaluators
without resetting vectors. The setter affects `evaluate_correlated` and
inherited requests in `evaluate_correlated_many`, not `evaluate` or
`evaluate_resolved`.

`Decimal` vector components retain their digits; a `(Decimal(real),
Decimal(imaginary))` pair represents one complex component without first
rounding it through Python's binary64 `complex`. For example:

```python
from decimal import Decimal

zero = Decimal(0)
correlated_runtime.set_spin_correlation_vectors({3: (
    zero, zero,
    (Decimal("1.0000000000000000000000001"), Decimal("2e-25")),
    zero,
)})
arb = correlated_runtime.evaluate_correlated((point,), precision=80)
dd = correlated_runtime.evaluate_correlated(
    (point,), arithmetic="double-double", precision=31,
)
correlated_runtime.set_spin_correlation_vectors(None)
```

The default is arbitrary precision. Explicit `arithmetic="double-double"`
uses genuine DoubleFloat arithmetic throughout sources, evaluation stages,
normalization and reduction; at most 31 result digits are supported. The same
keyword and Decimal vector syntax apply to `evaluate_correlated_many` and its
explicit request vectors. See the [grouped precision example](../correlators.md#decimal-inputs-and-double-double-arithmetic).

Artifacts without declarations, append generation, partial source coverage,
FFT, eager/recurrence/OTF correlated execution, replay reductions, and
nonidentity process permutations are not supported. Ordered `EmitGluon` and
`SplitGluon` operations have no fixed order cap; generic NkLO colour-connection
order is distinct from LC/NLC colour approximations and does not supply a
complete higher-order subtraction calculation. See [Born Correlations](../correlators.md)
for the bra-adjoint/ket convention, joint spin contractions, JSON declarations,
Ward checks, and current limitations.

The same generated colour catalogue and spin classes are available to the
[C/C++/Fortran/Rust SDKs](native-apis.md#correlated-born-evaluations), whose
correlated evaluation is binary64 and independent of Python/Symbolica.
The Python methods described here retain their precision-controlled executor.

## Select helicities and color flows

Batch-global selectors accept stable IDs or the typed objects exposed by
`runtime.physics`:

```python
selected = runtime.evaluate(
    points,
    helicities=[runtime.physics.helicities[0]],
    color_flows=[runtime.physics.color_flows[0]],
)
```

For one selector per point:

```python
mixed = runtime.evaluate(
    points,
    helicity_by_point=[
        runtime.physics.helicity_ids[index % 2]
        for index in range(len(points))
    ],
    color_flow_by_point=[
        runtime.physics.color_flow_ids[(index // 2) % 2]
        for index in range(len(points))
    ],
)
```

Batch-global and per-point selectors are mutually exclusive on the same axis.
Per-point color selection is LC-only. Rusticol groups mixed selectors while
returning values in original point order.

See [Runtime and Selectors](runtime-and-selectors.md) for the complete selector
contract.

## Update model parameters

```python
runtime.set_model_parameters({"aS": 0.1165, "MZ": 91.1876})
runtime.set_model_parameter("MT", 172.5)
```

Updates are atomic and refresh dependent derived parameters. Unknown,
immutable, non-finite, or invalid values reject the entire update.

`Runtime.load(..., model_parameters=...)` applies a complete mapping before the
runtime is returned. Complex inputs may be represented as Python complex values.

## Group equivalent subprocesses

```python
from pyamplicol.api import group_processes

summary = group_processes("artifacts/pp_jj")
for relation in summary["new_relations"]:
    print(relation["process"], "=", relation["factor"], "x", relation["representative"])
```

`group_processes` evaluates each group representative stored in
`process-groups.json` in f64 at three on-shell points, each with its own
deterministic variation of up to 10% of the external and normalization model
parameters other than masses and widths. Two subprocesses with the same
external particle types are grouped when, through a leg permutation (and
charge conjugation, unless `consider_charge_conjugation=False`), every
resolved helicity and colour
component of one is a common rational factor times the other's; numerator and
denominator are at most 99 and `only_unit_multiplier=True` accepts only 1.
Subprocesses generated with a selected helicity or colour subset are never
grouped. The map is updated in place and rerunning adds nothing; the artifact
ID and UMAMI files are unchanged. `analysis="none"` only writes the trivial map
of an artifact that has none. This is the `pyamplicol group-processes`
command; see [Process Grouping](process-grouping.md) for the details.

## Precision

```python
f64 = runtime.evaluate(points, precision=16)
high_precision = runtime.evaluate(points, precision=80)
```

For ordinary `evaluate` and `evaluate_resolved`, precision 16 uses the native
Rusticol runtime without Symbolica computations. Package startup imports
Symbolica to attempt registration of the pyAmpliCol library key; synchronous
registration failures retain the personal-license and restricted-mode paths.
Other positive precision requests load retained exact evaluator state when
the artifact supports it.
Decimal input preserves the supplied decimal digits;
binary64 input cannot gain information merely by requesting more arithmetic
precision.

`evaluate_correlated` always uses retained Symbolica evaluator states through
the separate Python exact executor, including at its default precision 16.
Its default `arithmetic="arbitrary"` is distinct from the explicit
`arithmetic="double-double", precision=31` mode; neither is selected by an
ordinary native SDK precision flag.
Its real and imaginary results remain `Decimal` values; `complex(value)` is an
explicit conversion to ordinary double precision.

Generated C/C++/Fortran/Rust standalone APIs support f64 (`precision=16`) only.

## Profile from Python

```python
from pyamplicol import BenchmarkConfig, BenchmarkRunner

config = BenchmarkConfig(
    target_runtime=1.0,
    batch_size=128,
    precision=16,
    warmup_runs=2,
    minimum_samples=5,
    color_flow_ids=("1",),
)

result = BenchmarkRunner(config).run(runtime, points=points)
print(result.wall_time_per_point)
print(result.evaluator_time_per_point)
print(result.uncertainty.standard_error)
print(result.uncertainty.relative_standard_error)
```

`BenchmarkResult` also contains calibrated repetitions, sample count,
environment metadata, native timing breakdown, stage attribution, and work
counters when the runtime exposes them.

The runnable reference is
[`examples/python/benchmark.py`](https://github.com/MadGraphTeam/pyamplicol/blob/main/examples/python/benchmark.py).

## Convenience functions

```python
from pyamplicol import benchmark, generate, load

generated = generate(
    "d d~ > z g",
    "artifacts/builtin_ddbar_to_zg",
    model=ModelSource.built_in_sm(),
    mode="replace",
)
runtime = load(generated.output)
profile = benchmark(runtime, points=points)
```

The class-based services are preferable when reusing configuration, progress
sinks, compiled models, or loaded runtimes.
The one-shot `generate(...)` function also accepts the same
`correlators=CorrelatorConfig(...)` keyword as `Generator.generate(...)`.

## Errors

All public failures derive from `PyAmpliColError`:

```python
from pyamplicol import EvaluationError, GenerationError, PyAmpliColError

try:
    runtime = Runtime.load("artifacts/pp_zjj", process="missing")
except EvaluationError as error:
    print(f"could not load runtime: {error}")
```

Useful subclasses include configuration, model, generation, artifact,
compatibility, dependency, and evaluation errors. Catch the narrowest type you
can handle; catch `PyAmpliColError` at an application boundary.

## API design notes

- Public result and metadata objects are immutable dataclasses or typed facades.
- Artifact paths are normalized to absolute paths at service boundaries.
- `Generator.generate(..., mode="error")` refuses an existing destination.
- `Runtime` is bound to one selected process; load another instance for another
  process in the same multiprocess artifact.
- Process artifacts are trusted executable inputs. Normal loading validates the
  schema, path confinement, references, target compatibility, and runtime ABI.
- Use explicit payload validation only for an intentional whole-artifact
  corruption audit; it is not required before every load.

## Further reading

- [Configuration](configuration.md)
- [Generation Modes and Evaluators](generation-modes-and-evaluators.md)
- [Runtime and Selectors](runtime-and-selectors.md)
- [Born Correlations](../correlators.md)
- [Correlation Conventions](../correlator-conventions.md)
- [Artifacts and Portability](artifacts-and-portability.md)
- [Native APIs](native-apis.md)

<!-- Page hierarchy for the MadGraph7 (Sphinx) documentation. -->
```{toctree}
:hidden:

runtime-and-selectors
native-apis
artifacts-and-portability
umami-api
../correlators
../correlator-conventions
```
