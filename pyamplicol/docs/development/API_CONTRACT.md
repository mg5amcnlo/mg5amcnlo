---
title: "Public API Contract"
nav_order: 1
parent: "Development Documentation"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Public API Contract

This document is normative for Python API version 1. Implementations may add
private helpers but may not widen the package-root export list without an API
review.

## Package-Root Exports

`pyamplicol` exports:

- Model and process types: `ModelSource`, `CompiledModel`, `CompiledModelInfo`,
  `CompiledModelSource`, `CompiledModelCapabilities`,
  `ModelCompilationIssue`, `ModelCompilationPhase`, `ProcessRequest`, and
  `ProcessSet`.
- Configuration: `RunConfig`, `GenerationConfig`, `EvaluationConfig`, and
  `BenchmarkConfig`.
- Correlation declarations: `CorrelatorConfig`, `ColorCorrelator`, `EmitGluon`,
  and `SplitGluon`.
- Services: `Generator`, `Runtime`, and `BenchmarkRunner`.
- Results: `GenerationPlan`, `GenerationResult`, `BenchmarkResult`,
  `BenchmarkStatistics`, `BenchmarkComponentTiming`, `BenchmarkStageTiming`,
  `BenchmarkTimingBreakdown`, `BenchmarkProfileCounters`, `ProcessPhysics`, `ExternalParticle`,
  `HelicityConfiguration`, `ColorComponent`, `ColorFlow`,
  `ContractedColorComponent`, `PhysicsReduction`, `ReductionGroup`,
  `ModelParameter`, `ResolvedEvaluation`, and `CorrelatedValue`.
- Process metadata: `ProcessAlias`.
- Helpers: `generate`, `load`, and `benchmark`.
- Package metadata: `__version__`.
- Errors: `PyAmpliColError`, `ConfigurationError`, `ModelError`,
  `GenerationError`, `ArtifactError`, `CompatibilityError`,
  `EvaluationError`, and `DependencyError`.

No DAG, evaluator, stage, color-sector, Symbolica, or legacy-reference class is
exported from the package root.

## Model And Process Requests

`ModelSource.built_in_sm()` identifies the built-in Standard Model.
`ModelSource.from_path(path, restriction=None, simplify=True)` accepts a UFO
directory, serialized JSON, or compiled-model file and records a resolved
source kind without importing Symbolica.

`ModelSource.compile(*, cache_dir=None, use_cache=True,
require_supported=True, prepared_output=None, evaluator=None) -> CompiledModel`
is the explicit public model compilation operation. `prepared_output` may also
write a reusable prepared-model bundle using the supplied typed evaluator
configuration. `CompiledModel` is an opaque immutable handle accepted by
generation; compiler-owned expression and tensor IR remain private.
`CompiledModel.info` exposes stable typed source, capability, parameter,
diagnostic, and phase-timing records. Convenience properties mirror those
records, while `write(path)` serializes the complete private payload and
`write_parameter_card(path)` writes mutable external-parameter defaults.

`ProcessRequest.parse(expression, *, name=None)` validates one process string.
`ProcessSet` contains a non-empty tuple of uniquely named requests and
explicit aliases. At runtime, stored aliases and uniquely inferred
side-preserving permutations use the same Rusticol mapping; incoming and
outgoing legs never cross the process boundary. Public physics, helicity,
color-flow, reduction, and momentum ordering follow the requested expression,
while internal representatives remain hidden.

## Generation

`Generator(config=None, progress=None)` accepts immutable typed configuration
or a `ConfigResolution`, plus an optional typed progress sink. Passing the
resolution preserves requested and effective TOML plus every clamp reason in
the generated artifact.

`Generator.plan(processes, *, model=None) -> GenerationPlan` is strictly
non-writing: it creates no model-cache or output files, directories, locks, or
temporary trees. It expands concrete processes and validates structural color
coverage without compiling a model, process DAG, evaluator, or artifact. A
built-in model, `CompiledModel`, or compiled-model file can always be planned.
For a UFO/JSON `ModelSource`, planning may read a valid existing model-cache
entry but never populates one; on a cache miss callers must first invoke
`ModelSource.compile()` and pass the returned `CompiledModel`.

Planning performs the same effective Symbolica license and resource resolution
as generation. Restricted mode clamps generation workers and Symbolica cores
to one. Licensed mode partitions the affinity-aware CPU budget after concrete
process expansion. `GenerationPlan.requested_settings` and
`GenerationPlan.effective_settings` retain both typed configurations, while
`GenerationPlan.adjustments` contains typed `ConfigClamp` values with the
requested value, effective value, path, and reason. `generate --dry-run` uses
this operation and has identical non-writing behavior.

`Generator.generate(processes, output, *, model=None, mode="error", correlators=None)
-> GenerationResult` writes a transactional schema-v3 artifact. `mode` is
`error`, `append`, or `replace`. Both `model` arguments accept a `ModelSource`,
the canonical `CompiledModel`, or `None` for the configured/default source.

`generate(processes, output, *, model=None, mode="error", config=None,
progress=None, correlators=None)` is a convenience wrapper with the same
generation semantics.

### Correlation declarations

`correlators=None` leaves ordinary generation unchanged. Passing
`CorrelatorConfig(color_correlations=(), spin_correlations=(),
all_color_through_order=None)` opts into tree-level correlated Born generation.
This declaration is separate from `RunConfig` and is not an argument to
`Generator.plan()`.

`ColorCorrelator(id, bra=(), ket=())` names an ordered pair of colour
connections; `ColorCorrelator.dipole(id, left_leg, right_leg)` requests
`<M|T_left . T_right|M>`. Connection steps are `EmitGluon(emitter_label,
emitted_label)` or `SplitGluon(parent_label, quark_label, antiquark_label)`.
There is no fixed step-count cap; both sides have the same step count, and
their final labelled colour representations must agree. Born legs use public
one-based labels; emitted auxiliary labels must be fresh. Colour IDs must be
unique; `"born"` is reserved and always supplies the ordinary overlap.
`spin_correlations` declares the exact nonempty sets of four-component vector
legs that may be replaced simultaneously. `CorrelatorConfig.to_json_dict()`
and `from_json_dict()` implement the CLI declaration-file schema.

`CorrelatorConfig.all_color(*, through_order, spin_correlations=())` accepts
a positive integer order `k` and resolves a complete, nonminimal tree-soft
catalogue at orders 1 through `k` separately for each process. Every active
coloured leg can emit; only auxiliary gluons split automatically. All compatible
directed bra/ket pairs are included, without numerical multiplicity or flavour
weights. Catalogue size grows rapidly. IDs of the form `N{k}LO/c{i}/c{j}` are
reserved when automatic generation is enabled; explicit requests may use other
IDs alongside `all_color_through_order`.

The correlation path applies the requested LC/NLC/full colour accuracy to
the inserted matrices while generating complete full-colour amplitudes.
It records `direct` contraction, `compiled` execution, and disabled numerical
current reuse as configuration adjustments where needed. It preserves arbitrary vector-source dependence
without helicity-specific parity/zero reductions or replay. Partial
colour/helicity generation, nonidentity process permutations, and append mode
are unsupported. A previously generated ordinary artifact cannot acquire this
capability merely by setting spin vectors. See
[Born Correlations](../correlators.md) for conventions and examples.

## Runtime

`Runtime.load(artifact, *, process=None, model_parameters=None,
mute_warnings=False) -> Runtime` validates the schema and identity-contract
marker, confined references, target, and ABI before loading executable state.
Call `pyamplicol.artifacts.validate_payloads()` for an explicit full checksum
audit.

`Runtime.physics` returns `ProcessPhysics` with stable particles, physical
helicity IDs, physical color-flow IDs, contraction metadata, coverage, and
selector capabilities. LC advertises `("helicity", "color_flow")`; NLC/full
advertise only `("helicity",)`. Their singleton contracted-color output axis is
metadata, not a selectable color flow. For an on-the-fly runtime, requesting
`physics` explicitly materializes the complete compact helicity and color-flow
axes and retains that compatibility view until the runtime is released;
`Runtime.clear()` does not discard it.

`Runtime.artifact_id` returns the lowercase 64-hex manifest identity
authenticated by the native loader. `Runtime.execution_mode` returns exactly
`"compiled"`, `"eager"`, `"recurrence"`, or `"on-the-fly"` from authenticated
runtime metadata. Both fail with `EvaluationError` when an injected backend
cannot supply a valid identity.

`Runtime.inspect() -> Mapping[str, object]` is the compact, observational
inspection path implemented by the built-in backend. It returns the
authenticated native metadata without opening `Runtime.physics`, together with
the live on-the-fly retained-state census when applicable. The OTF metadata
includes requested and host-effective query-construction thread counts, and its
inspection advertises `supported_precisions == (16,)`. Inspection is an optional
facade capability rather than a structural requirement of `RuntimeBackend`:
keeping it and the identity properties outside the protocol's original minimum
surface preserves compatibility with third-party backends.
The OTF state reports `family_cache_policy == "last-family-only"` and
`family_cache_limit == 1`; its semantic executor binding count is the exact
current-family binding cardinality rather than an all-seen cumulative count.

`Runtime.evaluate(momenta, *, helicities=None, color_flows=None,
helicity_by_point=None, color_flow_by_point=None, precision=16)` returns one
fully summed value per point. Global and per-point selectors are mutually
exclusive within each selector dimension. Momenta have shape
`(point, particle, [E, px, py, pz])`. On-the-fly execution currently supports
only `precision=16` (native f64); every other precision fails before selector
resolution or dense metadata access.

`Runtime.warm_up(momenta, *, precision=16, helicities=None,
color_flows=None, progress=None, n_cores=None) -> WarmUpResult` is available only for OTF
runtimes. `momenta` must contain exactly one binary64 phase-space point. The
operation transactionally constructs and retains the selected family, performs
its first evaluation, and reports elapsed time, total/new query counts, cache
reuse, and optional current/peak RSS. The optional `ProgressSink` observes
process preparation, query-family construction, family finalization, and first
evaluation. LC accepts helicity/flow selectors; contracted NLC/full accepts
helicity selectors only.
`n_cores` is a positive integer overriding the process's construction-worker
setting for this call; `None` retains that setting. Booleans, nonintegers, and
nonpositive counts are rejected. Query traces are constructed with bounded
parallelism, then merged serially. The override is not persisted and does not
change cache identity, family reuse, or later evaluation threading.

`Runtime.save(path) -> None` and `Runtime.load_cache(path) -> None` are
optional facade capabilities implemented by the native OTF backend. They
write and restore the currently retained completed structural cache, including
its selection and reduction mappings. Restoration requires a compatible
loaded process, rebinds its prepared kernels, and preserves its current model
parameters. Invalid snapshots leave the existing cache unchanged. Snapshots
contain neither evaluated currents nor native addresses, and do not resume
an unfinished construction. Ordinary non-OTF evaluation is unchanged.
The Rust core and C/C++/Fortran/Rust SDKs expose the same operations.

`Runtime.evaluate_resolved(momenta, *, helicities=None, color_flows=None,
precision=16) -> ResolvedEvaluation` returns LC values with shape `(point,
physical_helicity, physical_color_flow)` and NLC/full values with shape
`(point, physical_helicity, 1)`. `ResolvedEvaluation.total()` must reproduce
`evaluate()`.

`Runtime.set_spin_correlation_vectors(vectors)` sets the numerical vectors for
`evaluate_correlated()` and inherited requests in `evaluate_correlated_many()`.
`vectors` maps public one-based leg labels to
literal real or complex contravariant `(v0, vx, vy, vz)` vectors, or one such
vector per evaluated point. Its nonempty key set must exactly match a declared
spin class. Inputs are copied; invalid updates leave the old state intact.
Vectors are neither normalized nor projected nor conjugated on input. `None`
or `{}` restores ordinary helicity sources. These settings never change
`evaluate()` or `evaluate_resolved()`.

`Runtime.available_color_correlations() -> tuple[ColorCorrelator, ...]` lists
the selected process's resolved operators, including `"born"`. Each record
exposes `id`, `order`, `bra`, and `ket`. Listing reads and validates the stored
catalogue without initializing the amplitude executor; original leg ordering
and a correlation-enabled artifact are required.

`Runtime.evaluate_correlated(momenta, *, color_correlation="born",
helicities=None, precision=16, arithmetic="arbitrary") -> tuple[CorrelatedValue, ...]` selects a declared
colour ID and returns one value per point using the current model parameters
and spin vectors. Global helicity IDs or `HelicityConfiguration` records may
be selected; LC-flow and per-point selectors are not arguments to this method.
`CorrelatedValue.real` and `.imag` are `Decimal` values at the requested decimal
precision; `complex(value)` converts to binary64. The full directed
bra-adjoint/ket contraction may be complex. Each replaced spin leg counts
once, spectator helicities are summed incoherently, and ordinary initial-state
averages and identical-particle factors remain in place.

`Runtime.evaluate_correlated_many(momenta, requests, *, helicities=None,
precision=16, arithmetic="arbitrary") -> dict[str, tuple[CorrelatedValue, ...]]` accepts an ordered
mapping of user result labels to `CorrelatedRequest` records. Each record has
`color_correlation="born"` and `spin_vectors=None` defaults. `None` inherits a
snapshot of the setter; `{}` explicitly selects physical helicities. Request
vectors are copied and validated without changing setter state. Request labels
are distinct from generated colour IDs, allowing several spin projections of
one colour operator. The output preserves label order and input point order;
`values[label][point_index].real` and `.imag` select a single component.
Every series has `len(momenta)` entries. Spectator helicities are summed or
selected globally, not an additional output axis.

Both methods accept `arithmetic="double-double"` for genuine DoubleFloat
arithmetic with at most 31 output digits (`precision=31`). Their default
`"arbitrary"` retains arbitrary precision. Spin-vector real components may be
`Decimal`; complex components may be `(Decimal(real), Decimal(imaginary))`
pairs. Setter and request validation preserve their digits until conversion
to the selected arithmetic, and results always use plain `Decimal` fields.

Identical spin assignments share coherent amplitudes across colour IDs;
unchanged complete stage inputs can share stage outputs across assignments.
Numerical sharing is exact, signed-zero preserving, call-local and streamed
point by point. There is no persistent numerical cache or full spin-response
tensor. The underlying ordinary `evaluate()` remains full-colour; the
correlated `"born"` request uses the chosen correlated accuracy.

Python correlated evaluation uses the Symbolica-backed exact executor even
at `precision=16`. The C/C++/Fortran/Rust SDKs provide a separate binary64
RustiCol executor, catalogue enumeration, spin defaults and grouped requests.
Both paths require an explicitly correlated artifact and its original leg
ordering; both use the generation-time LC/NLC/full inserted colour matrices.
Native results are complex and request-major, with point-major Fortran array
access `values(point,request)`. Native APIs do not expose Python's arbitrary
precision or double-double arithmetic. Recurrence, eager, on-the-fly and FFT
correlated evaluation remain unsupported. See [Native APIs](../user/native-apis.md#correlated-born-evaluations)
for the typed SDK request and spin-vector conventions.

`Runtime.set_model_parameters(mapping)` validates the complete update before
committing it. `set_model_parameter(name, value)` is a convenience wrapper.
`mute_warnings()` and `unmute_warnings()` modify only that runtime handle.
`Runtime.clear()` discards warmed OTF process/family/scratch state while
retaining the loaded artifact, current model parameters, and any already
materialized `physics` compatibility view. It is a no-op for recurrence,
compiled, and eager ordinary execution state. If a correlated evaluator has
been initialized, `clear()` additionally releases its loaded stage/amplitude
evaluators while retaining declarations, colour matrices, and current spin
vectors. The next correlated call reloads that state; use
`set_spin_correlation_vectors(None)` to reset the vectors themselves.

`load(...)` is an alias for `Runtime.load(...)`.

## Benchmarking

`BenchmarkRunner(config=None, progress=None).run(target, *, points=None)
-> BenchmarkResult` accepts a runtime or artifact path. Results contain
requested/effective configuration, sample count, wall time per point, pure
evaluator time where available, uncertainty statistics, and environment
provenance. When native profiling is available,
`BenchmarkResult.timing_breakdown` contains typed top-level and internal timing
attribution, per-stage timings, and `BenchmarkProfileCounters`. Internal
leaf/backend/output-gather/remap attribution is non-additive: full-stage
evaluator envelopes own leaf gathering, while composed selected-chunk
input-pack envelopes own it. Movement and materialization counters are
normalized per profiled point; backend-call and explicit allocation counters
are normalized per runtime call. Compiled Direct-Arena engine/call counts and
its input/current-output/amplitude-output boundary-byte counters are likewise
normalized per runtime call. A valid fused Arena profile reports zero boundary
traffic and enough Arena calls to cover every evaluator backend call.

For on-the-fly execution, benchmarking snapshots the authenticated compact
runtime-state census immediately before and after its separately timed first
requested-workload evaluation, before configured warm-up runs. The environment
records both immutable snapshots, whether the initial runtime was cold or
already retained, and evidence that the first evaluation left a complete
retained family. A missing, mismatched, partial, or malformed census fails
closed. Other execution modes do not require this private census capability.

`benchmark(...)` is the convenience wrapper.

## Stability Rules

All public dataclasses are frozen. Public collections are tuples or immutable
mappings. Paths are accepted as `os.PathLike[str]` and returned as absolute
`Path` objects only when they describe user-created files.

Machine-readable methods return typed objects or documented JSON, never
volatile dictionaries. JSON output writes only to stdout; diagnostics use
logging/stderr.

## Typing Gate

`just typing` is the normative public Python typing gate and is also a
dependency of `just check`. It runs strict mypy over an explicit source list:
the package root, `pyamplicol.api`, the public configuration models/registry
and `pyamplicol.config` facade, `pyamplicol.reporting`, and
`pyamplicol.runtime`. Imported implementation modules are followed for type
information with their diagnostics silenced; no public target uses
`ignore_errors`.

A second mypy pass stages a temporary installed-style package, including
`py.typed` and the maintained `_rusticol.pyi`, then checks external consumers.
Those consumers assert exact Generator, Runtime, configuration, protocol, and
result types. Every name in the `__all__` lists of `pyamplicol` and its public
`api`, `config`, `reporting`, and `runtime` facades is also rejected if it
degrades to `Any`. The native consumer checks the maintained extension stub,
and the metadata check compares that stub with the Rust binding export list
without rebuilding Rust.

This is intentionally not a project-wide strictness claim. Dynamic generation,
model compilation, evaluator, color, artifact, CLI, and other private
implementation modules are outside the source target. Bundled vendor UFO
assets under `src/pyamplicol/assets/models/ufo/` and generated standalone API
templates under `src/pyamplicol/assets/api_templates/` are explicitly
excluded. Their exclusion does not weaken checks of the documented public
signatures that expose their results.
