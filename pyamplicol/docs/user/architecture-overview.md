---
title: "Architecture Overview"
nav_order: 6
has_children: true
---
<!-- SPDX-License-Identifier: 0BSD -->

# Architecture Overview

pyAmpliCol turns a model plus one or more process requests into a standalone
schema-v3 executable artifact. Python owns configuration, model compilation,
process planning, and artifact generation. Rusticol owns artifact loading,
physics-aware process selection, optimized f64 execution, profiling, and the
native ABI.

This page describes stable public boundaries rather than internal module names.

```text
CLI or typed Python API
          │
          ▼
schema-v1 configuration → model IR and process expansion
          │
          ▼
recurrence / compiled / eager generation
          │
          ▼
schema-v3 process artifact → Rusticol runtime
                                  ├── Python
                                  ├── C11
                                  ├── C++17
                                  ├── Fortran 2008
                                  └── Rust 2021
```

## User surfaces

One typed configuration registry drives:

- schema-v1 TOML cards;
- dedicated CLI flags and `--set` overrides;
- frozen Python configuration dataclasses.

The public Python root exports model/process types, configuration, `Generator`,
`Runtime`, `BenchmarkRunner`, typed results, convenience helpers, and documented
errors. Generation DAGs, evaluator payloads, Symbolica objects, and artifact
writer internals remain private.

See [Command-Line Interface](command-line-interface.md), [Configuration](configuration.md),
and [Python API](python-api.md).

## Model boundary

The primary portable user input is serialized JSON model IR. Trusted UFO and
JSON sources compile into the same canonical representation before process
planning. A compiled model records particle/anti-particle relations, spin and
statistics, color representation, mass and width policy, propagators,
interactions, exact quantum numbers, and normalized tensors.

Two model products have different roles:

- a portable compiled-model JSON contains model IR and exact expressions;
- a `.pyamplicol-model` bundle adds one prepared local-kernel backend for
  recurrence or eager execution.

When higher-point interactions are split into smaller kernels, model compilation
also reduces their internal auxiliary currents. It examines every consuming
kernel and both input orientations: unused components are removed, while
components with identical or opposite coefficient columns are replaced by the
corresponding signed sum of producer components. These are exact symbolic
relations valid for all model parameters, not numerical approximations.
Physical currents and externally prescribed bases are unchanged. The reduced
kernels are stored in the compiled/prepared model, so this work is not repeated
at the first process evaluation.

Symbolic current weights, propagator phases and projector fractions are kept
exact during this construction. In particular, multiplying an exact coefficient
by a complex unit does not round it to binary64. This preserves cancellations
when the same process output is evaluated with double-double or arbitrary
precision.

The hand-written built-in Standard Model is a compatibility model. Its aliases
and optimized kernels do not define generic UFO/JSON behavior. Unsupported
model features fail during preflight with structured diagnostics rather than
falling back to built-in assumptions.

See [Models and Processes](models-and-processes.md).

## Process planning and representative reuse

A single multiparticle request may expand into many concrete processes.
Generation retains one evaluator representative per equivalent incoming and
outgoing permutation class when possible. Stable IDs select representatives;
readable process expressions select a requested public ordering.

Rusticol resolves a unique side-preserving permutation and remaps momenta,
particles, helicities, LC flows, reductions, selectors, and resolved metadata
centrally. No leg crosses `>`, and ambiguous matches fail with their candidate
stable IDs.

See [Process Selection and Permutations](process-selection-and-permutations.md).

## Four generation lanes

All lanes implement the same public process artifact and runtime result:

- **recurrence** writes current-recursion schedules over prepared local
  kernels;
- **compiled** writes process-local stage evaluators;
- **eager** writes compact DAG invocation tables over prepared local kernels;
- **on-the-fly** writes a compact process seed and builds the selected query
  family on first use.

Reusable contracted recurrence artifacts also carry one all-helicity
physical-colour plan and a compact per-helicity row-group dispatch. The
support metadata is used only during cold selector binding; warmed execution
runs the selected rows directly. On-the-fly does not carry that recurrence
companion: it builds and retains the requested family from its compact seed.

The evaluator backend is a separate dimension: JIT, C++, or assembly. The
default direct JIT stores a SymJIT application. Prepared recurrence/eager/OTF
JIT kernels use the portable O2 storage contract.

LC exposes physical color flows; NLC/full expose one contracted color output
per helicity. The optimized total and resolved tensor are two views of the same
runtime plan, and `ResolvedEvaluation.total()` must reproduce the optimized
total.

See [Generation Modes and Evaluators](generation-modes-and-evaluators.md).

## Schema-v3 artifact boundary

A generated artifact is a transactional, standalone executable input. It
contains:

- model and process metadata;
- physical helicity/color axes and reductions;
- execution-mode plans;
- evaluator or prepared-kernel payloads;
- configuration and generation records;
- optional generated API drivers;
- deterministic validation kinematics.

The manifest confines payload paths and binds the runtime ABI, target,
capabilities, references, sizes, and required content digests. Normal runtime
loading checks the schema, identity-contract marker, path confinement, target,
and ABI at the cheapest authoritative boundary. An explicit full payload hash
audit is available when wanted; it is not repeated on every normal load.

Artifacts are executable and must come from a trusted producer. Internal
consistency hashes do not establish publisher identity.

Generated artifact formats intentionally have no automatic compatibility shim.
An old identity contract or runtime ABI fails with regeneration guidance.

See [Artifacts and Portability](artifacts-and-portability.md).

## Rusticol runtime boundary

Rusticol core is Python-independent. It:

- loads and validates the selected artifact/process;
- applies public process permutations;
- owns the optimized f64 execution engines;
- resolves global and per-point selectors;
- exposes resolved physical metadata;
- applies atomic model-parameter updates;
- measures runtime and native attribution;
- implements C ABI version 1.

The PyO3 extension and C ABI are boundary layers over the same core. The wheel
also owns a header-only C++ wrapper, a Fortran module source, a dependency-free
safe Rust wrapper, the static C ABI archive, and target link metadata.

One runtime handle has mutable parameter and warning state and is not
concurrently reentrant. Independent handles may run concurrently.

## Native SDK and generated APIs

Binary wheels contain a target-specific static SDK discovered with
`rusticol-config`. Generation can emit an `API/` bundle with Python, C11,
C++17, Fortran 2008, and Rust 2021 standalone checks.

All five drivers support:

- process selection by stable ID or side-preserving expression;
- bundled or user-supplied kinematics;
- UFO-style JSON parameter cards and direct overrides;
- resolved evaluation and explicit comparison with the optimized total.

Python supports precision-controlled exact evaluation when retained expressions
permit it. Native APIs expose f64 only.

Opt-in [Born Correlations](../correlators.md) use a separate Python exact
executor over retained Symbolica states, with full-colour direct compiled
amplitudes. This includes requests at precision 16; the native ABI and generated
standalone drivers remain ordinary total/resolved evaluation interfaces.

See [Native APIs](native-apis.md).

## Symbolica and SymJIT boundary

Python package startup imports Symbolica and attempts to register the
pyAmpliCol library key. While valid, that package license covers model
compilation, generation, and Python higher-precision or correlated evaluation
without requiring a personal user key. Synchronous registration failure leaves
the personal-license and restricted-mode paths available. Generation queries
Symbolica's effective license state and records resource clamps.

The default JIT artifact embeds a direct SymJIT application. Rusticol loads and
executes that f64 state without importing Symbolica or applying its generation-
time license/resource policy. Compatible precompiled C++/ASM payloads are also
Symbolica-independent at f64. Generating any of those payloads still uses
Symbolica.

See [Symbolica and Licensing](symbolica-and-licensing.md).

## Portability model

Portability is authenticated from the complete executable content:

- compiled all-JIT O1/O2 artifacts may use the `portable-64le` target and
  rebuild native code on a supported receiving host;
- JIT O0/O3, C++, ASM, and explicitly native-architecture payloads remain
  target-specific;
- prepared JIT O2 bundles are portable across supported x86-64 and AArch64
  release hosts;
- a single target-specific execution leaf makes the whole artifact
  target-specific.

Loaders reject incompatible target or CPU-feature requirements before running
the evaluator.

## Build and release boundary

One locked Rust workspace builds both the Python extension and static SDK. The
source distribution carries the release build backend and exact release
dependency contract; contributor dependency state is checkout-only and cannot
be substituted into a release build.

Release publication consumes already validated wheels and one source
distribution. It does not rebuild during upload. The supported platform matrix
and publication boundary are recorded in
[Release and Support](release-and-support.md).

See [Release and Support](release-and-support.md).

## Independent physics reference

The optional original-AmpliCol Fortran comparison is developer/campaign
infrastructure, not an installed runtime dependency. It provides an independent
numerical reference. Agreement among ordinary Python and native f64 APIs
validates a shared ABI and wrapper contract, but those operations all call
Rusticol and are therefore not independent physics implementations.

## Design principles

- Model-generic behavior comes from explicit model metadata, not hard-coded SM
  particle numbers or names.
- Unsupported capabilities fail explicitly.
- Public inputs and outputs follow the user's selected particle ordering.
- Configuration, Python results, and public metadata are typed and immutable.
- Normal runtime checks each invariant once at its authoritative boundary.
- Performance instrumentation must not redefine the ordinary evaluation path.
- Artifacts are self-contained; installations do not depend on another source
  checkout.

## Deeper contracts

The repository maintains concise normative contracts for contributors:

- [Public API contract](../development/API_CONTRACT.md)
- [Configuration contract](../development/CONFIG_CONTRACT.md)
- [Architecture decisions](../development/ARCHITECTURE_DECISIONS.md)
- [Packaging contract](../development/PACKAGING_CONTRACT.md)
- [Physics extraction contract](../development/PHYSICS_EXTRACTION_CONTRACT.md)
- [On-the-fly mode architecture](../development/ON_THE_FLY_MODE_ARCHITECTURE.md)

<!-- Page hierarchy for the MadGraph7 (Sphinx) documentation. -->
```{toctree}
:hidden:

symbolica-and-licensing
```
