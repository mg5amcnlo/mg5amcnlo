---
title: "Architecture Decisions"
nav_order: 3
parent: "Development Documentation"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Architecture Decisions

## Standalone Repository

`pyamplicol` is an independent repository and distribution. Build, runtime,
tests, examples, and release tooling may use only files contained in the source
checkout, source distribution, installed wheel, or explicitly downloaded
dependencies. No installed path or generated artifact may depend on another
checkout.

The Python distribution is named `pyamplicol`. Rusticol is the runtime and
native ABI name, not a second Python package.

## Completion States

**Candidate-complete** means source, API, physics, native SDK, wheel, source
distribution, and isolated-install tests pass against pinned contributor
inputs. Candidate package files are marked non-publishable.

**Release-complete** additionally requires exact compatible published
dependencies and the full supported-platform deployment matrix. Release mode
never substitutes candidate state.

## Model Boundary

Serialized external JSON is the primary user path. Trusted UFO and JSON inputs
compile into one canonical model IR before process planning. Model-owned
symbols have deterministic namespaces; executable compiled expressions may not
retain raw process-global UFO symbols.

The hand-coded built-in SM is a compatibility model contained under
`pyamplicol.models.builtin`. Its particle tables, aliases, synthetic fields,
and optimized kernels must not define generic external-model behavior.

Generic physics decisions use explicit model metadata: canonical species and
anti-relations, signed-PDG orientation, spin/statistics, color representation,
mass/width policy, propagator, source/crossing records, exact quantum numbers,
and normalized color/Lorentz tensors. Absolute SM PDG values and particle names
are interoperability labels, not generic taxonomy.

Unsupported model features fail in preflight with structured diagnostics.
Built-in optimizations are permitted only behind a proven applicability record
and a model-independent fallback.

## Processes And Artifacts

One process request may expand into many concrete processes. Stable process
names are artifact selectors and remain distinct from the output directory.
Generic `p`/`j` behavior is model/config data. Every model-derived catalog also
defines `all` from its propagating physical external states, excluding ghosts,
Goldstones, and auxiliary/non-propagating records. The complete legacy alias
table belongs only to the built-in compatibility model.

Schema-v3 artifacts are transactionally written executable inputs. Their
manifest verifies payload paths, digests, producer/runtime ABI, target, and
required capabilities. These checks detect accidental mutation; trust in the
artifact producer remains an external user decision.

The optimized total and resolved evaluation are two views of one runtime plan.
Resolved LC output retains physical color flows; NLC/full currently expose one
contracted SU(3) color component per helicity.

## Runtime And APIs

Rusticol core is Python-independent. Boundary crates expose one PyO3 extension
and C ABI v1. The wheel owns a header-only C++17 wrapper, Fortran 2008 module
source, static C ABI archive, and target link metadata.

Generation emits one root API bundle with Python, dependency-free Rust 2021,
C++17, and Fortran 2008 drivers. All drivers support concrete-process
selection, JSON and direct model-parameter updates, resolved evaluation, and
explicit total comparison. Python supports precision-controlled evaluation;
native APIs expose f64 only.

Runtime parameter batches are atomic. Mutable handles are not concurrently
reentrant; independent handles may run concurrently.

## Symbolica Boundary

Symbolica is imported lazily for model compilation, generation, and Python
precision paths that require it. Effective license state comes from
`symbolica.is_licensed()`.

The default JIT artifact embeds a direct SymJIT f64 application. Rusticol loads
that capability without importing Symbolica or requiring a Symbolica license.
ASM/C++ artifacts and Python precision paths advertise their retained Symbolica
runtime requirement explicitly.

## Build And Release

The PEP 517 backend and pinned Maturin build the Python extension and static SDK
from one locked Rust workspace. `dependencies/release-lock.toml` describes exact
published release inputs; contributor state is source-checkout-only and cannot
enter release package files.

Release validation uses standard package metadata, wheel `RECORD`, model and
artifact payload digests where the formats require them, `twine check`, platform
audits, clean installation, cross-language runtime tests, and PyPI Trusted
Publishing. Publication consumes already validated wheels and one source
distribution and performs no build.

## Legacy Physics Parity Reference

The legacy Fortran reference is developer-only. It is prepared as an isolated,
pinned checkout by developer tooling, is never imported by the installed
package, and is excluded from release package files. It provides a separately
implemented legacy parity authority and performance baseline. Because it shares
validation lineage and known defects with the current implementation, agreement
does not establish independent physics correctness. Future full-colour,
arbitrary-precision comparisons with MadGraph are required; agreement among
APIs that share Rusticol validates the ABI only.

## Supported Platforms

Binary releases target macOS arm64, macOS x86_64, and manylinux x86_64 with
`cp311-abi3`. Source installation may work elsewhere but is unsupported until
the complete native and physics matrix passes there.
