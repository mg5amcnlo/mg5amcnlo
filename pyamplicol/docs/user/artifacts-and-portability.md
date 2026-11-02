---
title: "Artifacts and Portability"
nav_order: 3
parent: "Python API"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Artifacts and Portability

pyAmpliCol separates model inputs, prepared model bundles, and generated
process artifacts. Understanding that boundary makes deployment predictable:
you can move the right object, know when native code is rebuilt, and avoid
mistaking an internal digest for publisher authentication.

## Three related but different objects

| Object | Typical suffix/path | Purpose |
| --- | --- | --- |
| Compiled model IR | `.pyAmplicol-model.json` | Architecture-independent normalized model, expressions, particles, parameters, and capabilities. |
| Prepared model bundle | `.pyamplicol-model` | Compiled model plus one prepared local-kernel backend for recurrence/eager/OTF generation. |
| Process artifact | directory containing `artifact.json` | Standalone selected processes, physics metadata, evaluators/schedules, model-parameter state, and optional API bundle. |

A raw JSON or UFO model can feed process-local **compiled** generation. The
default **recurrence**, **eager**, and **on-the-fly** modes require a compatible
prepared model bundle; the installed `built-in-sm` and `built-in-sm-heft`
models are exceptions because wheels already ship their portable JIT O2 packs.

See [Models and Processes](models-and-processes.md) and [Generation Modes and Evaluators](generation-modes-and-evaluators.md) for producing each
object.

An optional **OTF cache file**, written by `runtime.save(path)`, is different
from all three: it retains completed current-construction work for the loaded
process. Keep the original process directory, load it normally, then call
`runtime.load_cache(path)`. The cache contains no evaluated currents or model
parameter values and cannot be loaded as a standalone process output. Its
compatibility follows the matching process output and cache format, rather
than providing a separate portability guarantee. See the
[small save/restore example](runtime-and-selectors.md#saving-an-otf-warm-cache).

## Process-artifact layout

A generated root resembles:

```text
artifacts/pp_zjj/
  artifact.json
  process-groups.json
  evaluators.pacbin
  model/
  processes/
    p_p_to_z_j_j_1/
    p_p_to_z_j_j_2/
    ...
  API/
    python/
    c/
    cpp/
    fortran/
    rust/
```

The exact payload set depends on execution mode, backend, color accuracy,
coverage, and whether the API bundle is enabled. Do not infer the contract from
filenames; `artifact.json` declares all runtime-bearing payloads and their
roles.

Inspect without loading executable evaluator state:

```console
pyamplicol inspect artifacts/pp_zjj
pyamplicol inspect artifacts/pp_zjj --json
```

The inventory reports process IDs and aliases, target, execution mode,
capabilities, selector coverage, physical and logical evaluator counts,
payload sizes, and dependency metadata.

## Artifact identity

`runtime.artifact_id` is the lowercase SHA-256 identity of runtime-bearing
payload records under the current schema-v3 identity contract. It deliberately
excludes provenance that cannot change runtime behavior, such as requested and
effective configuration records, validation momenta, timing, the
`process-groups.json` map of subprocesses with equal matrix elements, and
report-only metadata.

```python
from pyamplicol import Runtime

runtime = Runtime.load("artifacts/pp_zjj", process="d d~ > g z g")
print(runtime.artifact_id)
```

This content label answers “is this the same declared runtime payload?” It does
**not** answer “who published this artifact?”

Artifacts must explicitly carry the current identity-contract marker and
runtime ABI. Old internal process-artifact formats are not automatically
migrated; regenerate them with the current pyAmpliCol version.

## Fast loading versus an explicit checksum audit

Normal loading validates the inexpensive authoritative boundary:

- JSON/schema and identity-contract shape;
- confined relative paths and references;
- process/runtime capability coherence;
- target and CPU compatibility;
- native runtime ABI.

It intentionally does not rehash every payload. Rehashing a multi-gigabyte
PACBIN file before every evaluation would duplicate work and can dominate
startup without changing the artifact.

When a full on-disk corruption audit is specifically wanted, request it once:

```python
from pyamplicol.artifacts import load_manifest, validate_payloads

manifest = load_manifest("artifacts/pp_zjj")
validate_payloads(manifest)
```

That explicit audit recomputes the manifest identity, sizes and SHA-256 hashes
of declared payloads, executable bits, symlink/path constraints, and target
compatibility.

Artifact writing always validates the structure and declared payload records
it creates. Optional post-build runtime validation is a different operation: it
reopens the completed artifact and compares optimized and resolved f64
evaluation. It is off by default because this second runtime pass does not
modify the artifact and can become disproportionately expensive for large
resolved axes. Enable it only when an immediate runtime smoke is valuable:

```console
pyamplicol generate --card run.toml --post-build-validation
```

The configured `generation.validation.samples` controls that optional reopened
runtime smoke. It is independent of high-precision numerical-current relation
certification performed during generation.

## Portability matrix

| Artifact/backend | Target metadata | Can move between supported x86-64 and arm64 hosts? |
| --- | --- | --- |
| Compiled all-JIT O1 or O2 process artifact | `portable-64le`, empty CPU-feature set | **Yes**, on supported 64-bit little-endian hosts. |
| Eager/recurrence process artifact using a prepared JIT O2 pack | `portable-64le`, empty CPU-feature set | **Yes**, on supported 64-bit little-endian hosts. |
| Prepared JIT O2 model bundle | portable SymJIT storage-v3 | **Yes**, between supported x86-64 and arm64 hosts. |
| Compiled JIT O0 or O3 process artifact | concrete target, empty CPU-feature set | No; regenerate on the destination. |
| C++ or ASM process artifact | concrete target; CPU features when native specialization is enabled | No; requires exact compatible target/features. |
| C++ or ASM prepared bundle | target-native | No. |
| Compiled model IR (`.json`) | architecture-independent | Yes; it contains no executable prepared backend. |

For a portable compiled all-JIT artifact, Rusticol verifies that every
executable leaf and nested selector path is O1 or O2 SymJIT. Eager and
recurrence artifacts separately require a portable, exact-O2 prepared pack.
In every case, a target-specific capability prevents the portable marker.
Rusticol rebuilds executable code for the receiving CPU when the artifact is
loaded.

`portable-64le` is an executable portability claim, not a claim that benchmark
results transfer between hosts. Profiling measurements and campaign currents
remain host/profile-specific.

## Why some JIT artifacts are still target-specific

JIT O0/O3 may specialize code and metadata for the producing host. C++ and ASM
artifacts contain target-native libraries. Their manifest records a concrete
triple and required CPU features, and loading fails before executable state is
opened if the destination is incompatible.

For compiled mode, O1 and the default O2 are both portable choices. Prepared
eager/recurrence packs use the exact-O2 portability contract. To maximize a
specific machine's process-local performance with explicit O3, plan to
generate and deploy on a matching target.

## PACBIN evaluator containers

JIT evaluator state is stored as indexed logical members in a single
`evaluators.pacbin` container. Inspection distinguishes:

- physical file count and on-disk size;
- indexed evaluator member count;
- logical unpacked evaluator size.

It does not extract one filesystem object per evaluator.

On Unix, runtime loading copies a PACBIN-backed artifact into anonymous memory,
removes write access from that mapping, and parses it there. Replacing,
truncating, or mutating the path afterward cannot change the running evaluator.
Plan for virtual memory and RAM or swap approximately equal to the loaded
PACBIN size for the handle's lifetime; no temporary filesystem space is
required.

Target-native C++/ASM evaluator libraries use an analogous executable snapshot
root controlled by `PYAMPLICOL_NATIVE_SNAPSHOT_ROOT`. Its filesystem must
permit executable mappings.

## Prepared models and standalone process artifacts

A prepared bundle is a **generation input**, not a runtime dependency of the
resulting process artifact. Eager, recurrence, and OTF generation copy only the
needed kernels plus their compact tables, schedules, or process seed into the
process artifact. A machine that evaluates that artifact does not need the
original `.pyamplicol-model` bundle.

For example:

```console
pyamplicol model compile models/json/sm/sm.json \
  models/ufo-sm-jit-o2.pyamplicol-model \
  --backend jit --jit-optimization-level 2

pyamplicol generate 'd d~ > z g g g' artifacts/ddbar_z3g_eager \
  --model models/ufo-sm-jit-o2.pyamplicol-model \
  --execution-mode eager
```

The second artifact is self-contained for supported runtime APIs.

## Transactional writing and replacement

Generation writes artifacts transactionally. The output mode is explicit:

| Mode | Behavior |
| --- | --- |
| `error` | Refuse an existing output (default). |
| `append` | Extend only under the supported compatible multiprocess contract. |
| `replace` | Build a replacement and atomically install it. |

Use a separate output for materially different configuration during
experimentation. Avoid manually editing a sealed artifact: its declared size,
digest, and semantic links will no longer match.

The one supported in-place update is `pyamplicol group-processes`, which
rewrites `process-groups.json` (payload role `process-groups`) and its record
in `artifact.json` under the same lock that serializes generation. The artifact
ID does not change. An append keeps the stored groups and adds the new
subprocesses.

## Copying and archiving

Copy the **entire** artifact directory. Preserve ordinary file contents and
executable bits; do not copy only `artifact.json` and selected payloads.

```console
rsync -a artifacts/pp_zjj/ remote:/srv/processes/pp_zjj/
```

After transfer, either load it normally (fast schema/target boundary) or run
one explicit `validate_payloads()` audit when the transport channel warrants
it. Do not add a full checksum pass to every runtime invocation.

Process artifacts are trusted executable inputs. Direct SymJIT applications
are lowered to native code, and C++/ASM artifacts contain native libraries.
Generate them yourself or receive them through a channel you trust.

## Common compatibility failures

| Message or symptom | Meaning | Next step |
| --- | --- | --- |
| Target triple mismatch | Target-native artifact moved to a different architecture/OS. | Regenerate there or use a portable JIT artifact: compiled O1/O2, or eager/recurrence with a prepared O2 pack. |
| Required CPU feature unavailable | Artifact was specialized beyond the destination CPU. | Regenerate with compatible settings. |
| Old identity contract / runtime ABI | Artifact predates the current internal format. | Regenerate; there is no compatibility shim. |
| Payload size/digest mismatch during explicit audit | Files changed or transfer was incomplete. | Restore from a trusted source or regenerate. |
| PACBIN load consumes substantial memory | Container is intentionally snapshotted into anonymous read-only memory. | Size deployment memory for the artifact or generate a smaller/specialized artifact. |

See [Troubleshooting](troubleshooting.md) for environment and API-driver failures.

## Related pages

- [Runtime and Selectors](runtime-and-selectors.md) — load, select, and evaluate a process.
- [Native APIs](native-apis.md) — consume the same artifact from five languages.
- [Models and Processes](models-and-processes.md) — compiled and prepared model inputs.
- [Release and Support](release-and-support.md) — validated wheel and source-distribution boundary.
- [Packaging contract](../development/PACKAGING_CONTRACT.md).
