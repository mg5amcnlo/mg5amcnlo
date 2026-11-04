---
title: "Troubleshooting"
nav_order: 4
parent: "Profiling and Benchmarking"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Troubleshooting

This page starts with cheap, authoritative checks and then narrows common
generation, runtime, and native-SDK failures. Avoid broad
rebuilds until the failing boundary is known.

## First checks

Confirm which Python and installation are active:

```console
command -v python
command -v pyamplicol
python -c 'import pyamplicol; print(pyamplicol.__version__); print(pyamplicol.__file__)'
```

Inspect the installed environment and compiler tools:

```console
pyamplicol doctor
```

Run the packaged direct-runtime self-test:

```console
pyamplicol self-test
```

On a healthy 0.2.0 macOS arm64 release-candidate installation, the colored
terminal tables report checks equivalent to this compact transcription:

```text
python          pass
model-assets    pass   67 verified files
rusticol-python pass   package 0.2.0, C ABI 1
native-sdk      pass   aarch64-apple-darwin; ABI 1; librusticol_capi.a
physics-f64     pass   d_dbar_to_z; shape (1, 12, 1); direct SymJIT
```

The Python version, target, paths, and package version naturally vary. A failure
in one named check points to a much smaller subsystem than a full rebuild.

## `ModuleNotFoundError: No module named 'pyamplicol'`

The generated Python driver uses the interpreter that launches it. Activate
the installed environment or invoke that environment's Python explicitly:

```console
. /path/to/venv/bin/activate
python artifacts/pp_zjj/API/python/check_standalone.py \
  --process 'd d~ > g z g' --json
```

or:

```console
/path/to/venv/bin/python \
  artifacts/pp_zjj/API/python/check_standalone.py \
  --process 'd d~ > g z g' --json
```

In the canonical source workflow, a copied workspace below a checkout can find
the checkout's `.venv` created by `just dev-install`. A workspace elsewhere
must use an active or explicit environment.

## `rusticol-config: command not found`

C, C++, Fortran, and Rust consumers discover their SDK through the installed
environment:

```console
. /path/to/venv/bin/activate
rusticol-config --json
```

An explicit override is also supported:

```console
export RUSTICOL_CONFIG=/path/to/venv/bin/rusticol-config
make -C artifacts/pp_zjj/API/c run \
  ARGS='--process "d d~ > g z g" --json'
```

Do not manually guess include, archive, framework, or system-library paths.
If `rusticol.h`, `rusticol.hpp`, the Fortran module source, or
`RUSTICOL_RUST_SOURCE` is missing, first check:

```console
rusticol-config --include-dir
rusticol-config --library
rusticol-config --fortran-source
rusticol-config --rust-source
```

See [Native APIs](native-apis.md) for the complete environment contract.

## `installed candidate wheel is stale for this checkout`

Contributor imports bind the staged native extension to relevant checkout
sources. If Rust/native inputs or the source revision changed, refresh the
candidate once:

```console
just dev-install
```

Do not copy an extension from another worktree or disable this check. Published
wheels do not use the checkout-candidate gate.

## An example card cannot find its artifact

Copied evaluation and benchmark cards use paths relative to the copied
workspace. Generate their prerequisite first and run from that workspace:

```console
pyamplicol examples copy /tmp/pyamplicol-examples --force
cd /tmp/pyamplicol-examples

pyamplicol generate_pp_zjj_from_ufo_sm.toml
pyamplicol evaluate_total.toml
pyamplicol evaluate_resolved.toml
pyamplicol benchmark.toml
```

Generating `qq_z6g_compiled_jit_o3.toml` creates a different artifact and does
not satisfy `benchmark.toml`'s `artifacts/pp_zjj` prerequisite.

List the packaged names and their actions before running one:

```console
pyamplicol examples list
pyamplicol examples run --help
```

See [Examples Gallery](examples-gallery.md).

## “no model-supported tree-level amplitudes”

pyAmpliCol generates tree-level amplitudes. The primary `p p > Z j j`
expansion encounters `g g > Z g g`, which is loop-induced in the Standard
Model. It is reported and omitted while the supported tree-level channels are
retained:

```text
Skipped 1 concrete subprocess with no model-supported tree-level amplitudes:
p_p_to_z_j_j_8 (g g > Z g g)
```

That warning is expected for the inclusive example. A request containing only
the loop-induced process has no supported representative and fails. Choose a
tree-level process or a model that provides the required tree-level amplitude;
pyAmpliCol is not silently performing a loop calculation.

## Recurrence, eager, or OTF says a prepared model is missing

Recurrence, eager, and OTF use prepared local kernels. A raw UFO directory,
serialized model JSON, or compiled model IR is not itself a prepared kernel
bundle.

Either compile a prepared bundle:

```console
pyamplicol model compile models/json/sm/sm.json \
  models/sm-jit-o2.pyamplicol-model \
  --backend jit --jit-optimization-level 2
```

and use it:

```console
pyamplicol generate 'd d~ > z g g' artifacts/builtin_ddbar_to_zgg \
  --model models/sm-jit-o2.pyamplicol-model \
  --execution-mode recurrence
```

or select process-local compiled mode explicitly for raw/compiled model input:

```console
pyamplicol generate --card run.toml --execution-mode compiled
```

The installed `built-in-sm` source automatically selects its packaged JIT O2
prepared bundle. There is no hidden recurrence-to-compiled fallback.

## A process expression does not resolve

Inspect the stored representatives:

```console
pyamplicol inspect artifacts/pp_zjj
```

The runtime permits a unique permutation within each side of `>`:

```console
pyamplicol inspect artifacts/pp_zjj --process 'd d~ > g z g'
```

It does not move an incoming particle to the outgoing side. If several
representatives have the same incoming/outgoing multisets, inference is
ambiguous and the error lists their stable IDs. Select one of those IDs
explicitly.

## Wrong kinematic shape or ordering

Runtime arrays have shape `(point, external particle, [E, px, py, pz])`.
Standalone `--kinematics` files contain exactly one `[external][4]` point or a
singleton batch around it. Legs follow the expression passed in `--process`,
including inferred permutations.

Use `runtime.physics.external_particles` or:

```console
pyamplicol inspect artifacts/pp_zjj --process 'd d~ > g z g'
```

to confirm public order. JSON booleans, non-finite values, multiple points, and
wrong ranks are rejected. Decimal strings are preserved only by Python exact
evaluation; native APIs convert to f64.

## Color-flow selection is rejected

Physical color-flow selection is an LC capability. NLC and full color are
contracted and expose one output color slot per helicity, not a selectable LC
flow.

```python
print(runtime.physics.selector_capabilities)
print(runtime.physics.color_flow_ids)
```

Use only helicity selectors for NLC/full. See [Runtime and Selectors](runtime-and-selectors.md).

## Target or CPU-feature incompatibility

Inspect the target:

```console
pyamplicol inspect artifacts/my_process
rusticol-config --target
```

Compiled all-JIT O1/O2 process artifacts report `portable-64le` and can move
between supported x86-64 and arm64 little-endian hosts. Prepared eager and
recurrence artifacts use the exact-O2 portable pack contract. JIT O0/O3, C++,
and ASM artifacts are target-specific and may require particular CPU features.
Regenerate a target-specific artifact on the destination, or use a portable
JIT configuration. See [Artifacts and Portability](artifacts-and-portability.md).

## Artifact loading or an explicit checksum audit fails

Treat a size/digest/path mismatch as a changed or incomplete artifact. Do not
edit `artifact.json` or payloads by hand. Restore the complete directory from a
trusted channel or regenerate it.

Normal runtime loading does not hash every payload. If you intentionally ran
the explicit full audit, remember it may read very large PACBIN files once:

```python
from pyamplicol.artifacts import load_manifest, validate_payloads

manifest = load_manifest("artifacts/my_process")
validate_payloads(manifest)
```

Do not put this full rehash in every evaluation loop.

## Generation appears to repeat expensive validation

Generation-time validation remains enabled and checks construction as it is
built. The separate **post-build** runtime reopen is off by default because it
does not change the completed artifact and can be expensive for large resolved
axes.

Check the resolved card:

```console
pyamplicol config resolve run.toml
```

`generation.validation.post_build_validation` should be `false` unless you
explicitly requested `--post-build-validation`. The sample count alone does not
enable the reopened pass.

## Symbolica license or restricted-mode messages

Run:

```console
pyamplicol doctor
```

Importing pyAmpliCol and using direct f64 runtime evaluation do not import
Symbolica. Model compilation and generation use Symbolica lazily. Without a
valid license, eligible users may continue under Symbolica's current restricted
terms, and pyAmpliCol clamps generation to one worker and one Symbolica core.

Python precision other than 16 also loads retained Symbolica state. Native
languages are f64-only and never perform a Symbolica license check. See
[Symbolica and Licensing](symbolica-and-licensing.md).

## Memory use while loading a large JIT artifact

PACBIN-backed runtimes snapshot the container into anonymous read-only memory.
The handle therefore needs RAM or swap comparable to the PACBIN size for its
lifetime. This protects a running evaluator from later path replacement or
truncation.

If that footprint is unsuitable, generate a smaller process set or a more
specialized artifact. Do not assume the container will be lazily read from a
mutable file.

## Reporting a reproducible issue

Include only the smallest evidence that identifies the failing boundary:

```console
python -c 'import pyamplicol; print(pyamplicol.__version__)'
pyamplicol doctor --json
pyamplicol inspect PATH_TO_ARTIFACT --json
```

Also include:

- the exact command and run card;
- host OS/architecture and Python version;
- execution mode, backend, color accuracy, and target;
- the complete error, not only its final status code;
- whether the artifact is reproducible from a public example.

Do **not** upload proprietary UFO code, license keys, or an untrusted native
artifact publicly. Open issues through [Release and Support](release-and-support.md).
