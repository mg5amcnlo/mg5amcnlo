---
title: "Release and Support"
nav_order: 5
parent: "Profiling and Benchmarking"
---
<!-- SPDX-License-Identifier: 0BSD -->

# Release and Support

pyAmpliCol 1.0.0 is the software release accompanying the official arXiv
publication, *pyAmpliCol: fast tree-level matrix elements*. Binary wheels and a
source distribution are available on [PyPI](https://pypi.org/project/pyamplicol/).
This page records the supported release boundary and explains how to report a
problem.

## Releases with MadGraph7

pyAmpliCol is now developed in the
[MadGraph7 repository](https://github.com/MadGraphTeam/MadGraph7/tree/main/pyamplicol)
and released with MadGraph7: every MadGraph7 release publishes pyAmpliCol to
PyPI with the same version number. These versions (0.2.x and later) sort below
the standalone 1.0.x releases described below, so a plain
`pip install pyamplicol` may still select 1.0.1; MadGraph7 always installs the
exact matching version, `pyamplicol==<MadGraph7 version>`. Report problems in
the [MadGraph7 issue tracker](https://github.com/MadGraphTeam/MadGraph7/issues).

## Current release boundary

Version `1.0.1` is represented by the immutable
[`v1.0.1` source snapshot](https://github.com/MadGraphTeam/pyamplicol/tree/v1.0.1),
the [GitHub release](https://github.com/MadGraphTeam/pyamplicol/releases/tag/v1.0.1),
and [PyPI release](https://pypi.org/project/pyamplicol/1.0.1/). Its validated
inventory is one source distribution and three `cp311-abi3` wheels:

- macOS 11 or newer on Apple silicon;
- macOS 11 or newer on x86-64;
- manylinux 2.28 on x86-64.

The [release workflow](https://github.com/MadGraphTeam/MadGraph7/actions/workflows/release.yml)
installs each wheel into a clean CPython 3.11 environment and exercises the
Python, C11, C++17, Fortran 2008, and Rust 2021 APIs. It also runs a CPython
3.14 abi3 smoke test, source preflight, and independent Fortran physics oracle.
Publication uploads these already validated files without rebuilding them.

Version 1.0.1 uses published Symbolica 3.0.0, SymJIT 2.26.4 and
ufo-model-loader 1.0.0, as recorded in the dependency lockfiles. No local
dependency source patches or unpublished wheels are needed.

## 1.0.1 release

Version `1.0.1` adds process grouping for multiprocess outputs. Subprocesses
with equal matrix elements are recorded in a new `process-groups.json`
artifact payload, with their leg permutation, rational factor, and charge
conjugation. `--flavor-blind` generates one representative per flavour
class, optionally including charge conjugation, and every other flavour
remains selectable from Python, the CLI, and the native SDKs, which route it
to its representative. `--analyze-process-groups numerical` and
`pyamplicol group-processes` find further relations numerically.
`pyamplicol inspect` and the generation summary display the groups, and UMAMI
exports flavour-blind members through their representative. UFO multiparticle
expansion now enumerates multisets and drops candidates that violate a charge
conserved by every vertex, which renumbers the process IDs of compiled-model
multiprocess requests. Artifacts written by 1.0.1 declare the new payload and
require a 1.0.1 runtime. See [Process Grouping](process-grouping.md).

## 1.0.0 release

Version `1.0.0` preserves exact model constants, colour weights, and
normalization factors through higher-precision evaluation. It also corrects
native evaluation and retains the exact next-to-leading-colour generation
shortcut. The release keeps the same supported platforms and Python versions.
Tensor algebra and prepared models have been updated for the published
Symbolica 3.0.0 API, including exact complex constants in the SymJIT bridge.
Equivalent exact and floating-point coefficients are again recognized when
selecting optimized recurrence kernels, including in the bundled SM and HEFT models.
Batched recurrence evaluations now fill the already allocated SIMD-aligned
workspace, avoiding unnecessarily small batches at higher multiplicities.
Completed on-the-fly caches can be saved and restored through Python and every
native SDK, without reconstructing the retained currents. See
[saving an OTF warm cache](runtime-and-selectors.md#saving-an-otf-warm-cache).
Symbolica's package license removes the need for a personal key for pyAmpliCol
use; personal-license and restricted-mode fallbacks remain available.
Compiled JIT O2 generation selects compressed code by default.

## 0.2.0 release

Version `0.2.0` adds exact symmetric-group FFT full-colour contraction,
helicity-parametric recurrence and on-the-fly scheduling, built-in SM+HEFT and
trusted UFO HEFT support, and the resumable fixed-helicity/helicity-sum FFT
profiling workflow with its final comparison reports.
See the runnable [scalar HEFT workflow](models-and-processes.md#built-in-scalar-heft-model).

## 0.1.4 release

Version `0.1.4` adds on-the-fly generation and its explicit one-point
`warm_up(...)` lifecycle, while retaining compiled, eager, and recurrence
generation. It also ships progress-aware warm-up entry points for the Python,
C, C++, Fortran, and Rust APIs and the packaged `p p > Z j j` OTF example.

## Install a release

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install pyamplicol
```

Binary wheels include the Rust runtime and native SDK. Wheel users do not need
a Rust compiler. A C, C++, Fortran, or Rust compiler is required only when
compiling a consumer in that language against the included SDK.

Verify the environment with:

```console
pyamplicol doctor
pyamplicol self-test
```

Then run the primary example from an editable copy:

```console
pyamplicol examples copy ./pyamplicol-examples --force
cd pyamplicol-examples
pyamplicol generate_pp_zjj_from_ufo_sm.toml
pyamplicol evaluate_total.toml
```

See [Installation](installation.md) and [Quick Start](quick-start.md).

## What a release contains

The release set consists of:

- one source distribution;
- supported-platform `cp311-abi3` wheels;
- the Python extension and target-specific static Rusticol SDK in each wheel;
- packaged model resources, examples, and user
  documentation required by installed workflows.

Source installation may work elsewhere, but a platform is not advertised as
supported until its installed Python and native API matrix has passed.

Rendered performance PDFs are repository documents, not PyPI wheel payloads or
release-CI benchmarks. Their links intentionally point to the current `main`
branch:

- [MacBook M3 report](https://github.com/MadGraphTeam/pyamplicol/blob/main/docs/performance_reports/macbook_M3_pyAmpliCol.pdf)
- [AMD EPYC report](https://github.com/MadGraphTeam/pyamplicol/blob/main/docs/performance_reports/EPYC_pyAmpliCol.pdf)

## How artifacts are validated and published

The authoritative release workflow builds and tests the source distribution
and platform wheels. Deployment tests install each wheel into a clean
environment and exercise the installed Python runtime and native SDK consumers.

Publication then selects that already validated artifact run and uploads its
exact files through PyPI Trusted Publishing. Upload does not rebuild packages.
This keeps the tested and published bytes identical without adding a second
release build.

GitHub workflow history is available under
[Actions](https://github.com/MadGraphTeam/pyamplicol/actions), and tagged source
snapshots under [Releases](https://github.com/MadGraphTeam/pyamplicol/releases).

## Compatibility promises

- The public Python API is versioned and documented in the repository's API
  contract.
- The native public boundary is C ABI version 1; C++, Fortran, and safe Rust
  wrappers sit on that ABI.
- Run cards use schema version 1 and reject unknown fields.
- Current process artifacts use schema version 3 and the current runtime
  identity/ABI contract.
- Internal generated-artifact formats are not automatically migrated. An old
  artifact that lacks the current contract fails with regeneration guidance.

Regenerate process artifacts with the installed version when upgrading across
an artifact-contract change. Do not edit a manifest or executable payload by
hand.

## Supported and intentionally separate components

- pyAmpliCol has no LHAPDF dependency.
- Symbolica is required for model compilation, generation, and Python exact
  paths; the default compatible f64 JIT runtime executes through SymJIT without
  importing Symbolica.
- Original AmpliCol is optional campaign/developer comparison infrastructure,
  not an installed-package runtime dependency.
- Source contributor builds and release builds have separate dependency
  boundaries. Contributor candidate wheels are deliberately non-publishable.

See [Symbolica and Licensing](symbolica-and-licensing.md) and
[Artifacts and Portability](artifacts-and-portability.md).

## Report a problem

Use [GitHub Issues](https://github.com/MadGraphTeam/pyamplicol/issues) for a
reproducible bug, compatibility failure, or documentation gap. Search existing
issues first.

Include the smallest information that distinguishes the failure:

1. exact pyAmpliCol version and installation method;
2. operating system, architecture, and Python version;
3. the command or short Python snippet;
4. the complete error message;
5. whether the model is built-in, JSON, compiled, prepared, or trusted UFO;
6. execution mode, backend, color accuracy, and process expression;
7. output from `pyamplicol doctor` when installation or SDK discovery is
   involved.

Useful read-only captures are:

```console
pyamplicol doctor --json > doctor.json
pyamplicol inspect ARTIFACT --json > inspect.json
pyamplicol config resolve RUN_CARD.toml --json > config.json
```

Review these files before attaching them. Do not publish private model data,
license keys, cluster paths, credentials, or an artifact you are not permitted
to redistribute.

For a generated native-driver failure, also include:

```console
rusticol-config --version
rusticol-config --target
rusticol-config --json
```

For a numerical report, provide the smallest kinematic point and parameter card
that reproduces it, the selected stable ID or process expression, precision,
and selectors. State whether the optimized total and explicit resolved sum
disagree.

## Triage checklist

Before filing, try the focused check matching the symptom:

| Symptom | First check |
| --- | --- |
| CLI/import failure | `pyamplicol doctor` |
| Installation/runtime smoke | `pyamplicol self-test` |
| Native headers or linker flags missing | Activate the environment, then `rusticol-config --json` |
| Artifact target/ABI rejection | `pyamplicol inspect ARTIFACT` and regenerate with the current release |
| Process expression not found | Inspect stable IDs; see [Process Selection and Permutations](process-selection-and-permutations.md) |
| Selector rejected | Inspect the selected process's helicity/color capabilities |
| Raw UFO/JSON fails in recurrence/eager/OTF | Prepare a compatible `.pyamplicol-model` bundle, or choose compiled mode |
| High-precision Python failure | Confirm Symbolica availability and license; f64 is `--precision 16` |
| Profiling result is surprising | Keep process, layout, selectors, batch, and precision fixed; see [Profiling and Benchmarking](profiling-and-benchmarking.md) |

More symptom-specific help is in [Troubleshooting](troubleshooting.md).

## Feature requests

A useful feature request explains:

- the physics or deployment workflow;
- why existing model, generation, runtime, or selector APIs are insufficient;
- the smallest representative process/model;
- expected public behavior, including error behavior;
- platform or performance constraints.

Avoid attaching a complete private project when a small public model/process
can demonstrate the need.

## Contributing a fix

Contributor setup is documented in [Installation](installation.md).
The repository policy favors focused tests for a demonstrated failure plus one
authoritative CI path. Changes should not add redundant manifests, repeated
hashing, dependency revalidation, or unrelated release ceremony.

Public API changes require corresponding documentation and typing coverage.
Runtime or artifact changes require proportionate native tests; documentation-
only changes do not justify rebuilding the platform release matrix.

## Security and artifact trust

Process artifacts are executable inputs. JIT applications are lowered to
native code at load time, and C++/ASM artifacts may contain native libraries.
Manifest validation establishes internal consistency and path confinement, not
publisher identity. Generate artifacts yourself or obtain them through a
trusted channel.

Do not post a suspected security vulnerability publicly before maintainers can
assess it. Use the repository's available private security-reporting channel if
enabled; otherwise contact the maintainers through the organization channels
without including exploit details in a public issue.

## Links

- [PyPI](https://pypi.org/project/pyamplicol/)
- [GitHub releases](https://github.com/MadGraphTeam/pyamplicol/releases)
- [Issue tracker](https://github.com/MadGraphTeam/pyamplicol/issues)
- [License](https://github.com/MadGraphTeam/pyamplicol/blob/main/LICENSE)
- [Third-party notices](https://github.com/MadGraphTeam/pyamplicol/blob/main/THIRD_PARTY_NOTICES.md)
