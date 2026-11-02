---
title: "Packaging Contract"
nav_order: 4
parent: "Development Documentation"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Packaging And Release Contract

This document is normative for standalone pyAmpliCol package builds and
publication.

## Canonical Build

- Distribution: `pyamplicol==0.4.1`, Python 3.11+, license `0BSD`.
- Backend: the in-tree PEP 517 wrapper delegates to pinned Maturin.
- Python extension: `pyamplicol._rusticol`, built from `rusticol-python` with
  `abi3-py311`.
- Native SDK: `rusticol-capi` links the Python-independent Rusticol core and
  must contain no PyO3, NumPy, Python symbols, or Python linker dependency.
- Version source: root Cargo workspace metadata. Installed Python obtains the
  version from distribution metadata.
- Rust resolution: the committed `Cargo.lock` is authoritative.

Release builds use a temporary allowlisted source overlay and an external
`CARGO_TARGET_DIR`. Opted-in contributor builds use a persistent
workspace-local cache under `.artifacts/dev-install`; neither mode rewrites
the source inputs.

## Native Build Identity

The native identity hashes the Rust/Cargo source closure, the pinned Rust and
Maturin toolchains, effective native Maturin settings, and the declarative
`tool.pyamplicol.native-build` contract. Python orchestration, distribution
metadata, documentation, reports, staging paths, provenance-only locks, and an
explicit inventory of wholly test-only Rust files are excluded. Candidate
versions use the first twelve digits of this same digest; there is no parallel
candidate fingerprint.

Rust files that mix production code with inline `#[cfg(test)]` modules remain
file-granular inputs. Avoiding that conservative invalidation requires moving
the tests into an explicitly audited test-only module; the identity code does
not attempt to parse or rewrite Rust source.

## Dependency Modes

Release mode reads `dependencies/release-lock.toml` and accepts exact published
package/crate versions, including Symbolica 3.0.0 and SymJIT 2.26.4. Canonical
`Cargo.lock` records standard crates.io resolution and checksums without source
patches. Git, path, editable, floating, and candidate dependencies are forbidden.

Candidate mode is available only from a full source checkout. It reads the
repository-only contributor contract and produces explicitly non-publishable
artifacts. Contributor dependency setup and local candidate inputs are excluded
from release wheels and source distributions.

Release and candidate Cargo resolution are physically separate. A release
command must fail closed when exact published inputs are unavailable; it must
never fall back to contributor state.

## Release Prepared-Model Production

Inside MadGraph7 neither the packs nor their metadata are tracked in Git. Each
MadGraph7 release attaches one archive, `pyamplicol-prepared-models.tar.gz`,
holding the metadata of every architecture pair and each distinct bundle once.
`tools/release/prepared_models_archive.py fetch` downloads the archive of the
matching release, else of the latest one, and writes the files into both
directories below; `unpack` does the same for a local archive, `pack` creates
one, and `regenerate` repeats the production described below on the current
host. The installer and CI validate fetched packs against the source tree and
regenerate them when they are stale. The release workflow attaches the archive
its wheels were built with.

The canonical package directory
`src/pyamplicol/assets/prepared_models` remains candidate-owned in a source
checkout. Contributor overlays and wheels use those files byte-for-byte and
never consult release identity.

Release-only source inputs live outside the Python package at
`release_assets/prepared_models`. A normal release build from a Git checkout
requires exactly the README and both architecture pairs in that directory. The
build overlay replaces the four canonical candidate payload files with the
release pairs, removes the auxiliary store, and then validates the projected
package assets against the active release lock. Missing, mixed, candidate, or
stale release packs are build errors; the PEP 517 interface has no release
bypass. The auxiliary store is deliberately outside the native-build identity,
so producing its packs cannot change the bootstrap runtime that produced them.

When those source-owned packs must be regenerated, run
`tools/release/prepared_models_archive.py regenerate` (CI does this through the
`pyamplicol_prepared_models` action). Its dedicated backend entry
point creates a temporary release-version bootstrap wheel with both package
payloads and the auxiliary store removed. That wheel is explicitly marked
`publishable: false` and
`release_prepared_model_bootstrap: true`, requires an exact clean Git revision,
and is rejected by the ordinary release artifact audit. It exists only long
enough to run the architecture-local producer.

The producer writes `mode: release`, an explicit null candidate fingerprint,
the producing package version, and a source identity derived only from
`dependencies/release-lock.toml` and canonical `Cargo.lock`. Producer version
is provenance: a pack may be reused by a later patch release when its schema,
compiler, model-source, dependency, and application ABI identities all match.
The producer writes each pair below `release_assets/prepared_models`; the packs
are portable, so the pair produced on one host is installed for both
architectures. The `pyamplicol-eager-portability.yml` workflow checks that
packs produced on one architecture work on the others.

## Wheel Contents

The wheel contains:

- the typed Python package and one native extension;
- model assets with their provenance and content manifest;
- packaged examples and
  Python/C11/C++17/Fortran 2008/Rust 2021 API templates;
- direct-JIT f64 self-test data;
- license and third-party notice files;
- the target-specific Rusticol C ABI archive, C/C++ headers, Fortran module
  source, SDK configuration, and relative link metadata.

The installed SDK layout is:

```text
pyamplicol/_sdk/
  include/rusticol.h
  include/rusticol.hpp
  fortran/rusticol.f90
  rust/rusticol.rs
  lib/librusticol_capi.a
  config.py
  metadata.json
  link.json
```

The backend captures one static archive from Cargo JSON messages, validates the
complete C ABI, rejects non-relocatable link inputs, scans for Python-family
symbols, and records only target-appropriate system libraries/frameworks.
`rusticol-config` exposes typed C/C++ flags, Fortran source, Rust linker flags,
and JSON metadata without a machine-specific build path.

Wheel audits permit only the `pyamplicol/` package, one `.dist-info/` directory,
and an optional platform repair directory. Standard wheel `RECORD` protects
installed-file integrity. Model and process-artifact digests remain scoped to
the package features that verify those payloads against accidental mutation.

## Source Distribution

One retained source distribution is the source of all release wheels. It must
build with `python -m pip install .` using release-locked published
packages/crates and contain:

- Python/Rust/build sources and lockfiles;
- schemas, tests, examples, user documentation, and release tooling;
- the README, licenses, and third-party notices.

The canonical package paths in the retained sdist contain only the validated
release pairs. The auxiliary `release_assets` store and the candidate package
payloads are absent, so building a wheel from that sdist validates and reuses
the canonical release payload without a second projection.

The sdist excludes dependency checkouts, contributor setup, local candidate
inputs, the auxiliary release source store, generated process outputs, build
products, caches, local environments, development histories, raw campaign
data, and rendered report PDFs.

`just sdist`, `just wheel-from-sdist`, and the PEP 517 hooks exercise the same
backend. The sdist audit checks required members and forbidden source-checkout
inputs before a wheel is retained.

## Validation Matrix

Release targets are:

- macOS 11 arm64, `cp311-abi3`;
- macOS 11 x86_64, `cp311-abi3`;
- manylinux 2.28 x86_64, `cp311-abi3`.

Each wheel receives the full installed-package deployment on CPython 3.11,
without Rust in the consumer environment. That deployment includes:

- `twine check` and platform/native-dependency audits;
- import, CLI, direct-JIT f64 self-test, and packaged-example checks;
- Python total/resolved runtime behavior with Symbolica imports blocked;
- C11, C++17, Fortran 2008, and Rust 2021 f64 driver compilation/execution;
- model-parameter card and direct UFO-parameter override behavior.

CPython 3.14 receives a focused abi3 installation, import, metadata, and
direct-runtime smoke test; it does not duplicate generation or the
multilanguage SDK deployment. The independent Fortran physics oracle runs as a
source-level release check. Performance campaigns are intentionally separate
from release CI.

Local entry points are:

```console
just source-gate
just test-deployment
just release-artifacts
just publish-dry-run
```

`publish-dry-run` validates ordinary package files and prints the upload command
without uploading anything.

## Publishing

The release inventory is exactly one source distribution and one wheel per
supported target. Publication never rebuilds these files.

The validated-artifact workflow is manually dispatched with read-only source
permissions. A separate manual publishing workflow downloads the successful
validated inventory, verifies the expected platforms and non-candidate version,
and publishes through a protected TestPyPI or PyPI environment using OIDC
Trusted Publishing. Only that final job receives `id-token: write`.

Published releases reference one successful exact-source validated-artifact
workflow run. Uploading its retained files is a separate manual action and
never rebuilds them.
