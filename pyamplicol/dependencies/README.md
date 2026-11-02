# Dependency Modes

pyAmpliCol defaults to published runtime dependencies and release-mode native
builds. Historical pinned-source candidate machinery is retained, but its
upstream pins need a coherent update before use with the current tensor APIs.
Release mode uses Symbolica **3.0.0** and SymJIT **2.26.4**. Historical candidate
mode retains SymJIT **2.26.0**; version numbers alone do not establish Python API
compatibility. Candidate builds do not
relabel dependency versions or modify upstream source files.

## Release Mode

`just dev-install` uses pip to resolve the Python runtime dependencies declared
in the root `pyproject.toml`, then replaces published Symbolica and
ufo-model-loader without resolving dependencies again. This replaces older
source candidates reporting the same version. The upcoming release requires
ufo-model-loader 1.0.0, which is not yet published. Until publication, explicitly
supply the locally built loader wheel for contributor validation:

```console
just dev-install --loader-wheel /path/to/ufo_model_loader-1.0.0-py3-none-any.whl
PYTHON=.venv/bin/python just dev-test
```

The installer never chooses a local loader wheel automatically; `--loader-wheel`
is an explicit dependency-development override. Native build and staging use
the existing release-mode workflow. Building a new release wheel requires a
clean Git checkpoint and complete release assets; published-dependency mode
does not relax publication guards. For dirty development checkouts,
`--wheel-directory PATH` reuses an already-built compatible release wheel, or
`--dependencies-only` skips the project build and staging while leaving the
existing native runtime untouched. `--no-build` also skips runtime dependency
installation.
Developer tools and explicitly requested optional references may still be
set up with `--no-build`. No editable installation is used. `just dev-test`
defaults to release mode and includes a fresh build, so it also requires a clean
checkpoint. Dirty edits can use focused tests against the staged native runtime.
The developer dashboard retains its pinned Ratatui 0.4.2 build and FFI checkout
in both lanes; it is not a core runtime dependency.

Release-equivalent builds use exact published versions. Canonical Cargo
resolution uses the published Symbolica 3.0.0 and SymJIT 2.26.4 crates without
source patches. `release-lock.toml` records those versions, and `Cargo.lock`
records normal registry resolution and checksums. The
ordinary locked Cargo build is the dependency check; no local dependency
patches or release-only source projections are needed.

Python Symbolica 3.0.0 is published on PyPI. The required ufo-model-loader 1.0.0
entry has no published wheel recorded in `python-runtime-lock.toml` yet. After
the loader is uploaded, record its official PyPI wheel metadata before running
the final release gates. A local wheel is not a published runtime-lock artifact.
Successful candidate builds do not establish release availability or make
candidate artifacts publishable.

Prepared models in `src/pyamplicol/assets/prepared_models` are the source-runtime
inputs. Release pairs use the release-locked dependencies and also
live in the source-only `release_assets/prepared_models` store and are regenerated
through the manual `release-prepared-models.yml`
workflow using `release-lock.toml` and canonical `Cargo.lock`. Its temporary
bootstrap wheel is non-publishable and omits prepared models and the portable
self-test fixture. The release overlay installs the release pairs at canonical
package paths and removes the auxiliary store, so the retained sdist contains
only release payloads. Prepared-pack compatibility uses model/compiler
identities, storage and plane ABIs, target, and payload hashes—not a separate
dependency checkout fingerprint.

## Historical Candidate Mode (Opt-in)

The retained `--candidate` dependency-development machinery selects managed
checkouts under `dependencies/checkouts`, pinned by `contributor-lock.toml`.
These historical pins are incompatible with the
current tensor implementation and must be updated together before use; this
is not a working current-source setup route. The retained pin table is:

| Dependency | Upstream source | Revision |
| --- | --- | --- |
| Symbolica 3.0.0, including Numerica and Graphica | `symbolica-dev/symbolica`, `main` | `7b31114c7a77571aabacf96a311f27ef0f44d007` |
| SymJIT 2.26.0 | `siravan/symjit-crate` | `530304a07d1be6d5abc80291aa5e92cf28ac5546` |
| GammaLoop, including Spenso/Idenso/Vakint | `alphal00p/gammaloop`, `main` | `312cf6aaf4cd1414f6742ad87ce39922c497a0b6` |
| symbolica-integrate 2.0.1 | `symbolica-dev/symbolica-integrate`, `main` | `716354a07f2660b38390c95702aa617d1f9472c7` |
| ufo-model-loader 0.1.8 | `alphal00p/ufo_model_loader`, `main` | `70ddee6b416f8c8b340e0d087646d77095c5d24b` |

The pinned Symbolica source declares 3.0.0, matching the loader's genuine
`symbolica>=3.0` requirement. The loader lives at
`dependencies/checkouts/ufo-model-loader`; no separate future-loader checkout
is needed. The symbolica-community revision is recorded in
`contributor-lock.toml`.

The generated candidate Cargo overlay selects these upstream sources for both
pyAmpliCol and the community extension without changing canonical release
resolution. The installer configures the community build manifest with the
managed source paths and Symbolica's matching allocator; upstream Rust source
files remain unchanged. Rusticol enables Symbolica's
`native_code_generation` feature. SymJIT supplies the standard P-kernel
interface; Rusticol owns scheduling, factors, accumulation, fanout, and artifact
binding. Its plane adapter uses actual row indices and identity output, with
descriptor lifetime, alignment, aliasing, and mutability checked by the caller.

Candidate mode is not for PyPI publication. Its retained test selector,
`PYAMPLICOL_BUILD_MODE=candidate`, does not make the historical pins compatible.
The retained build machinery uses the tracked prepared-model packs and
portable self-test fixture; an explicitly requested bootstrap omits both and
is non-publishable. Neither bootstrap nor `--update` repairs incompatible
upstream pins: `--update` only moves managed checkouts to the recorded revisions.
`--reset` archives managed state in the workspace-local `.trash` store before
recreating it.

## Optional Profiling References

These opt-ins are independent of the historical candidate machinery.

The original Fortran AmpliCol checkout is optional, developer-only, and used
only as an independent validation and benchmarking reference. Enable it with
`just dev-install --with-legacy-amplicol`. The pinned
`amplicol_with_patches` branch removes the unnecessary LHAPDF
link from its direct color probe, exposes complete recursion-kind diagnostics,
and reports each LC contraction-row partition. This
resolves the physical color component only for a genuinely single-flow case;
multi-flow fixtures use the rows only to verify the complete per-helicity
aggregate. None modifies
amplitude physics. `just legacy-physics` builds that probe and checks the
tracked low-multiplicity LC/NLC/full fixture, including every physical
helicity and every independently resolvable color component, against the
pinned Fortran implementation.

The pinned upstream revision contains no `LICENSE` or `COPYING` file. That fact
is recorded in contributor-only provenance, not the release dependency lock.
The checkout and its developer-only branch are never redistributed in a wheel
or sdist.

The Reference FFT checkout is likewise developer-only and is used only by the
FFT profiling and hardware-sensitive acceptance tools. Enable it with
`just dev-install --with-reference-fft`, which clones the exact
public `AllGluonsMultipletFFT` revision recorded in `contributor-lock.toml`
into `dependencies/checkouts/reference-fft`. That repository keeps the
upstream contraction implementation and adds the calibrated
batch/helicity-sum benchmark interface plus a high-multiplicity roundoff guard.
It is never redistributed in a wheel or sdist. Ordinary `just dev-install`
runs omit both external references. Enable both for the complete FFT profiling
toolchain with
`just dev-install --with-legacy-amplicol --with-reference-fft`.
Enabling either external profiling reference also installs the project-owned
`fft-profiling` Python extra into `.venv`; no editable project reinstall is
required or supported. When updating a source tree that already has the former
Reference FFT checkout, add `--update`; the installer migrates its managed
`origin` to the public repository before fetching the pinned revision.
