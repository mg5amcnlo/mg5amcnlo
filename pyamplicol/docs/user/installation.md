---
title: "Installation"
nav_order: 2
has_children: true
---
<!-- SPDX-License-Identifier: 0BSD -->
# Installation

pyAmpliCol is distributed as a Python package with a Rust-backed runtime. For
most users, installation is a normal `pip` operation: the published wheel
already contains the Python extension and the Rusticol native SDK.

> **Recommended path:** use Python 3.11 or newer in a fresh virtual
> environment and install from PyPI.

## Installation for MadGraph7

pyAmpliCol is developed inside the
[MadGraph7 repository](https://github.com/MadGraphTeam/MadGraph7) (in
`pyamplicol/`) and released together with MadGraph7, with the same version
number. MadGraph7 uses it for processes generated with the `[amplicol]`
matrix-element backend. Its installer works like the madspace one: by default
it installs into `pyamplicol/install` inside the MadGraph7 directory and leaves
the active Python environment untouched.

From the MadGraph7 command line:

```console
install pyamplicol
```

or directly:

```console
python pyamplicol/install.py            # interactive
python pyamplicol/install.py --bin      # wheel from PyPI (release tarballs)
python pyamplicol/install.py --source   # build from source
```

| Option | Meaning |
| --- | --- |
| `-y`, `--yes` | No prompts; reuse the previous install mode. |
| `--bin` | Install the PyPI wheel of this MadGraph7 version. Offered by default in release tarballs on platforms with a wheel. |
| `--source` | Build from source. |
| `-j N` | Parallel compilation jobs (`install pyamplicol` passes the `nb_core` option). |
| `--system` | Install into the active Python environment instead of `pyamplicol/install`. |
| `--clean` | Delete `pyamplicol/build` and `pyamplicol/install` first. |
| `--build-models` | Regenerate the prepared models instead of downloading them. |

A source build needs a C/C++ toolchain and the Rust version pinned in
`rust-toolchain.toml`. When `rustup` is not available, the installer installs
it into `pyamplicol/build/rust`, without touching `~/.rustup` or `~/.cargo`.
The Cargo build tree is kept in `pyamplicol/build`, so rebuilds after a source
change are incremental.

The prepared SM and SM+HEFT models shipped in the package are not stored in
Git. The installer downloads them from the MadGraph7 release (the release of
this version, else the latest one) and checks that they match the source tree.
When they do not, for example after a change to the model compiler, it
regenerates them locally, which takes one extra build. The same can be done by
hand with `tools/release/prepared_models_archive.py fetch` or `regenerate`.

After installing, the installer checks that Symbolica accepts pyAmpliCol's
library key (see [Symbolica licensing](#symbolica-licensing)). MadGraph7 also
warns when the installed package was built from different sources than the
ones in `pyamplicol/`; reinstall with `install pyamplicol -y` in that case.

## Install from PyPI

### macOS and Linux

```console
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pyamplicol
```

Verify the installation without generating a process:

```console
python -c 'import pyamplicol; print(pyamplicol.__version__)'
pyamplicol doctor
pyamplicol examples list
```

`pyamplicol doctor` reports the package, Python, runtime, platform, and
Symbolica environment that pyAmpliCol can see. `examples list` confirms that
the packaged run cards and data files are available.

## Supported binary-wheel platforms

The current release provides `cp311-abi3` wheels for Python 3.11 and newer on:

| Platform | Architecture | Minimum deployment target |
| --- | --- | --- |
| macOS | Apple silicon (`arm64`) | macOS 11 |
| macOS | Intel (`x86_64`) | macOS 11 |
| Linux | `x86_64` | manylinux 2.28 |

Wheel users do **not** need a Rust compiler. pyAmpliCol does not depend on
LHAPDF.

The generated C11, C++17, Fortran 2008, and Rust 2021 examples compile against
the wheel-owned Rusticol SDK. Install the compiler only for the language you
intend to use:

- C/C++ compiler for C11 or C++17 consumers;
- Fortran compiler for Fortran 2008 consumers;
- Rust compiler for the standalone Rust driver.

See [Native APIs](native-apis.md) for the generated driver workflow.

## Create an editable examples workspace

The examples are package resources. Copy them before editing or running them:

```console
pyamplicol examples copy ./pyamplicol-examples --force
cd pyamplicol-examples
```

Paths inside the supplied TOML cards are relative to the card, so the copied
workspace can be moved. Keep the environment containing pyAmpliCol activated,
or invoke its executable by absolute path.

Continue with [Quick Start](quick-start.md).

## Install a tagged source release

Use a tagged snapshot when your platform has no compatible wheel or when you
need a source build:

```console
git clone --branch v0.2.1 --depth 1 \
  https://github.com/MadGraphTeam/MadGraph7.git
cd MadGraph7/pyamplicol
python tools/release/prepared_models_archive.py fetch
python -m pip install .
```

Inside MadGraph7, `python pyamplicol/install.py --source --system` does the
same, including the prepared models.

A source build requires:

- Python 3.11 or newer;
- Rust 1.89 or newer;
- a C/C++ toolchain;
- network access to the release-locked Python, Cargo, and SymJIT inputs.

The repository-pinned CI toolchain may be newer than the minimum supported
Rust version. A Fortran compiler remains optional unless you build a Fortran
consumer.

To build from the source distribution published on PyPI:

```console
python -m pip download --no-binary pyamplicol pyamplicol
tar -xf pyamplicol-*.tar.gz
cd pyamplicol-*/
python -m pip install .
```

## Contributor installation

For development from a checkout, `just dev-install` creates a repository-managed
environment with published Python dependencies from `pyproject.toml`, including
Symbolica 3.0.0 and ufo-model-loader 1.0.0, and the release-mode native build with
published SymJIT 2.26.4. It does not use an editable installation. Building a new
release wheel requires a clean Git checkpoint and complete release assets;
published-dependency mode does not relax these publication guards.

```console
git clone https://github.com/MadGraphTeam/MadGraph7.git
cd MadGraph7/pyamplicol
python tools/release/prepared_models_archive.py fetch
just dev-install
PYTHON=.venv/bin/python just dev-test
```

Normal setup uses published PyPI dependencies. For dependency development,
`--loader-wheel PATH` explicitly selects a local loader wheel; the installer
does not search for one automatically.

The native build can take several minutes. For a dirty development checkout, add
`--wheel-directory PATH` to reuse an already-built compatible release wheel,
or use `--dependencies-only` to install dependencies without building or
staging pyAmpliCol, leaving the existing native runtime untouched.
`--no-build` skips both runtime dependency installation and the project
build/staging; developer tools and explicitly requested optional references
may still be set up. `just dev-test` includes a fresh release build and therefore
also requires a clean checkpoint; dirty edits can use focused Python tests
against the staged compatible native runtime.

On Nix or NixOS:

```console
nix develop
just dev-install
PYTHON=.venv/bin/python just dev-test
```

The Nix shell supplies Python, Rust, C/C++, Fortran, build libraries, and the
documentation/PDF tools. `just dev-install` prepares the published-dependency
environment in `.venv` and builds and stages the project in release mode.

The original AmpliCol and Reference FFT repositories are optional profiling
inputs and are omitted by default. Request either or both explicitly:

```console
just dev-install --with-legacy-amplicol --with-reference-fft
```

Either opt-in also installs the `fft-profiling` Python extra into `.venv`, so
the profiling driver and PDF renderer are ready without reinstalling the
project in editable mode.

### Historical candidate mode (opt-in)

The old pinned-source dependency machinery is retained behind `--candidate`,
but its pinned APIs are incompatible with the current tensor implementation,
even when the dependencies report the same version numbers. It needs updated,
coherent upstream pins before use and is not a working current-source setup
route. Its candidate wheels are non-publishable. Use the published dependency
lane above; ordinary `just dev-test` uses release mode.

## Symbolica licensing

Generation and Python arbitrary-precision execution use Symbolica. The native
binary64 runtime embedded in generated artifacts does not import Symbolica.
pyAmpliCol attempts to register its package key at startup, including in
spawned workers. A valid package key covers its Symbolica operations without a
personal license. If registration fails synchronously, the existing personal
license and restricted-mode paths remain available; restricted mode clamps
generation to one worker and one Symbolica core.

For details and the built-in personal license-request helpers, see
[Symbolica and Licensing](symbolica-and-licensing.md).

## Common installation issues

### `No matching distribution found`

Check your Python version and platform:

```console
python -VV
python -c 'import platform; print(platform.system(), platform.machine())'
```

If no wheel matches, use the tagged source installation above.

### `pyamplicol: command not found`

Activate the environment where the package was installed:

```console
. .venv/bin/activate
python -m pyamplicol --help
```

### Generated native driver cannot find `rusticol-config`

The SDK discovery command belongs to the installed wheel. Activate that wheel's
environment before invoking `make`, or set `RUSTICOL_CONFIG` to its executable:

```console
export RUSTICOL_CONFIG="$(python -c 'import sys; print(sys.prefix + "/bin/rusticol-config")')"
make -C artifacts/pp_zjj/API/c run
```

### A contributor import says the candidate wheel is stale

The local candidate no longer matches the native build inputs. The historical
candidate pins are also incompatible with the current tensor implementation;
rebuilding them is not a supported recovery route. Switch the same checkout to
the default published-dependency environment:

```console
just dev-install
```

Do not work around this by mixing an extension from a different checkout.

## Next steps

- [Quick Start](quick-start.md) — generate, inspect, evaluate, and profile a process.
- [Configuration](configuration.md) — run cards, overrides, color modes, and evaluators.
- [Models and Processes](models-and-processes.md) — built-in, JSON, UFO, and prepared models.
- [Troubleshooting](troubleshooting.md) — focused solutions for runtime and generation failures.

The exact build and publication boundary is recorded in
[Release and Support](release-and-support.md).

<!-- Page hierarchy for the MadGraph7 (Sphinx) documentation. -->
```{toctree}
:hidden:

quick-start
examples-gallery
```
