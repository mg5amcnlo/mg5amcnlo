<!-- SPDX-License-Identifier: 0BSD -->

<p align="center">
  <img src="https://raw.githubusercontent.com/MadGraphTeam/pyamplicol/main/docs/assets/pyamplicol_logo.png" alt="pyAmpliCol" width="760">
</p>

<p align="center">
  <a href="https://pypi.org/project/pyamplicol/"><img src="https://img.shields.io/pypi/v/pyamplicol.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/pyamplicol/"><img src="https://img.shields.io/pypi/pyversions/pyamplicol.svg" alt="Python versions"></a>
  <a href="https://madgraph.org/pyamplicol/"><img src="https://img.shields.io/badge/docs-User%20Guide-2f81f7.svg?logo=githubpages" alt="pyAmpliCol documentation"></a>
  <a href="https://github.com/MadGraphTeam/pyamplicol/actions/workflows/tests.yml"><img src="https://github.com/MadGraphTeam/pyamplicol/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/MadGraphTeam/pyamplicol/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-0BSD-blue.svg" alt="License: 0BSD"></a>
</p>

<p align="center"><strong>Fast color-ordered scattering amplitudes from Python and native APIs.</strong></p>

pyAmpliCol generates and evaluates color-ordered scattering amplitudes from
built-in, JSON, or UFO models. It provides a typed Python API and CLI, fast
Rust-backed execution, runtime helicity and color-flow selection, and generated
Python, C11, C++17, Fortran 2008, and Rust 2021 interfaces.

Explore the complete [pyAmpliCol documentation](https://madgraph.org/pyamplicol/)
for guided workflows, API examples, technical reference, and release support.

Opt-in [tree-level spin and colour correlations](docs/correlators.md) prepare
named colour operators at generation and accept spin-contraction vectors at
runtime. This opt-in path supports LC, NLC and full colour at generic
connection order. `CorrelatorConfig.all_color(through_order=k)` prepares the
complete, nonminimal tree-soft catalogue through any positive order `k`;
`Runtime.available_color_correlations()` lists the generated operators.
Catalogues grow rapidly with order. Multiple requests share amplitudes and
accept batches of phase-space points. The `correlators` development branch
also exposes these operations in the C, C++, Fortran and Rust SDKs with
binary64 arithmetic; Python supports double-double and arbitrary precision.

## Installation

Install the release from PyPI:

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install pyamplicol
```

The binary wheels include the Rust runtime and native SDK; wheel users do not
need a Rust compiler. pyAmpliCol has no LHAPDF dependency.

Build the tagged 1.0.1 source snapshot with:

```console
git clone --branch v1.0.1 --depth 1 https://github.com/MadGraphTeam/pyamplicol.git
cd pyamplicol
python -m pip install .
```

A source build requires Python 3.11 or newer, Rust 1.89 or newer, and a C/C++
toolchain. A Fortran compiler is required only for Fortran consumers.

Contributor setup defaults to published Python dependencies from
`pyproject.toml`, including Symbolica 3.0.0 and ufo-model-loader 1.0.0, and the
release-mode native build with published SymJIT 2.26.4.
Building a new release wheel requires a clean Git checkpoint and complete
release assets; the dependency mode does not relax publication guards.

```console
nix develop  # optional on Nix/NixOS
just dev-install
PYTHON=.venv/bin/python just dev-test
```

The `just dev-install` native build can take several minutes. Use
`--wheel-directory PATH` to avoid rebuilding an existing compatible wheel.
For a dirty development checkout, use `--wheel-directory PATH` to reuse an
already-built compatible release wheel, or `--dependencies-only` to install
dependencies without building or staging the project; the latter leaves the
existing native runtime untouched. An optional `--loader-wheel PATH` explicitly
selects a local wheel for dependency development; normal setup uses PyPI.
Editable installs are not used. `just dev-test` also includes a fresh release
build and requires a clean checkpoint; dirty edits can use focused tests against
the staged native runtime.
Historical dependency-development machinery remains behind `--candidate`,
but its pinned APIs are incompatible with the current tensor implementation.
It needs updated, coherent upstream pins before use; use the published lane
for current-source development and tests.

Full installation details are in the
[documentation](https://madgraph.org/pyamplicol/).

## Quick start

Copy the installed examples into an editable workspace:

```console
pyamplicol examples copy ./pyamplicol-examples --force
cd pyamplicol-examples
```

Keep the Python environment containing pyAmpliCol activated while using the
top-level CLI and Python examples, or invoke its executables by explicit path.
For a copy below a source checkout prepared by `just dev-install`, generated
artifact Python and native API drivers also find the nearest checkout `.venv`
automatically; explicit SDK overrides and an active environment take
precedence.

The primary example generates a multiprocess `p p > Z j j` artifact from the
packaged serialized Standard Model, then evaluates and profiles one concrete
subprocess. It expands to eight candidates, one per side-permutation class;
it stores the seven tree-level representatives and reports the omitted
loop-induced `g g > Z g g` class. The card inherits the portable JIT
O2 default, so its process artifact can be moved between supported 64-bit
little-endian macOS arm64, macOS x86_64, and Linux x86_64 hosts:

```console
pyamplicol generate_pp_zjj_from_ufo_sm.toml
pyamplicol evaluate_total.toml
pyamplicol evaluate_resolved.toml
pyamplicol benchmark.toml
```

For direct CLI use:

```console
pyamplicol generate "d d~ > z g" ./artifacts/builtin_ddbar_to_zg \
  --model built-in-sm

pyamplicol inspect ./artifacts/builtin_ddbar_to_zg
```

Process generation can also be steered directly from Python:

```python
from pyamplicol import GenerationConfig, Generator

generator = Generator(GenerationConfig(workers=4))
plan = generator.plan("d d~ > z g")  # Resolve without writing an artifact.
result = generator.generate(
    "d d~ > z g",
    "artifacts/builtin_ddbar_to_zg",
    mode="replace",
)
print(result.output)
```

The same runtime is available from Python:

```python
import json
from pathlib import Path

from pyamplicol import Runtime

momenta = json.loads(Path("data/pp_zjj_momenta.json").read_text())
runtime = Runtime.load("artifacts/pp_zjj", process="d d~ > g z g")
total = runtime.evaluate(momenta)
resolved = runtime.evaluate_resolved(momenta)
assert resolved.total() == total
```

Concrete process expressions may reorder particles within the incoming side or
within the outgoing side. Rusticol maps momenta, helicities, color flows, and
resolved metadata to that requested order; particles never cross the `>`
boundary. Stable process IDs remain available when more than one generated
representative could match an expression.

See the
[examples guide](https://github.com/MadGraphTeam/pyamplicol/blob/main/examples/README.md)
for complete cards, parameter updates, selector examples, and generated API
drivers.

## Models and execution

pyAmpliCol supports:

- the packaged built-in Standard Model;
- packaged serialized JSON and trusted UFO examples;
- user-supplied JSON or trusted UFO model paths;
- leading-color, contracted next-to-leading-color, and contracted full-color
  calculations;
- recurrence, compiled-DAG, eager, and on-the-fly execution modes;
- JIT, C++, and assembly evaluator backends where supported;
- native binary64 execution without Symbolica computations, plus precision-controlled
  Python evaluation when exact expressions are retained. On-the-fly execution
  currently supports native binary64 only.

Reusable artifacts preserve complete public helicity and color physics unless
the request explicitly fixes selectors at generation time. Runtime calls can
then select one flow or helicity globally or per phase-space point without
regenerating the artifact. On-the-fly artifacts always keep selection at
runtime and carry the complete contract in a compact query-local seed rather
than materializing the full axes; `inspect` reports their physical census
without constructing it.
Recurrence, eager, and on-the-fly execution reuse the same prepared model
kernel bundle.

Contracted NLC/full-colour recurrence and on-the-fly execution automatically
try exact `symmetric-group-fft` colour contraction with adaptive adjoint-basis
selection, falling back to direct contraction when the FFT plan is unsupported.
Select `--color-accuracy full` (or `nlc`) to use this default policy.
Certified pure-gluon Yang–Mills trees and single-insertion HEFT use two-anchor
DDM tensors: for `n` external gluons, `(n-2)!` ordered amplitudes replace the
trace basis's `(n-1)!`. Quark processes retain fundamental chains or their
products; uncertified adjoint processes retain trace tensors. The saved
`fft_basis_selection` records the actual representation and its reason, so a
fallback is not presented as a DDM reduction. Configuration defaults are
`contraction = "auto"` and `fft_basis = "adjoint"`; LC, compiled/eager execution,
and correlators retain direct contraction and the trace basis. Use
`--color-contraction direct` to opt out, or `--fft trace` / `--fft adjoint` to
force FFT with that basis and reject unsupported plans. In the benchmarked
pure-gluon family the adjoint basis was faster than trace for six or more
external gluons, by 3.9x per sample with 4x faster generation at ten gluons. See the
[FFT configuration guide](docs/user/configuration.md#color-accuracy-and-lc-layout).
FFT transforms certified permutation-orbit blocks and retains unsupported
terms as exact direct residuals.

Recurrence artifacts persist one helicity-parametric physical-colour
schedule, its helicity-support masks, and precomputed per-helicity row groups;
loading binds those groups once, so warmed evaluation does not rescan the
masks. On-the-fly execution instead constructs and caches the requested family
on first use, which is why that warm-up belongs to its plotted setup time.
Completed OTF work can be retained between programs with `runtime.save(path)`
and restored with `runtime.load_cache(path)` after loading the original process
output. Python and every native SDK expose these operations; see the
[small save/restore example](docs/user/runtime-and-selectors.md#saving-an-otf-warm-cache)
and [native equivalents](docs/user/native-apis.md#saving-and-restoring-an-otf-cache).

The public C ABI is version 1. Every generated artifact can include standalone
Python, C11, C++17, Fortran 2008, and dependency-free Rust 2021 drivers backed
by the wheel-owned static Rusticol SDK.

## Performance reports

Selected manual measurement snapshots are published in the upstream repository:

- [MacBook M3 report](https://github.com/MadGraphTeam/pyamplicol/blob/main/docs/performance_reports/macbook_M3_pyAmpliCol.pdf)
- [AMD EPYC report](https://github.com/MadGraphTeam/pyamplicol/blob/main/docs/performance_reports/EPYC_pyAmpliCol.pdf)
- [FullColor FFT selected-helicity snapshot](https://github.com/MadGraphTeam/pyamplicol/blob/main/docs/performance_reports/summary_plots_final.pdf)
- [FullColor FFT helicity-sum snapshot](https://github.com/MadGraphTeam/pyamplicol/blob/main/docs/performance_reports/summary_plots_final_helicity_sum.pdf)

## Documentation

Read the complete [pyAmpliCol documentation](https://madgraph.org/pyamplicol/).

## Dependencies and license

Release builds use pinned published dependencies plus SymJIT 2.25.0 from an
immutable revision of the official
[symjit-crate repository](https://github.com/siravan/symjit-crate).

pyAmpliCol is distributed under the 0BSD license. Third-party components and
model assets retain their own terms; see
[THIRD_PARTY_NOTICES.md](https://github.com/MadGraphTeam/pyamplicol/blob/main/THIRD_PARTY_NOTICES.md).
