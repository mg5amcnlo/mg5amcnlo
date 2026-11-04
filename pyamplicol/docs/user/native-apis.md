---
title: "Native APIs"
nav_order: 2
parent: "Python API"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Native APIs

Every binary pyAmpliCol wheel includes a target-specific Rusticol SDK and every
generated process artifact can include a ready-to-build API bundle. Python,
C11, C++17, Fortran 2008, and Rust 2021 select and perform ordinary total/resolved
evaluation of the same artifact through the same Rusticol core. C, C++,
Fortran, and the standalone Rust interface share the public C ABI v1; Python
uses the wheel's PyO3 binding for these native operations.

For a shared matrix-element provider that can be loaded by MadSpace or linked
into another executable, see [UMAMI and MadSpace](umami-api.md). Its generated
`API/umami` Makefile builds `libumami.so` and a linked standalone driver using
the same installed Rusticol SDK.

`rusticol-config` (also available as `python -m rusticol_config`) only reads SDK
paths, version and linker metadata. It does not initialize pyAmpliCol or
Symbolica, so build-system queries can run alongside numerical evaluations.

FFT basis is a generation-time choice, not a native runtime flag. Generation
with `--color-accuracy full` or Python's `ColorConfig(accuracy="full")`
automatically tries adaptive adjoint FFT for recurrence/on-the-fly execution,
falling back to direct for unsupported FFT plans. Certified gluon trees and
single-insertion HEFT use DDM; quark processes use fundamental chains, while
uncertified interactions retain trace.
The saved `fft_basis_selection` reports the actual representation and reason;
the defaults are automatic contraction and adaptive adjoint selection.
C, C++, Fortran and Rust load either generated artifact and use their existing
total/resolved evaluation calls unchanged. Both FFT choices require NLC/full
colour in recurrence or on-the-fly mode. Automatic LC, compiled/eager, and
correlated generation use direct contraction and trace basis. Select
`--color-contraction direct` to opt out; `--fft trace` or `--fft adjoint` forces
FFT and retains unsupported-plan errors. Forced adjoint FFT rejects correlations.

> **Prerequisites:** install a binary wheel as described in [Installation](installation.md),
> activate that environment, and generate the primary artifact from
> [Quick Start](quick-start.md). Native consumers need the corresponding language compiler;
> they do **not** need a Rust compiler unless the consumer itself is Rust.

## Correlated Born evaluations

The C, C++, Fortran and standalone Rust SDKs expose the same generation-time
colour catalogue, literal spin replacements and grouped correlated evaluations.
These additions are on the `correlators` development branch; use its matching
SDK and library, not an older installed release. Declare the operators with
`Generator.generate(..., correlators=CorrelatorConfig(...))` or
`generate ... --correlators FILE.json` as in [Born Correlations](../correlators.md).

Native correlated evaluation is **binary64 only**, like the ordinary native
SDK. It runs in RustiCol without Python or Symbolica. The separate Python
correlated executor retains Symbolica-backed double-double and arbitrary
precision; its `precision=16` path also uses that executor.

LC, NLC and full colour are selected at generation for each inserted colour
matrix, independently of connection order. The underlying amplitudes remain
complete and full-colour: ordinary `evaluate`/`evaluate_resolved` still return
that underlying process, while the correlated `"born"` request returns the
Born result at the requested correlated accuracy. Spin defaults affect only
correlated calls, never ordinary totals or resolved components.

### Catalogue, grouped requests and result indices

| Language | Catalogue | Grouped evaluation and component access |
| --- | --- | --- |
| C | `rusticol_runtime_color_correlation_count`, `_id`, `_catalogue_json` | `rusticol_runtime_evaluate_correlated_many_f64`; output offset `2*(request*point_count + point)` gives real then imaginary parts |
| C++ | `runtime.color_correlation_ids()`, `.color_correlation_catalogue_json()` | `runtime.evaluate_correlated_many(...)`; `result(request, point)` is `std::complex<double>` |
| Fortran | `runtime%color_correlation_ids()`, `%color_correlation_catalogue_json()` | `runtime%evaluate_correlated_many(...)`; `values(point, request)` is `complex(c_double_complex)` |
| Rust | `runtime.color_correlation_ids()`, `.color_correlation_catalogue_json()` | `runtime.evaluate_correlated_many_f64(...)`; `result.get(request, point)` returns `Option<Complex64>` |

The ID list and full JSON histories follow generation order, including
`"born"`. C/C++/Rust indices are zero-based; Fortran array indices are
one-based. Requests follow caller order and points follow input order.
Different requests share amplitudes and unchanged spin-dependent stages in
one call. All momenta use `[point][external particle][E,px,py,pz]`.

For example, using the registered `"T13"` dipole from the correlation guide:

```cpp
// runtime and flattened momenta have already been prepared.
std::vector<rusticol::SpinCorrelationVector> vectors{
    {3, {{{0., 0., 1., 0.}}}}};
std::vector<rusticol::SpinCorrelationVector> physical;
auto values = runtime.evaluate_correlated_many(momenta, point_count, {
    {"born", physical}, {"T13", physical}, {"T13", vectors},
});
std::complex<double> first_spin_dipole = values(2, 0);
runtime.set_spin_correlation_vectors(vectors);
auto inherited = runtime.evaluate_correlated(momenta, point_count, "T13");
runtime.set_spin_correlation_vectors();  // Clear defaults.
```

```fortran
type(rusticol_correlated_request) :: requests(2)
complex(c_double_complex), allocatable :: values(:, :)
! Unallocated spin_vectors inherits defaults; allocated size zero is physical.
requests(1)%color_correlation = "born"
allocate(requests(1)%spin_vectors(0), requests(2)%spin_vectors(1))
requests(2)%color_correlation = "T13"
requests(2)%spin_vectors(1)%leg = 3_c_size_t
allocate(requests(2)%spin_vectors(1)%components(4, 1))
requests(2)%spin_vectors(1)%components(:, 1) = cmplx([0,0,1,0], kind=c_double)
call runtime%evaluate_correlated_many(momenta, point_count, requests, values)
print *, values(1, 2)  ! First point, spin-correlated dipole.
```

```rust
let mut physical = rusticol::CorrelatedRequest::new("born");
physical.spin_vectors = Some(vec![]);
let mut spin = rusticol::CorrelatedRequest::new("T13");
spin.spin_vectors = Some(vec![rusticol::SpinCorrelationVector {
    leg: 3,
    components: vec![[rusticol::Complex64::new(0., 0.),
                      rusticol::Complex64::new(0., 0.),
                      rusticol::Complex64::new(1., 0.),
                      rusticol::Complex64::new(0., 0.)]],
}]);
let values = runtime.evaluate_correlated_many_f64(
    &momenta, point_count, &[physical, spin], &[])?;
let first_spin_dipole = values.get(1, 0).unwrap();
```

Spin labels are **one-based public external legs** in every language. Supply
one complex four-vector to broadcast, or one per momentum point. Fortran uses
`components(4,points)`; C++ and Rust use a vector of complex four-vectors. The
C representation explicitly interleaves real/imaginary doubles, avoiding any
assumption about a language's complex-number memory layout. The nonempty set
of replaced legs must match a declared spin class.

A C request sets `use_default_spin_vectors=1` with zero explicit vectors to
inherit the setter, or `0` to use its own vectors (zero means physical
helicities). C++ uses `std::nullopt` versus an explicit vector; Rust uses
`None` versus `Some(...)`; Fortran uses unallocated versus allocated
`spin_vectors`. Setters copy their input, and empty setter input clears it.
Optional helicity IDs select physical helicities of the unreplaced legs.

Complete examples for the mixed adjoint/fundamental N3LO interference are in
[`examples/native/correlated.c`](https://github.com/MadGraphTeam/pyamplicol/blob/correlators/examples/native/correlated.c),
[`correlated.cpp`](https://github.com/MadGraphTeam/pyamplicol/blob/correlators/examples/native/correlated.cpp),
[`correlated.f90`](https://github.com/MadGraphTeam/pyamplicol/blob/correlators/examples/native/correlated.f90), and
[`correlated.rs`](https://github.com/MadGraphTeam/pyamplicol/blob/correlators/examples/native/correlated.rs).
The [generation example](../correlators.md#native-correlated-evaluations)
prepares their input. Build all four with `make -C examples/native correlated`;
each executable accepts `ARTIFACT PROCESS`. The generated ordinary standalone
drivers still demonstrate total/resolved evaluation, not correlations.

## What the wheel provides

The installed SDK is owned by the wheel:

```text
pyamplicol/_sdk/
  include/rusticol.h
  include/rusticol.hpp
  rust/rusticol.rs
  fortran/rusticol.f90
  lib/librusticol_capi.a
  config.py
  metadata.json
  link.json
```

`rusticol-config` validates this SDK and emits correctly quoted paths and
target-specific link arguments:

```console
rusticol-config --abi-version
rusticol-config --version
rusticol-config --target
rusticol-config --include-dir
rusticol-config --library
rusticol-config --fortran-source
rusticol-config --rust-source
rusticol-config --cflags
rusticol-config --libs
rusticol-config --rustflags
rusticol-config --cargo-rustflags
rusticol-config --json
```

For a 0.2.0 macOS arm64 release-candidate wheel, the first three commands print:

```text
1
0.2.0
aarch64-apple-darwin
```

The target naturally differs on Intel macOS and manylinux. The ABI remains 1.

Run these commands from the environment that contains pyAmpliCol:

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install pyamplicol
rusticol-config --json
```

If pyAmpliCol was invoked through an explicit path, either activate that
environment or put its `bin` directory on `PATH`. An explicit
`RUSTICOL_CONFIG=/path/to/rusticol-config` also works.

In the source workflow, generated drivers below a checkout can find the nearest
`.venv/bin/rusticol-config` created by `just dev-install`. This is a
development convenience, not an embedded build-machine path.

`--cflags`, `--libs`, and `--rustflags` emit shell-escaped argument streams.
By contrast, `--cargo-rustflags` emits the same linker arguments using Cargo's
unit-separator encoding for `CARGO_ENCODED_RUSTFLAGS`; it is not a shell
argument stream. `--json` exposes typed arrays when a program should avoid
parsing either textual form.

## The generated API bundle

With `generation.emit_api_bundle = true` (the default for ordinary
generation), an artifact contains:

```text
artifacts/pp_zjj/API/
  validation_points.dat
  python/check_standalone.py
  c/check_standalone.c
  rust/check_standalone.rs
  cpp/check_standalone.cpp
  fortran/check_standalone.f90
  umami/umami_driver.c
  umami/metadata.json
```

The per-language Makefiles place binaries, objects, and Fortran modules in the
sibling `artifacts/.pyamplicol-api-build/` directory. The integrity-checked
artifact itself is not modified.

The ordinary language drivers:

- selects a process by stable ID or expression;
- accepts an optional JSON kinematic point;
- accepts a UFO-style JSON model-parameter card and direct overrides;
- evaluates all resolved components and sums them explicitly;
- compares the explicit sum with the optimized total;
- prints either a human result or JSON.

The [UMAMI driver](umami-api.md#generate-and-build) instead demonstrates the
shared-library ABI and its channel/flavour selectors. Its Makefile places
libraries and executables outside the artifact in the same way.

## One process expression in all five languages

The following commands deliberately request `d d~ > g z g`, even though its
stored representative uses another outgoing order. Rusticol resolves the
unique representative and permutes momenta, particles, helicities, color
flows, reductions, and resolved output metadata consistently.

```console
python artifacts/pp_zjj/API/python/check_standalone.py \
  --process 'd d~ > g z g' \
  --set-parameter aS 0.117 0 \
  --json

make -C artifacts/pp_zjj/API/c run \
  ARGS='--process "d d~ > g z g" --set-parameter aS 0.117 0 --json'

make -C artifacts/pp_zjj/API/rust run \
  ARGS='--process "d d~ > g z g" --set-parameter aS 0.117 0 --precision 16 --json'

make -C artifacts/pp_zjj/API/cpp run \
  ARGS='--process "d d~ > g z g" --set-parameter aS 0.117 0 --json'

make -C artifacts/pp_zjj/API/fortran run \
  ARGS='--process "d d~ > g z g" --set-parameter aS 0.117 0 --json'
```

No generated alias is required. Case and whitespace are normalized. Incoming
legs may permute only among incoming legs; outgoing legs may permute only among
outgoing legs. An ambiguous expression fails with the candidate stable IDs.

The drivers also accept the representative stable ID:

```console
--process p_p_to_z_j_j_3
```

After loading, the API exposes the representative stable key and active
representative-to-public permutation. The public process expression remains
the ordering requested by the caller.

The same selectors reach processes that a `--flavor-blind` generation listed
in `process-groups.json` without generating them. Rusticol evaluates the
representative, multiplies every matrix element by the member's exact factor,
and, for a charge-conjugated member, reverses the LC flow words (and maps flow
selectors back). Runtime metadata JSON reports `process_group_member_of`,
`process_group_factor` (`{"numerator", "denominator"}`), and
`charge_conjugated` in every language. The C ABI adds one getter:

```c
uint64_t numerator, denominator;
int32_t conjugated;
rusticol_runtime_process_group_relation(runtime, &numerator, &denominator, &conjugated);
```

Selections other than non-generated members report `1/1` and `0`. See
[Process-group members](process-selection-and-permutations.md#process-group-members).

## Custom kinematics

All five drivers accept `--kinematics PATH`. The file contains exactly one
point, either directly as `[external][4]` or inside a singleton batch
`[[external][4]]`. Its leg order is the order written in `--process`.

For `d d~ > g z g`, `my_sample_point.json` may be:

```json
[
  ["250.0", "0", "0", "250.0"],
  ["250.0", "0", "0", "-250.0"],
  ["204.406", "204.406", "0", "0"],
  ["91.188", "0", "0", "0"],
  ["204.406", "-204.406", "0", "0"]
]
```

Each vector is `[E, px, py, pz]`. Components may be JSON numbers or decimal
strings. Multiple points, booleans, non-finite values, incorrect rank, or an
incorrect particle count are rejected.

```console
python artifacts/pp_zjj/API/python/check_standalone.py \
  --process 'd d~ > g z g' \
  --kinematics my_sample_point.json \
  --precision 80
```

When an artifact retains an exact evaluator, the Python driver keeps decimal
strings as `Decimal` values at non-f64 precision without an intermediate f64
round trip. Native drivers convert the same input to f64 and accept only
precision 16. OTF artifacts are binary64-only through the Python driver too.

When `--kinematics` is omitted, the bundled representative validation point is
reordered into the public process order automatically.

## Model-parameter cards and overrides

`--model-parameters PATH` reads one flat JSON object. A real external value is
a finite number; a complex external value is `[real, imaginary]`:

```json
{
  "aS": 0.117,
  "MZ": 91.188,
  "complex_external_parameter": [1.0, -0.25]
}
```

Apply the card, then override one entry:

```console
make -C artifacts/pp_zjj/API/cpp run \
  ARGS='--process "d d~ > g z g" \
        --model-parameters data/model_parameters.json \
        --set-parameter aS 0.1165 0 --json'
```

Direct overrides are applied after the card and win atomically. Unknown,
immutable, non-finite, or otherwise invalid entries reject the complete update.

Compiled, eager, recurrence, and OTF artifacts use this same API surface when
the bundle is emitted. Eager artifacts carry compact invocation tables,
recurrence artifacts carry current schedules, and OTF artifacts carry the
referenced prepared kernels plus a compact process seed. A native caller never
needs the source `.pyamplicol-model` bundle used during generation.

## OTF warm-up from native APIs

OTF callers can make cold-path work explicit with
`rusticol_runtime_warm_up_f64_with_cores` in C, `rusticol::Runtime::warm_up` in C++,
`runtime%warm_up` in Fortran, or `Runtime::warm_up`/`warm_up_f64` in the safe
Rust wrapper. Each call accepts exactly one flattened binary64 point and
optional global helicity/color ID subsets, constructs and retains that family,
and performs its first evaluation.

In C the entry point is:

```c
const char *one_flow[] = {flow_id};
RusticolWarmUpResult result = {0};
int status = rusticol_runtime_warm_up_f64_with_cores(
    handle, point, momentum_count,
    NULL, 0, one_flow, 1,
    2, report_progress, user_data, &result);
```

The construction-core override is per call: C takes `size_t n_cores` before
the callback, with zero selecting the process-output default. The original
`rusticol_runtime_warm_up_f64` keeps its signature without `n_cores` and uses
that default. C++ takes a
trailing `std::size_t n_cores = 0`; Fortran accepts optional integer `n_cores`
and rejects an explicitly supplied value below one; Rust takes
`Option<usize>` before the callback, with `None` selecting the default and
`Some(0)` rejected. Positive values bound independent query-trace construction;
shared-cache merging and family finalization remain serial. The override does
not change later evaluation defaults or structural cache identity. Generated
API bundles use these same SDKs; their standalone drivers retain ordinary lazy
evaluation unless the caller explicitly invokes warm-up.

The optional fixed-layout callback reports stage, completed and total query
counts, elapsed time, construction workers, and current/peak RSS when the
platform exposes it. Updates are throttled and callbacks run only on the
coordinating caller thread. Returning zero from the C/Fortran callback, or
`false` from the C++/Rust callback, cancels at the next pre-commit boundary;
the final post-commit first-evaluation notification cannot be cancelled. With no
observer, ordinary evaluation and warm-up perform no progress-callback or
memory-sampling work.

The cache belongs to one runtime handle and retains only the last selected
family. Native wrappers do not expose Python's clear-without-unload
convenience: close/free/drop and reload to start fully cold. See the
[OTF lifecycle walkthrough](lc-workloads-and-execution-modes.md#the-otf-warm-state-lifecycle)
for the corresponding C++, Fortran, and Rust call fragments.

## Saving and restoring an OTF cache

All native APIs can save the completed structural cache of an OTF handle and
restore it into a handle loaded from the matching process output. The following
fragments use an OTF output for `g g > g g` at `artifacts/otf_gg_gg`; the
[Python example](runtime-and-selectors.md#saving-an-otf-warm-cache) gives the
generation command and explicit four-vector input.
Here `point` and `next_point` are two flattened binary64 phase-space points,
each containing 16 components in `[particle][E,px,py,pz]` order. They are arrays
in C/Fortran and vectors in C++/Rust. The first evaluation constructs the
recursion; after reopening, the second reuses it at different momenta.

In C, `check(status)` denotes the application's usual error handler: stop on
a nonzero status and retrieve the message with `rusticol_last_error_message`.

```c
RusticolRuntimeHandle *runtime = NULL;
double value;
check(rusticol_runtime_load("artifacts/otf_gg_gg", NULL, NULL, &runtime));
check(rusticol_runtime_evaluate_f64(runtime, point, 16, 1, &value, 1));
check(rusticol_runtime_save(runtime, "gg.otf-cache"));
check(rusticol_runtime_free(runtime));

check(rusticol_runtime_load("artifacts/otf_gg_gg", NULL, NULL, &runtime));
check(rusticol_runtime_load_cache(runtime, "gg.otf-cache"));
check(rusticol_runtime_evaluate_f64(runtime, next_point, 16, 1, &value, 1));
check(rusticol_runtime_free(runtime));
```

```cpp
{
    rusticol::Runtime runtime("artifacts/otf_gg_gg");
    runtime.evaluate(point, 1);
    runtime.save("gg.otf-cache");
} // The original handle is closed.
rusticol::Runtime restored("artifacts/otf_gg_gg");
restored.load_cache("gg.otf-cache");
auto values = restored.evaluate(next_point, 1);
```

```fortran
! Within a procedure using iso_c_binding and the rusticol module:
type(rusticol_runtime) :: runtime
real(c_double), allocatable :: values(:)
call runtime%load("artifacts/otf_gg_gg")
call runtime%evaluate(point, 1_c_size_t, values)
call runtime%save("gg.otf-cache")
call runtime%close()

call runtime%load("artifacts/otf_gg_gg")
call runtime%load_cache("gg.otf-cache")
call runtime%evaluate(next_point, 1_c_size_t, values)
call runtime%close()
```

```rust
{
    let mut runtime = rusticol::Runtime::load("artifacts/otf_gg_gg", None, None)?;
    runtime.evaluate_f64(&point, 1)?;
    runtime.save("gg.otf-cache")?;
} // The original handle is dropped.
let mut restored = rusticol::Runtime::load("artifacts/otf_gg_gg", None, None)?;
restored.load_cache("gg.otf-cache")?;
let values = restored.evaluate_f64(&next_point, 1)?;
```

The reopening block can run in a later invocation of the application. Unlike
the one-flow Python example, these examples use the default helicity/colour
sum; selected evaluations and explicit
`warm_up(...)` calls can be saved in exactly the same way. Reuse the same
selectors after loading to benefit from the saved family. Fortran reports
errors by stopping unless an optional `ierr=status` argument is supplied.

The Rust core's `NativeRuntime` exposes the same `save` and `load_cache`
methods as the dependency-free Rust SDK. C, C++, Fortran, Rust and Python
use the same snapshot format, so a compatible runtime can restore a cache
written through another API. Usual SDK status/exception handling applies.

The snapshot covers the currently retained selection, including LC sums and
contracted NLC/full-colour direct or FFT families. It stores structural rows
and reduction mappings, not machine addresses or evaluated currents. Loading
rebinds the original prepared kernels and allocates numeric workspaces;
the receiving handle's model parameters remain unchanged. The same saved
selection can then run at new phase-space points and batch sizes without
reconstructing its currents. Selecting a different family still follows OTF's
normal cache-replacement behavior. Non-OTF handles reject these operations.
See [Runtime and Selectors](runtime-and-selectors.md#saving-an-otf-warm-cache)
for the Python example and snapshot scope.

## Selector support

The C, C++, Fortran, and Rust total-evaluation entry points accept optional
zero-based selector arrays with one entry per point. They map to the physical
helicity and LC-flow order reported by runtime metadata.

- Batch-global string-ID subsets and per-point selectors are mutually
  exclusive on the same axis.
- LC supports physical helicity and color-flow selection.
- NLC/full supports helicity selection; color is contracted.
- Rusticol groups equal per-point selectors for contiguous execution and
  restores the original point order on output.
- Rectangular resolved evaluation remains batch-global.

The selector contract is consistent across compiled, eager, recurrence, and
OTF artifacts; each lane applies its own internal planner. For OTF, LC exposes
selectable flow families. NLC and full colour expose one contracted component
and reject LC-flow selectors. Those contracted families are correctness
capabilities without a high-multiplicity runtime-practicality promise.

See [Runtime and Selectors](runtime-and-selectors.md) for the same behavior in Python.

## C11

The C header exposes C ABI v1 as an opaque runtime handle, typed status codes,
metadata getters, parameter updates, warning access, explicit OTF warm-up,
OTF cache save/restore, total evaluation, and resolved evaluation. Use
`rusticol-config` rather than hard-coding include or library paths:

```console
eval "set -- $(rusticol-config --cflags) $(rusticol-config --libs)"
cc -std=c11 my_runtime.c "$@" -o my_runtime
```

The shell `eval` converts the command's shell-escaped output into an argument
vector and preserves SDK paths containing spaces.

## C++17

`rusticol.hpp` is a header-only RAII wrapper over C ABI v1. It exposes metadata,
parameters, selectors, warnings, one-point OTF warm-up, cache save/restore,
and total/resolved f64 evaluation:

```cpp
#include <rusticol.hpp>

#include <iostream>
#include <vector>

int main() {
    rusticol::Runtime runtime("artifacts/pp_zjj", "d d~ > g z g");
    runtime.set_model_parameter("aS", 0.117);

    std::vector<double> momenta = /* point-major [point][particle][4] */;
    auto total = runtime.evaluate(momenta, 1);
    auto resolved = runtime.evaluate_resolved(momenta, 1);

    std::cout << total.at(0) << "\n";
    std::cout << resolved.total().at(0) << "\n";
}
```

Compile with:

```console
eval "set -- $(rusticol-config --cflags) $(rusticol-config --libs)"
c++ -std=c++17 my_runtime.cpp "$@" -o my_runtime
```

## Fortran 2008

The wheel ships portable module source rather than a compiler-specific `.mod`
file:

```console
RUSTICOL_FORTRAN="$(rusticol-config --fortran-source)"
eval "set -- $(rusticol-config --libs)"
gfortran -std=f2008 "$RUSTICOL_FORTRAN" my_runtime.f90 "$@" -o my_runtime
```

`type(rusticol_runtime)` owns load/close, metadata, parameter, warning,
one-point OTF warm-up, cache save/restore, and total/resolved f64 methods.
Resolved Fortran storage is `(color, helicity, point)`, the column-major view
of the C ABI sequence `(point, helicity, color)`.

## Rust 2021

`rusticol.rs` is a dependency-free safe wrapper over C ABI v1. It owns the
handle, frees it on drop, exposes typed metadata, selectors, explicit
one-point OTF warm-up and cache save/restore, and remains bound to its creating
thread.

The generated driver is compiled directly with `rustc`:

```console
make -C artifacts/pp_zjj/API/rust check_standalone
make -C artifacts/pp_zjj/API/rust run \
  ARGS='--process "d d~ > g z g" --precision 16 --json'
```

For a hand-written source:

```console
RUSTICOL_RUST_SOURCE="$(rusticol-config --rust-source)"
eval "set -- $(rusticol-config --rustflags)"
RUSTICOL_RUST_SOURCE="$RUSTICOL_RUST_SOURCE" \
  rustc --edition=2021 my_runtime.rs -o my_runtime "$@"
```

`rust-script` is optional and is not a pyAmpliCol runtime requirement. The
generated source includes its minimal Cargo header, so an installed
`rust-script` can run it directly:

```console
RUSTICOL_RUST_SOURCE="$(rusticol-config --rust-source)" \
  CARGO_ENCODED_RUSTFLAGS="$(rusticol-config --cargo-rustflags)" \
  rust-script artifacts/pp_zjj/API/rust/check_standalone.rs -- \
  --process 'd d~ > g z g' --precision 16 --json
```

The generated Makefile exposes the same path as:

```console
make -C artifacts/pp_zjj/API/rust run-script \
  ARGS='--process "d d~ > g z g" --precision 16 --json'
```

## Runtime and precision boundary

All native APIs use the Symbolica-independent f64 runtime:

- direct JIT artifacts load the separate MIT-licensed SymJIT application;
- eager and recurrence artifacts execute their prepared native schedules;
- OTF artifacts construct a selected recurrence family from their compact seed
  and then execute the artifact-local prepared kernels;
- C++/ASM evaluator artifacts load only on a compatible target and CPU;
- no native call imports Symbolica or performs a Symbolica license check.

Python is the only standalone driver that can request Symbolica-backed exact
precision when the artifact retains an exact evaluator. OTF does not retain
that path and rejects non-f64 precision. See
[Symbolica and Licensing](symbolica-and-licensing.md).
The separate Python correlated API is also Symbolica-backed at precision 16;
the native correlated entry points instead use the independent RustiCol
binary64 executor described above.

## Common setup failures

| Symptom | Resolution |
| --- | --- |
| `ModuleNotFoundError: pyamplicol` | Run the Python driver with the environment's Python or activate that environment. |
| `rusticol-config: command not found` | Activate the same environment, set `RUSTICOL_CONFIG`, or use the supported checkout-local `.venv` workflow. |
| `rusticol.h` / `rusticol.hpp` not found | Do not invoke the compiler without the flags emitted by `rusticol-config`. |
| Empty `RUSTICOL_RUST_SOURCE` or `RUSTICOL_FORTRAN` | Verify `rusticol-config --rust-source` / `--fortran-source` in the active environment. |
| Target incompatibility | Use a portable JIT artifact (compiled O1/O2, or eager/recurrence/OTF from a prepared JIT O2 pack), or regenerate C++/ASM/O0/O3 on the destination target. |

See [Troubleshooting](troubleshooting.md) for a fuller decision tree.

## Related pages

- [Examples Gallery](examples-gallery.md) — complete copied examples and generated API commands.
- [Artifacts and Portability](artifacts-and-portability.md) — target rules and trusted-input boundary.
- [Runtime and Selectors](runtime-and-selectors.md) — process ordering and selector semantics.
- [Born Correlations](../correlators.md) — opt-in LC/NLC/full-colour correlations in Python and native SDKs.
