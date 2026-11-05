---
title: "Born Correlations"
parent: "Python API"
nav_order: 4
---
<!-- SPDX-License-Identifier: 0BSD -->
# Tree-level spin and colour correlations

For the mathematical definitions, charge normalization, emission ordering,
and translation of the original MadNkLO note, see
[Correlation Conventions](correlator-conventions.md).

Correlations are an explicit generation-time option. Declare the colour
operators and allowed joint spin replacements once, then select an operator
by ID and supply numerical spin vectors at runtime. This provides
four-dimensional tree-level correlated Born quantities, with a generic
construction of ordered colour connections at any finite perturbative order.
It is not restricted to NLO or NNLO. This is **not** a complete subtraction implementation: no
loop amplitudes, kinematic splitting kernels, integrated counterterms, or
extra-dimensional spin components are supplied.

The correlated path supports **LC, NLC and full colour** through Python and
the native C/C++/Fortran/Rust SDKs, with the approximation selected at generation. It retains the
complete colour basis and uses direct contraction. Ordinary uncorrelated
generation and evaluation are unchanged.

## Generate and evaluate

This small example uses the built-in Standard Model and a new output directory:

```python
from pyamplicol import (
    ColorCorrelator, CorrelatorConfig, Generator, ModelSource, Runtime,
)
from pyamplicol.config import ColorConfig, RunConfig

declarations = CorrelatorConfig(
    color_correlations=(ColorCorrelator.dipole("T13", 1, 3),),
    spin_correlations=((3,), (3, 4)),
)
config = RunConfig(action="generate", color=ColorConfig(accuracy="full"))
result = Generator(config).generate(
    "g g > g g",
    "artifacts/gg_correlated",
    model=ModelSource.built_in_sm(),
    correlators=declarations,
)
runtime = Runtime.load(result.output)
point = (
    (500, 0, 0, 500),
    (500, 0, 0, -500),
    (500, 300, 0, 400),
    (500, -300, 0, -400),
)

# "born" is always included: the ordinary full-colour overlap.
born = runtime.evaluate_correlated((point,), precision=40)[0]
colour = runtime.evaluate_correlated(
    (point,), color_correlation="T13", precision=40,
)[0]

# Replace leg 3's polarization by a literal transverse four-vector.
runtime.set_spin_correlation_vectors({3: (0, 0, 1, 0)})
spin = runtime.evaluate_correlated((point,), precision=40)[0]
spin_colour = runtime.evaluate_correlated(
    (point,), color_correlation="T13", precision=40,
)[0]
print(born.real, colour.real, spin.real, spin_colour.real, spin_colour.imag)
runtime.set_spin_correlation_vectors(None)
```

Thus `"born"` with no replacement gives the ordinary helicity-summed Born;
`"T13"` with no replacement gives a colour correlation; `"born"` with vectors
gives a spin correlation; and `"T13"` with vectors gives both together.
`helicities=None` sums all spectator helicities. A global `helicities=` selector
can instead restrict the physical helicity IDs, as for ordinary evaluation.

The result is a tuple of `CorrelatedValue` objects, one per phase-space point.
Each has `real` and `imag` fields of type `Decimal`; `precision` specifies
decimal digits and defaults to 16. `complex(value)` is convenient but converts
to ordinary double precision. An ordered colour correlation can be complex:
do not discard its imaginary part before forming the intended physical
combination. All calls use the runtime's current model parameters and
normalization settings.

## Several requests and several phase-space points

Use `evaluate_correlated_many` to share amplitudes between requested operators,
instead of calling `evaluate_correlated` in a loop. Requests are named by the
caller; several names can select the same generated colour ID with different
spin vectors. For the generated example above:

```python
from pyamplicol import CorrelatedRequest

points = (point, point)  # Replace with any batch of distinct physical points.
requests = {
    "born": CorrelatedRequest(color_correlation="born", spin_vectors={}),
    "dipole_13": CorrelatedRequest(color_correlation="T13", spin_vectors={}),
    "spin_3": CorrelatedRequest(
        color_correlation="born", spin_vectors={3: (0, 0, 1, 0)}
    ),
    "dipole_13_spin_3": CorrelatedRequest(
        color_correlation="T13", spin_vectors={3: (0, 0, 1, 0)}
    ),
}
values = runtime.evaluate_correlated_many(points, requests, precision=40)

# Axis 1 is the request name; axis 2 is the original phase-space point index.
entry = values["dipole_13_spin_3"][1]  # Second point, this specific contraction.
print(entry.real, entry.imag)          # Decimal, preserving complex correlations.
all_points_for_dipole = values["dipole_13"]
all_requests_at_second_point = {name: series[1] for name, series in values.items()}
assert all(len(series) == len(points) for series in values.values())
```

The returned dictionary preserves request order and each tuple preserves input
point order. No result axis is flattened or summed over the requested IDs.
Spectator helicities are still summed, or restricted by the batch-global
`helicities` selector; they are not a third returned axis. Per-point spin
vectors use the same vector-batch notation as the setter below.

`spin_vectors=None` (the default) inherits a snapshot of the current setter
state. An explicit `{}` instead requests physical helicities, even when the
setter holds vectors. Request-local vectors do not change that state. All
requests are validated before evaluation; an unknown colour ID or invalid spin
class does not partially update the runtime.

For each distinct spin assignment the amplitudes are calculated once and
shared by all its colour requests. Across assignments, existing evaluation
stages with exactly unchanged inputs can also be reused. This is stage-level
sharing, not construction of a full spin-density tensor. Its benefit depends
on which stages depend on the changed vectors. Numerical caches are local to
the call and streamed point by point; parameter or precision changes on later
calls cannot reuse an earlier numerical result.

The standalone [local NLO limit example](https://github.com/MadGraphTeam/pyamplicol/blob/main/examples/NLO_limits/nlo_limit_stability_tests.py)
uses both axes: all required dipoles are requested together, with the whole
approach to an unresolved limit supplied as a phase-space batch.

## Spin vectors and joint classes

Leg labels are one-based, in the generated public process order, and components
are contravariant `(v0, vx, vy, vz)` with metric `(+,-,-,-)`. Values may be real
or complex. A four-vector broadcasts to every point; a sequence of four-vectors
provides one vector per point:

```python
runtime.set_spin_correlation_vectors({3: ((0, 0, 1, 0), (0, 0, 2j, 0))})
batch = runtime.evaluate_correlated((point, point), precision=40)

runtime.set_spin_correlation_vectors({3: (0, 0, 1, 0), 4: (0, 0, 1, 0)})
joint = runtime.evaluate_correlated((point,), precision=40)[0]
runtime.set_spin_correlation_vectors({})  # Reset; None is equivalent.
```

### Decimal inputs and double-double arithmetic

Supply `Decimal` components directly when their digits matter. For a complex
component, use a `(real, imaginary)` pair of `Decimal` values; constructing a
Python `complex` first would round both parts to binary64. The setter and
`CorrelatedRequest.spin_vectors` both preserve these inputs until evaluation:

```python
from decimal import Decimal
from pyamplicol import CorrelatedRequest

zero = Decimal(0)
vector = (zero, zero, (
    Decimal("1.0000000000000000000000001"),
    Decimal("0.0000000000000000000000002"),
), zero)
runtime.set_spin_correlation_vectors({3: vector})
arb = runtime.evaluate_correlated((point,), precision=80)
dd = runtime.evaluate_correlated(
    (point,), arithmetic="double-double", precision=31,
)
grouped = runtime.evaluate_correlated_many(
    (point,), {"spin_dipole": CorrelatedRequest("T13", {3: vector})},
    arithmetic="double-double", precision=31,
)
print(grouped["spin_dipole"][0].real)  # Decimal result, no float conversion.
runtime.set_spin_correlation_vectors(None)
```

The default `arithmetic="arbitrary"` uses arbitrary-precision arithmetic at
the requested decimal precision. `arithmetic="double-double"` explicitly
selects genuine Symbolica DoubleFloat arithmetic for sources, retained
evaluation stages, colour contraction and normalization; `precision` then
controls output rounding and must not exceed 31 digits. Inputs finer than
that are rounded when entering DoubleFloat, not when defining the vector.
Both modes return `Decimal` real/imaginary parts. Supply `Decimal` momenta as
well when kinematic input accuracy beyond binary64 is needed. Native SDK
correlated calls are binary64 only; the Python arithmetic selection does not
change their numerical type.

The nonempty set of supplied legs must match a declared class **exactly**.
Here `{3}` and `{3, 4}` are allowed, but `{4}` is not. Every declared leg must
be a spin-1 source with four components. Batch lengths must match the number
of evaluated points. State belongs to each `Runtime` instance; setters copy
their inputs, and a failed setter leaves the previous vectors intact.

Vectors replace the external source literally, including its temporal
component. They are not normalized, projected onto helicity states, or
conjugated on input; the existing source crossing phase is applied once.
Use physical incoming/outgoing momenta in their normal public ordering, with
no manual crossing of the supplied vectors. The same vector enters the
amplitude and is conjugated through the bra, giving a rank-one spin
contraction. Joint replacements contract the same amplitude with several
vectors; they are not products of separate one-leg results. Independent
bra/ket spin vectors and full spin-density tensors are not exposed.

Each replaced leg counts once; spectator helicities are summed incoherently.
The original incoming-spin and colour averages and identical-particle factors
are retained, without an extra average over supplied vectors. Vector magnitude
is preserved: scaling one vector by `z` scales the correlation by `abs(z)**2`.
For a physical massless polarization choose `p.v = 0`; no such condition is
enforced by the setter. Setting, for example, `{3: point[2]}` supplies a literal
nonzero momentum vector for a Ward-identity check. Its vanishing is a
massless gauge-boson statement, not a generic test for massive vectors.

## Automatically register every colour correlation through N^kLO

Use `CorrelatorConfig.all_color(through_order=k)` when the required subset is
not known in advance. For example, this prepares every compatible pair of
one- and two-operation tree-soft colour connections, including the Born:

```python
declarations = CorrelatorConfig.all_color(
    through_order=2,  # 1: NLO; 2: through NNLO; 3: through N3LO; any k >= 1.
    spin_correlations=((3,),),
)
config = RunConfig(action="generate", color=ColorConfig(accuracy="full"))
result = Generator(config).generate(
    "g g > g g", "artifacts/gg_all_correlations",
    model=ModelSource.built_in_sm(), correlators=declarations,
)
runtime = Runtime.load(result.output)

catalogue = runtime.available_color_correlations()
for entry in catalogue:
    print(entry.id, entry.order, entry.bra, entry.ket)

# Every NNLO component, at every point, in one grouped call.
requests = {
    entry.id: CorrelatedRequest(entry.id, spin_vectors={})
    for entry in catalogue if entry.order == 2
}
values = runtime.evaluate_correlated_many(points, requests, precision=40)
first_id = next(iter(requests))
print(values[first_id][0].real, values[first_id][0].imag)
```

Here `points` is a batch of four-leg phase-space points as above. The catalogue
is specific to the selected process and includes `"born"` at order zero.
Listing it loads the stored declarations and matrices but does not initialize
amplitude evaluation. The ordinary `evaluate` path is unchanged.

This is **not** a factory for products of dipoles. At each step every active
coloured leg can emit; an emitted gluon can also split into a quark pair, and
its daughters can radiate later. Every final unresolved-label assignment is
included. The factory pairs all histories of the same order with identical
labelled final representations, including both directed orders and self-pairs.
It follows the note's soft-current construction: hard Born-gluon splitting is
not included automatically, but remains available in explicit declarations.

The recursion has no fixed order cap. Its complete, nonminimal catalogue grows
rapidly with order and multiplicity; it can be much larger than a chosen
subtraction basis. Chronology is the order of colour operations, **not** an
assumption of strongly ordered soft energies. The kinematic current, flavour
weights and identical-fermion exchange signs remain the caller's responsibility.
See the [completeness argument and encoding](correlator-conventions.md#automatic-generic-catalogue).

Automatic IDs have the form `N{r}LO/c{i}/c{j}`, with zero-based connection
indices `i,j` in the sorted order-`r` catalogue. They are process-local labels,
not universal operator names: inspect `bra` and `ket` to identify the physics.
For manually declared operators, IDs remain arbitrary user-chosen strings.
Both forms can be combined with
`CorrelatorConfig(color_correlations=(...), all_color_through_order=k)`;
the `N{r}LO/c{i}/c{j}` ID namespace is reserved when automatic registration
is enabled, so conflicts are rejected before generation starts.

The equivalent CLI declaration is simply:

```json
{"all_color_through_order": 3, "spin_correlations": [[3]]}
```

Pass this file to the existing `generate ... --correlators correlators.json`
option. Expansion happens separately for each generated process, using its
actual coloured leg labels and crossed representations.

## Ordered colour connections

`ColorCorrelator.dipole("T13", 1, 3)` prepares
`<M | T_1 . T_3 | M>`. More general requests supply ordered `bra` and `ket`
operations, with the same number of steps on each side and no fixed order cap:

```python
from pyamplicol import EmitGluon, SplitGluon

cascade = (
    EmitGluon(1, -1),       # Emitter, new gluon label.
    SplitGluon(-1, -2, -3), # Parent gluon, new quark, new antiquark.
    EmitGluon(-2, -4),      # A later step may act on an emitted leg.
)
request = ColorCorrelator("cascade", bra=cascade, ket=cascade)
# Include request in color_correlations before generating the artifact.
```

For example, the following declares two- and three-step gluon connections with
different emitters on the bra and ket:

```python
bra = (EmitGluon(1, -1), EmitGluon(2, -2), EmitGluon(-1, -3))
ket = (EmitGluon(3, -1), EmitGluon(4, -2), EmitGluon(-2, -3))
declarations = CorrelatorConfig(color_correlations=(
    ColorCorrelator("double", bra=bra[:2], ket=ket[:2]),
    ColorCorrelator("triple", bra=bra, ket=ket),
    request,
))
result = Generator().generate(
    "g g > g g", "artifacts/gg_higher_correlations",
    model=ModelSource.built_in_sm(), correlators=declarations,
)
runtime = Runtime.load(result.output)
double = runtime.evaluate_correlated((point,), color_correlation="double")[0]
triple = runtime.evaluate_correlated((point,), color_correlation="triple")[0]
```

These two- and three-step connections supply NNLO and N3LO **colour
structures**, respectively; the corresponding subtraction kinematics and
splitting functions remain the caller's responsibility.

Positive labels identify Born legs; fresh negative labels identify auxiliary
particles. Both sides must have the same number of steps and finish with the
same labelled colour representations. These are colour operations only:
**do not add unresolved momenta** to the Born phase-space input. Prepare only
the IDs you need; IDs must be unique and `"born"` is reserved.

The convention is explicitly **bra-adjoint times ket**. For connected basis
tensors, `K[c,d] = <C_bra,c | C_ket,d>` and the runtime computes
`normalization * sum_h,c,d conjugate(A[h,c]) * K[c,d] * A[h,d]`.
The full directed matrix is used, without assuming Hermiticity. Physical
SU(3) generators satisfy `tr(t^a t^b) = delta(a,b)/2`; colour roles use the
all-outgoing convention, with incoming fundamental representations crossed
by generation. No additional crossing sign should be supplied by the caller.
`SplitGluon` represents one labelled quark flavour with physical `T_R = 1/2`;
no flavour sum (`n_f`), extra `g_s` factors, unresolved phase-space factors, or
symmetry multiplicities are included automatically.
Build higher connections from ordered operations, not by multiplying Born
overlap matrices: those matrices are not action matrices in an orthonormal
closed basis.

## JSON and command line

The equivalent declaration file `correlators.json` is:

```json
{
  "color_correlations": [
    {
      "id": "T13",
      "bra": [{"kind": "emit-gluon", "emitter": 1, "emitted": -1}],
      "ket": [{"kind": "emit-gluon", "emitter": 3, "emitted": -1}]
    }
  ],
  "spin_correlations": [[3], [3, 4]]
}
```

```console
pyamplicol generate 'g g > g g' artifacts/gg_correlated_cli \
  --model built-in-sm --color-accuracy full --correlators correlators.json
```

`CorrelatorConfig.to_json_dict()` and `from_json_dict()` use this same schema.
A splitting step uses `{"kind": "split-gluon", "parent": -1,
"quark": -2, "antiquark": -3}`. The CLI option prepares the artifact;
numerical vector setting and correlated evaluation use Python or the native
SDKs rather than the CLI `evaluate` command.

## Native correlated evaluations

The `correlators` development branch provides binary64 correlated evaluation
in C, C++, Fortran and standalone Rust. These entry points use RustiCol and do
not load Python or Symbolica. Python's separate executor supports
double-double and arbitrary precision. See [Native APIs](user/native-apis.md#correlated-born-evaluations)
for the language-specific owned request types, catalogue enumeration, spin
setter and result indexing.

The complete native examples evaluate the Born overlap, a mixed
adjoint/fundamental N3LO interference and its spin-correlated counterpart
together over two points. Generate their input with:

```python
from pyamplicol import ColorCorrelator, CorrelatorConfig, EmitGluon
from pyamplicol import Generator, ModelSource, ProcessSet
from pyamplicol.config import ColorConfig, RunConfig

alpha = (EmitGluon(2, -3), EmitGluon(-3, -1), EmitGluon(-3, -2))
beta = (EmitGluon(1, -1), EmitGluon(3, -2), EmitGluon(4, -3))
declarations = CorrelatorConfig(
    color_correlations=(ColorCorrelator("cascade-interference", alpha, beta),),
    spin_correlations=((5,),),
)
config = RunConfig(action="generate", color=ColorConfig(accuracy="full"))
Generator(config).generate(
    ProcessSet.from_expressions(("d d~ > d d~ g",), names=("dd_ddg",)),
    "artifacts/native_correlated", model=ModelSource.built_in_sm(),
    correlators=declarations,
)
```

The auxiliary gluon emitted from leg 2 radiates twice in `alpha`; `beta`
attaches three gluons to distinct quark or antiquark legs. Their final auxiliary colour
labels match, so their overlap is the interference, not the square of either
history. With the matching development SDK installed, run from the checkout:

```console
make -C examples/native correlated
examples/native/correlated_cpp artifacts/native_correlated dd_ddg
examples/native/correlated_fortran artifacts/native_correlated dd_ddg
examples/native/correlated_rust artifacts/native_correlated dd_ddg
examples/native/correlated_c artifacts/native_correlated dd_ddg
```

Each reports `VALUE request_index point_index real imaginary`, with zero-based
indices and request order Born, colour interference, colour-plus-spin
interference. The programs also demonstrate reading catalogue IDs and full
histories, supplying default spin vectors, clearing them, and keeping ordinary
evaluation unchanged. C/C++/Rust grouped result access is request then point;
Fortran's natural array layout is `values(point,request)`.

## Colour accuracy and execution scope

| Requested quantity | Correlated support |
| --- | --- |
| Full colour | Exact SU(3) colour matrices, including interference |
| Leading colour (LC) | Leading powers of the inserted colour matrix |
| Next-to-leading colour (NLC) | Inherited first-subleading colour-matrix approximation |

`color.accuracy` (CLI `--color-accuracy lc|nlc|full`) selects the correlated
answer at generation. The ordinary default is LC; specify `full` explicitly
for exact SU(3) contractions, as above. Passing `correlators=` still selects
direct contraction and generic compiled amplitudes and disables unsupported
reductions, recording those adjustments in the effective configuration.
There is no runtime colour-accuracy or single-flow selector for either
correlated method. Generate separate outputs to compare accuracies.

The underlying amplitude output remains complete and full-colour. Ordinary
`Runtime.evaluate` on it therefore evaluates that underlying full-colour
process; use `evaluate_correlated(..., color_correlation="born")` for the
Born result at the requested correlated accuracy. See
[the colour-accuracy convention](correlator-conventions.md#colour-accuracy)
for the inherited LC/NLC prescription and its retained-order checks.

Correlated generation selects complete full-colour, direct, generic compiled
amplitudes and disables helicity-specific zero/parity reductions and current
reuse that would invalidate arbitrary source vectors. Python correlated
evaluation uses a separate exact executor over retained Symbolica evaluator
states. Native SDK correlated evaluation uses RustiCol's binary64 coherent
amplitude executor. Both share amplitudes and unchanged stages between
requests and contract the generation-time colour matrices directly; neither
uses FFT or physical-helicity reductions for the correlated result.

FFT, recurrence, on-the-fly execution, replay-based reductions, nonidentity
process aliases/permutations, partial helicity/colour generation, and append
mode are not supported by this correlation path. Existing artifacts without
the explicit declaration cannot be upgraded merely by setting vectors.
Ordinary generation and `Runtime.evaluate(...)` are unchanged; setting spin
vectors affects only `evaluate_correlated(...)` and inherited requests in
`evaluate_correlated_many(...)`.

## References

The [archived MadNkLO note](https://github.com/MadGraphTeam/pyamplicol/blob/80aec6f706c409b2f8f6c793dff8d8f04b118f7d/IMPLEMENTATION_DOCS/REFERENCES/ColorCorrelators_MadNkLO.pdf)
and [MadNkLO source at revision 646a3db](https://github.com/madnklo/madnklo/tree/646a3db9c8efd7b4cb00e9d89b9197cd5394c01b)
provide background. The note's oriented, non-conjugated connections are not
the runtime's literal bra-adjoint/ket convention: translate adjoints,
orientations, signs, and phases before comparing individual ordered entries.
