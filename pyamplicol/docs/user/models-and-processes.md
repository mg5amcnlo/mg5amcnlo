---
title: "Models and Processes"
nav_order: 1
parent: "Configuration"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Models and Processes

pyAmpliCol separates **model compilation** from **process generation**. A model
describes particles, parameters, interactions, propagators, Lorentz structures,
and color data. A process request selects external states and asks pyAmpliCol to
construct the supported tree amplitudes and executable runtime artifact.

## Model source choices

| Source | Typical use | Executes Python while loading? |
| --- | --- | --- |
| `built-in-sm` | Fast Standard Model compatibility workflows and prepared recurrence/eager/OTF kernels | No |
| `built-in-sm-heft` | Standard Model plus scalar Higgs-effective gluon interactions | No |
| Serialized JSON | Recommended portable external-model input | No |
| UFO directory | Trusted development input | **Yes** |
| Compiled model JSON | Reuse normalized model IR | No |
| `.pyamplicol-model` | Reuse model IR plus a prepared local-kernel backend | No |

## Built-in Standard Model

```toml
[model]
source = "built-in-sm"

[process]
entries = [{ expression = "u u~ > g g", name = "uubar_gg" }]
```

Direct CLI:

```console
pyamplicol generate 'u u~ > g g' artifacts/uubar_gg \
  --model built-in-sm
```

The built-in model selects the wheel-owned prepared JIT O2 bundle for default
recurrence or eager execution. It also retains compatibility aliases that an
arbitrary external model does not automatically inherit.

## Built-in scalar HEFT model

Select `built-in-sm-heft` for an independent model containing the packaged
Standard Model plus the scalar Hgg, Hggg, and Hgggg effective interactions:

```console
pyamplicol model inspect built-in-sm-heft
pyamplicol generate 'g g > H g g' artifacts/gg_hgg \
  --model built-in-sm-heft \
  --coupling-order-policy explicit --max-coupling-order HIG=1 \
  --color-accuracy full --execution-mode recurrence
pyamplicol inspect artifacts/gg_hgg
```

The equivalent packaged run card is `builtin_sm_heft.toml`:

```toml
[model]
source = "built-in-sm-heft"

[process]
entries = [{ expression = "g g > H g g", name = "gg_hgg" }]
coupling_order_policy = "explicit"
max_coupling_orders = { HIG = 1 }
```

The model inherits the packaged SM parameter defaults, including `MH = 125`
GeV, and has no runtime dependency on a UFO checkout. The effective coupling
`GH` depends on model parameters (`G`, `vev`, `MH`, and `MT`), not event
momenta; the momentum factors in the Hgg and Hggg Lorentz tensors are retained.
Update those external inputs through a runtime parameter card rather than
treating the derived `GH` as an independent input. The built-in selector uses
its canonical default restriction and simplification policy.

The wheel-owned portable JIT O2 pack makes `eager`, `recurrence`, and
`on-the-fly` execution available directly from this selector. Compiled mode
supports the `jit`, `asm`, and `cpp` evaluator backends. Explicitly prepared
HEFT bundles can likewise use any of those backends. `builtin_sm_heft`,
`sm-heft`, and `sm_heft` are accepted as configuration aliases.

The canonical selector is also available through the typed API:

```python
from pyamplicol import Generator, ModelSource

model = ModelSource.built_in_sm_heft()
plan = Generator().plan("g g > H g g", model=model)
print(plan.concrete_processes)
```

### Trusted external HEFT UFOs

A trusted scalar-HEFT UFO directory can be supplied through the ordinary UFO
path described below. Inspect it before starting process generation:

```console
pyamplicol model inspect /path/to/trusted/ScalarHEFT_UFO
pyamplicol generate 'g g > H g g' artifacts/ufo_heft_hgg \
  --model /path/to/trusted/ScalarHEFT_UFO \
  --coupling-order-policy explicit --max-coupling-order HIG=1 \
  --color-accuracy full --execution-mode compiled --backend jit
```

UFO modules execute Python while loading. pyAmpliCol recognizes the supported
scalar one-/two-structure-constant contact families and rejects unsupported
Lorentz, colour, or form-factor structures during model preflight; arbitrary
HEFT UFOs are therefore not assumed compatible.
The separately downloaded FeynRules HEFT UFO used by developer acceptance is
not packaged, redistributed, or required at runtime. For portable user
workflows, prefer `built-in-sm-heft`.

## Serialized JSON model

Serialized JSON is the recommended portable external-model form:

```toml
[model]
source = "models/json/sm/sm.json"
restriction = "default"
```

Copy the packaged JSON/UFO examples into an editable workspace:

```console
pyamplicol examples copy ./pyamplicol-examples --force
cd pyamplicol-examples
pyamplicol model inspect models/json/sm/sm.json
```

The wheel includes serialized `sm`, `scalars`, and `scalar_gravity` examples.
They become ordinary filesystem paths under the copied workspace; no code needs
to know the package installation directory.

If only the packaged models are wanted, the copied
`python/copy_packaged_models.py` helper can populate a separate empty model
workspace:

```console
python python/copy_packaged_models.py ../pyamplicol-models
```

## Trusted UFO model

A UFO directory uses the same source field:

```toml
[model]
source = "/path/to/MyUFO"
restriction = "default"

[process]
entries = [{ expression = "u u~ > z g" }]

[generation]
output = "artifacts/my_ufo_process"

[evaluator]
execution_mode = "compiled"
backend = "jit"
```

> UFO modules are Python code and execute during loading. Only load a UFO model
> from a source you trust. Serialize it for portable or automated workflows.

In a TOML card, `restriction` is normally a loader name such as `default`,
`no_widths`, or `none`. With the typed Python API, an explicit restriction file
may be supplied. Relative typed-API restriction paths are resolved from the
model directory; card names select the conventional `restrict_<name>.json` or
`restrict_<name>.dat` file.

## Model commands

Inspect model capabilities and diagnostics:

```console
pyamplicol model inspect models/json/sm/sm.json --restriction default
```

Compile portable normalized model IR:

```console
pyamplicol model compile \
  models/json/sm/sm.json models/sm.pyAmplicol-model.json \
  --restriction default
```

Enumerate concrete processes without generating a process artifact:

```console
pyamplicol model processes 'p p > Z j j' \
  --model models/sm.pyAmplicol-model.json \
  --multiparticle 'p=d,d~,g' \
  --multiparticle 'j=d,d~,g' \
  --flavor-scheme 2 --max-quark-lines 2
```

Unsupported model features are reported during model preflight, before an
expensive process build.

## Prepared kernel bundles

A file ending in `.pyamplicol-model` contains normalized model IR, exact
expressions, and one prepared local-kernel backend. It enables recurrence,
eager, or on-the-fly process generation without compiling each local kernel
again.

Create a JIT O2 prepared bundle:

```console
pyamplicol model compile \
  models/json/sm/sm.json models/ufo-sm-jit-o2.pyamplicol-model \
  --backend jit --jit-optimization-level 2
```

Use it for eager generation:

```console
pyamplicol generate 'd d~ > z g g g' artifacts/ddbar_z3g_eager \
  --model models/ufo-sm-jit-o2.pyamplicol-model \
  --execution-mode eager
```

The three prepared-kernel lanes store different process-level structures:
eager writes compact invocation tables, recurrence writes compact current
schedules, and on-the-fly writes a compact process seed from which the selected
query-local recurrence family is built. Each completed process artifact carries
the referenced kernels and is standalone; the source prepared bundle is not
needed at runtime.

At native `f64`, OTF supports LC helicity/flow selection and the singleton
contracted color axis used by NLC and full colour. Contracted color rejects LC
flow selectors and is intended as a low-multiplicity correctness capability,
not a high-multiplicity practicality promise. OTF is binary64-only; eager and
recurrence can retain the Symbolica-backed exact path when their artifact does.

Prepared JIT state uses the exact-O2 portability contract across supported
64-bit little-endian macOS and Linux hosts. Process-local compiled JIT O1 and
O2 are also portable; C++/ASM and explicit O0/O3 JIT remain target-native. See
[Artifacts and Portability](artifacts-and-portability.md).
See [Generation Modes and Evaluators](generation-modes-and-evaluators.md) for
the generation and warm-state trade-offs.

## Typed model API

```python
from pyamplicol import ModelSource

json_source = ModelSource.from_path("models/json/sm/sm.json")
ufo_source = ModelSource.from_path(
    "/path/to/MyUFO",
    restriction="restrict_default.dat",
)

compiled = json_source.compile()
print(compiled.name)
print(compiled.capabilities.supported_color_accuracies)
print(compiled.supported)

compiled.write("models/sm.pyAmplicol-model.json")
compiled.write_parameter_card("data/sm_parameters.json")
```

`CompiledModel` is an immutable public handle. Stable metadata is available
through `compiled.info`; compiler-owned expression/tensor IR remains private.
Compiled models are content-addressed by their source, restriction,
compiler/schema versions, and normalization/tensor policies.

See [Python API](python-api.md) for generation with a compiled handle.

## Parameter restrictions and runtime cards

Model compilation restrictions determine the model baked into the artifact.
Mutable external UFO parameters may then be updated at runtime.

A generated parameter card is a flat JSON mapping:

```json
{
  "aS": 0.118,
  "MZ": 91.1876,
  "complex_parameter": [1.0, 0.0]
}
```

Each value is a finite real number or `[real, imaginary]`. Apply it through the
runtime CLI:

```console
pyamplicol evaluate artifacts/pp_zjj \
  --process 'd d~ > g z g' \
  --model-parameters data/model_parameters.json \
  --momenta data/pp_zjj_momenta.json
```

The generated standalone API drivers also accept repeated
`--set-parameter NAME REAL IMAG` options; those overrides are applied after the
card and win over its values. The main `pyamplicol evaluate` command uses the
`--model-parameters` card, while Python callers can use
`Runtime.set_model_parameter()` or `Runtime.set_model_parameters()`. Updates
are atomic: unknown, immutable, invalid, or non-finite input rejects the whole
batch.

## Process expressions

A process has incoming and outgoing particles separated by `>`:

```text
d d~ > Z g g
```

Named requests provide stable IDs:

```toml
[process]
entries = [
  { expression = "u u~ > Z g", name = "uubar_Zg" },
  { expression = "u u~ > Z g g", name = "uubar_Zgg" },
]
```

Repetition syntax such as `3*scalar_0` is supported for particle names.

## Multiparticles

```toml
[process]
entries = [{ expression = "p p > Z j j" }]
flavor_scheme = 2
max_quark_lines = 2

[process.multiparticles]
p = ["d", "d~", "g"]
j = ["d", "d~", "g"]
```

Every model supplies `all`, in declaration order, containing its valid
propagating physical external states. Ghosts, Goldstones, non-propagating
records, and auxiliary particles are excluded. User labels merge over defaults;
an explicit `all` overrides the generic one.

Example enumeration:

```console
pyamplicol model processes 'p p > all all' \
  --model built-in-sm --flavor-scheme 1 --max-quark-lines 0
```

`all` is intentionally broad and may produce a combinatorial expansion for a
large UFO model. Prefer a focused custom label when possible.

With a UFO or other compiled model (and in `model processes`), expansion
enumerates each side of `>` as a multiset of particles, so two candidates
never differ only by an ordering within a side. Process IDs count these
candidates, so they differ from those of earlier pyAmpliCol versions, which
counted every ordered candidate: re-read them with `pyamplicol inspect` after
regenerating. The built-in SM catalogue keeps its own ordered enumeration.
When generating with a compiled model, candidates that violate an additive
charge conserved by every model vertex are dropped before the colour and
amplitude checks. The charges are derived from the vertex list itself: electric charge,
and quark or lepton numbers per generation whenever the vertices conserve
them (quark numbers reduce to baryon number with a non-diagonal CKM matrix).
With the shipped serialized Standard Model (`sm.json`, default
restriction) this leaves exactly the 385 tree-level subprocesses of
`p p > j j j j`. `model processes` applies the same filter, but not the
colour and amplitude checks. The built-in SM keeps its catalogue prefilters (`flavor_scheme`,
`max_quark_lines`). See
[Conserved charges in UFO expansion](process-grouping.md#conserved-charges-in-ufo-expansion).

## Permutation-equivalent processes

Multiprocess expansion stores one representative for candidates that differ
only by reordering within the incoming side and/or within the outgoing side.
At runtime, this is valid:

```console
pyamplicol inspect artifacts/pp_zjj --process 'd d~ > g z g'
```

even when the representative is stored as `d d~ > Z g g`. Rusticol maps the
public particle order centrally across momenta, particle metadata, helicities,
LC flow words, selectors, reductions, and resolved values.

Rules:

- incoming particles may only permute with incoming particles;
- outgoing particles may only permute with outgoing particles;
- repeated identical particles use deterministic first-unused matching;
- ambiguous representative matches fail with candidate stable IDs;
- stable process and explicit alias IDs take precedence over inferred matching.

The current [Born Correlations](../correlators.md) API is an exception: it
requires the generated external-leg ordering and rejects nonidentity
permutations. Declare spin and colour legs using that process's one-based
labels; a declaration used for a process set must be valid for every generated
member. Ordinary runtime permutation support is unchanged.

## Flavour-blind generation

`--flavor-blind` (`process.flavor_blind`) generates one subprocess per flavour
class: subprocesses related by a relabelling of flavours with equal spin,
colour, mass and width and, unless `--no-consider-charge-conjugation`, by
charge conjugation. For the built-in SM, `p p > j j j j` at `flavor_scheme = 5`
has 486 subprocesses, 33 flavour classes and 23 classes with charge
conjugation.

Process IDs are assigned before the classes are formed, so a generated
representative has the same ID with or without the flag. The representative
is the canonical member: smaller `|PDG|` first and particles before
antiparticles, so `d d~ > g g` represents `u u~ > g g` and `d g > d g`
represents `u~ g > u~ g`. The other members are not generated; they are
recorded in `process-groups.json` with the leg permutation onto the
representative. A subprocess with an explicit `ProcessAlias` keeps its own
artifact entry.

The equality is assumed, not verified. A member gets a warning, logged and
stored with it, when its relation is not a symmetry of the model vertices
that the representative is generated with: those within
`max_coupling_orders` and, under the default `minimal` coupling-order policy,
within the representative's minimal coupling-order envelope. A relabelling
must map each such vertex onto one with the same structure and coupling;
charge conjugation also requires every such vertex to couple fermions
vector-like. Pure-QCD jet classes therefore pass, while `p p > z j` warns: the
photon and Z couplings distinguish `u` from `d`, and the chiral Z coupling
breaks charge conjugation. The check looks at vertices rather than diagrams
and can warn about a relation that holds for the subprocess at hand.
[Process Grouping](process-grouping.md) covers flavour classes, warnings and
the numerical analysis in detail.

## Tree-level support and omitted candidates

Multiparticle expansion may find a syntactically valid process with no
model-supported tree-level amplitude. pyAmpliCol reports and omits it rather
than creating a zero or broken artifact. In the primary SM example,
`g g > Z g g` is loop-induced and therefore omitted.

This is distinct from an unsupported model feature: model preflight rejects
unsupported theory structures with an actionable diagnostic.

## Current generic UFO support

The generic path covers the proven subset used by shipped tests and examples,
including:

- scalar, Dirac-like fermion, vector, and proven massless spin-2 paths;
- SU(3) singlet, fundamental, antifundamental, and adjoint particles;
- implemented identity, gamma/gamma5, projector, sigma, metric, and momentum
  Lorentz structures;
- trilinear kernels, color-singlet contact trees, and proof-gated colored
  contacts;
- LC flows and contracted NLC/full SU(3) calculations.

These general model capabilities do not imply LC/NLC or every execution mode
is available for correlations. The opt-in correlated Born path currently uses
full SU(3), with singlet, fundamental, antifundamental, and adjoint colour
representations, and generic compiled amplitudes. Spin replacements apply only
to four-component vector sources, not fermion or spin-2 sources. Supported
model preflight remains required; see [Born Correlations](../correlators.md)
for the narrower execution and API scope.

Preflight rejects Majorana/FNV fermions, spin 3/2, sextets, epsilon color
tensors, multiple/non-SU(3) color groups, unknown form-factor functions,
unsupported custom propagator tensors, and unproven general colored
higher-point contacts. Massive spin-2 is experimental; the packaged massless
`scalar_gravity` model is a tested path.

The exact, code-derived list of every condition, rejection and diagnostic
code is kept in [UFO Model Coverage](ufo-model-coverage.md). For the
validated release boundary, see [Release and Support](release-and-support.md).
