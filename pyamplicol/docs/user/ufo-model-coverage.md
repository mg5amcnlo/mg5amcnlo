---
title: "UFO Model Coverage"
nav_order: 2
parent: "Configuration"
---
<!-- SPDX-License-Identifier: 0BSD -->
# UFO Model Coverage

This page is the exact reference of what a generic UFO (or serialized JSON)
model may contain for pyAmpliCol, what is accepted only under a condition, and
what is rejected or ignored. Every row was derived from the code of
pyAmpliCol 1.0.0 with ufo-model-loader 1.0.0 and Symbolica 3.0.0; the paper
summarises it in one table. Read
[Models and Processes](models-and-processes.md) first for the workflow, and
use `pyamplicol model inspect <source>` to see the issues reported for your
own model.

Status words used below:

| Status | Meaning |
| --- | --- |
| supported | accepted without further condition |
| conditional | accepted when the stated condition holds, otherwise the stated diagnostic |
| experimental | accepted, but the path is not validated against a reference |
| rejected | refused with the stated diagnostic |
| ignored | loads without error but never affects generated amplitudes |

## Summary

| Area | Supported | Rejected or restricted |
| --- | --- | --- |
| Input | UFO directories (trusted Python), serialized JSON, compiled IR files, prepared kernel bundles | anything else; ufo-model-loader older than 1.0.0 |
| Fields | scalars, Dirac fermions, vectors, massless spin-2 (validated); massive spin-2 (experimental) | Majorana and fermion-number-violating fermions, spin 3/2, higher spins; ghosts and Goldstones only internally |
| Colour | one SU(3) with 1, 3, 3̄, 8; external coloured legs are triplet fermions or octet vectors | sextets, other groups, N_c other than 3, d^abc, epsilon and sextet tensors |
| Lorentz | Identity, Gamma, Gamma5, ProjM, ProjP, Sigma, Metric, P; PSlash in propagators | Epsilon, C, IdentityL, form factors, functions outside the registry |
| Three-point vertices | any spins with colour 1, delta, T^a, f^abc | d^abc; one- and two-point terms are ignored |
| Four-point vertices | colour-singlet (any Lorentz structure); the two-f gauge contact; HEFT Hggg | other coloured contacts (four-fermion operators, delta couplings, d^abc) |
| Five-point and higher | colour-singlet contacts of any valence (tested to ten scalars); HEFT Hgggg | every other coloured contact |
| Propagators | standard scalar, Dirac, vector and spin-2 kernels with fixed widths; custom UFO propagators that lower exactly | complex-mass scheme; gauge choices other than Feynman (massless) and unitary (massive) |
| Parameters | external and internal, real or complex, the fixed function registry, restriction cards, runtime cards for external parameters | custom function bodies, epsilon-expanded values (finite part kept), counterterm vertices (dropped) |
| Processes | 2 to N tree level, LC, NLC and full colour, FFT contraction in the trace or adjoint basis | loop-induced channels (omitted) |

## Preflight diagnostic codes

`pyamplicol model inspect` lists these codes; any error code makes the model
unsupported for generation, warnings do not.

| Code | Severity | Trigger |
| --- | --- | --- |
| `unsupported-spin` | error | a particle spin code outside -1, 1, 2, 3, 5 |
| `unsupported-color-representation` | error | a particle colour code outside 1, 3, -3, 8 |
| `majorana-fermion` | error | a spin-1/2 particle whose name equals its antiname |
| `form-factors` | error | the model declares any form factor |
| `function-arity` | error | a declared function with a registry name but a different argument count |
| `unknown-functions` | error | a function name outside the registry and the UFO tensor heads in any parameter, coupling, propagator, Lorentz or colour expression |
| `unsupported-contact-color-lowering` | error | a vertex with more than three legs whose colour structure could not be decomposed into contact kernels (reported after tensor lowering) |
| `experimental-massive-spin-2` | warning | a spin-2 particle whose mass is not `ZERO` |

Some rejections happen at model compilation rather than at preflight and
carry a `ValueError` message instead of a code; they are marked below.

## Sources and loading

| Feature | Status | Condition and consequence |
| --- | --- | --- |
| Source kinds | supported | `built-in-sm`, `built-in-sm-heft`, a UFO directory containing `__init__.py`, a `.json` model in ufo-model-loader's schema, a compiled IR file `*.pyAmplicol-model.json`, or a prepared bundle `*.pyamplicol-model`. A directory without `__init__.py`, a missing path or unreadable JSON is a `ValueError`. |
| Built-in selector spelling | conditional | Use `built-in-sm`; the `builtin_sm` spelling is accepted by the config resolver but not by the typed model layer. |
| UFO Python execution | supported | A UFO directory is imported as Python with your privileges; only load trusted models and prefer the JSON form for automation. During the import the `UFO_SCALARS_MODEL_*` and `UFO_GRAVITY_MODEL_*` environment variables are hidden and no `.pyc` files are written. |
| ufo-model-loader version | conditional | External UFO and JSON models require an installed release 1.0.0 or newer (development builds are refused with a `RuntimeError`); built-in models do not need it. |
| Required UFO attributes | conditional | `all_orders`, `all_parameters`, `all_particles`, `all_lorentz`, `all_couplings` and `all_vertices` must exist; `all_propagators`, `all_functions`, `all_form_factors` and `all_CTparameters` are optional; objects of class `CTVertex` are dropped silently. |
| Serialized JSON schema | conditional | The file must end in `.json` and provide `name`, `restriction`, `orders`, `parameters`, `particles`, `propagators`, `lorentz_structures`, `couplings` and `vertex_rules`. |
| Loader errors | conditional | Errors raised inside ufo-model-loader (missing restriction file, external parameter without default, malformed JSON) surface as raw Python tracebacks, not as pyAmpliCol `ModelError` messages. |
| Restriction selector | supported | `default` applies `restrict_default.dat` (UFO) or `restrict_default.json` (JSON) when present, otherwise the model defaults without zero-pruning; `none` applies no card; any other name needs `restrict_<name>.dat` or `.json`. |
| Restriction file location | conditional | A restriction given as a path must be named `restrict_<name>.dat` (UFO, inside the model directory) or `restrict_<name>.json` (JSON, beside the model file); other names or locations are a `ValueError`. |
| What a restriction changes | supported | A card sets external parameter values only; `.dat` cards are SLHA parameter cards, `.json` cards map names to `[re, im]`. With `simplify=true` (default) parameters set to zero become constants and the couplings and vertices they kill are pruned. |
| Restrictions on built-in and precompiled sources | rejected | `restriction` and `simplify=false` apply only to UFO and JSON sources; built-in, compiled and prepared models refuse them. |
| Compilation cache | conditional | Compiled models are cached under `$PYAMPLICOL_CACHE_DIR/models` (or the platform cache) keyed by source contents, options and toolchain. Editing a `restrict_*.json` beside a JSON model does not change the key: pass `--no-model-cache` or the file path after editing it. |
| Dry run | conditional | `--dry-run` never compiles a UFO or JSON model; compile it or run `model inspect` first so the cache is populated. |
| Compiled IR files | conditional | Tied to the exact pyAmpliCol, Symbolica and ufo-model-loader versions that wrote them; regenerate after any upgrade. |
| Prepared bundles | conditional | Verified member by member (sizes and SHA-256) on every load; must match the current bundle schema and model-compiler version. |
| Parameter card | supported | `CompiledModel.write_parameter_card()` writes the runtime-adjustable external parameters after the restriction as `{name: [re, im]}`; parameters fixed to zero by the restriction are absent and immutable. |
| Model CLI | supported | `pyamplicol model inspect` loads even unsupported models and lists every issue; `model compile` and `model processes` refuse unsupported models. There is no separate preflight subcommand. |

## Particles

| Feature | Status | Condition and consequence |
| --- | --- | --- |
| Spin codes | supported | -1 (ghost), 1 (scalar), 2 (Dirac fermion), 3 (vector), 5 (spin-2); any other code is `unsupported-spin`. |
| Spin 3/2 and higher | rejected | `unsupported-spin`; no downstream path exists. |
| Scalars | supported | standard propagator i/(p^2 - m^2 + i m Gamma). |
| Dirac fermions | supported | particle and antiparticle must be distinct; standard massive or massless Dirac propagator. |
| Massless fermions | conditional | Treated as two-component Weyl spinors when the mass is literally `ZERO` or an internal parameter equal to zero and the propagator is the loader default; an external mass parameter that merely defaults to zero keeps four-component currents. |
| Majorana fermions | rejected | `majorana-fermion` when name equals antiname; fermion-number-violating flow is not implemented. |
| Vectors | supported | massless vectors get helicities -1, +1 and Feynman gauge; massive vectors get -1, 0, +1 and unitary gauge, decided from the restricted default mass at compile time. |
| Massless spin-2 | supported | 16-component tensor, helicities -2 and +2 only, de Donder propagator; validated by the packaged scalar-gravity model. A model without its own spin-2 propagator must define the `dim` parameter used by the loader default. |
| Massive spin-2 | experimental | accepted with `experimental-massive-spin-2`; the Fierz-Pauli propagator and five helicities are implemented but only the propagator projector is unit-tested; such a model must declare its own propagator because the loader has no default. |
| Ghosts | ignored | any particle with a nonzero ghost number loads, but it can never be external, is excluded from every current, and has no propagator kernel. |
| Spin -1 without ghost number | rejected | classified as a non-external auxiliary whose propagator has no contract; model preparation fails when it is needed. |
| Colour codes | supported | 1, 3, -3, 8; anything else is `unsupported-color-representation`. Colour always means SU(3) with N_c = 3. |
| Coloured external states | conditional | an external coloured leg must be a triplet or antitriplet fermion or an octet vector; coloured scalars, octet fermions and coloured tensors are refused when a process names them (`ValueError`). |
| External-state eligibility | supported | a particle can be named in a process only if it has positive spin, zero ghost number, is not a Goldstone, is propagating and is not a compiler auxiliary; names resolve exactly, then case-insensitively when unique. |
| Particle identity | supported | antiname must exist and be involutive, non-self-conjugate pairs use opposite-signed PDG codes, names and PDG codes are unique, self-conjugate particles are neutral. |
| Mass | conditional | `ZERO` or a model parameter; the massless or massive class (helicities, gauge, Weyl projection) is fixed once at compile time from the restricted default and cannot be changed by a runtime card. |
| Complex masses and widths | rejected | mass and width values must be real; there is no complex-mass scheme. |
| Width | supported | `ZERO` or a parameter; enters only the fixed-width denominator of massive propagators and can be overridden at runtime. |
| Goldstone bosons | conditional | must be propagating scalars; matched to the unique massive vector with identical colour, quantum numbers and mass and then absorbed into its unitary-gauge propagator; an ambiguous match is an error, an unmatched Goldstone stays an internal scalar. |
| Propagating flag | supported | non-propagating particles cannot be external but are compiled normally. |
| Electric charge | supported | recorded as a bookkeeping quantum number only; not used to check charge conservation. |
| Other attributes | ignored | `texname`, `antitexname`, `line`, `counterterm`, `LeptonNumber` and `Y` never affect amplitudes. |
| Custom propagator attribute | conditional | a particle whose propagator differs from the loader's Feynman default is used as written and loses the Weyl projection and Goldstone absorption. |

## Parameters, couplings and functions

| Feature | Status | Condition and consequence |
| --- | --- | --- |
| Parameter nature | supported | `external` (runtime input) or `internal` (substituted symbolically); the common `interal` typo is corrected; anything else fails in the loader. |
| Parameter type | supported | `real` or `complex`; complex external parameters are set as `[re, im]`; a real parameter can never receive an imaginary part. |
| External defaults | conditional | every external parameter needs a value in the UFO or in the selected restriction card. |
| Expression syntax | conditional | standard UFO spellings only: `cmath.sqrt`, `cmath.pi` or `pi`, `cmath.sin/cos/asin/acos`, `complex(x, y)`, `complexconjugate`, `cond`, `Theta`, `reglog`, bare `exp`, `log`, `tan`, `atan`, and `**`; other `cmath.`, `math.` or `numpy.` calls fail to load. |
| Scientific notation | rejected | a decimal mantissa with an exponent such as `1.5e-3` is mis-parsed by the loader; write `0.0015`, `15e-4` or a fraction. |
| Rational exponents | supported | `x**0.5` is evaluated exactly as `x^(1/2)`. |
| Parameter dependencies | supported | internal parameters may reference each other in any order; cycles are rejected. |
| Function registry | conditional | one-argument `Theta`, `abs`, `acos`, `acosh`, `acsc`, `asec`, `asin`, `asinh`, `atan`, `atanh`, `complexconjugate`, `conj`, `cos`, `cosh`, `csc`, `exp`, `im`, `log`, `log10`, `re`, `reglog`, `reglogm`, `reglogp`, `sec`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`; two-argument `complex`, `pow`; three-argument `cond`, `if`. Anything else is `unknown-functions`. |
| Evaluable subset | conditional | `re`, `im`, `log10`, `pow`, `reglogp` and `reglogm` pass preflight but have no runtime evaluator and fail when a parameter that uses them is evaluated; all other registry functions compute. |
| Declared functions | conditional | a declared `Function` is accepted only if it is a registry name with the registry arity; its body is ignored in favour of the built-in meaning (`function-arity` otherwise). |
| Constants | conditional | `pi` stays exact until numerical evaluation and `complex(0, 1)` is the exact imaginary unit; a bare `I` symbol is not supported. |
| Epsilon-expanded values | ignored | dictionary-valued parameters and couplings keep only the order-0 entry, with one loader warning. |
| Counterterms and decays | ignored | counterterm parameters are ordinary parameters, `CTVertex` objects are dropped, the decays block is never read. |
| Coupling records | supported | any expression in the parameters, any number of coupling-order names per coupling. |
| Coupling matrix | supported | one row per colour structure and one column per Lorentz structure; `None` entries are skipped; each cell becomes its own term and pieces with different orders are counted separately. |
| Coupling orders | conditional | `hierarchy` weights the minimal coupling-order policy; `expansion_order` is stored but has no effect; undeclared order names default to weight 1. |
| Coupling-order policies | supported | `minimal` (default) keeps the lowest hierarchy-weighted order; `explicit` applies only `--max-coupling-order NAME=N`, and misspelled names are not detected. |
| Runtime parameter cards | supported | flat JSON objects of finite real or `[re, im]` values via `--model-parameters`, `Runtime.load(model_parameters=...)` or `set_model_parameters`; an invalid batch leaves the runtime unchanged. |
| Runtime mutability | conditional | only surviving external parameters are mutable; derived parameters and couplings are recomputed and read-only. In recurrence mode an external parameter that the process never uses may be rejected as not used. |
| Masses and widths at runtime | conditional | external mass and width parameters can be varied, but a particle cannot switch between the massless and massive class without regenerating. |
| Numerical precision | supported | model constants are kept exact in the compiled model; only user-supplied values are binary64. |

## Lorentz structures and vertices

| Feature | Status | Condition and consequence |
| --- | --- | --- |
| Normalized tensor heads | supported | `Identity(i, j)`, `Gamma(mu, i, j)`, `Gamma5(i, j)`, `ProjM(i, j)`, `ProjP(i, j)`, `Sigma(mu, nu, i, j)`, `Metric(mu, nu)`, `P(mu, k)` with the standard UFO argument conventions; `PSlash(i, j)` only in its two-argument propagator form. |
| Other tensor heads | rejected | `Epsilon`, `EpsilonBar`, `C`, `IdentityL`, three-argument `PSlash` and any accepted head with a wrong arity fail model compilation with `ValueError: ... contains unsupported UFO tensors`. |
| Form factors | rejected | `form-factors`. |
| Sigma normalisation | experimental | expanded as (i/2)(gamma^mu gamma^nu - gamma^nu gamma^mu); vertices using `Sigma` have not been validated against a reference implementation. |
| Chiral structures | supported | `ProjM`, `ProjP` and `Gamma5` are validated through the packaged Standard Model. |
| Index conventions | conditional | 1-based leg indices, the `1000*component + leg` notation for spin-2 fields and negative dummy indices are supported; inconsistent indices are a `ValueError`. |
| Several structures per vertex | supported | any number of colour and Lorentz structures through the coupling matrix. |
| Three-point vertices | supported | any combination of scalars, Dirac fermions, vectors and spin-2 fields, momentum-dependent or not, with colour 1, delta, T^a or f^abc. |
| One- and two-point terms | ignored | tadpole and two-point mixing vertices are dropped without a warning. |
| Four-point colour-singlet contacts | supported | momentum-independent ones through the exact component-basis split; momentum-dependent ones through balanced contact trees. |
| Four-point two-f gauge contacts | conditional | accepted when written as a numeric factor times `f(a, b, x) * f(x, c, d)` with one shared adjoint index and no momentum dependence; otherwise `unsupported-contact-color-lowering`. |
| Other coloured four-point contacts | rejected | four-fermion operators, delta couplings to singlets, `d^abc` and single-f shapes outside the HEFT family give `unsupported-contact-color-lowering`. |
| Higher-point colour-singlet contacts | supported | any valence, momentum-dependent or not, tested up to ten scalars and five-point spin-2 vertices. |
| Scalar-HEFT contacts | conditional | four-point (8, 8, 8, 1) with colour exactly `f(i, j, k)` and five-point (8, 8, 8, 8, 1) with colour exactly `f(...) * f(...)` sharing one index, without a numeric prefactor in the colour string. |
| Other coloured contacts with five or more legs | rejected | `unsupported-contact-color-lowering`. |
| Four-fermion contacts | experimental | not gated at compile time but not validated; treat as unsupported. |
| Counterterm and loop vertices | ignored | only tree vertices are used; `type` and `loop_particles` attributes are never read. |
| Ghost vertices | ignored | compiled but never used. |
| Custom propagator tensors | conditional | the vertex tensor heads plus `PSlash(i, j)`, `P(1)` and `P(1)^2` for the propagating momentum. |

## Colour structures and colour accuracy

| Feature | Status | Condition and consequence |
| --- | --- | --- |
| Colour group | rejected | one SU(3) with N_c = 3 only; a second group or SU(N) with N other than 3 is not representable. |
| `1` | conditional | accepted for any valence when every leg is colour-neutral; on a vertex with coloured legs it is rejected. |
| `Identity(i, j)` | conditional | both indices must point at coloured legs of dual (or equal adjoint) representations. |
| `T(a, i, j)` | supported | adjoint, fundamental, antifundamental index order. |
| `f(a, b, c)` | supported | three adjoint legs or dummies; a product of two `f` is the only coloured four-point structure. |
| `d(a, b, c)` | rejected | recognised, then refused at compilation as an unsupported trilinear tensor or contact lowering. |
| `Epsilon`, `EpsilonBar`, `K6`, `K6Bar`, `T6` | rejected | pass the preflight name check but fail model compilation with `ValueError: color expression contains unsupported UFO tensors`. |
| Three-point projection | conditional | every three-point colour structure must be exactly proportional to 1, delta, T^a or f^abc with matching leg representations; the constant may be any number. |
| Colour accuracies | supported | `lc`, `nlc` and `full` for any accepted model; the choice is a run-card setting. |
| NLC and full colour weights | supported | exact for any number of quark lines; pure-gluon processes are exact up to 40 gluons. |
| Colour-charge balance | conditional | after crossing, fundamental and antifundamental legs must balance; otherwise the process has no colour plan. |
| Sector truncation | conditional | `process.max_color_sectors` produces a partial sum marked `selected` with direct contraction and is refused by on-the-fly, FFT and correlators. |
| Colour-flow selectors | conditional | LC outputs only; NLC and full outputs return one contracted entry per helicity. |
| `all-flow-union` layout | conditional | LC only, with complete coverage. |
| FFT contraction | conditional | Default `contraction="auto"` tries FFT for `nlc` or `full` with `recurrence` or `on-the-fly`, falling back to direct contraction for unsupported plans. FFT needs at most ten permutable gluons whose colour sectors form complete permutation orbits. LC, compiled/eager execution, and correlators retain direct/trace; `--color-contraction direct` opts out. Explicit `--fft trace` or `--fft adjoint` forces FFT and retains unsupported-plan errors. |
| Adjoint FFT selection | conditional | Default `fft_basis="adjoint"` adaptively uses DDM for certified Yang–Mills or single-insertion scalar HEFT (`HIG = 1`), fundamental chains/products for quarks, and trace tensors otherwise. Saved `fft_basis_selection` reports the actual representation; no universal DDM or speedup claim. Forced adjoint FFT with correlators is rejected; direct contraction ignores the FFT basis. |
| Shared-trace optimisation | conditional | applied only to pure gauge-boson processes whose Yang-Mills structure the model certificate proves; results are exact either way. |
| Correlators | conditional | representations 1, 3, -3, 8 from complete full-colour plans in compiled execution. |

## Propagators and gauge

| Feature | Status | Condition and consequence |
| --- | --- | --- |
| Default versus custom | supported | a UFO propagator counts as custom only if its expression differs from the loader's Feynman default for that particle. |
| Massless vectors | supported | Feynman gauge -i g/p^2, fixed. |
| Massive vectors | supported | unitary gauge with a fixed-width denominator; matched Goldstones are absorbed. |
| Feynman-gauge massive vectors | not applicable | no such kernel exists; Goldstone exchange appears only if the UFO supplies a custom massive-vector propagator, which is then used verbatim. |
| Massless spin-2 | supported | de Donder propagator with dimension parameter `dim` (default 4). |
| Massive spin-2 | experimental | Fierz-Pauli propagator behind `experimental-massive-spin-2`. |
| Widths | supported | fixed-width Breit-Wigner denominators only; zero widths are fine; complex masses are rejected. |
| Mass class | conditional | fixed by the restriction used at generation; runtime cards change values within that form, and setting a massive particle's mass to zero at runtime is not guarded. |
| Custom propagators | conditional | numerator and denominator must both be defined, use only the accepted tensors and registered functions, and lower exactly to the particle's component count; no gauge conversion is applied. |
| Disabled shortcuts | conditional | a custom propagator switches off the Weyl projection, transverse Yang-Mills lowering, Fierz auxiliary currents, reflection certificates and Goldstone removal for that particle. |
| Custom propagator coverage | conditional | plumbed through all execution modes but not exercised end to end by the test suite. |
| Ghost propagators | ignored | none; ghosts never propagate in tree amplitudes. |
| Non-propagating particles | conditional | excluded from external states but not from internal lines. |
| Gauge selection | not applicable | no option selects a gauge; the UFO must supply its own propagator for another choice. |
| Contact auxiliaries | supported | synthesized auxiliary currents carry no propagator and introduce no pole. |

## Process-level scope with a generic model

| Feature | Status | Condition and consequence |
| --- | --- | --- |
| Model inputs at generation | supported | any accepted source kind; restriction and simplification choices apply only when compiling from UFO or JSON. |
| Preflight gate | rejected | a model with any error code cannot generate or enumerate processes. |
| Process shape | supported | `a b > c d ...` with exactly two incoming and at least one outgoing particle; several requests may be joined with `\|`. |
| Loop-induced channels | rejected | tree level only; a channel without tree amplitudes is skipped with a warning inside a multiparticle request and is an error when requested alone. |
| Multiparticle labels | conditional | `all` always exists; `p` and `j` exist only when the model has massless triplet fermions or octet vectors; define your own labels otherwise. |
| Repetition syntax | conditional | write `3*g`; the built-in shorthands `3g` and `[d g]` are not understood by generic expansion. |
| `flavor_scheme` | ignored | no effect on generic models; use multiparticle labels instead. |
| `max_quark_lines` | conditional | bounds the number of external quark-antiquark pairs per subprocess; channels above it are dropped like loop-induced ones. |
| Permutation representatives | supported | one generated representative serves every reordering within the incoming and within the outgoing side. |
| Basis reductions | conditional | trace-reflection folding and the shared single-trace basis are applied only to certified pure Yang-Mills gluon processes; the unreduced exact basis is used otherwise. |
| Execution modes | conditional | `compiled` works from a raw UFO or JSON model; `recurrence` (default), `eager` and `on-the-fly` need a prepared bundle: `pyamplicol model compile MODEL out.pyamplicol-model --backend jit`. |
| Prepared-bundle clamps | conditional | the bundle fixes the backend and code-shaping settings; differing run-card values are overridden with a warning. |
| Backends | supported | `jit`, `asm`, `cpp` for compiled mode; only JIT artifacts support arbitrary precision. |
| Precision | conditional | `precision=16` needs no Symbolica; higher precision needs the Symbolica package and a JIT-backed recurrence, compiled or eager artifact, never on-the-fly. |
| Symbolica at generation | supported | generation from any model needs Symbolica; double-precision evaluation of a finished artifact does not. |
| Helicity selectors | supported | any generated helicity, by stable ID, for every accuracy and mode. |
| On-the-fly | conditional | keeps every selector for runtime, double precision only, one-point warm-up required. |
| Correlated Born evaluations | conditional | compiled full-colour generation with complete coverage; spin replacements on vector legs only. |
| Numerical current reuse | conditional | off by default; `--numerical-current-reuse` enables the certified search for compiled, eager and recurrence generation. |

## Related pages

- [Models and Processes](models-and-processes.md) for the model workflow and
  the `model` commands.
- [Configuration](configuration.md) for the run-card fields referenced above.
- [Release and Support](release-and-support.md) for the validated release
  boundary.
