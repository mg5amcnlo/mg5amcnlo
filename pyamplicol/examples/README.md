<!-- SPDX-License-Identifier: 0BSD -->

# Packaged Examples

All TOML cards use schema version 1 and resolve paths relative to themselves.
Create an editable workspace whose model/data assets are independent of the
installation location with:

```console
pyamplicol examples copy ./pyamplicol-examples
cd pyamplicol-examples
```

Keep the environment containing pyAmpliCol activated for top-level CLI and
Python commands below, or invoke its executables by explicit path. Generated
artifact API drivers additionally support the canonical source workflow: when
the copied workspace is below a checkout containing `.venv` from
`just dev-install`, the Python driver and C, C++, Fortran, and Rust Makefiles
find that environment automatically. Explicit `RUSTICOL_CONFIG` and an active
`PATH` still take precedence.

The primary example uses the serialized external Standard Model and generates
seven tree-level representatives for `p p > Z j j`. They cover all 18 ordered
tree-level channels through automatic incoming/outgoing permutations.
Expansion also finds `g g > Z g g`, which is loop-induced and therefore
omitted:

```console
pyamplicol generate_pp_zjj_from_ufo_sm.toml
pyamplicol evaluate_total.toml
pyamplicol evaluate_resolved.toml
pyamplicol benchmark.toml
```

`evaluate_total.toml` selects the concrete channel through the requested public
ordering `d d~ > g z g`. Its representative stable ID is
`p_p_to_z_j_j_3`. Its
parameter card updates the genuine UFO external inputs `aS` and `MZ`.
These three showcase cards print colorized terminal tables by default. Add
`--json` when a machine-readable result is wanted instead.

## Run Cards

| File | Coverage |
| --- | --- |
| `generate_pp_zjj_from_ufo_sm.toml` | Generate a portable compiled-JIT-O2 multiprocess `p p > Z j j` artifact from the serialized UFO SM |
| `evaluate_total.toml` | Optimized total for one `pp_zjj` subprocess |
| `evaluate_resolved.toml` | Helicity/color-resolved evaluation and explicit sum |
| `benchmark.toml` | Short topology-replay benchmark of the same subprocess at one selected flow |
| `qq_z6g_recurrence_jit_o2.toml` | `u u~ > Z + 6g` through the default recurrence schedule and prepared JIT O2 kernels |
| `qq_z6g_compiled_jit_o3.toml` | The same `Z + 6g` workload through process-local compiled-DAG JIT O3 execution |
| `qq_z6g_eager_jit_o2.toml` | The same `Z + 6g` workload through eager-DAG execution and prepared JIT O2 kernels |
| `benchmark_z6g_single_flow_helicity_sum.toml` | Profile the default topology-replay layout for runtime flow selection with a helicity sum in `u u~ > Z + 6g` |
| `benchmark_z6g_all_flows_single_helicity.toml` | Profile the explicit all-flow-union layout for an all-flow sum at one runtime-selected helicity in `u u~ > Z + 6g` |
| `benchmark_z6g_generation_specialized_flow_helicity_sum.toml` | Generation-specialized flow baseline for the reusable-selector comparison |
| `benchmark_z6g_generation_specialized_all_flows_single_helicity.toml` | Generation-specialized helicity baseline for the reusable-selector comparison |
| `process_set_mixed_multiplicity.toml` | Named UFO-SM 2-to-2 and 2-to-3 requests |
| `external_ufo_sm.toml` | Trusted UFO execution path |
| `external_json_scalars.toml` | Scalar contact model and repeated particles |
| `external_json_scalar_gravity.toml` | Proven massless spin-2 model path |
| `builtin_sm_lc.toml` | Built-in compatibility SM, default recurrence JIT O2, `u u~ > g g`, LC |
| `builtin_sm_nlc.toml` | Built-in compatibility SM, default recurrence JIT O2, contracted NLC |
| `builtin_sm_full.toml` | Built-in compatibility SM, explicit compiled C++, contracted full color |
| `builtin_sm_adjoint_fft.toml` | Certified pure-gluon tree, recurrence full color in the adjoint DDM FFT basis |
| `builtin_sm_heft.toml` | Packaged scalar HEFT `g g > H g g`, one insertion, recurrence JIT O2 with adjoint DDM FFT |
| `builtin_sm_eager.toml` | Built-in SM LC generation using the wheel-owned prepared JIT O2 pack |
| `builtin_sm_on_the_fly.toml` | Built-in SM LC generation as a compact on-the-fly seed using the same prepared JIT O2 pack |
| `otf_pp_zjj.toml` | Generate compact OTF LC `p p > Z j j`, then profile one flow with a helicity sum |
| `generate_pp_4j_flavor_blind.toml` | Flavour-blind built-in SM `p p > j j j j` at `QED = 0`, LC: 23 generated representatives for 486 subprocesses |
| `inspect_pp_4j.toml` | Summary of that artifact, including its process-group map |
| `evaluate_pp_4j_member.toml` | Evaluate the non-generated member `u~ g > u~ g g g` through its representative |
| `group_pp_4j.toml` | Numerical `group-processes` analysis of the generated representatives |
| `all_options.toml` | Every current schema field, active and commented |

## Generate Pure Gluons With Adjoint FFT

This is the primary FFT example for certified pure Yang–Mills trees; the card
selects adjoint explicitly rather than relying on the trace configuration
default.

```console
pyamplicol generate --card builtin_sm_adjoint_fft.toml
pyamplicol inspect artifacts/builtin_sm_adjoint_fft
```

The card keeps the same amplitude in the two-anchor DDM basis: for its five
external gluons, it retains `3!` ordered amplitudes instead of the trace
basis's `4!`. To use the original trace FFT, pass `--fft trace`. Either basis
also supports `--execution-mode on-the-fly`; choose a different output path
when comparing artifacts. The adjoint request adaptively selects DDM for
certified Yang–Mills and single-insertion HEFT trees, exact fundamental
chains/products for quarks, and trace tensors otherwise. Inspect saved
`fft_basis_selection` for the actual representation and selection reason.
Correlated generation is not supported by this option. In the benchmarked pure-gluon
family the adjoint basis was faster than trace for six or more external gluons
(3.9x per sample at ten gluons); compare setup and warmed runtime for other
workloads.
The generated Python, C, C++, Fortran, and Rust drivers need no new runtime
option to evaluate it.

## Generate Scalar HEFT Higgs Plus Two Jets

The scalar HEFT card uses the packaged `built-in-sm-heft` model, so it does not
need an external UFO checkout:

```console
pyamplicol generate --card builtin_sm_heft.toml
pyamplicol inspect artifacts/builtin_sm_heft
```

The explicit `HIG = 1` limit retains one effective Higgs-gluon insertion and
allows the certified DDM colour reduction. The four gluons in this example
have `2!` DDM orderings instead of `3!` trace orderings; the Higgs is not
counted as a coloured leg. The artifact uses full-colour recurrence over the
wheel-owned prepared JIT O2 kernels. Use `--fft trace` to compare the original
representation, without assuming a speedup at every multiplicity. See the website's
[Models and Processes](https://madgraph.org/pyamplicol/user/models-and-processes/)
guide for execution-mode and trusted-UFO details.

## Process Grouping With `p p > j j j j`

With five massless quark flavours, `p p > j j j j` expands into 486
subprocesses. In pure QCD they fall into 23 classes whose members share one
matrix element, related by a relabelling of quark flavours, charge
conjugation, and a reordering of the legs within each side. Four cards
generate one representative per class, show the grouping, evaluate a
subprocess that was never generated, and run the numerical analysis:

```console
pyamplicol generate_pp_4j_flavor_blind.toml
pyamplicol inspect_pp_4j.toml
pyamplicol evaluate_pp_4j_member.toml
pyamplicol group_pp_4j.toml
```

The same steps run in the private `examples run` workspace with
`pyamplicol examples run generate_pp_4j_flavor_blind`, then
`inspect_pp_4j`, `evaluate_pp_4j_member`, and `group_pp_4j`.

**Generate.** The card uses the built-in SM with its default five-flavour `p`
and `j`, `process.max_coupling_orders = { QED = 0 }`, and LC recurrence over
the wheel-owned JIT O2 kernels. `process.flavor_blind = true` generates only
the 23 class representatives into `artifacts/pp_4j_flavor_blind`, which takes
about 30 seconds on an Apple M3 Pro laptop. The other 463 subprocesses, 140
of them related by charge conjugation, are recorded in
`artifacts/pp_4j_flavor_blind/process-groups.json` as members of their class.
Equal matrix elements within a class are assumed, not verified. They hold for
massless pure-QCD jets, so this request produces no relabelling warning; with
photon, `Z` or `W` couplings allowed, generation warns about the relations
that may break. The card refuses an existing output; add
`--set generation.mode=replace` to regenerate.

**Inspect.** `inspect_pp_4j.toml` is equivalent to
`pyamplicol inspect artifacts/pp_4j_flavor_blind`. After the 23 generated
processes it shows the process-group map: a summary of the 486 subprocesses,
then one row per representative with the members routed through it, the
charge-conjugated ones marked `C`. Select a representative to list its group
member by member, or a member with `--full-physics` to see how it is routed:

```console
pyamplicol inspect artifacts/pp_4j_flavor_blind --process 'd g > d g g g'
pyamplicol inspect artifacts/pp_4j_flavor_blind \
  --process 'u~ g > u~ g g g' --full-physics
```

**Evaluate a member.** `u~ g > u~ g g g` is not in the artifact. The runtime
finds it in `process-groups.json` and evaluates its representative
`d g > d g g g` (relabelling `u -> d` plus charge conjugation) at the
correspondingly permuted momenta. `data/pp_4j_momenta.json` holds one massless
2-to-4 point at `sqrt(s) = 1 TeV` in the member's leg order. The total,
`7.271103230388e-08`, agrees with a separately generated
`u~ g > u~ g g g` to about `1e-14` relative. Any of the 486 subprocesses, in
any order of the legs within each side, can be selected the same way:

```console
pyamplicol evaluate_pp_4j_member.toml \
  --set 'evaluation.process=u~ u > u~ u g g'
pyamplicol evaluate --card evaluate_pp_4j_member.toml --resolved \
  --helicity 'h:-1,-1,-1,-1,-1,-1'
```

The resolved table lists helicities and LC flows in the member's leg order;
for a conjugated member each flow word is reversed. Without `--helicity` it
lists all 64 helicities times 24 flows. Python selects members
with `Runtime.load("artifacts/pp_4j_flavor_blind", process="u~ g > u~ g g g")`,
whose `representative_process_key`, `process_group_factor` and
`charge_conjugated` describe the routing. The generated API drivers accept
the same selection, for example:

```console
python artifacts/pp_4j_flavor_blind/API/python/check_standalone.py \
  --process 'u~ g > u~ g g g' --kinematics data/pp_4j_momenta.json --json
```

**Group numerically.** `group_pp_4j.toml` is equivalent to
`pyamplicol group-processes artifacts/pp_4j_flavor_blind`. It compares the 23
generated representatives at random points, merges those whose resolved
values agree up to a rational factor, and rewrites `process-groups.json` in
place. The card sets `process.analyze_process_groups = "numerical"` because
run cards otherwise use the schema default `none`; the `group-processes`
subcommand defaults to `numerical`. Here the analysis takes about a second
and reports no new relation: the 23 classes are already distinct. To test
the flavour-blind assumption itself, generate all 486 subprocesses without
`flavor_blind` (much slower) and let the analysis rebuild the same 23 groups;
see the [process grouping guide](https://madgraph.org/pyamplicol/user/process-grouping/).

## Run `q q~ > Z + 6g` With Recurrence, Compiled, And Eager

Each matched card generates and profiles the same complete-selector,
topology-replay LC workload:

```console
pyamplicol generate --card qq_z6g_recurrence_jit_o2.toml
pyamplicol profile --card qq_z6g_recurrence_jit_o2.toml

pyamplicol generate --card qq_z6g_compiled_jit_o3.toml
pyamplicol profile --card qq_z6g_compiled_jit_o3.toml

pyamplicol generate --card qq_z6g_eager_jit_o2.toml
pyamplicol profile --card qq_z6g_eager_jit_o2.toml
```

Recurrence JIT O2 is the default current-schedule lane, compiled JIT O3 builds
process-local DAG evaluators, and eager-DAG JIT O2 applies prepared kernels to
compact process tables. The explicit settings in these cards make the
materialized-mode comparison stable if project defaults change again.

## Generate A Compact On-The-Fly LC Artifact

The dedicated OTF walkthrough expands `p p > Z j j`, loads the concrete
ordering `d d~ > g z g`, warms one LC flow with a complete helicity sum, and
then profiles the same selector workload:

```console
pyamplicol generate --card otf_pp_zjj.toml
python python/otf_pp_zjj_warm_up.py
pyamplicol profile --card otf_pp_zjj.toml
```

The Python step calls
`Runtime.warm_up((point,), precision=16, color_flows=(flow,))`: its input is
exactly one phase-space point, native `f64` is explicit, and omitting a
helicity selector means sum every physical helicity. The standard terminal
progress dashboard tracks query-family progress and live current/peak RSS.
It is followed by a colored PrettyTable containing the selected stable flow
ID, warm-up wall time, query counts, memory, and matrix element. Use
`--color always` when capturing the ANSI-colored table through a pipe, or
`--color-flow N` to choose another one-based physical flow.

The `profile` command starts an independent runtime and prints pyAmpliCol's
colored human benchmark tables. Its 128-point batches are throughput and
benchmark-capacity measurements; they do not change the explicit
`warm_up(...)` API's one-point contract.

To reuse completed OTF preparation in another Python session or native program,
call `runtime.save("process.otf-cache")` on the warmed handle. Load the same
process output normally in the new program, then call
`runtime.load_cache("process.otf-cache")` before evaluating with the same
selectors. This is explicit API functionality: the separate CLI `profile`
command above does not automatically read a saved cache. Small Python and
C/C++/Fortran/Rust examples are in the online manual's
[save/restore guide](https://madgraph.org/pyamplicol/user/runtime-and-selectors/#saving-an-otf-warm-cache)
and [native API guide](https://madgraph.org/pyamplicol/user/native-apis/#saving-and-restoring-an-otf-cache).

The smaller single-process card remains useful for quick inspection and a CLI
profile without multiprocess expansion:

```console
pyamplicol generate --card builtin_sm_on_the_fly.toml
pyamplicol inspect artifacts/builtin_sm_on_the_fly
pyamplicol profile --card builtin_sm_on_the_fly.toml \
  --color-flow 1 --batch-size 128 --target-runtime 5
```

The inspection reports the physical helicity/color-flow census without
materializing either LC flow layout. The card's
`evaluator.optimization.cores = 4` requests four query-construction threads; it
does not promise four numerical-evaluation threads. Point-tile and workspace
settings are not currently applied by this mode. The profile measures one
runtime-selected LC flow with a helicity sum over 128-point batches. Generation
requires the wheel-owned prepared JIT O2 bundle selected for the built-in
model; the artifact carries the referenced kernels it needs at runtime.
The walkthrough intentionally targets the practical OTF workload: LC,
single-flow, helicity-sum execution at native `f64`. Contracted NLC and full
color are available for low multiplicities but are not showcased here because
their OTF construction cost grows quickly; higher-precision OTF execution is
not available.

## Reproduce The Z-Ladder Workloads

These acceptance cards use two separately generated compiled-mode artifacts so
their generation and profile logs remain independent. The first keeps the
default `topology-replay` LC layout, selects flow ordinal `1` only at profile
time, and sums all helicities. The second explicitly sets
`color.lc_flow_layout = "all-flow-union"`, sums all flows, and selects one
stable helicity ID only at profile time. The union shares currents across the
physical flows and is optimized for this second workload.

Both artifacts retain all physical LC flows and all physical helicities. Either
can select any retained flow or helicity globally or per point at runtime;
neither card fixes a selector at generation time. Generate and profile them
independently:

```console
pyamplicol generate --card benchmark_z6g_single_flow_helicity_sum.toml
pyamplicol profile --card benchmark_z6g_single_flow_helicity_sum.toml

pyamplicol generate --card benchmark_z6g_all_flows_single_helicity.toml
pyamplicol profile --card benchmark_z6g_all_flows_single_helicity.toml
```

Both profiles use native Rusticol wall timing, compressed JIT O3, 64-point
batches, two warmups, at least five samples, and a twenty-second target. The
cards use `u u~ > Z g g g g g g` as requested; the PDF's literal reference
family uses `d d~ > Z + (n-1)g`, so replace `u u~` with `d d~` when reproducing
that exact row rather than the equivalent up-quark topology.

The runtime selector can be changed without regenerating. For example:

```console
pyamplicol profile \
  --card benchmark_z6g_single_flow_helicity_sum.toml \
  --color-flow 2

pyamplicol profile \
  --card benchmark_z6g_all_flows_single_helicity.toml \
  --helicity h:-1,+1,-1,+1,+1,-1,+1,-1,+1
```

Python's `Runtime.evaluate(..., color_flow_by_point=..., helicity_by_point=...)`
accepts one physical selector per phase-space point. Homogeneous, alternating,
and randomized selector batches all use the same complete artifact.

The layout is also a dedicated CLI option. For example, this creates a union
artifact from the otherwise topology-replay card:

```console
pyamplicol generate \
  --card benchmark_z6g_single_flow_helicity_sum.toml \
  --lc-flow-layout all-flow-union \
  --set generation.output=artifacts/uubar_z6g_all_flows_override
```

`topology-replay` remains the global default because it is optimized for a
single runtime-selected flow with a helicity sum. `all-flow-union` is LC-only;
configuration rejects it for NLC/full and for LC requests with a
generation-selected color sector, a generation-selected helicity, or truncated
color coverage.

The execution mode can be overridden directly at invocation time. Use a
different output so the eager artifact cannot collide with the compiled one:

```console
pyamplicol generate \
  --card benchmark_z6g_single_flow_helicity_sum.toml \
  --execution-mode eager \
  --set generation.output=artifacts/uubar_z6g_single_flow_helicity_sum_eager
```

The eager override keeps the same complete reusable-selector contract and the
card's LC flow layout. Eager generation and execution apply flow and helicity
choices through Rusticol's selector schedule rather than generation-time
specialization.

`examples copy` also materializes wheel-owned `sm`, `scalars`, and
`scalar_gravity` resources into `models/`. The included
`python/copy_packaged_models.py` helper performs the resource-only operation for
a separate workspace and refuses to merge into a non-empty destination unless
`--force` is supplied. The public model API accepts filesystem paths, so no
example relies on an installation directory.

## Typed Python

Plan or generate the primary process through the typed API:

```console
python python/typed_generation.py artifacts/pp_zjj_typed --plan-only
python python/typed_generation.py artifacts/pp_zjj_typed
```

The script compiles the external JSON model before planning, carries explicit
`p`/`j` definitions in the resolved configuration, and writes the same root
API bundle as the CLI.

Evaluate with a parameter card plus a direct override:

```console
python python/runtime_evaluation.py \
  artifacts/pp_zjj data/pp_zjj_momenta.json \
  --process 'd d~ > g z g' \
  --parameters data/model_parameters.json \
  --set-parameter aS=0.1165
```

The JSON output includes the resolved tensor `shape`, flattened row-major
`values`, its explicit `resolved_sum`, and the optimized
`compatibility_total`, in addition to the selected physics-axis IDs.

Benchmark the selected process:

```console
python python/benchmark.py artifacts/pp_zjj \
  --process 'd d~ > g z g' \
  --momenta data/pp_zjj_momenta.json \
  --color-flow 1
```

The compact OTF counterpart uses the same phase-space point but explicitly
prepares just one LC flow before evaluating it:

```console
python python/otf_pp_zjj_warm_up.py
```

`python/external_models.py` demonstrates explicit JSON and trusted-UFO
`ModelSource` construction:

```console
python python/external_models.py models/json/sm/sm.json models/ufo/sm
```

## Generated Python, C, Rust, C++, And Fortran

Every generated artifact contains one `API/` bundle. All drivers select a
process, accept JSON/direct model-parameter updates, evaluate resolved values,
sum them, and compare with the optimized total:

```console
python artifacts/pp_zjj/API/python/check_standalone.py \
  --process 'd d~ > g z g' --set-parameter aS 0.117 0 --json
make -C artifacts/pp_zjj/API/c run \
  ARGS='--process "d d~ > g z g" --set-parameter aS 0.117 0 --json'
make -C artifacts/pp_zjj/API/rust run \
  ARGS='--process "d d~ > g z g" --set-parameter aS 0.117 0 --precision 16 --json'
make -C artifacts/pp_zjj/API/cpp run \
  ARGS='--process "d d~ > g z g" --set-parameter aS 0.117 0 --json'
make -C artifacts/pp_zjj/API/fortran run \
  ARGS='--process "d d~ > g z g" --set-parameter aS 0.117 0 --json'
```

The requested expression need not be a stored alias. Rusticol identifies a
unique generated representative with the same incoming and outgoing particle
multisets, then applies the side-preserving permutation to input momenta,
helicity and LC-flow selectors, and resolved output metadata. Incoming and
outgoing legs never cross `>`; an ambiguous match fails with the candidate
stable process IDs.

All five drivers also accept `--kinematics PATH`. The JSON file contains one
point as `[external][4]`, or the same point wrapped as a singleton batch
`[[external][4]]`, in the exact particle order written in `--process`. Each
component may be a JSON number or a decimal string. Decimal strings avoid an
f64 round trip in the arbitrary-precision Python driver; native drivers parse
the same representation for f64. Multiple points, non-finite values, booleans,
and mismatched shapes are rejected. Without this option the driver reorders
the artifact's bundled validation point automatically.

`--model-parameters data/model_parameters.json` applies a UFO-style flat JSON
object before evaluation. Each mutable external parameter maps to a finite
number or `[real, imaginary]`, matching a serialized UFO restriction card.
Repeated `--set-parameter NAME REAL IMAG` options are applied afterward and
therefore override the card atomically.

In a copied workspace below a source checkout, this exact block works even in
a later shell where `.venv` has not been activated. Artifacts generated by an
older pyAmpliCol retain their older API files and must be regenerated once:

```console
../.venv/bin/pyamplicol generate_pp_zjj_from_ufo_sm.toml \
  --set generation.mode=replace
```

The generated Rust source includes the wheel-owned safe wrapper located by
`rusticol-config --rust-source` and is compiled directly with `rustc` plus
`rusticol-config --rustflags`; no Rust crate dependency is needed. The Makefile
also has an optional `run-script` target for separately installed
`rust-script`, using `rusticol-config --cargo-rustflags`. C, Rust, C++, and
Fortran support f64 (`--precision 16`) only. At f64, direct SymJIT,
target-compatible ASM/C++, and eager JIT artifacts run without a Symbolica
runtime. The Python driver also exposes precision-controlled Symbolica
evaluation when exact expressions are available.

## Hand-Written Native Examples

`native/runtime.cpp`, `native/runtime.f90`, and `native/Makefile` consume only
the SDK discovered from an installed wheel:

```console
make -C native
native/runtime_cpp artifacts/pp_zjj p_p_to_z_j_j_3 \
  data/model_parameters.json
native/runtime_fortran artifacts/pp_zjj p_p_to_z_j_j_3 \
  data/model_parameters.json
```

Both examples apply a direct `aS` override, evaluate the five-particle
validation point, and verify that resolved components reproduce the total.

Current release validation and upload status are listed in
the [release and support guide](https://madgraph.org/pyamplicol/user/release-and-support/).
