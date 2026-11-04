---
title: "Examples Gallery"
nav_order: 2
parent: "Installation"
---
<!-- SPDX-License-Identifier: 0BSD -->

# Examples Gallery

The wheel ships runnable TOML cards, Python programs, model/data resources, and
native consumers. Copy them into an ordinary directory before experimenting:

```console
pyamplicol examples copy ./pyamplicol-examples --force
cd pyamplicol-examples
```

Paths inside a card are resolved relative to that card, so the copied workspace
can be moved without referring back to the installation.

> [!IMPORTANT]
> Keep the environment containing pyAmpliCol activated, or invoke its
> executables by explicit path. Generated native drivers discover the SDK
> through `rusticol-config` from that same environment.

## See what is available

```console
pyamplicol examples list
```

The command prints a colored table with each card stem, action, and one-line
description. Use `--json` when a script needs the same inventory as stable,
uncolored machine-readable output:

```console
pyamplicol examples list --json
```

A card can also run in a versioned private cache workspace:

```console
pyamplicol examples run evaluate_total
```

`examples run` accepts the stem shown by `examples list`; it is not an
arbitrary TOML path. Use the copied workspace when you want to edit cards,
inspect artifacts, or run several related steps.

## The four-command tour

The primary sequence generates a portable multiprocess `p p > Z j j` artifact
from the serialized UFO Standard Model, then evaluates and profiles one
subprocess:

```console
pyamplicol generate_pp_zjj_from_ufo_sm.toml
pyamplicol evaluate_total.toml
pyamplicol evaluate_resolved.toml
pyamplicol benchmark.toml
```

What each card demonstrates:

| Card | Demonstration |
| --- | --- |
| `generate_pp_zjj_from_ufo_sm.toml` | JSON model input, multiparticle expansion, compiled JIT O2, API bundle |
| `evaluate_total.toml` | Optimized total, model-parameter card, reordered process expression |
| `evaluate_resolved.toml` | Physical helicity/LC-flow components and explicit sum |
| `benchmark.toml` | One-second calibrated profile of the optimized total |

The generation request finds eight candidates, one per side-permutation class,
stores seven tree-level representatives, and reports the loop-induced
`g g > Z g g` class as omitted. Evaluation selects the public
ordering `d d~ > g z g`; the stored representative is reused automatically.

The three result cards use colored human tables by default. Add `--json` for
machine-readable stdout.

## Run cards by topic

| File | Purpose | Expected cost |
| --- | --- | --- |
| `generate_pp_zjj_from_ufo_sm.toml` | Primary portable multiprocess artifact | Short functional example |
| `evaluate_total.toml` | Fully summed primary subprocess | Fast after generation |
| `evaluate_resolved.toml` | Resolved helicity/color tensor | Fast after generation |
| `benchmark.toml` | Short primary runtime profile | About the configured one-second target plus setup |
| `builtin_sm_lc.toml` | Built-in SM recurrence JIT O2, LC | Small generation example |
| `builtin_sm_nlc.toml` | Built-in SM recurrence JIT O2, contracted NLC | Small generation example |
| `builtin_sm_full.toml` | Built-in SM compiled C++, contracted full color | Requires a C++ toolchain during generation |
| `builtin_sm_adjoint_fft.toml` | Pure-gluon full color in the adjoint DDM FFT basis | Recurrence; also supports an on-the-fly override |
| `builtin_sm_heft.toml` | Packaged scalar HEFT `g g > H g g`, one insertion, recurrence JIT O2 with adjoint DDM FFT | Small HEFT workflow; no external UFO required |
| `builtin_sm_eager.toml` | Built-in SM eager execution with wheel-owned prepared kernels | Small eager example |
| `builtin_sm_on_the_fly.toml` | Built-in SM compact OTF LC artifact | Small generation example; first selected family is built at runtime |
| `otf_pp_zjj.toml` | `p p > Z j j` OTF generation, one-point warm-up, and profiling | Short guided OTF workflow |
| `generate_pp_4j_flavor_blind.toml` | Flavour-blind `p p > j j j j` at `QED = 0`: 23 representatives for 486 subprocesses | About 30 seconds |
| `inspect_pp_4j.toml` | Artifact summary and process-group map | Fast after generation |
| `evaluate_pp_4j_member.toml` | Non-generated member `u~ g > u~ g g g` routed through its representative | Fast after generation |
| `group_pp_4j.toml` | Numerical `group-processes` analysis of the representatives | About a second after generation |
| `process_set_mixed_multiplicity.toml` | Named 2-to-2 and 2-to-3 processes in one artifact | Small multiprocess example |
| `external_ufo_sm.toml` | Trusted UFO-directory loading | Imports trusted model Python |
| `external_json_scalars.toml` | Repeated scalar particles and a contact model | Small external-JSON example |
| `external_json_scalar_gravity.toml` | Proven massless spin-2 path | Small external-JSON example |
| `qq_z6g_recurrence_jit_o2.toml` | `q q~ > Z + 6g`, recurrence prepared JIT O2 | Substantial |
| `qq_z6g_compiled_jit_o3.toml` | Same process, compiled process-local JIT O3 | Substantial and host-specific |
| `qq_z6g_eager_jit_o2.toml` | Same process, eager prepared JIT O2 | Substantial |
| `benchmark_z6g_single_flow_helicity_sum.toml` | Reusable topology-replay selector workload | Substantial |
| `benchmark_z6g_all_flows_single_helicity.toml` | Reusable all-flow-union selector workload | Substantial |
| `benchmark_z6g_generation_specialized_flow_helicity_sum.toml` | Generation-selected flow baseline | Substantial |
| `benchmark_z6g_generation_specialized_all_flows_single_helicity.toml` | Generation-selected helicity baseline | Substantial |
| `all_options.toml` | Exhaustive commented schema reference | Reference only; not runnable via `examples run` |

Use the primary Z+jet sequence, not a six-gluon card, as an installation smoke.

## Packaged scalar HEFT

Generate a full-colour Higgs-plus-two-gluon artifact without an external UFO:

```console
pyamplicol generate --card builtin_sm_heft.toml
pyamplicol inspect artifacts/builtin_sm_heft
```

The card selects the packaged `built-in-sm-heft` model and explicitly limits
the effective coupling order to `HIG = 1`. Recurrence uses the wheel-owned JIT
O2 prepared kernels. See
[Models and Processes](models-and-processes.md#built-in-scalar-heft-model) for
the compiled, eager, on-the-fly, Python API, and trusted-UFO variants.

## Process grouping with `p p > j j j j`

Four cards show [process grouping](process-grouping.md) on the 486
five-flavour subprocesses of pure-QCD `p p > j j j j` in the built-in SM:

```console
pyamplicol generate_pp_4j_flavor_blind.toml
pyamplicol inspect_pp_4j.toml
pyamplicol evaluate_pp_4j_member.toml
pyamplicol group_pp_4j.toml
```

| Step | Equivalent command | Result |
| --- | --- | --- |
| Generate | `pyamplicol generate "p p > j j j j" artifacts/pp_4j_flavor_blind --model built-in-sm --flavor-blind --max-coupling-order QED=0` | 23 representatives at LC; 463 members, 140 of them conjugated, in `process-groups.json`; about 30 seconds |
| Inspect | `pyamplicol inspect artifacts/pp_4j_flavor_blind` | Generated processes and the process-group map |
| Evaluate a member | `pyamplicol evaluate artifacts/pp_4j_flavor_blind --process "u~ g > u~ g g g" --momenta data/pp_4j_momenta.json` | Routed through `d g > d g g g` with relabelling and charge conjugation |
| Group numerically | `pyamplicol group-processes artifacts/pp_4j_flavor_blind` | No new relation: the 23 classes are distinct |

`u~ g > u~ g g g` was never generated, yet its value agrees with a separately
generated copy to about `1e-14`. Any of the 486 subprocesses, in any leg order
within each side, can be selected in the same way, from the CLI, Python, or
the generated API drivers. Flavour-blind generation assumes equal matrix
elements within a class; see
[When to use it](process-grouping.md#when-to-use-it) before using it with
electroweak couplings.

## Three materialized execution modes on one process

The matched `Z + 6g` cards keep model, process, color accuracy, and LC layout
fixed while changing the execution mode:

```console
pyamplicol generate --card qq_z6g_recurrence_jit_o2.toml
pyamplicol profile  --card qq_z6g_recurrence_jit_o2.toml

pyamplicol generate --card qq_z6g_compiled_jit_o3.toml
pyamplicol profile  --card qq_z6g_compiled_jit_o3.toml

pyamplicol generate --card qq_z6g_eager_jit_o2.toml
pyamplicol profile  --card qq_z6g_eager_jit_o2.toml
```

Each writes a separate artifact. These are performance/acceptance workloads,
not quick examples; generation can take significant time and memory.

See [Generation Modes and Evaluators](generation-modes-and-evaluators.md) before
interpreting their differences.

## On-the-fly warm-up and profile

The OTF `p p > Z j j` example keeps generation compact, explicitly warms one
LC flow summed over helicities, and then profiles the retained workload:

```console
pyamplicol generate --card otf_pp_zjj.toml
python python/otf_pp_zjj_warm_up.py
pyamplicol profile --card otf_pp_zjj.toml
```

The Python warm-up takes exactly one double-precision phase-space point and
draws live progress while the selected family is constructed. Its colored
summary reports warm-up time, query counts, resident memory, and the matrix
element. The independent profiler then measures the configured 128-point
steady-state batch. See
[LC workloads and execution modes](lc-workloads-and-execution-modes.md#the-otf-warm-state-lifecycle)
for the cache lifecycle and native-language equivalents.

Completed OTF preparation can also be retained between programs with
`runtime.save(path)` and `runtime.load_cache(path)`. The
[four-gluon save/restore example](runtime-and-selectors.md#saving-an-otf-warm-cache)
shows generation, a complete input point, and a cache-reuse check; the
[native examples](native-apis.md#saving-and-restoring-an-otf-cache) cover every SDK.

## LC selector-layout examples

The two reusable-selector cards retain complete helicity and physical-flow
coverage:

```console
pyamplicol generate --card benchmark_z6g_single_flow_helicity_sum.toml
pyamplicol profile  --card benchmark_z6g_single_flow_helicity_sum.toml

pyamplicol generate --card benchmark_z6g_all_flows_single_helicity.toml
pyamplicol profile  --card benchmark_z6g_all_flows_single_helicity.toml
```

The first uses default `topology-replay`, optimized for one selected flow and a
helicity sum. The second uses `all-flow-union`, optimized for all flows at one
selected helicity. Change selectors without regenerating:

```console
pyamplicol profile \
  --card benchmark_z6g_single_flow_helicity_sum.toml \
  --color-flow 2

pyamplicol profile \
  --card benchmark_z6g_all_flows_single_helicity.toml \
  --helicity h:-1,+1,-1,+1,+1,-1,+1,-1,+1
```

## Typed Python generation

Plan first, then generate:

```console
python python/typed_generation.py artifacts/pp_zjj_typed --plan-only
python python/typed_generation.py artifacts/pp_zjj_typed
```

The script compiles the copied serialized model, constructs explicit `p` and
`j` multiparticle definitions, and calls `Generator.plan()` or
`Generator.generate()`. The generated artifact includes the same standalone API
bundle as a CLI build.

Evaluate through the typed runtime:

```console
python python/runtime_evaluation.py \
  artifacts/pp_zjj data/pp_zjj_momenta.json \
  --process 'd d~ > g z g' \
  --parameters data/model_parameters.json \
  --set-parameter aS=0.1165
```

The JSON result reports the resolved shape and flattened values, explicit
resolved sum, optimized total, and selected physics-axis IDs.

Profile the same process:

```console
python python/benchmark.py artifacts/pp_zjj \
  --process 'd d~ > g z g' \
  --momenta data/pp_zjj_momenta.json
```

`python/external_models.py` demonstrates typed JSON and trusted-UFO
`ModelSource` construction:

```console
python python/external_models.py models/json/sm/sm.json models/ufo/sm
```

## Generated five-language API bundle

After the primary generation, run the same public process ordering in every
generated driver:

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

All drivers load one bundled point, evaluate every resolved component, sum
those components, and compare the sum to the optimized total. They use the
wheel-owned static Rusticol SDK discovered through `rusticol-config`.

Every driver also accepts `--kinematics PATH` and `--model-parameters PATH`.
The full shape, precision, ordering, and precedence rules are documented in
[Native APIs](native-apis.md) and
[Process Selection and Permutations](process-selection-and-permutations.md).

## Hand-written C++ and Fortran consumers

The `native/` directory shows SDK use independent of generated driver sources:

```console
make -C native
native/runtime_cpp artifacts/pp_zjj p_p_to_z_j_j_3 \
  data/model_parameters.json
native/runtime_fortran artifacts/pp_zjj p_p_to_z_j_j_3 \
  data/model_parameters.json
```

Both programs apply an `aS` override, evaluate one five-particle point, and
verify that resolved components reproduce the total.

## Modify an example safely

Prefer a new output directory when changing execution mode, backend, color
accuracy, or flow layout:

```console
pyamplicol generate \
  --card benchmark_z6g_single_flow_helicity_sum.toml \
  --execution-mode eager \
  --set generation.output=artifacts/uubar_z6g_eager_experiment
```

Use `--set generation.mode=replace` only when you explicitly want to replace
an existing artifact transactionally.

## See also

- [Quick Start](quick-start.md)
- [Command-Line Interface](command-line-interface.md)
- [Python API](python-api.md)
- [Native APIs](native-apis.md)
- [Profiling and Benchmarking](profiling-and-benchmarking.md)
