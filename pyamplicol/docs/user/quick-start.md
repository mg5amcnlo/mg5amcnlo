---
title: "Quick Start"
nav_order: 1
parent: "Installation"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Quick Start

This tutorial follows the primary shipped example: generate a portable
leading-color `p p > Z j j` artifact, select the concrete process
`d d~ > g z g`, evaluate it, inspect its resolved components, and profile the
same runtime path.

## 1. Install and copy the examples

```console
python3 -m venv .venv
. .venv/bin/activate
python -m pip install pyamplicol

pyamplicol examples copy ./pyamplicol-examples --force
cd pyamplicol-examples
```

The copied directory is self-contained: it includes run cards, model resources,
phase-space data, parameter cards, typed Python examples, and native driver
sources.

List the available packaged examples at any time:

```console
pyamplicol examples list
pyamplicol examples run --help
```

## 2. Generate `p p > Z j j`

```console
pyamplicol generate_pp_zjj_from_ufo_sm.toml
```

The run card uses the packaged serialized UFO Standard Model, explicit
two-flavor `p` and `j` multiparticles, leading color, and process-local compiled
JIT O2 execution. The result is written to:

```text
artifacts/pp_zjj/
```

The inclusive request expands to ordered partonic candidates, merges candidates
that differ only by a permutation within the incoming or outgoing side, and
stores one representative per class. The loop-induced `g g > Z g g` candidate
has no supported tree-level amplitude and is reported rather than generated.

> Generation can take appreciably longer than evaluation because it constructs,
> optimizes, validates, and serializes reusable evaluators.

## 3. Inspect the artifact

```console
pyamplicol inspect artifacts/pp_zjj
```

The default terminal report uses colored tables for the artifact, model,
runtime target, concrete processes, aliases, helicities, color flows, and
payload inventory. Select one process for deeper physics metadata:

```console
pyamplicol inspect artifacts/pp_zjj --process 'd d~ > g z g'
```

Machine-readable output is available without changing the artifact:

```console
pyamplicol inspect artifacts/pp_zjj --json > inspection.json
```

## 4. Evaluate totals and resolved components

```console
pyamplicol evaluate_total.toml
pyamplicol evaluate_resolved.toml
```

Both cards load `artifacts/pp_zjj`, apply the UFO parameter card in
`data/model_parameters.json`, and read `data/pp_zjj_momenta.json`.

- `evaluate_total.toml` calls the optimized fully summed runtime path.
- `evaluate_resolved.toml` shows the physical helicity/color components and an
  explicit per-point sum.

The two totals agree. Nonzero matrix elements are displayed in scientific
notation; exact zeros remain `0`. Add `--json` for scripts:

```console
pyamplicol evaluate_total.toml --json
```

## 5. Understand process ordering

The artifact stores a representative process, but the runtime accepts a unique
permutation within each side of the process arrow:

```console
pyamplicol evaluate artifacts/pp_zjj \
  --process 'd d~ > g z g' \
  --momenta data/pp_zjj_momenta.json \
  --model-parameters data/model_parameters.json
```

The input point is interpreted in the exact public ordering written in
`--process`. Rusticol consistently remaps momenta, external-particle metadata,
helicities, LC flows, selectors, reductions, and resolved outputs. No leg may
cross `>`.

If multiple stored representatives could match, selection fails with the
candidate stable IDs instead of guessing. A stable ID such as
`p_p_to_z_j_j_3` is always an unambiguous alternative.

## 6. Profile the runtime

```console
pyamplicol benchmark.toml
```

This short profile uses the same optimized total path, a 128-point batch, two
warmups, at least five measured blocks, and a one-second target. The human
report separates headline wall/evaluator timing from native component
attribution and work counters; rows with exactly zero measured time are omitted.

The direct equivalent is:

```console
pyamplicol profile artifacts/pp_zjj \
  --process 'd d~ > g z g' \
  --momenta data/pp_zjj_momenta.json \
  --target-runtime 1.0 --batch-size 128 --color-flow 1 --precision 16
```

`pyamplicol benchmark` is a compatibility alias for `profile`.

## 7. Use the Python runtime

```python
import json
import math
from pathlib import Path

from pyamplicol import Runtime

points = json.loads(Path("data/pp_zjj_momenta.json").read_text())
runtime = Runtime.load(
    "artifacts/pp_zjj",
    process="d d~ > g z g",
    model_parameters={"aS": 0.117},
)

total = runtime.evaluate(points)
resolved = runtime.evaluate_resolved(points)

print(runtime.physics.process)
print(total[0])
for optimized, explicit in zip(total, resolved.total(), strict=True):
    assert math.isclose(optimized.real, explicit.real, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(optimized.imag, explicit.imag, rel_tol=1e-12, abs_tol=1e-15)
```

The runtime exposes stable selector IDs:

```python
one_component = runtime.evaluate(
    points,
    helicities=[runtime.physics.helicity_ids[0]],
    color_flows=[runtime.physics.color_flow_ids[0]],
)
```

See [Python API](python-api.md) and [Runtime and Selectors](runtime-and-selectors.md)
for typed results, per-point selectors, arbitrary precision, and benchmarking.

## 8. Try a direct built-in-model command

For a smaller single-process example:

```console
pyamplicol generate 'd d~ > z g' artifacts/builtin_ddbar_to_zg \
  --model built-in-sm
pyamplicol inspect artifacts/builtin_ddbar_to_zg
```

The built-in Standard Model automatically uses the wheel-owned prepared JIT O2
kernel pack for the default recurrence execution mode.

## 9. Try generated language APIs

The generated `artifacts/pp_zjj/API/` tree contains standalone Python, C11,
C++17, Fortran 2008, and Rust 2021 drivers. For example:

```console
python artifacts/pp_zjj/API/python/check_standalone.py \
  --process 'd d~ > g z g' --set-parameter aS 0.117 0 --json
```

See [Native APIs](native-apis.md) for all five invocations, custom kinematics, and
SDK discovery.

## Where to go next

- [Examples Gallery](examples-gallery.md) — choose another shipped card.
- [Configuration](configuration.md) — adapt a TOML card or use direct CLI overrides.
- [Models and Processes](models-and-processes.md) — load your own JSON/UFO model.
- [Artifacts and Portability](artifacts-and-portability.md) — understand schema-v3 artifacts, portable compiled JIT O1/O2, and prepared O2 packs.
