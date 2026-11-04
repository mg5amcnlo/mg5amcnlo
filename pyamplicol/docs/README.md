---
title: "Home"
nav_order: 1
description: "Generate and evaluate color-ordered scattering amplitudes from Python and native APIs."
---
<!-- SPDX-License-Identifier: 0BSD -->
<p align="center">
  <img src="https://raw.githubusercontent.com/MadGraphTeam/pyamplicol/main/docs/assets/pyamplicol_logo.png" alt="pyAmpliCol" width="680">
</p>

<p align="center">
  <a href="https://pypi.org/project/pyamplicol/"><img src="https://img.shields.io/pypi/v/pyamplicol.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/pyamplicol/"><img src="https://img.shields.io/pypi/pyversions/pyamplicol.svg" alt="Python versions"></a>
  <a href="https://github.com/MadGraphTeam/pyamplicol/actions/workflows/tests.yml"><img src="https://github.com/MadGraphTeam/pyamplicol/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/MadGraphTeam/pyamplicol/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-0BSD-blue.svg" alt="License: 0BSD"></a>
</p>

<p align="center"><strong>Fast color-ordered scattering amplitudes from Python and native APIs.</strong></p>

pyAmpliCol generates portable process artifacts from built-in, JSON, and UFO
models, then evaluates them through a fast Rust runtime. It supports a typed
Python API and CLI, runtime helicity and color-flow selection, and generated
Python, C11, C++17, Fortran 2008, and Rust 2021 interfaces.

## Start here

| I want to… | Read |
| --- | --- |
| follow a physicist-friendly, MadGraph-style walkthrough | [Get started: a gentle walkthrough](user/gentle-walkthrough.md) |
| install pyAmpliCol and verify it | [Installation](user/installation.md) |
| generate and evaluate my first process | [Quick Start](user/quick-start.md) |
| choose a model, process, color approximation, or evaluator | [Configuration](user/configuration.md) and [Models and Processes](user/models-and-processes.md), including the packaged scalar HEFT workflow |
| check whether my UFO model is supported | [UFO Model Coverage](user/ufo-model-coverage.md), the exact list of accepted, conditional and rejected model features |
| call pyAmpliCol from Python | [Python API](user/python-api.md) |
| prepare and evaluate tree-level spin and colour correlations | [Born Correlations](correlators.md) and [Correlation Conventions](correlator-conventions.md), LC/NLC/full colour and grouped Python evaluation |
| use C, C++, Fortran, Rust, or generated Python drivers | [Native APIs](user/native-apis.md) |
| build a UMAMI shared library or integrate with MadSpace | [UMAMI and MadSpace](user/umami-api.md) |
| benchmark or profile a process | [Profiling and Benchmarking](user/profiling-and-benchmarking.md) |
| diagnose an error | [Troubleshooting](user/troubleshooting.md) |

## Install and try it

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install pyamplicol
pyamplicol self-test
```

Generate a small built-in Standard Model process and inspect the artifact:

```console
pyamplicol generate "d d~ > z" ./artifacts/ddbar_z \
  --model built-in-sm
pyamplicol inspect ./artifacts/ddbar_z
```

The generated process appears in the inspection table as:

```text
+---------+-------------+------------------+-------+--------------+---------------+
| default | stable ID   | concrete process | color | helicities  | color outputs |
+---------+-------------+------------------+-------+--------------+---------------+
|    *    | d_dbar_to_z | d d~ > z         | lc    | 12 (6 eval.) | 1             |
+---------+-------------+------------------+-------+--------------+---------------+
```

The CLI uses color automatically on an interactive terminal; the plain capture
above remains readable in logs and documentation.

## One artifact, several interfaces

```text
model + process request
          │
          ▼
generated process artifact
          ├── Python Runtime
          ├── C / C++ / Fortran / Rust APIs
          ├── selectors and arbitrary precision
          └── benchmarking and profiling
```

Ordinary Python evaluation uses the same runtime as the native interfaces:

```python
import math

from pyamplicol import Runtime

runtime = Runtime.load("artifacts/ddbar_z", process="d d~ > z")
energy = 91.188 / 2.0
momenta = [[
    [energy, 0.0, 0.0, energy],
    [energy, 0.0, 0.0, -energy],
    [2.0 * energy, 0.0, 0.0, 0.0],
]]
value = runtime.evaluate(momenta)
resolved = runtime.evaluate_resolved(momenta)
for optimized, explicit in zip(value, resolved.total(), strict=True):
    assert math.isclose(optimized.real, explicit.real, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(optimized.imag, explicit.imag, rel_tol=1e-12, abs_tol=1e-15)
```

Process expressions may reorder particles within the incoming side or within
the outgoing side. pyAmpliCol consistently permutes momenta, helicities, color
flows, and resolved metadata; particles never cross the `>` boundary. See
[Process Selection and Permutations](user/process-selection-and-permutations.md).

## Explore the documentation

- **Using pyAmpliCol:** [Examples Gallery](user/examples-gallery.md), [Command-Line Interface](user/command-line-interface.md), [Runtime and Selectors](user/runtime-and-selectors.md)
- **Generation:** [Models and Processes](user/models-and-processes.md), including scalar HEFT, [Generation Modes and Evaluators](user/generation-modes-and-evaluators.md), and [Process Grouping](user/process-grouping.md) of subprocesses with equal matrix elements
- **Guided workflows:** [LC workloads and execution modes](user/lc-workloads-and-execution-modes.md)
- **Deployment:** [Artifacts and Portability](user/artifacts-and-portability.md), [Release and Platform Support](user/release-and-support.md)
- **Background:** [Architecture Overview](user/architecture-overview.md), [Symbolica and Licensing](user/symbolica-and-licensing.md)

This site is the authoritative user documentation for the current `main`
branch. [Release and Support](user/release-and-support.md) records the published
platform and validation boundary, while public API stability is defined by the
[API contract](development/API_CONTRACT.md).
