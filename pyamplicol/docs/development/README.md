---
title: "Development Documentation"
nav_order: 7
has_children: true
---
<!-- SPDX-License-Identifier: 0BSD -->
# Development Documentation

This directory contains the stable contracts and decisions that supplement the
source and tests:

- [`API_CONTRACT.md`](API_CONTRACT.md)
- [`CONFIG_CONTRACT.md`](CONFIG_CONTRACT.md)
- [`PACKAGING_CONTRACT.md`](PACKAGING_CONTRACT.md)
- [`PHYSICS_EXTRACTION_CONTRACT.md`](PHYSICS_EXTRACTION_CONTRACT.md)
- [`ARCHITECTURE_DECISIONS.md`](ARCHITECTURE_DECISIONS.md)
- [`ON_THE_FLY_MODE_ARCHITECTURE.md`](ON_THE_FLY_MODE_ARCHITECTURE.md)
- [`MODEL_ASSET_PROVENANCE.md`](MODEL_ASSET_PROVENANCE.md) and
  [`model-assets/`](model-assets/README.md)

Historical feature plans, audit trails, milestone ledgers, and duplicated
machine-readable summaries are deliberately not maintained here. Current
behavior is defined by the implementation, tests, and contracts above.
The on-the-fly architecture page is retained because it records the current
production contract and validation boundary; its historical dispositions are
explicitly labelled as such.

## Linux source-test wheels

The contributor installer and `just source-runtime` build a host-compatible
Linux wheel for local tests, rather than claiming compatibility with an older
manylinux platform. They explicitly pass the backend setting
`--config-setting pyamplicol.host-wheel=true` to `python -m build --wheel`.
Published dependencies are unchanged. Normal release builds still target
`manylinux_2_28` and must run in the appropriate build environment; a host wheel
is not a substitute for the audited release artifact.

## Focused SymJIT dependency tests

The source checkout includes small opt-in numerical/compiler regressions in
`tests/integration/test_symjit_upstream_regressions.py`. They require Python
3.11+, pytest and Cargo, but no pyAmpliCol native build or Symbolica installation.
They are separate from full process-generation and performance tests.
The reproducer bundles in `SYMJIT_FOLLOW_UP_FIXES/` and `DEPENDENCY_FIXES/`
are local, untracked inputs; retain or supply those directories before running
this optional suite. They are not included in a fresh checkout.

```sh
# Published SymJIT 2.25.4; outstanding regressions fail rather than being xfailed.
PYAMPLICOL_RUN_SYMJIT_REGRESSIONS=1 \
  python3 -m pytest tests/integration/test_symjit_upstream_regressions.py -q

# Evaluate a local SymJIT source checkout instead, without modifying it.
PYAMPLICOL_RUN_SYMJIT_REGRESSIONS=1 \
PYAMPLICOL_SYMJIT_SOURCE=/absolute/path/to/symjit \
  python3 -m pytest tests/integration/test_symjit_upstream_regressions.py -q
```

`just symjit-regressions` runs the same suite. The source-selection environment
variable is honoured by either command. ARM-specific tests are skipped on other
architectures. The root `SYMJIT_FOLLOW_UP_FIXES/README.md` maps the checks to the
reported defects and explains the relevant pyAmpliCol evaluation paths.

<!-- Page hierarchy for the MadGraph7 (Sphinx) documentation. -->
```{toctree}
:hidden:

API_CONTRACT
CONFIG_CONTRACT
ARCHITECTURE_DECISIONS
PACKAGING_CONTRACT
PHYSICS_EXTRACTION_CONTRACT
MODEL_ASSET_PROVENANCE
model-assets/README
ON_THE_FLY_MODE_ARCHITECTURE
```
