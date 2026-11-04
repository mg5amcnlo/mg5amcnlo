<!-- SPDX-License-Identifier: 0BSD -->

# Release Prepared-Model Store

This source-only directory owns the architecture-specific portable JIT O2
prepared-model pairs for both `built-in-sm` and `built-in-sm-heft`. It is
deliberately outside the Python package: contributor builds continue to use the
candidate assets under `src/pyamplicol/assets/prepared_models`.

Generate the release pairs with the manual `release-prepared-models.yml`
workflow, then copy only these eight files from its architecture artifacts into
this directory:

- `built-in-sm-heft-jit-o2-aarch64.metadata.json`
- `built-in-sm-heft-jit-o2-aarch64.pyamplicol-model`
- `built-in-sm-heft-jit-o2-x86_64.metadata.json`
- `built-in-sm-heft-jit-o2-x86_64.pyamplicol-model`
- `built-in-sm-jit-o2-aarch64.metadata.json`
- `built-in-sm-jit-o2-aarch64.pyamplicol-model`
- `built-in-sm-jit-o2-x86_64.metadata.json`
- `built-in-sm-jit-o2-x86_64.pyamplicol-model`

Normal release builds fail until all eight files exist and match the active
release lock. The build overlay projects them into the canonical package asset
directory and removes this auxiliary store before creating a wheel or source
distribution.
