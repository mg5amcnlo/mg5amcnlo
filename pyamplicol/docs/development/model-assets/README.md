---
title: "Packaged Model Assets"
nav_order: 7
parent: "Development Documentation"
has_children: true
---
<!-- SPDX-License-Identifier: 0BSD -->
# Packaged Model Assets

This directory records the release classification for the model resources under
`pyamplicol/assets/models`. The machine-readable authority is
`PROVENANCE.toml` in that package directory; `MANIFEST.sha256` is the single
content manifest.

## Selection Boundary

The external-model inventory starts from the 67 tracked files under
`pyAmpliCol/src/pyamplicol/assets/models` at AmpliCol revision
`643bc6f99d7b2249af0a85204768df243e612411`: one historical manifest and 66
payload files. Only the 66 payload Git blobs were selected. The duplicate
`pyAmpliCol/assets/models` tree, ignored files, caches, bytecode, and source
working-tree contents were not used.

The packaged release set contains complete UFO directories and loader JSON
examples for `sm`, `scalars`, and `scalar_gravity`. The historical built-in SM
is code-backed by `pyamplicol/models/builtin.py`; its exact baseline source and
companion symbol-table hashes are recorded as the `built-in-sm` model entry in
`PROVENANCE.toml`. No second serialization of that implementation was invented.

## Deterministic Changes

The scalar UFO uses the historical defaults of three scalar particles and
contact valences three through ten. Host environment variables can no longer
change those values. The scalar-gravity UFO similarly fixes three scalar
particles and removes its unused interaction-option branch, which referenced
an undefined variable when enabled.

JSON models and restriction cards were regenerated with
`ufo-model-loader==0.1.8` at revision
`70ddee6b416f8c8b340e0d087646d77095c5d24b`. Full model serializations use
`restriction="full"`, simplification enabled, and `JSONLook.VERBOSE`.
`sm_wrapped_indices.json` additionally enables Lorentz-index wrapping. The
tracked `scalars_2p_3p.json` variant is reserialized from its own canonical
content and freezes `N_SCALARS=3` with contact valences two and three; it does
not retain a runtime environment option.

The current JSONs also include the numerical parameter and coupling defaults
computed by the loader. Their symbolic expressions and model contents are
unchanged; restriction cards and the frozen scalar variant retain their
existing serialization when regeneration produces identical contents.

Each JSON restriction card is a complete serialization of its matching UFO
`restrict_*.dat` card. Zero-valued parameters are therefore explicit. A
separately supplied sparse JSON card remains an override map: omitted entries
retain the unrestricted model defaults and are never inferred to be zero.

Every changed source and result hash is listed in
[TRANSFORMATIONS.md](TRANSFORMATIONS.md) and in `PROVENANCE.toml`.

## Redistribution Classification

The applicable notices are the root `LICENSE`,
`licenses/GammaLoop-model-assets.txt`, and
`licenses/MadGraph5_aMCatNLO.txt`. The exact loader license is preserved at
`licenses/model-assets/ufo-model-loader-MIT.txt`. The MadGraph notice applies
specifically to `ufo/sm/build_restrict.py`. Loader-generated JSON retains the
terms of its source model; the loader itself is MIT licensed.

All primary source identities point to exact AmpliCol baseline Git objects.
For additional traceability, each record compares that source blob with the
same path at GammaLoop revision
`db79edc84f6a1580decbcc4ede7ea0b1c79d9a08`. Twenty-eight of 66 baseline
payload blobs match that pinned revision exactly. The other 38 are not claimed
as exact GammaLoop-revision content; their exact provenance stops at the
AmpliCol baseline while GammaLoop remains the documented distribution origin.

Under the repository's existing notices this inventory is release-eligible,
so no file is in `nonrelease/`. If a notice is withdrawn or found insufficient,
the affected complete model set must move to an explicitly excluded
`nonrelease/` directory and release tests must fail with that reason. Individual
files must not be silently dropped from a UFO model.

## Integrity Contract

`MANIFEST.sha256` lists every packaged file below the model root except the
manifest itself, including `PROVENANCE.toml`. Paths are sorted POSIX-relative
paths. Release tests require exact set equality, SHA-256 agreement, no symlinks,
no traversal or absolute-path references, complete UFO module sets, JSON
parseability, and `importlib.resources` access.

Raw UFO modules execute Python and must only be loaded from trusted sources.
The serialized JSON forms are preferred for portable workflows.

<!-- Page hierarchy for the MadGraph7 (Sphinx) documentation. -->
```{toctree}
:hidden:

TRANSFORMATIONS
```
