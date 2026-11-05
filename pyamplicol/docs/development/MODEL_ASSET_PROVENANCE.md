---
title: "Model Asset Provenance"
nav_order: 6
parent: "Development Documentation"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Model Asset Provenance

Package-owned model assets live in `src/pyamplicol/assets/models`. Their exact
upstream revisions and file checksums are recorded next to the payloads in
`PROVENANCE.toml` and `MANIFEST.sha256`; no duplicate top-level asset tree is
used.

| Asset | Origin | Intended notice | Port status |
|---|---|---|---|
| Built-in SM | Original pyAmpliCol model implementation | 0BSD | Eligible; code-backed implementation is ported separately |
| SM UFO 1.3 | FeynRules UFO by N. Christensen and C. Duhr, distributed through GammaLoop | Preserve authorship, UFO/FeynRules provenance, and all upstream notices | Extracted and manifest-covered |
| Scalar UFO | Model by Valentin Hirschi, generated with FeynRules 1.7.69 and distributed through GammaLoop | 0BSD for original model material; preserve generated UFO helper attribution | Extracted with deterministic static defaults |
| Scalar-gravity UFO | Model by Valentin Hirschi, generated with FeynRules 1.7.69 and distributed through GammaLoop | 0BSD for original model material; preserve generated UFO helper attribution | Extracted with deterministic static defaults |
| Serialized JSON | Deterministic `ufo-model-loader` serialization of the corresponding UFO | Same terms and provenance as source UFO; loader is MIT | Regenerated and manifest-covered |

The SM UFO contains `build_restrict.py` from MadGraph5_aMC@NLO. Its license is
included as `licenses/MadGraph5_aMCatNLO.txt`; model, GammaLoop, and loader
provenance are recorded separately so generated JSON does not obscure the
terms of its source UFO.

The completed extraction:

1. Inventories 66 payload files plus `PROVENANCE.toml` from the pinned source.
2. Excludes caches, interpreter output, symlinks, and host paths.
3. Freezes scalar model environment switches to their historical defaults.
4. Regenerates JSON with `ufo-model-loader` 0.1.8 and records both source and
   package hashes.
5. Records each deterministic transformation in
   `model-assets/TRANSFORMATIONS.md` and verifies all package payloads through
   `MANIFEST.sha256`.

Numerical UFO/JSON/compiled-model parity remains an integration gate owned by
the model compiler rather than an asset-copying claim.
