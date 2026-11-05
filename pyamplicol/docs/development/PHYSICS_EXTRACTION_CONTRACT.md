---
title: "Physics Extraction Contract"
nav_order: 5
parent: "Development Documentation"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Physics Extraction Contract

This document records the first independent physics audit. It is normative
for the behavior-preserving extraction from source revision
`643bc6f99d7b2249af0a85204768df243e612411`.

## Port Boundary

Freeze numerical behavior before reorganizing:

- color contraction and planning;
- process IR, enumeration, and all-outgoing crossing;
- current identity and DAG construction;
- lowering, phase-space generation, stage compilation, and Symbolica
  evaluator construction;
- built-in model parameters and kernels;
- UFO model IR, tensor projection, contact decomposition, and propagators.

Rewrite the CLI, artifact writer, loader boundary, logging, reporting, model
asset access, generated API bundle, and Python runtime facade around those
locked contracts. Validation-only Fortran adapters and reference kernels stay
under developer tooling and are excluded from distributions.

The extraction initially preserves both Standard Model implementations. The
built-in path still uses its hard-coded production model, while UFO/JSON uses
the compiled external-model path. Converging those implementations is a
later, separately validated refactor.

## Required Invariants

- UFO and serialized JSON pass through the same `ufo-model-loader` model and
  produce identical restricted, wrapped-index, canonical, and compiled model
  representations.
- External filtering, particle/antiparticle crossing, external order,
  helicity bases, Weyl projection, propagators, spin-2 polarizations, and
  n-ary contact decomposition are preserved.
- A contact coupling is applied exactly once.
- `CurrentIndex` retains every physics identity field. Optimized and
  unoptimized DAGs agree point by point.
- LC flow, reflection, helicity, averaging, coupling, and identical-particle
  weights are each applied exactly once.
- NLC/full performs one coherent sparse color contraction, applies the
  off-diagonal factor two exactly once, and never reapplies an LC color
  factor.
- Inconsistent helicity weights in a contraction group are errors. They must
  never be silently averaged.
- Unsupported colored representations are errors. They must never be treated
  as singlets.
- Final-state permutation reuse is validated against fully materialized
  direct artifacts. It is not described as generic crossing reuse.
- Resolved evaluation executes one DAG. LC exposes physical
  `(point, helicity, flow)` values; NLC/full exposes one contracted color
  component per helicity. Structural zeros remain exact and resolved members
  sum to the compatibility total.
- A color-singlet LC axis may have an empty ordered color word. Every nonempty
  LC word contains exactly once every external label whose model-derived color
  representation is non-singlet.

## Model Determinism

The historical scalar UFO modules read `UFO_SCALARS_MODEL_*` and
`UFO_GRAVITY_MODEL_*` environment variables without including them in the
compiled-model cache key. Standalone model loading must start from a sanitized
environment. Any retained model option becomes a typed input included in
canonical content, provenance, and cache identity. The historical undefined
`N_POINT_INTERACTIONS` gravity branch must be removed or corrected before the
asset is shipped.

## Reference Ladder

The first frozen fixture-v2 bundle contains:

- `d d~ > z`, `d d~ > z g`, and `d d~ > z g g` in LC/NLC/full;
- scalar contact `scalar_0 scalar_0 > scalar_0 scalar_0`;
- spin-2 `scalar_0 scalar_0 > graviton graviton`.

The built-in-SM `d d~ > z` and `d d~ > z g` rows are also replayed through
the pinned Fortran AmpliCol color probe in LC, NLC, and full color. This
independent check covers the summed matrix element and every recorded nonzero
physical-helicity component. The probe is developer-only, links no LHAPDF,
and is never imported or packaged by the installed distribution.

The next declarative capture revision first completes all low-multiplicity
classes before retaining any higher-multiplicity output. It adds `u d~ > w+`,
`d d~ > e- e+`, mixed- and same-flavour two-quark-line scattering,
`g g > g g`, and `g g > t t~` in LC/NLC/full. Built-in masses and exact
Fortran process-row, matching-row, and colour-order counts are independent
capture contracts. The massive top case includes a high-precision threshold
point. These cases become release evidence only after a clean candidate built
from the committed recipe regenerates all fixture and evidence documents.

Before the model/generation milestone is accepted the ladder then expands to:

- process-set `d d~ > z g` and `d d~ > z g g`;
- `d d~ > t t~` and `g g > t t~ g` at the next multiplicity;
- identical-vector/Higgs interference;
- scalar contacts with two, three, and five final scalars;
- scalar propagation with mass, width, and coupling variations;
- two and three final gravitons;
- reordered final states for permutation reuse.

Use at least three distinct deterministic generic points and one quantified
unstable high-precision point for each substantive row. Each point records
the external masses used for its on-shell check, and capture validates both
four-momentum conservation and mass shells. Cross-language agreement checks
one Rust core; release physics additionally requires pinned Fortran, analytic
scalar, exact-color, and independent spin-2 oracles.

Fixture schema v2 must retain decimal-string expectations per point rather
than round-tripping high-precision values through binary64. It records complete
helicity and color axes, every structural zero, topology and coverage, the
complete reduction-group partition, normalization inputs, model and dependency
provenance, and the independent oracle used for each point. Arithmetic
precision, serialization width, and independently certified digits are
distinct metadata; stored values are never claimed beyond the supporting
oracle's certified accuracy. Fast gates may select one generic point from the
complete fixture, but substantive tracked cases cannot weaken their declared
physical axes. Each case hash covers its model dependencies, exact points,
process, and the complete transitive source chain of any process alias.
Compiled-model identity hashes the canonical semantic payload and excludes only
conversion-duration and phase-timing diagnostics, which naturally differ
between otherwise identical color-accuracy builds.

Completeness is not inferred from the emitted axes. Each process records the
external UFO spin code, color representation, mass, and runtime label in model
order. The strict reader reconstructs the physical helicity Cartesian product
and the exact set of labels required in each LC word. Every normalized oracle
record has a recomputable whole-record hash in addition to any raw-transcript
digest. Each reduction plan has a separate canonical hash and must partition
every nonzero physical helicity/color cell exactly once. The three fixture
documents become readable only after the bundle-v1 hash manifest is published
last.

The pinned Fortran adapter follows these additional rules:

- run probes in isolated temporary working directories so the dependency
  checkout remains read-only;
- prefer an exact ordered external-PDG row, and persist the selected group,
  integral, row PDGs, color order, source permutation, and match count;
- identify repeated-PDG legs with stable external IDs and compare every
  physical helicity. A single-flow LC case may be resolved directly; for
  multiple flows the row-wise quadratic-contraction partitions are diagnostic
  only, so the independent Fortran evidence certifies the per-helicity
  aggregate rather than mislabeling those rows as physical resolved cells;
- require every resolved or helicity-aggregate observation of a structural
  zero to be exactly zero, independent of numerical tolerances;
- treat the probe's LC, NLC, and full outputs as cumulative contractions, not
  additive components;
- compare fresh compiler output numerically with the tracked tolerances while
  keeping process identity, axes, command, dependency provenance, structural
  zeros, and capture metadata exact;
- do not require the legacy oracle for three-quark-line cases, which exceed
  the pinned implementation's supported process class.

The next independent built-in-SM ladder proceeds through `d d~ > z`,
`u d~ > w+`, `d d~ > e+ e-`, `d d~ > u u~`, `g g > g g`,
`g g > t t~`, and `d d~ > g g g`. The last case is the smallest planned
check with genuinely distinct LC, NLC, and full contractions.

## Known Pre-Port Limitations

- Existing schema-v2 artifacts have unconfined paths, no payload hashes,
  ambiguous serialized-state fallbacks, nontransactional writes, and Python
  pickle diagnostics. They are rejected, not migrated.
- Resolved topology replay is not proven in the reference runtime. Schema v3
  must not claim that capability until direct expanded-versus-replayed
  resolved values pass.
- The old 180-case cache came from a dirty, older revision and contains
  absolute paths. It is not a release fixture and will be regenerated.
