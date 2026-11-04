---
title: "On-the-fly mode architecture"
nav_order: 8
parent: "Development Documentation"
---
<!-- SPDX-License-Identifier: 0BSD -->

# On-the-fly Mode Architecture and Implementation Record

## Status and scope

This document records the architecture selected from the on-the-fly research
prototypes based on source revision
`a08feed4aacf39c00dfedaeedd3a82a9666f1565` and the production implementation
that followed. The prototype commits are historical evidence, not code that
was merged wholesale.

On-the-fly execution is a distinct artifact and native runtime lane. Python
produces a compact process source projection; Rust authenticates it against the
prepared template/executor catalogs, builds and encodes the binary
`OnTheFlyProcessSeedV1`, and decodes and validates that seed at load. The lane
never constructs a `DirectRecurrencePlan`.

For LC, the native runtime accepts one arbitrary requested Cartesian family of
helicity ordinals times flow ordinals, subject to checked native `usize` sizes
and available resources. That family may range from one selector pair to the
complete all-helicity times all-flow product. Selected-flow/helicity-sum and
all-flow/single-helicity are the two campaign workloads; they are not
capability limits. NLC/full use an authenticated recurrence-v3 colour metric,
evaluate only canonical structural-owner amplitudes, and expose one public
`color:contracted` component rather than a selectable colour-flow axis.

Cold preparation constructs transient selector-local traces and groups their
operations into one executable family. Only the last successfully evaluated
family is retained. The mode is integrated with generation, artifact loading
and inspection, selected/resolved/total evaluation, `Runtime.clear()`,
profiling, the CLI, explicit one-point warm-up progress, and the native SDKs.
It is native-f64-only. Contracted NLC/full correctness is covered through the
complete n<=4 acceptance catalog; their rapidly growing cold selector product
is not claimed to be practical at high multiplicity.

Numerical comparisons now include independent full-colour MadGraph targets at
precision 200 and strict scale-only relative tolerance 1e-10. OTF full colour
is replayed at native f64 against those frozen targets; OTF NLC is replayed
against recurrence-p200 targets captured only after the full-colour lane was
MadGraph-gated. Legacy-only evidence is not treated as independent authority.

## Prototype evidence and dispositions

| Research commit | Evidence | Production disposition |
|---|---|---|
| `ce8037749c97ae04bae6cec42161991754ee2078` | Algorithmic helicity and LC-flow codecs addressed one selector without cardinality-sized tables. | Retain the algorithms and fixtures. Production ordinals and Cartesian products are bounded by native `usize`. |
| `4c9a1d6cedb082140c5b492171bd2fe7873b0f98` | A synthetic plain 8-byte VM was bitwise equal to its interpreter and used zero warm allocations; generic superinstructions were slower. | Historical only. Production executes grouped family rows and has no 8-byte VM. |
| `84f2246f0de8f5d9e598eb76f2ff8e85bd0dbd36` | Ordered prefixes carry distinct color-ordered amplitudes. LC contraction was a diagonal second moment in the checked domains; coherent summation was wrong. | Reject shared-amplitude DP. Retain exact LC selector construction and diagonal reduction. |
| `2312b1137c7e2f3d6bdf9064a902592c16bac58c` | Candidate A built a selected GenericDAG query without global color/projection/axis materialization. Retained n=5 cases agreed with a separate p32 recurrence path. | Port the selector-local construction to Rust. Treat p32 as historical regression evidence, not independent physics authority. |
| `caa500dd7fa67a7e496238e63c6b823aab4b8dd6` | A demand-first memo used synthetic kernels and the wrong ownership boundary. | Reject the evaluator; retain only useful reverse-index ideas. |
| `6b4f011d90d5262807d8316d73f28b106ea31fb9` | Candidate C explored a VM, byte-budgeted LRU, and ephemeral oversize plans. | Historical only. Production uses transactional `last-family-only` retention. |
| `dc171ac357c2817465f20baf3ff0efa91086b53a` | Query-sized structural construction reached n=10 under the 30 GiB watchdog without building the full selector product. | Retain as historical scaling evidence, not a numerical or completion gate. |
| `cf8de3820152be262bafc6f93e5af9a6f423b271` | Compact seeds retained crossed sources, pairing classes, coupling policy, source slots, and parity. | Adopt the seed concept; authenticate and build the production seed in Rust. |
| `e71161cb656ed5eeaa69d18221735b242c60554d` | A prepared pool served compact plans, but the prototype used incomplete traces and hard-coded executor IDs. | Adopt the plan-independent pool seam; use authenticated semantic lookup and a process/lane-owned source binding. |

The production implementation resolves the prototype-era gaps in complete LC
family construction, semantic executor lookup, source-domain ownership, and
the binary seed/load boundary. The VM/LRU proposal and its byte-accounting
gates were not adopted.

## Current production contract

### Binary seed and execution manifest

The Python projection contains process-owned source facts and policy inputs,
but no recurrence DAG, color plan, public-flow table, or direct plan. Rust
cross-checks it with the validated recurrence-template catalog, prepared
direct-executor catalog, and prepared-kernel-pack identity before producing the
PACBIN seed member.

`OnTheFlyProcessSeedV1` stores exactly these classes of data:

- process, compiled-model, recurrence-template-catalog,
  prepared-kernel-pack, and direct-template-catalog digests;
- the normalization semantic digest and convention, with seed construction
  requiring the raw-amplitude normalization factor to be exact one;
- source anchors: source/public slots, initial-state flag, LC color role,
  statistics, optional pairing-contract digest, and all concrete source
  states;
- per-state public/source helicities, source/current template identities and
  semantic digests, momentum sign, exact crossing phase, spin/chirality,
  flavour and quantum flow, color-seed proof, wavefunction family,
  particle/antiparticle/self-conjugate orientation, and optional prepared mass
  slot;
- the public-to-construction external permutation;
- coupling-order policy, positive hierarchy weights, and optional explicit
  user hard caps; and
- species pairing classes with balanced fundamental/antifundamental endpoints,
  plus the seed's semantic digest.

The seed does not store enumerated public axes, selector census, reference
color word, reflection policy, executable rows, or the runtime normalization
factors. Those facts have different owners:

- `execution.selector_policy` stores complete LC coverage, the optional
  reference color word, whether trace reflections are folded, and the
  authenticated physical helicity/flow census;
- `execution.runtime_metadata.normalization` stores LC color, averaging,
  identical-particle, global-coupling, and coupling-policy runtime factors;
  the runtime metadata also carries parameters, external legs, masses, and the
  decoded seed identity; and
- the runtime container manifest owns the canonical PACBIN and seed-member
  paths.

The selector adapter derives axes from the decoded seed plus manifest policy.
The normal load/evaluate/profile path does not materialize dense public axis
records. Explicit `.physics` access may populate a separate metadata cache;
that does not change the seed or native load contract.

There is no fallback to ordinary recurrence construction. In particular, OTF
must not call the global color-plan/projection/recurrence builders,
`build_replay_targets`, `finish_program`, or any `DirectRecurrencePlan`
constructor or loader.

### Coupling policy and first cold warm-up

The seed stores policy inputs, not always the effective runtime limits:

- `Explicit` uses the seed's optional hard-cap vector directly and performs no
  minimal-policy topology sweep.
- `Minimal` performs a process-global cold topology sweep. The sweep erases
  public helicity labels while retaining source support, state/spin, flavour
  and quantum ancestry, exact LC color, and fermion-pairing lineage. Explicit
  hard caps constrain that sweep. The hierarchy-minimal viable total orders
  define the effective per-order envelope used by every subsequent query until
  the process preparation is cleared.

Grammar preparation and minimal-policy resolution happen on the first genuinely
cold family preparation, not at artifact load or seed generation. Their result
is process-wide and reused by subsequent families. `Runtime.clear()` removes
the prepared grammar, effective policy, timing, families, and family semantic
bindings, so the next cold evaluation recomputes this warm-up. Generation and
warm-up therefore remain separately reportable timings.

The cold sweep indexes parent currents by source-support size and visits only
size-complementary pairs. Final closure additionally requires a singleton
anchor. These filters preserve the original candidate order and exact colour,
flavour and fermion-pairing checks; they avoid scanning pairs that cannot
possibly form the requested current.

### Selector family, transient traces, and retained rows

The public selector adapter accepts arbitrary subsets of the helicity and flow
axes. It checks the requested Cartesian product in `usize`, collects the
selected indices/IDs and native-internal `DecodedLcQueryV1` requests during
cold preparation, and preserves public order. Requesting both complete axes is
supported by the contract, although it can be expensive.

For each nonzero query, production constructs an
`OnTheFlyStructuralTraceV1`. This is the transient cold IR: it carries
query-local current/color identities, source/contribution/finalization/closure
operations, momentum forms, exact factors, pairing owners, workspace layout,
and structural proof. Prepared-executor parent order is authenticated and bound
before lowering.

The family builder consumes those traces, unions repeated currents, groups
executor rows, and records amplitude destinations. Transient traces are then
dropped; retained row groups keep the semantic keys and member bindings needed
by the executor, while the retained-state census reports zero query-local
traces. Structural-zero requests retain their public placement but need no
executable destination.

Family replacement is transactional. A candidate remains pending until its
first successful evaluation; a failed candidate is discarded without evicting
the last successful family. Successful promotion clears the old family and
retains exactly one selection and family. An executable family has its row
handle, destination map, and matching semantic bindings; a structural-zero
family has none of those executable members. `Runtime.clear()` leaves all
family-state counts at zero.

Compactness means that the artifact/load boundary has no dense recurrence plan
and that retained executable state belongs only to the requested Cartesian
family. It does not promise memory independent of that family's size: an
all-helicity times all-flow request may itself be large.

Warmed momentum preparation traverses contiguous phase-space-point planes.
Source bounds are checked once when the immutable family workspace is prepared;
each point retains the original momentum-term summation order. Only active
lanes are refreshed, with no warmed reallocation.

### Completed-cache snapshots

`NativeRuntime::save(path)` streams the completed retained OTF structure to a
snapshot; `load_cache(path)` restores it against a matching loaded process.
The Python facade and C/C++/Fortran/Rust SDKs expose the same operations and
format. This is independent of explicit warm-up: an ordinary successful
evaluation also leaves saveable state.
Small usage examples are maintained in
[Runtime and Selectors](../user/runtime-and-selectors.md#saving-an-otf-warm-cache)
and [Native APIs](../user/native-apis.md#saving-and-restoring-an-otf-cache).

The snapshot retains process preparation, grouped current rows, exact factors,
momentum forms, semantic executor references, and the public-selection and
amplitude-reduction mappings. Restoring the resolved coupling policy avoids
repeating its process-global topology sweep. Prepared kernels are rebound
from the original process output, while numeric workspaces are created afresh
and the receiving runtime's parameters remain in effect. No machine addresses
or evaluated currents are serialized. Successful restoration replaces the
previous cache transactionally; rejection leaves it unchanged.

Snapshot compatibility is tied to the existing process/seed/prepared-kernel
identities and a snapshot format version. A saved family follows the same
warmed execution path as one constructed in memory. The last-family retention
policy is unchanged: this is neither a history of evicted selections nor a
checkpoint of a partially constructed family.

### Prepared executors, source binding, and checked resources

The prepared executor pool is model/catalog-owned. At artifact load, source
domains are derived once from the process seed and template catalog and bound
to that pool. `OnTheFlySourceDomainBinding` is process/lane-owned and
query-independent; it survives family replacement and `Runtime.clear()`.
Family-specific semantic executor mappings are resolved and committed
transactionally beside the grouped rows.

Ownership order ensures grouped rows and resolved handles disappear before the
process source binding, which in turn disappears before the model-level pool.
Missing, ambiguous, or digest-mismatched semantic mappings fail closed.

Claims about checked resources are intentionally narrow. Selector counts,
Cartesian products, row/workspace offsets, and point/destination products use
checked size arithmetic. Specific large growth points use fallible reservation,
including seed/source projections, selector metadata caches, selected-family
and trace vectors, retained-family slots, and point/amplitude scratch buffers.
This is not a blanket claim that every Rust allocation is fallible.

All expansion-prone development commands run under the 30 GiB process-tree
watchdog.

### Public surface and limitations

OTF participates in the normal Python generation/load/runtime APIs, compact
inspection, explicit one-point warm-up, selected/resolved/total evaluation,
lifecycle clearing, benchmark/profile reporting, CLI workflows, and native
Rust/C/C++/Fortran APIs. The profile path reports generation separately from
generation plus first warm-up, and it records the native retained-family
census.

The runtime rejects non-f64 precision before expensive selector/physics work.
LC exposes flow selection. Contracted NLC/full reject flow selectors, apply the
authenticated colour factor exactly once, and reuse the same structural
query-family executor. External UFO-SM evidence includes the complete bounded
n<=4 acceptance catalog; this remains a bounded claim, not universal UFO
correctness.

## Original staged plan and dispositions (historical)

The original plan separated a structural interpreter, an optional 8-byte VM,
a bounded LRU, selector streaming, a compact artifact, and final public/UFO
integration into successive gates. Its useful dispositions are:

- the compact seed, transient structural trace, semantic lookup, distinct
  artifact/lane, and LC diagonal reduction were adopted;
- the 8-byte VM, general LRU, ephemeral oversize-plan policy, and exact cache
  byte-accounting gates were not adopted;
- the production family API permits any requested helicity×flow Cartesian
  product and may collect its requests during cold preparation, so strict
  one-selector-at-a-time streaming is not a current gate;
- the old requirement for both compiled and recurrence authorities was
  superseded by one designated legacy authority per cell; and
- broad UFO, p32, and exact n=10 gates were superseded by the bounded current
  evidence and completion scope below.

The deleted prototype-era file map and detailed stage gates described planned
filenames and sequencing, not the implemented contract. The source tree and
the sections above are authoritative.

## Validation evidence and completion scope

Focused tests cover seed/manifest validation, selector order and checked
products, source and semantic binding, transient trace construction, grouped
family rows, structural zeros, transactional failure/promotion, parameter
updates, `Runtime.clear()`, aliases/permutations, identical-fermion signs,
multi-line pairing, LC diagonal reduction, compact inspection, CLI/profile
workflows, and the generated C API. Built-in SM and the external JSON
scalar/contact lifecycle have both been exercised.

The live fail-fast n<=4 process-table campaign completed from native candidate
`238a4ad`: all 66 OTF cells passed their matching recurrence authority, and all
66 legacy AmpliCol timing baselines were available. The campaign produced 132
fresh successful OTF/recurrence results, reused 66 independently rehashed
AmpliCol results, and observed a maximum conditioned OTF/recurrence residual of
`3.93e-15`. This establishes legacy LC parity, not independent physics
correctness.

The later contracted-colour acceptance suite covers the complete bounded
n<=4 catalog. OTF full colour is replayed against frozen independent MadGraph
targets produced at precision 200; OTF NLC is replayed against recurrence-p200
targets admitted only after the matching full-colour lane passed its MadGraph
gate. This is the current bounded NLC/full correctness claim.

Retained selected-flow/helicity-sum high-multiplicity comparisons have passed:

- n=5: process IDs 7, 8, 11, 13, and 15 against legacy AmpliCol;
- n=6: process IDs 7, 8, 11, 13, and 15 against legacy AmpliCol, plus
  four-quark-line process ID 14 against recurrence; and
- n=7: process IDs 7, 11, 13, and 15 against legacy AmpliCol, plus
  four-quark-line process ID 14 against recurrence.

The process-ID-8 pure-gluon n=7 probe reached its one-hour timeout without a
completed report. It is a bounded engineering probe, not a parity failure or
completion gate.

After required parity, scaling exploration and performance investigation each
have a separate one-hour wall-clock cap. Scaling may attempt OTF-only n=8/n=9
selected-flow/helicity-sum and bounded arbitrary-family feasibility under the
30 GiB watchdog; those probes require no numerical authority. Performance uses
batch size 128 and implements only obvious, generic, low-risk improvements
found within the hour. The former 2x AmpliCol aspiration remains context, not a
completion gate.

## Original LC milestone definition of complete

The LC OTF milestone is complete when:

- the final live n<=4 campaign and designated bounded parity cells pass;
- the relevant source-tree and installed-package tests, API/CLI/profile/native
  lifecycle checks, and generated C API test pass;
- the separate one-hour scaling and performance investigations conclude,
  whether or not they find an improvement; and
- the intentional dirty worktree is preserved and the implementation is
  committed and pushed.

That original LC-only milestone did not require a p32 comparison, n=10
reproduction, 2x runtime ratio, NLC/full-color OTF, or repair of shared legacy
physics bugs. The production scope has since added the bounded n<=4 contracted
NLC/full capability and independent full-colour MadGraph gate described above;
it still makes no high-multiplicity practicality claim for those families.
