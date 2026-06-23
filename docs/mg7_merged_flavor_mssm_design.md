# mg7/madmatrix merged-flavor support for single-merged-leg vertices (MSSM)

## Resolution (this PR): Option B — clean guard

This PR ships **Option B**: the madmatrix (mg7/standalone_mg7) model export now
**refuses, with a clear and actionable error**, the merged-flavor configurations
it cannot yet generate correctly, instead of crashing or emitting
wrong/uncompilable code. Concretely, `UFOModelConverterCPP` gained
`_assert_flv_couplings_supported`, called from both `write_flv_couplings`
copies (`export_cpp.py` and `madmatrix/model_handling.py`); it raises
`InvalidCmd` when a used flavored coupling either connects a number of merged
legs other than two, or is an event-by-event ("dependent", running-αs) coupling.

Result for `generate p p > go go; output standalone_mg7`:
```
Command "output standalone_mg7 ..." interrupted with error:
InvalidCmd : merged-flavor C++ output (mg7/standalone_mg7) does not yet support
this process: flavor coupling FLV_54 references an event-by-event (running-alphas)
coupling. ... Use 'output madevent' or 'output standalone' for this process.
```
The guard is scoped to the process's *used* couplings (`coups_flv_dep/indep` are
filtered to `wanted_couplings`), so valid SM two-leg merged cases (e.g.
`u u~ > j j QCD=0`) still generate. `output madevent`/`standalone` keep working.

The full feature (Option A) is **deferred**; the diagnosis and plan below are
kept for the follow-up. The rest of this note describes Option A.

---

Status of Option A: **deferred / not implemented**. This note records the
diagnosis, the discovered scope, and the implementation plan, so a follow-up PR
is self-contained.

## Symptom

```
./tests/test_manager.py -pA test_generation_from_file_1 -t 0
```
crashes (MSSM `generate p p > go go`, default `output` = `mg7` = madmatrix):

```
File ".../madmatrix/model_handling.py", line 942, in write_flv_couplings
    k1, k2 = [i for i in key if i!=0]
ValueError: not enough values to unpack (expected 2, got 1)
```

This is **pre-existing on the feat-madmatrix branch** (both `default='mg7'` and
`write_flv_couplings` exist in `9d726705c`); it is not caused by the
`aloha_obj_wmerged` goodhel merge. The Fortran `output madevent` path handles
`p p > go go` fine (it uses legacy flavor grouping, `P1_qq_gogo` has no
`PARTNER`/`flv_index`), so the gap is **madmatrix/mg7-specific**.

## Root cause: the merged-flavor model assumes two merged partner legs

`coupl.flavors` keys are per-leg tuples; a non-zero entry is the 1-based flavor
index of a leg in a merged-particle group, `0` otherwise. The whole FLV
mechanism assumes a flavored vertex has **exactly two** merged legs forming a
partner pair (the SM `q q̄ V` topology), routing flavor `k1 → k2`:

- `write_flv_couplings` (model side) does `k1, k2 = [i for i in key if i!=0]`.
- `MadMatrixALOHAWriter.get_coupling_def` (vertex side, `model_handling.py`
  ~408-526) is hardwired to two fermions `F1`/`F2` and requires
  `partner1[flv_index1] == flv_index2`.
- `FLV_COUPLING { int partner1[]; int partner2[]; cxtype* value[]; }` encodes
  exactly this pairing.

MSSM has vertices with a **single** merged leg. Confirmed by probing the model:

```
INTERACTION id=94  particles: [('_quark',F,81), ('x1+',F,1000024), ('su1',S,1000002)]
   FLV_1 flavors = {(1, 0, 0): 'GC_385'}      # only the quark (leg 0) is merged
```
i.e. `quark(merged) · chargino(unmerged) · squark(scalar)`. The key has one
non-zero entry → the 2-tuple unpack throws.

Semantically a single-leg coupling is a **selection/gate**: "this interaction is
active only when the single merged leg has flavor index k" — there is no second
merged leg to partner with. Each squark flavor gets its own interaction
(su1↔flavor1, su2↔flavor3, ...).

## Discovered scope (larger than the initial framing)

Empirically generating `p p > go go` as `standalone_mg7` after fixing the
unpack reveals **three** independent problems; all are needed for MSSM:

1. **Single-leg topology** (this crash).
   - Model side: `write_flv_couplings` must serialize a one-merged-leg key.
     *(done — see below, for the independent-coupling case.)*
   - Vertex side: `get_coupling_def` (and likely the generic
     `aloha/aloha_writers.py`, plus the Fortran writer) need a single-leg
     branch that gates on the one merged fermion and does **not** require a
     second fermion partner. This needs the writer to know *which* leg is
     flavored (today the `M` tag is binary — `helas_objects.py:2099` — and the
     writer just assumes `F1`/`F2`).

2. **Dependent (event-by-event) flavored couplings.**
   `set_flv_couplings` is emitted inside `Parameters::setIndependentCouplings()`
   and points `FLV_xx.value[k] = &GC_yyy`. Every FLV coupling in the *model* is
   serialized there. MSSM flavored couplings such as `GC_106` are **dependent**
   (running-αs #373 made dependent couplings event-by-event SIMD data in
   `DependentCouplings_sv`), so they are not addressable as a fixed pointer in
   `setIndependentCouplings`:
   ```
   Parameters.cc: error: use of undeclared identifier 'GC_106'
   ```
   The `cxtype* value[]` pointer mechanism is fundamentally incompatible with
   per-event dependent couplings. This needs a different representation — e.g.
   store a flavor → dependent-coupling *index* and have the consumer select the
   per-event dependent coupling by flavor.

3. **Per-flavor couplings are not emitted.** The FLV couplings actually used by
   `p p > go go` (P1_QQx_gogo) are *all* single-leg and reference couplings such
   as `GC_452`, `GC_223`, which are **not generated in `Parameters` at all**, so
   the references would not even link. The merged-flavor model export must emit
   the per-flavor couplings that `value[]` entries point at.

The minimal first crash (#1) was masking #2 and #3.

## Plan (for the Option A follow-up)

1. [done] single-leg serialization in `write_flv_couplings`.
2. dependent flavored couplings (#2): redesign the FLV coupling value storage to
   select a per-event dependent coupling by flavor index (not a fixed pointer).
3. ensure per-flavor couplings (#3) are generated/declared in `Parameters`.
4. single-leg consumer gating (#1 vertex side) in `get_coupling_def` (+ generic
   aloha writers, + Fortran), threading which-leg-is-flavored to the writer.
5. validate `p p > go go`: generate `standalone_mg7`, build, and compare the
   per-flavor |M|² (check_sa.exe matrix mode) against the Fortran
   `output standalone` / `madevent` reference; add a regression test.

## Open questions for review

- Is full MSSM merged-flavor C++ support (items 2-4) in scope now, or should the
  branch ship a clear "unsupported topology" guard (Option B) for models with
  single-merged-leg / dependent flavored couplings while the feature is built?
- Preferred representation for per-event dependent flavored couplings.
- Whether `output` should keep defaulting to `mg7` while these gaps exist
  (separate policy question; `output madevent`/`standalone` already work).
