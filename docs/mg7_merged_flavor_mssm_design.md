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

## How the Fortran/madevent side does it (the template to mirror)

`output madevent`/`standalone` supports `p p > go go`. From the generated
`Source/MODEL/flavor_couplings.f`:

```fortran
TYPE COUPPTR
  DOUBLE COMPLEX, POINTER :: P
END TYPE
TYPE FLV_COUPLING
  INTEGER :: PARTNER(4)
  INTEGER :: PARTNER2(4)
  TYPE(COUPPTR) :: VAL(4)        ! pointer per flavor index
END TYPE
! single merged leg (quark flavor k), unmerged partner (gluino) carries flavor 1:
FLV_56(J)%PARTNER(3)=1 ; FLV_56(J)%PARTNER2(1)=3 ; FLV_56(J)%VAL(3)%P => GC_106(J)
```

Two ideas make it work:

1. **Single-leg = two-leg with an unmerged partner of flavor 1.** Unmerged legs
   carry flavor index 1 (Fortran) / 0 (C++, since `get_flavor_matrix` subtracts
   1). So a single-leg key `(k,0,..)` is serialized exactly like a two-leg one
   with the partner flavor = 1: `partner1[k-1]=0, partner2[0]=k-1, value[k-1]=coup`.
   Each squark is a separate diagram, gated by the merged-quark flavor.
2. **`VAL` is a pointer into the per-event coupling array** (`=> GC_106(J)`), so
   running-αs ("dependent") couplings work uniformly with independent ones.

## Empirical finding: serialization alone is NOT sufficient (consumer is wrong)

An attempt that (a) serialized single-leg as above (`k2 = 1`) and (b) relaxed
the guard to allow single-leg *independent* couplings was validated on
`p p > n1 n1 QCD=0` (electroweak neutralino pair: single merged quark leg +
unmerged neutralino + squark, with **independent** couplings, so the dependent
gap is out of the way). It **compiles and runs** but gives **wrong** per-flavor
|M|² — only the first flavor matches:

| flavor (q q~ > n1 n1) | standalone_mg7 | Fortran standalone |
|---|---|---|
| d d~ (idx 0) | 2.4022949e-05 | 2.4022949e-05  ✓ |
| u u~ (idx 1) | 3.5226867e-07 | 3.8121047e-04  ✗ |
| s s~ (idx 2) | 4.5463454e-07 | 2.4022949e-05  ✗ |
| c c~ (idx 3) | 3.5226867e-07 | 3.8121047e-04  ✗ |

The generated cudacpp **does** emit separate, correctly-mass-ed squark diagrams
(`FFS1M_3(..., cIPD[3]=Msd1, ...)`, `cIPD[5]=Msd4`, `cIPD[9]=Msu1`, …), so the
propagator masses are fine and the goodhel-union filter is not the cause
(re-tested with it applied — no change). The bug is the **consumer gating** in
`MadMatrixALOHAWriter.get_coupling_def` (the `FFSxM` routine): it is hard-wired
to read `F1`/`F2` and assumes the *merged* fermion is `F1`. For these vertices
the unmerged fermion can be `F1` (flv_index 0), so `partner1[flv_index1]` indexes
the wrong slot and only flavor 0 (where `partner1[0]==0`) survives. Fortran
handles this by branching on the fermion *position parity* and using `PARTNER`
vs `PARTNER2` (aloha_writers.py ~757-786); the cudacpp port of that logic does
not correctly cover the single-merged-leg case.

So the single-leg work is **(1) serialization [trivial, done in the attempt] +
(2) a real consumer fix** that picks the merged fermion's flavor (mirroring the
Fortran parity/`PARTNER2` branching) — NOT serialization alone. The attempt was
reverted; the guard still (correctly) blocks single-leg so no wrong physics
ships.

## Plan (for the Option A follow-up)

1. single-leg serialization in `write_flv_couplings` (trivial: unmerged partner
   flavor = 1, i.e. the two-leg formula with `k2 = 1`). Validated structurally
   against `flavor_couplings.f`.
2. **single-leg consumer fix** in `get_coupling_def` (+ the generic
   `aloha_writers.py`): identify and index by the *merged* fermion leg (parity /
   `partner2`), mirroring Fortran. This is the actual correctness fix; validate
   on `p p > n1 n1 QCD=0` against the Fortran per-flavor |M|².
3. dependent flavored couplings: redesign the FLV value storage to select a
   per-event dependent coupling by flavor index (idcoup into the per-event
   `couplings` buffer + a `CD_ACCESS` `FLV_COUPLING` view), the analogue of
   Fortran's `VAL%P => GC(J)`. This is the large piece and unblocks `p p > go go`.
4. relax the guard incrementally (single-leg once step 2 is validated; dependent
   once step 3 is); keep it for any residue.
5. validate `p p > go go`: build `standalone_mg7`, compare per-flavor |M|²
   against the `test_madevent_mssm_gogo` reference; convert
   `test_mssm_gogo_mg7_unsupported` into a positive consistency check.

## Open questions for review

- Is full MSSM merged-flavor C++ support (items 2-4) in scope now, or should the
  branch ship a clear "unsupported topology" guard (Option B) for models with
  single-merged-leg / dependent flavored couplings while the feature is built?
- Preferred representation for per-event dependent flavored couplings.
- Whether `output` should keep defaulting to `mg7` while these gaps exist
  (separate policy question; `output madevent`/`standalone` already work).
