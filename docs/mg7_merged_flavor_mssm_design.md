# mg7/madmatrix merged-flavor support for single-merged-leg vertices (MSSM)

## Status (this PR)

Two things ship here:

1. **Single-merged-leg flavored couplings are now supported** (the MSSM
   topology: one merged fermion + an unmerged partner + a squark/boson), for
   flavor-*independent* couplings. Two parts:
   - **serialization** (`write_flv_couplings`, both copies): a single-leg key is
     written exactly like the Fortran side — the unmerged partner gets flavor
     index 1, i.e. the two-leg formula with `k2 = 1`.
   - **consumer fix** (`MadMatrixALOHAWriter.get_coupling_def`): the flavored
     coupling is selected by the *merged* fermion leg, which (unlike Fortran)
     can be either `F1` or `F2` in the cudacpp argument order — pick whichever
     leg is the partner-populated one.

   Validated: `p p > n1 n1 QCD=0` (t-channel squark, EW/independent couplings)
   now reproduces the Fortran standalone per-flavor |M|² for **all** flavors
   (before the consumer fix only flavor 0 matched). Regression test:
   `test_standalone_mg7_mssm_single_leg`. The two-leg path is unchanged
   (`test_standalone_mg7_vs_cpp` still passes).

2. **A guard for the remaining gap.** `UFOModelConverterCPP.
   _assert_flv_couplings_supported` (called from both `write_flv_couplings`)
   still raises a clear `InvalidCmd` for the cases not yet handled — an
   event-by-event ("dependent", running-αs) flavored coupling, or a vertex with
   >2 merged legs. So `generate p p > go go; output standalone_mg7` still fails
   cleanly (its SUSY-QCD couplings are dependent):
   ```
   InvalidCmd : merged-flavor C++ output (mg7/standalone_mg7) does not yet support
   this process: flavor coupling FLV_54 references an event-by-event (running-alphas)
   coupling. ... Use 'output madevent' or 'output standalone' for this process.
   ```
   The guard is scoped to the process's *used* couplings, so SM two-leg cases
   and single-leg independent cases generate; `output madevent`/`standalone`
   keep working.

**Remaining for full MSSM (`p p > go go`): the dependent-coupling mechanism**
(Step 3 below) — the large piece.

---

The rest of this note records the diagnosis and the remaining plan.

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

So the single-leg work is **(1) serialization [trivial] + (2) a real consumer
fix** that picks the merged fermion's flavor — NOT serialization alone.
**Both are now done** (see the Status section at the top): the consumer selects
whichever fermion leg is the partner-populated (merged) one, and
`p p > n1 n1 QCD=0` matches Fortran for all flavors. The guard now only blocks
the dependent-coupling case (Step 3).

## Plan

1. [done] single-leg serialization in `write_flv_couplings`.
2. [done] single-leg consumer fix in `get_coupling_def` (select by the merged
   fermion leg). Validated on `p p > n1 n1 QCD=0`.
3. **dependent flavored couplings (remaining, the large piece)** — see below.
4. relax the guard for dependent once step 3 is validated.
5. validate `p p > go go` vs the `test_madevent_mssm_gogo` reference; convert
   `test_mssm_gogo_mg7_unsupported` into a positive consistency check.

## Step 3 design: dependent (event-by-event) flavored couplings

Today the cudacpp FLV mechanism (`model_handling.py` ~1591-1660 + the
`CPPProcess.cc` template) **bakes** the per-flavor coupling values into a
constant device array `cIPF_value[nIPF*nMF*2]` at construction
(`tIPF_value[..] = *tFLV[i].value[j]`), and the vertex routines read a value-
based `FLV_COUPLING_VIEW`. This works only for *independent* couplings; a
running-αs coupling like `GC_106` changes per event and cannot be a fixed
pointer/constant (and isn't even in scope in `setIndependentCouplings`).

**Chosen approach — per-event gather, reuse the consumer.** The dependent
couplings are already computed per event into `allcouplings` and exposed in the
kernel as `allCOUPs[idcoup]` (`CD_ACCESS::idcoupAccessBufferConst`). So for a
dependent flavored coupling we keep an `idcoup` per (coupling, flavor) slot and,
**per event page in `calculate_jamps`, gather** the current values into a
`dpf_value[nDPF*nMF*2]` array, then build an ordinary value-based
`FLV_COUPLING_VIEW` over it. The vertex routines and `get_coupling_def` are
**unchanged** (they already consume a value-based view) — this is the key
simplification, and it is the direct analogue of Fortran's `VAL%P => GC(J)`.

Concrete surface:

1. **Split** `couporderflv` into independent (existing `cIPF`) and dependent
   (`cDPF`) flavored couplings (using `coups_flv_dep`).
2. **Model-side** (`model_handling.py`): emit `cDPF_partner1/2[nDPF*nMF]` (as
   today) plus `cDPF_idcoup[nDPF*nMF]` — the `idcoup` of the dependent coupling
   each `value[j]` slot points at (the position of that GC in `coups_dep`;
   `value[j]` is null for unused slots → idcoup `-1`).
3. **Kernel** (`CPPProcess.cc`/`process_function_definitions.inc`): right after
   `allCOUPs` is set up, gather
   `dpf_value[i*nMF+j] = (cDPF_idcoup[i*nMF+j] >= 0) ? COUPs[cDPF_idcoup[i*nMF+j]] : 0`
   (per event / SIMD page), then
   `FLV_COUPLING_ARRAY<nDPF,nMF> flvCOUPs_dep{ cDPF_partner1, cDPF_partner2, dpf_value }`.
4. **Routing** (`model_handling.py` helas-call writer, ~2251-2287): a dependent
   flavored coupling resolves to `flvCOUPs_dep[idx]` instead of `flvCOUPs[idx]`.
5. **Guard**: drop the `is_dep` rejection in `_assert_flv_couplings_supported`.
6. **Validate**: `p p > go go` standalone_mg7 vs the Fortran reference
   (`test_madevent_mssm_gogo`); flip `test_mssm_gogo_mg7_unsupported` to a
   positive consistency test.

Risk: this is the deepest cudacpp codegen change (touches the SIMD/CUDA gather
and the per-coupling routing); needs CPU **and** GPU validation.
