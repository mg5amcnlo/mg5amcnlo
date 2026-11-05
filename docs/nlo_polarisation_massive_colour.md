# Polarised NLO: may a *coloured* particle be polarised in the subtracted regime?

**Answer as shipped: yes if it is massive, no if it is massless.**

This is the closure study that decided it, reproduced verbatim from the run
directory it was written in (`tt_closure_run/results/RECOMMENDATION.md`) so the
evidence lives in git rather than in a scratch directory. The full per-seed
tables it summarises are in that run directory: `SUMMARY_FO.md`,
`SUMMARY_PS.md`, `FO_emitter_table.txt`, `PS_emitter_table.txt`,
`FO_xsec_table.md`, `PS_xsec_table.md` and the `.HwU` histograms.

**What actually shipped is narrower than the recommendation below.** The study
recommends removing the colour check outright for the subtracted QCD regime.
It was instead *narrowed*: the guard now refuses only a **massless** coloured
polarised particle. Every process the study ran had a massive coloured leg (the
top); a massless coloured polarised leg — a gluon, a light quark, and therefore
`p` and `j`, which contain one — was never exercised, and section 5 below is
explicit that a case which was not run should not be relied on. So the refusal
that remains is an evidence gap, not a statement of principle: it can be
narrowed further by anyone who runs the corresponding study, and the natural
one is a massless-emitter closure where a genuine collinear singularity sits on
the polarised leg (the top has none — see section 3).

Both limitations that survive the narrowing are recorded in the guard's own
comment in `madgraph/interface/madgraph_interface.py` and in
`docs/nlo_polarisation_boost_plan.md`, M6.

## A prerequisite the study did not exercise: the real must keep the polarisation

Allowing a *coloured* polarised particle makes it an **FKS emitter**, and until
this branch the FKS real emission threw its polarisation away.
`fks_common.split_leg` built the daughter legs from a bare dict, so they took
the `Leg` default `polarization = []`, and `insert_legs` then *replaced* the
polarised Born leg by that unpolarised daughter. The result was a polarised
Born subtracted against a helicity-summed real.

This was harmless while polarised legs were forced colour-singlet — a singlet
never splits in QCD, so it stayed a spectator that the `deepcopy` preserved —
and `p p > t t~ [QCD]`, the process this study ran, never showed it either:
`find_reals` enumerates the initial-state splittings first and reals are
deduplicated on **PDGs alone**, so the surviving amplitude is the ISR one, in
which the top *is* a spectator and does keep its polarisation. The FSR
configurations merge onto it. The study's numbers are therefore valid, but they
never exercised the broken path.

A process with **no ISR QCD splitting giving the same real PDGs** does exercise
it: a leptonic or photonic initial state, or a 1→N decay. On `54dc8a361`,
`e+ e- > t{+} t~ [QCD]` exported a Born `e+ e- > t{R} t~` with `NCOMB = 8`
against a real `e+ e- > t t~ g` with `NCOMB = 32`, and `launch NLO -f` aborted:
`test_ME` (`test_soft_col_limits`, which the launch always runs) reported
`Soft test 1 FAILED. Fraction of failures: 1.00` and the same for `Soft test 2`.

The fix (`fks_common.carry_polarization`) carries the mother's polarisation
onto the daughter **j**, which is the leg the Born's `ij` becomes in the
singular limit. It is only defined when `j` has the mother's identity — true
for `t -> t g`, the only QCD splitting of a massive quark, and *false* for
`g -> q q~`. The function raises rather than silently dropping the polarisation
when that fails, so relaxing the massless refusal cannot reintroduce the bug
quietly. This is a second, independent reason to keep massless coloured
refused, on top of the evidence gap.

The same raise also fires for a coloured particle in the **initial** state,
where the splitting runs backwards (`g -> q(-> Born) q~`) and the polarised
parton is an internal line of the real. That is a topological statement, not a
mass one, so it is a rule of its own: `check_process_format` refuses a
polarised coloured *initial-state* leg whatever its mass, and only a **decay**
(one particle before the `>`, never split by `find_reals`) is exempt. Two
processes that hit it: `b{+} b~ > h [QCD]`, and the more realistic
`e- b{+} > e- b [QCD]` (DIS with a massive `b`) — both die on `6a7717288` with
a raw `FKSProcessError` traceback and are a clean INITIAL-state `InvalidCmd`
here.

After that guard, `carry_polarization`'s raise is unreachable from any accepted
process — which is what a backstop should be — and that is not a property of
the `sm` alone. What the guard leaves through is a **massive** coloured
final-state mother, and for such a mother `find_splittings` returns
`[mother, g]` and nothing else: `find_mothers` keeps only splittings in which
one daughter carries the *mother's own mass symbol*, and the `nsoft >= 1`
filter then forces the other daughter to be a massless coloured particle, i.e.
the gluon. `g -> q q~`, the channel that would break the identity, needs a
massless mother and is refused by rule (b). Scanned exhaustively over every
particle of `sm`, `loop_sm` and `MSSM_SLHA2`, in both flavour-grouping modes:
**zero** massive coloured final-state mothers with a splitting other than
`[self, g]` — the gluino and each squark give exactly one splitting, `[self, g]`,
with no `go -> q sq~` channel surviving. So a BSM model is not an escape hatch
either.

(The soft set itself is *not* gluon-only, contrary to what it looks like under
the default flavour grouping: `find_pert_particles_interactions(model, 'QCD')`
gives `soft_particles = {21, ±81}` when quarks are grouped but `{21, ±1..±4}`
when they are not. That extra freedom only ever produces `g -> q q~`, off a
massless mother.)

Two reals may now share one amplitude only if their PDGs **and** their
per-leg polarisations agree (`FKSRealProcess.pdgs_pols`). For any process
without a polarisation restriction the key is the PDG tuple plus a tuple of
empty tuples, so nothing changes.

This is **defensive**, not a fix for an observable symptom. The obvious way to
collide the old PDG-only key,

```
generate    p p > t{+} t~ [real=QCD]
add process p p > t{-} t~ [real=QCD]
```

does not in fact reach it: all 18 FKS matrix elements are generated (9 `t{R}`
plus 9 `t{L}`, both visible in the log) and then only **3** P directories are
written, all `tRtx` — `t{L}` appears in no file at all. The second
polarisation is eaten earlier, by `FKSHelasProcess.__eq__`
(`madgraph/fks/fks_helas_objects.py`), which compares Borns through
`helas_objects.IdentifyMETag.create_tag`, and `IdentifyMETag.link_from_leg`
does not include `polarization`: the two Borns compare equal and
`add_process` merges one away. That behaviour is identical on the parent
commit and here, it is a separate pre-existing bug tracked elsewhere, and it
is **not** fixed on this branch. LO is unaffected. The `pdgs_pols` key would
have collided had the reals ever got that far, and it becomes load-bearing the
moment the `IdentifyMETag` bug is fixed.

After the fix, `e+ e- > t{+} t~ [QCD]` exports a real `e+ e- > t{R} t~ g` with
`NCOMB = 16` (Born `NCOMB = 8` times the gluon's two helicities), `check_poles`
cancels 20/20, `test_ME`'s two soft tests pass at failure fraction 0.00, and
the run completes (`sigma = 6.678e-02 +- 2.9e-04 pb` at a 1 TeV `e+ e-`
collider, `me_frame = [3,4]`). `p p > t{+} t~ [QCD]` generates the identical set
of reals, with the identical amplitude-sharing pattern, before and after — so
every number in this document stands unchanged.

Note that `check_poles` is **blind** to this bug: the poles are proportional to
the polarised Born on both sides, and it reported the same 20/20 and the same
worst single-pole miscancellation `3.17e-14` before and after the fix.

---

# Should the colour restriction on polarised particles be kept or removed?

## Recommendation: **REMOVE it** for the subtracted NLO regime (QCD), keeping every other guard in place.

Evidence from a five-configuration closure study of `p p > t t~ [QCD]` at
13 TeV, run on `origin/claude/nlo-polarised-boost-assessment-d97794` at
`54dc8a361` with **exactly one** guard bypassed -- the
`if p.get('color') != 1` check inside `if subtracted_boost_ok:` in
`madgraph/interface/madgraph_interface.py`. `check_poles`, `test_ME` and
`test_soft_col_limits` ran exactly as shipped, at shipped tolerances; nothing
was skipped, loosened or re-run to make a result look better.

The top is spin-1/2, so `{+}`/`{-}` on each of `t` and `t~` gives the complete
spin sum in four combinations. Frame: the t t~ rest frame,
`me_frame = [3,4]` -> `FRAME_ID = 24`, verified in `Source/run_card.inc` for
all 24 polarised runs (and 0 for all 12 unpolarised ones).

## 1. The closure holds

| | sum(4) - unpolarised | relative | pull |
|---|---|---|---|
| NLO fixed order, 6 seeds/config | +0.0050 +- 0.196 pb | **+0.0006 %** | **+0.03 sigma** |
| NLO+PS, parton level, 6 seeds/config | -1.66 +- 1.45 pb | -0.19 % | -1.15 sigma |
| NLO+PS, shower level (histogram integrals) | +4.3 to +5.3 pb | +0.49 to +0.62 % | +1.2 to +1.4 sigma |

The fixed-order number is the decisive one: the closure identity is satisfied
to six parts in a million, with a bar of 2 parts in 10 000. The NLO+PS numbers
are statistics-limited and consistent with zero.

Differentially, the ratio panel `sum(4)/unpolarised` sits on 1 across
pT(t), y(t), m(t t~), pT(t t~) and dphi(t, t~), with chi2/ndf of
0.39 / 0.44 / 0.46 / 0.67 / 1.66 (fixed order) and
1.18 / 0.52 / 0.85 / 1.09 / 0.55 (NLO+PS).

## 2. `check_poles` is unaffected by the polarisation

Every P directory of every configuration cancelled **20 points out of 20** at
the shipped tolerance 1.0e-5 (`IRPoleCheckThreshold` untouched). Beyond
pass/fail, the largest relative miscancellation
`|MadFKS - OLP| / max(|MadFKS|,|OLP|)` actually attained over the 20 points:

| configuration | worst double pole | worst single pole |
|---|---|---|
| unpolarised | 6.8e-14 | 2.9e-12 |
| t{+} t~{+} | 3.1e-13 | 2.0e-12 |
| t{+} t~{-} | 7.4e-14 | 4.2e-13 |
| t{-} t~{+} | 7.6e-14 | 5.1e-12 |
| t{-} t~{-} | 1.9e-13 | 1.7e-12 |

All are at double-precision round-off, and the polarised runs are in the same
range as the unpolarised reference -- eleven orders of magnitude inside the
tolerance. Since the poles are proportional to the Born, a polarisation /
subtraction mismatch would have shown here first. It does not.

## 3. `test_soft_col_limits` is unaffected

`test_ME` **is** `test_soft_col_limits` (see
`amcatnlo_run_interface.compile_dir`: both `test_ME` and `test_MC` run
`test_exe='test_soft_col_limits'`), driven by `test_ME_input.txt` = mode 2,
ME/ME(limit), 100 soft and 100 collinear points per FKS configuration.

No test FAILED in any P directory of any configuration. The failure fractions:

| configuration | max soft/collinear failure fraction (gg / uu~ / u~u) |
|---|---|
| unpolarised | 0.03 / 0.02 / 0.01 |
| t{+} t~{+} | 0.00 / 0.05 / 0.04 |
| t{+} t~{-} | 0.02 / 0.02 / 0.01 |
| t{-} t~{+} | 0.02 / 0.02 / 0.01 |
| t{-} t~{-} | 0.00 / 0.05 / 0.04 |

The polarised spread (0.00-0.05) brackets the unpolarised reference
(0.01-0.03): 1-5 points in 100 sit just outside the internal tolerance in both,
which is the ordinary noise of the limit test, not a polarisation effect.

### The case the restriction was written for

The stated worry is that a coloured polarised particle is also an FKS emitter.
Those are exactly the FKS configurations with `j_fks` = the top. They exist and
they were tested -- configurations 3 and 4 in every P directory, `i = 21`
(gluon), `j = 6` / `j = -6`. `j_fks` is massive there, so there is no collinear
singularity and the soft test is the whole test; it is precisely the soft limit
of a gluon emitted off the polarised leg:

```
30 such FKS configurations across the 5 configurations x 3 P directories:
   28 PASSED with failure fraction 0.00
    2 PASSED with failure fraction 0.01  (1 point in 100, in P0_uxu_*)
```
The unpolarised reference scores 0.00 on all six of its own. The two 0.01s are
in `t{+} t~{+}` and `t{-} t~{-}`, and are the same size as fluctuations seen in
the unpolarised run's *other* FKS configurations. Nothing degrades.

### On `test_MC`

`test_MC` also ran (NLO+PS mode only) and reported PASSED with failure fraction
**exactly 0.00** everywhere. That is **not** quoted as evidence: this project
has established that `test_MC` on `main` is structurally vacuous, and these logs
corroborate it -- a test that can fail (`test_ME`) produces 0.01-0.05 on the
same phase-space points, while `test_MC` produces a perfect 0.00 in all 15 P
directories. `test_ME` is what is quoted throughout.

## 4. Why the concern turns out not to bite

The FKS soft/collinear counterterms multiply the **Born** matrix element. The
polarisation projection acts on that Born (and on the real and the virtual)
through the same frame-boost wrappers that the `subtracted_boost_ok` path
already threads for a colourless massive boson. Colour enters the subtraction
only through colour-linked Born matrix elements and the FKS partition, neither
of which touches the spin projection. So the polarisation axis and the singular
regions do not in fact interact: the boost is a fixed Lorentz transformation
determined by the t t~ momentum sum, which is itself infrared-safe here because
the frame is built from two final-state legs whose momenta are unchanged by a
soft emission at leading power. Two legs also mean `boost_to_me_frame` never
runs its `nsel == 1` zeroing, so no leg is left exactly at rest and neither the
HELAS quantisation-axis subtlety nor the `improve_ps` last-leg corruption can
bite.

## 5. What removal should and should not cover

- **Remove** the colour check for the `subtracted_boost_ok` regime as it stands:
  `nlo_mode in ('loonly', 'real', 'all')` **and** `pert_orders == 'qcd'`.
  That is the regime tested here.
- **Keep** the QCD-only restriction. Nothing in the QED sector was exercised;
  the QED counterterms go through the same wrappers but have never been
  validated, polarised or not.
- **Keep** the refusal for other NLO modes.
- A caveat worth stating in the commit message: this study uses a
  **frame built from two final-state legs**. A single-leg frame on a coloured
  polarised particle (`me_frame = [3]` with leg 3 coloured) was *not* tested,
  and it is the configuration that triggers the `nsel == 1` zeroing plus the
  HELAS at-rest axis choice. If the restriction is dropped, that case deserves
  its own check before anyone relies on it.
