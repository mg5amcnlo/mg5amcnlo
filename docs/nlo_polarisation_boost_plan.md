# Polarised cross-sections at NLO: frame-boost implementation plan

This document records the assessment of the existing (dead) polarised-NLO code
and the plan to make `p p > z{0} z{0} j [QCD]` work.

**It works.** `check_poles` cancels 20 of 20 points in all 12 P dirs and
`calculate_xsect NLO` gives 4.779e-01 +- 3.3e-03 pb (`me_frame=[3]`, nn23lo1,
fixed scales 91.188, ptj 30, etaj 4.0). See M3 for the gate results.

Status:

| milestone | state |
|---|---|
| M0 plumbing | **done** — run_card option, frame bookkeeping, boost routine, all inert |
| M1 `[LOonly=QCD]` | **done** — Born boosted, validated against LO madevent |
| M2 step 0 | **done** — ISR and FSR emission azimuth carried covariantly |
| M2 rest | **done** — reals + counterterms boosted, azimuthal wiring in, `test_soft_col_limits` passes, `[real=QCD]` enabled |
| M3 `[QCD]` | **done** — `check_poles` cancels 20/20 in every P dir, `calculate_xsect NLO` runs end to end, `[QCD]` enabled |
| M4 | **done** — guard open, `help_polarization` rewritten, acceptance test written, run and wired into CI |
| M5 NLO+PS | **done** — MC counterterm azimuth boosted, `generate_events` validated, second acceptance test in CI |
| M6 regimes + colour | **done** — the three NLO regimes checked separately; coloured polarisation allowed when *massive*, on a measured `p p > t t~ [QCD]` closure |

`[LOonly=QCD]`, `[real=QCD]`, `[QCD]` and `[virt=…]` all accept a polarised
massive particle today. The first three get their frame from `me_frame` in the
run_card; `[virt=…]` is standalone MadLoop, where the user supplies the
phase-space point and therefore picks the frame themselves (see M3).

**A polarised *coloured* particle is allowed too, if it is massive**:
`p p > t{+} t~{-} [QCD]` works, validated by a five-configuration closure
study (`+0.03 sigma`, `check_poles` 20/20 everywhere, `test_ME` clean on the
FKS configurations whose emitter *is* the polarised top). A **massless**
coloured one — a gluon, a light quark, and hence `p` and `j` — is still
refused, because it was never tested. See M6 and
`docs/nlo_polarisation_massive_colour.md`.

**NLO+PS works too**, not only fixed order: `generate_events` on
`p p > z{0} j [QCD]` with `me_frame=[3]` gives 2.189e+03 +- 8.2e+00 pb with all
of `test_ME`, `test_MC` and `check_poles` (20/20 in every P dir) passing. See
M5 for what had to be fixed and for the number that actually detects it.

Fixed along the way, and **live at LO on its own**: the `me_frame` boost left
the selected leg at rest only to rounding, and HELAS switches quantisation axis
at exactly zero. See M2 step 2. It shifts the shipped LO polarised
cross-section by 1.8% (5 sigma).

## 1. Assessment of the existing implementation

This section records the state **before** this work started; the guard it
quotes has since been rewritten (M3/M4).

### The guard

`madgraph/interface/madgraph_interface.py:1240`

```python
if '[' in process and '{' in process:
    if 'noborn' in process or 'sqrvirt' in process: valid = True
    else: raise InvalidCmd('Polarization restriction can not be used for NLO processes')
    # below are the check when [QCD] will be valid for computation
    # if order.strip().lower() != 'qcd':
    #     raise InvalidCmd('...generic NLO computations')
```

The commented-out block at `:1251` is the only authored trace of an intended
`[QCD]` path. The surviving `check(p)` at `:1254` rejects colour-charged **and
massive** particles, so even `p p > z{0} z{0} [noborn=QCD]` is refused today.
The live NLO polarisation path is massless-colourless loop-induced only, which
is exported through the **LO** madevent template and therefore inherits
`me_frame` for free. That is why an NLO boost was never needed.

### Generation is alive and correct

`FKSLeg` inherits `Leg`, so `polarization` survives; `fks_common.py:747`
(`to_leg`) copies it explicitly. With the guard bypassed in-memory,
`p p > z{0} z{0} j [QCD]` generates cleanly — 8 born processes, with `{0}` on
the Z legs of every born **and** every real:

```
born:  d~ g > z{0} z{0} d~        legs: [(-1,[]), (21,[]), (23,[0]), (23,[0]), (-1,[])]
real:  d~ g > z{0} z{0} d~ g      legs: [(-1,[]), (21,[]), (23,[0]), (23,[0]), (-1,[]), (21,[])]
```

The front end is not the blocker. Everything downstream is.

### The frame machinery is LO-only

| piece | location |
|---|---|
| `boost_to_frame` (pure `boostx`, no rotation) | `Template/LO/SubProcesses/genps.f:1761` |
| `common /to_frame_me/frame_id` | `Template/LO/Source/run.inc:27` |
| call sites | `madgraph/iolibs/template_files/auto_dsig_v4.inc:183`, `:452`, `:488` |
| `mapid` (`ids(i)=btest(id,i)`) | `Template/LO/SubProcesses/cluster.f:128` |
| near-duplicate | `MadSpin/src/driver.f:1924` |

`Template/NLO/` has **zero** occurrences of `frame_id`, `me_frame` or
`boost_to_frame`.

### The six checks

| | status |
|---|---|
| Born boost | **No.** `call sborn(p_born,…)` x20 in `fks_singular.f`, always raw partonic-CM momenta |
| Virtual boost | **No.** `Call BinothLHA(p_born,…)` — `fks_singular.f:7087` |
| Real boost | **No.** `call smatrix_real(pp,wgt)` — `fks_singular.f:4701` |
| Counterterm boost | **No**, for any of them — soft `sbornsoft`, collinear `sborncol_isr`/`sborncol_fsr`, soft-collinear, degenerate `sreal_deg`, `bornsoftvirtual`, `extra_cnt`. Each has its own reduced kinematics (`p1_cnt(0,1,0/1/2)`, `p_born`, `p_born_used`) and so needs its own frame |
| Boost before eikonal/splitting kernel | **No — and this is the hard part** (see M2 step 4) |
| run_card option | **No.** `me_frame`/`frame_id` are added in `RunCardLO` (`banner.py:4438-4440`), resolved at `:4869`, and the `frame` display block is appended only from the LO process-dependent setup (`:5191`). `RunCardNLO` (`:5758`) has none of it |

### On rotations

A rotation is **not** needed for the squared MEs. HELAS builds polarisation
vectors in each particle's own momentum-direction helicity basis, so a rotation
`R` gives `eps_lambda(Rp) = R eps_lambda(p) * exp(-i*lambda*dphi)` — a pure
per-particle phase that cancels in `|M|^2` when every external helicity is
fixed. Born, real and virtual are all rotation-invariant.

Where orientation *does* matter is the collinear counterterm azimuthal phases
(`getaziangles`, and the ISR `cphi_mother=1` assumption). That is where the
"the two Z should be on the z axis" concern actually lands.

## 2. Design decisions

### D1 — Where to apply the boost

Options: inside the generated ME wrappers (`SBORN`, `SMATRIX_REAL`, …), or
caller-side in `fks_singular.f`.

**Recommend caller-side.** `SBORN` caches amplitudes against
`savemom(j,1)=p1(0,j), savemom(j,2)=p1(3,j)`
(`madgraph/iolibs/template_files/born_fks.inc:85-97`) and that cache is shared
with `sborn_sf`, `born_hel` and `extra_cnt` via `calculatedBorn`/`saveamp`. If
one entry point boosts and another does not, the cache silently returns
amplitudes from the wrong frame — a bug with no symptom. Boosting once per
kinematic configuration in the caller makes coherence structural, and gives the
azimuthal code access to the same boost vector.

Concretely: one helper `boost_to_me_frame(p, npart, pboost_out, p_out)` in the
NLO template, plus a common block holding the boost vector for the current
configuration.

### D2 — How `me_frame` indexes legs

At LO, run_card positions == fortran positions, so `mapid` is enough. At NLO
this breaks twice:

- `FKSLegList.sort()` (`fks_common.py:793`) reorders born legs relative to the
  user's process, so user leg *n* != born position *n*.
- Born has `nexternal-1` legs, each real has `nexternal`, with the map given by
  `set_pdg` (`Template/NLO/SubProcesses/chooser_functions.f:454-540`): real *k*
  -> born *k* for `k < i_fks`, born *k-1* for `k > i_fks`, and `i_fks` has no
  born counterpart.

**Decided:** keep `me_frame` in **user process numbering**.

Note the premise above was worse than stated: `sort_proc` (`fks_common.py:731`)
*renumbers* the legs (`leg['number'] = n + 1`) after permuting them, so the
user's numbering is not merely reordered, it is destroyed. Verified: the user
writes `p p > j z{0}` and the born comes out `g d > z{0} d` with the Z carrying
`number=3`, not 4. It therefore cannot be recovered downstream and has to be
captured before `sort_proc` runs.

**Implemented in M0** (see the M0 section for status):

- `fks_common.get_user_leg_order()` computes the FKS ordering on a copy and
  returns the pre-sort numbers; `FKSProcess.__init__` calls it *before*
  `sort_proc` and stores `self.user_leg_order`, which `FKSHelasProcess` carries
  to the exporter.
- `frame_info.inc` per P dir holds only `frame_map_born(nexternal-1)`.
- **No `frame_map_real` table is needed.** The FKS convention already fixes the
  real -> underlying-Born correspondence in terms of `i_fks` alone (`set_pdg`,
  `chooser_functions.f:454-540`): real *r* -> Born *r* for `r < i_fks`, Born
  *r-1* for `r > i_fks`, and `i_fks` itself has no Born counterpart. So
  `get_frame_mask_real` derives the real mask at runtime from the Born one.
  This drops the per-FKS-config table the plan originally called for.

### D3 — Smoothness and equivariance

The frame must be a continuous function of the momenta converging in the
singular limits, or the subtraction stops cancelling. "Rest frame of the sum of
the selected legs, recomputed from *each* configuration's own momenta"
satisfies this: as `xi -> 0` / `y -> 1` the Born's ZZ sum -> the real's ZZ sum.

**Do not** compute the frame once from the real and reuse it for the Born — it
looks simpler and is wrong away from the limit.

**The subtler requirement (equivariance).** A pure boost to the rest frame of a
system is *not* equivariant under longitudinal boosts: `B(L_z P) . L_z != B(P)`
whenever `P` has transverse momentum, the difference being a finite rotation.
So if the reduced Born lived in its own partonic CM rather than in the same
"tilde" frame as the real event, the polarisation axes of the two would differ
by an `O(xi)` rotation, the `1/(1-y)` pole would not cancel, and no azimuthal
fix could repair it.

FKS bookkeeping gives us this for free in the ISR case, and it has been
checked: at `y=1` the ISR mapping applies to the final-state spectators only a
transverse boost whose rapidity vanishes identically
(`genps_fks.f:3055-3068`), and `xp` was initialised from the Born momenta
(`genps_fks.f:1206-1222`). In the tilde frame the real event's spectators are
therefore bit-identical to the Born's -> same sum of selected momenta -> same
`me_frame` boost for real and counter-event -> no residual Wigner rotation of
the Z polarisation axes.

**Action:** assert this at runtime (or at minimum comment it) — check that the
boost 4-vector derived from `p_born_used` and from the real `p` agree at the
counter-event. It is cheap and it protects a property the whole subtraction
rests on.

**Not yet confirmed for FSR:** `generate_momenta_massless_final` applies a
non-transverse `boostwdir2(chybst,…)` to spectators
(`genps_fks.f:2324-2339`). Confirm `shybst -> 0` in the collinear limit before
assuming FSR inherits the same property.

**Equivariance is not enough: the frame must not depend on the FKS sector
(B13, 2026-11-01).** The tilde frame makes each sector consistent with its own
counterterms, but it is not the same frame in every sector: for a final-state
emission it is the real's own c.m., for an initial-state one the c.m. of that
sector's underlying Born, a boost along z away. The same real event therefore
had its single-leg spin axis tilted by 5 degrees on average (up to 70) from
one sector to the next in `p p > w+{0} j`, and changing only the FKS
partition (`fks_b` 1.5 -> 3.0 in `fks_powers.inc`, which leaves any
well-defined NLO observable alone) moved the polarised NLO cross-section by
+2.25 +- 0.30% (`w+{0}`) and -0.9 +- 0.3% (`w+{L}`), against -0.14 +- 0.44%
unpolarised. The real's own c.m. is no fix: it is not infrared safe (test_ME
fails the initial-state collinear tests at 73-94%).

The fix is a frame that does not depend on which boost along z the momenta
come in: boost along the beam to zero rapidity of the selected system, then
transversely to its rest frame (`get_me_frame_boosts`,
`Template/Common/Source/impose_frame_rest.f`). For a single leg this is the
Collins-Soper frame. It depends only on the momentum of the selected system,
so it is infrared safe, the same in every sector (partition shift -0.35 +-
0.22%), and the same from the partonic c.m. (madevent), the lab (madspace,
MadSpin, the reweighting) and the FKS frames -- so LO, NLO and the decay and
reweighting tools now share one definition. It changes the single-leg
polarisation fractions: for `p p > w+ j` at 13 TeV (anti-kT 0.4, pT(j) > 30
GeV), f0 goes from 0.212 to 0.250 at LO and from 0.223 to 0.258 at NLO.
Decays keep the single boost from the parent's rest frame: no beam, no
initial-state sector.

**The partonic c.m. (`[1,2]`, the default) -- resolved 2026-11-01.** It had
the same dependence and no fix of this kind, since the c.m. of the real is
exactly what is not infrared safe. `me_frame = [1,2]` was skipped, i.e. the
real was evaluated in whatever tilde frame its sector supplied. Measured,
`fks_b` 1.5 -> 3.0 only, 5 seeds each (15 for WW), `req_acc_FO` 0.005, 13
TeV, nn23nlo, anti-kT 0.4, pT(j) > 30 GeV:

| process, `me_frame = [1,2]` | `fks_b` 1.5 | `fks_b` 3.0 | shift |
|---|---|---|---|
| `p p > w+{0} j [QCD]` | 2588.6(6.0) pb | 2835.5(8.3) pb | **+9.5(4)%** |
| `p p > z{0} j [QCD]` | 1521.6(2.6) pb | 1694.0(6.1) pb | **+11.3(4)%** |
| `p p > w+{0} w- [QCD]` | 17.712(16) pb | 17.716(14) pb | +0.02(12)% |
| `e+ e- > t{L} t~ [QCD]`, 1 TeV | 0.10865(19) pb | 0.10867(25) pb | +0.02(29)% |

The lepton collider has no initial-state sector, so every real comes in its
own c.m. and nothing moves, as expected. The diboson does not move visibly
either, which does not make its frame a definition: the real is still
evaluated in a frame that depends on its sector. For a single boson at rest in the
Born it is worse than a partition dependence: `p p > z{T} [QCD]` with `[1,2]`
fails `test_ME` on 100% of the soft and collinear points, since the real's
spin axis is the direction of a momentum that goes to zero from any
direction.

Candidates:

- **The lab frame.** Infrared safe and the same in every sector (the real's
  lab momenta are physical, and they tend to the Born's), and already mg7's
  default `[]`. But madevent's LO default is the partonic c.m., so LO/NLO
  consistency would change the LO polarised default too (madevent, MadSpin,
  the reweighting): every LO polarised number in the default frame since
  arXiv:1912.01725. Lab helicities also change under a boost along z, so they
  depend on the beam energies, and analyses of diboson polarisation use the
  diboson rest frame, of single bosons the Collins-Soper or helicity frame.
- **Refuse `[1,2]` whenever there is initial-state QCD radiation.** LO is
  unchanged, but every polarised `p p` run with the default card stops,
  including dibosons and single bosons, where an exact continuation exists.
- **Keep it, with a warning.** The result stays a function of an internal
  parameter at the 10% level, and Drell-Yan does not even pass `test_ME`.

**Decision.** LO does not change.

**The default `me_frame` of the NLO run_card depends on the process**
(`fks_common.me_frame_characteristics`, written into the card by
`RunCardNLO.create_default_for_process`):

- **With initial-state QCD radiation (`p p`): the rest frame of the massive
  final-state particles.** For example `[3,4]` for `p p > w+{0} w- [QCD]` (the
  diboson rest frame of the literature), `[3]` for `p p > w+{0} j [QCD]` (the
  Collins-Soper frame of the W), and `[3,4]` for `p p > t{+} t~ j [QCD]`. It
  depends only on the physical momenta of those particles, so it is infrared
  safe and the same in every sector. It is the same for `[LOonly=QCD]`, so the
  multiplicities of a merged sample share one frame.
- **Otherwise (`e+ e-`, decays): `[1,2]`, the partonic c.m.**

`me_frame` is one list for every subprocess of an output, so a leg number is
only part of the default if it is massive in every process that has it.
- With `p p > z{0} j` and `p p > z{0} z` in one output, the default is `[3]`,
  not `[3,4]`, which would put the jet of the first in its frame.
- For `define v = z a; p p > v{T} j`, leg 3 is massless in the photon process.
  It is left out, and the default is `[1,2]`, at LO as at NLO. At LO, keeping
  leg 3 would have given the photon process a single massless particle as its
  frame, which has no rest frame. At NLO no single frame is infrared safe for
  both subprocesses, so `[1,2]` is refused at launch and the user has to
  choose.
- Merged multiplicities are not affected: `p p > w+{0}` has no leg 4, so with
  `p p > w+{0} j` the default stays `[3]`.

The rule is the same at LO and NLO: a multiparticle label at NLO counts as
massive only if all its particles are.

What `[1,2]` means is resolved when the run is launched
(`RunCardNLO.get_me_frame_legs`, from `proc_characteristics`). That covers the
default of `e+ e-` and decays, and anyone who sets it by hand:

- No initial-state QCD splitting (`e+ e-`): unchanged. The tilde frame is the
  partonic c.m. in every sector.
- A decay: the decaying particle, `[1]`. `frame_id` 6 used to select it
  *and* leg 2, the first decay product, so the frame depended on the order
  the final state was written in. At NLO, at matched seeds,
  `t > b w+{0} QED=1 [QCD]` came out 0.04% below `t > w+{0} b QED=1 [QCD]`.
  The two are now bit-identical.
- Initial-state QCD radiation, no massless coloured parton in the Born final
  state: **the rest frame of the Born final state**.
  - For the Born, the virtual and every counter-event this is exactly the
    partonic c.m.; they stay skipped, bit for bit.
  - For the real it depends only on its physical momenta, so it is infrared
    safe and the same in every sector.
  - When the whole final state is massive (dibosons), this is also the default.
  - `p p > w+{0} w- [QCD]` (15 seeds): 18.484(15) pb at `fks_b` 1.5 and
    18.472(19) pb at 3.0, a -0.06(13)% shift.
  - That is **+4.35(13)% above the old `[1,2]` value**. This is a change of
    definition of the real emission; the Born is identical.
  - `p p > z{T} [QCD]` now passes `test_ME`, 49012(199) pb.
- Initial-state QCD radiation and a jet in the Born final state, with `[1,2]`
  set by hand (`p p > w+{0} j`): **refused** at launch. No frame equal to the
  partonic c.m. of the Born is infrared safe: it would have to contain the
  jet, which a final-state collinear splitting changes by a finite amount.
  The message points to the default, e.g. `[3]`.

**Which legs may enter a frame at NLO.** These are the legs whose momentum
no infrared limit changes (`me_frame_unsafe_legs` lists the others):

- Colourless or massive final-state particles.
- The decaying particle of a `1 -> n` process, even a coloured one. FKS never
  splits it (`fks_base.find_reals`), and its momentum is the total momentum
  in every configuration. `me_frame = [1]` used to be refused for a decay as
  "part of the initial state".
- A beam that is not charged under the perturbation, e.g. `e+ e-` with
  `[QCD]`. `[1,3]` on `e+ e- > t{L} t~ [QCD]` used to be refused for mixing
  initial and final legs.

Refused are a beam that radiates (the partons of `p p`) and a massless
coloured final-state parton. Without process information (a card edited
outside a launch) the old blanket rule still applies. In the same way, any
`me_frame` containing a massless coloured final-state parton is refused. `[3,4]` on `p p > w+{0} j` used to pass and
built the rest frame of the W and `j_fks`. Runs of the Born alone (`launch
LO`, `aMC@LO`) refuse nothing.

**LO follows (same rule, 2026-11-01).** madevent and mg7 use the same
default (`banner.default_me_frame_for_process`), so LO and NLO defaults agree
and a K-factor compares like with like:

- A collision with a coloured initial parton gets the rest frame of the
  massive final-state particles of the core process.
  - At LO a decayed particle is represented by all of its decay products.
    For example, `p p > w+{0} z, w+ > e+ ve, z > e+ e-` gets `[3,4,5,6]`
    (`Process.get_massive_final_legs_with_decays`).
- e+ e- gets `[1,2]`.
- A decay gets `[1,2]` in madevent and `[1]` in mg7, which are the two
  spellings of the rest frame of the decaying particle.

**Only when the frame matters.** This default is applied only when a massive
particle is polarised, or when a spin-density matrix is asked for
(`output ... --density`, recorded as `density_matrix` in
`proc_characteristics`). Everything else keeps its old default: `[1,2]`
(hidden) in madevent and at NLO, and `[]` (no boost) in mg7. There the
matrix element is Lorentz invariant and nothing is boosted. For a process
whose final state is all massive (dibosons, `t t~`), the new default is the
partonic c.m. at the Born level, so LO results do not change. For one with a
jet, a photon or a lepton, they do. For example, polarised `p p > w+{0} j`
now uses the Collins-Soper frame of the W instead of its helicity in the
partonic c.m. The size of that LO change was not measured here.

## 3. Milestones

### M0 — Plumbing, no physics — **DONE**

Delivered:

| what | where |
|---|---|
| `me_frame` + `frame_id` in the NLO run card | `banner.py` `RunCardNLO.default_setup`, `frame_id` resolved in `update_system_parameter_for_include` |
| `frame_block` exposed only for polarised massive legs | `RunCardNLO.blocks` + `create_default_for_process` |
| `$frame` placeholder | `Template/NLO/Cards/run_card.dat`, after the beam energies (mirrors the LO card) |
| `common/to_frame_me/frame_id` | `Template/NLO/Source/run.inc` |
| `mapid_frame`, `get_frame_mask_born`, `get_frame_mask_real`, `get_me_frame_boost`, `boost_to_me_frame` | `Template/NLO/SubProcesses/boost_to_frame.f` (new) |
| `boost_to_frame.o` in every executable | `makefile_fks_dir` `FILES` |
| `frame_info.inc` per P dir | `export_fks.write_frame_info_file` |
| user leg order captured pre-sort | `fks_common.get_user_leg_order`, `FKSProcess`, `FKSHelasProcess` |

Gate met. The change is **runtime-inert by construction**, which is a stronger
statement than a bit-identical rerun: `frame_id` is assigned in `run_card.inc`
but read nowhere, and nothing calls the boost routines yet. Verified by grep
over a generated process dir. `p p > z j [QCD]` builds, links and runs clean.

`get_me_frame_boost` returns `trivial=.true.` both when nothing is selected and
when the selected system already has exactly zero 3-momentum, so the default
`me_frame=[1,2]` costs nothing and cannot perturb existing results even once
call sites are added in M1.

Unit-tested numerically (rest frame reached, invariant mass preserved,
4-momentum conservation preserved, in-place aliasing safe).

Also fixed on the way: adding a block to `RunCardNLO` without a `$frame`
placeholder in the card template made `banner.py:3178` append the block at the
end of the file, which broke the `test_default` round-trip in
`tests/unit_tests/various/test_banner.py`. The placeholder is the fix.

Original task list, for reference:

1. `me_frame` + `frame_id` into `RunCardNLO` (`banner.py:5758`), mirroring
   `banner.py:4438-4440`; replicate the `frame_id = sum(2**n)` resolution from
   `:4869` and the `frame` display-block trigger from `:5191` into the NLO
   process-dependent setup.
2. `integer frame_id / common/to_frame_me/frame_id` into
   `Template/NLO/Source/run.inc`.
3. `boost_to_me_frame` + `mapid` into the NLO SubProcesses (port of
   `genps.f:1761`, pure `boostx`, keep the trivial-boost short circuit).
4. Exporter writes `frame_info.inc` per P dir (D2). Add alongside the other
   per-dir includes at `export_fks.py:502`.

**Gate:** `frame_id=6` (default) reproduces current NLO results bit-for-bit on
an unpolarised `p p > z j [QCD]`. Nothing else changes yet.

### M1 — `[LOonly=QCD]`: Born boost only — **DONE**

Gate met. `p p > z{0} j`, matched PDF (nn23lo1), fixed scales (91.188), ptj 30,
etaj 4.0:

| | LO madevent | NLO LOonly | |
|---|---|---|---|
| unpolarised (control) | 6966 +- 6 | 6964 +- 18 | 0.03% |
| `z{0}`, Z rest frame `me_frame=[3]` | 1370 +- 1.7 | 1366 +- 3.1 | 1.1 sigma |
| `z{0}`, partonic c.m. `me_frame=[1,2]` | 974.8 +- 1.0 | 977.3 +- 2.3 | 1.0 sigma |

The unpolarised control matters as much as the polarised rows: it proves the
PDF/scale/cut/channel matching is right, so a polarised disagreement cannot be
blamed on setup. The frame dependence is large (40%) and both codes reproduce
it.

Null test, `p p > z{0} z{0} [LOonly=QCD]`: `me_frame=[3,4]` gives
0.5392 +- 0.0016 pb and `me_frame=[1,2]` gives 0.5401 +- 0.0017 pb (0.4 sigma).
At Born level the ZZ system *is* the partonic c.m., so the boost must be the
identity and the answer must not depend on `me_frame` -- it does not.

Incidentally this also confirms `nhel` is a variance knob, not a correctness
one, for polarised LO: `nhel=1` gives 1369 +- 1.7 against `nhel=0`'s
1370 +- 1.7. `help_polarization` currently implies `nhel=1` is required.

Converted call sites (all three reachable with `abrv='born'`):
`compute_born` (`fks_singular.f:37`), `include_multichannel_enhance`
(`:1505` and the cache-isolated `:1550`), and `bornsoftvirtual` (`:6896`,
not reachable in LOonly but converted for M3).

Guard relaxed for `LOonly` only (`madgraph_interface.py:1240`), including the
massive-particle rejection; `[QCD]` and the other NLO modes are still refused.

Two things learned the hard way; record them before touching M2.

**The fixed-order Born does not come from `bornsoftvirtual`.** It comes from
`compute_born` (`fks_singular.f:1`, the first routine in the file), called from
`driver_mintFO.f:445`, which does its own `call sborn(p_born,wgt_c)` and then
`add_wgt(2,...)`. `bornsoftvirtual` (`:494`) jumps straight to label 549 when
`abrv='born'` and leaves `amp_split_wgtnstmp` at zero, so its `add_wgt(3,...)`
contributes nothing. Converting only `bornsoftvirtual` therefore changed the
answer without ever boosting the Born that is actually integrated. Both sites
are now converted.

**B5 is not hypothetical.** With only `bornsoftvirtual` converted, the run did
not simply stay unboosted: it moved by ~3% (974.8 -> 946.7). Two callers of
`SBORN` disagreed about the frame within one event, and they share both the
`amp_split` common and the `calculatedBorn`/`savemom` cache, so the second
caller either reused amplitudes from the other frame or thrashed the cache.
This is exactly the failure mode D1 predicts, and it is silent -- no crash, no
warning, just a wrong number. Every `SBORN` caller reachable in a given mode
must be converted together, or none.

Original task list:

`[LOonly=QCD]` generates a fake FKS config with no reals and no virtuals
(`export_fks.py:3676`), so this milestone touches exactly one ME call and zero
counterterms. Cheapest possible validation of D1 + D2 + the run_card.

- Boost `p_born` before `call sborn` at the LOonly-reachable sites.
- Relax the guard at `madgraph_interface.py:1240` for `LOonly` only, and drop
  the massive-particle rejection at `:1257` on that branch (today it blocks
  `z{0}` even in the allowed `noborn` mode).

**Gate — acceptance test 1 (`p p > z{0} j`):** `[LOonly=QCD]`
`calculate_xsect LO` vs. LO madevent `generate p p > z{0} j`, same `me_frame`,
same cuts/PDF/scales, `group_subprocesses False`. Must agree inside MC error.
This validates the NLO Born boost against an independently-validated
implementation. Pin `nhel` identically on both sides.

`p p > z{0} z{0}` is a deliberate **null** test here: at Born level the ZZ
system *is* the partonic CM, so the boost must be a no-op and the polarised
LOonly result must be `me_frame`-independent.

### M2 — `[real=QCD]`: reals + counterterms (the hard milestone)

**Step 0 (ISR half) DONE.** The azimuthal factor can now be rebuilt covariantly.

`genps_fks.f` stores `xij_kperp` (common `/cxij_kperp/`) next to `xij_aor` at
both ISR generation sites (`generate_momenta_initial`,
`generate_momenta_initial_noevpr`), zeroed wherever `xij_aor` is zeroed.
`azifact_from_kperp` in `boost_to_frame.f` rebuilds `-exp(2 i psi)` from the
mother direction and that vector.

The reconstruction turned out cleaner than the plan assumed. Using the standard
helicity basis of the mother direction,

    e1 = ( cos(th)cos(ph), cos(th)sin(ph), -sin(th) )
    e2 = (       -sin(ph),       cos(ph),        0  )

reproduces **both** incoming legs with no `idir` bookkeeping at all:

    j_fks=1, n=+z : psi = phi_i      -> -exp( 2 i phi_i)
    j_fks=2, n=-z : psi = pi - phi_i -> -exp(-2 i phi_i)

because the doubled angle turns the basis flip into a harmless 2 pi. This is
*why* B3 is right that the `R_y(pi)` flip must be deleted rather than boosted:
it was only ever compensating for a basis convention that the psi form handles
by construction.

Validated at Lambda=1 against `-exp(2 idir i phi_i)` over 37 azimuths on both
beams: worst deviation 2.4e-16, i.e. one ulp. Exact bit-identity is not
reachable through this route (`(a+ib)^2` vs `exp(2 i phi)` differ in the last
ulp), so the shipped code keeps using `xij_aor` whenever the boost is trivial
and only takes the covariant route when a frame is actually requested --
unpolarised runs stay bit-identical by construction rather than by measurement.

A longitudinal boost leaves the reconstruction *exactly* unchanged (0.0, not
just small), confirming the plan's prediction that a longitudinal-only
`me_frame` is a genuine null and therefore a useful intermediate test.

**Step 0 (FSR half) DONE, and B4 is resolved -- both parts favourably.**

*Which branch runs.* The same one as ISR. `sreal` dispatches to
`sborncol_fsr`/`sborncol_isr` on `pmass(j_fks)` and `j_fks<=nincoming`, but the
collinear and soft-collinear counter-events pass the literal `one` for
`y_ij_fks` (`fks_singular.f:793`, `:904`) in both cases. So FSR also always
takes `azifact = xij_aor`, and its `IXXXXX`/`OXXXXX` branch is live only in the
sliver `vtiny=1e-8 <= 1-y < tiny=1e-6` -- never during `colltest`, where
`tiny=1d-12`.

*Equivariance.* Holds for FSR too. The spectator boost has
`shybst = -(shat-sumrec^2)/(2 sumrec sqrt(shat))` with
`recoil = p_total - p_mother`, which in the partonic c.m. vanishes exactly when
`E_m = |p_m|`, i.e. for a massless mother. Since
`m_mother^2 = 2 E_i E_j (1-y) -> 0`, `shybst = O(1-y) -> 0` in the collinear
limit. So the reduced Born and the real event share a frame in the singular
region and there is no residual rotation of the polarisation axes.

*The FSR reconstruction needs no new machinery.* The emission is generated
about a `+z` mother as `(cos(phi_i),sin(phi_i),0)` and then passed through
`rotate_invar`, which is `R_z(phi) R_y(theta)` and therefore maps `x,y` onto
exactly the `e1,e2` of the helicity basis above. So the rotated vector has
azimuth `phi_i` again, and the *same* `azifact_from_kperp` serves both cases.
The only difference is a conjugation, which is just spacelike vs timelike
splitting:

    ISR net factor = -dconjg( azifact_from_kperp(mother, kperp) )
    FSR net factor = -        azifact_from_kperp(mother, kperp)

Checked at Lambda=1 against the shipped `-(cphi_m - i sphi_m)^2 * xij_aor` over
672 configurations of `(theta_m, phi_m, phi_i)`: worst deviation 1.4e-15.

This also retires the worry that FSR would need its own analysis: the two
cases differ by a conjugation, not by structure.

`[real=QCD]` gives the full FKS subtraction without virtuals — everything hard,
nothing MadLoop.

1. Boost the real momenta before `call smatrix_real(pp,wgt)`
   (`fks_singular.f:4701`).
2. Boost each reduced configuration with its *own* frame: `p1_cnt(0,1,0/1/2)`,
   `p_born`, `p_born_used`, and the `extra_cnt` argument. Sites:
   `fks_singular.f:494` (`bornsoftvirtual`), `:705`/`:793`/`:904` (the
   `sreal`/`sreal_deg` counterterms), plus the ~20 `call sborn` sites.
3. Soft is easy: the eikonal `p_m.p_n/(p_m.k)(p_n.k)` is invariant and
   `sborn_sf` is colour-linked only (no spin correlation), so `sbornsoft` needs
   nothing beyond a boosted `p_born`.
4. **The azimuthal phases — the real work.** `SBORN` returns
   `ANS(2) = BORNTILDE`, the +/- gluon-helicity interference built from
   `JAMPH(1,.)`/`JAMPH(2,.)` (`born_fks.inc:277`). The correction multiplying
   it collapses to `-exp(-2*i*psi)`, with `psi` the emission azimuth about the
   mother in the HELAS `(e1,e2)` basis.

   **No extra Wigner phase is needed on `BORNTILDE`.** The mother's
   little-group phase `exp(-+2*i*theta_W)` cancels against the `exp(+2*i*theta_W)`
   picked up by a *boosted* `<ij>/[ij]` — same `theta_W`, since i, j and the
   mother are collinear. The controlling identity, exact in any single frame
   and for any mother direction, is `arg(azifact)/2 = phi_m + psi`. There is no
   `theta_m` dependence (`R_y(theta)` has a real SU(2) matrix), which is
   exactly why `getaziangles` returning only `phi` suffices. The generator's own
   analytic limits confirm it independently: `genps_fks.f:2344` gives
   `-exp(2i(phi_mother+phi_i))` (FSR), `:3099` gives `-exp(2*idir*i*phi_i)` (ISR).

   The massive-leg (Z) Wigner transformation is *not* a phase but a genuine
   helicity mixing — that is the polarised observable itself, not an error to
   correct. It is why `SBORN` must be handed boosted momenta.

   **ISR — BLOCKER B1, and the branch that actually runs.**
   `fks_singular.f:5034-5035` hardcodes `cphi_mother=1, sphi_mother=0`; that is
   numerically identical to `getaziangles` on a beam-axis momentum
   (its `sth.ne.0` guard at `:4111` returns exactly `(1,0)` for `cth=+-1`) and
   it dies under a boost. But repairing it is not the main problem:

   The ISR counter-events are called with `y_ij_fks = one` **exactly**
   (`fks_singular.f:793`, `:904`; `y_ij_fks_matrix(1)=y_ij_fks_matrix(2)=1.d0`
   at `genps_fks.f:2742`), so `1d0-y_ij_fks .lt. vtiny` always fires and
   `azifact = xij_aor` is taken every time (`:5006-5007`). The
   `IXXXXX`/`OXXXXX` spinor branch below it is live only in the sliver
   `vtiny=1e-8 <= 1-y < tiny`, where `tiny=1d-6` in production and `1d-12`
   under `colltest` (`:4658`) — i.e. it is **never exercised by
   `test_soft_col_limits`**. Both branches still have to be made consistent or
   that sliver becomes a discontinuity nothing catches.

   `xij_aor = -exp(2*idir*i*phi_i_fks)` is a precomputed partonic-CM constant
   (`genps_fks.f:3096-3101`) and a `0/0` limit of `<ij>/[ij]`. A boost maps
   exactly-parallel null vectors onto exactly-parallel null vectors, so **the
   degeneracy survives in every frame — it can never be recomputed after
   boosting.** The azimuthal direction must be carried through as a stored
   4-vector.

   Proposed fix: store `k_perp_dir = (0, cos(phi_i_fks), sin(phi_i_fks), 0)`
   alongside `xij_aor`. One vector covers both beams (for `j_fks=2` the basis
   is `(x, -y)`, so it yields `psi = -phi_i` — precisely the `idir` sign
   already in `xij_aor`, and the `idir` special case disappears). Then boost
   it, project out the mother component, take
   `psi' = atan2(k'_perp . e2, k'_perp . e1)`, apply `-exp(-2*i*psi')`. At
   `Lambda=1` this reproduces current numbers bit-for-bit, so it lands as a
   safe refactor **before** any boost is wired in.

   `azifact` is invariant under independent positive rescalings, so
   `p_i_fks_ev` being the rescaled null vector `p_i/xi`
   (`genps_fks.f:3085-3094`) is harmless: `Lambda(lambda*p) = lambda*(Lambda p)`.

   **ISR is a package of three changes** — do any subset and you apply part of
   the correction, which is worse than the current code:
   (a) carry `k_perp_dir` through the boost as above;
   (b) `getaziangles` on the boosted mother instead of the hardcode;
   (c) **delete** the `j_fks.eq.2` `R_y(pi)` flip (`:5013-5020`) — after a boost
   it no longer maps the mother onto +z. Note `montecarlocounter.f:3000-3006`
   double-encodes the same thing (both the rotation and `cphi_mother=-1.d0`);
   numerically identical today since only the square enters, but do not carry
   both forward.

   **FSR:** `getaziangles(p_born(0,imother_fks), …)` (`:4844`) and `azifact`
   built by `IXXXXX/OXXXXX` on `p_i_fks_ev`, `p(.,j_fks)` (`:4830`) must both be
   evaluated on momenta carrying the *same* boost as the Born. The FSR
   equivalent of B1 has not been checked — establish whether the FSR
   counter-events also collapse onto the `vtiny` branch.

   **Scope narrowing:** `Qterms_reduced_spacelike` returns a real, z-only
   number (`:5538-5584`), non-zero only for a **vector mother** (`col1=8`); the
   `abs(m_type).eq.3 .or. ch_m.ne.0d0` branch that zeroes `Q` and `wgt1(2)` is
   the same statement one level up. So **only gluon-mother ISR configurations
   are affected at all** — a process whose ISR mothers are all quarks passes
   the collinear test even with a completely wrong azimuthal phase. Choose the
   validation channel deliberately.

   This is the frame/orientation consistency issue: it lands on the
   counterterms, not on `|M|^2`.

5. **Same construct elsewhere**, needing the same treatment if polarised
   aMC@NLO (not just fixed-order) is in scope: `montecarlocounter.f:2969-3012`
   and `montecarlocounter_alt.f:1445-1533`. `sreal_deg` (`fks_singular.f:5844`)
   has no azimuthal term — it needs the boost for the ordinary Born only.
   **DONE in M5**, and it was real: the phase was wrong by a constant
   37 to 157 degrees.

**Step 1 (the B5 sweep) DONE.** Every ME entry point now goes through a frame
wrapper, so the boost is a pure function of the momenta passed in.

That is what makes B5 go away *structurally* rather than by inspection: two
callers that pass the same momenta necessarily get the same boost, and two
callers that pass different momenta were already required to reset the cache
between them. No reachability argument is needed, which matters because the
reachability argument is exactly what I got wrong twice in M1.

Wrappers in `boost_to_frame.f`: `sborn_frame`, `sborn_sf_frame`,
`extra_cnt_frame`, `smatrix_real_frame` (the last uses the *real* mask, derived
from the Born one and `i_fks` of the current FKS configuration).

43 call sites converted:

| file | sites |
|---|---|
| `fks_singular.f` | 30 |
| `montecarlocounter.f` | 5 |
| `montecarlocounter_alt.f` | 5 |
| `add_write_info.f` | 1 (this retires **B7**) |
| `check_poles.f` | 1 |
| `test_soft_col_limits.f` | 1 |

Deliberately **not** converted, and why: the EW Sudakov paths
(`check_sudakov*.f`, `ewsudakov_functions_dummy.f`,
`sa_ewsudakov_dummyfcts.f`) are a separate feature, and `symmetry_fks_v3.f` is
`gensym`, a separate executable whose Born calls only seed the integration
grid. Neither shares a process with the integration, so neither can trip the
cache.

**The sweep also exposed a wrong default, worth recording.** With all 43 sites
converted, unpolarised `p p > z j [QCD]` moved from 2.854e+04 to 2.834e+04
+- 1.9e+02 -- only 0.74 sigma, but the two previous runs had been *bit*
identical, so the pipeline is deterministic and a change meant a real change.

Cause: MadFKS does not hand the matrix elements momenta in their own partonic
c.m. `shy_lbst = -xi_i_fks*yijdir/bstfact` is non-zero for any real emission,
so `xp(0,1) != xp(0,2)` (`genps_fks.f:3074-3082`) and the real event lives in a
frame boosted along z. Honouring `me_frame=[1,2]` literally therefore applies a
genuine longitudinal boost to every configuration -- an identity for `|M|^2`,
but not bit for bit, and enough to send the VEGAS grids down a different path.

Confirmed by control: the same build with `me_frame=[0]` (`frame_id=1`, every
mask empty, boost skipped) reproduces 2.854e+04 +- 1.9e+02 exactly. So the
sweep itself is sound and the whole shift came from the boost.

The semantics were right -- boosting to the initial-parton rest frame is what
`[1,2]` means at LO, and the M1 cross-check agreed with LO at 1.0 sigma -- but
the *default* was wrong: an unpolarised run must not pay for machinery it never
asked for. `frame_id` is now 0, meaning "skip", unless `me_frame` actually
appears in the run_card:

| case | frame_id | |
|---|---|---|
| unpolarised (no `me_frame` in the card) | 0 | boost skipped, bit-identical |
| polarised, default | 6 | partonic c.m., as at LO |
| polarised, `me_frame=[3]` | 8 | Z rest frame |

The LO path is untouched: there the momenta really do arrive in the lab frame,
so `[1,2]` has always been a meaningful boost and still is.

**Skip the identity boost structurally, as LO does.** Applying a boost that is
mathematically the identity is not free -- it goes through `boostx` and
perturbs the momenta in the last bits, which is enough to move an adaptive
integration. LO avoids this in *two* places, and the NLO port now folds in
both:

- the call site skips `frame_id=6` outright (`auto_dsig_v4.inc:183`);
- `boost_to_frame()` skips the all-final-state selection (`genps.f:1782`),
  whose comment says of the initial-state case: *"1 1 0 0 0 .... should not go
  within this function"*.

`get_me_frame_boost` now returns `trivial` for four cases: nothing selected,
exactly the initial state, exactly the whole final state, and an already-at-
rest system. Relying on the last one alone (as the first version did) is not
enough at NLO, because the tilde frame means it never fires.

**`me_frame` must not be built from the initial state at NLO -- it is not
infrared safe.** This is a stronger statement than the numerical one above and
it constrains what the feature may *accept*, not just what it skips. The real
emission and the reduced Born carry momentum fractions that differ by a finite
amount even in the singular limit, so a frame defined from the initial state is
discontinuous across that limit and the subtraction stops cancelling. No
azimuthal fix can repair it -- this is the same failure mode as the
equivariance discussion in D3, but caused by the frame *definition* rather than
by the mapping.

Consequently `RunCardNLO.check_validity` rejects an `me_frame` that mixes
initial-state legs with final-state ones. `[1,2]` and the full final state stay
accepted, since they merely name the partonic c.m. and are skipped. Frames
should be defined from final-state particles only, which is what the physics
wants anyway: the polarised system (the Z, the ZZ pair) is a set of final-state
spectators, present unchanged in both the real and the Born.

Re-validated after this change -- unpolarised 2.854e+04 +- 1.9e+02 (baseline),
`z{0} j` Z-frame 1366 +- 3.1 and partonic c.m. 977.3 +- 2.3, both identical to
the M1 numbers. The skip reproduces the boost exactly for `[LOonly=QCD]`, as it
must: with no emission `shy_lbst=0` and the Born really is in the partonic c.m.

**Step 2 (azimuthal wiring) DONE. The M2 gate PASSES and `[real=QCD]` is
enabled.**

`azifact_me_frame` rebuilds the frame from the same Born momenta the matching
`sborn_frame` call gets, boosts the mother and `xij_kperp` together, and
returns `-exp(2 i psi)`. Both `sborncol_isr` and `sborncol_fsr` use it in their
boosted branch and, there, apply neither the `cphi_mother=1` shortcut, the
`R_y(pi)` flip, nor `getaziangles` -- the psi form carries all of them (**B3**).
The legacy branch is untouched, so unpolarised runs stay bit-identical.

It first failed the gate: `test_ME` on `p p > z{0} j [real=QCD]` with
`me_frame=[3]` gave soft 0.48, collinear 0.37. The section below is the
diagnosis. It was *not* a frame-consistency bug and the azimuthal work was
never implicated.

### The failure: the quantisation axis was decided by rounding

**Root cause.** For a one-leg `me_frame` selection -- `me_frame=[3]`, "the Z
rest frame", which is the standard way to ask for Z polarisation -- the boost
puts that leg at rest, and *there the boost has to be exactly right rather than
right to rounding*.

HELAS changes convention at exactly zero. `vxxxxx` (`aloha_functions.f`) builds
the polarisation vectors of a massive vector from

    pp = min(|p(0)|, sqrt(p(1)^2+p(2)^2+p(3)^2))

and branches on `pp.eq.rZero`: at exactly zero it uses the **z axis** of the
current frame as the quantisation axis, and otherwise it uses the **momentum
direction**, giving `eps_L = (pp/m, E*p/(m*pp))`.

`boostx` does not reach `p=0` exactly. With `q=(E,-P)` and `p=P` it forms
`p(i)+q(i)*lf` where `lf` is 1 only up to the rounding of `(q(0)-m)+p(0)`, so
the residual is a few `1d-14` pointing in a direction that is pure noise.
Whether it rounds to zero is a coin flip, decided independently for the real
and for the reduced Born. Every configuration that misses zero gets `eps_L`
pointed along rounding noise instead of along z, and the subtraction is then
comparing two different observables.

**Evidence.** Instrumented `boost_to_me_frame` to print `|p|` of the selected
leg after the boost, alongside the soft-limit ratio table, on
`p p > z{0} z{0} [real=QCD]` with `me_frame=[3]`. One-to-one, no intermediate
values:

| residual `|p_Z|` after the boost | soft ratio |
|---|---|
| 0 (exact) | 0.9033, 0.9901, 0.9990, 0.9999, 1.0000, 1.0000, 1.0000 |
| 2.95e-14 | 0.0021716, 0.0021716, 0.0021716 |

The Born landed on exact zero for every point sampled; the real missed it 30%
of the time. A factor 460 in `|M|^2`, from a 3e-14 momentum.

**Fix** (`boost_to_me_frame`, `Template/NLO/SubProcesses/boost_to_frame.f`):
when the selection is a single leg, zero that leg's 3-momentum exactly after
boosting. This imposes the defining property of the frame instead of hoping
for it, and removes the choice. Only `nsel=1` needs it -- with two or more
selected legs it is their *sum* that is at rest, no individual leg sits on the
branch point, and the residual of the sum never reaches HELAS.

### `p p > z z` was the right thing to run first

It was decisive, but not in either of the two ways this document predicted.
`me_frame=[3]` on `p p > z{0} z{0} [real=QCD]` reproduces the `z{0} j` symptom
exactly -- soft 0.55/0.68, collinear 0.54/0.57, uniform -- and there every ISR
mother is a quark, so `Q` vanishes identically and the azimuthal phase is
provably not involved. That reduced the search to the boost itself on a process
that builds in a quarter of the time.

The `me_frame=[3,4]` spelling suggested in the previous revision would have been
a much weaker probe: it fails only a split-order soft test (0.43/0.30) while the
summed test passes, and for a *different*, purely numerical reason -- see below.

**What the D3 instrumentation actually showed.** The boost 4-vectors were
printed for the real and for the reduced Born at the same counter-event, as D3
asks. They converge cleanly: on `me_frame=[3,4]` the real's boost runs
`(553.24, -13.00, 25.52, ...)` down to `(552.50, -1.2e-8, 2.4e-8, ...)` against
the Born's constant `(552.50, 0, 0, ...)`, i.e. the transverse part vanishes
like xi exactly as D3 requires. **The working hypothesis of a finite frame
mismatch was wrong**, and so was the reasoning that led to it. There is no need
for a permanent D3 assertion: the property holds, and the thing that broke was
downstream of it.

### A second, independent finding: the same bug is live at LO

`boost_to_frame` (`Template/LO/SubProcesses/genps.f:1759`) has the identical
defect, and `me_frame` is a *shipped* LO feature. Measured on LO madevent,
`p p > z{0} j`, `me_frame=[3]`, nn23lo1, fixed scales 91.188, ptj 20, etaj 5:

| | cross-section |
|---|---|
| before the fix | 1399 +- 3.4 pb |
| after the fix | 1424 +- 3.8 pb |

1.8%, 5 sigma. So a fraction of LO events had been getting a rounding-noise
polarisation axis all along. The same fix is applied there.

This also matters for M1: that milestone's gate is NLO `[LOonly=QCD]` against
LO madevent, so fixing only the NLO side would have made the two disagree.
Re-validated with both fixed, same cuts and PDF: LO madevent 1424 +- 3.8 vs
NLO `[LOonly=QCD]` 1427 +- 3.0, **0.6 sigma**.

### A precision floor, and the betamin guard

Separately, `me_frame=[3,4]` on `p p > z{0} z{0}` failed the *split-order* soft
test at 0.43/0.30 while the summed test passed. That one is not physics.

That selection is the whole Born final state, so its boost degenerates to the
identity as xi -> 0. Applying a near-identity boost is actively harmful: it
tilts the incoming partons off the beam axis by O(beta), and the HELAS massless
spinors are built from `sqrt(p(0)+p(3))`. On the axis that sum is exactly zero
and the exact branch is taken; tilted by beta it is `E*beta^2/2`, computed as a
difference of two numbers of size E, hence carrying absolute error `ulp(E)` and
relative error `2*eps/beta^2`. Measured down the soft scan, `p(0)+p(3)` of the
boosted incoming leg and the deviation of the ratio from 1:

| `p(0)+p(3)` | 3.3e-9 | 3.3e-11 | 3.4e-13 | 0 | 0 | 0 |
|---|---|---|---|---|---|---|
| ratio - 1 | 9e-6 | 6.5e-4 | 2.1e-2 | 2e-6 | 4e-7 | 2e-6 |

Exactly `1/beta^2` while `p(0)+p(3)` is a few ulp, clean again once it
underflows to exactly zero.

Balancing what the boost delivers (O(beta)) against what it costs
(`30*2*eps/beta^2`, the 30 measured above) gives `beta* = 2.4e-5`.
`get_me_frame_boost` now skips the boost below `betamin=1d-4`, one safety
decade above that. Scanned: `1d-5` is not enough (0.16 still failing), `1d-4`
and `1d-3` both give 0.01. This costs no physics -- it fires only where the
selected system is at rest to one part in 10^4, which never happens in an
integration, only in the artificial deep-soft scan of `test_soft_col_limits`.
Confirmed neutral: the `[real=QCD]` cross-section is 2.604e+03 +- 2.8e+01 pb
without the guard and 2.600e+03 +- 2.2e+01 pb with it, 0.1 sigma.

**Update 2026-09-15: the guard is removed.** PR #111 (on main) rewrote the
massless spinors to take the small light-cone component from
`p+ p- = pT^2` whenever `p3 < 0` and `pT != 0`, a sum of positives where
`sqrt(p(0)+p(3))` was a cancelling difference -- exactly the tilted
backward beam above. Re-measured on `p p > z{0} z{0} [real=QCD]`,
`me_frame=[3,4]` (`FRAME_ID = 24`), the split-order soft test that exposed
this, three trees generated from scratch:

| HELAS | guard | split-order soft, 16 tests |
|---|---|---|
| before #111 | removed | **6 FAILED**, worst 0.43, run aborts |
| with #111 | kept | 0 failed, worst 0.01 |
| with #111 | removed | 0 failed, worst 0.02 |

The first row reproduces the 0.43 above exactly, so the test still sees the
defect; the last shows the spinors no longer need protecting from it.

### Gate results

`p p > z{0} j [real=QCD]`, `me_frame=[3]`, nn23lo1, fixed scales -- the
process this document names as the **discriminating** one, and the one whose
gluon-initiated channels exercise the azimuthal `Q` term:

| | test_ME |
|---|---|
| before | 268 FAILED of 512 test lines, all 12 P dirs |
| after | **0 FAILED of 512**, worst failure fraction 0.08 against a 0.30 threshold |

`calculate_xsect NLO` then runs end to end: **2.600e+03 +- 2.2e+01 pb**.

`p p > z{0} z{0} [real=QCD]`, all frames, `P0_ddx_z0z0`:

| `me_frame` | frame_id | before | after |
|---|---|---|---|
| none | 0 | passes (0.01) | passes (0.01), unchanged |
| `[3]` | 8 | FAILS 0.55/0.68 | passes, worst 0.08 |
| `[3,4]` | 24 | FAILS 0.43/0.30 | passes, worst 0.01 |

Unpolarised runs are untouched **by construction**, not by measurement:
`frame_id=0` makes every mask empty, `get_me_frame_boost` returns `trivial` on
`nsel=0` before reaching either new branch, and `boost_to_me_frame` returns
before the `nsel=1` block.

The guard at `madgraph_interface.py` now accepts `real` alongside `loonly`.

### One thing that was checked and left alone

The "exactly the whole final state" skip in `get_me_frame_boost` is
multiplicity-dependent, so it does not mean the same thing for the Born and for
the real: `me_frame=[3,4]` on `p p > z z` is the whole Born final state but only
two of three real legs. That asymmetry looked like the bug and is not: removing
the skip leaves the failure fractions **bit-identical**, because `p_born` is
handed to us in the Born partonic c.m., so the zero-3-momentum test catches the
same case anyway. Reverted, and the reason recorded in the source.

**Recommended order within M2:**

0. Refactor the ISR azimuthal factor to the `psi`-form at `Lambda=1` (the
   `k_perp_dir` construction above, boost = identity). Demand **bit-identical**
   `test_ME`. This de-risks the hardest change before any frame logic exists.
1. Wire in a **longitudinal-only** `me_frame` boost. Must also be unchanged —
   the mother stays on the beam axis and the z-rotation Wigner phase of a
   z-directed massless particle is trivial, so the existing hardcode remains
   exactly correct. A useful null.
2. Then the general boost.

**Gate — `test_soft_col_limits`.** Already runs automatically as `test_ME` on
every `calculate_xsect` (`amcatnlo_run_interface.py:5494`,
`Template/NLO/SubProcesses/test_soft_col_limits.f`; the collinear scan at
`:617-663` drives `1-y = 1e-2 … 1e-10`). The `Q(z)` azimuthal term is the
*same order* in the collinear limit as the `AP(z)` term, so a wrong phase does
not cancel — the ratio plateaus off 1. Run it before any cross-section
comparison.

**Gate — acceptance test 2 (`p p > z{0} z{0}`):** the Born boost is trivial but
the real has `z z j`, so in the *finite* region the ZZ frame != partonic CM.
This exercises the real boost with the Born held fixed.

**But it does not discriminate the azimuthal phase.** In the singular region —
which is precisely where `test_soft_col_limits` probes — the reduced Born is
`p p > z z` with zero pT(ZZ), so the `me_frame` boost degenerates to
longitudinal and the phase issue does not bite. `p p > z{0} z{0}` will pass
either way. The discriminating processes are those whose `me_frame` system
recoils already at Born level: **`p p > z{0} j`** and **`p p > z{0} z{0} j`**,
restricted to gluon-initiated channels (see the scope narrowing above).

### M3 — `[QCD]`: virtual — **DONE, GATE PASSES**

`binothlha_frame` boosts the momenta handed to `BinothLHA`, which passes them
straight to `sloopmatrix_thres` and takes `born_wgt` from the caller rather
than recomputing it, so one call site covers the virtual.

**Resolved.** The integration was already right; the *gate* was the thing
comparing two frames. `check_poles.f` had been converted by the M2 sweep on
its Born line only — `call sborn_frame(p_born, born)` — while the line
immediately below it still called `BinothLHA(p_born, …)` directly, on the
unboosted `p_born`. So the driver handed MadFKS a Born in the Z rest frame
and MadLoop a phase-space point in the partonic c.m., and then compared the
poles the two produced. Routing it through `binothlha_frame`, exactly as
`bornsoftvirtual` already does at `fks_singular.f:7123`, makes the check
pass. The rest of this section is the original failure report followed by the
diagnosis, the evidence and the gate results.

#### What the failure was

`check_poles` on `p p > z{0} j [QCD]` with `me_frame=[3]`: **poles do not
cancel**, 20 miscancellations. `test_ME` was clean on the same run (0 FAILED,
40 PASSED), which correctly said M2 was unaffected and the problem was on the
virtual side.

The pole coefficients were read as saying it is not a normalisation:

    COEFFICIENT DOUBLE POLE:  MadFKS -3.23e-3  OLP -4.88e-5   ratio 66
    COEFFICIENT DOUBLE POLE:  MadFKS -5.13e-3  OLP -9.43e-5   ratio 54
    COEFFICIENT SINGLE POLE:  MadFKS -4.06e-3  OLP -7.56e-5   ratio 54

"the ratio is O(50-60) and *not* constant, so MadLoop is evaluating a
different polarisation state rather than the same one with a wrong factor."

**That reading was wrong, and it is what sent the search into MadLoop.** The
three lines above come from three *different* phase-space points (points 4, 5
and 6 of the scan), one line each — so the varying ratio is a comparison
across points, not within one. Grouped by point, the ratio is *exactly*
constant: point 1 gives 58.03 for the double pole and 58.03 for the single,
point 2 gives 81.01 and 81.01, and only the per-point value moves. Both poles
are proportional to the Born, so a per-point constant ratio says the two
sides disagree about the Born by an overall factor, i.e. about the frame.
Group diagnostics by phase-space point before drawing conclusions from them.

Two MadLoop settings were suspected and both were correctly **excluded by
test**:

- `NRotations_DP`/`NRotations_QP` re-evaluate the loop at a rotated PS point and
  expect `|M|^2` unchanged -- which fails for a particle at rest, whose axis is
  the frame z axis and does not rotate with the momenta. But both **default to
  0**, so the rotation test never runs. Latent hazard, not this bug. Keep them
  at 0 for polarised runs.
- `ImprovePSPoint` (default 2) deforms the PS point to restore exact
  onshellness and could move the deliberately-zeroed leg off zero. Setting it
  to `-1` (no deformation in double precision) **does not fix it**: still 20
  miscancellations, ratio still ~58.

#### The diagnosis

**MadLoop's polarisation restriction was never the problem, and that was
checked before anything else was touched.** MadLoop reads its helicity
configurations from `MadLoop5_resources/<prefix>HelConfigs.dat`, and for
`P0_gd_z0d` that file holds exactly the eight rows of `born.f`'s `NHEL`
table, with the Z entry pinned to `0`:

    -1  1  0 -1        -1 -1  0 -1         1  1  0 -1         1 -1  0 -1
    -1  1  0  1        -1 -1  0  1         1  1  0  1         1 -1  0  1

`NCOMB=8` in `loop_matrix.f` against `MAX_BHEL=8` in `born_nhel.inc`. The
Born and the virtual sum over the same helicity set. `hel_avg_factor` is
restored as `loop_exporters.py:1552` intends. Nothing to fix there.

**Instrumentation closed it.** A `diag_frame` routine in `check_poles.f`
prints, for each point, the selected leg's `|p|` before and after the boost
and the Born evaluated both ways (`sborn` on the raw `p_born`, and
`sborn_frame`), with `calculatedborn` reset between the two so the shared
amplitude cache does not fight. The Z goes from `|p| = 318.6` to exactly `0`,
as `boost_to_me_frame` guarantees for `nsel=1`, and:

| point | Born(me_frame)/Born(partonic c.m.) | MadFKS/OLP pole ratio | rel. diff |
|---|---|---|---|
| 1 | 58.0338377763 | 58.0338377763 | 1.7e-14 |
| 2 | 81.0065550136 | 81.0065550136 | 6.9e-14 |
| 3 | 64.0169796788 | 64.0169796788 | 4.4e-15 |
| … | | | |
| 20 | 57.8773835460 | 57.8773835460 | 1.3e-14 |

All 20 points agree to 1e-13. The pole mismatch *is* the ratio of the two
Borns, with nothing else in it.

The chain that produces it: `getpoles` (`fks_singular.f:7574`) ignores its
`p` argument for the Born and reads `common/pborn/p_born`, then calls
`sborn_frame` on it — so the MadFKS poles are always in the me_frame.
MadLoop's poles come from whatever momenta `BinothLHA` was handed. In the
integration that is `binothlha_frame`'s boosted copy and the two agree; in
`check_poles.f` it was the raw `p_born` and they did not.

#### The fix

One line, `Template/NLO/SubProcesses/check_poles.f:228`:

```fortran
-          call BinothLHA(p_born, born, virt_wgt)
+          call binothlha_frame(p_born, born, virt_wgt)
```

`check_poles.f` is compiled only into the `check_poles` executable (`POLES`
in the P-dir makefile; the integration uses `driver_mintFO.o`), so no
cross-section can move as a result of this change — only the diagnostic that
was misreporting.

Unpolarised runs are unaffected, by measurement as well as by construction:
with `frame_id=0` every mask is empty, `get_me_frame_boost` returns `trivial`
and `boost_to_me_frame` copies the momenta unchanged. Rebuilt
`P0_gd_zd/check_poles` of unpolarised `p p > z j [QCD]` both ways — the two
logs are **byte-identical**.

The two settings excluded by the previous session stay excluded, and the
passing run now confirms it positively rather than by elimination: it ran
with the shipped defaults `ImprovePSPoint = 2`, `NRotations_DP = 0`,
`NRotations_QP = 0`.

#### Gate results

`p p > z{0} j [QCD]`, `me_frame=[3]`, nn23lo1, fixed scales 91.188, ptj 30,
etaj 4.0, `req_acc_fo 0.05`:

| | before | after |
|---|---|---|
| `check_poles`, `P0_gd_z0d` | 0 of 20 pass | **20 of 20** |
| `check_poles`, all 12 P dirs | — | **20 of 20 in every one** |
| `test_ME`, all 12 P dirs | passed | passed, unchanged |
| `calculate_xsect NLO` | — | **2.193e+03 +- 1.2e+01 pb** |

Agreement is now at the level of the loop-library accuracy, not the
tolerance: e.g. MadFKS `-5.1257915456849783e-03` against OLP
`-5.1257915456856262e-03`.

Unpolarised control, `p p > z j [QCD]`, same cuts and scales: `check_poles`
20 of 20 in all 12 P dirs, `calculate_xsect NLO` 1.029e+04 +- 5.5e+01 pb.

That control also gives a physics sanity check the pole cancellation cannot,
since the M1 table has the Born of both at the same cuts and scales:

| | Born (`[LOonly=QCD]`) | NLO (`[QCD]`) | K |
|---|---|---|---|
| unpolarised | 6964 +- 18 | 10290 +- 55 | 1.48 |
| `z{0}`, `me_frame=[3]` | 1366 +- 3.1 | 2193 +- 12 | 1.61 |

Both K factors are the O(1.5) expected for Z+jet at `muR=muF=mZ` with
`ptj>30`, and the polarised one is close to but not equal to the unpolarised
one — which is what a correct polarised NLO should look like. A frame bug of
the kind M3 hit would not have landed anywhere near here.

Two further processes, same settings:

| process | `me_frame` | P dirs | `check_poles` | `test_ME` | xsec |
|---|---|---|---|---|---|
| `p p > z{0} z{0} [QCD]` | `[3]` | 4 | 20/20 each | passed | 1.217e+00 +- 6.6e-03 pb |
| `p p > z{0} z{0} j [QCD]` | `[3]` | 12 | 20/20 each | passed | 4.779e-01 +- 3.3e-03 pb |

The second is the process this document names as the M3 acceptance test.

#### What the guard now accepts, and what it still refuses

A polarised massive particle needs a frame. There are **two** ways to have
one, and the guard now recognises both.

*Frame from the run_card.* Modes `loonly`, `real` and `all` — the last being
what the parser calls `[QCD]` with no mode keyword (`LoopOption='all'`,
`madgraph_interface.py:5104`) — **for a purely QCD perturbation**. These are
the modes whose matrix elements this work boosts.

*Frame from the user.* Mode `virt` outputs standalone MadLoop: the caller
supplies the phase-space point, so the frame is theirs to choose and there is
no `me_frame` and nothing for the code to get wrong. Accepted for **any**
perturbation orders, since the frame argument does not depend on them. This
was initially refused on the grounds that "it has no run_card, so it has no
frame" — which had the implication backwards. Confirmed by running it:
`p p > z{0} j [virt=QCD]` does reject `order=NLO`, `fixed_order=ON` and
`set me_frame [3]` at the launch prompt, because it is a standalone check and
not an integrable process directory. That is the point, not a problem.

**Unblocking `virt` exposed a separate bug, which had to be fixed for the
unblock to mean anything.** `p p > z{0} j [virt=QCD]` was generating
`P0_gu_zu` — no `z0` — with 24 helicity configurations covering all three Z
polarisations. The user would have asked for a polarised loop and silently
received the unpolarised one.

The polarisation is dropped by the **multiparticle expansion**, and only on
the MadLoop path. `u u~ > z{0} g [virt=QCD]` (explicit flavours) is correct —
`P0_uux_z0g`, 8 configurations, Z pinned to `0`. Tree level is correct with
multiparticles too. The culprit is `ProcessDefinition.__iter__`
(`base_objects.py:4670`), whose own docstring says *"not used by MG which
used some smarter version (use by ML)"*: it rebuilds each expanded leg as
`Leg({'id':id, 'state':…})` and never carries `polarization` across.
`loop_interface.py:885` iterates it and re-issues each combination as a text
command, so by the time `nice_string` runs the `{0}` is already gone.
`MultiProcess.generate_multi_amplitudes` — the path MG5 itself uses — copies
`polarization` properly, which is why only the loop side was affected.

Fixed by carrying the multileg's `polarization` onto each expanded leg.
`p p > z{0} j [virt=QCD]` now gives `P0_gu_z0u` with 8 configurations in all
seven subprocess MEs; unpolarised `p p > z j [virt=QCD]` still gives 24, so
nothing else moves.

Validated by running the standalone check, which has its own version of the
check_poles argument: the IR pole coefficient is a colour factor times the
Born, so `1eps/(born*ao2pi)` must be the same whether or not the helicities
are restricted, while the Born itself must differ.

| `g u > z u` | Born | `1eps/(born*ao2pi)` |
|---|---|---|
| `z{0}` | 3.1963946124086368e-04 | -4.1666666666641756 |
| unpolarised | 0.55343246189280271 | -4.1666666666665897 |

Both give -25/6. Had the loop summed all helicities while the Born stayed
restricted, this ratio would have been wrong — the same signature that gave
M3 away.

**The sibling defect, and why it has to land with this change.**
`ProcessDefinition.get_process` (`base_objects.py:4849`) hardcodes
`'polarization':[]` the same way. Verified on this branch:

    multileg pol : [[], [], [0], []]
    get_process  : [(2, []), (-2, []), (23, []), (21, [])]     shell: 0_uux_zg

Its callers are the `check` command paths. That matters here because
`check_check` runs `check_process_format` (`madgraph_interface.py:1156`), so
**opening `virt` in the guard also newly admits
`check timing|stability|profile <polarised massive process> [virt=QCD]`**,
which was refused before. Without the `get_process` fix that check silently
runs the *unpolarised* loop. Tree-level `check` is unaffected by the guard —
it has no brackets, so the polarisation branch never fires — and its version
of the bug is purely pre-existing.

Fixed separately on branch `claude/quizzical-ptolemy-16f0c8`, commit
`2b151185d`, which also found a **third** instance:
`process_checks.py:2032` `run_multiprocs_no_crossings` built its legs with no
polarization at all, and that is the function every tree-level `check` goes
through — so fixing `get_process` alone would have left `check
gauge/lorentz/permutation` unpolarised. Their evidence, `check permutation
p p > z{…} j --seed=1`:

| | value |
|---|---|
| `g Q > z{0} Q` | 3.1783208585e-04 |
| `g Q > z{T} Q` | 1.0612625837e-01 |
| `g Q > z Q` (unpolarised) | 1.0644409045e-01 |

the two polarised pieces summing exactly to the unpolarised one.

**So `2b151185d` should land together with this change, not after it.** The
guard here opens the door; that commit is what makes what is behind it
correct. Its loop-side caller (`process_checks.py:2111`,
`generate_loop_matrix_element` — the `check timing/stability/profile` path)
is a pure data carry but is **unverified at runtime** on either branch, and
it is exactly the combination this guard newly admits. Worth one polarised
`check timing` run before relying on it.

Two further things to know before using `check` on these processes, neither
of them caused by this work:

- `run_multiprocs_no_crossings` builds its legs with no `flavor` key, so
  `_flavor_group_pos` returns 1 for every leg and **only flavour index 1 is
  actually gauge-checked** once flavour grouping is on (the default). A
  passing `check` covers less than the summary line suggests;
  `set apply_flavor_grouping False` restores the per-flavour table.
- FD/axial gauge on a process with a cross-base FLV routine (`FFVx_FFSy`,
  classically massive-lepton Goldstone-Yukawa) was broken in every backend
  until `6a8530f84` on `claude/bold-mestorf-777ea7` — the call site emitted
  `FFV2_FFS3_0` against ALOHA's `FFV2_FFS3M_0`. It does **not** affect the
  processes in this document: grepping the generated `p p > z{0} j [QCD]`
  tree finds no cross-base combined routines at all, so `tags` is empty and
  the fix is a no-op here. Checked rather than assumed.

Still refused, each for a reason rather than by omission:

- `[QED QCD]` and `[QED]` in the boosted modes. The frame wrappers are
  order-agnostic and the QED counterterms (`particle_charge`, the charge
  links, `sreal_deg`) go through the same `sborn_frame`/`sborn_sf_frame`, so
  this is likely to work — but nothing in the QED sector has been run, and
  the plan does not cover it. `[virt=QED]` is fine, being standalone.
- `[noborn=QCD]` with a *massive* polarised particle. Pre-existing and
  unchanged: loop-induced is exported through the LO madevent template and so
  already has `me_frame`, but `check()` has always rejected massive there.
  Worth revisiting; not touched here because nothing in this work exercises
  it. See section 1. **Superseded in M6**, which is where it was revisited:
  loop-induced now has no restriction at all.

`tests/unit_tests/interface/test_cmd.py::test_check_generate` encoded the old
restriction (`'u u~ > w+{L} [QCD]'` in the *invalid* list) and is updated: the
massive-colourless QCD forms and both `[virt=…]` spellings move to the
accepted list, `[QED QCD]` and `[QED]` stay in the rejected one. It was that
test that caught the first version of this change quietly opening mixed-order
NLO.

#### What this milestone cost, and why

The instrumentation-over-reasoning lesson from M2 held again, and more
sharply. Every hypothesis on the list — the helicity restriction,
`ImprovePSPoint`, `NRotations`, quadruple-precision promotion — was internal
to MadLoop, because the one misread number said "MadLoop is computing
something else". The bug was one line above the call, in the driver, in code
this project had already touched. Both times the search went to the most
intricate component available, and both times the answer was in the plumbing.

Original plan for this milestone:


Boost `p_born` before `Call BinothLHA(p_born,…)` (`fks_singular.f:7087`).
MadLoop already handles polarised MEs (`loop_exporters.py:1552` restores the
helicity-averaging factor). Note MadLoop has its own stability rescue that
re-evaluates in rotated/boosted frames — check it does not fight the imposed
frame.

**Gate:** `check_poles` must still pass (the poles are proportional to the
Born, so a Born/virtual frame mismatch shows up as a pole mismatch). Then
`p p > z{0} z{0} j [QCD]` end to end.

### M4 — Unblock and document — **DONE**

`help_polarization` now covers fixed-order NLO: which modes accept a polarised
massive particle and where each gets its frame. Two things it previously got
wrong are corrected -- `nhel=1` was presented as a requirement when it is only
a variance choice (measured: 1369 +- 1.7 against 1370 +- 1.7), and the
QCD-only restriction on the run_card modes was not stated at all, so a reader
would have assumed `[QED]` worked. It also warns that a frame must be built
from final-state particles, an initial-state one not being infrared safe.

The acceptance test `test_polarised_nlo_me_frame` covers
`p p > z{0} z{0} j [QCD]` with `me_frame=[3,4]` and asserts on `check_poles`
rather than on a cross-section, since the poles are proportional to the Born
and are what a frame mismatch actually breaks. Run and verified to do real
work, not to pass vacuously: 12 P dirs, 12 `check_poles.log`, zero
miscancellations, `generate p p > z{0} z{0} j [QCD]` and `[3, 4] = me_frame`
read back from the generated cards, 3.084e-01 +- 3.0e-03 pb.

Wired into CI as `acceptancetest_105` in
`.github/workflows/acceptancetest.yml`, modelled on the existing
`test_calculate_xsect_nlo` job. Without that the test would never have run:
the workflow invokes tests by explicit name.


Done in M3: the guard now accepts `loonly`, `real` and `all`, deriving the
mode the same way the parser does (no keyword in the brackets means `all`),
and `check()` no longer rejects a massive particle in those modes. The stale
commented-out `order != 'qcd'` guard is deleted, along with the dead `order`
it parsed.

Remaining:

- `help_polarization` (`:803`) does not say `me_frame` is LO-only, but it
  does not say it works at fixed-order NLO either — add that, and say which
  modes accept it;
- the acceptance test in section 4 is written up but not yet added to
  `tests/acceptance_tests/test_cmd_amcatnlo.py`;
- decide whether `[noborn=QCD]` should accept a massive polarised particle.
  It is loop-induced, exported through the LO madevent template, so it has
  `me_frame` already; `check()` has always refused massive there and nothing
  in this work exercises it.

### M5 — NLO+PS (MC@NLO matching) — **DONE**

Fixed order was already covered by M3. This milestone is about
`generate_events`, i.e. the MC@NLO-matched sample, which adds two things the
cross-section run never touches: the MC (shower) counterterms, and the event
record written to the LHE.

**Nothing blocked it.** No guard in `madgraph_interface.py` or
`amcatnlo_run_interface.py` distinguishes fixed order from NLO+PS for a
polarised process, and `generate_events aMC@NLO --parton` ran to completion on
the unfixed code. The gap was silent and numerical.

#### The gap

`get_mbar` (`montecarlocounter.f:2820`) takes `borntilde` from `ans_cnt(2,·)`,
which `sborn_frame` filled with the Born evaluated in the me_frame, and then
multiplies it by an azimuthal phase built entirely in the partonic c.m.:
`azifact` from the unboosted emission spinors, and `cphi_mother=+-1` from the
mother lying on the beam axis. `borntilde` is the +/- gluon-helicity
interference, so the mother's little-group phase enters it; that phase cancels
against the one in the azimuthal factor **only if the two are computed in the
same frame**. They were not.

The FKS collinear counterterms had exactly this problem and it was solved in
M2 with `azifact_me_frame`. The MC counterterms are different in one useful
respect: they are evaluated at generic `y_ij_fks`, where `<ij>/[ij]` is a
regular function of the momenta rather than the `0/0` limit B1 is about, so it
can simply be recomputed on boosted momenta. No new covariant machinery is
needed away from the exactly-collinear point, and `azifact_from_kperp` covers
that point.

#### The fix

Three new routines in `boost_to_frame.f`:

- `me_frame_born_boost` — the boost `sborn_frame` would apply to these Born
  momenta, exposed so the phase can use the same one;
- `azifact_from_spinors` — `<ij>/[ij]`, the expression the legacy branches
  build inline, factored out so the boosted branch can reuse it;
- `azifact_mc_frame` — `azifact` plus the mother azimuth, both in the
  me_frame, returned in the *legacy convention* so the caller's multiply is
  textually unchanged and unpolarised runs stay bit-identical by construction.

Both `get_mbar` sites (ISR `ileg=1,2` and FSR `ileg=3,4`) gain an
`if (me_boosted) … else <legacy verbatim> endif`. The `R_y(pi)` flip of the
`j_fks=2` ISR branch is **skipped** when a frame is requested — it only ever
existed to put the mother back on `+z`, which is meaningless once the mother
is off the beam axis, and B3 already said it must be deleted rather than
boosted. Skipping it also removes a second hazard that only bites a polarised
run: for a one-leg `me_frame` the selected leg sits at rest, HELAS quantises it
along the frame `z` axis rather than along its own momentum, and a rotation
therefore does *not* carry the polarisation axis with it. Harmless for `{0}`
(`eps_L -> -eps_L`), but it would swap `{+}` and `{-}`.

`montecarlocounter_alt.f` gets the same edit. It is dead code — neither
`export_fks.py` nor `makefile_fks_dir` mentions it — but the plan named it and
leaving the two copies inconsistent is a trap.

**The collinear branch is taken at `kt` exactly zero and nowhere else.**
`azifact_mc_frame` can reach `<ij>/[ij]` two ways: recompute it from the
boosted spinors, or rebuild its collinear limit from `xij_kperp` through
`azifact_from_kperp`. The first is exact for any `kt>0`; the second is exact
only at `kt=0`, and its error away from there was measured in review to scale
as `(kt/E)/betaT`, with `betaT` the *transverse* velocity of the me_frame
boost — 1.1e-3 at `kt/E=1e-6, betaT=1e-2`, but 1.36 at `betaT=1e-5`. The
former `betamin` guard (removed, see B9) bounded `|beta|`, not `betaT`, so it
never covered this. The condition is therefore `1-y_ij_fks <= 0` rather than
the legacy `< 1d-12`: at `y=1` the spinor product is genuinely `0/0` and the
`kperp` route is the only one available; everywhere else the exact route is.

Not reachable in the channel validated here (`ptj 30` forces the Z's pT above
30 GeV, so `betaT` is O(1)), but reachable whenever the frame-defining system
can have small transverse momentum. The change is inside `if (me_boosted)`, so
the no-frame path is untouched. `azifact_me_frame`, which the fixed-order
counterterms use and which is *always* handed exactly `y=1`, is deliberately
left alone: there the `kperp` route is not an approximation but the only
correct one (B1).

The identity that makes the two routes agree,
`lim_{kt->0} <ij>/[ij] = -exp(2 i psi) exp(2 i phi_m)`, is what the
convergence table below measures — the generic branch is its left-hand side
and the reference its right-hand side.

Measured cost of the change: none. A freshly generated directory carrying it,
run at `me_frame=[3]` with the same card and seed, reproduces `res_0.txt`,
`res_1.txt` and `res_2.txt` byte for byte against the run made before it
(ratio 1.999850 both ways), and the LHE differs only in the banner's git hash
and output path. The same directory at `me_frame=[1,2]` is likewise byte for
byte identical to the **unmodified base branch** (1.748300e+03 pb, ratio
1.971314), so the no-frame guarantee survives the follow-up. Neither is a
surprise -- production kinematics never reach `1-y = 0`, and the only caller
that passes exactly `y=1` is the counter-event -- but it is cheap to check and
the branch condition is the kind of thing that is easy to get subtly wrong.

Read the byte-identity as structural, not as coverage: `xmcsubt_wrap` has no
production caller at all (all five call sites are in `test_soft_col_limits.f`),
so after this change the `kperp` branch is unreachable in an integration. The
change removes the small-`betaT` exposure rather than measuring it.

#### What was measured

*The azimuthal phase itself.* A debug print in `get_mbar` next to a call to
`azifact_me_frame`, which is the covariant collinear reference validated in M2,
run through the collinear scan of `test_soft_col_limits` on
`p p > z{0} j [QCD]`, `me_frame=[3]`. The quantity is the net factor
multiplying `borntilde`; the reference is exact only as `y_ij_fks -> 1`, so
what matters is whether the difference goes to zero.

| `1-y` | ISR, before | ISR, after | FSR, before | FSR, after |
|---|---|---|---|---|
| 1e-1  | 156.59 deg | 10.07 deg | 0.1920 deg | 0.0371 deg |
| 1e-3  | 156.59 deg | 0.97 deg  | 0.0352 deg | 0.0050 deg |
| 1e-5  | 156.59 deg | 0.097 deg | 0.0036 deg | 0.0005 deg |
| 1e-10 | 156.59 deg | 0.0003 deg| 0.0000 deg | 0.0000 deg |

(`P0_uux_z0g`; `P0_gu_z0u` gives constant errors of 37.38 and 63.38 degrees on
its two scans.) Before the fix the ISR phase error is a **constant** — it never
converges, at any depth. After it, it falls like `sqrt(1-y)`, i.e. like
`k_T/E`, which is the right rate for an azimuth approaching a collinear limit.

FSR was already nearly right here and gets better by a factor 5. That is an
accident of the channel, not a general statement: the reduced Born of
`p p > z j` is 2 -> 2, so the `me_frame=[3]` boost is almost along the FSR
mother, and a boost along the mother has no Wigner rotation. A process with a
2 -> 3 reduced Born would not be so lucky.

*The physics.* `generate_events aMC@NLO --parton -f`, 1000 events,
`iseed=33`, nn23lo1, fixed scales 91.188, `ptj 30`, `etaj 4.0`, `me_frame=[3]`
(`FRAME_ID = 8` read back from `Source/run_card.inc`):

| | before | after |
|---|---|---|
| total cross-section | 2.172e+03 +- 8.3e+00 pb | 2.189e+03 +- 8.2e+00 pb |
| absolute cross-section | 4.837e+03 +- 1.2e+01 pb | 4.378e+03 +- 9.8e+00 pb |
| ratio | 2.227 | 2.000 |

**The total is the observable that must not move and does not** (1.4 sigma):
the MC counterterm enters the S event with one sign and the H event with the
other and cancels in the sum, which is the defining property of MC@NLO. The
absolute cross-section is the one that must move and does, by 30 sigma: it
measures how well the subtraction cancels *locally*, which is exactly what the
azimuthal phase controls. A wrong phase makes the counterterm less local, and
the price is a 9.5% larger `int |dsigma|`, i.e. more unweighting work and a
noisier sample for the same cross-section.

That pair — total unchanged, absolute cross-section down — is what the
acceptance test asserts on. A cross-section comparison alone would have seen
nothing.

Repeated with `iseed=7717` on a freshly generated pair of directories, to make
sure the ratio is not a property of one MINT grid history: 2.2220 before,
2.0060 after, with the totals agreeing at 0.02 sigma (2.1868e+03 +- 8.8e+00 vs
2.1871e+03 +- 9.5e+00). The threshold in the test is 2.15.

*A two-leg frame.* `p p > z{0} z{0} j [QCD]` with `me_frame=[3,4]`
(`FRAME_ID = 24`), same cuts and seed, `generate_events aMC@NLO --parton -f`:
3.134e-01 +- 2.2e-03 pb over 1000 events, with `test_ME`, `test_MC` and
`check_poles` (20/20) passing in all 12 P dirs and zero miscancellations. That
agrees at 1.3 sigma with the fixed-order 3.084e-01 +- 3.0e-03 pb of M4. No new
assertion is added for it: with `nsel >= 2` `boost_to_me_frame` never runs its
`nsel.eq.1` zeroing, so no leg sits on the HELAS at-rest branch and the two-leg
case is structurally the safer of the two. The run is here as coverage.

The full shower step could not be exercised on the development machine: the
`MG5aMC_PY8_interface` there was installed against MG5aMC v3.5.15 and its C++
link now fails (`ld: symbol(s) not found for architecture arm64`). That is a
local toolchain problem, not a polarisation one -- it happens after
`events.lhe` is written, and the run that hit it had already produced its 1000
events at 2.212e+03 +- 8.2e+00 pb. Everything under test here happens before
the handover to PYTHIA8.

#### The test was checked against the unfixed code

`test_polarised_nlo_ps_me_frame` was run twice through `./tests/test_manager.py`:
it passes on the branch (399 s) and, with `montecarlocounter.f` alone reverted
to its pre-M5 state, fails on exactly the intended assertion with
`AssertionError: 2.22732091828047 not less than 2.15`. That reproduces the
by-hand measurement above to every digit, so the test really drives the code
path it claims to.

*No frame requested stays bit-identical.* Same process, `me_frame=[1,2]`
(`FRAME_ID = 6`, the partonic c.m., which `get_me_frame_boost` skips), run
twice from two copies of one output directory differing only in
`montecarlocounter.f`/`boost_to_frame.f`: `res_0.txt`, `res_1.txt`,
`res_2.txt` and the 2.1 MB `events.lhe.gz` all compare byte for byte identical
(1.748e+03 +- 1.0e+01 pb both times). Note this run also shows the frame is not
cosmetic: the same process with `me_frame=[3]` gives 2.189e+03 pb.

#### Audited and left alone

- Every ME entry point in the shipped NLO template already goes through a
  frame wrapper (the M2 step 1 sweep); a fresh grep for `call sborn`,
  `smatrix_real`, `sborn_sf`, `extra_cnt`, `BinothLHA`, `sreal`, `compute_born`
  finds nothing outside them except the EW Sudakov files and
  `symmetry_fks_v3.f`, both excluded on purpose in M2.
- `add_write_info.f` (the LHE record) uses `sborn_frame` already — B7.
- `reweight_xsec_events.f` does not evaluate matrix elements; it reweights
  stored contributions with PDFs and scales, which are frame-independent.
- The shower-scale machinery (`assign_emsca`, `assign_ref_scale`,
  `compute_pTparton`) and `fill_MC_mshell.f` are kinematic only.
- The shower itself is handed an LHE file: momenta and colour, no
  polarisation. As at LO, the polarised sample is defined by the hard process.

#### A pre-existing bug found on the way: `test_MC` is vacuous

`test_soft_col_limits.f` offers three modes. `ilim=2` (`ME/ME(limit)`) is what
`test_ME` runs and is correct. `ilim=1` (`MC/ME(limit)`) is what `test_MC`
runs, and for every scan point after the first it assigns *the same* value to
both sides of the comparison:

```fortran
call xmcsubt_wrap(p,xi_i_fks_ev,y_ij_fks_ev,fx)
fxl(i)=fx*wgt      ! the "limit" side
...
limit(i)=fx*wgt    ! the value being tested
```

`checkres2` then sees a ratio of exactly 1 everywhere and returns PASS
unconditionally. The counter-event `sreal` that `fxl(i)` is supposed to hold is
simply never recomputed for `i>=2`. `ilim=0` has the same shape.

`ilim=0` (`MC/MC(limit)`) is broken twice over: it has the same shape, and it
also stops outright with `Third bit of MCcntcalled should not be set yet` as
soon as it reaches the collinear section, because the driver does not reset
`MCcntcalled` between the counter-event and the real-event calls. Verified to
happen identically on unmodified code with no frame at all. The split-order
sub-check of both modes is broken in a third way, pointed out in review: it
compares the same matrix element under two different jacobians
(`amp_split_mc*jac_cnt(1)` against `amp_split_mc*wgt`), which prints a
convincing-looking convergence that means nothing.

This is on `origin/main`, is not related to polarisation, and is **not fixed
here**: the MC counterterms only reproduce the real emission inside the
shower's phase space, so switching the comparison on would very likely fail for
unrelated processes and is its own piece of work. It is recorded because it
explains why `test_MC` passed on the broken code, and why the M5 validation had
to be built by hand.

### M6 — The three regimes, and colour — **DONE**

#### The three regimes get three different checks

One flag used to gate all of NLO polarisation, and the two halves of it
disagreed: the first admitted `noborn`/`sqrvirt` explicitly, the second
refused them again for being massive, so the first clause was dead and
loop-induced polarisation was refused outright — including
`g g > z{0} z{0} [noborn=QCD]`, the case this milestone validated. They do not
share a hazard, so they no longer share a check
(`madgraph_interface.py`, `check_process_format`):

| regime | `nlo_mode` | where the frame comes from | restriction |
|---|---|---|---|
| standalone MadLoop | `virt` | the caller's phase-space point | none, any perturbation order |
| loop-induced | `noborn`, `sqrvirt` | `noborn` uses the LO `boost_to_frame`, exactly like a tree process; `sqrvirt` squares one amplitude standalone | none, any perturbation order |
| subtracted | `all` (= a bare `[QCD]`), `real`, `loonly` | `me_frame` in the run_card, threaded by hand through Born (M1), real + FKS counterterms (M2), virtual (M3) and the MC-counterterm azimuth (M5) | QCD only, and no *massless* coloured polarised particle |

The mode is read out of the brackets the way the parser reads it (no keyword
means `all`), not by a substring test on the process line, so a stray `real`
elsewhere in the process cannot match. Everything reaching the per-particle
walk now has a frame, so the old mass check there was unreachable and is gone.

Verified that `g g > z{0} z{0} [sqrvirt=QCD]` is not merely admitted but
generated: `P0_gg_z0z0` with `NCOMB=4` (2x2 gluon x 1x1 longitudinal Z)
against `NCOMB=36` unpolarised. `[noborn=QCD]` outputs madevent.

#### Colour: massive yes, massless no

The subtracted regime is the only one that restricts colour, and it now
refuses a polarised particle only when that particle is **both colour-charged
and massless**:

```python
if p.get('color') != 1 and p.get('mass').lower() == 'zero':
    raise self.InvalidCmd(...)
```

The worry the restriction was written for is that a coloured polarised
particle is also an FKS emitter, so its polarisation axis and the
subtraction's singular regions meet somewhere the boost threading was never
validated — the threading having been validated on colourless massive bosons
(`p p > z{0} j [QCD]`, `z{0} z{0} j`).

**For a massive emitter that was measured, and it does not bite.** A closure
study of `p p > t t~ [QCD]` at 13 TeV was run on this branch's tip
(`54dc8a361`) with **exactly one** guard bypassed — this colour check —
in all four top-helicity combinations plus the unpolarised reference, at
fixed order and at NLO+PS, `me_frame = [3,4]` (the `t t~` rest frame,
`FRAME_ID = 24`, read back out of `Source/run_card.inc` after every run).
`check_poles` and `test_ME` ran exactly as shipped, at shipped tolerances.
The full write-up is `docs/nlo_polarisation_massive_colour.md`.

The top is spin-1/2, so `{+}`/`{-}` on each of `t` and `t~` is the complete
spin sum and `sum(4) = unpolarised` is an exact identity up to Monte Carlo.

| | `sum(4) - unpolarised` | relative | pull |
|---|---|---|---|
| NLO fixed order, 6 seeds/config | +0.0050 +- 0.196 pb | **+0.0006 %** | **+0.03 sigma** |
| NLO+PS, parton level, 6 seeds/config | -1.66 +- 1.45 pb | -0.19 % | -1.15 sigma |

Differentially the ratio `sum(4)/unpolarised` sits on 1 across pT(t), y(t),
m(t t~), pT(t t~) and dphi(t, t~). `sigma(++) = sigma(--)` and
`sigma(+-) = sigma(-+)` are reproduced to 5 digits, unforced.

`check_poles` cancelled **20 points out of 20 in every P directory of every
configuration** at the shipped tolerance `1.0e-5` (`IRPoleCheckThreshold`
untouched). The worst relative miscancellation actually attained was 3.1e-13
(double pole) and 5.1e-12 (single pole) for the polarised runs, against
6.8e-14 and 2.9e-12 unpolarised — the same range, eleven orders inside
tolerance. The poles are proportional to the Born, so a polarisation /
subtraction mismatch would surface here first (that is exactly how the M3
blocker gave itself away).

`test_ME` (= `test_soft_col_limits`, mode 2, 100 soft and 100 collinear points
per FKS configuration) failed nowhere. The configurations that matter are the
ones whose emitter `j_fks` **is** the polarised top — the exact case the guard
was written for. They were isolated: **30 such configurations across five
polarisation states, 28 at failure fraction 0.00 and 2 at 0.01**,
indistinguishable from the unpolarised run's own fluctuations.

`test_MC` reported a perfect 0.00 everywhere and is **not** quoted as
evidence: it is structurally vacuous on this codebase (M5), and these logs
corroborate that — a test that can fail gives 0.01-0.05 on the same points.

Why it turns out not to bite: the FKS counterterms multiply the **Born**, and
the polarisation projection acts on that Born through the same frame wrappers
the subtracted path already threads for a colourless massive boson. Colour
enters the subtraction only through colour-linked Borns and the FKS partition,
neither of which touches the spin projection.

#### A prerequisite the study could not have caught

Making a coloured particle polarisable also makes it an **FKS emitter**, and
`fks_common.split_leg` used to build the daughter legs from a bare dict, so
they took `polarization = []` and `insert_legs` replaced the polarised Born
leg with an unpolarised one: a polarised Born subtracted against a
helicity-summed real. `p p > t t~` never showed it, because reals are
deduplicated on PDGs alone and the ISR-generated real — in which the top *is* a
spectator and does keep its polarisation — is enumerated first and survives.
`e+ e- > t{+} t~ [QCD]`, which has no ISR QCD splitting, did: the launch aborted
on `test_ME` with `Soft test FAILED, fraction 1.00`. Fixed by
`fks_common.carry_polarization`, which puts the mother's polarisation on leg
`j` and **raises** when `j` is not the same particle as the mother — the
massless case. Details and the before/after numbers:
`docs/nlo_polarisation_massive_colour.md`. `p p > t t~ [QCD]` generates the
identical reals and the identical amplitude sharing before and after, so every
number in this section stands.

#### Three limitations, which is why this narrows rather than removes

1. **A coloured particle in the INITIAL state stays refused, whatever its
   mass.** The initial-state splitting is read backwards,
   `g -> q(-> Born) q~`: the polarised parton that enters the Born is an
   *internal* line of the real emission, so there is no external leg to carry
   the projection and nothing for `get_frame_mask_real` to select. That is a
   statement about the topology, not about the mass, which is exactly why the
   massless-only rule below does not catch it: `b{+} b~ > h [QCD]` has a
   massive `b` in the default `sm`, used to pass `check_process_format`, and
   then died inside generation with a raw
   `fks_common.FKSProcessError: Cannot carry the polarization [1] of the born
   leg with id 5 through the splitting into ids [21, -5]`. On `0198d8a92` it
   generated and *silently dropped* the polarisation in 2 of its 3 reals.
   `e- b{+} > e- b [QCD]` — DIS with a massive `b` — is the same failure on a
   more realistic process, and is likewise a clean `InvalidCmd` now.
   Note also that the closure study above had its polarised legs in the
   **final state only**, so there is no evidence for an initial-state one
   either. The refusal is per leg and wholesale: the same initial-state quark
   also has the perfectly well-defined `q -> q g` splitting, but `split_leg`
   builds every splitting of the leg and the backward one raises, so the leg
   is refused as a whole rather than channel by channel.
   A **1 → N decay is exempt**: `fks_base.find_reals` skips initial-state
   splittings for a decay process, so `t{+} > w+ b QED=1 [QCD]` and
   `h > b{+} b~ QED=1 [QCD]` are accepted and their reals do carry the
   polarisation. `check_process_format` recognises a decay as "exactly one
   particle before the `>`".
2. **Massless coloured is untested, so it stays refused** — and it is no
   longer only an evidence gap. `fks_common.carry_polarization` can only put
   the Born's polarisation on the real's leg `j` when `j` is the same particle
   as the mother; `g -> q q~` has no such correspondence, and
   `boost_to_frame.f`'s real-side frame mask relies on the same equality.
   Lifting the refusal therefore needs a definition of what the polarisation
   even *means* across the splitting, not just a closure run. If one is found,
   the natural test is a closure where a genuine collinear singularity sits on
   the polarised leg. A massive `j_fks` has none — the soft limit was the
   whole of the test above.
3. **Only a two-leg frame was run.** `me_frame = [3,4]` never triggers
   `boost_to_me_frame`'s `nsel == 1` zeroing, so no leg sits exactly at rest.
   A **single-leg** frame on a coloured polarised particle (`me_frame = [3]`
   with leg 3 coloured) is untested, and it is precisely the configuration
   that exercises the HELAS at-rest quantisation axis — the mechanism behind
   B8, B11 and the `improve_ps` last-leg bug fixed in PR #91. Anyone relying
   on that combination should run it first.

Unchanged and deliberate: `[QED]` and mixed orders stay refused in the
subtracted regime (nothing in the QED sector has been validated), `[tree=…]`
stays refused (no frame), and loop-induced and `virt` stay unrestricted.

#### A flavour-scheme consequence that will surprise someone

The predicate reads the *model's* mass, so the same process string is accepted
or refused depending on which model is loaded:

| model | `b` mass | `p p > b{+} b~ [QCD]` |
|---|---|---|
| `loop_sm` (4-flavour, massive b) | `mdl_MB` | accepted |
| `loop_sm-no_b_mass` (5-flavour) | `ZERO` | refused |

That is correct — a massless b *is* the untested case — but it is worth
knowing before someone reports it as a bug.

The predicate also reads the **symbolic** mass, once, at generation time.
`import model loop_sm` then `p p > b{+} b~ [QCD]` is accepted because the
model says `mdl_MB`; setting `MB = 0` in the param card afterwards puts the
run squarely in the refused regime and nothing re-checks it. Stated in
`help polarization`.

#### `p` and `j` are refused through their constituents

The per-particle walk expands a multiparticle and calls the check on every
member, so `p{+}`/`j{+}` are refused through their gluon (and through their
massless quarks). This is the reason the change is safe for the common case
and it is tested rather than assumed:
`test_polarisation_nlo_regimes` asserts on `p{+} p > t t~ [QCD]`,
`p p > j{+} j [QCD]` and — to show the refusal is not keyed on the *gluon* —
a gluon-free `define qlight = u u~ d d~`.

That last case does **not** show the walk reaches every constituent. MG5
orders multiparticle members gluon first, then by `|pdg|` ascending, so
`qlight` is stored as `[2, 1, -2, -1]` and the refusal fires on the **first**
member. In the SM one cannot build a multiparticle whose first member is
massive-coloured and a later member massless-coloured, so "reaches every
constituent" has no test case there.

`tests/unit_tests/interface/test_cmd.py::test_check_generate` and
`::test_polarisation_nlo_regimes` both encoded the blanket colour refusal
(`'u u~ > t{L} t~ [QCD]'` in the rejected list) and are updated: the massive
coloured forms move to the accepted list, the massless coloured and
multiparticle forms are added to the rejected one. Both fail on the base
branch and pass here, and both load a model first — without `import model`
the colour lookup is skipped and any assertion about it passes vacuously.

## 4. Acceptance test

One new test in `tests/acceptance_tests/test_cmd_amcatnlo.py`, modelled on
`MECmdShell.generate` + `self.do('calculate_xsect LO -f')` (`:722`), with the
reference LO numbers produced the way `tests/acceptance_tests/test_cmd_madevent.py:2870`
does it.

| stage | process | mode | assertion |
|---|---|---|---|
| M0 | `p p > z j [QCD]` unpolarised | NLO | xsec unchanged vs. today (regression) |
| M1 | `p p > z{0} j` | `[LOonly=QCD]` vs LO madevent | agree within MC error |
| M1 | `p p > z{0} z{0}` | `[LOonly=QCD]` | `me_frame`-independent (null test) |
| M2 | `p p > z{0} z{0}` | `[real=QCD]` | real boost: `test_ME` passes; xsec != partonic-CM value. **Weak** — see below |
| M2 | `p p > z{0} j` | `[real=QCD]` | **discriminating**: `test_ME` on gluon-initiated channels |
| M3 | `p p > z{0} j` | `[QCD]` | **run**: `check_poles` 20/20 in all 12 P dirs, 2.193e+03 +- 1.2e+01 pb |
| M3 | `p p > z{0} z{0}` | `[QCD]` | **run**: `check_poles` 20/20 in all 4 P dirs, 1.217e+00 +- 6.6e-03 pb |
| M3 | `p p > z{0} z{0} j` | `[QCD]` | **run**: `check_poles` 20/20 in all 12 P dirs, 4.779e-01 +- 3.3e-03 pb |
| M5 | `p p > z{0} j` | `[QCD]` + `generate_events` | **run**: 2.189e+03 +- 8.2e+00 pb, absolute cross-section ratio 2.00 (2.23 before the fix) |
| M5 | `p p > z{0} z{0} j` | `[QCD]` + `generate_events`, `me_frame=[3,4]` | **run**: 3.134e-01 +- 2.2e-03 pb, `test_ME`/`test_MC`/`check_poles` 20/20 in all 12 P dirs |

**Which process tests what.** The two are complementary, but not in the obvious
way:

- `p p > z{0} j` — the Z recoils against the jet already at Born level, so the
  `me_frame` boost is non-trivial and non-longitudinal in *both* the Born and
  the counter-event, including in the singular region. This is the process that
  discriminates the azimuthal phase, and it doubles as the M1 cross-check
  against LO madevent.
- `p p > z{0} z{0}` — the Born boost is the identity (the ZZ system *is* the
  partonic CM), so it is a clean null test at M1 and exercises the real boost in
  the finite region at M2. In the singular region the boost degenerates to
  longitudinal, so it will **not** catch a wrong azimuthal phase.

Keep M1/M2 in the always-run set (fast, no MadLoop) and gate M3 behind the slow
marker. Run via `./tests/test_manager.py`, not pytest.

## 5. Open blockers

- **B1 — `xij_aor` cannot be recomputed after a boost.** See M2 step 4. Drives
  the `k_perp_dir` design. *Confirmed.*
- **B2 — `me_frame` leg remapping** between Born and real multiplicities (D2,
  plus the `skip` logic at `genps_fks.f:1206-1222`, `:1542-1553`).
  Sub-claim resolved: the `2**n` vs `2**(n-1)` convention is **not** a bug —
  `mapid` is `ids(i)=btest(id,i)` (`cluster.f:141`), which matches
  `sum(2**n)` at `banner.py:4869`. The comment at `genps.f:1758` saying
  `sum 2**(N-1)` is stale; fix the comment.
- **B3 — RESOLVED.** The `R_y(pi)` flip is deleted rather than boosted in
  `sborncol_isr` (M2) and skipped in `get_mbar` when a frame is requested
  (M5); `montecarlocounter.f` no longer applies it together with
  `cphi_mother=-1`.
- **B4 — RESOLVED, both parts.** FSR counter-events do collapse onto the
  `vtiny` branch, exactly like ISR (they are handed the literal `one` for
  `y_ij_fks`). And `shybst = O(1-y) -> 0` in the FSR collinear limit, because
  it vanishes iff the mother is massless and `m_mother^2 = 2 E_i E_j (1-y)`.
  So FSR inherits the D3 equivariance property. See the M2 step 0 section.
- **B5 — RESOLVED structurally.** Rather than argue about which callers are
  reachable, every ME entry point was routed through a frame wrapper, making
  the boost a pure function of the momenta passed. Callers that share momenta
  then agree by construction. Worth stressing that this was not a theoretical
  hazard: in M1 a partial conversion produced a silent 3% shift once and the
  `momenta not the same in Born` stop once. See M2 step 1.
- **B6 — no guard against a lightlike/null `me_frame` sum** in `boost_to_frame`
  (`Template/LO/SubProcesses/genps.f:1761`). Pre-existing at LO; not replicated
  into the NLO port: `get_me_frame_boost` stops with a diagnostic instead.
- **B7 — RESOLVED.** `add_write_info.f:278` now goes through `sborn_frame`
  like everything else, so polarised event generation cannot write weights
  computed in a different frame from the cross section.
- **B8 — RESOLVED, and it was the M2 blocker.** A one-leg `me_frame` left that
  leg at rest only to rounding, and HELAS `vxxxxx` picks the quantisation axis
  from the momentum direction unless the 3-momentum is *exactly* zero. Fixed in
  both `boost_to_me_frame` (NLO) and `boost_to_frame` (LO). See M2 step 2.
- **B9 — RESOLVED by PR #111, guard removed.** A boost whose velocity
  degenerates to zero tilted the beams just enough to wreck `sqrt(p(0)+p(3))`
  in the HELAS massless spinors, without reaching the exact on-axis branch.
  It was guarded by `betamin=1d-4` in `get_me_frame_boost`. #111 fixed the
  spinors themselves (the component now comes from `pT^2`, no cancellation),
  in every backend, so the guard is gone; the split-order soft test that
  failed 0.43 without it passes at 0.02. See M2 step 2.
- **B13 — RESOLVED.** The FKS tilde frame differs between sectors by a
  boost along z, so a frame reached by a single boost from it depended on
  the FKS partition. Single-leg frames now go along the beam first
  (Collins-Soper). The old default `[1,2]` (+9.5% on `p p > w+{0} j`,
  +11.3% on `p p > z{0} j` under a change of `fks_b`) is replaced, for `p p`,
  by the rest frame of the massive final state; `[1,2]` set by hand is the
  rest frame of the Born final state, refused when that has a jet. See D3.
- **B12 — RESOLVED here, and it has a twin that must land with it.**
  `ProcessDefinition.__iter__` dropped `polarization` when expanding
  multiparticles, so any polarised MadLoop-standalone process written with
  `p`/`j` came back silently unpolarised. Pre-existing, exposed by unblocking
  `[virt=…]`. Fixed in `base_objects.py` on this branch. The same defect sits
  in `ProcessDefinition.get_process` and in
  `process_checks.run_multiprocs_no_crossings`, both feeding the `check`
  command — fixed on `claude/quizzical-ptolemy-16f0c8` (`2b151185d`), which
  **must be merged alongside this change**, because opening `virt` in the
  guard is what makes `check … [virt=QCD]` reachable for a polarised massive
  particle in the first place. See M3.
- **B10 — RESOLVED, and it was the M3 blocker.** `check_poles.f` had been
  half-converted: its Born line went through `sborn_frame` while the
  `BinothLHA` call one line below still got the unboosted `p_born`, so the
  gate compared MadFKS poles in the me_frame against MadLoop poles in the
  partonic c.m.. This is B5 all over again — two consumers of one kinematic
  configuration disagreeing about the frame — but in a *driver* rather than
  in the integration, which is why the M2 sweep's "route every ME entry
  point through a wrapper" rule did not catch it: `BinothLHA` is not an ME
  entry point, it is a wrapper of one. Fixed at `check_poles.f:228`. See M3.
- **B11 — RESOLVED: rotations about z when a leg is at rest.**
  `NRotations_DP`/`NRotations_QP` re-evaluate the loop at a rotated
  phase-space point and expect `|M|^2` unchanged. The two rotations were axis
  permutations moving z, and a particle put at rest by a one-leg `me_frame`
  has its quantisation axis pinned to the frame z axis, so the rotated
  evaluation was a different polarisation state. It was worse than a false
  flag: on `u u~ > z{0} z{0} [virt=QCD]` with leg 3 exactly at rest,
  `NRotations=2` returned code 420 (EPS), relative accuracy 6.1, and a finite
  part of -2.129e-06 against the correct -4.838e-06 (56% off). `ROTATE_PS`
  (DP and MP) now detects an exactly-at-rest external leg and rotates by
  pi/2 and pi about z instead -- exact permutations that leave that spin axis
  alone. After: code 216, accuracy 6.3e-11, finite equal to the no-rotation
  value to 5e-13. Points with no leg at rest take the old permutations and are
  bit-identical. Regression test:
  `test_polarised_rotation_stability_leg_at_rest`. Same root as B8: the HELAS
  branch at exactly zero momentum.

## 6. Risk

M2 step 4 is the only genuinely hard item; everything else is mechanical. D2 is
the decision most likely to need revisiting, and M1 proves it out cheaply.
