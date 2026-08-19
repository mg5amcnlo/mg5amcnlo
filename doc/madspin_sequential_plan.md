# MadSpin: the density-mode unweighting schemes, the polarisation axis, and the pure-interference mode

Design record for the density spin modes (`PA`, `onshell`, `madspin`/`full`):
why the per-particle ("sequential") accept/reject is exact, what `unweighting =
auto` resolves to and on what measurements, which frame the polarisation braces
are defined in, and how the pure-interference mode works. It is written as a
record of *why the code is what it is*, not as a plan: everything described here
is built unless the text says otherwise.

Code references are to `MadSpin/interface_madspin.py` and `MadSpin/decay.py`;
line numbers are indicative and drift.

**Section numbers are stable and are referenced from the code** (`section 12`,
`sections 13.17 and 13.18`, ...), so sections that were retired leave a gap in
the numbering rather than being renumbered.

---

## 1. Why it is exactly equivalent (and where it is not)

### Notation

`calculate_matrix_element_from_density` (interface_madspin.py:3027) builds, per
production event:

- `density_prod` = rho, over the *joint* helicity index of the n decaying
  particles (dimension = prod_i n_i, with n_i = len(hel_dict[spin_i]),
  `hel_dict = {1:[0], 2:[1,-1], 3:[-1,0,1]}` in MG5 2S+1 convention);
- per decaying particle i, `density_dec_tmp` = D_i (n_i x n_i);
- `density_dec` = tensor product of the D_i;
- `me = density_dec.scalar_multiplication(density_prod)`.

The weight actually used in the accept/reject (interface_madspin.py:3033) is

    wgt = full_me / (production_me * decay_me)
        = <rho, (x)_i Dhat_i> / Tr(rho),        Dhat_i = D_i / Tr(D_i)

(the color / symmetry / `prod_denominators` factors cancel between the numerator
and the two diagonals).

### Production polarisation restricts the double sum

`<rho, ...>` is `sum_i sum_j rho_prod(i,j) rho_dec(i,j)` over the *full*
helicity basis. When the production process carries a polarisation brace on a
particle MadSpin decays (`p p > t{L} t~`, `p p > w+{0} w-{T}`), that particle's
index is restricted:

- a single-state brace (`{0}`, `{+}`/`{R}`, `{-}`/`{L}`) keeps only the diagonal
  `rho_prod(X,X) rho_dec(X,X)` term;
- `{T}` (= `[-1,1]`) keeps the double sum but drops the `0` row and column;
- no brace leaves that index summed in full -- unrestricted runs are bit-for-bit
  unchanged.

The restriction is per decaying particle and combines multiplicatively over the
tensor product, so `t{0} t~{T}` masks index 1 to its `0` diagonal entry and
index 2 to the `-1/+1` block. It is carried by the production `DensityMatrix`
itself (`set_hel_restriction`, a bool row mask cached per
`(basis_id, restriction)`), so every `scalar_multiplication` and every `trace()`
-- including `N_k` in the sequential accept/reject -- applies it without any
call site passing it along. `Tr(rho)` is restricted with the same mask, which is
what keeps `N_0 = Tr(rho)/prod_i n_i` and hence the accept/reject normalisation
untouched, and matches the polarised `|M_prod|^2` the input events were
generated with. The *decay* diagonals stay unrestricted: the decay events come
from the full, unpolarised decay matrix element.

Two things this is NOT:

- it is not a speed-only consistency tweak. `GET_DENSITY`/`GET_ALL_INTER`
  overwrite the decaying particle's helicity from `ALLOW_HEL` for every entry
  they build, so `rho_prod` comes back **fully unpolarised** in those indices
  even for a polarised process -- verified by evaluating `M0_GET_DENSITY`
  directly for `p p > t{L} t~` and for `p p > t t~`: identical entries,
  including `rho(+1,+1)`. End to end, `p p > t{R} t~` before this change gave
  event blocks *byte-identical* to `p p > t t~`: the brace was ignored outright.
- it is not free of a basis reordering. `GET_DENSITY` selects the rows of the
  process' `NHEL` table by matching them against the *first* `ALLOW_HEL`
  combination, and a polarised process has no `NHEL` row outside its
  polarisation. With the default `hel_dict` order (`[1,-1]`, `[-1,0,1]`) a
  `{L}`/`{-}`/`{0}` production matched nothing and handed back an identically
  zero density matrix -- and a zero `rho_prod` rejects *every* accept/reject
  trial, so MadSpin did not fail, it **looped forever** regenerating decay-event
  pools (observed: `p p > t{L} t~` and `p p > w+{0} w-` both spin indefinitely
  on the pre-change code). `_apply_production_polarization` therefore puts an
  allowed helicity first in that particle's basis; the order is untouched when
  there is no brace.

Polarisation on a **decay** line is rejected outright in the density spin modes
(`do_decay`): the braces there would restrict the decay matrix element that
defines the branching ratio, not the density matrix that is contracted.

Validated end to end against the analytic decay distributions (`spinmode` in
parentheses; theta measured in the parent rest frame against the parent's
direction **in the `me_frame` frame**, which is the axis the helicity is
quantised along -- see the next section, which is what fixed that axis):

| production | observable | measured | expected |
|---|---|---|---|
| `p p > t t~` | `<cos>` of e+ from t | +0.015 +- 0.061 | ~0 (unpolarised) |
| `p p > t{R} t~` (madspin) | idem | +0.409 +- 0.045 | +1/3 |
| `p p > t{L} t~` (madspin) | idem | -0.352 +- 0.051 | -1/3 |
| `p p > w+ w-` (madspin) | `<cos^2>` of e+ from W+ | 0.347 +- 0.007 | SM mixture |
| `p p > w+{T} w-` (madspin / onshell) | idem | 0.393 / 0.398 +- 0.007 | 2/5 |
| `p p > w+{0} w-` (madspin / PA) | idem | 0.198 / 0.208 +- 0.005 | 1/5 |

In every polarised run the *un*polarised partner in the same event (the `t~`,
which carries no brace) stayed compatible with zero, confirming the mask is
per particle. A no-brace run is byte-identical to the pre-change code.

Those `w+` rows were measured against the parent's **lab** direction, which at
the time was also the axis MadSpin quantised on -- self-consistent, and the
wrong axis. See below.

### Which frame the brace is defined in (`me_frame`)

A polarised matrix element is **not Lorentz invariant**, so `w+{0}` is only a
statement once a frame is named. MG5 names it in the run_card:

    me_frame = 1, 2      # frame_id = sum(2**n) = 6

Measured, on 200 events of `p p > w+{0} w-` (`SMATRIX` from the library MadSpin
itself builds, evaluated on the LHE momenta and on the same momenta boosted into
the partonic CM):

    SMATRIX(lab) / SMATRIX(partonic CM)
      {0}       min 0.512   max 7.251   mean 1.562
      {T}       min 0.483   max 1.180   mean 0.892
      {0}+{T}   min 1.000   max 1.000   mean 1.000   (|ratio-1| <= 1.3e-8)

Each polarised piece moves by up to a factor 7 between the two frames; their sum
-- the unpolarised `|M|^2`, which *is* invariant -- is unchanged to numerical
precision. That is the cleanest statement of what is at stake: the frame does
not change any physics, it changes **which helicity `{0}` names**.

**MadEvent means the `me_frame` frame, and the default is the partonic CM.**
`auto_dsig_v4.inc:134` calls `boost_to_frame(PP, frame_id, P1)` and skips it
only for `frame_id.eq.6`, because `genps.f` (`x_to_f_arg`, `mom2cx` on
`p(0,-nbranch) = (sqrt(shat),0,0,0)`) already builds `PP` in the partonic CM;
`cm_rap` is carried separately for the rapidity cuts, and `unwgt.f`
(`zboost_with_beta`) boosts to the lab only on the way out to the LHE file. So
the momenta the polarised matrix element sees are the partonic-CM ones, and the
lab momenta in the event file are a *later* z boost. MadSpin's own v1 Fortran
driver agrees: `driver.f:266` calls `boost_to_frame(pfull, frame_id, P2)`
unconditionally, and its copy of `boost_to_frame` has no `frame_id.eq.6`
short-circuit, so it really does boost. Note also that no `me_frame` value can
name the lab frame -- it selects the rest frame of a subset of the external
momenta, and the lab is not one of those.

**The density spin modes did not.** `_frame_boost` opened with

    if self._beampol() is None:
        return None

so with unpolarised beams -- the common case, and every `p p >` run -- the
production and decay density matrices were both built from **lab** momenta. The
justification given when that guard was written (b231141fe) was explicit and, at
the time, correct: *"the frame cannot change an observable here at all"*, because
the contraction `sum_ij rho_prod(i,j) rho_dec(i,j)` is a trace and a boost acts
on it as a unitary change of basis that cancels between the two factors.

The previous section is exactly what breaks that argument. `set_hel_restriction`
is a **projection**, not a change of basis, and a projection does not commute
with one. The guard was safe before the restriction existed and is a live bug
after it.

Measured end to end, 10000 unweighted events per row, `p p > w+ w-` at 13 TeV,
`spinmode madspin`, `decay w+ > e+ ve`, one MadEvent sample per polarisation
reused by both MadSpin variants (`<|y_boost|>` = 1.57 / 1.48, so the lab and the
partonic CM are far apart):

| production | MadSpin | `<cos^2>` on the lab axis | `<cos^2>` on the me_frame axis |
|---|---|---|---|
| `p p > w+{0} w-` | before | **0.1977 +- 0.0021** | 0.2732 +- 0.0027 |
| `p p > w+{0} w-` | after  | 0.2738 +- 0.0027 | **0.1974 +- 0.0021** |
| `p p > w+{T} w-` | before | **0.4017 +- 0.0031** | 0.3783 +- 0.0031 |
| `p p > w+{T} w-` | after  | 0.3646 +- 0.0031 | **0.4019 +- 0.0031** |

(analytic: 1/5 for a pure `{0}`, 2/5 for a pure `{T}`, 1/3 for flat.)

The two rows of each pair simply swap which axis carries the textbook value. The
old code was self-consistent -- it put a clean `sin^2(theta)` on the lab axis --
but the events it was decaying had been generated with `{0}` meaning the
partonic-CM helicity, so it restricted the wrong one. Reading the "before" line
as a helicity decomposition on the axis MG5 actually meant, `0.4 - 0.2 f_0`
gives `f_0 = 0.63` for the nominally 100% longitudinal sample and `f_0 = 0.11`
for the nominally 0% one: roughly a third of the polarisation purity the user
asked for, thrown away by the frame alone. This is a live bug for ordinary
`p p >` runs, not a latent trap.

Two changes, both in `_frame_boost`:

- the guard is now `if self._beampol() is None and not
  self._production_polarization(): return None`. The frame is honoured when it
  can change an observable -- polarised beams, or a brace on a final-state
  particle of the production -- and skipped otherwise, so unpolarised density
  runs keep the bit-for-bit behaviour b231141fe was careful to preserve. The
  brace does not have to be on a particle MadSpin decays: a restricted helicity
  sum over any final-state leg is frame dependent and reshapes `rho_prod`.
- `_, orig_order, _, _ = self.get_pdir(event)` unpacked **four** values from a
  `get_pdir` that has returned five (`pdir, orig_order, prefix, pos, tag`) since
  6ba177c56, i.e. since well before b231141fe. `_frame_boost` therefore raised
  `ValueError: too many values to unpack (expected 4, got 5)` the first time it
  was ever reached. Nothing reached it, because the guard above turned it off
  for unpolarised beams -- so **the whole `me_frame` path in the density modes
  had never executed**, and `polbeam1`/`polbeam2` in a density spinmode crashed.
  The unit-test stub `_FrameStub.get_pdir` returned a 4-tuple, which is why the
  tests were green; it now returns the real 5-tuple.

Cross-checks: the integrated weight is untouched (10.60038 pb for `{0}`,
54.1049 pb for `{T}`, identical before and after and equal to the production
cross section -- the restriction enters `N_0 = Tr(rho)` too, so the normalisation
does not move); an unpolarised run is byte-identical to the pre-change code; and
a brace on a particle MadSpin does *not* decay (`p p > w+{0} w-` with
`decay w- > e- ve~`) runs through the newly-live frame path without incident.

### The partial weight

For a decay ordering sigma, define after k particles are fixed:

    N_k = <rho, (x)_{i<=k} Dhat_sigma(i)  (x)  (x)_{i>k} I_sigma(i)/n_sigma(i)>

so `N_0 = Tr(rho) / prod_i n_i` and `N_n = wgt * Tr(rho)`.

**Rule:** at slot k, accept the candidate decay with probability
`N_k / (C_k * N_{k-1})`; on reject, redraw *that particle only*.

### Two facts

**(a) Telescoping.** A full chain is accepted with probability
`prod_k N_k/(C_k N_{k-1}) = N_n / (N_0 * prod_k C_k)`, which is proportional to
`wgt`. Same target density as the joint test -- the method is exact, not an
approximation.

**(b) Normalisation.** `E_{d_k}[Dhat_k] = I/n_k`, because the decay density
matrix integrated over the *full* decay phase space is proportional to
delta_{hh'} (rotational invariance in the parent rest frame). Hence
`Integral p_uncorr(d_k) N_k dd_k = N_{k-1}`: each slot's conditional is already
normalised.

**(b) is a shared assumption, not a new one.** The current scheme also never
rejects the production event -- the accept/reject loop keeps `production` fixed
and only redraws the decays (`while 1:` in `_run_onshell_loop`,
interface_madspin.py:2325). That is only correct because
`Z(prod) = Integral p_uncorr * wgt dd = 1 / prod_i n_i`, i.e. a constant
independent of the production event -- which is exactly fact (b). Were (b) to
fail, the joint scheme would be biased too (it keeps every production event and
so cannot compensate a varying `Z`). Sequential therefore adds no new
assumption of this kind, and the equivalence in (a) is not conditional on
anything the current code does not already need.

### Where the gain comes from

Not from rejecting fewer production events -- neither scheme rejects any. The
production ME is also already preserved across retries (`prod_density_cached`
at :2324, and `production.me_wgt` inside `get_onshell_evt_and_wgt`). What a
rejection costs today is **all n decay density matrices** and **n decay events**
from the pools; sequentially it costs **one** of each.

Per production event, in `get_density` (f2py) calls -- the dominant cost:

    joint       ~ n / prod_k eff_k      (exponential in n)
    sequential  ~ sum_k 1/eff_k         (linear in n)

With n=2 and eff_k=0.5 that is 8 vs 4; with n=4 it is 64 vs 8. This is the
structural gain, and it is why the feature matters most exactly where MadSpin
is slowest (many decaying particles).

It also explains the ladder of section 5: `1/eff_k` *is* the expected number of
decay events slot k draws from its own pool, so the requested 1.5 / 2 / 2.5 / 3
are precisely per-slot consumption estimates.

### Where it bites

Not fact (b) (shared, see above). The real exposure is:

1. **Per-slot max weights.** n bounds `C_k` to estimate instead of one, each
   from a finite scan: n chances to under-estimate, and an under-estimated
   `C_k` biases silently. This is the main risk -- hence the overflow counter
   of section 6.
2. **Max-weight scan sampling density.** The real chain draws slot k
   conditioned on the *accepted* earlier decays, while the scan samples the
   pool uniformly. The support is the same, so the bound stays valid in
   principle, but the tail is explored with a different density, so the
   estimate is not the same quality as the joint one.
3. **Off-shell / Breit-Wigner mass sampling** in `get_onshell_evt_and_wgt`
   (interface_madspin.py:2949-2965): masses are drawn *sequentially* with
   `full_dqrts -= dec[0].new_mass`, so particle k's mass range depends on the
   earlier draws and `jac` accumulates across particles. In PA (default)
   `density_pole_approximation` is True; the block runs when
   `density_do_reshuffle` (`spinmode == 'PA'`). The jacobian is attributed to
   the slot that draws the mass; the coupling between slots is real but
   benign -- see "PA: mass sampling, jacobian, and kinematic failures" below,
   where it reduces to the restart rule.
4. `fixed_order` counter-events.
5. `density_debug` compares against the full ME and is only meaningful for a
   complete set.

**Scope: every density spin mode** -- `PA`, `onshell` and `madspin`/`full` (the
card default is `madspin`). An `onshell`-only feature would be inert for
essentially every user. `fixed_order` still falls back to the joint test.

### PA: mass sampling, jacobian, and kinematic failures

In PA `density_do_reshuffle` is True, so `get_onshell_evt_and_wgt`
(interface_madspin.py:2949-2965) draws each resonance mass from its
Breit-Wigner, depleting a shared budget (`full_dqrts -= dec[0].new_mass`), and
folds the sampling jacobian into the weight the accept/reject uses. Sequential
therefore has to deal with it. Three separate points, and only the third is a
real constraint:

1. **The trace property survives the mass sampling.** At a fixed mass,
   `Integral dOmega D_k` is proportional to I by rotational invariance, so any
   mass mixture stays proportional to I; and `Tr(Dhat_k) = 1` by construction,
   hence `E[Dhat_k] = I/n_k` *exactly*, whatever the mass distribution and even
   though the budget makes slot k's mass range depend on the earlier draws.
   Fact (b) is safe.

2. **The jacobian is a non-issue in practice.** The mass is generated
   *according to* the Breit-Wigner, so `jac_k` is small and quite flat across
   the phase space -- the effect washes out. Slot k's mass is drawn inside slot
   k's own accept/reject, so `jac_k` attributes to that slot naturally, and the
   per-slot weight is `(N_k/N_{k-1}) * jac_k` -- matching PA today, where the
   Breit-Wigner jacobian is already part of the accept/reject weight.

3. **Kinematically impossible mass sets are the real failure mode, and they
   fail in the *reshuffling*, not the jacobian.** They come in two kinds, with
   different scope and therefore different handling:

   *Decay side* -- the sampled mass cannot accommodate the decay products, e.g.
   `t > b j j` with a top mass below `MW + Mb`. This is local to one slot.
   **Do the decay reshuffling as part of drawing slot k**, right after its mass
   is sampled, so the failure surfaces where it can be retried cheaply: on
   failure **redraw the mass for that decay only, keeping the slots already
   accepted**, and use the jacobian of the draw that succeeds.

   *Production side* -- the drawn masses do not fit the production kinematics,
   e.g. a resonance decaying to two tops whose sampled masses sum above the
   resonance mass. This is a property of the whole mass *set*: it cannot be
   attributed to a slot, and can only be established once every slot has a mass.
   **Do the production reshuffling once, at the last stage.** Carry its
   *jacobian* at each step (below), but never perform the reshuffling itself per
   slot. If that final reshuffling turns out to be impossible, **trash the full
   set of decay events** and restart the chain from the first decay (keeping the
   production event).

   Neither retry touches the decay *angles* -- both reject on masses only -- so
   (1) is untouched: the solid-angle integral at fixed mass is still
   proportional to I, and reshaping the mass mixture is exactly what
   `E[Dhat_k] = I/n_k` is insensitive to.

4. **The production jacobian telescopes, exactly like N_k.** Write `J_k` for the
   production jacobian with the first k decays (in the ordering) put offshell,
   `J_0` being nothing offshell. Then:

   - slot 1 carries `J_1` -- which is precisely what the code computes today
     when a single decay is offshell;
   - slot k carries the **ratio** `J_k / J_{k-1}`.

   The chain multiplies out to `J_n / J_0`, the full production jacobian, so the
   per-slot weights telescope to the joint weight just as `N_k/N_{k-1}` does.
   The complete weight at slot k is therefore

       w_k = (N_k / N_{k-1}) * jac_k^decay * (J_k / J_{k-1})

   and `prod_k w_k` reproduces the joint weight, which is what makes the whole
   scheme exact. `J_k` is the jacobian only -- the production reshuffling itself
   still happens once, at the end, so this needs a way to evaluate `J_k` without
   reshuffling. That is possible, but not currently exposed; see below.

### Evaluating `J_k` without reshuffling (dedicated step)

**Checked: the jacobian can be had without touching the event, but no entry
point offers it.** `Event.reshuffle_production` (lhe_parser.py:3151) takes its
jacobian from `Event.mass_shuffle(old_momenta, sqrts, new_masses)`
(lhe_parser.py:2929), a *staticmethod* returning `(new_momenta, jac)`. It
mutates only the momenta list handed to it, and `reshuffle_production` hands it
`old_momenta = [FourMomentum(p) for p in production if p.status != -1]` -- a
fresh list of copies -- so the event's particles are never touched. The event is
mutated afterwards, by `reshuffle_production` itself.

So `J_k` is reachable, but `reshuffle_production` entangles it with three things
the sequential scheme must not inherit:

1. it applies `new_mom` to the event's particles;
2. it folds in the decay reshuffling (`jac *= self.reshuffle_decay(...)`), which
   in our scheme is `jac_k^decay` and belongs to the slot that drew the mass;
3. on `jac in [0, -1]` it *resamples the masses and recurses* -- its own retry
   policy, which would collide with the per-slot / whole-set rules.

**Dedicated step, before the loop:** add a jacobian-only entry point, e.g.
`Event.production_jacobian(new_masses)` (or
`reshuffle_production(jacobian_only=True)`), which

- does the `split_event_by_onshell_propagator` / `old_momenta` setup;
- returns 1 for the 2 -> 1 case (no phase space for RAMBO to redistribute);
- reports failure when `sum(new_masses) > sqrts`. This *is* the production-side
  kinematic test ("two tops summing above their resonance"), and it is cheap:
  no reshuffling is needed to know a mass set is impossible;
- otherwise returns `mass_shuffle`'s `jac` alone, discarding `new_momenta`,
  without the `reshuffle_decay` factor and without the retry recursion --
  `jac in [0, -1]` is a failure to report to the caller, not to retry here.

`reshuffle_production` should then be re-expressed in terms of it, so that the
two cannot drift apart.

### Mass ownership: what phase 4 has to untangle first

The mass logic is currently spread over three places which do not compose once
the draw has to happen per slot:

1. **The draw** is in `get_onshell_evt_and_wgt` (:2949-2965). It runs on every
   trial, walks `decays` in a single pass, depletes a shared `full_dqrts` and
   accumulates `jac` over all of them at once.
2. **The copy onto the production particles** (`particle.new_mass = ...`) is in
   `calculate_matrix_element_from_density`, *inside* the `prod_static` cache
   guard (:3133). For PA that guard only fires on the first trial (it reads
   `not prod_static or prod_static.get('decays_key') != decays_key`), so on a
   retry the decays carry freshly drawn masses while the production particles
   still hold trial 1's. This looks benign today only because acceptance
   rebuilds the event with `lhe_parser.Event(str(production))`, which drops the
   python attribute -- it must not be inherited by a per-slot rewrite.
3. **The reshuffle** then runs on that rebuilt event, after acceptance for PA,
   with its jacobian discarded.

**Resolved: in PA the decays already own their masses, and (2) is dead code.**
`add_decay_to_particle` (lhe_parser.py:2358) copies `new_mass` and
`reshuffle_info` from the decay event onto the production particle it attaches
it to:

    if hasattr(decay_particle, 'new_mass'):
        this_particle.new_mass = decay_particle.new_mass
        this_particle.reshuffle_info = decay_particle.reshuffle_info

So on acceptance the masses reach `reshuffle_production` through
`Event(str(production)).add_decays(decays)` -- carried by the *decays*, not by
the `production` object. The copy in (2) is therefore:

- **dead** for PA: nothing reads `production`'s `new_mass`, which is why its
  staleness on a retry never showed up;
- **load-bearing** for non-PA only, where `calculate_matrix_element_from_density`
  calls `production.reshuffle_production()` on the production event itself --
  and there the guard is always true, so it already runs on every trial.

Consequences for phase 4, all favourable:

- the mass ownership PA needs is **already correct**: `_draw_offshell_mass`
  leaves `new_mass` on `dec[0]`, `add_decays` carries it to the merged event,
  and `reshuffle_production` / `production_jacobian` read it there. Nothing has
  to be re-plumbed; the copy just must not be extended to the sequential path;
- the basis setup can be split out (or left alone) purely on readability
  grounds. It is **not a blocker and not an optimisation** -- for PA the guard
  already computes it once per production event and reuses it across retries,
  and `production._ms_density_static` is readable by the loop as it stands;
- **`J_k` falls out for free**: `Event(str(production)).add_decays(decays_so_far)`
  has exactly the first k resonances carrying a `new_mass` and the rest at their
  nominal mass, so `production_jacobian()` on it *is* "the production jacobian
  with the first k decays put offshell". That is the definition `J_k / J_{k-1}`
  needs, with no extra bookkeeping.

---

## 2. The key lever: identity substitution (no new tensor algebra)

`N_k` is the *existing* contraction with the not-yet-fixed particles' `Dhat`
replaced by `I/n`:

    density_dec = (x)_i ( Dhat_i if fixed else I_i/n_i )
    N_k         = density_dec.scalar_multiplication(density_prod)

So **no partial trace is needed**. `DensityMatrix.from_components(...)` plus the
cached `_diag_mask` (decay.py:4567+) give an `identity_like()` in a few lines.
`scalar_multiplication`, `tensor_product`, `trace` are untouched, and the
`_tp_hel_cache` / `basis_id` caching keeps working because the basis is
unchanged.

Cost: n contractions per production event instead of 1, but a contraction is
numpy over a prod_i n_i vector while `get_density` is the f2py ME -- the
expensive one, whose call count sequential *reduces*. If profiling later shows
the contraction dominating at large n, phase 3 is a true partial contraction
(fold fixed indices away so later steps act on a smaller tensor). Not needed
for a first cut.

**Slot-order constraint (important).** The tensor slot order must remain the
`position` order (interface_madspin.py:3090) -- the production event's particle
order -- because `helicities`, `init_part`, `allowed_hel` and the whole basis
cache derive from it. `get_decay_from_file` (:2720) also walks the production
particles in that order. The decay *ordering* must therefore only change **which
slot is filled next**, never permute the tensor. Implement sigma as an index
list over the existing (pdg-group, index-within-group) slots; identity fills the
rest.

**Free check:** for a spin-0 parent n_i = 1, so `Dhat_i = I_i = [1]` and
`N_k/N_{k-1} = 1` identically -- a scalar can never be rejected. Good unit test,
and it is why the ladder must not charge scalars (section 5).

---

## 3. The options

The scheme is selected by a single enumerated card option, whose values and the
measurements behind them are in section 10 ("The option: one knob, four
schemes") and section 12:

    set unweighting auto | joint | sequential | sequential_global_retry
                         | sequential_with_mass

`auto` is the default and what it resolves to is section 12. Two companions keep
their own names, being an ordering and a check rather than modes:

- `sequential_spin_order` (default `2 3 1`) decides which particle is
  accept/rejected first -- section 4;
- `sequential_debug` recomputes the joint weight on every accepted chain and
  checks the stage weights against it -- section 10, "the weight identity".

The scheme is forced back to `joint` when the spinmode carries no density
matrix, when `fixed_order` is on, when the decays are grouped with `@` tags
(`doc/madspin_decay_groups.md`), under `pure_interference` (13.4) and under
`decay_output = weighted` (13.18).

---

## 4. Ordering

`_decay_slot_order(decaying_spins)` -> list of slot indices, stable-sorted by
`sequential_spin_order.index(spin)`, ties broken by slot index (keeps the run
reproducible and independent of dict ordering). Default `2 3 1` = spin 1/2
first, spin 1 in the middle, spin 0 last.

Rationale: the first particle sees rho traced over everything else (close to
unpolarised -> mild modulation -> high acceptance); each subsequent one sees a
more conditioned, more polarised parent -> wider ratio spread -> lower
acceptance. Scalars never reject, so they are parked at the end.

---

## 5. Pool sizing ladder

The joint scheme sizes a pool per pdg from a flat per-spin efficiency guess
(1.1 for a scalar, 2.0 otherwise). Sequentially each slot burns its own pool at
its own rate, so the guess becomes a **ladder by position, capped by spin**:

```python
# position k (0-based) in the decay ordering
efficiency = 1.1 if spin == 1 else 1.5 + 0.5 * k     # 1.5, 2.0, 2.5, 3.0, ...
```

- scalars keep 1.1 at whatever position they land (their ratio is identically
  1, so a bigger pool would be pure waste);
- spin 1/2 and spin 1 take the ladder value **at their own index**;
- beyond 4 particles the formula keeps going (3.5, 4.0, ...).

The `+ nevents_for_max` term and `decay_event_mult` are unchanged. Note the
pool is sized per *pdg*, while the ladder is per *slot*: for several identical
parents (same pdg, several slots) take the max ladder value over that pdg's
slots -- the same file feeds them.

This is the one part of the scheme that is a heuristic rather than a
derivation. `1/eff_k` *is* the expected number of decay events slot k draws
from its own pool, so the requested 1.5 / 2 / 2.5 / 3 are per-slot consumption
estimates and nothing more; the per-slot acceptances the run logs are what
would recalibrate them.

---

## 6. Max weights: one bound per slot

The joint scan records one `maxwgt` per production event and combines them as
`1.05 * (mean + nb_sigma*std)` over the per-event maxima, refined over the top
20/30/40/50. Sequentially there are `n` independent bounds `C_k`, one per slot:
for each PS point the scan computes the n ratios `N_k/N_{k-1}` for the sampled
set, tracks the per-event max of **each**, and runs the same statistical
combination independently per slot.

The scan keeps sampling decay sets uniformly from the pool even though the real
chain conditions on earlier accepted decays: uniform sampling explores the same
support, so the max over uniform draws remains a valid estimator of the same
bound. It does change the *sampling density* of the ratio, so the tail estimate
is not the same quality as the joint one -- which is the argument for keeping
the `nb_sigma`/`1.05` margins and for the overflow counter.

The cached bound in `ms_dir` is therefore a vector, not a float, and it is
written under its own file name and format so a stale scalar cache cannot be
read back as one (the up-front schemes go further and carry their `Z_k` tables
in the same file -- section 10, "Implementation").

The per-slot **overflow counter** counts `N_k/N_{k-1} > C_k` and is logged at
the end: the joint path has the same exposure on a single bound, but n bounds
mean n chances to under-estimate, and an under-estimated `C_k` biases silently.
A non-zero count is the first thing to look at when two schemes disagree.

---

## 10. spinmode = madspin (full offshell): the up-front mass draw and `Z_k`

The per-particle decomposition of sections 1-6 needs a production density that
is fixed while the chain is built. Offshell it is not, so the offshell schemes
draw every virtuality up front. This section is the derivation of that split,
of the running-width factor `Z_k` it makes necessary, and of the measurements
that fixed the scheme names and the `auto` rule (section 12).

### Why madspin is different

For `PA`/`onshell` the production density rho is evaluated at the *onshell*
production momenta -- fixed per production event -- and any reshuffling is a
separate kinematic dressing. That fixed rho is what the per-particle
decomposition needs.

`madspin` (density_pole_approximation = False) instead, in
`calculate_matrix_element_from_density` (interface_madspin.py ~3915):
1. computes the denominators |M_prod|^2 and Prod |M_dec|^2 at **onshell**;
2. `production.reshuffle_production()` -- redistributes **all** production
   momenta to fit the sampled masses;
3. reshuffles each decay to its (possibly resampled) mass;
4. computes the numerator density rho at the **reshuffled (offshell)** momenta.

So rho depends on the whole set of decay masses jointly -> not fixed while the
chain is built -> the decomposition does not apply as-is.

### The fix: draw all masses up front

Draw the invariant mass of every decaying particle **before** the per-particle
loop, then reshuffle the production once, up front, and reuse the resulting
fixed offshell rho for the whole chain. Concretely, per production event:

1. For each decaying particle, sample its virtuality from its Breit-Wigner.
2. Reshuffle the production with that full mass set.
   - **Production infeasible** (sum of masses > sqrt(shat), reshuffle returns
     -1): restart from step 1 (redraw the whole set). This validity check
     happens *early*, before any decay is drawn.
3. Compute rho once at the reshuffled momenta (fixed for the chain).
4. Per-particle accept/reject loop, exactly as onshell but: each drawn decay
   event is reshuffled to its particle's pre-drawn mass before its density is
   taken, and boosted to the offshell parent.
   - **Decay infeasible** (the drawn mass cannot accommodate that decay's
     products): the candidate is an ordinary **rejection**, not a restart --
     see "`jac_dec == 0` is a rejection, not a restart" under Implementation
     below, which is where the first version of this scheme got it wrong.

With rho fixed, the loop and its telescoping are the onshell case again.

### Normalisation: onshell denominators with an offshell numerator (resolved)

Fixing rho alone is not enough: joint madspin's weight is

    <rho_off, (x) D_off> * jac / ( |M_prod|^2_on * Prod |M_dec|^2_on )

-- offshell **numerator**, onshell **denominators**. The per-particle test must
divide by the onshell decay ME, not the offshell trace. Olivier's telescoping
does exactly that. Define the partial weight after k particles

    W_k = <rho_off, D1_off (x) ... (x) Dk_off (x) I (x) ... (x) I> *
          (jac_prod jac_1 ... jac_k) / ( |M_prod|^2_on |M_1,dec|^2_on ... |M_k,dec|^2_on )

and accept slot k with probability proportional to W_k / W_{k-1}. The product
telescopes to W_n / W_0; W_n is the joint madspin weight and W_0 is a per-event
constant, so the chain samples the joint distribution -- exact.

In the ratio W_k/W_{k-1} the `|M_prod|^2_on`, `jac_prod` and the slots < k all
cancel, leaving

    W_k/W_{k-1} = [ P_k / P_{k-1} ] * jac_dec_k / |M_k,dec|^2_on,
    P_k = <rho_off, D1_off (x) ... (x) Dk_off (x) I (x) ... (x) I>

i.e. the existing N_k/N_{k-1} contraction evaluated with **offshell** rho and
Dk, times the decay-reshuffle jacobian over the **onshell** decay ME. This
unifies with what is built:

    onshell / PA:  Dhat_k = D_k^on  / Tr(D_k^on)
    madspin:       Dhat_k = D_k^off / Tr(D_k^on)   -- same denominator, offshell top

So the only extra cost is one ME evaluation per decay: `D_k^off` (numerator,
after the reshuffle) and `Tr(D_k^on) = |M_k,dec|^2_on` (denominator, before it).
The un-drawn-slot identity keeps whatever per-particle constant (the
offshell/onshell rate ratio) it carries; that is absorbed into `C_k`, so plain
`I` is correct there.

Why the up-front mass draw is exact, not an approximation: the mass is drawn
from its Breit-Wigner, and the BW sampling jacobian sits in the weight (jac_k),
so the BW *prior* cancels the BW *jacobian* and the accepted events' mass
distribution is the true physical marginal `Integral physical(m,Omega) dOmega`
-- even though the mass is fixed for the chain and only the decay angles are
accept/rejected.

### Why there is a mass-set stage at all

The offshell path is `_upfront_production` (up-front mass draw + reshuffle of a
copy + fixed rho) plus the `offshell` branch of `sequential_accept_reject`
(offshell density on a copy of the decay so the drawn decay stays onshell for
the final add_decays + reshuffle; weight
`(N_k/N_{k-1}) * jac_bw_k * Tr(D_k^off)/|M_k|^2_on`).

Without a stage of its own for the mass set, that scheme is **slower than the
joint test on ttbar**: ~340 decay-ME evaluations per event (slot 0 ~313, slot 1
~27) against joint madspin's ~122 (61 trials x 2 decays). The cause: madspin is
inherently peaked (joint itself needs 61 trials/event), and the per-mass-set
production reshuffling jacobian `jac_reshuffle` plus the offshell weight tail
land in **slot 0's per-angle accept/reject**. Since the mass is fixed per chain,
an unlucky mass draw cannot be escaped by redrawing angles, so slot 0's bound
(max weight ~322) is huge and its acceptance ~1/313.

Hence the mass-set-level accept/reject, a step before the per-angle loop with
weight `w_mass = Tr(rho_off) * jac_reshuffle * prod jac_bw_k` (normalised by
`|M_prod|^2_on`, see below) and the per-angle factors reduced to
`(N_k/N_{k-1}) * Tr(D_k^off)/|M_k|^2_on`. It isolates the reshuffling jacobian,
its bound is modest (`C_mass` ~ 14 on ttbar before the normalisation, ~3 after)
and -- for physical resonant decays -- it makes sequential madspin **faster
than the joint test**. Validated end to end on `p p > t t~`,
`t > w+ b, w+ > l+ vl` (fully leptonic), same production events, `nb_core=1`:

- efficiency: sequential 5.6 decay-ME evaluations/event (slot 0 = 2.1, slot 1 =
  3.5) vs joint density madspin's 8.9 (4.46 trials x 2 decays);
- physics: cross section 23.7742 vs joint 23.7750 (0.003%), Delta-phi(l+,l-)
  within 0.32 sigma on the full 10000-event dilepton sample.

**The earlier "sequential madspin is hopeless" finding was an artifact of the
inclusive `w+ > all all` decay.** That channel makes density madspin itself
blow up (61 trials/event, vs ~4.5 for the leptonic decay and ~7.8 for
madspin_v1), because the top's *non-resonant* contributions get a huge offshell
reweighting tail (`Tr(D^off)/|M|^2_on`) when the decay is reshuffled over
`BW_cut` = 15 widths. That is a **separate density-madspin issue**, shared by
joint and sequential, and orthogonal to the per-particle factorisation: it hits
`w+ > all all` regardless of which accept/reject is used. For resonant decays
the density mode is efficient and the sequential version improves on it.

What `auto` does with madspin/full was settled later, by the multiplicity scan
of section 12: joint up to two decaying particles, `sequential` from three. The
open item this subsection leaves is the `w+ > all all` non-resonant blow-up in
the density-madspin *weight* (BW_cut too wide for unweighting the reshuffle?
reweighting normalisation? keep those channels weighted?), which would help
joint madspin too and is not a property of any accept/reject scheme.

### Fixed (measured): the mass-set stage missed the per-slot normalisation

The claim above that the un-drawn-slot identity "keeps whatever per-particle
constant (the offshell/onshell rate ratio) it carries; that is absorbed into
C_k" is **wrong when that ratio depends on the sampled virtuality**, which is
the madspin case. The per-angle loop redraws until it accepts, so slot k's
accepted angles are distributed as `p_pool * w_k / Z_k` with

    Z_k(m) = Integral p_pool(Omega) w_k(Omega, m) dOmega ~ Gamma_k(m)/Gamma_k(on)

Redrawing-until-accept divides `Z_k` out, so the accepted *mass sets* are
missing `prod_k Z_k(m_k)` relative to the joint weight -- the running-width
factor. For PA/onshell this cannot happen: fact (b) makes `Z_k == 1`. For
madspin it is a real bias of the two-stage (mass set, then angles) split.

Measured on `p p > t t~`, `t > w+ b, w+ > l+ vl`, 10000 events (probe-mode dump
of `E[w_0 | m]`):

    Z(155) = 0.61   Z(165) = 0.81   Z(173) = 1.00   Z(180) = 1.19   Z(190) = 1.49

and the consequence on the reconstructed top lineshape:

    sequential madspin  <m_top> = 172.937 +- 0.022
    joint madspin       <m_top> = 173.185 +- 0.022     (-7.8 sigma)
    sequential x Z(m_t)Z(m_tbar) = 173.190 +- 0.023    (closure, 0.14 sigma)

i.e. the whole discrepancy is that one factor -- it is not the decay
reshuffling jacobian (added since, and verified chain-by-chain: with it the
sequential product equals `prod(n_i) * |M_prod|^2_on * wgt_joint` to 1.5e-7,
against a 1.4% rms spread without it), and it is not max-weight truncation
(doubling `C_mass` via `nb_sigma = 3` leaves the shift at -0.26 GeV). The mass
distribution sequential madspin produces is the PA/Breit-Wigner one
(`<m_top>` = 172.916 for PA joint), because the decay-side offshell
reweighting of the virtuality is exactly what gets normalised away.

#### What Z_k is, and why it can be tabulated

Working the contraction through, the pool sampling density (proportional to
`|M_dec|^2_on`) cancels the onshell denominator and rotational invariance in the
parent rest frame turns the angular integral of `D^off` into the identity, so

    Z_k(m) = Integral dPhi_off(m) |M_dec|^2 / Integral dPhi_on |M_dec|^2
           = (m/M) * Gamma_k(m) / Gamma_k(M)

-- the `m/M` because this ratio carries no `1/2m` flux factor. Three things drop
out of it: the production event, the other slots' virtualities, and the angles
already accepted. It is a smooth function of **that slot's virtuality alone**,
which is what makes a one-dimensional table per slot the right object. The
narrow-width formula for `t > W b` reproduces the measured values above to 1-2%
(0.601 / 0.807 / 1 / 1.194 / 1.512 against 0.61 / 0.81 / 1.00 / 1.19 / 1.49).

The same derivation gives a cheaper estimator than `w_k` itself: with
`s_k = jac_dec * Tr(D^off) / |M_dec|^2_on`, `E[s_k|m] = E[w_k|m] = Z_k(m)`
exactly -- the density ratio `N_k/N_{k-1}` averages to one at fixed m -- so the
table is built from `s_k`, which carries no polarisation modulation and hence
less variance.

#### An imperfect Z_hat does *not* cancel

The per-angle stage is **invariant** under any rescaling of `w_k` by a factor
that does not depend on the angles: `w_k -> w_k / Z_hat_k` changes its
normalisation to `Z_k / Z_hat_k` and leaves the accepted angles alone. So
whatever weight it is given, it still divides out the *true* `Z_k`, and the
residual bias of a tabulated scheme is exactly `Z_hat / Z`. Accuracy is
therefore a requirement, not a nicety -- unless the per-angle stage is stopped
from normalising at all, which is what `sequential_global_retry` does.

How accurate: the full factor moves `<m_top>` by 0.248 GeV, so a fractional
error `eps` in the slope of `ln Z` leaves `0.25 * eps` GeV behind. Against the
0.031 GeV combined MC error of a 10000-event A/B that is `eps < 12%` -- loose,
and about ten times looser than what the probe delivers.

#### Implementation

`_zhat` / `_build_z_tables` / `_z_slot_keys` (interface_madspin.py). The samples
are free: the max-weight probe already draws `Nevents_for_max_weight *
max_weight_ps_point` = 75 * 500 chains, each giving one `(m_k, s_k)` pair per
slot, so `sequential_accept_reject` records them in probe mode
(`probe_extra`) and `get_sequential_maxwgt` fits the table before combining the
bounds. The fit is a weighted quadratic in `ln(m/pole)` through the *bin means*
(Z is an expectation, so the mean estimates it and the mean of the logarithms
would not), held constant outside the probed range and reported in the log
against the running width it estimates. `C_mass` is then derived from the
completed weights `w_mass * prod_k Z_hat_k` (`_complete_upfront_probe`), which
is why the probe now keeps its chains instead of maxing them online.

Two related points fell out:

- **`jac_dec == 0` is a rejection, not a restart.** The offshell branch used to
  trash the whole mass set when a drawn decay could not be reshuffled onto its
  virtuality. That is a *second* mass-dependent normalisation -- the set then
  survives slot k with probability `Z_k / (Z_k + q_k C_k)`, not `Z_k` -- which
  would defeat the correction near a threshold (it is invisible on `t > W b`,
  where `m_t > M_W + M_b` always). A zero-weight candidate is an ordinary
  rejection; counting it as one makes `Z_k`, which includes those zeros, the
  exact correction again. A virtuality no pool decay can reach is killed by the
  table itself (`zero_below`), with a 200-draw fail-safe behind it.
- The `max_wgt_sequential` cache splits: the up-front-mass bounds travel with
  their tables (and depend on `sequential_global_retry`), so they get their own file
  name and a JSON format (`_read_upfront_cache` / `_UPFRONT_CACHE_FORMAT`). The
  file name still carries the spinmode family that wrote it
  (`max_wgt_sequential_offshell...` / `max_wgt_sequential_pa...`), since the
  mass-set weight is a different quantity in each and neither cache may be read
  back for the other.

#### `sequential_global_retry`: the escape hatch

New option, offshell spinmodes only. The mass stage pays `Z_hat_k`, each slot
divides it back out, and a **rejected decay trashes the mass set** instead of
being redrawn. The chain is then accepted with probability proportional to
`w_mass * prod_k w_k` -- the joint weight -- so `Z_hat` cancels identically and
is reduced to an efficiency preconditioner: exact whatever the table says, or
even with no table at all. The price is that the per-angle stage no longer
recovers from a rejection, so the acceptance falls back towards the joint one
and only the early-exit saving survives (worth little at n=2, more at n>=3). Use
it to bound the residual bias of the tabulated path without needing the joint
run as the yardstick.

#### A/B after the fix

Same 10000 production events, `p p > t t~`, `t > w+ b, w+ > l+ vl`, seed 42,
`nb_core 1`, joint madspin as the reference:

                        <m(l+ v b)>          <m(l- v~ b~)>        lineshape
    joint            173.2024 +- 0.0318   173.1681 +- 0.0318      --
    sequential + Z   173.1278 +- 0.0319   173.1554 +- 0.0318      chi2/ndf 19.0/22
    sequential_global_retry 173.1914 +- 0.0323   173.1906 +- 0.0323      chi2/ndf 10.4/22

Over both resonances that is a shift of -0.044 +- 0.032 GeV for the tabulated
path and +0.006 +- 0.032 GeV for the exact one, against **-0.248 GeV (-7.8
sigma)** and chi2/ndf 75.8/22 before the fix. `m(l+ vl)` and `dphi(l+,l-)`, the
no-regression checks, stay within 0.3-0.6 and 1.0-1.9 sigma.

The -1.66 sigma on `m(l+ v b)` alone is a fluctuation, not a residual. Four
sequential replicas over the same production events with independent MadSpin
seeds (42, 43, 44, 45) average both resonances to 173.1416, 173.1527, 173.2167
and 173.1453 -- 173.1641 +- 0.0177 -- against two joint replicas at 173.1853 and
173.1867. That is a residual of **-0.022 +- 0.024 GeV**, a tenth of what was
there before and consistent with zero.

The joint-vs-joint replica is the control that makes the rest readable: it comes
out at chi2/ndf 21.7/22 on the lineshape and 2.1 sigma on `m(l+ vl)`, i.e. the
scatter between two runs of the *same* scheme is as large as anything the
sequential replicas show (14.8, 18.9, 23.1 on the lineshape; 1.2-2.5 sigma on
`m(l+ vl)`, whose naive standard error is optimistic because the Breit-Wigner
tail reaches 15 widths). Sequential-vs-joint is now indistinguishable from
joint-vs-joint.

(Replicating needs a *fresh factory per seed*: MadSpin seeds its RNG on the
first `set seed` of the card and ignores every later one, so an extra `set seed`
appended to the card silently reproduces the same run.)

The tabulated `Z` is accurate far beyond what is needed: the fit reports
`Z(150.7) = 0.53`, `Z(173) = 1`, `Z(195.4) = 1.71` with bin-to-fit deviations
under 2%, against the narrow-width `(m/M) Gamma(m)/Gamma(M)` values 0.525 and
1.704 -- i.e. under 0.5% on the shape, where 12% would do.

#### The mass weight must be normalised by |M_prod|^2 on shell (Olivier)

The mass-set weight above was written `Tr(rho_off) * jac_reshuffle * prod jac_bw`
-- an offshell production matrix element in the numerator with nothing under it,
while the joint weight divides by the **onshell** one
(`calculate_matrix_element_from_density`: `MEdenom_prod` is evaluated before
`reshuffle_production` and returned as `prod_diag`). The missing denominator is

    w_mass = [ Tr(rho_off) / |M_prod|^2_on ] * jac_reshuffle * prod_k jac_bw_k
                                            * prod_k Z_hat_k(m_k)

**It is not a bias.** `|M_prod|^2_on` depends on the production event alone, and
the chain never redraws the production event -- the mass stage resamples
virtualities, nothing else. A factor constant over everything the chain
resamples cancels between the weight and its bound, and since every production
event is kept and retried to acceptance it cannot reweight events against each
other either. Which is why the A/B closed without it.

**It is still wrong**, because `C_mass` is a single number shared by every
production event. Left out, the absolute scale of `|M_prod|^2` rides inside
`w_mass` and varies across the sample by orders of magnitude: the bound is set
by the loudest kinematics, the quiet ones pay for it in acceptance, and the loud
ones exceed it and are silently truncated. Both of the runs reported above did
log the overflow CRITICAL -- 10 weights (sequential) and 28 (exact) -- and the
per-slot lines carry no overflow annotation, so every one of them was at the
mass stage.

Measured, same 10000 events, before -> after normalising:

    C_mass                       17.1  ->  3.09
    mass sets / accepted event   26.2  ->  3.20     (sequential)
                                104.5  -> 12.79     (sequential_global_retry)
    weights above their bound      10  ->  1        (sequential)
                                   28  ->  3        (sequential_global_retry)
    decay phase                  28.7s -> 19.7s     (sequential)
                                 83.9s -> 31.4s     (sequential_global_retry)

with the lineshape unchanged, as the constancy argument requires: over both
resonances the sequential mean moves from 173.1641 to 173.17 and the exact one
sits at 173.1877 against joint's 173.1853. Cost: one onshell production matrix
element per production event, cached under `me_wgt` -- the same attribute, and
the same quantity, the joint path already caches there.

#### `two_stage`: one bound over all the angles

Suggested by Olivier. Keep the mass-set stage, but replace the *per-slot*
accept/reject by a single test on the product of every slot's weight, redrawing
the whole angle set on a rejection and **keeping the mass set**:

    stage 1   w_mass  = [Tr(rho_off)/|M_prod|^2_on] * jac_reshuffle
                        * prod_k jac_bw_k * prod_k Z_hat_k(m_k)
    stage 2   w_angle = prod_k [ (N_k/N_{k-1}) * jac_dec_k
                                 * Tr(D_k^off)/|M_k,dec|^2_on / Z_hat_k ]

This is the same target distribution as the per-slot scheme -- same mass stage,
same self-normalising angle stage, only the granularity of the test changes --
and the measurement says so: over four replicas each, `two_stage` gives
173.1704 +- 0.0101 and the per-slot scheme 173.1703 +- 0.0062, agreeing to
0.0001 GeV while their replica scatters are 0.010-0.012. It needs `Z_hat` for
exactly the same reason the per-slot scheme does: stage 2 redraws to acceptance
and so divides out its own normalisation.

**What it buys is reuse.** With the mass set frozen across angle retries, the
production reshuffling and the offshell production density are evaluated once
per *accepted mass set* rather than once per trial -- which the joint test
cannot do, because a rejection there redraws the virtualities too. Per event on
ttbar: 3.25 mass sets and 5.74 decay-ME evaluations, against joint's 4.46 trials
and 8.92.

The `Z_hat` division in stage 2 is not needed for correctness (stage 2 is
invariant under any rescaling by a function of the masses) but is kept for two
reasons: it flattens the virtuality dependence out of `C_angle`, and it makes
the bound estimated by the probe -- which samples masses from the prior, not
from the accepted mass distribution -- closer to what the run actually tests.

An infeasible decay kills the whole angle set rather than being redrawn in
place. Redrawing one slot would propose from the *feasible* part of the pool,
making the normalisation stage 2 divides out `Z_k/(1 - q_k(m))` instead of
`Z_k` -- a different function of the virtuality than the tabulated one, so the
mass stage's `Z_hat` would no longer compensate it. The same argument applies to
`sequential_global_retry`, and both were fixed together.

**A fifth combination was measured and is not offered.** One angle bound *and* a
mass-set restart on a rejected angle set -- i.e. `two_stage` crossed with
`sequential_global_retry` -- would make `Z_hat` cancel between the stages and be
exact whatever the table says. It costs the reuse above (9.25 mass sets per
accepted event instead of 3.25) and, over four replicas, sat 0.034 GeV below
joint: the *largest* deviation of any scheme tried, in the one that should have
been the most exact. The weight-identity check below then cleared its weight
algebra, so the deviation is a sampling or statistics question and remains
unexplained -- but the scheme is slower than `two_stage` either way, so it was
dropped rather than chased, and there is no card spelling for it.

#### The option: one knob, four schemes

The schemes are mutually exclusive alternatives rather than independent
switches, so they are selected by a single enumerated option:

    set unweighting auto | joint | sequential | sequential_global_retry
                         | sequential_with_mass

    mode                     mass stage   angle test              a rejection redraws
    joint                    --           everything at once      everything
    two_stage                yes          all angles, one bound   the angles only
    sequential               yes          per particle            that particle
    sequential_global_retry  yes          per particle            the virtualities too
    sequential_with_mass     no           per particle            that particle and its mass

`two_stage` is in the table but **not in the card's advertised values**: section
12 measured it and it is not the fastest scheme at any multiplicity, so `auto`
never picks it and it is not offered in the completion or in the "allowed values
are ..." message (`MadSpinOptions.hidden_unweighting_modes`). An explicit `set
unweighting two_stage` is still honoured -- it is the one staged scheme whose
angle stage is a single joint test, which makes it the natural cross-check
against joint, and the benchmarks and parallel tests still exercise it.
`sequential_with_mass` is section 11.

`sequential_spin_order` and `sequential_debug` keep their names: an ordering and
a check, not modes.

**`sequential_exact` was renamed, not kept.** "Exact" advertised a distinction of
~0.001 GeV on the top lineshape -- the tabulated factor is good to ~0.5%, and the
lineshape sensitivity is 0.25 GeV per unit fractional slope error -- inside a pole
approximation whose own error is of order `Gamma/m` ~ 0.9%, three orders of
magnitude larger. The name would have pushed users to a 2-3x slower scheme for a
difference nobody can measure. `sequential_global_retry` says what the mode does
and leaves the accuracy statement to the documentation, where it can carry the
numbers.

**`auto` resolves once per run**, not per event: the modes carry different
bounds, and one that changed event to event would be testing weights against
the wrong ones. What it resolves *to* is section 12 (measured over the decay
multiplicity) plus the polarised-production override. An explicit setting is
always honoured, so any scheme stays available as a cross-check.

#### How to compare these numbers (measurement notes, learned the hard way)

**Wall clocks are only comparable within one campaign.** The per-slot scheme
measured 19.74 s in one campaign and 13.33 s in another, with byte-identical
counters (same bounds, same trial counts, same seed) -- a 48% swing from machine
load alone. Several cost claims in earlier revisions of this document were built
on cross-campaign timings and were wrong. Quote the counters (mass sets, decay
evaluations, acceptances, overflows); quote wall time only from a single
campaign, with joint as an anchor in it.

**Most of the decay phase is not the accept/reject.** Counter differences of
30-40% between schemes move the clock by about 10%, so a large per-event fixed
cost -- reading the production event, `add_decays`, the final
`reshuffle_production`, writing the LHE -- dominates. A three-point fit put it
near 1.05 ms/event, but PA's total is *below* that, so the fit is
ill-conditioned and the figure too high; what survives is the qualitative
statement. Profile before optimising the accept/reject further.

**Replicas share production events.** Replicas of one scheme (same production
sample, different MadSpin seed) scatter by 0.005 (joint) to 0.020 (the
sequential schemes) on `<m_top>` over both resonances, while the naive
per-run MC error is 0.0225. The replicas are therefore strongly correlated and
neither error is right for a scheme-to-scheme difference: the naive one is too
conservative, the replica scatter probably too optimistic. This is unresolved,
and it is why the residuals below are quoted with both.

#### Speed, measured within one campaign

Decay phase for the same 10000 production events, `p p > t t~`,
`t > w+ b, w+ > l+ vl`, `nb_core 1`:

    spinmode   scheme                 decay phase   per event
    PA         joint                     9.17 s     3.14 trials -> 6.28 decay ME
    PA         sequential_with_mass     11.19 s     1.88 + 3.13 -> 5.01 decay ME
    madspin    two_stage                13.55 s     3.25 mass sets, 5.74 decay ME
    madspin    joint                    14.61 s     4.46 trials -> 8.92 decay ME

(PA's per-particle scheme was still the one that draws each slot's mass inside
its own accept/reject when this campaign was run; section 11 named it
`sequential_with_mass` and built the up-front alternative that replaced it.)

So full offshell matrix elements with `two_stage` cost about **1.5x PA-joint**,
where madspin-joint costs 1.6x, and `two_stage` is **7-9% faster than
madspin-joint** (13.06-13.55 s against 14.43-14.61 s over two campaigns).

Two observations about PA, both independent of the offshell work:

- **PA's per-particle scheme is 22% slower than PA joint on this process**,
  despite drawing fewer decay events (5.01 against 6.28). With
  `density_keep_jacobian` on, every slot trial calls
  `_production_jacobian_for` -- an `Event(str(production))` copy and a reshuffle
  -- so 5.01 production reshufflings per event against joint's 3.14. The
  per-slot decomposition is supposed to pay off as n grows; at n = 2 it does
  not. **This is what section 11 fixed**, by giving PA the up-front mass draw:
  1.95 reshufflings per event, and the decay phase level with joint.
- **It logged 11 weight overflows** (9 at slot 0, 2 at slot 1) against
  `two_stage`'s 1 and joint's 0. Its per-slot bounds are under-estimated here, so
  that sample is slightly biased. Worth a look on its own.

#### Where the offshell path stands

`two_stage` is faster than joint on n = 2 at this sample size and correct as far
as the statistics can tell. (Section 12 re-measured that at 50000 events, where
the wider `nb_sigma` margin turns the comparison around and `auto` takes joint
at n <= 2 offshell; the residual below is what had to be settled first either
way.)

**Resolved: the residual is statistical, and Z_hat is not the limiting factor.**
Earlier revisions of this section recorded that every tabulated scheme sat below
joint with the same sign in all twelve replicas, and flagged a possible `Z_hat`
inaccuracy. Tested directly by raising `max_weight_ps_point` from 500 to 2500,
i.e. 187500 probe samples per slot instead of 37500:

    Z table         Z(150.6)      Z(195.4)     bin/fit deviation
    1x probe      0.527 / 0.529  1.705/1.710      1.8% / 0.8%
    5x probe      0.524 / 0.524  1.707/1.705      0.4% / 0.6%
    analytic          0.5245        1.7036

The table is already converged at the default statistics: five times the samples
move its endpoints by under 1%, the bin-to-fit deviation falls like 1/sqrt(N)
(so it was statistical, not a wrong fit form), and the deep table reproduces the
narrow-width `(m/M) Gamma(m)/Gamma(M)` to 0.1-0.2%. Against a lineshape
sensitivity of 0.25 GeV per unit fractional error in the slope of `ln Z`, the
observed residual would have needed a ~4.4% slope error -- an order of magnitude
more than the table's ~0.5%.

And the lineshape moved the wrong way for a systematic, over four replicas each:

    two_stage, 1x probe   173.1704 +- 0.0101    -0.011 +- 0.010   (-1.1 sigma)
    two_stage, 5x probe   173.1985 +- 0.0182    +0.017 +- 0.018   (+0.9 sigma)
    joint                 173.1818 +- 0.0024

-- it flipped sign rather than shrinking, and three of the four deep-probe
replicas sit *above* joint, which retires the "same sign every time" pattern.
Pooling all eight `two_stage` replicas gives **173.1844 +- 0.0110 against joint's
173.1818, i.e. +0.003 +- 0.011 (+0.23 sigma)**. `two_stage` agrees with the joint
accept/reject.

`max_weight_ps_point = 500` is therefore sufficient for the Z table; the deeper
probe costs 169 s against 76 s per 10000-event run (the probe is fixed setup, so
it amortises on larger samples) and buys nothing.

**What this leaves open.** The dropped fifth combination's -0.034 GeV was
quoted at "-4.3 sigma" on the replica-scatter error model; that significance is
not trustworthy. The
replica scatter itself ranges from 0.005 (joint) to 0.036 (`two_stage`, deep
probe) across schemes estimated from four points each -- a ~40% uncertainty on
the error bar before any comparison is made -- and the two error models
(naive per-run MC error, and replica scatter) disagree by a factor of three.
Any future claim at the few-hundredths-of-a-GeV level needs either many more
replicas or, better, the deterministic check below.

**The weight identity holds (`sequential_debug`).** On every accepted chain, recompute the joint weight with the joint code --
on copies, for the same production event, the same virtualities and the same
decays -- and compare with the product of the stage weights.

The weights compared are the ones taken *before* any `Z_hat` division, since
`Z_hat` cancels between the mass stage and the angle stage. What is tested is
therefore the decomposition itself, not the table. And what is tested is
*proportionality*, not equality: the two differ by a constant -- the number of
helicity states, and the normalisation the density path applies to the decay
matrix elements relative to `calculate_matrix_element` -- which the bounds
absorb. A constant ratio is the identity, whatever its value; a scheme sampling
the wrong distribution has a ratio that varies chain to chain.

Measured over 2000 accepted chains each:

    two_stage            spread 1.71e-07   ratio 1108198261
    sequential per-slot  spread 1.55e-07   ratio 1108198255
    the dropped fifth    spread 1.54e-07   ratio 1108198258

The spread is float32 epsilon (1.19e-7) -- the density matrices are
`complex64`, so that is the floor of the arithmetic and not physics -- and the
three schemes agree on the constant to nine significant figures. The threshold
is `density_tolerance`, for the same reason `density_debug` uses it: two
evaluation routes through single-precision matrix elements cannot agree better
than that.

**What this settles, and what it does not.** It settles the *weight algebra* in
all three schemes: no missing jacobian, no wrong normalisation, no mis-assigned
factor. That is the class of bug a lineshape comparison detects only indirectly
and that no amount of Monte Carlo could have excluded.

It does not settle the *sampling*, which additionally requires that nothing
self-normalising is left uncompensated -- the `Z_hat ~ Z` requirement for
`two_stage` and the per-slot scheme (measured at ~0.5%, far inside the ~12%
tolerance), automatic for the restart schemes. So the dropped combination's
-0.034 GeV is not a broken weight. With the weights verified and the error models in the state
described above, the honest summary is: weights correct, deviation unexplained,
dropped because it is slower than `two_stage` anyway.

## 11. PA: the up-front mass draw, and `sequential_with_mass`

Section 10 built the up-front mass draw for the offshell spinmodes, where it is
a *necessity*: rho depends on the whole mass set, so it has to be fixed before
the per-particle loop or the decomposition does not apply. Under PA the same
split is optional -- rho is evaluated at the onshell momenta and is already
fixed per production event -- and this section is about doing it anyway,
because of what else the mass set freezes.

### The scheme PA had is a fifth scheme, not a variant

What PA did before this section is now called **`sequential_with_mass`**: one
test per decaying particle, with that particle's virtuality drawn *inside* its
own accept/reject (`_draw_offshell_mass` in the slot loop) and redrawn together
with its angles. That is a genuinely different scheme, not a flavour of
`sequential`. Nothing is ever frozen, so no stage redraws-to-acceptance under a
condition it then divides out, so there is no `Z_k` to tabulate and no
`Z_hat/Z` residual to argue about. It is also the reason `two_stage` and
`sequential_global_retry` used to be refused under PA: they split the
accept/reject at a mass draw that did not exist there.

It is still available by name, and the bit-for-bit check below is what makes
the rename safe. It was `auto`'s choice for PA and onshell when this section was
written; section 12 moved that to `sequential`, for the reasons the end of this
section gives.

### `sequential_with_mass` offshell: asked, and the answer is no

An earlier revision of this section left open whether the offshell spinmodes
should offer `sequential_with_mass` too, purely so that both families expose the
same option set. They cannot, and the obstacle is the one that made
`_upfront_production` exist in the first place: offshell, `rho_off` depends on
the whole mass set jointly, so redrawing one slot's virtuality invalidates it
*and* every slot already accepted against it. The scheme is therefore refused
there and falls back to `sequential`, with a log line saying why
(`_unweighting_mode`, `with_mass_pa_only`). The symmetry is not free and is not
worth faking.

### What PA gains by freezing the masses

Not `rho`, which is cached per production event either way
(`production._ms_density_prod`). What it gains is the **production reshuffling
jacobian**. With `density_keep_jacobian` on -- the default -- the per-slot
scheme needs `J_k`, the jacobian with the slots drawn so far offshell, at
*every slot trial*: an `Event(str(production))` copy and a `reshuffle_production`
each time, with the ratios `J_k/J_{k-1}` telescoping over the chain. Freeze the
mass set and there is one `J` for the whole set, evaluated once, with no
telescoping at all.

The weight splits the way it does offshell, minus the pieces PA does not have:

    stage 1   w_mass  = J(m_1..m_n) * prod_k jac_bw_k * prod_k Z_hat_k(m_k)
    stage 2   w_k     = (N_k/N_{k-1}) * jac_dec_k

`Tr(rho)` is absent because it cancels between `N_n` and `N_0`, which is also
why PA needs no `|M_prod|^2_on` normalisation of the mass weight (section 10)
-- there is no production matrix element in `w_mass` to normalise. The product
over the chain is `J * prod jac_bw * prod jac_dec * N_n/N_0`, which is the joint
PA weight: `reshuffle_production` on the *complete* event returns exactly
`J_RAMBO * prod_k jac_dec_k`, the two pieces the chain computes separately
through `_production_jacobian_for` and `_decay_reshuffle_jacobian`.

### Z_k under PA is *not* the running width

Freezing the masses immediately buys the problem section 10 spent itself on: the
angle stage redraws until it accepts, so it divides out

    Z_k^PA(m) = E_pool[ w_k ] = E_pool[ jac_dec(m, Omega) ]

-- the density ratio `N_k/N_{k-1}` averages to one at fixed m, so what is left
is the decay reshuffling jacobian alone. This is **not** the offshell
integrand. Offshell, `w_k` also carries `Tr(D^off)/|M_dec|^2_on` and `Z_k`
comes out as the running partial width `(m/M) Gamma(m)/Gamma(M)`; PA evaluates
every matrix element on shell, so there is no offshell reweighting to
normalise and only the phase-space cost of mapping a pool decay onto the
sampled virtuality survives.

That makes it a much gentler function, and the measured table says so. On
`p p > t t~`, `t > w+ b, w+ > l+ vl`, from the 37500 probe samples per slot the
scan collects for free:

    slot    Z(150.7)   Z(173)   Z(195.3)   bin/fit deviation
    6_0       0.913      1        1.059         0.2%
    -6_0      0.912      1        1.059         0.1%

against the offshell table's 0.53 / 1 / 1.71 over the same window. A factor 1.16
across the Breit-Wigner window where offshell has a factor 3.2. The machinery is
the same -- `_z_slot_keys`, `_build_z_tables`, `_zhat`, samples recorded by
`sequential_accept_reject` in probe mode -- only the quantity averaged differs
(`rate = jac_dec` instead of `jac_dec * Tr(D^off)/|M_dec|^2_on`).

**It is still needed.** The argument of section 10 does not care how big the
factor is: the angle stage divides out the *true* `Z_k` whatever weight it is
given, so omitting the compensation leaves the accepted virtualities
Breit-Wigner distributed rather than physically distributed, and the residual
is exactly `Z_hat/Z`. What the small factor does change is the *sensitivity of
the lineshape test*: see below.

`density_keep_jacobian = False` is the degenerate case. There the reshuffle is a
post-acceptance dressing and `jac_dec` is in no weight, so `w_k` is the density
ratio alone and `Z_k(m)` collapses to the fraction of the pool that can reach
`m` -- still a function of the virtuality, still tabulated by the same code
(a decay that cannot be reshuffled onto `m` records a zero, exactly as
offshell), and identically one wherever the whole pool is reachable.

### Validation

**Bit-for-bit, first.** `sequential_with_mass` on 10000 `p p > t t~` events is
byte-identical to what the branch produced before this section: same seed, same
production sample, every event record in the decayed LHE the same, and the same
counters (5.01 decay events per accepted event, 1.88 + 3.13 per slot, 11
weights above their bound, 17 chains restarted). The only difference anywhere in
the file is the path of the input LHE echoed in the banner, which is the test
harness giving each mode its own directory.

**The weight identity, per chain.** `sequential_debug` now covers the PA
up-front schemes: it rebuilds the joint PA weight for the same production event,
the same virtualities and the same decays -- `calculate_matrix_element_from_density`
for the density part (PA leaves the momenta onshell there, so it returns
`jac_reshuffle = 1`), the Breit-Wigner jacobians recomputed from the sampling
window, and the reshuffling jacobian from a `reshuffle_production` of the
*complete* rebuilt event, which is the route the joint path takes and is
therefore an independent check of the chain's two separate pieces. Over 10000
accepted chains each:

    sequential                spread 7.91e-08   ratio 1108198227
    two_stage                 spread 1.23e-07   ratio 1108198225
    sequential_global_retry   spread 8.54e-08   ratio 1108198230

Float32 epsilon is 1.19e-7 and the density matrices are `complex64`, so that is
the floor of the arithmetic. Worth noting that the constant is the *same* one
the offshell schemes report (1108198255-261, section 10) to nine significant
figures, from a different spinmode and a different set of factors -- the
number is the helicity/normalisation constant of the density path, as claimed.

**The lineshape, and how to make the test sensitive.** `m(l+ vl b)` is the
observable the missing factor distorts, and under PA it *is* the sampled
virtuality (the accepted decay is reshuffled onto it), so this is a direct look
at the accepted mass distribution. Four replicas of each scheme over the same
10000 production events with independent MadSpin seeds (42-45), against a
four-replica joint reference:

    scheme                   <m(top)>, both resonances   vs joint        chi2/ndf
    joint                    172.9469 +- 0.0135           --              --
    sequential_with_mass     172.9480 +- 0.0104          +0.0011 (+0.06)  12.6/24
    sequential               172.9558 +- 0.0068          +0.0089 (+0.59)   9.4/24
    two_stage                172.9589 +- 0.0067          +0.0119 (+0.79)  11.3/24
    sequential_global_retry  172.9632 +- 0.0057          +0.0162 (+1.11)  20.3/24
    sequential, Z_hat = 1    172.9083 +- 0.0087          -0.0386 (-2.40)  12.3/24

(errors are the replica scatter; the naive per-run MC error on the pooled sample
is 0.0111 and gives the same significances to within 0.08 sigma.)

The three up-front schemes sit 0.6 to 1.1 sigma above joint, all on the same
side. That is not the table: `sequential_global_retry` needs no table at all --
`Z_hat` cancels identically there -- and it is the *highest* of the three. What
it is is the joint reference, whose own replica scatter (0.0135) is the largest
of the six and whose seed-42 replica (172.9161) is a visible low outlier. As in
section 10, the scatter between two runs of the same scheme is as large as
anything the scheme-to-scheme differences show.

The last row is the point. PA's `Z_k` spans a factor 1.16 where the offshell
one spans 3.2, so the bias it protects against is ~0.04 GeV rather than the
0.25 GeV of section 10 -- at the edge of what a 10000-event A/B can see, which
would have made a plain "sequential agrees with joint" statement
uninformative. Measuring it directly instead, with a scratch build whose
`_zhat` returns 1, puts the factor at **-0.039 +- 0.016 GeV** (-0.047 +- 0.011
against `sequential` itself, -4.3 sigma) and in the direction section 10
predicts: down, towards the Breit-Wigner prior. Restoring it recovers joint to
+0.009 +- 0.015.

The two no-regression observables behave as section 10 says they do -- blind to
this class of bug. Even with `Z_hat` forced to 1, `m(l+ vl)` moves by -0.17
sigma and `dphi(l+,l-)` by -0.21 sigma, while the resonance mass is off by 2.4.
Anyone checking a new unweighting scheme on those two alone would have passed
this build.

**What the up-front branch must not have disturbed.** The offshell path now
shares `_upfront_production` and the merged slot body with PA, so it was
re-run against the base commit: `spinmode = madspin` on the same 10000 events
gives identical event records, the same Z table to every digit
(0.527 / 1 / 1.705 and 0.529 / 1 / 1.710), the same bounds (3.09, 2.881) and the
same 57448 trials. `spinmode = onshell` with an explicit `sequential` -- no
virtuality anywhere, so the mass stage is the degenerate one -- runs and gives
3.98 decay events per accepted event with no overflow. `nb_core = 4` reproduces
the cross section and the counters up to the workers' own RNG streams (2.09
mass sets per event against 1.95 serial), i.e. the tables survive the fork.

(The offshell re-run earned its place: the first version of this branch passed
PA's `draw_mass` flag straight into `_upfront_production`, which under an
offshell spinmode is False, and skipped the mass draw entirely. Offshell always
samples -- its rho is only defined at the reshuffled momenta -- and the PA flag
only ever described the PA draw.)

**One thing the merge with the me_frame work had to fix.** `frame_id` picks the
frame the helicity basis is defined in, and the production density and every
decay density contracted against it must be taken in the same one. That boost
was computed only on a *cache miss* of `production._ms_density_prod` -- so the
first chain of a production event got it and every later chain got `None`, while
the cached rho still carried it. The max-weight probe draws 500 chains per
production event, so 499 of them would have contracted lab-frame decay densities
against an me_frame production one. It is latent on a default card
(`_frame_boost` returns None for unpolarised beams) and silent when it is not,
which is the worst combination. The boost is now cached alongside the density it
belongs to, as `production._ms_frame_boost`.

### Speed

Decay phase and counters for the same 10000 production events,
`p p > t t~`, `t > w+ b, w+ > l+ vl`, `nb_core 1`, two campaigns with joint as
the anchor in each. The counters are byte-identical between the two (same
seed), so the pair of clocks is a read on the machine, not on the schemes:

    scheme                  decay phase   decay MEs/event   mass sets   prod reshuffles
    joint                   7.6 / 7.6 s   6.28 (3.14 x 2)      --           3.14
    sequential              7.4 / 7.3 s   4.06 (1.21 + 2.85)  1.95          1.95
    sequential_with_mass    9.4 / 9.3 s   5.01 (1.88 + 3.13)   --           5.01
    two_stage              10.4 / 8.9 s   5.73 (2.87 + 2.87)  1.93          1.93
    sequential_global_retry 12.2 / 12.1 s 6.22 (3.38 + 2.84)  6.61          6.61

(`two_stage`'s 10.4 s is the one entry the second campaign does not reproduce;
that run also took 38% longer to generate its matrix elements, so it was load.
Section 10's warning about cross-campaign clocks applies inside a campaign too
when the difference being read is 10%.)

The observation section 10 closed on -- "PA sequential is 22% slower than PA
joint on this process ... 5.01 production reshufflings per event against joint's
3.14" -- is what the up-front draw was built for, and it is fixed: **1.95
reshufflings per event**, a factor 2.6, and the decay phase goes from 22% above
joint to level with it. The decay-ME count drops too (4.06 against 5.01),
because the per-slot bounds no longer have to cover the production jacobian's
spread: slot 0's bound falls from 1.836 to 1.21 and its acceptance rises from
1/1.88 to 1/1.21.

`two_stage` loses here, unlike offshell. Its single angle bound (2.865) is
barely tighter than the product of the per-slot ones (1.21 x 2.863 = 3.46),
because under PA slot 0's weight is nearly flat, so it pays the lost early exit
for almost nothing. `sequential_global_retry` costs 3.4x the mass sets, as it
does offshell, and is a cross-check rather than a candidate.

The remaining overflow counts are 11 (`sequential_with_mass`, unchanged), 6
(`sequential`), 9 (`two_stage`) and 11 (`sequential_global_retry`) out of 10000
events -- the "PA sequential logged 11 weight overflows" observation of section
10 is improved but not removed, and remains worth a look on its own.

**At 250000 events the ordering changes, and the gap widens.** The run above
is dominated by costs the scheme does not touch, so it was repeated on a single
250000-event sample where the decay phase is 51-66% of the wall clock:

    scheme                  decay phase  total wall  decay MEs/ev  prod reshuffles  mass sets  over bound
    sequential                226.0 s      446.5 s       4.56           2.83          2.83         40
    two_stage                 285.2 s      507.3 s       6.57           2.83          2.83         42
    joint                     324.3 s      544.6 s       9.06           4.53           --           0
    sequential_with_mass      403.7 s      629.8 s       8.07           8.07           --          37
    sequential_global_retry   436.6 s      660.2 s       7.22          11.15         11.15        180

Against joint: `sequential` is 30% faster in the decay phase and 18% on the
whole run; against `sequential_with_mass`, which is what `auto` picks, it is
**44% and 29%**. All five agree on the cross section to 0.003% (23.75487 joint,
23.755604 for the other four).

Two things move between the two sample sizes, and both favour the up-front
split as N grows.

- The fixed costs stop hiding it. At 10000 events the decay-pool generation
  (~35 s) and the probe dwarf a 2 s difference; at 250000 the pool costs ~165 s
  against a 226-436 s decay phase.
- `nb_sigma` is `max(4.5, log_7.7 N)`, so it goes from 4.51 to 6.09 and every
  bound widens. That costs the schemes unequally: `sequential_with_mass`'s
  per-slot weights carry the Breit-Wigner jacobian and `J_k/J_{k-1}`, so they
  are the broadest and lose the most (slot 1's bound 5.201, acceptance 1/5.31,
  against the up-front `sequential`'s 3.287 and 1/3.29). Moving those factors
  into a mass stage that is bounded once is worth more the wider the margin
  gets. `two_stage` overtakes `sequential_with_mass` for the same reason.

So the 10000-event ordering (sequential < joint < with_mass < two_stage <
global_retry) is not the asymptotic one; at 250000 it is sequential < two_stage
< joint < with_mass < global_retry, and the advantage of the up-front draw over
the scheme PA shipped with is a factor 1.8 on the accept/reject.

**`auto` took `sequential_with_mass` under PA when this section was written,
and no longer does** -- section 12 scanned the decay multiplicity and switched
it to `sequential`. At 10000 events the gain was inside the noise; at 250000 the
up-front draw takes 29% off the whole run, and the margin grows with N. What it
costs is exactness by construction: `sequential_with_mass` freezes nothing and
needs no table, where `sequential` carries a tabulated factor accurate to ~0.2%
on something worth 0.04 GeV, i.e. ~0.0001 GeV of residual -- three orders of
magnitude inside the pole approximation's own error. `sequential_with_mass`
remains available, and is the scheme to reach for if that residual ever needs
to be excluded rather than bounded.

## 12. Which scheme should be the default: a multiplicity scan

Sections 10 and 11 each measured one process with two decaying particles, which
is exactly the multiplicity at which the schemes are closest. This scans the
number of decaying particles instead, 50000 events per point, `nb_core 1`, seed
42, one campaign per process (so the clocks are comparable within a block, not
across blocks):

    n=1   p p > w+ j        w+ > l+ vl
    n=2   p p > t t~        t > w+ b, w+ > l+ vl                (both tops)
    n=3   p p > t t~ z      as above, plus z > l+ l-
    n=4   p p > t t~ t t~   as above, all four tops

All 24 runs agree on the cross section: identical within a process except for
the 5e-5 relative offset between the joint runs and the rest, which is the
Breit-Wigner sampling and not the scheme.

### PA

    n  scheme                decay phase   total wall   decay MEs/ev   mass sets/ev
    1  joint                    50.3 s       92.9 s         6.12            --
    1  sequential_with_mass     58.7 s       99.5 s         6.39            --
    1  sequential               40.4 s       81.8 s         5.09           2.42
    2  joint                    83.4 s      178.2 s        10.96            --
    2  sequential_with_mass    127.0 s      231.6 s        12.27            --
    2  sequential               39.8 s      139.0 s         4.39           3.23
    3  joint                   186.6 s      292.4 s        24.30            --
    3  sequential_with_mass     88.7 s      205.4 s         9.78            --
    3  sequential               66.0 s      177.0 s         7.73           4.48
    4  joint                   324.3 s      472.4 s        42.56            --
    4  sequential_with_mass    110.5 s      298.2 s         9.47            --
    4  sequential               85.5 s      263.9 s         8.79           2.42

**`sequential` wins at every multiplicity**, by 20% at n=1 and by a factor 3.8
at n=4, and it is never worse than the other two on any counter. The joint
test's cost grows as n x (trials per event) because a single rejection throws
away every decay; the per-particle test's grows far more slowly.

`sequential_with_mass` -- today's PA default -- is the *worst* of the three at
n=1 and n=2 on this campaign, and slower than `sequential` everywhere. Note how
much worse it looks at 50000 events than at the 10000 of section 11 (n=2: 12.27
decay MEs per event against 5.01): `nb_sigma` is `max(4.5, log_7.7 N)`, and its
per-slot weights carry the Breit-Wigner jacobian and the `J_k/J_{k-1}` ratios,
so they are the broadest and lose the most as the safety margin widens. That is
the same effect section 11 saw between 10000 and 250000, and it says the
measured gap is a lower bound on what a production-sized run would see.

### madspin (full offshell)

    n  scheme          decay phase   total wall   decay MEs/ev   mass sets/ev
    1  joint              65.5 s      112.5 s         6.57            --
    1  sequential       2534.6 s     2576.7 s         8.11         786.61
    1  two_stage        2495.1 s     2539.8 s         4.96         787.32
    2  joint              54.3 s      151.1 s         8.08            --
    2  sequential         70.1 s      169.6 s         6.73           3.51
    2  two_stage          59.5 s      162.7 s         6.24           3.50
    3  joint             231.4 s      338.5 s        25.32            --
    3  sequential        107.0 s      223.0 s        11.79           3.59
    3  two_stage         244.7 s      359.4 s        24.94           3.59
    4  joint             722.4 s      882.7 s        50.60            --
    4  sequential        167.3 s      365.7 s        13.21           3.23
    4  two_stage         315.8 s      514.3 s        28.70           3.22

Three separate findings.

**n=1 offshell is a disaster, and it is the mass stage.** 787 mass sets per
accepted event, `C_mass` = 781.6, a decay phase 38x the joint one. `two_stage`
gives the same 787, which localises it precisely: not the angle granularity,
not `Z_k` (the table is clean, bin/fit deviation 0.0%), but the mass-set weight
`Tr(rho_off)/|M_prod|^2_on`. On `p p > w+ j` the decaying particle carries
essentially all of the production matrix element's virtuality dependence, so
that ratio spans orders of magnitude over the 15-width window and no single
bound can cover it. The PA run of the same process has `C_mass` = 2.37, which
confirms the diagnosis -- PA evaluates rho on shell, so its mass weight has no
production matrix element in it at all. `auto` already routes n=1 to joint, so
nothing is broken; this is why that rule has to stay.

**n=2 offshell still belongs to joint.** 54.3 s against 59.5 (`two_stage`) and
70.1 (`sequential`), even though both draw *fewer* decay matrix elements (6.24
and 6.73 against 8.08): each mass set costs a production reshuffle and an
offshell production density, and at 3.5 mass sets per event that outweighs the
decays saved. Section 10 measured the opposite at 10000 events (`two_stage`
13.55 s against joint's 14.61 s); the difference is again `nb_sigma`, 4.51 there
against 5.60 here, which widens `C_mass` and costs the staged schemes.

**From n=3 `sequential` wins outright**, 2.2x at n=3 and 4.3x at n=4 on the
decay phase, and `two_stage` is not competitive there: its single bound over the
product forces every slot to be redrawn together, 8.31 angle sets per event at
n=3 and 7.17 at n=4, so it draws as many decay matrix elements as the joint test
while also paying the mass stage.

### What this says about `auto`

The current rule is: 1 -> joint; PA/onshell -> `sequential_with_mass`; 2 ->
`two_stage`; 3+ -> `sequential`. The scan says two of those four branches are
wrong and one is unnecessary:

    spinmode         n      current               measured best
    PA / onshell     any    sequential_with_mass  sequential  (1.2x - 3.8x)
    madspin / full   1      joint                 joint       (correct)
    madspin / full   2      two_stage             joint       (1.1x)
    madspin / full   3+     sequential            sequential  (correct)

`two_stage` is not the fastest scheme at any point measured here, in either
spinmode: joint beats it at n<=2 and `sequential` beats it at n>=3. It remains
worth keeping as an option -- it is the one staged scheme whose angle stage is a
single joint test, which makes it the natural cross-check against joint -- but
it does not earn a branch in `auto`.

**`auto` now implements the two-line rule** (`_unweighting_mode`):

    PA / onshell     ->  sequential
    madspin / full   ->  joint for n <= 2, sequential from n = 3

with the caveat that every number above is one process per multiplicity on one
machine, and that the n=2 offshell call is a 10% difference that went the other
way at a smaller sample size -- so that boundary is the one to revisit if a
process is found where a staged scheme pays off at two decays. The n=1 offshell
and the n>=3 conclusions are not close and are safe.

What changes for a user who never set `unweighting`: PA and onshell runs move
from `sequential_with_mass` to `sequential` (faster everywhere measured, at the
price of the tabulated factor -- section 11 bounds its effect on the top
lineshape at ~0.0001 GeV); offshell runs with two decaying particles move from
`two_stage` to `joint`, i.e. back to the historical scheme. Nothing changes for
offshell runs with one or with three or more decaying particles.

---

## 13. Pure-interference mode -- feasibility assessment

**Status: implemented and validated end to end** (section 13.12). This section
was written as a feasibility assessment before the mode existed; it is kept as
the derivation, because every design decision below is still the one in the
code. What changed since it was written is that 13.9's "not implemented" list
is now empty -- see 13.9 for the final state of the tree.

**Verdict as assessed: feasible with caveats, and the caveats are not small.**
The tensor algebra is clean. The *mode* -- syntax, signed unweighting, zero
cross-section bookkeeping -- is a structural change to the accept/reject loop,
is incompatible with the sequential scheme, and produces an LHE file whose
`<init>` cross-section is zero, which several downstream tools cannot consume.
All of that held up; the accept/reject rework (13.7b) was indeed the hard part.

The request, verbatim:

> check the possibility to handle pure interference term: that mode should be
> similar to [the production-polarisation restriction] in term of syntax, but
> should allow to specify production/decay polarization (and non overlapping
> ones). In that case the convolution should be for one index done like in the
> restriction of the production and for the other like the convolution specified
> by the decay. Full cross-section of the sample should be set to zero, but
> weight of the events should keep the same absolute value as now but one need
> to assign the sign of the weight according to the sign of the convolution. A
> check should assert if the sum of the weight are compatible with a zero
> cross-section.

### 13.1 Correcting two premises

**(a) The matrices are not stored packed upper-triangular in memory.** The
*Fortran* `INTER` buffer is (length `n(n+1)/2`), but `get_map_density_matrix`
builds keys for the conjugate labels too, and `get_map_template`'s `conj_mask`
conjugates them, so `DensityMatrix.values` holds all `n^2` entries of the full
hermitian matrix, each labelled by its `(bra, ket)` pair. A row mask can
therefore select any subset of `(i,j)` pairs, including asymmetric ones -- there
is no "lower triangle is implied" obstacle. (`diag_elements` / the packed
indexing survive only in `identity`, which builds a packed array to hand to the
normal constructor.)

**(b) The existing restriction is symmetric, and that is not an accident.** The
reading that pure interference needs an asymmetric rule is right -- but an
asymmetric rule *alone* is not well-defined. The correction is in 13.3.

### 13.2 What the convolution is, and why the full sum is real

`scalar_multiplication` computes, over the joint helicity index,

    W = sum_i sum_j  rho_prod(i,j) rho_dec(i,j)

Both matrices are hermitian: `rho(j,i) = conj(rho(i,j))`. Hence the term at
`(j,i)` is the complex conjugate of the term at `(i,j)`:

    rho_prod(j,i) rho_dec(j,i) = conj(rho_prod(i,j)) conj(rho_dec(i,j))
                               = conj( rho_prod(i,j) rho_dec(i,j) )

So **any** index set closed under `(i,j) -> (j,i)` sums to a real number, and
any set that is not closed does not. The full sum is closed (trivially). The
existing symmetric restriction -- entry survives iff both `i` and `j` are
allowed -- is closed, which is why `me.real` in
`calculate_matrix_element_from_density` (interface_madspin.py:6469) has always
been a no-op safety rather than a projection.

### 13.3 The crux: `P x D` alone is complex; the mode needs `P x D` u `D x P`

Take a decaying particle with production-side set `P` and decay-side set `D`,
with `P` and `D` disjoint. The literal reading of the request -- "one index like
the production restriction, the other like the decay" -- is the set `P x D`,
i.e. `bra in P and ket in D`. Its transpose is `D x P`, which is *disjoint* from
it because `P` and `D` are. So `P x D` is **not** closed under transposition and

    sum_{(i,j) in PxD} rho_prod(i,j) rho_dec(i,j)   is in general complex.

Numerically, for a random hermitian pair on the vector basis `[-1,0,1]` with
`P={0}`, `D={-1,+1}`:

    sum over P x D    = 0.7224 + 0.6042i
    sum over D x P    = 0.7224 - 0.6042i      (= its conjugate, as required)
    sum over the union = 1.4448 + 0i          (= 2 Re[P x D])

There is therefore **no "sign of the convolution"** for `P x D` on its own: a
complex number has no sign. Taking `Re[...]` by hand and calling that the answer
is not an arbitrary convention either -- it is *exactly* half the union, so the
two prescriptions agree up to a factor 2. The physically meaningful object is
the union, and it is real **by construction** rather than by projection:

    W_int = sum_{PxD} + sum_{DxP} = 2 Re sum_{PxD}

This is also the only choice that makes the decomposition close. With `P u D`
the whole basis,

    W_full = W_PP + W_DD + W_int

verified numerically to float32 precision (test
`test_the_three_blocks_add_up_to_the_full_convolution`). Under the `Re`-of-half
convention the three pieces would miss `W_full` by `W_int/2`.

**Conclusion: the mode is well-defined, and it needs no explicit real part --
provided the restriction keeps both index orderings.** The asymmetric reading is
correct in substance (bra from the production set, ket from the decay set, hence
`i != j`) and needs one amendment: the hermitian partner must be kept, which is
what turns "asymmetric" into "off-diagonal block", and is precisely the
difference between summing `i<j` and summing `i!=j`.

### 13.4 Zero cross-section, and why the sequential scheme dies

Integrating a decay density matrix over the full solid angle gives `delta_ij/n`
(`DensityMatrix.identity`, and the rotational-invariance argument in section 1).
`P x D u D x P` contains **no diagonal entry** when `P` and `D` are disjoint, so

    <rho_prod, I/n>  restricted to the interference block  =  0    exactly

(test `test_cross_contraction_against_the_identity_vanishes`). Two consequences,
one wanted and one fatal to a feature we just built:

* **Wanted:** the interference term integrates to zero over the decay phase
  space. That *is* the "full cross-section of the sample should be set to zero"
  of the request -- it is a theorem, not a convention, and it is what the
  statistical check of 13.8 is testing.
* **Fatal:** the sequential accept/reject (sections 1-6) substitutes `I/n` for
  every decay slot not yet drawn. Every partial weight in this mode is therefore
  *identically zero*, for every prefix, and there is nothing to unweight
  against. `_partial_density_contraction` and `N_k` collapse. **The
  pure-interference mode must force `unweighting = joint`** and refuse the
  sequential / two-stage schemes with a clear error rather than silently
  producing zero weights and hanging in the redraw loop.

The same zero shows up in `trace()`: the restricted trace of an interference
block is exactly zero (test `test_cross_restricted_trace_vanishes`). Since the
weight is `full_me / (production_me * decay_me)` and `production_me` comes from
`prod_diag = density_prod.trace().real` (interface_madspin.py:6460), **the
restriction must not be applied to the normalising trace**. This is the one
place where PR #349's "the restriction rides on the matrix so every call site
picks it up" design has to be broken: the contraction restriction and the
normalisation restriction stop being the same object. Concretely, a second
attribute (`hel_restriction_trace`, defaulting to the contraction one so nothing
symmetric moves) read by `trace()` and by `normalized()`. Its value in
interference mode is the *symmetric* restriction to `P u D` -- i.e. whatever the
production process' own braces impose, and `None` for an unpolarised production
process.

### 13.5 Which sample the mode may be run on

The interference between `P` and `D` amplitudes only exists if both are present
in the sample the events were drawn from. `p p > t{L} t~` events are distributed
as `|M_L|^2`; reweighting them by an `L`-`T` interference term is meaningless.
So:

* the production process **must not** carry a brace that excludes either side
  (`P u D` must be contained in, and should equal, the production polarisation),
  and normally is fully unpolarised;
* consequently the production-side set `P` cannot be read from the banner's
  `proc_card` the way `_production_polarization` does -- **both** sets have to
  come from the MadSpin card. This is the reason the request says the mode
  "should allow to specify production/decay polarization": there is no
  production brace to inherit.

`_production_polarization()` is still needed, but only as a *validation* input:
if the banner does carry a brace on that pdg, assert `P u D` equals it, and use
it as the symmetric trace restriction of 13.4.

### 13.6 Syntax

The decay-side brace is refused today (`do_decay`, interface_madspin.py:764-776)
with an `InvalidCmd` that explains that a brace on a `decay` line would project
the decay matrix element that defines the branching ratio, which is not what is
wanted. That reasoning is still right, and it is *not* what this mode does: here
the decay-side set restricts one index of the **convolution**, and the decay
matrix element (hence the BR) stays fully inclusive. So a carve-out would be
principled, not a loophole -- but the mode needs its own spelling anyway,
because reusing `decay t{T} > ...` would mean the same characters requesting two
different things depending on a mode flag.

Two candidate spellings:

1. **Two braces on the decay line**, `decay t{0}{T} > w+ b`: closest to
   "similar in syntax", but `{0}{T}` is not grammar MG5's `extract_process`
   accepts, so it would need a change in `madgraph_interface`'s process parser
   -- shared code, wide blast radius, and MadSpin is not its only consumer.
   Rejected.
2. **A dedicated MadSpin-card option** (recommended):

       set pure_interference t = 0 T          # or: 6 = 0 T

   a dict-valued parameter `pdg -> (production_pol, decay_pol)`, parsed with the
   same `{0}/{+}/{R}/{-}/{L}/{T}` vocabulary `_apply_production_polarization`
   already validates against the `hel_dict` basis. Validation at parse time:
   both sides expressible in the basis; the two sides **disjoint** (an overlap
   re-admits diagonal entries, so the trace stops vanishing and the mode stops
   being "pure interference" -- refuse rather than warn); spinmode is a density
   one; `unweighting` is not a sequential scheme. `do_decay`'s existing
   `InvalidCmd` is then left exactly as it is -- no carve-out needed at all,
   which also keeps the diff away from a file other agents are editing.

Setting the option is what switches the mode on; no separate boolean.

### 13.7 The hard part: signed unweighting

This is where the feature stops being cheap.

**(a) The accept/reject test rejects everything.** interface_madspin.py:3449 is

    if random.random()*maxwgt < wgt*jac:

With `wgt` free to be negative this never fires and `while 1:` spins forever.
It has to become `< abs(wgt*jac)`, with `sign = math.copysign(1.0, wgt*jac)`
carried to the output weight. Likewise `_joint_maxwgt_range`
(interface_madspin.py:4302) accumulates `maxwgt = max(wgt*jac, maxwgt)` from a
`0` seed, so it currently bounds only the positive excursions: it must bound
`abs(wgt*jac)`. `_combine_maxwgt`'s mean/sigma statistics are then applied to
`|w|` and need no further change.

**(b) Redraw-until-accept is statistically wrong here, and this is the real
blocker.** The current loop draws decay configurations until one is accepted, so
**every** production event yields exactly one output event of weight
`w_p * branching_ratio`. That is correct only because

    <wgt>_decay-phase-space = 1 / prod_i n_i

is the *same constant for every production event* -- so forcing one output per
input does not distort the production-side distribution. In interference mode
that mean is `0` for every event (13.4), and the quantity that now varies from
event to event is `Int_p = <|wgt|>`, which is exactly what measures how much
interference that production point carries. Redraw-until-accept normalises
`Int_p` away: every production event would contribute `+-w_p` with the same
magnitude, and the interference would be represented by its sign pattern alone,
with the production-side shape wrong.

The fix is to stop redrawing: **draw one decay configuration, accept with
probability `|wgt|/maxwgt`, and on rejection write nothing and move on.** Then
the number of kept events per production point is proportional to `Int_p`, each
carries `+- w_p * BR`, and for any observable `O` the sum over kept events of
`w_p sign(W) O` estimates the integral of `W(Omega) O(Omega)` -- the
interference distribution, correctly normalised relative to the parent sample.
The expected weight sum is the integral of `W`, i.e. zero, as required.

This is a different control flow from `while 1:` -- but not an unprecedented
one: the BR-equalization path in `_unweight_range` (interface_madspin.py:3369)
already does `nb_loose_skip += 1; continue` without writing, and
`_apply_accounting` already handles `n_written < n_processed`, rewrites the
banner cross-section by `n_written/n_processed`, and reports the kept fraction
as `self.efficiency` for the downstream `nb_event` bookkeeping. So the machinery
to write fewer events than were read exists and is exercised; what is new is
that in this mode it is the *normal* path rather than a correction, and that the
banner rewrite must not be applied (13.7c). It also interacts with `fixed_order`
(the counter-event group would have to be dropped as a unit) and with the
`nb_core` sharding (each shard reports its own counts, which already merge
additively).

**(c) The `<init>` cross-section.** `run_onshell` writes the banner via
`self.banner.scale_init_cross(self.branching_ratio)`
(interface_madspin.py:2537), and `scale_init_cross` (banner.py:220) rescales
`XSECUP`, `XERRUP` and `XMAXUP` per subprocess. "Set the full cross-section to
zero" means writing `XSECUP = 0` for every subprocess line -- reachable through
`modify_init_cross({pid: 0.0}, allow_zero=True)`, which sets `ratio = 0` and so
zeroes `XERRUP` and `XMAXUP` as well.

That is what was asked, and it is also a loaded gun. `get_cross` sums the
`XSECUP` column, so the banner then reads `sigma = 0`; any downstream consumer
that normalises "events -> picobarns" by `XSECUP / N` divides by zero, and
`XMAXUP = 0` is outside the LHE spec's intent for the `IDWTUP` schemes that use
it. Pythia8 in particular takes the process cross-section from the `<init>`
block. The honest engineering answer:

* write `XSECUP = 0` (the physics is that this sample has no rate), **and**
* record the *measured* weight sum and its MC error, plus the parent sample's
  `sigma * BR` as the reference normalisation, in a `<MGGenerationInfo>`-style
  banner note and in the log, so a user can renormalise by hand;
* log a loud warning that the output is a signed differential sample and is not
  directly showerable without an externally supplied normalisation.

An option (`set interference_init_cross measured|zero|reference`) is cheap
insurance if a user needs a showerable file; default `zero` per the request.

### 13.8 The statistical check

After the loop, with kept weights `w_i` (each `+- w_p * BR`):

    S     = sum_i w_i
    delta = sqrt( sum_i w_i^2 )        # MC error on S; there is no cancellation
                                       # in the second moment, so this is the
                                       # right scale to compare S against
    z     = S / delta

Report `z`, and fail the check when `|z| > nb_sigma` (the card already has
`nb_sigma`, default 3; 5 is the more usual threshold for an automatic assert and
is the value I would pick, so that a legitimate 3-sigma fluctuation in a large
run does not cry wolf).

*Where:* accumulate `sum_w` and `sum_w2` into the stats dict `_unweight_range`
already returns -- it is picklable and merged additively over the forked shards
in `_apply_accounting`, so one shard or many gives an identical answer. Emit the
report from `_apply_accounting`, next to the existing unweighting-efficiency
line.

*On failure:* `logger.critical`, not an exception. A non-zero `z` has three
possible causes -- a genuine fluctuation, an under-estimated `max_weight` (the
overweight events bias `S`, and this test would be the most sensitive monitor of
that we have), or a bug -- and none of them is worth discarding a completed run
over after the CPU has been spent. The message should print `S`, `delta`, `z`
and the overweight count so the three are distinguishable. `density_debug` can
promote it to a `RuntimeError` for the test suite.

Caveat to document: the test assumes the `w_i` are independent, which is true
event-to-event here but *not* across a production sample that itself came from a
correlated MC (multi-weight / reweighted samples). It is a sanity check, not a
proof of correctness.

### 13.9 What is implemented

**The algebra** (`MadSpin/decay.py`, plus 11 tests in
`tests/unit_tests/madspin/test_madspin.py::TestPureInterferenceRestriction`):
the cross restriction at the `DensityMatrix` level. A per-particle entry of
`hel_restriction` may now be a `(P, D)` pair instead of a flat set of allowed
helicities, and `_restriction_row_mask` builds

    (bra in P and ket in D) or (bra in D and ket in P)

for that particle. Crucially this **keeps the per-particle AND structure** of
the mask: the union is taken inside one particle's factor, not across particles,
so `_restriction_row_mask` stays a product of per-index conditions, the
`(basis_id, restriction)` cache key still works, and `tensor_product` still
concatenates entries. Each factor is separately closed under
`bra_k <-> ket_k`, so the product is closed under the global transposition and
the contraction is real for any mixture of `None`, symmetric and cross entries
(test `test_multi_particle_cross_stays_a_per_index_product`).

That per-particle union is a deliberate choice over the "global" alternative
`(all k in P) x (all k in D)` union its transpose. The two coincide whenever
only one particle carries a `(P, D)` pair -- the dominant, and arguably the only
physically motivated, use case -- and they differ only in whether mixed terms
(particle 1 taken production-side, particle 2 decay-side) are kept. The
per-particle form keeps them, which is what makes the polarised decomposition of
`W_full` close particle by particle; the global form does not, and would need a
union-of-two-product-masks in `_restriction_row_mask`, breaking its structure.
If the global variant is ever wanted it is a separate normalised form, not a
tweak.

Normalisation rules: `(S, S)` collapses to the symmetric `S`; an empty side
falls back to `None`; a non-2-element pair raises. Nothing symmetric moves --
same normalised values, same cached mask objects, same numbers
(`test_symmetric_restrictions_are_untouched`).

**The mode** (`MadSpin/interface_madspin.py`, plus `TestPureInterferenceMode`
and the frame test `test_frame_follows_the_pure_interference_mode`):

* `hel_restriction_trace` on `DensityMatrix`, consulted by `trace()` /
  `normalized()` **only** when `hel_restriction` is a cross one --
  `_trace_restriction` returns a symmetric restriction untouched, so no
  pre-existing path can move. Its value is the symmetric restriction the
  production braces impose, i.e. `None` (the full trace) for the unpolarised
  production the mode requires.
* the `pure_interference` card option, parsed by `_pure_interference` into
  `pdg -> (P, D)` and validated by `_validate_pure_interference`: density
  spinmode, disjoint sides, a pdg something actually decays, and a production
  process that is not braced away from either side.
* `_apply_pure_interference` overlays the cross restriction on whatever
  `_apply_production_polarization` produced and returns the trace restriction
  beside it; `_density_basis` carries both and `get_density` attaches both.
* `_unweighting_mode` forces `joint`.
* `_frame_boost` stays on for the mode (see 13.10 step 9).
* `_joint_maxwgt_range` bounds `|w|`; `_unweight_range` accepts on `|w|/maxwgt`
  with **one** draw and writes nothing on rejection, carrying `wsign` onto
  `full_evt.wgt` and onto every entry of `parse_reweight()`.
* `<init>` zeroing plus the `<MGPureInterference>` banner note, and the
  `sum_w` / `sum_w2` / overweight counters with the `z` report in
  `_report_pure_interference`.

**Known boundary:** `fixed_order` is handled (the counter-event group is
dropped as a unit by the same `continue`, and the sign is applied to every
member of the group) but is **not validated** -- no fixed-order sample was run
through the mode.

### 13.10 Implementation plan -- all steps done

1. *(done)* Cross entries in `normalize_hel_restriction` /
   `_restriction_row_mask`, with the algebra tests.
2. `hel_restriction_trace` on `DensityMatrix`, defaulting to `hel_restriction`,
   read by `trace()` and `normalized()`. Behaviour-neutral; one unit test that a
   cross restriction with a `P u D` trace restriction gives a zero numerator
   over a non-zero denominator.
3. `pure_interference` card option, parsing and validation (13.6), feeding
   `_apply_production_polarization` -> `_density_basis['hel_restriction']` and
   the new trace restriction. Refuse sequential/two-stage `unweighting`, refuse
   a non-density spinmode, refuse overlapping sets, cross-check against the
   banner braces (13.5). Unit-testable with the existing `_Stub` pattern in
   `TestProductionPolarizationPlumbing` -- no f2py needed.
4. `abs()` in `_joint_maxwgt_range` and in the accept test, sign carried onto
   `full_evt.wgt` and onto every entry of `parse_reweight()`. Gated on the mode
   so unrelated runs are untouched.
5. Drop-on-reject in `_unweight_range` (13.7b), gated on the mode; suppress the
   `_apply_accounting` BR rewrite and the `efficiency`-driven `nb_event`
   rescaling for this mode, since here a low keep-rate is physics, not a
   correction.
6. `<init>` zeroing plus the reference-normalisation banner note and warning.
7. `sum_w` / `sum_w2` in the stats dict and the `z` report.
8. Validation: steps 1-4 and 7 are unit-testable in-process. Steps 5, 6 and the
   physics closure test need a working end-to-end MadSpin run -- see 13.12.
9. **(added during implementation)** The frame boost. #355 established that the
   polarisation axis must be MG5's `me_frame`, because `set_hel_restriction` is
   a projection and a projection does not commute with the change of helicity
   basis a boost induces. Its guard switches the boost on for a polarised beam
   or a production brace. A cross restriction is a projection for exactly the
   same reason -- and it names two helicity *sets*, which only mean something
   once the axis is fixed -- but the mode's production is unpolarised by
   construction, so that guard would find nothing and leave the momenta in the
   lab. The clause added is:

       if (self._beampol() is None and not self._production_polarization()
               and not self._pure_interference()):
           return None

   A parallel branch factors the same condition into a `_needs_frame_axis()`
   helper; the `pure_interference` clause belongs in that helper once the two
   are merged.

### 13.11 Environment

The assessment was written in an environment where `f2py` could not build
extension modules -- it generated the wrappers and then died with
`meson: command not found` (NumPy drops the distutils backend for Python
>= 3.12). That is fixed: the `mg-3.14` pyenv carries `f2py`, `meson` and
`ninja`, and everything in 13.12 was measured there, not reasoned from source.

### 13.12 End-to-end validation

Sample: `p p > w+ w-` at 13 TeV, 20k unweighted events (`sigma = 64.66 pb`),
`spinmode = madspin` (offshell), `decay w+ > e+ ve` / `decay w- > e- ve~`, and

    set pure_interference w+ = 0 T

i.e. the interference between the longitudinal and the transverse W+.

**The weight sum is compatible with zero.** Five independent seeds, each over
the same 20k production events (`z = S / sqrt(sum w^2)`):

| seed | kept / read | S | sqrt(sum w^2) | z | overweight |
|---|---|---|---|---|---|
| 42   | 1166 / 20000 (5.83%) | +73.476 | 27.272 | +2.694 | 0 |
| 7    | 1218 / 20000 (6.09%) | -20.764 | 27.872 | -0.745 | 0 |
| 99   | 1314 / 20000 (6.57%) | -59.103 | 28.952 | -2.041 | 0 |
| 555  | 1180 / 20000 (5.90%) | -8.9e-16 | 27.434 | -0.000 | 0 |
| 2024 | 1144 / 20000 (5.72%) | -22.361 | 27.012 | -0.828 | 0 |

Combined over the 100k production events: `S = -28.75`, `sqrt(sum w^2) = 61.98`,
**`z = -0.46`**. The per-seed mean is `-0.18 +- 0.79`. The `+2.69` of seed 42 is
the reason the threshold is 5 sigma and not 3: a 2-3 sigma excursion turns up
readily in a handful of runs, and the spread across seeds is what shows it is a
fluctuation rather than a bias. No trial anywhere exceeded the maximum weight,
so the one mechanism that *would* have biased `S` was not active.

**The events carry both signs**, in roughly equal numbers, as they must for a
sample whose integral vanishes: 629+/537- (seed 42), 620+/694- (seed 99).

**`|w|` is the unpolarised magnitude.** Every written event carries exactly one
value, `|w| = sigma * BR = 0.79865722`, and the unpolarised run of the same seed
writes that identical number (relative difference `0.000e+00`). Across seeds it
moves by `4e-5`, which is just the per-run MC estimate of the branching ratio.
This is the design working as intended: the magnitude is constant and the
interference is carried entirely by *which* production events survive -- the
keep rate, ~6% here -- which is exactly what redraw-until-accept would have
normalised away (13.7b).

**A run with `pure_interference` unset is byte-identical.** The same card
without the option, run against the pre-implementation tree (`7e35f7780`) and
against the implementation, produces the same 135,011,930-byte file --
identical whole-file, banner included, not merely in the event blocks.

`tests/test_manager.py test_madspin -t0`: 184 tests, OK (161 before this work).

Caveats, stated rather than glossed:

* only `spinmode = madspin` was exercised end to end; `PA` / `onshell` go
  through the same `_unweight_range` and the same restriction, but were not run.
* `nb_core > 1` was used throughout (18 workers), so the additive merge of
  `sum_w` / `sum_w2` across shards is exercised; the serial path is not
  separately measured.
* `fixed_order` is implemented but unvalidated (13.9).
* the z test assumes the `w_i` are independent. That is true event to event
  here, but not across a production sample that itself came from a correlated
  MC (multi-weight / reweighted samples). It is a sanity check, not a proof.

### 13.13 The fully weighted output -- what replaced the signed accept/reject

**Status: implemented.** This supersedes the accept/reject of 13.7b and the
`+- sigma*BR` weight of 13.7c. The algebra of 13.7b is unchanged and is what
justifies the replacement; only the *representation* of `<|W|>` moves, from the
keep rate into the weight.

**What is written.** Every production event yields exactly one output event --
one decay configuration is drawn, and it is kept, with no accept/reject at all:

    w = sigma_parent * BR * W / c

`W = wgt*jac` is the signed convolution of that trial, and

    c = <W_full>_Omega

is the decay-phase-space mean of the **unrestricted** convolution: the same
quantity, with the cross restriction swapped for the symmetric one that already
normalises it (`hel_restriction_trace`). `c` is a *decay-side* constant -- the
production density matrix cancels between the restricted contraction and its
normalising trace -- which is exactly the constancy that made ordinary
redraw-until-accept unbiased in the first place (13.7b).

**Why this is the right weight.** MG5 writes LHE files with `IDWTUP = -4`, the
convention in which the cross-section is the **mean** of the event weights, not
their sum. (Measured on three sample files in the tree: `ttbar.lhe.gz`,
`wj_zj.lhe.gz`, `hj_heft.lhe.gz` all carry `XWGTUP = sum(XSECUP)` on every
event.) Under that convention:

* `mean(w) = sigma*BR*<W>/c = 0` -- the sample's own cross-section, which is
  correct and consistent with the `XSECUP = 0` the mode writes into `<init>`;
* `sum_bin(w) / N_file` is the interference contribution to that bin, **in pb**,
  with `N_file = N_read`. The file is genuinely self-normalising.

The relation to the ordinary scheme is the one derived in 13.7b: for an
observable `O`,

    S_ord(O) = N_read * sigma*BR * <W_ord O> / c

and writing `w = sigma*BR*W/c` per event makes `sum_events w O` estimate exactly
the interference part of that. `max_weight` does not appear.

**What this fixes, and what it costs.**

* `max_weight` leaves the normalisation completely. Under the accept/reject the
  sample's physical normalisation was `sigma*BR*maxwgt/c` per *read* event --
  i.e. it depended on an internal bound that depends on `nb_sigma`,
  `Nevents_for_max_weight` and the process, and that was written nowhere. The
  closure test had to reconstruct it by hand.
* The overweight-bias channel disappears with the bound: there is no trial that
  can be "accepted with probability 1 instead of `|w|/maxwgt`", so `nb_pi_overflow`
  and its critical message are gone, and the `z` test loses its most likely
  failure mode. The remaining counter is `nb_pi_dead` (a non-finite convolution,
  written with weight 0 and reported loudly -- a dead matrix element, not
  physics).
* All `N` production events are used instead of the 3-9% the accept/reject kept,
  so the statistics per production event are strictly better.
* `n_written == n_processed`, so `_apply_accounting`'s efficiency is 1 and
  nothing downstream is rescaled. (It is still computed from the counts rather
  than hard-coded, so a BR-equalization drop in the same run is still reported.)
* The `<rwgt>` entries follow automatically: `W/c` rides on the `br` factor,
  which is applied to `full_evt.wgt` and to every entry of `parse_reweight()`
  through the same multiplication.
* Cost: the output is a **weighted** sample. Tools that assume unit weights
  break -- but they break on a signed zero-cross-section sample anyway.

**13.7b's objection does not apply.** It argued against *redraw-until-accept*,
which normalises `<|W|>` away by forcing one same-magnitude event per production
point. Carrying `W` in the weight preserves `<|W|>` just as well as carrying it
in the keep rate, with less variance.

**How `c` is obtained.** Inside the existing maximum-weight probe
(`_joint_maxwgt_range`), one extra contraction per draw with
`density_prod.hel_restriction` temporarily swapped for
`density_prod.hel_restriction_trace` -- the same save/swap/contract/restore trick
`_polarization_ratios` already uses, on matrices that are alive anyway. The
probe's own statistics (`Nevents_for_max_weight * max_weight_ps_point`, typically
75 x 400 = 30000 trials) give it to a few tenths of a percent. The sum/sumsq/n
are merged additively across the forked scan workers, so one shard or many gives
the identical estimate, and they are cached beside `max_wgt` in `ms_dir` (a
cached `max_wgt` is only reused when the matching `c` cache is there too).

**The analytic candidate, checked.** Averaging `rho_dec/dec_diag` over the decay
phase space gives `delta_ij/n` (`DensityMatrix.identity`), and everything else in

    W = me * density_iden_prod * density_iden_decay
        / (iden_p * sym_prod * prod_color * prod_denominators * sym_decay)
        / (prod_diag * dec_diag) * jac

cancels -- `prod(spin)/n = 1`, `prod(color)/prod_color = 1`, the trace against
`prod_diag` -- leaving

    c = <jac> / (prod_denominators * sym_factor_decay)

with `prod_denominators = prod_i (m_i Gamma_i)^2`. So the lead was right in form:
`c = 1/prod_denominators` **exactly, but only where the chain carries no
reshuffling jacobian and no decay symmetry factor** -- i.e. `spinmode = onshell`
with one decay per pdg. Under `madspin`/`full` (offshell) and under `PA` the
Breit-Wigner sampling jacobian is inside `W` and `<jac> != 1`. The analytic form
is therefore **not** used: `c` is measured, and `1/(prod_denominators *
sym_decay)` is computed beside it and reported as a cross-check (the ratio of the
two is in the log and in the banner block).

**The banner block, kept.** `<MGPureInterference>` is no longer the
normalisation -- the file normalises itself -- but `XSECUP = 0` deletes the
reference cross-section from the file and the diagnostics have nowhere else to
live. It carries the reference `sigma*BR`, `N_read`, `c` and its error, the
analytic cross-check, the probed `max|W|` (diagnostic only), `S`,
`sqrt(sum w^2)`, `z`, and `mean(w)`.

### 13.14 Card syntax: what is expressible, and the two things that are not

**Repeated `set` lines accumulate.** `extended_cmd.Cmd.precmd` splits *every*
card line on `;` and dispatches the pieces as separate commands, so

    set pure_interference t = + - ; t~ = + -

can never work: the `t~` half is dispatched as its own command, lands in
`Cmd.default`, and used to produce nothing but a generic
`Command "t~" not recognized` warning while the run continued with a
valid-looking single-particle sample. That is a silently-wrong-physics failure,
so two things changed: `do_set` **accumulates** repeated `pure_interference`
lines (`ACCUMULATING_OPTIONS`), and `MadSpinInterface.default` **raises** when
the unrecognised line parses as a bare `particle = polA polB` entry. The
multi-particle spelling is therefore

    set pure_interference t  = + -
    set pure_interference t~ = + -

**Diagonal blocks are nameable.** Two *disjoint* sides give that particle's
interference block `I`; two *identical* sides give its diagonal block `D_S`
(`normalize_hel_restriction` already collapses `(S, S)` back to the symmetric
`S`). So `(I, D-)` of `t t~` is `t = + -` plus `t~ = - -`, from the card alone,
on an unpolarised production -- it no longer needs a production brace on the
other leg the way the closure test did. A **partial** overlap (`T +`) is still
refused: it is neither block, and it puts diagonal entries into an off-diagonal
one so the restricted trace stops vanishing. At least one particle must carry a
genuine interference pair, otherwise the mode's whole apparatus (zeroed `<init>`,
signed weights, separate trace restriction) is being applied to an ordinary
polarised sub-sample. A particle the option does **not** name is left
*unrestricted*, i.e. summed over its whole basis -- which is neither `D+`, nor
`D-`, nor `I`, but their sum. That is what makes `x_t = (I,D+) + (I,D-) + (I,I)`
in the closure test.

**Braces are not an option.** `madgraph_interface.py:5151` hard-rejects
`t{0}{T}` (`rest = '{T}'` -> "A space is required after the "}" symbol"), and a
leg carries a single flat `Leg['polarization']` list of ints that
`polarization` appears 221 times across 15 modules under `madgraph/` reasoning
about. A second, semantically different brace on the same leg has no
representation in that data model. This is a data-model change in shared code,
not a grammar accident -- 13.6's "rejected" stands.

**Leg-index keys are not an option either.** Keying the option on the MG5
process-line leg number (`set pure_interference 3 = + -`) is appealing because
pdg keys cannot say "the first `t` is `I` and the second `t` is `D+`" -- a pdg
entry is broadcast to *every* slot of that pdg. It does not work, for three
independent reasons:

1. **The label carries no physics.** #353's proof that the n-th same-pdg leg
   maps to the n-th density slot rests on `Process.identical_particle_factor`
   keying on `(id, polarization)` (`base_objects.py:3757`): two same-pdg legs
   with *different* braces are not identical to MG5, so nothing permutes them.
   This mode requires an **unpolarised** production, so that protection is
   absent: the two `t` of `p p > t t~ t t~` key to the same `(6, ())`, the
   amplitude is symmetrised, and a `2! 2!` identical-particle factor is applied.
2. **The event record's ordering is not stable.** In grouped output the momenta
   written out are permuted per channel
   (`SWITCHMOM(PP,P1,PERMS(1,MAPCONFIG(ICONFIG)),...)`,
   `super_auto_dsig_group_v4.inc:805`); in ungrouped output `unwgt.f:582-600`
   draws an identical-particle permutation uniformly at random *per event*. So
   "the first `t` in the event record" is set by which channel or which random
   draw produced the event.
3. **The spelling is already taken.** `set pure_interference 3 = + -` parses
   today as *pdg 3* (`pdg = int(name)` when the name is not in `name2pdg`), so an
   integer key would be ambiguous with the existing pdg-code spelling.

Consequence for `p p > t t~ t t~`: the card can ask for one block per *species*
(both `t` slots `I`, both `t~` slots `D+`, ...), which is the symmetrised
statement, and that is the only statement the sample supports. Per-slot
attribution would need a label the sample does not carry.

### 13.15 Why nine blocks and not ten -- the counting, settled

The question comes up every time someone lists the hermitian terms of the joint
`4 x 4` matrix for `t t~`, so it is recorded here rather than re-derived.

**Both counts are right; they count different things.** The six distinct
off-diagonal hermitian *pairs* are

    (++;+-) (++;-+) (++;--) (+-;-+) (+-;--) (-+;--)

so 4 diagonal + 6 pairs = **10 hermitian terms** = `4 + 2*6` = **16 matrix
entries**. The decomposition the code produces has **9 blocks** covering the same
16 entries as `4*1 + 4*2 + 1*4`. The whole difference is that the `(I,I)` block
**bundles two** of the ten terms:

| user term | particle 1 | particle 2 | block |
|---|---|---|---|
| `(++;+-)` | `(+,+)` diagonal | `(+,-)` flip | `(D+, I)` |
| `(++;-+)` | `(+,-)` flip | `(+,+)` diagonal | `(I, D+)` |
| `(+-;--)` | `(+,-)` flip | `(-,-)` diagonal | `(I, D-)` |
| `(-+;--)` | `(-,-)` diagonal | `(+,-)` flip | `(D-, I)` |
| `(++;--)` | `(+,-)` flip | `(+,-)` flip | `(I, I)` |
| `(+-;-+)` | `(+,-)` flip | `(-,+)` flip | `(I, I)` |

**The bundle cannot be split by any legal restriction.** `_restriction_rows`
builds the mask as a strict *product of per-particle conditions*: one loop
iteration per decaying particle, each touching only that particle's own
`(bra, ket)` columns, combined with `&=`. Keeping only `(++;--)` needs
`((+,-)&(+,-)) OR ((-,+)&(-,+))` -- a **union of two products**, not a product of
unions. Any product mask containing both `(+,-)&(+,-)` and `(-,+)&(-,+)` must
offer `{(+,-),(-,+)}` on *each* particle and therefore also contains
`(+,-)&(-,+)`. Nor can it be recovered afterwards: the nine blocks are the atoms
of the lattice of legal (transposition-closed, hence real) product masks, they
are pairwise disjoint and they tile the 16 entries, so a set that is not a union
of them is not any signed sum of them either. The `(I,I) = x_t - (I,D+) -
(I,D-)` subtraction has no analogue one level down.

**And neither half is an observable on its own.** Writing
`rho = 1/4 [1x1 + B+.sigma x 1 + 1 x B-.sigma + C_ij sigma_i x sigma_j]` in the
helicity basis with `(x,y,z) = (r,n,k)`,

    4 Re rho(++;--) = C_rr - C_nn
    4 Re rho(+-;-+) = C_rr + C_nn

so each both-flip term is the half-sum `(C_rr +- C_nn)/2`; `C_nn` is their
difference and `C_rr` is their sum. Even with a union-of-products
generalisation there would be no "`C_nn`-only" sample -- only two samples whose
difference is `C_nn`.

**In practice the bundling costs nothing.** The `(I,I)` sample contracts the
whole block against the decay density matrix, which supplies the angular
structure separating `C_nn` from `C_rr` at the *observable* level, and the
closure measured both from the same events: the interference contributes
`+0.03626 +- 0.00090` to `<C_nn>` and `+0.00247 +- 0.00091` to `<C_rr>`, and the
whole `C_nn` effect sits in `(I,I)` while the four singly-interfering blocks are
flat at zero in it. What the bundling costs is only the ability to attribute a
given *event* to one of the two terms.

**Decision: do not split `(I,I)`.** It would require a union-of-products
normalised form in `decay.py` plus card syntax naming the correlation between
two particles' flip directions -- a separate feature, exactly as 13.9 says of its
"global" cousin -- and it would buy a distinction between two quantities neither
of which is an observable and both of which are already measurable from the
single `(I,I)` sample.

### 13.16 End-to-end validation of the fully weighted mode

`p p > t t~` at 13 TeV, NNPDF23LO, `me_frame = [1,2]`, **50 000** production
events (`iseed = 4321`), `spinmode = onshell`, `BW_cut = 15`,
`max_weight_ps_point = 400`, `decay t > b w+, w+ > l+ vl` and the conjugate,
`l = e, mu`, 8 cores. The card uses the accumulating spelling, i.e. the
`(I, I)` block -- the one the closure test could only reach by subtraction:

    set pure_interference t  = + -
    set pure_interference t~ = + -

**The weight sum is compatible with zero, and every event is written.**

| | |
|---|---|
| events written / read | 50000 / 50000 (the mode no longer rejects) |
| `<init>` `XSECUP` / `IDWTUP` | `0.0` / `-4` |
| reference `sigma*BR` | 23.763645 pb |
| `S = sum w` | `-9.777639e+02` |
| `sqrt(sum w^2)` | `9.815884e+02` |
| `z = S / sqrt(sum w^2)` | **`-0.996`** |
| `mean(w)` | `-1.9555e-02 +- 1.9632e-02`, i.e. `-0.082%` of `sigma_ref` |
| positive / negative weights | 24918 / 25082 |
| `mean|w| / sigma_ref` | 0.13007 (0.13011 on an independent 2 000-event run) |
| trials with a dead weight | 0 |

`mean(w) = 0` is the sample's own cross-section under `IDWTUP = -4`, and it
agrees with the `XSECUP = 0` written into `<init>`.

**`c` agrees with the analytic form.** Measured `c = 2.258515e-10 +- 0.13%` over
44 016 probe trials against `1/(prod_denominators * sym_decay) = 2.255914e-10`
for the default SM card (`m_t = 173.0`, `Gamma_t = 1.4915`, so
`1/(m_t Gamma_t)^4`): **ratio 1.001153**, i.e. 0.9 sigma of the measurement.
The independent 2 000-event run gave `2.259903e-10 +- 0.15%`, ratio 1.0018. So
under `spinmode = onshell` -- where `<jac> = 1` and `sym_factor_decay = 1` --
the analytic form is confirmed. It is still not what the code uses, because
`<jac> != 1` under `madspin`/`full` and `PA`; it is recorded in the banner as a
cross-check.

**The physics closes against the independent closure test.** The interference
contribution to an observable is `sum_i w_i O_i / N_read`, divided by
`sigma_ref` to compare with the closure's `<O>` shifts (`RESULTS.md` section 6):

| observable | this run, `(I,I)` alone, 50k | closure, all 5 interference blocks, 5 x 50k | pull |
|---|---|---|---|
| `<C_nn>` | **+0.037205 +- 0.000364** | **+0.03626 +- 0.00090** | +0.97 |
| `<C_rr>` | +0.002246 +- 0.000375 | +0.00247 +- 0.00091 | -0.22 |
| `<C_kk>` | -0.000085 +- 0.000165 | -0.00034 +- 0.00062 | +0.39 |
| `<cos phi_ll>` | +0.039366 +- 0.000564 | +0.03839 +- 0.00145 | +0.63 |
| `<Delta phi>` | -0.066684 +- 0.001761 | -0.07051 +- 0.00491 | +0.73 |
| `<pT(t)>` (null test) | -0.63 +- 2.86 | +0.267 +- 0.351 | -0.31 |

Every entry agrees within one sigma, `(I,I)` alone reproduces the whole
interference (13.15 and `RESULTS.md` 6b), and `pT(t)` -- a production-level
observable, which the interference must not touch -- is flat at zero.

**The statistics are 2.5-2.8x better per observable from one fifth of the
production events**, i.e. roughly a factor 30-40 in variance per production
event, which is what dropping the 3-9% accept/reject and carrying `<|W|>` in the
weight buys.

**A run with `pure_interference` unset is byte-identical.** The same card
without the option, re-run through `decay_events` on the same 2 000-event parent
against the implementation and against the pre-change tree (`bdf383554`),
produces the same 3 633 126-byte file, identical SHA-256
(`160f0cff56d2f1ad138a9950de607c8a8366df1d69005ef8bf2c1aad8cf71fc3`), banner
included.

**Both `set` lines accumulate and the `;` spelling fails loudly.** The run above
logs `pure_interference is ON for particle(s) -6, 6` and the banner block lists
both pdgs; `set pure_interference t = + - ; t~ = + -` raises `InvalidCmd`.

`tests/test_manager.py test_madspin -t0`: **284 tests, OK** (269 before this
work).

Caveats:

* only `spinmode = onshell` was exercised end to end here (13.12 exercised
  `madspin`); `PA` and `fixed_order` were not run;
* `c` is measured with the probe's own statistics, so it carries a flat scale
  error on every weight -- 0.13% here, warned about above 5%;
* the analytic cross-check is confirmed only in the `<jac> = 1` case, which is
  exactly where the derivation says it should hold. It has **not** been checked
  offshell, where the derivation says it should *not* hold.

### 13.17 The unweighted-up-to-a-sign output -- `decay_output = unweighted`

**Status: implemented and validated end to end.** The fully weighted output of
13.13 stays the **default**; `set decay_output = unweighted` selects the other
representation of the same estimator, in which the sample carries exactly two
weight magnitudes.

> **Option name.** This was originally a separate option,
> `pure_interference_output`, with its own `weighted`/`unweighted` pair. It has
> been folded into `decay_output` (13.18, 13.19): one option now answers "does
> MadSpin unweight?" in both modes, and `decay_output = auto` -- the default --
> resolves to `weighted` here and to `unweighted` for an ordinary run, which is
> each mode's own historical default.

**The derivation.** Unweight on `|W|` against any bound `M >= max|W|`, ONE
decay draw per production event, nothing written on rejection, and give each
accepted event `w = sign(W) * w0`. Then `N_file = N_read * <|W|>/M` and, for
any observable `O`,

    sum_written w O = N_read * w0 * <W O> / M         (because |W| sign(W) = W)

    (1/N_file) sum_written w O = w0 * <W O> / <|W|>

The `M` of the acceptance probability cancels against the `M` of the file
size. Matching the interference contribution `sigma*BR*<W O>/c` -- the same
target the fully weighted output hits, per read event -- gives

    w0 = sigma_ref * BR * <|W|> / c

with **no `max_weight` in it**. `mean(w) = w0 <W>/<|W|> = 0` still holds,
because `<W> = 0`, so `XSECUP = 0` remains correct and both variants obey the
same `IDWTUP = -4` rule: `sum_bin(w) / N_file` is the contribution in pb, with
`N_file` the number of events *in the file* (which is `N_read` only for the
fully weighted output).

**The design notes' `w = +- sigma*BR*maxwgt/c` is not wrong physics; it is
normalised per event READ.** `maxwgt/c = (<|W|>/c) * (N_read/N_file)`, so the
two differ by exactly `N_read/N_file`. An LHE file carries no `N_read`, and
`IDWTUP = -4` says the cross-section is the mean of the weights over the file,
so a consumer that divides by `N_file` -- the only count it has -- would be off
by `M/<|W|>` (a factor 13 in the run below), and the factor depends on the
internal bound. That is what made `maxwgt` look load-bearing. It is not.

**`<|W|>` must NOT come from the maximum-weight probe.** This is the one thing
the derivation does not tell you and the run does. Unlike `c`, `<|W|>` is not
a decay-side constant -- it is the *local size of the interference* and varies
from production point to production point, which is the whole content of
13.7b. The probe sees `Nevents_for_max_weight` production events (112 in the
run below) and its `max_weight_ps_point` draws on each are all correlated
through that point's `|W|` scale, so:

* the trial-level error is meaningless here. The probe claimed
  `<|W|> = 3.168e-11 +- 0.46%`; blocked by production event the error is
  **5.0%**, and the truth was `2.895e-11`, i.e. the probe was **9.4% high**;
* it is not an ordering bias -- 2000 random 110-event subsamples of the same
  file have a 9.5% spread and the probe's value sits 0.8 sigma inside it. It
  is simply the wrong sample size for the quantity;
* a 9.4% error on `<|W|>` is a 9.4% error on every physics number the file
  produces, because the estimator is linear in `w0`.

**The run normalises with its own keep rate instead, which is exact.** Putting
`<|W|> = (N_file/N_drawn) * M` into `w0` makes `N_file` cancel out of the
estimator altogether:

    (1/N_file) sum w O = (M sigma_ref BR / (c N_drawn)) sum_accepted sign(W) O

whose expectation is `sigma_ref*BR*<W O>/c` exactly, with no estimate of
`<|W|>` in it anywhere and no residual `M` dependence even in principle. The
run therefore writes the probe's provisional magnitude during the loop and
divides it out afterwards, in the same pass that inserts the
`<MGPureInterference>` note (`_rewrite_lhe_banner_cross(event_scale=...)`);
`S`, `sqrt(sum w^2)` and `mean(w)` are rescaled with it and `z` is invariant.
The probe's `<|W|>`, its blocked error and the correction factor are all in
the banner block, and a correction beyond 25% warns.

**What comes back that the fully weighted output had lost:** the bound is live
again -- the acceptance probability clips at 1 when `|W| > M` -- so
`nb_pi_overflow` and its `logger.critical` are back on this path (and only on
it). `_dead_trial` is still not on the pure-interference path at all: a
negative weight is normal here, and `nb_pi_dead` counts the genuinely
non-finite trials.

**End-to-end validation.** Same setup as 13.16 -- `p p > t t~` at 13 TeV,
NNPDF23LO, `me_frame = [1,2]`, 50 000 production events (`iseed = 4321`),
`spinmode = onshell`, `BW_cut = 15`, `max_weight_ps_point = 400`, `l = e, mu`,
8 cores, the `(I, I)` block -- run four times on the **same** parent file:
fully weighted, and unweighted against three bounds set by `nb_sigma`.

| | weighted | `nb_sigma = 0` | `nb_sigma = 10` | `nb_sigma = 40` |
|---|---|---|---|---|
| `M = max\|W\|` used | (none) | 3.845e-10 | 6.351e-10 (1.65x) | 2.235e-09 (5.81x) |
| events written / read | 50000 / 50000 | 3764 / 50000 | 2305 / 50000 | 620 / 50000 |
| distinct \|w\| magnitudes | 49985 | **1** | **1** | **1** |
| positive / negative | 24921 / 25079 | 1855 / 1909 | 1135 / 1170 | 326 / 294 |
| `\|w\|` (pb) | -- | 3.05767 | 3.09265 | 2.92710 |
| `<\|W\|>` realised | 2.928e-11 | 2.895e-11 | 2.928e-11 | 2.771e-11 |
| `S` | -4.4436e+02 | -1.6511e+02 | -1.0824e+02 | +9.3667e+01 |
| `sqrt(sum w^2)` | 9.8358e+02 | 1.8759e+02 | 1.4848e+02 | 7.2884e+01 |
| `z` | **-0.452** | **-0.880** | **-0.729** | **+1.285** |
| `mean(w)` | -8.887e-03 | -4.387e-02 | -4.696e-02 | +1.511e-01 |
| trials above `M` | -- | 0 | 0 | 0 |

**`M` cancels.** The three bounds span 5.8x and the files they produce span
6.1x in size, and the physics is the same:

| observable | weighted | `M` | `1.65 M` | `5.81 M` |
|---|---|---|---|---|
| `<C_nn>` | +0.037286 +- 0.000366 | +0.036567 +- 0.000859 | +0.037049 +- 0.001111 | +0.035850 +- 0.002011 |
| `<C_rr>` | +0.001744 +- 0.000375 | +0.002458 +- 0.000830 | +0.003282 +- 0.001073 | +0.004786 +- 0.001952 |
| `<C_kk>` | +0.000108 +- 0.000165 | -0.000273 +- 0.000530 | -0.000605 +- 0.000690 | +0.000543 +- 0.001247 |
| `<cos phi_ll>` | +0.039139 +- 0.000564 | +0.038753 +- 0.001328 | +0.039726 +- 0.001716 | +0.041178 +- 0.003132 |
| `<Delta phi>` | -0.065226 +- 0.001768 | -0.066889 +- 0.004384 | -0.068263 +- 0.005661 | -0.047423 +- 0.010122 |
| `<pT(t)>` (null) | +0.04 +- 0.12 | -0.24 +- 0.30 | -0.38 +- 0.39 | +0.40 +- 0.68 |

Every unweighted entry is within one sigma of the fully weighted one on the
same events (largest pull -0.88, on the `pT(t)` null test). The three
independent measurements of `<|W|>` the runs realise -- 2.895, 2.928,
2.771e-11 -- agree to their own 1.6% / 2.1% / 4.0% counting errors, which is
the same statement one level down.

**The closure.** Against the committed `interference_closure_v2` numbers
(`RESULTS.md` section 4), for the `nb_sigma = 0` run:

| | closure v2 | closure v1 | this run, unweighted | pull vs v2 | pull vs v1 |
|---|---|---|---|---|---|
| `<C_nn>` | +0.03657 +- 0.00059 | +0.03626 +- 0.00090 | **+0.036567 +- 0.000859** | **-0.00** | +0.25 |
| `<C_rr>` | +0.00104 +- 0.00066 | +0.00247 +- 0.00091 | +0.002458 +- 0.000830 | +1.34 | -0.01 |

**The variance penalty is confirmed at 5.5-6.2x, from the other direction.**
Same 50 000 production events, ratio of the errors squared, unweighted over
fully weighted:

| observable | error ratio | variance ratio |
|---|---|---|
| `<C_nn>` | 2.35 | **5.5** |
| `<cos phi_ll>` | 2.35 | **5.5** |
| `<Delta phi>` | 2.48 | **6.2** |
| `<C_rr>` | 2.21 | 4.9 |

13.16 measured 5.8 / 5.7 / 6.1 by comparing against a different (5x larger)
reference; this is a direct like-for-like measurement on identical events and
it agrees. That is why the default stays `weighted`.

**Unchanged elsewhere.** A run with `pure_interference` unset produces the
identical 90 368 979-byte file, SHA-256
`767da240c5221ecc0d7193a3031044b3304a457f909212158728c2ab7f242855`, against
the base branch and against this one. The fully weighted run's event stream is
byte-identical too (`event_scale` is `None` there, so
`_rewrite_lhe_banner_cross` does not touch an event).

`tests/test_manager.py test_madspin -t0`: **325 tests, OK** (291 before this
work, counting 13.18's).

Caveats:

* only `spinmode = onshell` was exercised for this variant;
* the normalisation is exact in expectation but `N_file` is itself random, so
  `w0` carries a `1/sqrt(N_file)` relative error (1.6% at 3764 events). It is
  a *self-normalisation* error of the ratio estimator, not a bias, and it is
  small next to the 5.5x variance penalty;
* an `ms_dir` reused across MadSpin runs gave `branching_ratio = 0` (hence
  zero weights and a zero reference cross-section) on a card that ran
  correctly with a fresh `ms_dir`. That is **pre-existing** -- nothing in this
  work touches the branching-ratio path -- but it was hit while validating and
  is recorded here.

### 13.18 `decay_output = weighted` -- the same trick for an ordinary run

**Status: implemented and validated end to end.** `set decay_output =
weighted` drops the accept/reject for an ordinary (non-interference) MadSpin
run: one decay configuration is drawn per production event and kept, with

    w = w_prod * BR * W / c

exactly the fully weighted path of 13.13, only with `W` unrestricted. Default
`unweighted`, i.e. every existing card is untouched.

**The normalisation needs nothing new.** `c = <W>` is a decay-side constant --
that is the 13.7b argument, and it is what makes redraw-until-accept unbiased
in the first place -- so

    mean(w) = sigma_ref * BR * <W> / c = sigma_ref * BR

MG5 writes `IDWTUP = -4`, under which the cross-section *is* the mean of the
event weights, so `<init>` keeps its ordinary value and nothing downstream has
to be told a new rule. That is the whole difference from the interference
mode, where `<W> = 0` forces `XSECUP = 0`.

It also gives the mode a free self-check with no analogue elsewhere: `mean(w)`
against `sigma_ref * BR` **is** the statement that `c` was measured right,
because `mean(w)/(sigma*BR) = <W>/c` by construction. It is the exact
analogue of the interference mode's `z` test, with the target at 1 instead of
0, and it is reported in the log and in the `<MGWeightedDecay>` banner block,
`logger.critical` beyond 5 sigma.

**`c` is available on every path this covers**, from the same probe: it is
measured in `_joint_maxwgt_range` with the cross restriction swapped for
`hel_restriction_trace`, and outside the interference mode those two are the
same object, so the swap is a no-op and what comes out is `<W>` itself. The
probe is *not* skipped when the option is on -- `max_weight` goes unused, but
`c` does not, and the probe is the only thing that measures it. (`<|W|>` is
collected on the same draws and is unused here.)

**Scope: the density spin modes only.** `madspin`/`full`, `PA`, `onshell`.
`madspin_v1`, `onshell_v1` and `spinmode = none` build no density matrix and
have no `W`; the option raises `InvalidCmd` there rather than being ignored.
Under `pure_interference` it does **not** step aside -- it chooses that mode's
output shape instead (13.17, 13.19). The two constraints compose without
conflict, because `pure_interference` needs a density spinmode as well;
`_validate_pure_interference` runs first so the message names the more
fundamental of the two.

**It forces the joint path**, for the plain reason that there is no
accept/reject left to stage: the sequential and two-stage schemes exist to
split a test that is not being made.

**Interactions.** `fixed_order`: the counter-event group already rides along
through the same `br` multiplication, unchanged -- implemented, not validated,
exactly as in 13.9. `keep_weight_for_polarization_*`: allowed and still
meaningful, unlike in the interference mode -- the ratios multiply a nominal
weight that is now weighted, and the ratio itself is untouched. BR
equalization: unchanged, and it now shares the banner-rewrite pass with the
note. `_dead_trial` is **not** on this path: it exists to break a `while 1`
that no longer runs. In its place, a trial with `W <= 0` or non-finite is
written with **weight 0** and counted. `W < 0` outside the interference mode
means `jac <= 0`, i.e. a mass set the production could not be reshuffled onto,
which the accept/reject would have redrawn; there is no redraw here and the
event would carry the failed reshuffle's kinematics, so it gets the weight
that region contributes to the integral (zero) rather than the negative one
that would make the bookkeeping add up on an unphysical event. Zero such
trials occurred in the run below.

**End-to-end validation.** `p p > t t~` at 13 TeV, 50 000 production events
(`iseed = 4321`), `spinmode = onshell`, `BW_cut = 15`,
`max_weight_ps_point = 400`, `decay t > b w+, w+ > l+ vl` and the conjugate,
`l = e, mu`, 8 cores -- the same parent file as 13.17, run once with the card
default and once with `set decay_output weighted`.

| | default | `decay_output = weighted` |
|---|---|---|
| scheme taken | `sequential` (`auto`) | `joint` (forced) |
| decay trials per written event | **4.13** (206 289 / 50 000) | **1** |
| events written | 50 000 | 50 000 |
| `<init>` XSECUP | 23.779781 | 23.779781 (unchanged) |
| `sd(w)/mean(w)` | 0.0000 | 0.2779 |
| `mean(w)` | 23.779781 | **23.783611 +- 0.029563** |
| `mean(w)` / `sigma_ref*BR` | 1 | **1.000161 (0.13 sigma)** |
| `c` measured | -- | 2.251356e-10 +- 0.13% |
| trials with a dead weight | -- | 0 |

`c` is the same 2.251356e-10 the interference runs of 13.17 measured on the
same parent, which it has to be: it is the unrestricted convolution's mean
either way.

**The physics is the same and the variance penalty is small.**

| observable | default | weighted | var ratio | pull |
|---|---|---|---|---|
| `<C_nn>` | +0.037409 +- 0.001479 | +0.036763 +- 0.001520 | 1.06 | -0.30 |
| `<C_rr>` | +0.000651 +- 0.001485 | +0.000982 +- 0.001540 | 1.07 | +0.15 |
| `<C_kk>` | +0.036655 +- 0.001481 | +0.035017 +- 0.001576 | 1.13 | -0.76 |
| `<cos phi_ll>` | +0.074716 +- 0.002554 | +0.072762 +- 0.002671 | 1.09 | -0.53 |
| `<Delta phi>` | +1.749616 +- 0.004042 | +1.748802 +- 0.004232 | 1.10 | -0.14 |
| `<pT(t)>` | 119.864 +- 0.348 | 120.045 +- 0.367 | 1.11 | +0.36 |

**So: 4.13x fewer decay trials for a 6-13% larger variance per production
event, i.e. roughly a 3.7-3.9x variance reduction per unit of CPU spent in
the unweighting loop.**

Note this is a *different*, and much smaller, effect than the interference
mode's ~6x **per production event** (13.17). There the accept/reject discards
whole production events, so the weighted output buys statistics outright;
here the accept/reject redraws and every production event yields an output
event either way, so the weighted output is strictly noisier per event -- it
is importance sampling against exact sampling -- and the win is entirely in
CPU. Which of the two matters depends on whether MadSpin or the parent
generation is the bottleneck. That is why the default stays `unweighted`.

**Byte-identical with the option off.** The same card without
`decay_output` (i.e. at the default), run against the base branch and against
this one, produces
the same 90 368 979-byte file, SHA-256
`767da240c5221ecc0d7193a3031044b3304a457f909212158728c2ab7f242855`.

`tests/test_manager.py test_madspin -t0`: **325 tests, OK** (291 before this
work).

**One regression, and what caught it.** Adding this option replaced
`_unweight_range`'s exit condition `if pure_interference or
random.random()*maxwgt < test` by a flag that was true only for the paths
keeping *every* trial. 13.17's 'unweighted' path does not keep every trial --
but it has already decided, on `|W|`, further up -- so it fell through to the
signed test and had every negative weight rejected a second time: 356 events
instead of 3764, z = +18.9. The condition is "no ordinary joint test is made
below", which is true for all three paths.

The mode's own `z` check did catch it (far past the 5-sigma
`logger.critical`, a `RuntimeError` under `density_debug`) -- but only after a
full run. Six tests now drive the real `_unweight_range` through all three
shapes with the matrix element stubbed out; two of them fail on the bad
condition.

Caveats, stated rather than glossed:

* only `spinmode = onshell` was exercised end to end; `madspin`/`full` and
  `PA` go through the same `_unweight_range` and the same `c`, but were not
  run. `PA` in particular is the one path where the outer reshuffling
  jacobian is live, i.e. where the `W <= 0` handling above can actually fire,
  and it has **not** been exercised;
* `fixed_order` is implemented but unvalidated;
* **no downstream consumer was checked.** A weighted LHE is not what most
  tooling expects from MadSpin, and I have not run Pythia8, Delphes or any
  analysis framework on one of these files. `IDWTUP = -4` with non-constant
  `XWGTUP` is legal LHEF and Pythia8's reader does take the per-event
  `XWGTUP`, but I did not verify it, and anything that counts events instead
  of summing weights will be wrong. The option comment and the banner block
  both say so;
* `c` carries a flat scale error on every weight (0.13% here). Unlike the
  interference mode's `<|W|>`, `c` really is a decay-side constant, so the
  probe's few production events are enough for it -- and `mean(w)` against
  `sigma*BR` is the direct measurement of whether that held.

### 13.19 One option: `decay_output`, with `auto`

**Status: implemented; behaviour-preserving by construction, checked against
the base branch.** 13.17 and 13.18 arrived as two options with the same value
space and the same question behind them -- *does MadSpin unweight?* --
answered separately for the interference mode (`pure_interference_output`) and
for an ordinary run (`decay_output`). They are now one.

* `pure_interference_output` is **removed**. Nothing maps onto it and no
  deprecated spelling survives: none of this has been in a release, so there
  are no cards in the wild to protect.
* `decay_output` gains **`auto`**, and `auto` is the default.
* `auto` resolves to `weighted` when `pure_interference` is set and to
  `unweighted` otherwise (`_decay_output`).

**Why those two directions, and why this preserves behaviour exactly.** The
old defaults were `decay_output = unweighted` and
`pure_interference_output = weighted`, and each mode saw only its own option
(`decay_output` warned and stepped aside under `pure_interference`). So the
pair (ordinary run, interference run) had exactly the resolved defaults
(`unweighted`, `weighted`) -- which is what `auto` now computes. A card that
does not mention either option therefore lands on the same path as before, in
both modes.

They point opposite ways for a reason rather than by accident. The ordinary
run writes one event per production event either way and its accept/reject is
the *exact* sampler, so unweighting is the safe default and the weighted path
buys CPU at the cost of a weighted file. The interference mode has no exact
sampler to fall back on -- its weights are signed and its cross-section is
zero by construction -- and unweighting on `|W|` there keeps only a few
percent of the production events, for ~6x the variance on exactly the
observables the mode exists to measure (13.17).

**The step-aside is gone.** `_validate_weighted_decay` used to warn and return
under `pure_interference`, on the grounds that the other option governed
there. There is no other option now, so it governs. What the step-aside was
avoiding was a *contradiction* between two live options, not a code hazard:
the two flags reach the worker separately (`weighted_decay` and
`pure_interference_unweighted` in the run context) and `_weighted_decay` still
returns False under `pure_interference`, because the interference mode reaches
the same "keep every trial" branch by its own route, with a signed `W` and a
zeroed `<init>`. Only the *source* of the interference mode's choice changed.

**The two spinmode restrictions compose.** `decay_output = weighted` needs a
density spinmode (there is no `W` otherwise) and so does `pure_interference`,
so the constraints never disagree -- but a card that violates both would get
two refusals in a row, the less useful one first. `_validate_pure_interference`
is therefore now called *before* `_validate_weighted_decay`, and both are
called before the `if self._density_spinmode():` branch. `decay_output` is
then silent under `pure_interference`: the mode announces its own output shape,
spinmode requirement included.

That reordering fixes a **pre-existing gap** found on the way:
`_validate_pure_interference` was called only *inside* the density branch, so
`set spinmode none` together with `set pure_interference ...` reached no
validation at all and the mode was silently inert while the card asked for it.
It now raises, which is the error that was always intended (the raise existed;
it was unreachable).

**`auto` announces itself** through `_announce_decay_output`, on the same
`_log_once` convention as `_announce_mode`:

    MadSpin: decay_output = unweighted (auto, ordinary run)
    MadSpin: decay_output = weighted (auto, pure_interference is set)
    MadSpin: decay_output = weighted (set explicitly)

**Removed alongside it**, for the same "not in a release" reason, two other
deprecated spellings that were pure load-time translations with no run-time
reader: `sequential_decay` (mapped onto `unweighting`: `True` ->
`sequential`, `False` -> `joint`) and `keep_weight_for_polarization` (the
singular alias that set both `keep_weight_for_polarization_vector` and
`_fermion`). The per-species options and the refusal of
`keep_weight_for_polarization_*` under `pure_interference` are untouched.

**The one behaviour change, stated rather than glossed.** A card that combined
`set pure_interference ...` with an *explicit* `set decay_output unweighted`
used to get the fully weighted interference output (the explicit
`decay_output` was warned about and ignored, and `pure_interference_output`
kept its `weighted` default); it now gets the unweighted-up-to-a-sign output.
That is the intended meaning of the unification -- the option no longer steps
aside -- and it is the only combination whose resolved behaviour differs. A
card that does not set `decay_output` is unaffected in either mode.
