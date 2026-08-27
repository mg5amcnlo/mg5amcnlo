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

### Mass ownership

The mass logic sits in three places, which do not obviously compose once the
draw has to happen per slot:

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

Consequences, all favourable:

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
expensive one, whose call count sequential *reduces*. If profiling ever shows
the contraction dominating at large n, the next step would be a true partial
contraction (fold fixed indices away so later steps act on a smaller tensor).
It has not been needed.

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
measurements behind them are in section 10 ("The option: one knob, five
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

The other two options this document is about are

    set decay_output auto | unweighted | weighted    # does MadSpin unweight at
                                                     # all -- 13.13, 13.17,
                                                     # 13.18, 13.19
    set pure_interference t = 0 T                    # section 13

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

Keep the mass-set stage, but replace the *per-slot* accept/reject by a single
test on the product of every slot's weight, redrawing the whole angle set on a
rejection and **keeping the mass set**:

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

#### The option: one knob, five schemes

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

The rule the scan replaced was: 1 -> joint; PA/onshell ->
`sequential_with_mass`; 2 -> `two_stage`; 3+ -> `sequential`. Two of those four
branches were wrong and one was unnecessary:

    spinmode         n      before                measured best
    PA / onshell     any    sequential_with_mass  sequential  (1.2x - 3.8x)
    madspin / full   1      joint                 joint       (correct)
    madspin / full   2      two_stage             joint       (1.1x)
    madspin / full   3+     sequential            sequential  (correct)

`two_stage` is not the fastest scheme at any point measured here, in either
spinmode: joint beats it at n<=2 and `sequential` beats it at n>=3. It is worth
keeping reachable -- it is the one staged scheme whose angle stage is a single
joint test, which makes it the natural cross-check against joint -- but it does
not earn a branch in `auto`, and it is no longer offered in the card's
advertised values either.

**The multiplicity rule `auto` implements** (`_auto_unweighting_mode`) is
therefore two lines:

    PA / onshell     ->  sequential
    madspin / full   ->  joint for n <= 2, sequential from n = 3

with the caveat that every number above is one process per multiplicity on one
machine, and that the n=2 offshell call is a 10% difference that went the other
way at a smaller sample size -- so that boundary is the one to revisit if a
process is found where a staged scheme pays off at two decays. The n=1 offshell
and the n>=3 conclusions are not close and are safe.

### A polarised production overrides the multiplicity rule

Measured after this scan, and the one branch of `auto` the scan does not
describe: when the production process carries a polarisation brace
(`_production_polarization`), `auto` takes `sequential` at **every**
multiplicity, offshell included.

The brace restricts the production/decay convolution to a polarisation
subspace, which peaks the joint weight far below the bound the max-weight scan
hands it -- and the joint test has no way to recover, because its bound is a
single number over the whole chain. On `p p > t t~` with both tops decayed
(n = 2, so the rule above would say joint), trials per accepted event over 500
events:

    production          joint    sequential
    t t~ (unpolarised)    3.3       6.1
    t{+}t~{+}           112         9.1
    t{+}t~{-}           162         8.4

and at 50000 events, where the max-weight scan is longer and `nb_sigma` larger,
the joint column rises to 4.05 unpolarised, 204-213 like-helicity and 5800-6300
opposite-helicity against 8.59 sequential. The gap *widens* with statistics,
because the bound the joint test must clear keeps growing while the bulk of the
restricted weight distribution does not.

The asymmetry is what decides it: taking `sequential` where joint would have
done costs the ~2x of the first row, taking joint where the convolution is
restricted costs 30-1500x. So the clause fires on any brace in the production
line -- including one on a particle MadSpin does not decay, which cannot be the
thing peaking the weight. The other reason it must fire unconditionally is that
the resolved mode has to be the same at every call site (it names the max-weight
cache files and picks which bound the accept/reject tests against) while the set
of decayed pdgs is not known everywhere `_unweighting_mode` is called; a clause
that consulted it could resolve two ways in one run.

An explicit `set unweighting joint` is still honoured: only `auto` comes
through here.

---

## 13. Pure-interference mode

**Status: implemented and validated end to end** (13.12, 13.16, 13.17). 13.1
to 13.9 were written as a feasibility assessment before the mode existed and are
kept as the derivation, because every design decision in them is still the one
in the code -- the one exception being the *shape of the output*, which 13.13
and 13.17 replaced and which is flagged where it appears.

The caveats the assessment listed all held up: the mode is a structural change
to the accept/reject loop, it is incompatible with the sequential schemes, and
it produces an LHE file whose `<init>` cross-section is zero, which several
downstream tools cannot consume. The output rework was indeed the hard part.

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
  pure-interference mode therefore forces `unweighting = joint`**, announcing
  it once in the log, rather than silently producing zero weights and hanging
  in the redraw loop.

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
2. **A dedicated MadSpin-card option**, which is what was built:

       set pure_interference t = 0 T          # or: 6 = 0 T

   a dict-valued parameter `pdg -> (production_pol, decay_pol)`, parsed with the
   same `{0}/{+}/{R}/{-}/{L}/{T}` vocabulary `_apply_production_polarization`
   already validates against the `hel_dict` basis. Validation at parse time:
   both sides expressible in the basis; the two sides **disjoint** (an overlap
   re-admits diagonal entries, so the trace stops vanishing and the mode stops
   being "pure interference" -- refuse rather than warn); spinmode is a density
   one. (`unweighting` is not validated but forced to `joint`, 13.4.) `do_decay`'s existing
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
the number of kept events per production point is proportional to `Int_p`, and
for any observable `O` the sum over kept events of `sign(W) O` estimates the
integral of `W(Omega) O(Omega)` -- the interference distribution, correctly
normalised relative to the parent sample. The expected weight sum is the
integral of `W`, i.e. zero, as required.

*(The argument above is what the code rests on and is unchanged. Its two
conclusions about the output are not what shipped: the accepted event's weight
is `sign(W) * sigma_ref*BR*<|W|>/c`, not `+- w_p*BR` -- see 13.17, which
derives it and explains why the design note's per-read-event normalisation
would be off by `M/<|W|>` in an LHE file -- and the accept/reject itself is not
what `auto` picks, 13.13 having replaced it with a fully weighted output that
carries `<|W|>` in the weight instead of in the keep rate.)*

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

That is what the mode does; the banner note it writes is `<MGPureInterference>`
and 13.13 lists what ended up in it. There is no option to write a non-zero
`XSECUP` instead -- once the weights carry `W/c` the file normalises itself
under `IDWTUP = -4` (13.13), so there is nothing for such an option to buy.

### 13.8 The statistical check

After the loop, with kept weights `w_i` (each `+- w_p * BR`):

    S     = sum_i w_i
    delta = sqrt( sum_i w_i^2 )        # MC error on S; there is no cancellation
                                       # in the second moment, so this is the
                                       # right scale to compare S against
    z     = S / delta

Report `z`, and fail the check when `|z|` exceeds 5 -- not the card's `nb_sigma`
(default 3), so that a legitimate 3-sigma fluctuation in a large run does not
cry wolf. 13.12 shows a +2.69 turning up in five seeds, which is why.

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
* **the frame boost stays on for the mode.** The `me_frame` section of section 1
  established that the polarisation axis must be MG5's, because
  `set_hel_restriction` is a projection and a projection does not commute with
  the change of helicity basis a boost induces. A cross restriction is a
  projection for exactly the same reason -- and it names two helicity *sets*,
  which only mean something once the axis is fixed -- but the mode's production
  is unpolarised by construction, so the clauses that switch the boost on for a
  polarised beam or a production brace would find nothing and leave the momenta
  in the lab. `pure_interference` is therefore its own clause of
  `_needs_frame_axis`, beside those two and beside
  `keep_weight_for_polarization_*`.
* `_joint_maxwgt_range` bounds `|w|` and measures `c = <W_full>`;
  `_unweight_range` carries `wsign` onto `full_evt.wgt` and onto every entry of
  `parse_reweight()`. Which of the two output shapes it writes is
  `decay_output` (13.13, 13.17, 13.19).
* `<init>` zeroing plus the `<MGPureInterference>` banner note, and the
  `sum_w` / `sum_w2` / overweight counters with the `z` report in
  `_report_pure_interference`.

**Known boundary:** `fixed_order` is handled (the sign is applied to every
member of the counter-event group) but is **not validated** -- no fixed-order
sample was run through the mode. Note also that `fixed_order` now requires
`spinmode = onshell` or `onshell_v1`: the modes that reshuffle the production
onto sampled virtualities refuse it, because only the born member of a group
would be reshuffled.

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
| `mean\|w\| / sigma_ref` | 0.13007 (0.13011 on an independent 2 000-event run) |
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

**Status: implemented and validated end to end.** `decay_output = auto` -- the
default -- resolves to the fully weighted output of 13.13 in this mode; `set
decay_output unweighted` selects the other representation of the same
estimator, in which the sample carries exactly two weight magnitudes. Why
`auto` points that way here and the other way for an ordinary run is 13.19.

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
it agrees. That is why `auto` resolves to `weighted` in this mode.

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

exactly the fully weighted path of 13.13, only with `W` unrestricted. `auto`
resolves to `unweighted` outside `pure_interference`, i.e. every existing card
is untouched.

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
generation is the bottleneck. That is why `auto` resolves to `unweighted` for
an ordinary run.

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

13.17 and 13.18 are two answers to the same question -- *does MadSpin
unweight?* -- one for the interference mode and one for an ordinary run. One
option asks it:

    set decay_output auto | unweighted | weighted

* `auto` is the default, and resolves to `weighted` when `pure_interference` is
  set and to `unweighted` otherwise (`_decay_output`).
* an explicit value governs in **both** modes: under `pure_interference` it
  chooses that mode's output shape (13.17) rather than stepping aside.

**Why the two directions.** They point opposite ways for a reason. The ordinary
run writes one event per production event either way and its accept/reject is
the *exact* sampler, so unweighting is the safe default and the weighted path
buys CPU at the cost of a weighted file. The interference mode has no exact
sampler to fall back on -- its weights are signed and its cross-section is
zero by construction -- and unweighting on `|W|` there keeps only a few
percent of the production events, for ~6x the variance on exactly the
observables the mode exists to measure (13.17).

Internally the interference mode still reaches the "keep every trial" branch by
its own route -- with a signed `W` and a zeroed `<init>` -- so `_weighted_decay`
returns False under `pure_interference`; only the *source* of that mode's
choice is `decay_output`.

**The two spinmode restrictions compose.** `decay_output = weighted` needs a
density spinmode (there is no `W` otherwise) and so does `pure_interference`,
so the constraints never disagree -- but a card that violates both would get
two refusals in a row, the less useful one first. `_validate_pure_interference`
is therefore called *before* `_validate_weighted_decay`, and both are called
before the `if self._density_spinmode():` branch, so that
`set spinmode none` together with `set pure_interference ...` raises rather
than leaving the mode silently inert. `decay_output` is then silent under
`pure_interference`: the mode announces its own output shape, spinmode
requirement included.

**`auto` announces itself** through `_announce_decay_output`, on the same
`_log_once` convention as `_announce_mode`:

    MadSpin: decay_output = unweighted (auto, ordinary run)
    MadSpin: decay_output = weighted (auto, pure_interference is set)
    MadSpin: decay_output = weighted (set explicitly)

## 14. The overweight safety net: carry the excess instead of clipping it

Every accept/reject in MadSpin -- the joint one, the sequential mass stage,
each angle stage, and the legacy `spinmode = madspin_v1` Fortran loop -- has
the same shape:

    accept the trial with probability  min(1, w / C)

with `C` the bound the maximum-weight probe measured. When a trial comes back
with `w > C` that probability *clips at 1*: the trial is accepted, and the
factor `w/C - 1` by which it should have counted more than an ordinary
accepted trial is thrown away. The counters (`nb_overflow_mass`,
`nb_overflow_<k>`, `nb_overflow_angles`, `report['over_weight']`) have always
seen this happen; what they could not say is *how much* it was worth, and the
sample went out silently biased low in exactly the region where the bound is
too tight -- which for `p p > t t~` under PA is the `t t~` threshold, where the
reshuffling jacobian goes like `1/beta_t`.

**The fix is one multiplication.** The loop stops on the trial it accepts with
probability proportional to `min(1, w/C)`, so writing that event with the
weight `max(1, w/C)` restores the sampled density exactly:

    min(1, x) * max(1, x) = x     identically, for every x > 0

-- the accepted-and-carried density is `q(m) w(m)`, which is the target, and it
no longer depends on `C` at all. Nothing about the accept/reject changes: the
same random numbers are drawn, the same trials are accepted, the same events
are written. Only their weight moves, and only for the events that overflowed.

**Where it is applied.** The factor rides the branching ratio, the hook the
pure-interference mode already uses (`br = self.branching_ratio * pi_factor`),
so one multiplication reaches `full_evt.wgt` *and* every `parse_reweight()`
entry, and the LHEF v3 multiweights stay proportional to the nominal one.
`decay_all_events.decaying_events` uses `change_wgt(factor=...)`, which does
the same thing for the legacy path.

**Composition.** A chain can clip in more than one place, and the factors
*multiply*. `sequential_accept_reject` keeps two of them, each reset where the
quantity it describes is redrawn:

| | reset at | composed by |
|---|---|---|
| `carry_mass` | the top of the mass-set loop | assignment (one mass set per chain) |
| `carry_angles` | the top of the angle loop, beside `w_slots` | `*=` on each accepted slot (or a single assignment under `two_stage`, whose one bound covers the whole angle set) |

so a rejected trial -- which is redrawn -- contributes nothing, and only the
accepted chain's factors survive. The product is handed to `_unweight_range`
through `stats['overweight_factor']`, which is set **only when it is not 1**,
so the caller's `pop(..., 1.0)` returns the literal `1.0` on the common path.

**Exactness when nothing overflowed.** The factor is never built by a division
that could return `0.9999999999`: it is the literal `1.0` unless the code took
an `if w > C` branch, and the multiplication into `br` is skipped entirely in
that case. Measured on `p p > t t~`, 10 000 events, `spinmode = PA`,
`unweighting = sequential`, same seed and same cached bounds, against the
pre-change code: the two decayed LHE files are **byte-identical except for the
weight field of the three events the log reports as carrying**, whose weights
are the old ones times `2.038877`, `1.235717` and `1.010500`. Every momentum,
every other weight, and every `<rwgt>` entry is bit-for-bit the same.

**What it costs, and what it does not fix.** The unit-weight guarantee, for the
handful of events that overflow -- 3 in 10 000 at the shipped bound above,
worth 0.013 % of the sample's normalisation. MadSpin prefers dropping events to
weighting them elsewhere (the BR-equalization path), so this is a deliberate
exception; `EventFile.unweight` has always done the same thing
(`written_weight(max(wgt, max_wgt))`, reported as "truncation"), so a
downstream tool that cannot survive a non-unit weight could not survive an
ordinary MG5 unweighted sample either. It is **not** a substitute for a bound
that is high enough: it makes the mass stage exact, but for the *angle* stages
it only fixes the angular shape. Those stages redraw until they accept and so
divide out their own conditional normalisation, which the tabulated `Z_hat`
models as `E[w | m]` -- an identity that itself assumes the bound dominates.
Carrying restores `p(theta) w` at fixed `m` exactly; the residual `m`-dependence
of `E[min(1, w/C) | m]` against `Z_hat(m)` survives, and only a larger bound
removes it. So the counters still warn.

**The end-of-run measurement.** `_report_overweight` turns the counters into
the number that was missing:

    MadSpin overweight safety net: 3/10000 written events (0.03%) carried a
    non-unit weight because a trial weight exceeded its accept/reject bound
    (largest factor 2.0389). Carrying it added +1.53474 to the summed event
    weight, i.e. +0.0129% of the sample's cross-section (IDWTUP = -4: sigma is
    the mean weight and the event count does not change, so this is the
    relative shift). Clipping it -- what MadSpin did before -- would have
    discarded that silently.

and, when nothing overflowed, says so at INFO rather than staying silent. The
joint accept/reject gained a counter of its own here (`nb_overflow_joint`); it
had none before.

**Why the number is a weight and not a count.** Under `IDWTUP = -4` the
cross-section is the *mean* of the event weights, and carrying changes no event
count, so what the clipping used to discard is

    d(sum w) / sum w        both over the file as clipping would have written it

with `d(sum w) = sum_over (factor - 1) * w_nominal`. For a sample of identical
positive weights that is exactly `sum(factor - 1)/n_written`, so an ordinary
unweighted MadSpin run prints the same number either way. It stops being the
same as soon as the input carries **MC@NLO counter-events**: a negative event
whose trial overflowed makes the cross-section *more* negative, so its excess
subtracts, and a count would have claimed a shift of the wrong sign. Measured on
a 10 000-event `p p > t t~` sample with 25 % of the weights flipped negative and
the mass bound forced 30x low: `d(sum w) = +1.48865e6`, `+1252 %` of the clipped
`sum w`, against `+1250 %` for the count-based number -- close here only because
the sign and the overflow are uncorrelated in that construction, and not equal
in general.

**And why `sum w` needs a guard.** It is zero *by construction* under
`pure_interference`, so it cannot simply be divided by. The test is not a
magnitude cut but the same `z = S / sqrt(sum w^2)` the mode already uses for its
zero-cross-section check: `sum w` is used as a denominator only when it is at
least `_OVERWEIGHT_MIN_Z = 5` of its own Monte Carlo errors from zero. An
unweighted sample of N events has `z = sqrt(N)` and never trips it; a
pure-interference sample has `z = O(1)` and always does, and then the shift is
quoted against `sum |w|` -- which cannot cancel -- with the line saying which
convention it used:

    MadSpin overweight safety net: 6740/8274 written events (81.5%) carried a
    non-unit weight because a trial weight exceeded its accept/reject bound
    (largest factor 29.3165). Carrying it added -389.844 to the summed event
    weight and +52253.7 to the summed |weight|. The summed weight is -134
    against a Monte Carlo error of 290.2 (z = 0.46), i.e. consistent with the
    zero cross-section this sample has by construction, so it is not a usable
    denominator and the shift is quoted against sum|w| = 2.64e+04 instead:
    +198%. Clipping it -- what MadSpin did before -- would have discarded that
    silently.

**Negative weights, twice over.** Two unrelated things make a MadSpin weight
negative, and the carry is blind to both because it is never built from a signed
quantity:

* an **MC@NLO counter-event** in the input. The accept/reject tests a
  matrix-element weight -- a ratio of density contractions times jacobians,
  positive by construction -- and the event's own LHE weight never enters it, so
  the factor is unsigned and the event keeps its sign and only grows in
  magnitude. Measured: over 10 000 events with 25 % counter-events and the mass
  bound forced low, **0 sign flips**, 2496 of the 2500 counter-events carried a
  factor, and **0** carried factors below 1;
* `pure_interference`, where the written weight is signed but the accept/reject
  tests `|W|/M`. The factor is therefore built from `abs(signed)` -- the same
  modulus the test used -- and the sign is applied exactly once, by `pi_factor`,
  inside `br`. Had it been built from the signed `W`, `w > C` would be false for
  every negative trial and **half the sample would still be silently clipped**.
  Measured on a forced-low run: of the 6740 carrying events, **3417 are
  negative** and 3323 positive, matching the 50.3 % negative fraction of the
  file; **0** factors below 1; and, pairing every output event back onto its
  input, **0** `<rwgt>` entries whose ratio to the nominal or whose sign differs.

The closure for the interference case is sharp, because the mean magnitude of
the weight estimates `sigma_ref * BR * E[|W|] / c`, which
`decay_output = weighted` computes exactly:

| | value (pb) |
|---|---|
| forced-low bound, **carried** | 3.29775 +- 0.04628  (+0.2 sd) |
| forced-low bound, **clipped** (the old behaviour) | 1.10689 +- 0.01217  (**-46 sd**) |
| shipped bound, no overflow | 3.30300 +- 0.11514  (+0.2 sd) |
| `decay_output = weighted` (exact reference) | 3.28440 +- 0.04565 |

With the bound 30x too low, clipping under-estimates the size of the
interference by a factor 3 and says so with a *small* error bar; carrying
recovers the exact answer, and recovers the correct error bar with it.

**Validation.** With the mass-stage bound forced 30x low so that 99.9 % of the
10 000 events overflow (mean carried factor 13.5, largest 74.3), the mean of
`m(t) + m(t~)` over the `sqrt(shat) < 400 GeV` slice, where the mass weight
actually varies:

| | mean `m(t)+m(t~)` [GeV] |
|---|---|
| forced-low bound, carried | 345.584 +- 0.095 |
| forced-low bound, clipped (the old behaviour, same events) | 345.926 +- 0.094 |
| shipped bound | 345.539 +- 0.094 |
| `decay_output = weighted` (joint path, fully weighted) | 345.446 +- 0.105 |

The paired carried-minus-clipped difference is `-0.342 +- 0.036` (9.5 sigma):
clipping does move the spectrum. The carried result sits `+0.3` sigma from the
shipped-bound run and `+1.0` sigma from the weighted reference; the clipped one
sits `+2.9` and `+3.4` sigma away. In the `sqrt(shat) < 360 GeV` threshold slice
the carried result is `-0.1` / `+0.0` sigma from the two references and the
clipped one `+3.0` / `+2.6`. A factor 30 in the bound leaves the carried
physics where it was, which is the whole content of `min(1,x) max(1,x) = x`.

## 15. The mass stage's per-event bound

Section 14 made an under-estimated bound harmless. This makes it impossible, at
the mass stage, and cheaper at the same time.

### What the bound is

Under `PA`/`onshell` the mass-set weight is

    w_mass = J(m) . prod_s jac_BW_s(m) . prod_s Zhat_s(m_s)

with `J` the RAMBO production reshuffling jacobian, `jac_BW_s = gap/pi` the
width in `R = atan((m^2-pole^2)/(pole.Gamma))` of slot `s`'s Breit-Wigner
window, and `Zhat_s` the tabulated rate factor. Every factor is non-negative, so
a product of per-factor maxima dominates it. All three maxima are exact and cost
O(n) per production event:

* **`max J` is at the low corner of the windows.** `J` is monotone *decreasing*
  in every new mass at fixed `sqrt(shat)` and fixed configuration -- proved in
  the comment above `_mass_stage_bound`, `d ln J/d mu_k <= (5-2n)/(2 E_k' G
  chi^2) < 0` for `n >= 3`, and the `lambda^(1/2)` ratio for `n = 2`. Checked on
  1.6e6 directional probes over `n = 2..10` with `|p|` and `m` each spanning
  eight decades: zero violations.
* **`jac_BW_s` does not depend on `m_s` at all** -- it is the *window's* width,
  a function of the budget `sqrt(shat) - sum of the masses drawn before it`. It
  is increasing in that budget, which is largest at the same low corner. (The
  earlier design note treated `jac_BW` as a function of the drawn mass and
  proposed maximising `jac_BW . Zhat` jointly; there is nothing to maximise
  jointly.)
* **`max Zhat_s` is a 1-D maximum of `exp` of a quadratic in `ln(m/pole)` over
  the clamped range** -- endpoint or vertex, `_zhat_max`. Deliberately *not*
  `1.1 . <jac_BW . Zhat>`: a sample mean is not a bound, and `Zhat` is allowed
  to have structure.

The window is not a box (`sum m <= sqrt(shat)` couples the slots), but the
coupled region is a subset of the box that contains the low corner whenever it
contains anything, so a monotone function's maximum over it is at that corner
and a scan over the coupled region cannot beat it for `J`.

### Not restricted to `2 -> 2`

`Event.mass_shuffle_jacobian` evaluates RAMBO eqs. (4.3)/(4.9) from the
per-event data `Event.mass_shuffle_frame` returns -- `(E_i, m_i^2, |p_i|^2)` in
the reshuffling CM frame, computed once per event -- so a candidate mass set is
one scalar Newton solve plus O(n) arithmetic, for any `n`. The `2 -> 2` closed
form `lambda^(1/2)/lambda^(1/2)` is the case where the solve is explicit; it is
not the only case that is cheap. Against `_production_jacobian_for` (which
re-parses the event from a string, splits it, boosts it and rebuilds the
momenta) the kernel agrees to **8.9e-16** at `n = 2` and **6.6e-16 / 6.7e-16 /
8.8e-16** at `n = 3, 4, 5` on identical inputs, reproduces the `0` and `-1`
verdicts exactly (7683 such cases), and is 6-7x faster (5.3 vs 33 us at `n = 2`,
7.8 vs 55 us at `n = 5`). Fed the *untruncated* event it differs from the
shipped path by up to 4.4e-5, which is the `%.10e` truncation `str(Event)`
applies -- so the bound builds its frame from the round-tripped event, and sees
exactly what the accept/reject computes.

### What it is worth

`p p > t t~` at 6.5+6.5 TeV, 20 000 events, `spinmode = PA`,
`unweighting = sequential`, `BW_cut = 15`, both tops to `e nu b`. Offline probe,
2500 production events, ~1e6 free mass sets, same slicing as
`doc/madspin_pa_mass_stage/bound_design.md` section 4B:

| `sqrt(shat)` [GeV] | % of sample | `eps_m` global `C` | `eps_m` per-event `C_e` | overflows global | overflows per-event |
|---|---|---|---|---|---|
| 346-350 | 0.40 | 1.96 | 5.50 | 3841 | 0 |
| 350-355 | 1.20 | 1.89 | 3.26 | 1127 | 0 |
| 355-360 | 1.64 | 1.86 | 2.46 | 3 | 0 |
| 360-370 | 3.64 | 1.83 | 2.04 | 0 | 0 |
| 370-380 | 4.64 | 1.82 | 1.75 | 0 | 0 |
| 380-400 | 8.68 | 1.82 | 1.56 | 0 | 0 |
| 400-450 | 19.96 | 1.81 | 1.38 | 0 | 0 |
| 450-500 | 16.24 | 1.81 | 1.27 | 0 | 0 |
| 500-600 | 21.32 | 1.81 | 1.22 | 0 | 0 |
| 600-800 | 15.56 | 1.81 | 1.17 | 0 | 0 |
| > 800 | 6.72 | 1.81 | 1.14 | 0 | 0 |
| **all** | 100 | **1.82** | **1.39** | **4971** | **0** |

Same shape as the earlier study: worse than the global bound in the first few
GeV above threshold, where `J` at the corner is large and the Breit-Wigner
essentially never goes there, and better everywhere else. The largest
`max(w)/C_e` seen anywhere in that scan is **0.9635** -- tight, and never
exceeded.

End to end, against the same sample, the same seed and the *same* `ms_dir`
cache (so the `Zhat` tables and the global bound are identical):

| run | `eps_m` before | `eps_m` after | overweight events before | after |
|---|---|---|---|---|
| `p p > t t~` (2 -> 2) | 1.80 | **1.39** | 20 / 20 000 | **0** |
| `p p > t t~ j` (2 -> 3) | 2.76 | **1.50** | 0 / 8 000 | **0** |
| `p p > t t~`, `BW_cut = 70` | 1.74 | **1.68** | 23 / 20 000 | **0** |

The third row is the case where `Zhat` is *not* smooth: with a 70-width window
the top's `b W` threshold at 85 GeV falls inside it, and the fitted table runs
`Z(70.3) = 0.002`, `Z(173) = 1`, `Z(275.5) = 0.263` -- a factor 500 across the
window, with a 35 % bin-to-fit deviation. There the exact `max Zhat` is far
above the typical `Zhat`, so the bound is barely tighter than the global one.
It is still a *bound*, which the global one was not: 23 overflows became 0.

### The safety case: the bound cancels

The mass stage redraws until it accepts, so the accepted density is
proportional to `q_e(m) . min(1, w/C)`, i.e. to `q_e(m) w(m)` for **any**
`C >= max w`. Changing `C` changes the trial sequence and the cost and nothing
else. Measured, not asserted -- same production sample, same seed, base bound
against per-event bound, accepted virtualities of both parents:

| run | pdg | two-sample chi2 / d.o.f. | KS p |
|---|---|---|---|
| `t t~` | 6 | 19.3 / 24 | 0.22 |
| `t t~` | -6 | 14.0 / 24 | 0.71 |
| `t t~ j` | 6 | 30.1 / 24 | 0.94 |
| `t t~ j` | -6 | 17.1 / 24 | 0.56 |
| `t t~`, `BW_cut = 70` | 6 | 23.5 / 22 | 0.32 |
| `t t~`, `BW_cut = 70` | -6 | 22.3 / 22 | 0.99 |

(The base runs carry a handful of non-unit weights from section 14, worth
0.03 % of the normalisation; the histograms above ignore weights, which is
three orders of magnitude below their resolution.)

### Would a scan do better?

Only through `Zhat`. `J` and every `jac_BW` peak at the same corner, so the
only slack is `prod_s max Zhat_s` against `Zhat` where `w` actually peaks. A
120x120 grid over the true coupled region measures `C_corner/C_scan` at median
**1.26** (min 1.10, max 1.35 for `BW_cut = 15`; max 1.40 for `BW_cut = 70`), so
a scan would buy about 25 % of acceptance for ~14 000 kernel evaluations per
production event against ~2 -- and a grid maximum is not a bound, so it would
need a margin back. Not taken.

### Behaviour change, and the fallbacks

`nb_sigma` and `Nevents_for_max_weight` no longer set the **mass** stage's
bound; they still set every angle stage's. This is logged once per run, not
silent, and it is what makes the mass stage's cost reproducible -- the
probe-based bound was measured scattering +-40 % run to run without converging.
The probe still measures `maxwgts[0]`, and it is still what the mass stage uses
when the per-event bound does not apply:

* the offshell spinmodes (`madspin`/`full`), whose `w_mass` carries
  `Tr(rho_off)/|M_prod|^2_on` -- see the next subsection, which is why it is
  not a gap waiting to be filled;
* a production event with an onshell propagator (status 2), where
  `reshuffle_production` folds in a `reshuffle_decay` jacobian per sub-decay;
* a window that does not fit (`sum` of the minima above `sqrt(shat)`, or a
  budget below a window's own floor);
* a jacobian that is infeasible or not finite at the corner.

The end-of-run report says how many production events took each path -- for
every up-front-mass run, offshell included. (Until this was fixed the counters
and the announcement were gated on `draw_mass`, which is the *PA* half of "this
event has a mass stage": the offshell spinmodes appeared in neither column and
said nothing, while `sequential_with_mass` -- not an up-front scheme at all, and
whose `mass_bound` is dead code -- announced a bound it does not use.)

### The offshell spinmodes: measured, and left alone

`Tr(rho_off)/|M_prod|^2_on` has no cheap maximum, so the exact construction
above does not extend. The obvious partial bound does:

    C_e  =  J(low corner of the windows)_e  x  max_sample[ everything else ]

per-event and provable in the first factor, run-level in the rest; still a
bound, since it is a product of maxima over non-negative factors; and tighter
than today's global bound on every event whose `J_corner` is below the
sample-wide worst. **It is worth 2 %, and the measurement says so before any of
it is built.**

`p p > t t~` at 6.5+6.5 TeV, 10 000 production events, both tops to `e nu b`,
`BW_cut = 15`, `spinmode = madspin`, `unweighting = sequential`. Offline probe
on the *run's own* production events -- 200-400 **free** mass sets each through
`_upfront_production`, 3.90e6 in total, the run's own `ms_dir` cache, with the
weight

    w_mass = R . J . prod_s jac_BW_s . prod_s Zhat_s ,   R = Tr(rho_off)/|M_prod|^2_on

kept factorised. `eps_m = mean_e(C_e/A_e)`, with `A_e` the event's mean weight
over its free draws (infeasible sets counted as zero, as redraw-until-accept
does). The estimator reproduces the runs' own log line: it predicts **3.06**
for the shipped offshell bound and the run reports **3.06**, and **1.400** for
the shipped PA per-event bound against a reported **1.41**.

| the mass stage's bound | `eps_m` | events whose free draws exceed it |
|---|---|---|
| global `maxwgts[0]` -- shipped | **3.06** | 10 / 10 000, worst `w/C` = 1.98 |
| `J_corner . combine(max R.jac_BW.Zhat)` | **3.00** | 0 |
| `J_corner . max_sample(R.jac_BW.Zhat)` | 3.26 | 0 |
| `J_corner . jac_BW_corner . max Zhat . combine(max R)` | 5.00 | 0 |
| the per-event supremum of `w` (not reachable) | 1.42 | 0 |

The second row is the proposal, with its run-level factor built exactly the way
`maxwgts[0]` is (`_combine_maxwgt` over the first 75 probe events). The third is
the same construction with that factor measured honestly, as the sample-wide
maximum instead of a `mean + 4.5 sd` extrapolation of 75 events -- and it is
*worse* than the bound it replaces. A 2 % gain that turns negative when the
estimate it rests on is replaced by the quantity it estimates is not a gain.

**Why, exactly.** Not the offshell ratio. Over 3.90e6 mass sets `R` is
`1.00000 +- 0.0119`, range `[0.733, 1.314]` -- the factor the fallback is
*named* after is the flattest thing in the weight. Two other things do the work:

* **`Zhat` is steep offshell and flat under PA.** Same process, same sample,
  same windows: the fitted table runs `Z(150.6) = 0.522, Z(173) = 1,
  Z(195.3) = 1.699` offshell against `0.912 / 1 / 1.059` under PA. So the
  event-independent `jac_BW_corner . max Zhat` over the two resonances is
  **2.775** offshell and **1.074** under PA. The corner construction multiplies
  `max jac_BW` (at the *low* end of the window) by `max Zhat` (at the *high*
  end); under PA that costs 7 %, offshell it costs 190 %, which is the whole
  difference between `eps_m = 1.40` and `eps_m = 5.00`.
* **`J` and the rest are anti-correlated across events**, `corr = -0.49`: both
  are driven by `sqrt(shat)`, `J_corner` blowing up at threshold (median 1.13,
  mean 1.25, p99 2.9, max 13.2) exactly where the coupled window `sum m <=
  sqrt(shat)` squeezes `jac_BW . Zhat` down. A maximum *of the product* -- which
  is what the probe measures -- sees that; a product of maxima throws it away.
  Over this sample `max_e max_draw(w) / [max_e J_corner . max_e (R.jac_BW.Zhat)]`
  is **0.176**, i.e. fully factorising costs a factor 5.7.

Under PA the same probe, on the same 10 000 events, gives `eps_m` 2.29 for the
global bound (with 49 events over it, worst `w/C` = 3.94) against 1.40 for the
shipped per-event one. That is the shape the offshell case does *not* have.

**And a run-level table for `R`?** Tabulating `max_configurations R` against the
virtualities the way `Zhat` is tabulated, conservative but free at
accept/reject time, is the fuller option. It is not worth building either: `R`
is 1.00 to a percent, a 7x7 table of its maximum in `(ln m_1, ln m_2)` runs
1.02-1.31 against a single global 1.31, so the whole table is worth at most 25 %
*of a factor that is already 1* -- and it would cost the probe an extra record
per free mass set plus a `_UPFRONT_CACHE_FORMAT` bump. The slack offshell is in
`max Zhat`, and `Zhat` is already tabulated and already exact.

So the offshell spinmodes keep `maxwgts[0]`, section 14 keeps carrying the
handful of overflows it leaves (1-2 events in 10 000, largest factor 1.38), and
the fallback is now announced and counted rather than silent.

### Where the overflows it leaves actually are

Re-measured on the current tip, because the campaign logs the question was
first asked of predate sections 15 and 16 and their mass stage ran at 350-860
mass sets per accepted event against 3.6 today.

**First, which stage.** `spinmode = madspin` with two decaying particles
resolves `unweighting = auto` to **joint** (section 12), and the joint test has
no mass stage at all. So the overweights users of the offshell modes actually
see are `nb_overflow_joint`, not `nb_overflow_mass`; the mass stage only enters
when `unweighting = sequential*` is asked for explicitly, or from three
decaying particles up. Both were measured:

| run | overweight events | largest factor |
|---|---|---|
| `p p > t t~`, `unweighting = sequential` | 1 / 50 000 | 1.65 |
| `p p > t t~`, `unweighting = joint` (the default) | 17 / 500 000 | 7.81 |
| `p p > t t~ j`, `unweighting = joint` (the default) | 265 / 300 000 | 48.9 |

`p p > t t~` at 6.5+6.5 TeV, `BW_cut = 15`, both tops to `e nu b`. The 2 -> 3
row is 26x the 2 -> 2 rate and six times the tail, and it is the one that
matters.

**Second, where.** Offline probe on the sequential run's own production events,
400 **free** mass sets each through `_upfront_production`, 2.0e7 draws in total,
the estimator of the table above (it predicts `eps_m = 3.58` against the run's
own reported 3.61):

| | value |
|---|---|
| free draws above the shipped bound | 239 / 2.0e7 |
| production events with at least one | 14 / 50 000 |
| their `sqrt(shat) - 2 m_t`, largest | **0.70 GeV = 0.24 (Gamma_t + Gamma_t~)** |
| the sample's own `sqrt(shat) - 2 m_t`, median | 135 GeV = 45 summed widths |
| `J_corner` on those 14 events | 7.8 to 15.9, against a sample median of 1.12 |

Every one of them, and every one of the 239 draws, sits inside a quarter of a
summed width of the `2 m_t` threshold -- a region holding **0.036 %** of the
sample. The single overweight the run itself realised is there too
(`sqrt(shat) = 346.265`, `S + 0.089 G`). The joint run says the same thing on
an independent 500 000 events: all **17** of its overflows are at
`sqrt(shat) - 2 m_t < 0.24` summed widths, the largest (factor 7.81) at
`S + 0.001 G`.

The mechanism is not the offshell ratio and not `Zhat`: it is `J`, the jacobian
of the reshuffle that moves the production onto the drawn mass set. At
threshold there is no recoil momentum left to absorb the change, and `J`
diverges -- monotonically in the distance to threshold, 15.9 at 0.17 GeV above
it and 7 at 0.84 GeV.

**Third, and against the hypothesis: `p p > t t~ j` is not this.** Its 265
overflows are nowhere near threshold and could not be -- the sample's own
minimum `sqrt(shat)` is 369 GeV, 7.7 summed widths above `2 m_t`. Nor are they
near it in the invariant mass of the `t t~` system, which is the physics
variable rather than the code's mass-draw budget: there they are
*anti*-correlated with the threshold (2.0 % of the overflows inside 10 summed
widths against 7.6 % of the sample; 25 % inside 50 against 49 %). What they
have instead is a resonance at the **low corner of its own Breit-Wigner
window**: 95 % of them have one virtuality in the bottom 2 % of its sampling
variable (4 % by chance), at `(m - pole)/Gamma = -10.7`.

**This is NOT the same divergence of `J`** -- that sentence stood here and was
wrong. Measured on the offending trials, `J` is **1.07** on them (0.98 to 2.47)
against a sample median `J_corner` of 1.22: the reshuffle does nothing. The
whole factor is the matrix element, and the low corner matters because it is
what makes the *production* process's own resonance reachable -- section 17,
which measures it and reruns the `BW_cut` dependence it predicts.

So the threshold picture is **exact for 2 -> 2 and empty for 2 -> 3**, and the
population that dominates the overweight count is the second one -- whose
mechanism is section 17.

### The per-event constructions, re-measured against overflow removal

The table above ranked them on `eps_m`. Asked instead to *remove* the
overflows -- speed explicitly not the objective -- the same 50 000 events and
2.0e7 free draws say:

| the mass stage's bound | `eps_m` | events with a draw over it | worst `w/C` |
|---|---|---|---|
| global `maxwgts[0]` -- shipped | **3.58** | 14 / 50 000 | 1.76 |
| `J_corner . combine(max R.jac_BW.Zhat)` | 3.26 | **4 / 50 000** | 1.04 |
| `J_corner . max_sample(R.jac_BW.Zhat)` | 3.66 | 0 | 0.92 |
| `J_corner . jac_BW_corner . max Zhat . combine(max R)` | 4.82 | 0 | 0.70 |
| `J_corner . jac_BW_corner . max Zhat . max_sample(R)` | 4.94 | 0 | 0.69 |
| the per-event supremum of `w` (not reachable) | 1.48 | 0 | 1.00 |

**The proposal's zero was zero-by-small-numbers.** At five times the statistics
`J_corner . combine(max R.jac_BW.Zhat)` overflows on 4 events in 50 000 -- and
it has to, because `combine` is `mean + nb_sigma . sd` over the first 75 probe
events, an extrapolation of a tail and not a bound. It buys `3.58 -> 3.26` and
removes 10 of the 14, which is a different trade from "removes them all".

The two rows that do reach zero replace that extrapolation by the sample-wide
maximum, and cost `eps_m` 3.66 and 4.94 against 3.58. Their zeros are
**empirical, not provable**: `R = Tr(rho_off)/|M_prod|^2_on` has no analytic
maximum, so offshell no per-event construction can be a theorem the way
`J_corner . jac_BW_corner . max Zhat` is under PA. Making one provable needs a
bound on `R` over the window, i.e. either a matrix-element evaluation per
candidate mass set -- which is the whole cost the mass stage exists to avoid --
or a tabulated `max R(m_1, ..., m_n)` built during the probe. Section 15 priced
the table: `R` is `1.00000 +- 0.0119` over 3.9e6 draws, a 7x7 grid of its
maximum runs 1.02-1.31 against a single global 1.31, so the table is worth at
most 25 % of a factor that is already 1, at the price of an extra probe record
per free mass set and a `_UPFRONT_CACHE_FORMAT` bump.

And none of it would touch the population that dominates the count, because
that one is in the joint accept/reject, which has no mass stage.

### What is reported, and how loudly

Since section 14 an overweight is carried on the event weight rather than
clipped, so none of these is a silent bias any more: what is left to decide is
how loudly to say it. Near the sum-of-poles threshold there is nothing a user
could do about it -- the factorisation evaluates the production with every
resonance ON its pole, and an event that has not got the invariant mass to put
them there is asking the approximation for something it does not have. Away
from it, an overweight still says the bound does not dominate for a reason
nobody has explained, which is worth a warning.

`_near_nwa_threshold` splits them:

    sqrt(shat)  <  sum_r pole_r  +  _NWA_THRESHOLD_WIDTHS * sum_r Gamma_r

over the final-state particles the event actually decays, counted with
multiplicity. `sqrt(shat)` and not the resonance system's mass, because
`sqrt(shat)` is the quantity MadSpin itself spends as the mass-draw budget --
`_upfront_production` and the joint path both start from
`budget = production.sqrts` -- so this is the condition under which its own
windows stop being set by `BW_cut`. The margin is in summed **widths**, the
only scale in the problem that says how far off its pole a resonance may go;
`_NWA_THRESHOLD_WIDTHS = 1.0` says "this event has not got one width of room to
share", and the measurement above puts every observed overflow a factor four
inside it while the region holds 0.31 % of that sample.

The end-of-run line drops from `warning` to `info` only when **every** carried
overweight is inside the region, and it quotes both halves either way. The
total stays the first number on the line, and the arithmetic -- the count, the
largest factor, the cross-section shift -- is untouched: this is a report, not
an accounting change. On the measured runs that means `p p > t t~` goes quiet
and `p p > t t~ j` does not, which is the intended behaviour and not a
side-effect.

---

## 16. The Breit-Wigner truncation of the reported cross-section

Sections 14 and 15 were about the accept/reject bound. This one is about the
number MadSpin prints, and it was wrong in a way that is easy to state:

**`sigma` came out identical for `BW_cut = 15`, `BW_cut = 1`, or any other
value.** MadSpin samples each resonance's virtuality only inside `+- BW_cut`
widths of the pole, then normalises the sample with the *full* width --
`sigma_prod . BR` on the density side, the param-card BR of the chain on the v1
side. It reported the whole rate while producing only the part of the
Breit-Wigner inside the window.

### The size of it, measured

`p p > t t~ j` at 13 TeV, against a truth sample `p p > t t~ j, t > w+ b,
t~ > w- b~` generated by MG5 -- which *rejects* out-of-window points (`myamp.f`,
`gForceBW = 1` sets `cut_bw = .true.`, so the truncation is in the truth's
integrated cross section, the same convention as `BW_cut`):

| | truth (pb) | MadSpin before (pb) | truth/MadSpin | MadSpin after (pb) | truth/MadSpin |
|---|---|---|---|---|---|
| `BW_cut = 15` | 649.35 +- 2.19 | 674.44 | **0.9628** | 646.02 | **1.0052** |
| `BW_cut = 1`  | 334.28 +- 1.13 | 674.44 | **0.4956** | 335.05 | **0.9977** |

The `BW_cut = 1` row is the one that settles it: the old code could not track a
changing window at all, and reported the same 674.44 pb for a sample it had cut
in half.

**Reproduced independently**, on a fresh 30 000-event production sample and two
fresh 30 000-event truth samples at `bwcutoff = 15` and `bwcutoff = 1`
(`spinmode PA`, so the overweight carry of section 14 is out of the way):

| | truth (pb) | MadSpin before (pb) | truth/MadSpin | MadSpin after (pb) | truth/MadSpin |
|---|---|---|---|---|---|
| `BW_cut = 15` | 653.12 +- 1.14 | 674.16 | **0.9688** | 645.741 | **1.0114 +- 0.0024** |
| `BW_cut = 1`  | 335.11 +- 0.68 | 674.16 | **0.4971** | 334.910 | **1.0006 +- 0.0027** |

and the two reported cross-sections stand in the ratio
`645.741 / 334.910 = 1.928103` against the `f(15)^2 / f(1)^2 = 1.928102` the
closed form predicts -- seven digits, because the factor *is* the sampler's
normalisation and not a fit to it. (`onshell` at both cuts reports the same
674.1565 pb, unchanged and equal to the production sample's own, which is what
a mode that draws no virtuality has to do.)

The `BW_cut = 15` residual reads +1.1 % here against +0.5 % in the row above,
and neither is precise: both truth samples are 30 000-event runs quoting 0.17 %,
and 653.12 +- 1.14 against 649.35 +- 2.19 is a 1.5 sigma spread about the
5 000 000-event value RESULTS.md integrates, 651.8 +- 0.22. Folded against that
one and the high-statistics production 674.4 +- 0.21, the sharpest number
available is `651.8 / (674.4 * 0.95785) = ` **+0.90 % +- 0.05 %**, inside the
+0.4 %..+1.0 % band the study bounded. The `BW_cut = 1` residual is not
statistics-limited in the same way, because there the correction is half the
rate and the residual is a twentieth of the error bar.

**The second `bwcutoff` is what the earlier study said it needed, and it lands
where the residual's explanation predicts.** RESULTS.md could only bound the
residual to +0.4 %..+1.0 % because it ran truth at `bwcutoff = 15` only, and it
attributed the residual to the decay numerator `m.Gamma(m)/(m_t.Gamma_t)`
running 0.52 to 1.71 across the `+-22.4 GeV` window while the factor holds it
flat. If that is the cause, then narrowing the window to `+-1.5 GeV` -- where
the numerator barely moves -- has to collapse the residual, and it does:
**+1.1 % at `BW_cut = 15`, +0.06 % at `BW_cut = 1`**, the latter statistically
indistinguishable from zero. So the flat-numerator approximation is not a
uniform offset to be lived with; it is a window-width effect, and it vanishes
exactly where the correction itself is largest (50 % of the rate). That is the
strongest evidence available that the *form* of the factor is right and only its
numerator is approximate.

### The factor

`bw_retained_fraction(M, Gamma, N)` in `MadSpin/decay.py`:

    f = [ atan((M^2 - m_min^2)/(M.Gamma)) + atan((m_max^2 - M^2)/(M.Gamma)) ] / pi
    m_min = max(M - N.Gamma, 0),   m_max = M + N.Gamma

**This is the sampler's own normalisation, not an approximation of it.** Both
generators draw `m^2` flat in `R = atan((m^2 - M^2)/(M.Gamma))`, whose full
range is `pi`: `_mass_window` returns exactly this quantity as its `gap/pi`
jacobian (section 15's `jac_BW`), and `generate_inv_mass_sch` in `src/driver.f`
computes it as `bwdelf`. Integrating the density the code samples from over the
window it samples in is closed-form, so there is no reason to fall back on the
linearised `2/pi . atan(2N)` that the `m^2 - M^2 ~ 2M(m - M)` substitution gives
(0.97879 against 0.97869 here, for a top at `N = 15`).

**What no self-consistent calculation can supply is the numerator.** The rate
integrand is `BW(m^2)` times the decay matrix element and its phase space, and
the retained fraction of the *product* needs the numerator's integral over the
part of the Breit-Wigner that was never sampled -- which a sample that never
leaves the window cannot measure. `m.Gamma(m)/(m_t.Gamma_t)` alone runs 0.52 to
1.71 across a `+-15 Gamma` window for a top, and putting it in moves the `t t~`
pair factor from 0.95785 to 0.96249. That is the residual: **a few tenths of a
percent**, +0.4 % to +1.0 % for a `t t~` pair at `BW_cut = 15`, which is what
the 1.0052 in the table above is made of.

### Which resonances, and which modes

Not the same list on the two sides, and the reason is the *normalisation*, not
the sampling:

* **Density (`madspin`, `full`, `PA`): the top-level virtualities only.**
  `_draw_mass_value` is called once per decaying production particle and never
  for a nested propagator -- for `t > w+ b, w+ > l+ vl` the W's virtuality comes
  from the decay events MG5 generated in `decay_*_*` and is only boosted and
  rotated afterwards (`rotateboost_decay`). Its window is that generation's own
  run_card `bwcutoff`, and the truncation it causes is already inside the
  partial width measured there -- which is the numerator of the branching ratio.
  Correcting it again would double-count it.
* **v1 (`madspin_v1`): every resonance of the chain, nested included.**
  `merge_itree` marks every decay-side s-channel invariant free
  (`keep_inv(i) = .FALSE.`; only the production ones are frozen) and
  `generate_inv_mass_sch` BW-samples each of them inside `+- BW_cut` widths. And
  the v1 branching ratio is `AllMatrixElement.get_br` -- the param card's,
  recursive over the chain, carrying no truncation at all. So all of them have
  to be corrected here. Measured on `t > w+ b, w+ > all all` at `BW_cut = 15`:
  0.95785 for the density path (top^2) against 0.91614 for v1 (top^2 . W^2).
* **`onshell`, `onshell_v1`, `none`: no correction, and that is the point.**
  They sample no virtuality -- the density `onshell` skips the draw
  (`_density_do_reshuffle` is False), `onshell_v1` takes the `mode == 'onshell'`
  branch of `get_onshell_evt_and_wgt`, and the bridge takes MG5's decay event
  whole. Inventing a loss for them would be the same error with the sign
  flipped. Verified: all three report the identical cross-section at
  `BW_cut = 15` and `BW_cut = 1`.
* **2 -> 1 production: no correction.** `sqrt(shat)` fixes the single
  resonance's virtuality and `get_onshell_evt_and_wgt` draws nothing
  (`nb_prod_final > 1`), so the loop that accumulates the factor applies the
  same guard. This is why the `p p > w+` / `p p > w-` acceptance-test cross
  sections are unchanged.

### Where the number lands

Into `branching_ratio`, before anything reads it -- one number that reaches both
the `<init>` block (`scale_init_cross`, or `br_per_id` on the v1 side) and every
event weight. That is what keeps `sigma = mean(w)` true under `IDWTUP = -4`, and
it composes by construction with the later rewrites: the BR equalization of
`_apply_accounting`, `decay_output = weighted`, and section 14's overweight
carry (which still shows up as a `mean(w)` above `XSECUP` by the amount it
reports -- +0.153 % on the 10 000-event `t t~ j` run above).

The multiplicity can differ event to event, so the density side accumulates the
per-event product and writes the **mean** over the file. A per-event factor
would turn an unweighted sample into a weighted one, which is not what an
unweighted MadSpin file is for.

The two-sided asymmetry above is measured, not argued. Same production sample,
same `BW_cut = 15`, `decay t > w+ b, w+ > e+ ve` / `decay t~ > w- b~,
w- > e- ve~` on both paths:

| | reported factor | = |
|---|---|---|
| density (`PA`), nested chain | 0.95785 | top^2 only |
| `madspin_v1`, nested chain | 0.91614 | top^2 . W^2 |
| `madspin_v1`, `t > w+ b` (no nested resonance) | 0.95785 | top^2, i.e. it agrees with the density path when there is nothing nested to disagree about |

### What it moves in the test suite

One number, and it is worth knowing that it is one. A sweep of every
cross-section, branching-ratio and event-weight assertion in `tests/`:

* **`tests/acceptance_tests/test_cmd_madevent.py`, `test_complex_mass_scheme`**
  -- the post-`decay_events` target, `440.779 -> 431.39` (**-2.13 %**, one top
  at `bw_retained_fraction(173.0, 1.491257, 15) = 0.9786983`). Measured:
  production 442.887 +- 4.815, decayed 433.4528 +- 4.712, ratio 0.9786983 to
  seven digits. It did **not** fail -- `4*err1` on a 100-event run is +-4.3 %
  and swallowed the shift -- which is the reason to update it rather than leave
  it: a tolerance wide enough to hide a systematic is not a check on it.
* **The `p p > w+` / `p p > w-` acceptance cross-sections are unmoved**, all
  six of them, because a 2 -> 1 production draws no virtuality. Had they been
  2 -> 2 the W factor 0.97799 would have shifted `100521.5` by 2212 pb against
  an `error` of 800 -- so the 2 -> 1 guard is load-bearing, not decoration.
* **`test_wj_production_with_ms_decay`** (`p p > w+ j`, `spinmode madspin`) is
  the one other affected path, but it asserts event counts only; its cross
  references are already omitted. If they are ever restored they must carry one
  W factor.
* Everything else is an on-shell/`none` mode, a stub constant, or a count.

### What this does not cover

* **The numerator residual**, now measured at both ends: **+1.1 % at
  `BW_cut = 15`, +0.06 % at `BW_cut = 1`** -- a window-width effect, not a
  uniform offset. See the reproduction table above.
* **`BW_cut` does not narrow the nested windows on the density side.** The decay
  events are generated by MG5 with the *run_card*'s `bwcutoff`, so an explicit
  `set BW_cut 1` narrows the top's window and not the W's. That is not a new
  asymmetry -- it is what the code has always sampled -- and the branching ratio
  stays consistent with it, since it is built from the partial width measured in
  that same generation. But it does mean the density-side factor is a function
  of the *top-level* windows only, and a user who wants the nested one narrowed
  has to narrow `bwcutoff` in the production run_card (which `BW_cut` then
  inherits by default, making the two agree again).
* **Mixed-pdg samples equalized by dropping events.** The factor is the mean
  over the input events; the drop correction (`br_correction` in
  `_apply_accounting`) multiplies it afterwards. The two are treated as
  independent, which they are unless the truncation correlates with which events
  the equalization drops -- it cannot, since the drop probability depends only
  on the pdg's total BR.

---

## 17. The joint test's `2 -> 3` overweights: the production's own resonance

Section 15 measured *where* the offshell overweights are and got the `2 -> 2`
case right and the `2 -> 3` case wrong. It closed with

> What they have instead is a resonance at the low corner of its own
> Breit-Wigner window ... **Same divergence of `J`**, reached by the mass draw
> rather than by the energy budget.

The low corner is right. `J` is not. Measured on the offending events, the
production reshuffling jacobian of the 265 over-bound trials is **1.07**
(minimum 0.98, maximum 2.47) against a sample-wide `J_corner` median of 1.22 --
it does nothing at all. The whole factor is in the matrix-element ratio, and
the reason it is there is a mechanism that does not exist in `2 -> 2`.

### The mechanism

The joint weight of the offshell spinmodes is

    w  =  ME . jac_BW . J^P . prod_k J^D_k ,
    ME =  <rho_prod, rho_dec>(offshell) / [ prod_r (M_r Gamma_r)^2 ]
          / [ |M_prod|^2(onshell) . prod_k |M_D_k|^2(pole) ]

with `J^P` the production reshuffle and `J^D_k` the decay reshuffles. Once the
production process has a final state **besides** the resonances -- a jet -- its
matrix element contains a propagator of the resonance *itself*, on the line the
jet is radiated from, at

    (p_r + p_j)^2  =  m_r^2 + 2 p_r.p_j .

With `m_r` **on** its pole that is `>= M_r^2` for any real jet: the singularity
sits exactly on the boundary of phase space and is unreachable. Sampling `m_r`
**below** the pole -- which is the whole point of `BW_cut` -- opens it. The
equation `2 p_r.p_j = M_r^2 - m_r^2` acquires solutions, and on them the
production matrix element is a Breit-Wigner peak of its own, regulated only by
`M_r Gamma_r`. Physically the event is `p p > t t~` with a gluon radiated off an
**on-shell** top, which MadSpin has drawn as `p p > t t~ j` with an off-shell
one. The weight there is correct. It is simply enormous, and no single
run-level bound dominates it.

In `2 -> 2` the region does not exist. The only internal resonance line of
`g g > t t~` is t-channel, `(p_in - p_r)^2 = m_r^2 - 2 p_in.p_r < m_r^2 <=
M_r^2`, i.e. spacelike whatever the drawn virtuality; `q q~ > t t~` has no top
propagator at all. **That asymmetry is the whole of the 26x.**

### The measurement

`p p > t t~ (j)` at 6.5+6.5 TeV, `spinmode madspin`, `unweighting joint`,
`BW_cut = 15`, both tops to `w b`, seed 42. The runs reproduce section 15's
numbers exactly (265/300 000, largest 48.9071 for `t t~ j`; 17/500 000, largest
7.8090 for `t t~`), with every joint trial above `0.3 C` recording its own
factorisation. Medians over the trials that went **over** the bound:

| | `t t~` (2 -> 2), 17 of them | `t t~ j` (2 -> 3), 265 of them |
|---|---|---|
| `J^P`, the production reshuffle | **5.35** (4.3 to 34.8) | **1.07** (0.98 to 2.47) |
| `J^D`, the decay reshuffles | 0.971 | 0.940 |
| `jac_BW` | 0.924 | 0.958 |
| `ME`, in units of the bound | 0.268 (0.24-0.32) | **everything else** |
| `J_corner` of the event | 12.3 (sample 1.12) | 1.19 (sample 1.22) |
| `sqrt(shat) - 2 m_t`, in summed widths | **0.09** (sample 45.6) | 96.5 (sample 124) |
| `M(t t~) - 2 m_t`, in summed widths | 0.09 | **83.2** (sample 51.0) |

Two different populations. The `2 -> 2` one is section 15's threshold story and
is exactly as described there. The `2 -> 3` one is not near threshold in
`sqrt(shat)` (it cannot be -- the sample's own minimum is 7.7 summed widths
above `2 m_t`) and is *anti*-correlated with it in `M(t t~)`, which section 15
already noticed and could not explain.

The explanation is one number. For each over-bound trial, the distance of the
internal propagator from its pole,

    d  =  |(p_r + p_j)^2 - M_r^2| / (M_r Gamma_r) ,

minimised over the two tops, on the reshuffled momenta the trial actually used:

| | min | p10 | median | p90 | max |
|---|---|---|---|---|---|
| the 265 over-bound trials | **0.000** | 0.181 | **0.946** | 2.59 | **5.2** |
| the 431 trials with `0.3 < w/C < 1` | 0.015 | 1.13 | 3.30 | 6.61 | 20.2 |

**All 265 are inside `d = 5.2`; 51 % inside one.** `corr(ln w/C, ln d) =
-0.62`. The largest overweight of the run -- factor 48.9 -- sits at `d = 0.12`,
the second (30.95) at `d = 0.27`, the third (19.46) at `d = 0.94`. Their
`sqrt(shat)` runs 578 to 1650 GeV -- nowhere near threshold, and what the jet
has to do is *land* at `2 p_r.p_j = M_r^2 - m_r^2`, not be soft. Over the whole
population `sqrt(shat)` is mildly *below* the sample's (median 634 against 716);
the next subsection takes that apart.

### Why the probe misses it, quantified

The joint bound is `_combine_maxwgt` over a probe of `Nevents_for_max_weight`
production events x `max_weight_ps_point` decay draws -- here 304 x 492 =
149 568 draws, combined as `1.10 x (mean + 6.18 sd)` of the per-event maxima.
The region it has to find is narrow in *both* directions:

* the internal pole is reachable at some virtuality inside `BW_cut = 15` for
  only **3.5 %** of the production events (it needs `2 p_r.p_j <= M_r^2 -
  m_min^2`, i.e. a jet soft enough or collinear enough relative to the
  resonance);
* on those events, the band `d < 1` occupies a fraction **1.5e-3** of the
  Breit-Wigner sampling variable.

Averaged over all production events that is **5.15e-5 per mass draw** -- so the
probe expects **7.7** such draws and the run expects **208** in its 4.04e6
trials, against 265 over-bound trials observed. The probe does land in the
region; what it cannot do is follow it. Eight draws scattered over eight of the
304 events, at a random `d`, produce per-event maxima that a `mean + 6.18 sd`
extrapolation of a *smooth* distribution flattens away -- and the weight is a
power law in `d`, not a Gaussian tail.

The root's position also matches the corner section 15 saw: the virtuality that
puts the internal propagator on shell is at `(m - pole)/Gamma` between **-14.4
and -8.1** (p10-p90, median -11.8), against the `-10.7` section 15 measured for
the over-bound trials. It is the same thing seen from the other side.

### Is it `shat` near its minimum? Four readings, all refuted

Asked directly, and worth the answer in full because "minimal `shat`" has
several readings that are not equivalent. Populations: the **265 over-bound**
trials, the **431 near-bound** ones (`0.3 < w/C < 1`, the control that is also
heavy), and **40 000 ordinary mass draws** made on the run's own production
events the way `_draw_mass_value` makes them.

The sharp statistic is the AUC of each variable between over-bound and
near-bound -- both heavy, so it isolates what makes a heavy trial *overflow* --
beside the AUC between heavy and ordinary, which is what makes a trial heavy at
all. `0.500` means the variable says nothing.

| variable | over-bound vs near-bound | heavy vs ordinary draw |
|---|---|---|
| `sqrt(shat)` (readings 1 and 3) | 0.523 | 0.388 |
| `sqrt(shat) - sum m_i'` (reading 2) | 0.525 | 0.408 |
| `sum m_i' / sqrt(shat)` ("fill"; 1 = infeasible) | 0.474 | 0.575 |
| `|chi - 1|` (reading 4) | 0.490 | **0.885** |
| `(min m_i' - pole)/Gamma` | 0.452 | **0.013** |
| **`d`** | **0.133** | **0.000** |

**(1) and (3), `sqrt(shat)` near the sample or generation boundary.** Refuted,
and the median alone did not say so -- the over-bound trials are mildly shifted
*down* in `sqrt(shat)` (median 634 GeV against the sample's 716; 43.8 % below
600 GeV against 31.9 %), the opposite of what section 17's "hard jet" reading
suggests. But shifted is not concentrated: their own **minimum is 387.8 GeV**,
19 GeV above the sample's 368.9, only **1.5 %** are below 400 GeV, and the
sample's lowest `sqrt(shat)` decile holds 16.2 % of them -- a 1.6x enrichment.
Against `d`, where 100 % of them are inside 5.2 and an ordinary draw sits at 196.
The mild enrichment is a *consequence* of the mechanism: the resonance needs
`2 p_r.p_j <= M_r^2 - m_min^2`, and a lower `sqrt(shat)` gives a softer jet
more often. Reading 3 is reading 1 shifted by the `ptj = 20` cut and has the
identical AUC; `sqrt(shat) - (2 m_t + 2 ptj)` is 43 GeV at the over-bound 5th
percentile and 248 GeV at their median.

**(2), `sqrt(shat)` minimal *given the drawn masses*.** The reading that would
have been interesting, and it is refuted the hardest -- backwards, in fact:

| | min slack | p1 | p5 | median | max fill |
|---|---|---|---|---|---|
| over-bound | **60.9 GeV** | 70.9 | 101.0 | 302.2 | **0.843** |
| near-bound | 57.5 | 67.5 | 91.4 | 276.6 | 0.852 |
| ordinary draw | **21.4** | 59.8 | 100.4 | 372.2 | **0.943** |

The over-bound trials never come within 61 GeV of the reshuffle boundary and
never fill more than 84 % of `sqrt(shat)`, while ordinary draws reach 21 GeV and
94 %. They are **further** from their own boundary than a random draw is, and
the AUC against the near-bound control is 0.525.

**(4), the RAMBO solve near its edge.** `|chi - 1|` does separate heavy from
ordinary strongly (AUC 0.885, median 0.021 against 0.0017) -- but that is the
same statement as "a mass was drawn low", and it does **not** separate
over-bound from near-bound (0.490). It is what makes `J^P` 1.07 instead of
1.00, and 1.07 is not 48.9.

**Does any of it add anything beyond `d`?** No. Spearman `r(ln w/C, d) =
-0.688` on the 696 recorded trials; the partial correlations given `d` are
`+0.14` for `sqrt(shat)`, `+0.15` for the slack, `-0.08` for `|chi - 1|`. The
raw correlations are `+0.04` to `+0.05` before conditioning, so there is barely
anything for `d` to be a proxy *of*. What residual there is has the **wrong
sign** for the hypothesis: within quartiles of `d`, higher `sqrt(shat)` gives a
higher weight (`r = +0.18` to `+0.22`), not a lower one.

**Are "drawn low" and the mechanism two views of one thing?** No, and this is
the number that settles it. Over ordinary draws, `Spearman(d, min m') = 0.007`
-- independent. Of the draws that put a resonance more than 8 widths below its
pole (2.02 % of draws, which is the corner section 15 identified), only
**0.87 %** land inside `d < 5`. Drawing low is *necessary* -- it is what makes
`2 p_r.p_j = M_r^2 - m_r^2` solvable at all -- and short of sufficient by a
factor 115. The extra condition is on the jet, and it is the whole content of
the mechanism.

### The `BW_cut` prediction, run

If the mechanism is right, closing the window below `~8` widths must close the
population. Same sample, same seed, only `BW_cut` changed:

| `BW_cut` | events that can reach the internal pole | overweights | largest factor | sigma shift | trials/event |
|---|---|---|---|---|---|
| 15 (the default) | 3.49 % | **265** | **48.91** | +0.245 % | 13.47 |
| 10 | 0.91 % | **97** | 46.56 | +0.0943 % | 6.90 |
| 5 | 0.03 % | **13** | **6.95** | +0.0052 % | 4.30 |

The count falls by 20x and the *tail* collapses -- 48.9 to 6.9 -- exactly where
the internal pole leaves the window. The 13 that survive at `BW_cut = 5` are the
ordinary tail of the weight, not this population. (The full reachability scan,
per `(resonance, jet)` pair: 0 % at `BW_cut = 3`, 0.05 % at 5, 0.6 % at 8, 1.8 %
at 10, 6.8 % at 15, 13.8 % at 20, 21.2 % at 25, 30.5 % at 30. It is reachable at
*some* virtuality for 26 % of the pairs, but usually 40+ widths below the pole.)

### What it actually costs

The overweight is **carried**, not clipped (section 14), so none of this is a
bias. What it costs is variance and a shape:

| | `t t~ j`, 300 000 events |
|---|---|
| cross-section the carry restores | **+0.245 %** |
| `N_eff = (sum w)^2 / sum w^2` | 292 719 of 300 000, i.e. **-2.43 %** of the statistics |
| ... of which the single largest event | **-0.77 %** |

and it is not spread evenly. Binned in the *lower* of the two reconstructed top
virtualities:

| `min(m_t, m_t~)` | events | carried | excess | relative |
|---|---|---|---|---|
| 150.6 - 155 | 1283 | 98 | 318.3 | **+24.8 %** |
| 155 - 160 | 2482 | 108 | 333.4 | **+13.4 %** |
| 160 - 165 | 5635 | 46 | 73.6 | +1.3 % |
| 165 - 170 | 25 943 | 13 | 8.4 | +0.03 % |
| 170 - 176 | 262 811 | 0 | 0 | 0 |

So the pre-#375 clipping was leaving the **low tail of the top lineshape 25 %
low** on a `2 -> 3` sample, which is the number that says #375 was worth
building. In `M(t t~)` the effect rises with the jet's hardness, as the
mechanism says: +0.07 % below 400 GeV, +0.31 % at 500-700, **+0.84 % above
1 TeV**.

### The options, with their numbers

**Raise the bound.** Zero overflows needs `C' = 48.9 C`, and the joint
acceptance is `<w>/C` exactly, so the run goes from 13.5 to ~660 trials per
event -- 352 s becomes ~5 h. Dead.

**A per-event joint bound `C_e = J_corner(e) . K`**, the construction section 15
built for the mass stage (`J^P` is monotone decreasing in every drawn mass, so
`J_corner` -- the RAMBO kernel at the window's low corner, one Newton solve --
dominates it), with `K` the run-level maximum of `w / J^P`. Measured on the
runs' own trials:

| | `t t~` (2 -> 2) | `t t~ j` (2 -> 3) |
|---|---|---|
| `K = max(w / J^P)` | **0.8072 C** | **44.3 C** |
| `J_corner`: median / mean / max | 1.121 / 1.226 / 97.1 | 1.225 / - / 62.1 |
| `C_e` median / **mean** | 0.905 C / **0.990 C** | 54.3 C / **62.3 C** |
| trials over the shipped bound | 17 | 265 |
| trials over `C_e` | **0** (worst `w/C_e` = 0.949) | 0 (worst 0.895) |
| trials per accepted event, shipped -> per-event | **3.46 -> 3.43** | 13.5 -> ~840 |

**It fixes the case that does not matter and destroys the one that does.** In
`2 -> 2` the tail *is* `J^P`, so dividing it out shrinks the run-level factor to
0.81 C: the per-event bound is **free** -- 3.43 trials per accepted event
against the 3.46 the shipped bound predicts and the 3.44 the run reports -- and
all 17 overflows are gone, with the worst weight it ever sees at 0.949 of its
own bound. That row is measured over **all** 1 702 395 trials of the run, not a
tail sample. In `2 -> 3` the tail is in `ME` and `J^P` is 1, so dividing by it
shrinks nothing: `K` stays at the top of the tail and *every* event's bound is
multiplied by it, a 62x slowdown. (There `K` is measured over the top 0.02 % of
trials, so it is a lower bound on the true maximum -- which makes the verdict
stronger, not weaker.)

**A better probe** -- deliberately sampling the low-virtuality corner -- would
find the region, and then hand back a bound 49x too large. The population is not
a sampling gap that a bound can absorb; it is a genuine narrow peak of the
production matrix element inside the sampled region.

**Lower `BW_cut`.** Measured above, and it is the only lever that removes the
*cause*. It is also a physics choice (it truncates the Breit-Wigner; section 16
is about exactly that), so it belongs to the user and not to a default.

### What was done

Nothing to the bound. The report was taught the second region, the same way the
tip of section 15 taught it the first:

`_near_production_resonance(full_evt, production, evt_decayfile)` asks, on the
event that carried an overweight and only on that event, whether some decayed
resonance `r` and some other **production-level** final state `k` satisfy

    |m^2(r + k) - M_r^2|  <=  _PRODUCTION_RESONANCE_WIDTHS . M_r Gamma_r

with `_PRODUCTION_RESONANCE_WIDTHS = 10.0` -- twice the measured envelope of
the whole population (max `d = 5.2`), so it is not fitted to one sample, and
still specific: the fraction of joint trials that land inside it is 9.2e-4
(2.9e-4 at a margin of 5, 5.3e-5 at 1) against an overweight rate of 6.6e-5, so
it cannot silence an overweight by coincidence. It
reads the *reshuffled* momenta, i.e. the virtuality the event actually carries,
and it is `O(n^2)` four-vector arithmetic on an event that is already built.
The first `len(production)` entries of `full_evt` are the production block
(`add_decay_to_particle` appends decay products after them), so decay products
cannot pair up with their own parent. It returns False rather than raising, for
the same reason `_near_nwa_threshold` does: it decides how loudly a diagnostic
prints.

The end-of-run overweight line now splits three ways -- threshold, production
resonance, neither, in that order of precedence -- quotes each, and drops from
`warning` to `info` only when the third is empty.

Verified rather than asserted, on the same sample, same seed, same `nb_core`:

* `p p > t t~ j` tags **265 of 265** and the run's log goes from one WARNING to
  none. At a margin of 5 it tagged 264 and kept the warning for the one at
  `d = 5.2` -- which is the same population -- which is why the margin is 10.
* the 300 000 written events are **byte-identical** to the base tip's (SHA-256
  over the `<event>` blocks), and every number on the line is unchanged:
  265/300 000, largest 48.9071, `+473984`, `+0.245 %`, 13.47 trials per event.
  This is a report and nothing else.
* `p p > t t~` is untouched: `2 -> 2` returns False at the `len(finals) < 3`
  line before it looks at anything, so its 17 overweights keep going through
  the threshold branch.
* `tests/test_manager.py test_madspin -t0` is green at **473** tests (462 on
  the base, 11 new: five on the predicate -- the `M Gamma`-in-`s` window, the
  `2 -> 2` exclusion, an undecayed particle, the production-block slice, a
  zero-width particle, and that it never raises -- and six on the split,
  including that it moves no arithmetic).

### What this does not cover

* **`R = Tr(rho_off)/|M_prod|^2_on` in the sequential mass stage.** Section 15
  measured it at `1.00000 +- 0.0119`, range `[0.733, 1.314]`, and called it
  "the flattest thing in the weight". That was measured on `p p > t t~`, and
  `R` is precisely the ratio that carries this resonance -- so on a `2 -> 3`
  sample it must have the same heavy tail, and the mass-stage constructions
  section 15 ranked on it would have to be re-ranked there. Not measured here:
  `auto` sends two decaying particles to `joint`, so the mass stage is only
  reached on a `2 -> 3` sample by an explicit `set unweighting sequential` or
  from three decaying particles up.
* **The residual `sqrt(shat)` dependence.** After conditioning on `d` there is
  a small positive one (partial `r = +0.14`, `+0.18` to `+0.22` within `d`
  quartiles): at fixed distance from the internal pole, a harder event gives a
  bigger weight. Plausibly the resonant diagram's share of the matrix element
  growing against the non-resonant background, but not measured -- it would
  need the diagram-level decomposition, which the density path does not expose.
  It is the wrong sign for a threshold reading either way.
* **Higher multiplicity.** The mechanism gets *more* available with every extra
  production-level parton (more `(r, k)` pairs, and pairs of partons as well as
  single ones). Only `2 -> 3` was measured. The predicate tests pairs only.
* **The `2 -> 2` per-event joint bound**, which the table above says is free
  (3.43 trials/event against 3.46) and removes all 17 of its overflows, on the
  full 1.7e6-trial sample. It is a change to a shipped bound, so it is left as
  a recommendation and not taken here. Two things it would need before it
  could be: `K` has to come from the probe rather than from the run (the probe
  would have to record `w / J^P` per draw, which is one extra float and the
  jacobian it already computes), and it has to keep the fallbacks
  `_mass_stage_bound` already has -- an onshell propagator in the production
  event, an empty window, a jacobian that is infeasible at the corner. Note it
  is only ever *tighter* than a run-level `K . max_e J_corner`, never a
  different distribution: redraw-until-accept makes the accepted density
  independent of `C` for any `C >= max w`, the same argument as #377.

## 18. The joint max-weight scan reuses a stale `rho_prod` (assessment)

Found in passing during milestone 3 of the axial work (`2ba47b8d8`), where it
masked the `density_debug` closure signal; that commit made `density_debug`
skip the affected trials and left the scan alone, because touching it would
have broken the milestone's bit-identity proof. This section is the assessment
that was asked for afterwards. **Nothing shipped is changed by it.** Every
number below was taken with four environment-gated diagnostics whose default
is off; the patch is reproduced at the end.

### 18.1 The defect, confirmed

The joint bound is measured by `_joint_maxwgt_range`
(`MadSpin/interface_madspin.py:6735`). For each probe production event it takes
`nb_ps_point` decay draws and keeps the largest weight. Off shell
(`spinmode madspin`/`full`) each draw is a genuinely different phase-space
point: line `6808` builds a fresh **on-shell** copy of the production event per
draw, and `calculate_matrix_element_from_density` then reshuffles that copy onto
*that draw's* virtualities --

    :10627   if not density_pole_approximation or \
    :10628       (not prod_static or prod_static.get('decays_key') != decays_key):
    ...
    :10677           jac *= production.reshuffle_production()

-- the first disjunct being unconditionally true off shell. The production
density is taken **after** that reshuffle:

    :10735   if prod_density_cached is None:
    :10736       density_prod = self.get_density(production, ...)

and the scan hands the same `prod_density_cached` back in on every draw after
the first:

    :6809   if density_matrix_prod is None:
    :6810       _, wgt, density_matrix_prod = self.get_onshell_evt_and_wgt(
    :6811           prod_draw, decays, decay_dict, build_event=False)
    :6812   else:
    :6813       wgt = self.get_onshell_evt_and_wgt(
    :6814           prod_draw, decays, decay_dict, density_matrix_prod,
    :6815           build_event=False)[1]

So draws `j >= 1` contract `rho_prod` evaluated at **draw 0's** virtualities
against a `rho_dec` and a `jac_reshuffle` taken at draw `j`'s. The event loop
carries exactly the guard the scan is missing:

    :4747   if prod_density_cached is None or not density_pole_approximation:

-- `or not density_pole_approximation` forces a fresh `rho_prod` on every
off-shell trial, and the cache is used only under `PA`/`onshell`, where the
production event is never reshuffled in place and the cached matrix is the
right one. The existing comment at `:10750` already states the diagnosis; this
section confirms it and prices it.

Only `rho_prod` is stale. `MEdenom_prod` (`:10669`) is taken **before** the
reshuffle, on the per-draw on-shell copy, so it is the same number on every
draw; `prod_static`, `me_wgt` and the jacobians are all rebuilt per draw
because `prod_draw` is a fresh `Event`. The defect is one object.

Confirmed empirically as well as by reading: running the same probe twice with
the same seed, once as shipped and once with the density re-derived per draw,
the two agree **exactly** on all 304 `j = 0` trials (0 disagreements) and
differ on the 149 264 later ones -- which is what "the draws are identical, only
`rho_prod` differs" predicts. The per-draw resonance distance `d` of section
17 is bit-identical between the two runs, confirming the draws are aligned.

### 18.2 Which modes and schemes

| | affected? | why |
|---|---|---|
| `joint` + `madspin`/`full` | **yes** | `_joint_maxwgt_range`, above |
| `joint` + `PA` | no | the reshuffle is on a separate copy (`:6817-6823`); `rho_prod` is evaluated at `base_event`'s unchanged on-shell momenta, so the cache is *correct* -- which is what the event loop's `or not density_pole_approximation` says |
| `joint` + `onshell` | no | not the density path at all |
| `sequential`, `sequential_global_retry`, `two_stage`, `sequential_with_mass` | no | their probe (`_scan_maxwgt_range`, `:6645`) calls `sequential_accept_reject` once per draw, and that derives `rho` per chain in `_upfront_production` (`:8990`) whenever `offshell`; the cached `_ms_density_prod` at `:9702-9714` is guarded by `if not offshell` |

Checked, not only reasoned: the same `p p > t t~ j` probe under
`set spinmode PA` returns the **identical** bound
`3.4098530956016246e-09` with the diagnostic on and off, to the last digit.

The *joint* scheme has no mass stage, so there is no second place for this to
hide. But every mode that **forces** `joint` inherits it whenever the spinmode
is off shell: `fixed_order`, `@` decay groups, `pure_interference`,
`decay_output = weighted`, and -- new on this branch -- `consider_axial`
(`:3690-3706`), which *additionally requires* `madspin`/`full`. **Every
`consider_axial` run therefore takes the defective path by construction.**

### 18.3 What it does to the bound

`p p > t t~ (j)` at 6.5+6.5 TeV, `spinmode madspin`, `unweighting joint`,
`BW_cut = 15`, both tops to `w b`, seed 42 -- section 17's samples, unchanged.
The probe is the default one (`Nevents_for_max_weight = 300 -> 304`,
`max_weight_ps_point -> 492`, `nb_sigma = 6.77`), i.e. ~149 500 trials.
Both columns are MadSpin's own printed bound on identical draws.

| | shipped (stale `rho`) | correct (per-draw `rho`) | ratio |
|---|---|---|---|
| `p p > t t~ j` (2 -> 3) | `2.925218e-09` | `1.0178792e-07` | **x 34.80** |
| `p p > t t~` (2 -> 2) | `7.442786e-10` | `7.525693e-10` | x 1.0111 |

**The direction is too LOW, and by a factor 35 on the `2 -> 3` process.** Not
"wasteful"; the dangerous direction.

Seed 42 is the *mildest* of the four seeds tried, because the shipped bound is
not a stable quantity -- it depends on which virtuality each probe event
happened to draw **first**, and the corrected one does not:

| seed | shipped | correct | ratio |
|---|---|---|---|
| 42 | `2.925218e-09` | `1.0178792e-07` | 34.80 |
| 43 | `1.176710e-09` | `9.326181e-08` | 79.26 |
| 44 | `1.556063e-09` | `8.078270e-08` | 51.91 |
| 45 | `1.402010e-09` | `9.210736e-08` | 65.70 |
| | sd/mean **44.7 %**, max/min 2.49 | sd/mean **9.4 %**, max/min 1.26 | median **58.8** |

That second column is what a max-weight scan is supposed to look like: 304
production events, 492 draws each, and a 9 % seed-to-seed spread on the answer.
The first is a 2.5x range on the same probe, because each of its 304 per-event
maxima is conditioned on the virtuality that event's **first** draw happened to
pick: 492 draws that explore only one `rho_prod`. **The factor is 35 to 79
depending on the seed, not 35.** Section 17's un-costed "raising it needs 49x"
sits in the middle of that range.

Per trial the mismatch is small in the bulk and unbounded in the tail
(`|1 - w_stale/w_fresh|`, off-shell draws only):

| | median | p90 | p99 | max | `> 1 %` | `> 10 %` |
|---|---|---|---|---|---|---|
| `t t~ j` | 0.72 % | 4.8 % | 27 % | **6.10** | 41.5 % | 4.3 % |
| `t t~` | 0.33 % | 2.1 % | 7.2 % | 0.456 | 23.9 % | 0.35 % |

(the milestone-3 commit quotes "median 1.5 %, up to 42 %" for `u d~ > w+ h`;
the numbers are process dependent, and the shape is the same.)

### 18.4 It is section 17's resonance, not a second candidate

The brief asked whether a badly placed bound is an *independent* candidate for
the 265/300 000 overweights that section 17 attributed to the production matrix
element's own resonance propagator. It is not independent. It is the same
mechanism seen from the other side, and this is what closes the hypothesis.

Recording section 17's distance `d = min_{r,k} |(p_r+p_k)^2 - M_r^2| /
(M_r Gamma_r)` on every probe draw (on the *reshuffled* momenta, the ones
`rho_prod` is supposed to be evaluated at), the 10 probe events that set the
correct bound take their maximum **on the resonance**, and the shipped scan
takes its maximum on the same events far away from it:

| evt | `w` correct | `w` shipped | ratio | `d` at the correct max | `d` at the shipped max |
|---|---|---|---|---|---|
| 290 | 4.460e-08 | 3.362e-10 | 132.6 | 0.216 | 51.1 |
| 195 | 2.867e-08 | 3.058e-10 | 93.7 | 0.468 | 27.5 |
| 61 | 2.454e-08 | 1.663e-09 | 14.8 | 0.509 | 32.9 |
| 188 | 2.151e-08 | 9.453e-10 | 22.8 | 0.188 | 42.7 |
| 258 | 1.533e-08 | 3.357e-10 | 45.7 | 0.234 | 23.6 |
| 177 | 1.220e-08 | 4.390e-10 | 27.8 | 0.837 | 21.3 |
| 296 | 8.165e-09 | 4.074e-10 | 20.0 | 0.186 | 25.9 |
| 216 | 3.043e-09 | 3.162e-10 | 9.6 | 1.029 | 54.7 |
| 281 | 2.073e-09 | 3.326e-10 | 6.2 | 2.340 | 18.0 |
| 261 | 1.465e-09 | 3.118e-10 | 4.7 | 2.328 | 23.1 |

median `d` at the correct per-event max, over these ten: **0.49**; at the
shipped one: **26.7**. (Over all 304 probe events the two medians are 231 and
224 -- the shift is entirely in the tail that sets the bound.) Binned over all 149 264 off-shell draws:

| `d` | trials | median &#124;1 - stale/fresh&#124; | max |
|---|---|---|---|
| `[0, 1)` | 15 | **0.980** | 0.997 |
| `[1, 3)` | 18 | 0.929 | 0.990 |
| `[3, 10)` | 120 | 0.539 | 2.23 |
| `[10, 30)` | 6 209 | 0.023 | 5.29 |
| `[30, inf)` | 142 902 | 0.007 | 6.10 |

On the pole the stale weight is **fifty times too small**. `rho_prod` is where
the internal propagator lives -- `MEdenom_prod` is the *on-shell* production
matrix element and carries none of it -- so freezing `rho_prod` at draw 0's
(generic) virtuality is precisely the thing that blinds the scan to the region.
The scan visits the region once per probe event instead of once per draw, a
factor ~492 fewer looks, and then `_combine_maxwgt`'s
`mean + nb_sigma*sd` has nothing in its top-50 to raise the bound with. That the effect is 1.01x in
`2 -> 2`, where section 17 showed the region does not exist, and 34.8x in
`2 -> 3`, where it does, is the same asymmetry that section 17 called "the whole
of the 26x".

Section 17 wrote "no bound is changed. Raising it needs 49x and costs 49x". The
49x it could not justify is now *measured*, from the probe, on the same sample:
34.8x at seed 42 and 35 to 79x over four seeds, with 49x in the middle of that
range. 18.8 shows what raising it does, and 18.8's second half separates this
from section 17's own diagnosis rather than assuming they are the same.

### 18.5 What correcting it does to the overweight population

The 265 carried events of the shipped `t t~ j` run are exactly the trials with
`w > C`, and each is written with `carry = w/C`. So the shipped run's own carry
histogram is a full-statistics measurement of the trial tail over its 4 041 115
trials:

    carry > 2      151      carry > 10      18
    carry > 5       52      carry > 20       2
                            carry > 34.80    1     (the maximum, 48.9071)

Under a bound `x` times larger the trial count scales by `x` (the acceptance is
`w/C`) and the overweights are the trials above the new bound, so

    E[n_overweight(x . C)]  =  x . #{carry > x}  =  34.80 x 1  =  35 ,

and the largest known factor becomes `48.9071 / 34.80 = 1.41`. That predicts
**~35 events with factors near 1.4**, but the `35` rests on a single trial in
the tail (`N = 1`, so a 95 % Poisson interval of `[0.9, 194]`) and is worth very
little. **The direct measurement in 18.8 is 8 events, largest 1.67** -- inside
that interval, and the number to quote. The `1.41` for the severity holds up
(measured: 1.67).

Note what this does *not* say: the shipped output is not biased. `#375`'s carry
is exactly unbiased (accept with `min(1, w/C)`, write `max(1, w/C)`; the
expected written weight is `w/C` in both regimes), so a bound below the true
maximum turns the joint accept/reject into a *partially unweighted* sample
rather than a wrong one. What the too-low bound costs is 265 non-unit weights,
`+0.245 %` of the cross-section arriving through the carry, and 2.4 % of the
effective statistics (section 17). What it buys is a factor 34.5 in speed. It
would have been an outright bias before `#375`, where the excess was clipped --
section 17 measured that as leaving the low tail of the top lineshape 25 % low.

One consequence is *not* covered by the carry. `pure_interference` and
`decay_output = weighted` force `joint`, and the same scan trials measure
`c = <W>` and `<|W|>`, which normalise the **written weights** rather than a
bound. On `p p > t t~ j` with `decay_output = weighted`:

| | `c = <W>` | quoted error |
|---|---|---|
| shipped | `2.212664e-10` | 0.08 % |
| correct | `2.196368e-10` | 0.29 % |

a `+0.74 %` shift, and an error understated by 3.6x -- because suppressing the
tail also suppresses the spread the error is taken from. `c` divides the written
weight (`pi_factor = signed / pure_interference_c`), so `mean(w)` comes out
`0.74 %` low against `sigma_ref * BR`; small, but it lands on the **output**,
not on a bound, and it is larger than the precision the run claims for it.
`_weighted_decay_note`'s `mean(w)` test and the interference mode's `z` test are
the monitors that would see it.

The interference mode's *other* variant is fine. `pure_interference +
decay_output = unweighted` does accept/reject and its bound is live, but that
path carries its overflow too (`carry = abs(signed)/maxwgt`), and it normalises
the written magnitude with the run's **own** `<|W|> = (N_file/N_drawn) . M`,
which the code's own comment says "makes the accept/reject bound cancel exactly
rather than on average". Between the carry and that rescaling a low bound should
come out in the wash. The `logger.critical` there ("an under-estimated one
biases the sample") reads as predating `#375`'s carry; not chased down here, and
worth a separate look.

### 18.6 Options, priced

1. **Re-derive `rho_prod` per draw** (drop the cache off shell; one extra
   `or not density_pole_approximation`, the event loop's own condition).
   * scan cost: `+17 %` wall (`6.20 s -> 7.27 s` for 149 568 trials on 8 cores,
     `t t~ j`). A scan trial becomes as expensive as an event-loop trial, which
     is what it is supposed to be simulating.
   * run cost: the bound rises, so the acceptance falls by the same factor.
     Measured `13.47 -> 471` trials per event on `p p > t t~ j` (x 35.0);
     x 1.01 on `p p > t t~`. On the 300 000-event sample that is a measured
     **350 s -> 6.9 h**, on 8 cores.
   * **and it needs a decay-pool fix first.** 35x less acceptance is 35x more
     pool regeneration -- 2 refills became 74 -- and that raced two workers into
     the same `decay_6_0` build and killed one of them (18.8.1). Shipping (1)
     without fixing that ships a run that loses a worker.
   * **it changes the accepted sample.** The per-event mass-bound argument of
     `#377` -- redraw-until-accept makes the accepted density independent of
     `C` for any `C >= max w`, so a bound that dominates cancels -- applies to a
     bound that is *raised past* the maximum. This one is currently *below* it,
     so the shipped sample is the partially-weighted one described above and
     the corrected sample is a different (properly unweighted) one. This is the
     "smaller bound changes what gets written" case, seen from the other end.
2. **Correct the scan, then deliberately lower the bound.** Measure the true
   maximum and divide by a declared factor, keeping the carry to stay unbiased.
   This is the honest form of what the code does today by accident, and it makes
   the speed/weight trade an option rather than a bug. Cost: the `+17 %` scan
   only; the run is unchanged at whatever factor reproduces today's bound. It
   also removes the 2.5x seed-to-seed swing of 18.3 -- today's bound is not
   reproducible run to run, a declared fraction of the correct one is.
3. **Correct the scan only where the measurement is used as a measurement.**
   `c` and `<|W|>` (18.5) are not bounds and have no carry protecting them.
   Deriving `rho_prod` per draw only when `probe_c` is set costs the `+17 %` on
   `pure_interference` / `decay_output = weighted` runs and nothing elsewhere,
   and fixes the only place the defect reaches the output.
4. **Leave it, document it.** Nothing is biased today. The cost is that the
   overweight report (`_near_production_resonance`, `9c729f3f3`) tags all 265 as
   the production resonance and drops the line to `INFO`, which is a true
   statement about the *mechanism* and now also, unintentionally, hides a scan
   that cannot see it.

Recommendation: **(3) then (2)**, and not (1) on its own.

(3) is the only part of this that reaches an output and it is nearly free.

(1) alone buys, measured (18.8), `263 -> 8` carried weights and `48.9 -> 1.67`
on the worst one, and costs `350 s -> 6.9 h` plus a decay-pool race that has to
be fixed first. That is a bad trade at face value: `#375`'s carry already makes
the shipped sample unbiased, so what (1) buys is tidiness (a genuinely
unit-weight file) rather than correctness. It should be the user's explicit
choice, which is what (2) turns it into -- and (2) additionally removes the
2.5x seed-to-seed swing in today's bound, which is a reproducibility problem
independent of where the bound sits.

Whatever is chosen, the scan should stop **silently** measuring a different
quantity from the one the event loop computes. Even under (4) the constant in
(2) should be written down, because "the bound is 35 to 79x below the maximum,
deliberately, and the carry absorbs it" is a defensible design and "the scan
freezes `rho_prod`" is not.

### 18.7 Not settled

* How the factor scales. Two processes and four seeds (18.3), so the `35 to 79`
  range is measured but the *process* dependence is two points. The mechanism
  says it should grow with production multiplicity (section 17's closing note),
  so `>= 2 -> 4` is likely worse; not measured.
* `p p > t t~ j` has a **pre-existing** `density_debug` closure failure of
  1.4 % on the *first, non-stale* trial (`full = 0.0016651567985569238,
  density = 0.0016421164116264863, ratio = 1.0140309096037938`). Reproduced bit
  for bit on the untouched base tip, so it is not this defect and not the
  instrumentation; it is presumably the same off-shell-propagator-numerator
  story that milestone 3 solved for massive *vectors* (`{A}`), unsolved for the
  off-shell *fermion* `(p-slash + m)`. It means `density_debug` cannot be used
  as an independent check on this process at the default tolerance, which is why
  18.1's confirmation is by weight comparison instead.
* The `c` / `<|W|>` shift was measured on one process and only through the log;
  its effect on a written `pure_interference` sample was not measured.
* The corrected-bound run (18.8) lost a worker to a build race, so it is
  296 302 events rather than 300 000. The comparison is made shard by shard on
  the matched production events, so the counts are directly comparable, but the
  8 survivors are a Poisson-8 measurement and the `x 32.9` carries that.
* The decay-pool build race of 18.8.1 was observed once and not diagnosed.

### 18.8 The direct overweight measurement

The prediction of 18.5 was `~35`, from a single trial in the tail; it is worth
what a Poisson `N = 1` is worth. So the corrected bound was run: the identical
`p p > t t~ j` card and seed, `MS_SCAN_FRESH_RHO=1`, everything downstream
untouched -- 6.9 hours against 350 s, which is the price of 18.6 (1) made
concrete.

One of the eight workers died 33 802 events in (18.8.1), so the comparison below
is made **shard by shard on the same production events**, `A[:m]` against
`B[:m]` for `m = min(len_A, len_B)` -- the forked workers slice the input file
identically in both runs, so shard `k` is the same block of production events in
each. 296 302 events matched:

| shard | matched events | shipped: overweights (largest) | corrected: overweights (largest) |
|---|---|---|---|
| 0 | 37 500 | 32 (12.48) | 0 (--) |
| 1 | 33 802 | 33 (11.11) | 2 (1.146) |
| 2 | 37 500 | 38 (**48.91**) | 1 (1.068) |
| 3 | 37 500 | 32 (15.51) | 0 (--) |
| 4 | 37 500 | 33 (15.67) | 1 (1.304) |
| 5 | 37 500 | 26 (19.46) | 0 (--) |
| 6 | 37 500 | 31 (7.11) | 1 (1.177) |
| 7 | 37 500 | 38 (30.95) | 3 (**1.668**) |
| **total** | **296 302** | **263** (0.0888 %), largest **48.9071** | **8** (0.0027 %), largest **1.6679** |

**x 32.9 fewer, x 29.3 less severe.** The carry contribution to the summed
weight goes from `+0.245 %` to `+0.0007 %`. The written sample stops being
partially weighted: 8 non-unit weights in 296 302, none above 1.67.

#### Is 48.9 inside the 35-to-79 range a clue or a coincidence?

Neither, and this is the part worth being careful about. The two numbers are
**not the same quantity**, but they share a denominator, so of course they are
the same order:

    48.9   =  (largest trial weight the run realised) / (shipped bound)
    34.8   =  (correct bound)                         / (shipped bound)      [seed 42]

Dividing one by the other removes the shipped bound and leaves the only number
that means anything on its own:

    48.9071 / 34.797  =  1.406  =  (largest realised weight) / (correct bound)

and run B measured its own version of that directly: **1.6679**. So a 300 000-
event run's largest trial weight sits about `1.4` to `1.7` above a bound
measured from 304 probe events. That is the ordinary tail under-coverage of any
max-weight scan, it is what the 8 survivors are, and it is the *entire* residual
once the stale `rho` is gone. The `48.9` decomposes as `34.8 x 1.41`: the scan
defect supplies the first factor and normal probe statistics the second.

The seed-to-seed spread of 18.3 (`35` to `79`) is a spread in the **shipped**
bound, not in the correct one (44.7 % against 9.4 %). `48.9` landing inside it is
therefore an arithmetic consequence of both being divided by that same unstable
number, not evidence of a relationship. Read the stable quantity instead: the
correct bound is `9.20e-08 +- 9 %` over four seeds, and the largest weight a
300 k run realises is `1.4-1.7` times it.

#### Are the two explanations independent? No -- they are one.

The brief asked whether a badly placed bound is a *second*, independent
candidate for the 265. It is not. Three separate pieces of evidence say the
scan defect and section 17's production resonance are the same thing seen from
opposite ends:

1. **Where the correct bound comes from** (18.4). The probe draws that set it
   sit at `d = 0.49` (median over the top ten events) -- on the production
   resonance, the region section 17 named. The shipped scan's own per-event
   maxima on those same events sit at `d = 26.7`. At `d < 1` the stale weight
   is **50x too small**. `rho_prod` is where the internal propagator lives;
   freezing it is precisely what hides the region.
2. **The `2 -> 2` control.** Section 17 proved the region cannot exist there
   (the only internal resonance line is spacelike). The stale-`rho` effect
   there is **x 1.011**, against x 34.8 in `2 -> 3`. Same asymmetry, same cause.
3. **This run.** Correcting the scan removes **255 of the 263**. If the
   population had a second, independent cause, it would not.

So section 17's physics is right and unchanged -- the overweights *are* the
production matrix element's own resonance propagator opened by a low
virtuality. What this section adds is the reason the **bound never covered
it**: the scan cannot see that region, because `rho_prod` is exactly the object
it freezes. Section 17 recorded "raising it needs 49x and costs 49x" without
being able to say where 49x came from. It comes from here.

#### 18.8.1 A cost of the corrected bound that was not anticipated

Worker 1 of 8 did not finish. It died at 33 802/37 500 events with

    ar: ../../lib/libmodel.a: Inappropriate file type or format
    FileNotFoundError: .../decay_6_0/Source/param_card.inc.tmp

-- two workers rebuilding the same `decay_6_0` directory at once. The shipped
run does **2** decay-pool regenerations; the corrected-bound run did **74**,
because a bound 35x larger is an acceptance 35x smaller and the pools drain 35x
faster. The refill path has a known race history (`d6605efb6`, "publish a refill
generation only once its files exist"); this is a different one, in the
*compilation* rather than the publication, and it only becomes likely under the
regeneration pressure the corrected bound creates. It is a real, priced cost of
18.6 option (1) and it would have to be fixed first. Not chased down here.


### 18.9 The diagnostics used, and the bit-identity check

Four environment-gated switches on `MadSpin/interface_madspin.py`, all off
unless the variable is set, none of them committed. Reproduce with:

    MS_SCAN_FRESH_RHO=1   re-derive rho_prod on every off-shell scan draw
                          (i.e. the event loop's own condition, applied to
                          _joint_maxwgt_range)
    MS_SCAN_DUMP=<dir>    dump (event, draw, weight, d) per scan trial
    MS_SCAN_ONLY=1        os._exit(0) once the joint bound has been printed
    MS_NO_STALE_SKIP=1    stop density_debug skipping the stale scan trials

    --- a/MadSpin/interface_madspin.py
    +++ b/MadSpin/interface_madspin.py
    @@ _joint_maxwgt_range, at the shipped ``if density_matrix_prod is None:``
    -                if density_matrix_prod is None:
    +                if (density_matrix_prod is None
    +                        or (_MS_SCAN_FRESH_RHO and offshell_density)):
                         _, wgt, density_matrix_prod = self.get_onshell_evt_and_wgt(
                             prod_draw, decays, decay_dict, build_event=False)

plus the dump append, an `os._exit(0)` after `self._pi_max_weight = ...` in
`get_maxwgt_for_onshell`, `and not _MS_NO_STALE_SKIP` on
`self._density_debug_stale`, and a `_ms_scan_resonance_distance` helper that is
called only from the dump.

Two checks that the shipped path is untouched:

* with every switch off (only `MS_SCAN_DUMP` set, which appends to a list and
  writes a file), the full `p p > t t~ j` run reproduces the reference sample's
  **300 000 events byte for byte** -- `sha256` over the event blocks
  `f76f89c4a963ffa16b2c81c3edf02f96a61aad1da94a401dbb520ddf5e67b73c`, identical
  to the pre-axial reference run -- and reports the identical
  `265/300000 ... largest factor 48.9071 ... +0.245 %`;
* the pre-existing `density_debug` failure of 18.7 reproduces to the last digit
  against an untouched checkout of the same tip.

The instrumentation is **not committed**: `MadSpin/interface_madspin.py` in this
commit is byte-identical to `2ba47b8d8` (`git diff` against that tip is empty),
so milestone 3's own bit-identity property is untouched by this section.

`python tests/test_manager.py test_madspin -t0`: **498 OK**, on the reverted
tree.
