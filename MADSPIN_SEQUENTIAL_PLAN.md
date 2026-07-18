# MadSpin: sequential (per-particle) accept/reject in density mode

Plan for replacing the joint accept/reject over all decaying particles by a
per-particle one, in `density_method` mode (now the default). Opt-out flag so
each process can be A/B tested.

Code references are to the current tree (`MadSpin/interface_madspin.py`,
`MadSpin/decay.py`).

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

### Where it bites (why the flag is mandatory)

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

**Scope: `spinmode = PA` (the default, banner.py `add_param('spinmode', "PA")`)
is in.** An `onshell`-only feature would be inert for essentially every user.
`fixed_order` still falls back to the joint test.

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

## 3. New options (MadSpinCard, interface_madspin.py:~58)

```python
self.add_param("sequential_decay", True,
               comment="accept/reject one decaying particle at a time "
                       "(density mode). Set to False for the historical "
                       "joint accept/reject.")
self.add_param("sequential_spin_order", "2 3 1", hidden=True,
               comment="spin order (MG5 2S+1 convention) used to decide which "
                       "particle is decayed first: default fermions, then "
                       "vectors, then scalars.")
```

- `sequential_decay` defaults to **True** (opt-out, per request); forced False
  when `density_method` is off, when `fixed_order` is on, and when only one
  particle decays (then it is identical to the joint test -- fall back rather
  than pay for the identity machinery).
- `sequential_spin_order` is hidden and lets the ordering itself be A/B tested
  per process without a code change.

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

## 5. Pool sizing ladder (interface_madspin.py:1796-1830)

Today:

```python
spin = self.model.get_particle(pdg).get('spin')
if spin == 1:      # MG5 convention: scalar
    efficiency = 1.1
else:
    efficiency = 2.0
```

Sequential replacement -- **ladder by position, capped by spin** (per decision):

```python
# position k (0-based) in the decay ordering
efficiency = 1.1 if spin == 1 else 1.5 + 0.5 * k     # 1.5, 2.0, 2.5, 3.0, ...
```

- scalars keep 1.1 at whatever position they land (their ratio is identically
  1, so a bigger pool would be pure waste);
- spin 1/2 and spin 1 take the ladder value **at their own index**;
- beyond 4 particles the formula keeps going (3.5, 4.0, ...); consider a cap
  once measured.

The `+ nevents_for_max` term and `decay_event_mult` are unchanged. Note the
pool is sized per *pdg*, while the ladder is per *slot*: for several identical
parents (same pdg, several slots) take the max ladder value over that pdg's
slots -- the same file feeds them.

This is the one part of the plan that is a heuristic rather than a derivation;
the real efficiencies per slot should be logged (section 8) and the ladder
revisited against measurement.

---

## 6. Max weights: one bound per slot

`get_maxwgt_for_onshell` (:2810) currently records one `maxwgt` per production
event, then combines: `1.05 * (mean + nb_sigma*std)` over the per-event maxima,
refined over the top 20/30/40/50 and against `all_maxwgt[1]`.

Generalise to `n` independent bounds `C_k`, one per slot:

- during the scan, for each PS point compute the n ratios `N_k/N_{k-1}` for the
  sampled set and track the per-event max of **each**;
- `all_maxwgt` becomes a list of n-vectors; run the existing statistical
  combination independently per slot.

The scan may keep sampling decay sets uniformly from the pool even though the
real chain conditions on earlier accepted decays: uniform sampling explores the
same support, so the max over uniform draws remains a valid estimator of the
same bound. It does change the *sampling density* of the ratio, so the tail
estimate is not identical -- an argument for keeping `nb_sigma`/`1.05` margins
and for the overflow counter below.

`ms_dir`'s cached `max_wgt` file holds a single float: bump it to a list
(and invalidate the old format, e.g. by name `max_wgt_seq`) so a stale cache
cannot be silently read as a scalar.

Add a per-slot **overflow counter**: count `N_k/N_{k-1} > C_k` and log it at the
end (the joint path has the same exposure on a single bound, but n bounds mean
n chances to under-estimate). A non-zero count is the first thing to look at when A/B
disagrees.

---

## 7. Code changes, file by file

**`MadSpin/decay.py`**
- `DensityMatrix.identity_like(cls, template)` (or `identity_for(helicities)`):
  same basis / `basis_id`, values = 1 on `_diag_mask`, 0 elsewhere, scaled
  1/n. Must produce the exact row order of the template so the
  `scalar_multiplication` fast path (`map_density_matrix_ind is other...`)
  stays live.

**`MadSpin/interface_madspin.py`**
- `MadSpinCard`: the two options above (:~58).
- `get_decay_from_file` (:2720): extract the per-particle body (file choice by
  cross-section, `next(decay_file)`, the refill/`StopIteration` path) into
  `_draw_one_decay(particle, i, ids, evt_decayfile, nb_remain)`. The existing
  function becomes a loop over it -- **the joint path must stay byte-identical**.
- `calculate_matrix_element_from_density` (:3027): accept an optional
  `fixed_slots` set; build `density_dec` with `identity_like` for the unfixed
  slots. Return `N_k` alongside what it returns today. Keep the current
  signature working (all slots fixed = today's behaviour).
- new `_sequential_accept_reject(production, ...)`: the loop of section 1,
  replacing the `while 1:` block in `_run_onshell_loop` (:2325-2385) when the
  flag is on. Reuses `prod_density_cached` exactly as today (:2324) -- it is
  computed once per production event and is now reused across *all* slots and
  retries, which is strictly more valuable than before.
- `get_maxwgt_for_onshell` (:2810): per-slot bounds (section 6).
- pool sizing (:1796-1830): the ladder (section 5).
- `_run_onshell_loop`: efficiency bookkeeping is currently
  `self.efficiency = (curr_event+1)/nb_try` and feeds the refill estimate in
  `_draw_one_decay`. Sequential needs **per-slot** efficiency (each slot burns
  its own pool at its own rate) -- otherwise the refill sizing, which already
  reasons about `burn` per pdg, will be wrong. This is the subtlest piece of
  the wiring.

**Interaction with work already committed**
- The parallel workers (fork) each run their own loop; per-slot efficiency and
  overflow counters must join the per-shard stats dict already marshalled back
  (`n_processed`, `n_written`, `nb_try`, `nb_loose_skip`) and be summed in
  `_apply_accounting`. Keep them order-independent sums, like the existing ones.
- The BR-equalization drop (`drop_prob_per_pdg`) happens before any ME work and
  is unaffected.

---

## 8. Validation

The whole point of the flag is A/B, so the plan is measurement-first:

1. **Unit** — spin-0 slot: `N_k/N_{k-1} == 1` exactly.
2. **Unit** — `identity_like`: trace 1, `scalar_multiplication` against a known
   rho reproduces `Tr(rho)/prod n_i`; all-slots-fixed reproduces today's `wgt`
   bit-for-bit.
3. **Unit** — ordering: `_decay_slot_order` for mixed spins, ties stable;
   ladder values per slot incl. the scalar cap and the several-identical-parents
   max rule.
4. **Physics A/B** (the real test) — same seed, same events, `sequential_decay`
   True/False, compare distributions sensitive to spin correlation:
   - `t t~` semi-leptonic: lepton angular distribution / `cos(theta*)`, the
     classic MadSpin observable;
   - a process with two spin-1/2 and one scalar to exercise ordering;
   - `W+ W-` (two vectors) for the 3x3 blocks.
   Compare against the *joint* result, not against theory: they must agree
   within MC error. Any disagreement points at section 1's "where it bites".
5. **Efficiency** — log per-slot acceptance and total decay events consumed per
   production event, both modes. That is the number that justifies the feature
   and calibrates the ladder.
6. `density_debug` must still pass in joint mode (unchanged code path).

---

## 9. Suggested phasing

1. `identity_like` + `fixed_slots` in the contraction + unit tests 1-2.
   (No behaviour change: joint path untouched.)
2. `_draw_one_decay` refactor + unit test that the joint path is unchanged.
3. Options, ordering, ladder (+ tests 3).
4. Two preparatory steps first: untangle the mass ownership (section 1, "Mass
   ownership") -- the draw moves into the per-slot loop, the basis setup comes
   out from under the `prod_static` cache guard -- and add the jacobian-only
   production entry point (section 1, "Evaluating J_k"), with a test that it
   returns the same jacobian as `reshuffle_production` while leaving the event
   untouched. Then `_sequential_accept_reject` + per-slot max
   weights + per-slot efficiency, for `spinmode` in PA/onshell (`fixed_order`
   falls back). PA draws slot k's Breit-Wigner mass inside slot k's
   accept/reject, weight `(N_k/N_{k-1}) * jac_k`, reshuffles that decay there
   and redraws its mass on failure; the production reshuffling happens once at
   the end and, if impossible, trashes the whole set of decays (section 1).
5. A/B campaign (8). Only then the partial-contraction optimisation.

---

## 10. Extending to spinmode = madspin (full offshell) -- DESIGN, not yet built

Status: `onshell` and `PA` are implemented and validated end-to-end (ttbar
A/B: cross section 0.008%, dilepton Delta-phi within ~1 sigma). `madspin` still
falls back to the joint accept/reject. This section records how to lift it.

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

### Fix (Olivier): draw all masses up front

Draw the invariant mass of every decaying particle **before** the per-particle
loop, then reshuffle the production once, up front, and reuse the resulting
fixed offshell rho for the whole chain. Concretely, per production event:

1. For each decaying particle, sample its virtuality from its Breit-Wigner.
2. Reshuffle the production with that full mass set.
   - **Production infeasible** (sum of masses > sqrt(shat), reshuffle returns
     -1): restart from step 1 (redraw the whole set). This validity check now
     happens *early*, before any decay is drawn -- an advantage over PA, where
     it is deferred to the end.
3. Compute rho once at the reshuffled momenta (fixed for the chain).
4. Per-particle accept/reject loop, exactly as onshell but: each drawn decay
   event is reshuffled to its particle's pre-drawn mass before its density is
   taken, and boosted to the offshell parent.
   - **Decay infeasible** (the drawn mass cannot accommodate that decay's
     products): the mass is fixed before the loop, so it cannot be redrawn for
     one slot without invalidating the pre-computed reshuffle/rho -> **restart
     from step 1** (redraw the whole set). This is the cost of the fixed-rho
     simplification.

With rho fixed, the loop and its telescoping are the onshell case again.

### The remaining subtlety: onshell denominators vs offshell numerator

Not resolved by fixing rho. Joint madspin's weight is

    <rho_off, (x) D_off> * jac / ( |M_prod|^2_on * Prod |M_dec|^2_on )

-- offshell **numerator**, onshell **denominators**. The per-particle method
normalises each slot by `Dhat = D / Tr(D)`, i.e. by the **offshell** trace
Tr(D_off), and rho by Tr(rho_off) = N_0. Since Tr(D_off) != |M_dec|^2_on and
Tr(rho_off) != |M_prod|^2_on in general, the naive sequential weight targets a
*different* distribution than joint madspin.

To match, each slot's factor has to be the offshell density over the **onshell**
decay ME, `D_off,k / |M_dec,k|^2_on`, and the production factor rho_off over
|M_prod|^2_on -- i.e. compute the onshell decay ME (un-reshuffled decay) as the
normaliser in addition to the offshell density. That is one extra ME evaluation
per decay and per production, and the telescoping must be re-derived with these
mixed normalisers before trusting it. This is the piece to get right (and A/B
against joint madspin) when implementing; do not ship madspin sequential without
that check.

### Gating

`_sequential_active` currently returns False for spinmode not in
['PA','onshell']. Extending to 'madspin' is the last step, after the
normalisation above is implemented and the ttbar (and a genuinely offshell,
e.g. large-width) A/B against joint madspin passes.
