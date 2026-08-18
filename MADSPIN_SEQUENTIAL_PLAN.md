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
6. ~~`density_debug` must still pass in joint mode (unchanged code path).~~
   **`density_debug` is itself broken** and cannot be used as a validation
   instrument -- see the note at the end of section 10.

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

### Status: implemented, gated OFF -- efficiency blocker

The offshell path IS implemented: `_offshell_production` (up-front mass draw +
reshuffle of a copy + fixed rho) and the `offshell` branch of
`sequential_accept_reject` (offshell density on a copy of the decay so the
drawn decay stays onshell for the final add_decays + reshuffle; weight
`(N_k/N_{k-1}) * jac_bw_k * Tr(D_k^off)/|M_k|^2_on`, with `jac_reshuffle` on slot
0). It runs end to end and produces kinematically valid events (no crash).

But `_sequential_active` still returns False for madspin/full, because it is
**slower than the joint test on ttbar**: ~340 decay-ME evaluations per event
(slot 0 ~313, slot 1 ~27) against joint madspin's ~122 (61 trials x 2 decays).
The cause: madspin is inherently peaked (joint itself needs 61 trials/event),
and the per-mass-set production reshuffling jacobian `jac_reshuffle` plus the
offshell weight tail land in **slot 0's per-angle accept/reject**. Since the
mass is fixed per chain, an unlucky mass draw cannot be escaped by redrawing
angles, so slot 0's bound (max weight ~322) is huge and its acceptance ~1/313.

The mass-set-level accept/reject was implemented (a step before the per-angle
loop, weight `w_mass = Tr(rho_off) * jac_reshuffle * prod jac_bw_k`, with the
per-angle factors reduced to `(N_k/N_{k-1}) * Tr(D_k^off)/|M_k|^2_on`). It works
and isolates the reshuffling jacobian: its bound is modest (C_mass ~ 14 on
ttbar). It works, isolates the reshuffling jacobian (C_mass ~ 14 on ttbar), and -- for
physical resonant decays -- makes sequential madspin **faster than the joint
test**. Validated end to end on `p p > t t~`, `t > w+ b, w+ > l+ vl` (fully
leptonic), same production events, `nb_core=1`:

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

**madspin/full are reachable but not the default** in `_sequential_active`:
`sequential_decay = auto` resolves to sequential for PA/onshell and to the joint
test for madspin/full, and the wall-time measurement below says it should stay
that way. The open item is the
`w+ > all all` non-resonant blow-up in the density-madspin *weight* (BW_cut too
wide for unweighting the reshuffle? reweighting normalisation? keep those
channels weighted?), which would help the default joint madspin too.

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
from normalising at all, which is what `sequential_exact` does.

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
  their tables (and depend on `sequential_exact`), so they get their own file
  name and a JSON format (`_read_upfront_cache` / `_UPFRONT_CACHE_FORMAT`). The
  file name still carries the spinmode family that wrote it
  (`max_wgt_sequential_offshell...` / `max_wgt_sequential_pa...`), since the
  mass-set weight is a different quantity in each and neither cache may be read
  back for the other.

#### `sequential_exact`: the escape hatch

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
    sequential_exact 173.1914 +- 0.0323   173.1906 +- 0.0323      chi2/ndf 10.4/22

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
                                104.5  -> 12.79     (sequential_exact)
    weights above their bound      10  ->  1        (sequential)
                                   28  ->  3        (sequential_exact)
    decay phase                  28.7s -> 19.7s     (sequential)
                                 83.9s -> 31.4s     (sequential_exact)

with the lineshape unchanged, as the constancy argument requires: over both
resonances the sequential mean moves from 173.1641 to 173.17 and the exact one
sits at 173.1877 against joint's 173.1853. Cost: one onshell production matrix
element per production event, cached under `me_wgt` -- the same attribute, and
the same quantity, the joint path already caches there.

#### `sequential_joint_angles` (variant A): one bound over all the angles

Suggested by Olivier. Keep the mass-set stage, but replace the *per-slot*
accept/reject by a single test on the product of every slot's weight, redrawing
the whole angle set on a rejection and **keeping the mass set**:

    stage 1   w_mass  = [Tr(rho_off)/|M_prod|^2_on] * jac_reshuffle
                        * prod_k jac_bw_k * prod_k Z_hat_k(m_k)
    stage 2   w_angle = prod_k [ (N_k/N_{k-1}) * jac_dec_k
                                 * Tr(D_k^off)/|M_k,dec|^2_on / Z_hat_k ]

This is the same target distribution as the per-slot scheme -- same mass stage,
same self-normalising angle stage, only the granularity of the test changes --
and the measurement says so: over four replicas each, variant A gives
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
`sequential_exact`, and both were fixed together.

#### Variant B (`sequential_joint_angles` + `sequential_exact`): dropped

The same single angle bound, but a rejected angle set trashes the mass set. That
makes `Z_hat` cancel between the stages and the scheme exact whatever the table
says, and it costs the reuse above (9.25 mass sets per accepted event instead of
3.25). It was measured and **dropped**: over four replicas it sits 0.034 GeV
below joint, which is the *largest* deviation of any scheme tried and in the
scheme that should have been the most exact. That is not understood. Either the
error model below is wrong or that implementation is; the combination is
reachable in the code but should not be used until the weight-identity check
settles it.

#### The option: one knob, four schemes

`sequential_decay`, `sequential_exact` and `sequential_joint_angles` are replaced
by a single enumerated option -- the schemes are mutually exclusive alternatives,
not independent switches, and after variant B was dropped the three booleans no
longer spanned a clean 2x2:

    set unweighting auto | joint | two_stage | sequential | sequential_global_retry

    mode                     mass stage   angle test              a rejection redraws
    joint                    --           everything at once      everything
    two_stage                yes          all angles, one bound   the angles only
    sequential               yes          per particle            that particle
    sequential_global_retry  yes          per particle            the virtualities too

`sequential_decay` survives as a deprecated alias (`True` -> sequential,
`False` -> joint, warning once), so cards written against the earlier revisions
of this branch keep working. `sequential_spin_order` and `sequential_debug` keep
their names: an ordering and a check, not modes.

**`sequential_exact` was renamed, not kept.** "Exact" advertised a distinction of
~0.001 GeV on the top lineshape -- the tabulated factor is good to ~0.5%, and the
lineshape sensitivity is 0.25 GeV per unit fractional slope error -- inside a pole
approximation whose own error is of order `Gamma/m` ~ 0.9%, three orders of
magnitude larger. The name would have pushed users to a 2-3x slower scheme for a
difference nobody can measure. `sequential_global_retry` says what the mode does
and leaves the accuracy statement to the documentation, where it can carry the
numbers.

**`auto`** (as of this section; **superseded by section 12**, which measured
the schemes over the decay multiplicity and replaced the four branches below
with two) resolves once per run, from the number of decaying particles counted
where `to_decay` is built -- not per event, since the modes carry different
bounds and one that changed event to event would be testing against the wrong
ones:

- **one decaying particle -> `joint`**, in every spinmode. Every split
  degenerates there: the per-particle test *is* the joint test (section 3), and
  the mass/angle one only moves the same factors between two stages. Nothing to
  win, so the identity machinery is pure cost.
- **PA/onshell -> `sequential`**. `two_stage` and `sequential_global_retry` split
  the accept/reject at the up-front mass draw, which those modes do not have;
  asked for explicitly they log why and fall back to `sequential`.
- **madspin/full -> `two_stage` for two, `sequential` from three.** One bound
  over all the angles is tighter than the product of per-particle bounds, while
  testing each particle as it is drawn lets a rejection skip the decays not yet
  drawn. The first wins while there is little to skip, the second as the chain
  lengthens.

That last line makes `auto` non-joint for madspin/full, where `two_stage` is
faster than the joint test (3.25 production densities and 5.74 decay MEs per
event against 4.46 and 8.92) and agrees with it at +0.23 sigma over eight
replicas. An explicit setting is always honoured, including the degenerate
single-particle case, so any of the four stays available as a cross-check.

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
    PA         sequential (default)     11.19 s     1.88 + 3.13 -> 5.01 decay ME
    madspin    variant A                13.55 s     3.25 mass sets, 5.74 decay ME
    madspin    joint                    14.61 s     4.46 trials -> 8.92 decay ME

So full offshell matrix elements with variant A cost about **1.5x PA-joint**,
where madspin-joint costs 1.6x, and variant A is **7-9% faster than
madspin-joint** (13.06-13.55 s against 14.43-14.61 s over two campaigns).

Two observations about PA, both independent of this work:

- **PA sequential is 22% slower than PA joint on this process**, despite drawing
  fewer decay events (5.01 against 6.28). With `density_keep_jacobian` on, every
  slot trial calls `_production_jacobian_for` -- an `Event(str(production))` copy
  and a reshuffle -- so 5.01 production reshufflings per event against joint's
  3.14. The per-slot decomposition is supposed to pay off as n grows; at n = 2 it
  does not, and `sequential_decay = auto` makes sequential the default for PA.
- **PA sequential logged 11 weight overflows** (9 at slot 0, 2 at slot 1) against
  variant A's 1 and joint's 0. Its per-slot bounds are under-estimated here, so
  that sample is slightly biased. Worth a look on its own.

#### Where the offshell path stands

`sequential_decay = auto` still routes madspin/full to the joint accept/reject.
Variant A is now faster than joint on n = 2 and correct as far as the statistics
can tell, so that default is worth revisiting -- but not before the residual
below is understood.

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

    variant A, 1x probe   173.1704 +- 0.0101    -0.011 +- 0.010   (-1.1 sigma)
    variant A, 5x probe   173.1985 +- 0.0182    +0.017 +- 0.018   (+0.9 sigma)
    joint                 173.1818 +- 0.0024

-- it flipped sign rather than shrinking, and three of the four deep-probe
replicas sit *above* joint, which retires the "same sign every time" pattern.
Pooling all eight variant A replicas gives **173.1844 +- 0.0110 against joint's
173.1818, i.e. +0.003 +- 0.011 (+0.23 sigma)**. Variant A agrees with the joint
accept/reject.

`max_weight_ps_point = 500` is therefore sufficient for the Z table; the deeper
probe costs 169 s against 76 s per 10000-event run (the probe is fixed setup, so
it amortises on larger samples) and buys nothing.

**What this leaves open.** Variant B's -0.034 GeV was quoted at "-4.3 sigma" on
the replica-scatter error model; that significance is not trustworthy. The
replica scatter itself ranges from 0.005 (joint) to 0.036 (variant A, deep
probe) across schemes estimated from four points each -- a ~40% uncertainty on
the error bar before any comparison is made -- and the two error models
(naive per-run MC error, and replica scatter) disagree by a factor of three.
Any future claim at the few-hundredths-of-a-GeV level needs either many more
replicas or, better, the deterministic check below.

**Done: the weight identity holds (`sequential_debug`).** New option, offshell
only: on every accepted chain, recompute the joint weight with the joint code --
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

    variant A            spread 1.71e-07   ratio 1108198261
    sequential per-slot  spread 1.55e-07   ratio 1108198255
    variant B            spread 1.54e-07   ratio 1108198258

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
variant A and the per-slot scheme (measured at ~0.5%, far inside the ~12%
tolerance), automatic for the restart schemes. So variant B's -0.034 GeV is not
a broken weight. With the weights verified and the error models in the state
described above, the honest summary is: weights correct, deviation unexplained,
dropped because it is slower than variant A anyway.

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

It stays the `auto` choice for PA and onshell. The bit-for-bit check below is
what makes the rename safe.

### `sequential_with_mass` offshell: asked, and the answer is no

The planning revision of this section left open whether the offshell spinmodes
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
