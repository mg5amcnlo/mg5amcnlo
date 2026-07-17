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
   `density_do_reshuffle` (`spinmode == 'PA'`). The jacobian must then be
   attributed to the slot that draws it. This is a genuine coupling between
   slots and the reason phase 1 stays on `spinmode = onshell`.
4. `fixed_order` counter-events.
5. `density_debug` compares against the full ME and is only meaningful for a
   complete set.

**Recommendation:** phase 1 supports `spinmode = onshell` (PA without
reshuffling, ratio uncontaminated by `jac`) and refuses / falls back for
`fixed_order`. `spinmode = PA` with mass sampling is phase 2, once the
per-slot jacobian attribution is settled.

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
4. `_sequential_accept_reject` + per-slot max weights + per-slot efficiency,
   `spinmode = onshell` only, `fixed_order` falls back.
5. A/B campaign (4-5). Only then consider `spinmode = PA` mass sampling
   (jacobian attribution) and the phase-3 partial contraction.
