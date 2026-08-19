# The PA mass stage: is `<w>` a per-event constant, and what should be done

Design assessment, built on `findings.md` (same directory), which established
that `eps_m = C/<w>` exactly in all three modes, that `<w> = 0.955` globally in
all of them, and that the whole cost difference is therefore the bound `C`,
which is set by a `1/beta_t` tail of the reshuffling jacobian at the `t t~`
threshold and is neither stable nor high enough (11 overflows in 375 715
trials). Nothing of that is re-derived here.

Process throughout: `p p > t t~` at 6.5+6.5 TeV, 50 000 unweighted production
events, both tops leptonic, `BW_cut = 15`, `spinmode = PA`,
`unweighting = sequential`, `density_keep_jacobian = True` (the default). The
`Zhat` tables and the bound `C = 3.2199` come from a real MadSpin run on that
sample (`data/max_wgt_sequential_pa.json`, `data/logs/`), which logged
`eps_m = 3.31` against `C/<w> = 3.371`.

---

## 0. The answer, in one paragraph

**`<w | production event>` is not a constant.** It is flat to within 0.2 % over
the 80 % of the sample above 400 GeV, and 95 % of production events sit within
1 % of the global mean -- and then it falls to **-5.8 %** around
`sqrt(shat) = 349 GeV` and **diverges** as `sqrt(shat) -> 2 m_t`: `+19 %` at
346.5 GeV, `+124 %` at 346.1 GeV, without bound, like `1/beta_t`. Over 6000 production events the
event-to-event spread is `0.95 %` rms against a `0.053 %` Monte Carlo error --
eighteen times the noise -- with 4.8 % of events beyond 1 % and 0.4 % beyond
5 %. Two independent measurements agree to 0.08 %.

The consequence is **not** the one the three candidate options were framed
around. Redraw-until-accept divides `A_e = <w|e>` out whatever bound it uses, so

* **option B (a per-event bound) is not gated on this result at all.** The
  accepted mass distribution inside one production event is `q_e w` for *any*
  bound that dominates the weight -- the bound cancels. B is a pure
  efficiency-and-overflow change (`eps_m` 3.37 -> 1.20), and it is safe whatever
  the answer to the constancy question is;
* **neither A nor B nor C fixes the bias that the non-constancy causes**, which
  is a *between-event* normalisation, not a within-event shape. Fixing that
  needs a fourth thing: carrying `A_e` on the event weight (section 5). For a
  `2 -> 2` production `A_e` is an exact two-dimensional quadrature costing
  3.3 ms per event, because the reshuffling jacobian is analytic there
  (section 3).

---

## 1. How it was measured

MadSpin's own trials cannot answer the question. The mass stage redraws until
it accepts, so an event contributes a *stopped* sequence whose last member is
accepted and whose earlier members were rejected -- not an i.i.d. sample of the
proposal. `per_event_weight.py` therefore draws `N` **free** mass sets per
production event, with no accept/reject and no stopping rule, through the
shipped functions and nothing else:

    MadSpinInterface._draw_mass_value          the Breit-Wigner draw and jac_BW
    MadSpinInterface._production_jacobian_for  the RAMBO reshuffling jacobian J
    MadSpinInterface._zhat                     the tabulated Zhat

on a shim carrying only the banner, `BW_cut` and the `z_tables` the real run
left in its `ms_dir` cache. `N = 400` above 450 GeV, 2000 between 380 and 450,
8000 below 380 -- 6000 production events, **1.03e7 mass sets**.

Two quantities, because two questions want different denominators:

| | |
|---|---|
| `A_e = sum(w)/N` | infeasible mass sets counted as `w = 0`. `_upfront_production` returns None on those and the chain restarts *without* reaching the accept/reject, so they are invisible to MadSpin's counters -- but they belong to the event's normalisation. **This is what redraw-until-accept divides out.** (Measured: infeasible sets never happen for `t t~`; the draw is budget-capped, so the feasible fraction is 1.000 in every slice.) |
| `A_e^f = sum(w)/n_feasible` | the mean of what the accept/reject tests, i.e. what sets the acceptance `C/A_e^f`. |

**Independent check.** For a `2 -> 2` production the whole thing is analytic
(section 3), so `A_e` can be computed by quadrature with no Monte Carlo at all:
the sampler is uniform in `R = atan((m^2-pole^2)/(pole Gamma))` and
`jac_BW = gap/pi` is exactly that window's width in `R` over `pi`, so

    A_e = (1/pi^2) Int dR_1 Int dR_2  J(m_1,m_2;s) Zhat(m_1) Zhat(m_2)

with the `R_2` range capped by the remaining budget. 48-point Gauss-Legendre in
both, 3.3 ms per event (`jacobian_analytic.analytic_A`).

    Monte Carlo / quadrature over 4000 production events:
        mean 0.99999,  sd 0.00078,  range [0.9913, 1.0120]
        pull (A_MC - A_quad)/sigma_MC:  mean 0.04,  sd 1.03

The two methods agree at the level of the Monte Carlo error, and the pull is a
unit Gaussian. Everything below quotes the quadrature where a noise-free number
is wanted and the Monte Carlo where a sample average is.

## 2. The result

`plots/per_event_mean_weight.png`.

| | `A_e/<A>` |
|---|---|
| rms over 6000 production events | **0.0095** |
| mean absolute deviation | 0.0030 |
| Monte Carlo error on one `A_e` | **0.00053** |
| range | **[0.938, 1.512]** |
| events beyond 1 % | 4.82 % |
| events beyond 5 % | 0.38 % |

The spread is 18 times the measurement error, so it is real and not noise. The
bottom-left panel of the plot makes the point directly: the measured histogram
against the histogram the same statistics would have produced if `A_e` really
were one number.

The shape, from the quadrature (exact, no Monte Carlo):

| `sqrt(shat)` [GeV] | `A_e` | `A_e`/plateau | `A_e`/plateau, `PA (no jac.)` |
|---|---|---|---|
| 346.1 | 2.1456 | **2.241** | **0.522** |
| 346.5 | 1.1348 | 1.186 | 0.607 |
| 347 | 0.9655 | 1.009 | 0.696 |
| 348 | 0.9054 | 0.946 | 0.809 |
| 349 | 0.9017 | **0.942** | 0.869 |
| 350 | 0.9059 | 0.946 | 0.903 |
| 352 | 0.9162 | 0.957 | 0.941 |
| 355 | 0.9275 | 0.969 | 0.967 |
| 360 | 0.9393 | 0.981 | 0.987 |
| 370 | 0.9505 | 0.993 | 1.000 |
| 400 | 0.9557 | 0.998 | 1.000 |
| 500 | 0.9570 | 1.000 | 1.000 |
| 1000 | 0.9573 | 1.000 | 1.000 |

So the honest answer has three parts, and the nuance is the point:

1. **Constant to 0.2 % for the 80 % of the sample above 400 GeV** (and to
   0.06 % above 450). For those events a single global `<w>` is exactly right,
   and the earlier probe's global 0.955 is their number.
2. **A 5-6 % deficit** between 348 and 355 GeV (about 2 % of the sample) --
   `A_e` is *smaller* there, so redraw-until-accept *over*-populates those bins.
3. **An unbounded excess in the last GeV**: `A_e` diverges like `1/beta_t` as
   `sqrt(shat) -> 2 m_t`, reaching 2.24x the plateau at 346.1 GeV, which is the
   lowest `sqrt(shat)` the production sample actually contains. It is not
   "approximately constant near threshold"; it has no maximum.

**Why**, from the right-hand panel of the plot: two factors that nearly, but not
quite, cancel.

* `<J>` rises like `1/beta_t` (it *is* `1/beta_t`, section 3) -- 1.66x the
  plateau at 346.1 GeV;
* `<jac_BW . Zhat>` falls, because the second top's Breit-Wigner window is
  capped at `budget = sqrt(shat) - m_1`, and below `sqrt(shat) ~ 368 GeV` that
  cap bites: `gap/pi` shrinks and with it the weight. 0.67x the plateau at
  346.1 GeV.

The product overshoots on one side of 347 GeV and undershoots on the other.

**The mode with no jacobian tail at all is the more biased one.** The last
column of the table is `PA (no jac.)` (`density_keep_jacobian = False`), whose
weight is just `prod jac_BW` -- the mode `findings.md` showed is constant to
0.2 % over 99.9 % of trials and costs `eps_m = 1.10`. Its `A_e` is a
*deterministic* function of `sqrt(shat)` and is **48 % low** at 346.1 GeV, 30 %
low at 347, 10 % low at 350. Its accept/reject has nothing to unweight and its
bound is reproducible to four digits, and it still has this. **The
non-constancy is not a property of the jacobian tail. It is the truncated
Breit-Wigner window, and every mode has it**, including `madspin`, which
additionally carries the event-dependent `Tr(rho_off)/|M|^2_on`.

**End-to-end cross-check, and its limits.** `set decay_output = weighted`
(section 13.18 of `doc/madspin_sequential_plan.md`) writes one event per
production event carrying `w = w_prod * BR * W / c`, i.e. it keeps `W` instead
of normalising it away, and the probe applies the reshuffling jacobian to `W`
under PA. A 50 000-event run (`data/logs/decay_output_weighted_50000.log`)
passes its own normalisation self-check (`mean(w) = 23.7911 +- 0.0304` against
`sigma*BR = 23.7622`, 0.95 sigma) and its per-bin mean weights are compatible
with flat: `chi2 = 8.6` against `chi2 = 13.7` for the predicted curve over the
15 bins below 400 GeV, with an aggregate `+0.84 % +- 0.38 %` pull away from the
curve. **It is not evidence against the result, and it is not evidence for it
either**: the per-bin errors are 6-12 % where the effect is 5 %, and, more
importantly, `decay_output = weighted` takes the *joint* path, whose mass
proposal is the decay pool's virtuality with an untruncated window and an
internal retry, not the budget-capped `_draw_mass_value` the sequential mass
stage uses. It is a different sampler of the same physics. The two measurements
that *are* of the sequential mass stage's own weight -- 1.03e7 draws and the
quadrature -- agree to 0.08 %.

## 3. The reshuffling jacobian is analytic for a `2 -> 2` production

`jacobian_analytic.py`. `Event.mass_shuffle` scales every spatial momentum in
the production CM by a common `chi`; for `n = 2` the two momenta are back to
back with the same modulus, `chi = |p'|/|p|`, and every factor of RAMBO eq. 4.9
collapses:

    J = chi^3 . (E1 E2)/(E1' E2') . [|p|^2 sqrt(s)/(E1 E2)] / [|p'|^2 sqrt(s)/(E1' E2')]
      = chi^3 |p|^2/|p'|^2  =  chi  =  |p'|/|p|
      = lambda^(1/2)(s, m1'^2, m2'^2) / lambda^(1/2)(s, m1^2, m2^2)

the two-body phase-space volume ratio and nothing else. Verified to
**2.4e-09** over 1988 random `(sqrt(shat), mass set, orientation)`. This is the
`1/beta_t` divergence in closed form: on shell at threshold `|p| -> 0`.

Two things follow, and they are what make options B and C cheap:

* the **per-event maximum is exact and free** -- `J` grows monotonically as the
  masses fall, so `J_max(e) = |p'|(s, m_lo, m_lo)/|p|(s, m_pole, m_pole)`, two
  square roots:

  | `sqrt(shat)` | 346.5 | 350 | 360 | 380 | 450 | 700 | 1000 |
  |---|---|---|---|---|---|---|---|
  | `J_max` | 9.20 | 3.38 | 1.98 | 1.47 | 1.16 | 1.04 | 1.02 |

* the **per-event normalisation is a quadrature**, 3.3 ms (section 1).

For `n >= 3` neither is available: `J` is not a function of the mass set alone.
The same masses at the same `sqrt(shat)` give `J` from 0.922 to 1.056 across
eight random three-body configurations -- a 14 % spread at fixed masses.

## 4. The three options

### A. Carry the overweight instead of clipping it

**What it fixes.** The clipping bias, and only that. When `w > C` the trial is
accepted with probability 1 instead of `w/C`; writing it with weight `w/C`
restores the correct within-event shape, because the loop picks the trial it
stops on with probability proportional to `min(1, w/C)` and
`min(1,x) . max(1,x) = x` identically.

**How big.** Measured against the run's own `C = 3.2199`:

| | |
|---|---|
| overweight events per 100 000 **written** | **9.9 +- 4.8** |
| production events that can ever produce one | 0.18 % |
| the sample's mean weight, if carried | 1.000161 |

The 9.9 per 100 000 reproduces the 11 per 375 715 trials `findings.md` observed
in an independent 100 000-event run, which is a good closure on the accounting.
The bias being removed is therefore **0.016 % of the sample's normalisation**,
concentrated on ~10 events per 100 000 which each carry up to `w_max/C ~ 2`.

**Does the output path support it.** Yes, with no new machinery. The write path
already multiplies a per-event factor into the branching ratio,

    br = self.branching_ratio * pi_factor  if (pure_interference or weighted_decay)
    full_evt.wgt *= br;  for key in wgts: wgts[key] *= br

so `pi_factor` is exactly the hook, and it reaches `full_evt.wgt` and every
`parse_reweight()` entry through the same multiplication. On the sequential
path it is 1.0 today and the accept/reject writes unit weights.

**What it costs.** The unit-weight guarantee, for ~10 events per 100 000. MadSpin
is explicit elsewhere that it prefers *dropping* events to weighting them
("the output sample stays unweighted", the BR-equalization path), so this is a
deliberate departure. Two ways to keep unit weights instead: write the
overweight event `floor(w/C)` times plus one more with probability `frac(w/C)`
(duplication, changes `n_written`), or -- better -- make the overflow not happen,
which is option B.

**What it does not fix.** Nothing about efficiency, and nothing about the
between-event normalisation of section 2.

**What could go wrong.** `_apply_accounting` and the `<init>` cross-section are
computed from counts, not from a weight sum; a handful of non-unit weights does
not disturb them under `IDWTUP = -4` (where sigma is the *mean* weight) but the
0.016 % shift has to go somewhere -- either into the banner or nowhere, and
"nowhere" is what happens today. Downstream tools that assume unit weights see
10 events in 100 000 that are not.

**Validation.** `tests/test_manager.py test_madspin -t0`; a run with the bound
forced low enough that overflows are common, checked against a
`decay_output = weighted` run of the same sample; and the existing
`nb_overflow_mass` counter turned from a warning into a measured weight sum.

### B. A per-event bound `C_e`

**Its validity does not depend on section 2.** This is the part of the brief
that needs correcting. Redraw-until-accept inside production event `e` yields
the accepted density proportional to `q_e(m) min(1, w/C)`, and exactly one
output event. Any bound that dominates the weight gives `q_e w` -- **the bound
cancels out of the accepted distribution entirely**. A per-event bound therefore
cannot introduce a bias that a global bound does not already have, and it
*removes* one (the clipping). The `A_e` variation is dropped identically before
and after. So B is safe whatever the constancy answer is; it simply does not
address the bias that the constancy answer reveals.

**What it fixes.** The cost, and the overflow.

| `sqrt(shat)` [GeV] | % of sample | `eps_m`, global `C` | `eps_m`, `C_e = 1.1 max_e w` | `eps_m`, analytic `C_e` |
|---|---|---|---|---|
| 346-350 | 0.50 | 3.55 | 3.48 | 4.73 |
| 350-355 | 1.18 | 3.50 | 2.36 | 3.08 |
| 355-360 | 1.55 | 3.44 | 1.85 | 2.39 |
| 360-370 | 3.28 | 3.40 | 1.55 | 1.98 |
| 370-380 | 4.30 | 3.38 | 1.37 | 1.70 |
| 380-400 | 9.38 | 3.37 | 1.23 | 1.52 |
| 400-450 | 21.3 | 3.37 | 1.13 | 1.34 |
| 450-500 | 16.1 | 3.37 | 1.10 | 1.25 |
| 500-600 | 20.6 | 3.36 | 1.12 | 1.19 |
| 600-800 | 15.1 | 3.36 | 1.14 | 1.15 |
| > 800 | 6.8 | 3.36 | 1.15 | 1.12 |
| **all** | 100 | **3.37** | **1.20** | **1.36** |

`plots/per_event_bound.png`. A probe-built per-event bound takes the mass stage
from 3.37 to **1.20** mass sets per accepted event -- a factor 2.8 -- and the
analytic `2 -> 2` bound, which needs no probe at all, gives **1.36**. The
analytic one is more conservative because it sits at the corner of the window,
where both tops are 15 widths low and the Breit-Wigner essentially never goes;
in the first 4 GeV above threshold it is actually *worse* than the global bound
(4.73 against 3.55) for the same reason `J_max` is 9.2 there. That is 0.5 % of
the sample, and it buys an overflow probability of exactly zero.

**Overflow.** With `C_e >= max_e w` there is nothing left to clip, and with the
analytic bound that inequality is a theorem rather than a probe result. This is
strictly better than A at the same job.

**How big a change.** `get_sequential_maxwgt` already computes `per_event`
before `_combine_maxwgt` collapses it -- but those are the maxima of the
*probe's* events, which are the first `Nevents_for_max_weight` (139) events of
the file, not the events being decayed. So the quantity exists but is not
addressable by event. Three sub-variants, increasing invasiveness:

1. **analytic, `2 -> 2` only** (recommended): compute `C_e` from `sqrt(shat)`
   and the Breit-Wigner window at the top of `_unweight_range`'s per-event loop.
   Needs `pmag` and the pole/width, no probe, no cache, no format change; falls
   back to the global bound for `n >= 3`. Perhaps 30 lines.
2. **binned in `sqrt(shat)`**: have the probe record `(sqrt(shat), maxwgt)` and
   `_combine_maxwgt` return a step function. Works for any `n`, changes the
   cache format (`_UPFRONT_CACHE_FORMAT` is already versioned for this), and
   needs enough probe events per bin -- 139 events is not enough for the
   threshold bin, where 0.5 % of the sample lives.
3. **per-event probe**: a handful of draws per decayed event before the
   accept/reject. Exact for any `n`, but pays a reshuffle per probe draw on
   every event, which is most of what the per-event bound was meant to save.

**What could go wrong.** A bound that is too *tight* for one event silently
clips it, so a per-event bound has to be provably above the weight (variant 1)
or carry A as a safety net (variants 2 and 3). And under variant 1 `nb_sigma` /
`Nevents_for_max_weight` stop reaching the mass stage's bound (they still set
the angle stages'), a user-visible change of behaviour -- though it is also what
makes `eps_m` reproducible at last, since `findings.md` measured exactly that
knob dependence as a +-40 % run-to-run scatter that never converges.

**Validation.** Byte-comparison of the *distribution* (not the file) against a
current run: the accepted mass spectrum must be unchanged, which is the whole
point -- the bound cancels. Plus `eps_m` in the log, the overflow counter at
zero, and `test_madspin -t0`.

### C. Fold `1/beta_t` into the mass proposal

**What it means concretely.** Sample the mass set from
`q'_e(m) ~ BW(m_1) BW(m_2) J(m_1,m_2;s) Zhat Zhat` instead of from the plain
Breit-Wigner. Then `w` is constant in `m` -- it *is* `A_e` -- the accept/reject
becomes a no-op, `eps_m = 1.000`, and the tail, the bound, the bound's
instability and the overflow all cease to exist. It fixes the cause.

**Is the proposal normalisable?** Yes, and its normalisation is exactly the
`A_e` of section 1, which is finite for every `sqrt(shat) > 2(m_pole - 15 Gamma)`
-- the integrand's `1/|p|` is an `sqrt(shat)`-dependent *constant*, not a
divergence in `m`. What diverges is `A_e` itself as a function of the event, and
that is a normalisation, not an integrability problem.

**Does it change the physics of the mass draw?** No. The accepted virtualities
are already distributed as `q_e w ~ BW BW J Zhat Zhat` -- that is what the
accept/reject achieves. C draws from that distribution directly instead of
reaching it by rejection. Sampling only.

**How invasive.** For `2 -> 2` this is genuinely tractable: `J = |p'|/|p|` is
closed form (section 3), so `q'_e` is a two-dimensional density on the
`(R_1, R_2)` square whose marginal in `R_1` is a 1-D quadrature and whose
conditional in `R_2` is another -- an inverse-CDF sampler on a 48x48 tabulation
built once per production event (3.3 ms, the same quadrature). Or, much simpler
and equivalent in effect: keep the Breit-Wigner draw and accept/reject against
the *analytic* per-event bound, which is option B variant 1. **For `2 -> 2`, C
and B converge**: the only difference is whether the residual factor
`J/J_max(e)` is applied by rejection (B, `eps_m = 1.36`) or by inversion (C,
`eps_m = 1.00`), and B is a tenth of the code.

For `n >= 3` C has no closed form to fold in (section 3): `J` depends on the
momentum configuration, and building a per-event proposal would mean tabulating
a numerical `J` over the mass set on every production event -- one Newton solve
per tabulation point, per event. That is a large piece of machinery, it changes
the sampler for every `spinmode` that draws virtualities, and it invalidates
every cached `max_wgt` on disk.

**What could go wrong.** The proposal has to match the weight *exactly* or the
accept/reject is testing a different quantity from the one the probe bounded --
`findings.md` already had to rule that out once. A tabulated inverse CDF has
interpolation error, and that error goes straight into the physical virtuality
distribution with no accept/reject left to correct it. That is the real risk:
C removes the safety net that the accept/reject provides.

**Validation.** The accepted virtuality spectrum against a current run,
per `sqrt(shat)` slice, at high statistics; the `2 -> 2` sampler against the
quadrature; and a `n >= 3` process (`p p > t t~ j`) if the general case is
attempted.

### Are A and B orthogonal? Should they be done together?

They overlap on the bias and are complementary in role. Both fix the clipping;
B does it by making the overflow impossible, A by making it harmless. With the
analytic per-event bound there is no overflow left for A to carry, so A becomes
a *safety net* -- which is exactly what it should be, because it is the thing
that turns "the sample is biased" from a `logger.critical` into a number. **Do
both, in that spirit: B for the cost, A so that any future bound that is too
tight degrades gracefully instead of silently.**

## 5. What none of them fixes

`A_e` varies (section 2), redraw-until-accept divides it out (any bound, section
4B), and so:

* the `sqrt(shat)` spectrum of the decayed sample is the spectrum of the
  production sample, exactly, whereas the correct PA answer reweights it by
  `A_e/<A>`;
* that is **over-population of the 348-355 GeV region by up to 6 %** and a
  growing under-population of the last GeV above threshold, unbounded as
  `sqrt(shat) -> 2 m_t` (bottom-right panel of
  `plots/per_event_mean_weight.png`);
* globally it is small -- 4.8 % of events are off by more than 1 %, 0.38 % by
  more than 5 %, and the mean absolute distortion over all events is 0.30 % --
  but it is a *shape* error sitting exactly where the off-shell treatment is
  the reason one reaches for PA in the first place;
* it is present in `PA`, worse in `PA (no jac.)` (-48 % in the first bin), and
  present with a third mechanism in `madspin`;
* it is present in the **joint** accept/reject too, which also draws until it
  accepts once per production event with `wgt*jac` in the weight.

This is the same mechanism the code already documents twice: the decay-group
fallback ("the per-particle test redraws one slot until it is accepted, which
divides `E[w_k | group]` out of the chain -- and that expectation differs
between groups") and the 13.7b argument that made redraw-until-accept legal for
the interference mode precisely *because* `<W>` there is a per-event constant.
The mass stage's `A_e` is not.

**The fix that does address it** is to stop discarding `A_e`: write the event
with weight proportional to `A_e/<A>` (the `pi_factor` hook of section 4A), or
equivalently take the `decay_output = weighted` path for the mass stage. For a
`2 -> 2` production `A_e` is the exact quadrature of section 1 -- 3.3 ms per
event, no probe, no cache, no Monte Carlo. For `n >= 3` it would have to be
estimated, and a noisy `A_e` in the numerator is its own bias, so that case
needs thought.

Cost: the sample is weighted, with weights of rms 0.95 % and a range of
[0.94, 1.51] over this sample -- far gentler than `decay_output = weighted`,
but no longer unit. Keeping it unit would need an accept/reject on `A_e`
itself, and `A_e` is unbounded, so that route reduces to option A at the
production-event level.

I am **not** recommending this be implemented on the strength of this
assessment: it changes what MadSpin's PA output *is*, it applies to the joint
path as much as the sequential one, and the physical size (0.30 % mean, 5 % in
2 % of the sample) is worth a decision by someone who owns the physics, not a
patch. It is recorded here because the three options on the table were framed
as if one of them addressed it, and none does.

## 6. Recommendation

1. **B first, in its analytic `2 -> 2` form.** It is the smallest change with
   the largest effect: `eps_m` 3.37 -> 1.36 with a bound that is provably above
   the weight, no probe, no cache format change, no run-card knob, and it makes
   the overflow -- `findings.md`'s "the sample is biased" -- structurally
   impossible for the process class that covers `t t~`, `W W`, `Z Z` and most of
   MadSpin's use. It is also the one whose correctness argument is airtight: the
   bound cancels out of the accepted distribution, so the output is unchanged by
   construction and the validation is a distribution comparison rather than a
   physics argument. Keep the existing global bound as the `n >= 3` fallback.
2. **A next, as the safety net.** Small, uses an existing hook, and turns the
   remaining overflow (10 per 100 000 on the `n >= 3` fallback path) from a
   silent bias into a weight. Do not do A *instead* of B: on its own it fixes
   0.016 % of the normalisation and nothing else.
3. **C: not now, and for `2 -> 2` not ever** -- B variant 1 already collects
   nine tenths of its benefit for a tenth of the code and keeps the accept/reject
   as a safety net. For `n >= 3` C is the only thing that would remove the tail
   at its cause, but there `J` has no closed form and the change is large. If
   `n >= 3` efficiency becomes the complaint, revisit C then; today the tail is a
   `2 -> 2` threshold effect and B handles it.
4. **Separately, and above all three in importance if it is real to the
   physics: the `A_e` normalisation of section 5.** It is the only measured
   *bias* in the mass stage that survives every one of A, B and C, and it is not
   specific to the sequential scheme. It deserves its own decision.

## Reproducing

```bash
export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"   # MadSpin
# one PA sequential run to produce the Zhat tables and the bound
python3 -O MadSpin/madspin card_pa.dat          # set ms_dir, unweighting sequential
# and one decay_output = weighted run for the end-to-end cross-check
python3 -O MadSpin/madspin card_wgt.dat

export PATH="$HOME/.pyenv/versions/mg-3.12/bin:$PATH"   # numpy / matplotlib
python3 doc/madspin_pa_mass_stage/jacobian_analytic.py \
    --json doc/madspin_pa_mass_stage/data/jacobian_analytic.json
python3 doc/madspin_pa_mass_stage/per_event_weight.py \
    --events <production .lhe.gz> --out <dir> \
    --ztables <ms_dir>/max_wgt_sequential_pa --pool 6000 --draws 400
python3 doc/madspin_pa_mass_stage/analyse_per_event.py \
    --data <dir> --plots doc/madspin_pa_mass_stage/plots \
    --out doc/madspin_pa_mass_stage/data \
    --ztables <ms_dir>/max_wgt_sequential_pa
python3 doc/madspin_pa_mass_stage/weighted_crosscheck.py \
    --weighted <events_decayed.lhe.gz> --out <dir> \
    --ztables <ms_dir>/max_wgt_sequential_pa
```

No shipped code was modified. `per_event.npz` (6000 events) and
`weighted_crosscheck.json` are committed; the 50 000-event weighted LHE is not.
