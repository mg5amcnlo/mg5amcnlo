# `A_e`: what correcting it costs, and what it changes

Measurement, not a merge.  `doc/madspin_pa_mass_stage/bound_design.md` section 5
established the effect and declined to recommend acting on it; this is the
evidence a decision needs.  Section 7 is the recommendation and it is
deliberately the shortest section.

Process throughout: `p p > t t~` at 6.5+6.5 TeV, LO, **200 000** unweighted
production events, both tops fully leptonic, `BW_cut = 15`,
`unweighting = sequential`, `nb_core = 8`, seed 42 -- the same sample, statistics
and observable as `MadSpin/validation/mt_lineshape/`, whose *measured* replica
noise floor is the bar used below.

---

## 0. The answer in one paragraph

Correcting `A_e` costs **77 microseconds of CPU per event** for a `2 -> 2`
production -- **0.9 % of a MadSpin run**, not the 3.3 ms an earlier measurement
reported, and the factor 43 does not come from PR #377's kernel (section 1).
For the offshell spinmodes there is no closed form and it costs **7 % of the run**
(section 1.3).  It changes the `sqrt(shat)`
spectrum of the decayed sample by **-5 to -9 % between 348 and 355 GeV** and by
**+14 to +23 % in the first GeV above threshold**, in all three modes, with the
`PA (no jac.)` mode the worst at **-38 %** in that first GeV (section 3).  It
moves `<m(t)>` by **+0.6 MeV** (`PA`), **+1.2 MeV** (`PA no jac.`) and
**+1.5 MeV** (`madspin`), the same sign on `t` and on `t~` in every mode -- real,
and **16 to 6 times smaller than the +-10.1 MeV** two runs of the same code
differ by at this statistics (section 4).  It leaves the cross section exactly
unchanged *if* the normalisation constant is the sample's own mean of `A_e`, and
that is the one piece the prototype does not solve (section 5).  It makes the
output weighted, with an rms of 1.2-1.7 % and a cost of **0.014 % of the
statistics** (section 4.4).  The offshell `Tr(rho_off)/|M|^2_on` factor
**compensates, weakly**: +1.7 % where the truncation is -10 % (section 2.2).

---

## 1. Cost

### 1.1 The 3.3 ms was not the kernel's fault, and the kernel does not fix it

The brief for this work assumed that `doc/madspin_pa_mass_stage`'s 3.3 ms per
event came from the quadrature calling the shipped `_production_jacobian_for`
(which rebuilds an `Event` from a string and runs a full `reshuffle_production`
per evaluation), and that redoing it on PR #377's `Event.mass_shuffle_jacobian`
kernel would collapse it.  **Neither half is right, and the second one is
backwards.**

`jacobian_analytic.analytic_A` never called `_production_jacobian_for`.  For a
`2 -> 2` production it already used the closed form `J = |p'|/|p|`, two square
roots.  The 3.3 ms was the *interpreter*: 48x48 = 2304 nodes in a plain Python
double loop.

Measured here on the same events (`ae_kernel.py`, `data/cost.json`), 48x48 nodes
throughout:

| `A_e`, one production event | ms | vs the closed form |
|---|---|---|
| Python loop, closed-form `J` (what was measured before) | **2.81** | 46x |
| **numpy, closed-form `J`** | **0.061** | **1** |
| Python loop, `Event.mass_shuffle_jacobian` (the #377 kernel) | 20.1 | 330x |
| Python loop, `_production_jacobian_for` (shipped, extrapolated from 8x8) | 140 | 2300x |

and per single jacobian evaluation: `_production_jacobian_for` **67.0 us**, the
#377 kernel **7.59 us** (8.8x faster, consistent with the 5-8 us that commit
reported), the `2 -> 2` closed form **2.14 us** as a Python call and **26 ns**
inside a vectorised grid.

So: **the kernel is 8.8x cheaper than the shipped jacobian and irrelevant to
this quadrature**, which for `2 -> 2` never needed either.  Where the kernel
does earn its place is the Monte Carlo estimator that `n >= 3` needs -- section
6.1 -- where it is the difference between 9 us and 76 us per draw.

Batched over a whole sample the cost falls again, to **30-39 us per event**
(`measure_ae.py` does the whole 200 000-event file as three numpy expressions:
38.6 us/event for `PA`, 30.8 for `PA no jac.`, 32.9 for `madspin`'s `R = 1`
part).

### 1.2 Quadrature order

The 48x48 grid is not free choice; measured against a 96x96 reference over 3000
events:

| nodes | us/event | max relative error | median |
|---|---|---|---|
| 8x8 | 27 | 9.0e-3 | 7.0e-5 |
| 16x16 | 25 | 1.4e-3 | 5.6e-6 |
| 24x24 | 28 | 7.1e-4 | 3.2e-7 |
| **48x48** | **45** | **1.2e-4** | **3.4e-11** |
| 64x64 | 61 | 5.8e-5 | 6.8e-14 |

The worst case is always the threshold slice, where the integrand's `m1 + m2 ->
sqrt(shat)` edge is what the quadrature has to resolve -- and where the effect
being corrected is 10-100 %.  A 1.2e-4 error against a 20 % effect is not the
limiting term; 24x24 would do and 16x16 probably would.

### 1.3 In context

`data/timing.json`.  50 000 production events, `nb_core = 8`, **warm `ms_dir`**
so no pair differs by a max-weight probe or a first-generation decay pool.  The
honest denominator is the CPU a run spends per event, not its wall clock -- an
8-core run divides the wall clock by 8 and would flatter the number.

A MadSpin `PA`/sequential run of this process costs **8.35 ms of CPU per event**
(417.7 s over 50 000, summed over the eight workers); `madspin`/sequential costs
**9.55 ms**.

| | added per event | as % of the run's CPU |
|---|---|---|
| `PA`, exact quadrature (`2 -> 2`) | **76.9 us** | **0.92 %** |
| `madspin`, Monte Carlo, 8 draws | 0.70 ms | 7.3 % |
| `madspin`, Monte Carlo, 24 draws | 2.24 ms | 23.4 % |

The 76.9 us is the shipped method micro-benchmarked on 300 real production
events with its caches warm -- 16 us above the bare 61 us numpy quadrature, the
rest being the two `Zhat` evaluations and the window arithmetic.  The two
`madspin` rows give **90 us per offshell free draw**, which is what a mass set
costs when its weight carries `Tr(rho_off)` and there is no arithmetic-only path
to it (section 6.1).

End to end, the two independent `PA` pairs measured **+3.0 %** and **+4.7 %** of
wall clock and **+4.9 %** and **+8.3 %** of CPU -- both above the 0.92 % the
quadrature accounts for, which is the run-to-run scatter of this benchmark
(the two baselines alone differ by 0.7 % of CPU, and the decay pools are redrawn
between runs).  Take 0.9 % as the cost and a few percent as the noise on
measuring it.

**One trap worth recording**, because it is exactly the sort of thing that makes
a 45 us method look like a 450 us one: `numpy.polynomial.legendre.leggauss(48)`
is a root-find and costs **379 us**, six times the quadrature it sets up.  The
first `PA` pair above was measured before the nodes were cached and shows it
(+8.3 % of CPU against +4.9 % after).

---

## 2. `A_e`, sliced in `sqrt(shat)`, for all three modes

`measure_ae.py`, exact quadrature over the whole 200 000-event sample, each mode
with **its own** `Zhat` tables read from that mode's own run cache (they are very
different: the offshell fit's linear coefficient is 4.52 against `PA`'s 0.56).

`A_e` relative to its own plateau (the `> 800 GeV` mean):

| `sqrt(shat)` [GeV] | events | % of sample | `PA` | `PA (no jac.)` | `madspin` |
|---|---|---|---|---|---|
| 346 - 347 | 131 | 0.066 | **1.2307** | **0.6233** | **1.1289** |
| 347 - 348 | 235 | 0.117 | 0.9640 | 0.7625 | 0.9112 |
| 348 - 350 | 689 | 0.344 | **0.9432** | 0.8672 | **0.9045** |
| 350 - 352 | 875 | 0.438 | 0.9521 | 0.9252 | 0.9207 |
| 352 - 355 | 1608 | 0.804 | 0.9636 | 0.9561 | 0.9376 |
| 355 - 360 | 3167 | 1.583 | 0.9759 | 0.9787 | 0.9555 |
| 360 - 370 | 7543 | 3.772 | 0.9882 | 0.9951 | 0.9743 |
| 370 - 380 | 8468 | 4.234 | 0.9950 | 0.9999 | 0.9858 |
| 380 - 400 | 18196 | 9.098 | 0.9976 | 1.0000 | 0.9913 |
| 400 - 450 | 42479 | 21.24 | 0.9990 | 1.0000 | 0.9955 |
| 450 - 500 | 32556 | 16.28 | 0.9996 | 1.0000 | 0.9975 |
| 500 - 600 | 39798 | 19.90 | 0.9998 | 1.0000 | 0.9987 |
| 600 - 800 | 30399 | 15.20 | 0.9999 | 1.0000 | 0.9995 |
| 800 - 1200 | 11784 | 5.892 | 1.0000 | 1.0000 | 1.0000 |
| > 1200 | 2072 | 1.036 | 1.0000 | 1.0000 | 1.0002 |

Whole-sample summary:

| | `PA` | `PA (no jac.)` | `madspin` |
|---|---|---|---|
| `<A_e>` | 0.955154 | 0.955753 | 0.956769 |
| relative sd | **1.18 %** | **1.65 %** | **1.42 %** |
| range | [0.9012, 2.4436] | [0.4943, 0.9579] | [0.8661, 2.2168] |
| events beyond 1 % | 4.93 % | 3.48 % | 8.03 % |
| events beyond 5 % | 0.47 % | 1.12 % | 1.67 % |
| mean \|`A_e/<A>` - 1\| | 0.31 % | 0.41 % | 0.63 % |

This reproduces `bound_design.md` section 2 -- 0.955 global, `PA (no jac.)` 48 %
low in the first bin (0.6233/1.2 of the plateau is the same statement) -- at 33x
the statistics, which is why the `PA` range now reaches 2.44 rather than 1.51.
**`A_e` has no maximum**; a bigger sample finds a bigger one, because it diverges
like `1/beta_t`.  That matters in section 6.3.

**The three modes are not variations on one curve.**  `PA` overshoots the
plateau in the last GeV and undershoots below it; `PA (no jac.)` only ever
undershoots, and by far the most (a factor 1.6); `madspin` sits between them but
its deficit reaches further out in `sqrt(shat)` -- it is still 0.5 % low at
500 GeV where `PA` is 0.02 % low.  That last point is the offshell `Zhat` being
steep: the truncated window costs more when the density it truncates is steeper.

### 2.2 Does the offshell ratio compensate or compound?  It compensates, weakly

This was the open question.  `madspin`/`full` carry
`R = Tr(rho_off)/|M_prod|^2_on` in the mass-stage weight, measured elsewhere at
`1.00000 +- 0.0119` *averaged over a whole sample* -- an average dominated by the
80 % of events above 400 GeV.

Measured here in the last GeV, where the effect lives.  No new probe was needed:
a run with the prototype on writes `A_e` (with `R`) onto every event weight, and
`measure_ae.py --mode madspin` gives the same integral with `R = 1` exactly, so
their ratio is `R` averaged over the mass sets the stage actually weights
(`ratio_R.py`, 24 free draws per event, 4.8e6 offshell mass sets in all).

| `sqrt(shat)` [GeV] | `<R>_w` |
|---|---|
| 346 - 347 | 1.0063 +- 0.0134 |
| 347 - 348 | **1.0169 +- 0.0074** |
| 348 - 350 | **1.0110 +- 0.0030** |
| 350 - 352 | 1.0069 +- 0.0019 |
| 352 - 355 | 1.0043 +- 0.0010 |
| 355 - 360 | 1.0021 +- 0.0005 |
| 360 - 370 | 1.0001 +- 0.0002 |
| 370 - 400 | 0.9991 +- 0.0001 |
| 400 - 800 | 0.9995 - 1.0003 |
| whole sample | **0.999956 +- 0.000049** |

**It compensates.**  `R` is 1.7 % *high* exactly where the truncated window makes
`A_e` 10 % *low*, and the sign is consistent across every slice below 360 GeV.
But it is a 1.7 % correction to a 10 % effect: it removes a sixth of it and
leaves `madspin`'s threshold deficit the largest of the three modes anyway (the
`madspin` column of the table above already has `R` folded in).  The whole-sample
average confirms the `1.00000 +- 0.0119` result to five decimals and, exactly as
suspected, says nothing about the last GeV.

Two caveats.  The estimator is Monte Carlo, so the per-slice errors above are
real; the two smallest slices are 0.5 and 2.3 sigma from 1 individually and it is
the *pattern* across six consecutive slices that carries the signal.  And `R` was
folded into the `madspin` column by interpolating these slice means in
`ln sqrt(shat)`, i.e. treating `R` as a function of `sqrt(shat)` alone -- which it
is not exactly, though its residual event-to-event spread at fixed `sqrt(shat)`
is well inside the Monte Carlo noise here.

---

## 3. What correcting it does to the `sqrt(shat)` spectrum

The correction multiplies the written event by `A_e/<A>` and **changes no random
decision**.  That is not an argument, it is measured: over 3000 events the
decayed LHE of a `mass_normalisation` run is byte-identical to the baseline's
except in the weight field, and the weights it carries reproduce the offline
quadrature to 2.6e-8 (the LHE's own weight precision).  So reweighting a baseline
run *is* the corrected run, and every number in sections 3-5 is computed that
way, on the real 200 000-event decayed samples.

Relative change of the decayed sample's `sqrt(shat)` spectrum:

| `sqrt(shat)` [GeV] | % of sample | `PA` | `PA (no jac.)` | `madspin` |
|---|---|---|---|---|
| 346 - 347 | 0.066 | **+23.3 %** | **-37.5 %** | **+13.6 %** |
| 347 - 348 | 0.118 | -3.4 % | -23.6 % | -8.3 % |
| 348 - 350 | 0.345 | **-5.5 %** | -13.1 % | **-9.0 %** |
| 350 - 352 | 0.438 | -4.6 % | -7.3 % | -7.4 % |
| 352 - 355 | 0.804 | -3.4 % | -4.2 % | -5.7 % |
| 355 - 360 | 1.583 | -2.2 % | -1.9 % | -3.9 % |
| 360 - 370 | 3.771 | -1.0 % | -0.3 % | -2.0 % |
| 370 - 380 | 4.234 | -0.3 % | +0.2 % | -0.8 % |
| 380 - 400 | 9.098 | -0.0 % | +0.2 % | -0.3 % |
| 400 - 500 | 37.52 | +0.1 % | +0.2 % | +0.2 % |
| > 500 | 42.03 | +0.2 % | +0.2 % | +0.5 % |

`bound_design.md` predicted "over-population of 348-355 GeV by up to 6 % and a
growing under-population of the last GeV".  Confirmed for `PA` (-5.5 % is the
correction, i.e. the current sample over-populates that region by 5.8 %), and
the prediction understates the other two modes: `PA (no jac.)` is off by 13-38 %
over the first 4 GeV and `madspin` by 8-9 %.

The counterpart is a **+0.2 % excess everywhere above 400 GeV** (+0.5 % for
`madspin`) -- which is where 80 % of the sample lives, and is where the
normalisation the threshold region loses has to go.

---

## 4. What it does to everything else

### 4.1 `m(t)` and `m(t~)`

Same 200 000 events, same observable and same 70-bin scheme as
`MadSpin/validation/mt_lineshape/RESULTS.md`, so these numbers can be read
straight against that campaign's measured noise floor -- `chi2` **139.2/138** and
**135.7/138** for two runs of one scheme at different seeds, and **+-10.1 MeV**
per resonance / **+-7.1 MeV** combined on the mean.

| mode | `<m(t)>` shift | `<m(t~)>` shift | same sign? | `chi2` a rerun would see, /138 |
|---|---|---|---|---|
| `PA` | **+0.62 +- 0.12 MeV** | **+0.56 +- 0.11 MeV** | yes | **0.020** |
| `PA (no jac.)` | **+1.22 +- 0.11 MeV** | **+1.06 +- 0.11 MeV** | yes | **0.052** |
| `madspin` | **+1.50 +- 0.14 MeV** | **+1.73 +- 0.10 MeV** | yes | **0.075** |

Two readings, and both are true.

**It is real.**  The errors quoted are *paired* -- the two histograms are the
same events, so the error on the shift is that of `(f_e - 1)(m_e - <m>)` and
vanishes when the correction is the identity.  On that error every shift is 5-17
sigma, and, decisively, **both resonances move the same way in all three modes**.
`RESULTS.md` made the point that only a same-sign move is evidence (it found a
+41.7 MeV single-resonance shift there that was -22.0 MeV on the other and was a
fluctuation).  Six same-sign entries out of six is not a fluctuation.  The `rms`
of the lineshape moves too, by +0.4 to +1.4 MeV, in the same direction in every
mode.

**Nobody would ever see it.**  The last column is the `chi2` that comparing a
corrected run against a current one would produce from this shift alone, on the
70-bin scheme, with the errors such a comparison uses.  It is **0.02 to 0.08
against a noise floor of 138**, i.e. three to four orders of magnitude below the
point at which two runs of the *same* code differ.  Since that `chi2` grows
linearly with the statistics, reaching even a 3-sigma excess over 138 dof
(`Delta chi2 = 50`) needs `5e8` events for `PA` and `1.3e8` for `madspin`.  The
largest single 70-bin move anywhere is **-0.47 % +- 0.15 %** (`madspin`, the
152 GeV bin) against a per-bin statistical error that runs from 0.69 % at the
peak to 8.0 % in the sparsest bin.

The two statements are consistent and both belong in the decision: the
correction fixes a **real, deterministic** distortion of the lineshape that is
**far below** what this generator's users can resolve on the lineshape.  It is
*not* far below what they can resolve on the `sqrt(shat)` spectrum near
threshold, which is where it should be argued (section 3).

### 4.2 Cross section

**Exactly unchanged, by construction** -- measured at `-4.4e-16`, `0.0` and
`+2.5e-6` relative for the three modes, the first two being floating-point zero
and the third the residual of a normalisation constant supplied to 6 digits.

That is a property of the *choice of normaliser*, not of the correction, and it
is the one thing the prototype does not solve; see section 5.

### 4.3 It is exactly a reweighting, and nothing else

`efficiency`, `eps_m`, the overweight counters and every drawn quantity are
untouched (`PA`: 0.2222 with the correction on against 0.2222 off, same trials).
The mass-stage bound is *not* affected either, and this is worth stating plainly
because the two questions are constantly confused: the accepted mass density is
`q_e(m) . min(1, w/C)` and any `C >= max w` gives `q_e w`, so the bound cancels
out of the sample.  `A_e` is a *between*-event normalisation and no choice of
bound -- global, per-event, PR #377's or any other -- touches it.

### 4.4 The weight distribution it introduces

| | `PA` | `PA (no jac.)` | `madspin` |
|---|---|---|---|
| rms of `A_e/<A>` | 1.18 % | 1.65 % | 1.42 % |
| range | [0.944, 2.558] | [0.517, 1.002] | [0.905, 2.317] |
| 0.1 / 99.9 percentile | 0.944 / 1.0022 | 0.741 / 1.0022 | -- |
| **`N_eff/N`** | **0.99986** | **0.99973** | **0.99980** |

The output stops being unit-weighted.  What that costs in statistics is
**0.014 % to 0.027 %** -- for comparison, `decay_output = weighted` carries the
full matrix-element weight and costs orders of magnitude more.  The distribution
is one-sided and very tight: for `PA`, 95 % of events sit in [0.990, 1.0022].

The real cost is not statistical, it is that downstream tools which assume unit
weights see a file that is not.  Note this already happens today: the overweight
safety net (PR #375) writes non-unit weights on roughly 1 event in 10 000.  The
difference is one of degree -- 1e-4 of the sample against all of it.

---

## 5. The normalisation constant, which the prototype does not solve

`A_e/<A>` needs `<A>`, and `<A>` is a property of the whole production sample.
The prototype takes it as the option's value and every number above used the
sample's own mean, which is why the cross section is unchanged to
floating point.  A real implementation has three routes and they are not
equivalent:

| route | error on `<A>` | cost |
|---|---|---|
| the max-weight probe's 80 events | **0.13 %** (`PA`), 0.18 %, 0.16 % | free |
| a 2000-event pre-pass | 0.026 %, 0.037 %, 0.032 % | 0.08 s |
| a full pre-pass over the input file | **0** | 7.7 s on 200 000 events |

The middle column is `sd(A)/(<A> sqrt(k))` and it lands **straight on the cross
section**.  Using the probe's 80 events would inject a 0.13 % random shift into
`sigma` in order to remove a shape distortion whose mean is zero -- a bad trade,
and one that would show up as an irreproducible cross section run to run.  The
full pre-pass is 7.7 s of numpy on 200 000 events (the quadrature is the same
one, batched) and makes the shift exactly zero; the file has to be walked twice,
which under `nb_core > 1` means once before the fork.

**This is the piece that would need designing.**  It is not hard, but "which
`<A>`" is a decision with a visible consequence and it should not be made by
default.

---

## 6. The cases that need thought

### 6.1 `n >= 3`: how noisy is too noisy?

`bound_design.md` says "a noisy `A_e` in the numerator is its own bias".  **That
is not right as stated, and getting it right changes the answer.**

`A_e` enters the weight *linearly*.  If `A_hat_e` is an unbiased estimator built
from draws **independent of the accepted chain**, then `E[A_hat_e] = A_e` and the
reweighted estimator of any observable is unbiased.  Noise costs variance, not
bias.  There are exactly two ways it becomes a bias, and both are avoidable:

* **reusing the redraw loop's own trials.**  Those are a *stopped* sequence whose
  last member is accepted, so their mean is not `A_e` and, worse, they are
  correlated with the event that gets written -- which turns noise into bias
  directly.  The prototype therefore runs the estimator on its **own RNG
  stream**, forked once, and restores the main stream afterwards, so the sample
  it reweights is bit-for-bit the sample it would have written anyway.
* the denominator `<A_hat>` is a ratio estimator, so it has an `O(1/N)` bias.
  At `N = 2e5` and a few percent spread that is `~1e-9`.

So the criterion is the statistics loss, `N_eff/N ~ 1/((1 + sd_A^2)(1 + s^2))`
with `s` the estimator's relative noise.  `measure_ae.py` computes `E_q[w^2]` on
the same quadrature grid, so the *within-event* spread of `w` -- the thing a
Monte Carlo estimator has to average down -- is known exactly rather than
guessed:

| within-event `sd(w)/A_e` | `PA` | `PA (no jac.)` | `madspin` |
|---|---|---|---|
| median | 1.0 % | 0.0 % | 10.5 % |
| 95th percentile | 11.9 % | 3.8 % | 11.8 % |
| max | 100 % | 56 % | 89 % |

and the resulting cost of `k` free draws, *on top of* the 0.014-0.027 % the
correction costs anyway:

| draws `k` | extra statistics loss, `PA` | `madspin` |
|---|---|---|
| 4 | 0.105 % | 0.292 % |
| 8 | 0.053 % | 0.146 % |
| 16 | 0.026 % | 0.073 % |
| 32 | 0.013 % | 0.037 % |

**"Too noisy" is a long way away.**  Even four draws costs a third of a percent
of the statistics.  A sensible answer is `k = 8-16`, at which the estimator's
noise is 0.4 % (`PA`) to 2.6 % (`madspin`) per event and its statistics cost is
0.03-0.15 %.

Does the kernel make that affordable?  For the **PA-like** weight, yes and
comfortably: a free draw through `Event.mass_shuffle_jacobian` costs **9.0 us**
(measured: 0.29 ms for 32 draws), so `k = 16` is 144 us per event -- three times
the `2 -> 2` quadrature, still small.  Through `_production_jacobian_for` the
same 16 draws would be 1.2 ms, which is where it would start to hurt.  **This is
the one place PR #377's kernel is load-bearing for this work.**

For the **offshell** weight the kernel does not help at all, because every draw
needs a production reshuffle *and* a production density matrix -- there is no
arithmetic-only path to `Tr(rho_off)`.  Measured directly (section 1.3): 24 draws
per event cost `+2.06 ms/event` of CPU, i.e. **86 us per offshell free draw**,
ten times the kernel's.  `k = 8` would be 0.7 ms/event.

Two things this does *not* cover.  `J`'s within-event spread was measured on a
`2 -> 2` production; for `n >= 3` `J` is not a function of the mass set alone
(`jacobian_analytic.py` measured a 14 % spread at fixed masses across eight
three-body configurations), so the *per-event* variance is a property of the
event and could be larger -- the table above is a `2 -> 2` measurement being used
as a guide, not an `n >= 3` measurement.  And no `n >= 3` production was run
end to end here.

### 6.2 The joint path

`bound_design.md` says the joint path's mass proposal is "the decay pool's
virtuality, untruncated window, internal retry", different from the sequential
stage's budget-capped `_draw_mass_value`.  **On the code as it stands that is
wrong.**  `get_onshell_evt_and_wgt`'s offshell branch draws with

```python
full_dqrts = production.sqrts
for pdg in decays:
    for dec in decays[pdg]:
        full_dqrts, jac_dec = self._draw_offshell_mass(pdg, dec, full_dqrts)
```

and `_draw_offshell_mass` is a two-line wrapper around the *same*
`_draw_mass_value(pdg, budget)` the sequential mass stage calls, with the budget
chained down in exactly the same way.  **Same proposal, same truncated window,
same cause.**

The joint accept/reject also redraws until it accepts and writes exactly one
event per production event, so it divides out a per-event normalisation too.  Is
it the *same* `A_e`?  The argument says yes: the joint per-event normalisation is
`<W>_e`, the expectation of `wgt*jac` over both the mass draw and the decay-pool
draw, and the sequential decomposition is built so that the product of its stage
weights reproduces `W` -- with `Z_k(m_k)` *defined* as the angle stage's
normalisation at a given virtuality (`_build_z_tables`).  Integrating the mass
stage's weight against `prod Z_k` is exactly what `A_e` is, and it is what
`measure_ae.py` computes.

**This is an argument, not a measurement, and I did not measure it.**  The
prototype hooks the sequential path only.  What would settle it is a direct
measurement of `<W>_e` by free joint draws per production event, compared against
the `A_e` measured here; the two should agree up to the accuracy of the fitted
`Zhat` (about 0.5 %).  Until then, the safe statement is: the *mechanism* is
demonstrably identical, the correction is demonstrably needed there too, and the
*quantity* is very probably the same one.

### 6.3 Keeping unit weights

An accept/reject on `A_e` itself would need a bound on `A_e`, and `A_e` has none
-- it diverges like `1/beta_t`.  The measured maximum simply grows with the
sample: **1.512** over 6000 events (`bound_design.md`), **2.090** over 20 000,
**2.444** over 200 000.  Any "bound" is a sample maximum that the next run
exceeds.

Suppose one used it anyway, at the 200 000-event maximum.  The keep rate is
`<A>/A_max = 1/2.558 = 39 %`: **61 % of the sample thrown away** to remove a
distortion whose mean is zero and whose mean absolute size is 0.31 %.  Against
the weighted route's 0.014 % statistics cost that is a factor of **4400** worse,
and it would still be biased, because the events it clips are exactly the
threshold events the correction exists for.

The duplication variant (write `floor(f)` copies plus one more with probability
`frac(f)`) is the same thing in a different order: `f < 1` for 99.8 % of events
under `PA`, so it reduces to "drop the event with probability `1 - f`", changes
`n_written`, and costs the same statistics.

So the brief's own conclusion stands and the measurement sharpens it: **keeping
unit weights reduces to option A at the production-event level** -- carry the
excess on the weight -- **i.e. to being weighted anyway, having first thrown away
most of the sample.**

---

## 7. Recommendation

**What I would do: not ship this as a default, and not drop it either.  Ship the
measurement and offer the correction as an option that is off.**

The case for correcting it is real and the size is now known: a
**5-9 % over-population of 348-355 GeV and a 14-38 % error in the first GeV
above threshold**, in every mode, deterministic, and sitting exactly where the
off-shell treatment is the reason to reach for `PA` or `madspin` in the first
place.  It costs **77 us of CPU per event -- 0.9 % of a run** -- for the `2 -> 2`
processes that cover `t t~`, `W W`, `Z Z` and most of MadSpin's use, and
**0.014 % of the statistics**.  The offshell spinmodes have no closed form and
pay **7 %** of the run for an eight-draw estimator instead.

The case against making it the default is equally real.  It changes what
MadSpin's output *is*, in three ways: the sample becomes weighted; the
`sqrt(shat)` spectrum of the decayed sample stops being the production sample's
spectrum, which is a property users may well be relying on; and the run acquires
a normalisation constant that has to come from somewhere (section 5).  None of
those is a bug being fixed -- they are a definition being changed -- and 0.6 to
1.7 MeV on a lineshape that two runs of the same code disagree about by 10 MeV
is not the argument that forces it.

Concretely:

1. **Take the measurement.**  Sections 2 and 3 are the numbers the earlier
   assessment was missing, including the direct answer that the offshell ratio
   compensates by a sixth rather than compounding.
2. **Keep the option off.**  The prototype is 200 lines and it works end to end
   in all three modes; whether it becomes supported is a physics decision.
3. **If it is ever turned on, do the full pre-pass for `<A>`** (7.7 s per
   200 000 events), never the probe's 80 events, or the correction buys a shape
   fix at the price of a 0.13 % irreproducible cross section.
4. **Measure the joint path's `<W>_e` before extending it there** (section 6.2).
   The mechanism is identical; the quantity is an argument.
5. **Do not attempt the unit-weight route** (section 6.3).

---

## 8. What is prototype scaffolding

In `MadSpin/interface_madspin.py`, all of it under `mass_normalisation`, which
defaults to `0.0` = off.  With it off, no code added here runs and the output is
unchanged (verified: `p_pa` and `b_pa` differ only in the weight field, and the
efficiency and trial counts are identical).

| piece | state |
|---|---|
| `_mass_shuffle_data` | a straight extraction of eight lines that were inline in `_mass_stage_bound_compute`, which now calls it.  Not scaffolding. |
| `_mass_normalisation_quad` | correct, and validated against the offline quadrature to 2.6e-8 -- but it handles **only** a `2 -> 2` production with exactly two drawn slots and it imports numpy lazily.  A real version needs the general case and a decision about numpy. |
| `_mass_normalisation_mc` | correct.  The RNG fork is seeded from a **fixed constant**, not from the card's seed -- fine for a prototype, wrong for a shipped one. |
| `_mass_normalisation_factor` | takes `<A>` from the option.  **This is the scaffolding**: section 5. |
| the `stats['ae_factor']` hook and `_report_ae_normalisation` | the mechanism is the one PR #375 established and it is sound.  It reaches `full_evt.wgt` and every `parse_reweight()` entry through the same multiplication as the branching ratio. |
| joint path | **not hooked at all.**  Section 6.2. |
| `fixed_order`, `pure_interference`, BR equalization | not considered.  The factor rides `carry`, which those paths also use, so it should compose -- untested. |
| tests | none added.  `tests/unit_tests/madspin/test_madspin.py -t0` was run to confirm nothing existing broke. |

---

## 9. What this does not cover

* **No `n >= 3` production was run.**  Section 6.1's noise numbers are a `2 -> 2`
  measurement used as a guide; `J`'s per-event variance for `n >= 3` was not
  measured, and the 14 % spread at fixed masses that `jacobian_analytic.py` found
  says it could be larger.
* **The joint path was not measured** (section 6.2), only argued.
* **One process.**  Everything is `p p > t t~` at one energy with one decay
  chain.  `W W` and `Z Z` have much wider windows relative to their thresholds
  and the effect should be far smaller; that was not checked.
* **`R` was folded into the `madspin` column as a function of `sqrt(shat)`**, from
  slice means with real errors.  Its residual event-to-event spread at fixed
  `sqrt(shat)` is inside the Monte Carlo noise here but was not separately
  bounded.
* **The lineshape comparison against the noise floor is arithmetic, not a second
  pair of runs.**  It is exact -- the correction changes no random decision, which
  was verified byte for byte -- but nobody ran a corrected campaign and a
  baseline campaign at two seeds and compared them.
* **A pre-existing failure was hit twice** and worked around rather than fixed:
  on 8 cores the `spinmode madspin` runs died in the parallel decay-pool refill
  (`_open_refill_slice`, "refill pool ... is missing for worker 0").  Raising
  `decay_event_mult` avoids it.  It is unrelated to anything here -- the option
  was off, and the `PA` runs were fine -- but it is a real intermittent bug.

## 10. Where this contradicts the brief

Three places, all load-bearing:

1. **"T55 measured the quadrature at 3.3 ms -- but it used the shipped
   `_production_jacobian_for`... Re-do the quadrature on the kernel."**  It did
   not use it; it used the `2 -> 2` closed form, and the 3.3 ms was pure Python
   loop overhead.  Redoing it *on the kernel* makes it **seven times worse**
   (20.1 ms).  The right fix was vectorising the closed form: 0.061 ms.  The
   kernel's real contribution to this work is the `n >= 3` Monte Carlo estimator
   (section 6.1).
2. **"a noisy `A_e` in the numerator is its own bias".**  Not as stated.  An
   unbiased estimator built from draws independent of the accepted chain leaves
   the reweighting unbiased; noise costs variance only.  The bias comes from
   *correlation* with the accepted chain, which is a design constraint on where
   the draws come from, not a limit on how many you can afford (section 6.1).
3. **"the joint mass proposal differs (the decay pool's virtuality, untruncated
   window, internal retry)".**  On the current code the joint path calls
   `_draw_offshell_mass`, which is `_draw_mass_value` with the same budget chain
   -- the same truncated window (section 6.2).  This statement is in
   `bound_design.md` section 2 as well and should be corrected there.
