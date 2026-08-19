# Why the PA mass/virtuality stage costs `eps_m = 3.5` and the no-jacobian one `1.10`

Investigation of the `eps_m` column of `doc/madspin_unweighting_efficiency/table.tex`.
Process `p p > t t~`, both tops decayed leptonically, 100 000 events, one
production sample and one MadSpin seed throughout, `unweighting = sequential`.

| mode | `eps_m` (campaign) | `eps_m` (this reproduction) | bound `C` = `maxwgts[0]` | `<w>` | `C/<w>` |
|---|---|---|---|---|---|
| `PA` (`density_keep_jacobian=True`) | 3.50 | **3.76** | 3.5928 | 0.9551 | **3.762** |
| `PA (no jac.)` | 1.10 | **1.10** | 1.0540 | 0.9556 | **1.103** |
| `madspin` (offshell) | 3.31 | **3.33** | 3.2068 | 0.9563 | **3.353** |

**Verdict: not a bug. The premise is wrong -- but only half wrong, and in an
interesting way.** The reshuffling jacobian is *not* "small" in the sense that
matters. It is extraordinarily *narrow in the bulk* (median 1.000005, 90 % of
draws within 2 % of 1) and it has a clean power-law upper tail with index
`a = 3.6` that reaches 12 in 375 715 draws. An accept/reject is priced by
`C/<w>`, i.e. by the tail, so the bulk is irrelevant. The tail is real physics:
it is the phase-space volume ratio of an off-shell over an on-shell `t t~`
configuration at fixed `sqrt(shat)`, and it diverges like `1/beta_t` at the
`t t~` threshold.

---

## 1. `eps_m = C/<w>` holds exactly, for all three

`analyse.py` reads the weight `w` at the instant the accept/reject tests it
(the value `_dead_trial` is handed on the line above
`random.random() * maxwgts[0] >= w_mass`), so this is the tested quantity, not
a reconstruction.

| mode | trials | `C` | `<w>` | `C/<w>` | `eps_m` MadSpin logged |
|---|---|---|---|---|---|
| `PA` | 375 715 | 3.5928 | 0.95509 | 3.762 | **3.76** |
| `PA (no jac.)` | 110 458 | 1.0540 | 0.95556 | 1.103 | **1.10** |
| `madspin` | 333 457 | 3.2068 | 0.95634 | 3.353 | **3.33** |

The number of parsed trials equals MadSpin's own `drawn` counter exactly in
every case. **All three `eps_m` are `C/<w>` and nothing else** -- the mean of
the weight is the same 0.955-0.956 in all three modes, so the entire spread of
the table is the spread of the *bound*.

`PA (no jac.)`'s `eps_m = 1.10` is the 10 % safety margin of `_combine_maxwgt`
and nothing else: its weight is the *constant* 0.95785 for more than 99.9 % of
trials (`jac_bw` is a fixed function of the Breit-Wigner window, and `Z_hat`
is identically 1 there), so `C = 1.10 x 0.95785 = 1.054`. That stage is not
unweighting anything.

Plot: `plots/mass_weight_distribution.png` -- the three weights with their
bounds drawn on. One picture, three `eps_m`.

## 2. The jacobian: narrow bulk, power-law tail

`plots/jacobian_distribution.png`, 375 715 mass sets:

| | value |
|---|---|
| mean | **1.00040** |
| median | 1.000005 |
| 90 % | 1.0193 |
| 99 % | 1.1586 |
| 99.9 % | 1.7058 |
| 99.99 % | 3.4386 |
| max | **11.956** |
| Hill index `a` (top 0.1 %) | **3.57** |

So the intuition about the *bulk* is right to an almost startling degree: nine
draws in ten are within 2 % of unity, and the mean is 1 to four decimals. But
`P(J > x) ~ x^-3.6` over four decades (right panel of the plot), and a bound
must cover the tail.

An index near 3 has three consequences, all of them visible in the data:
the mean and the variance exist (so nothing is formally broken); the maximum
of `n` draws grows like `n^(1/3)` (so the bound never settles); and the
max-weight scan can never be deep enough (11 of 375 715 PA weights exceeded
`C`, and MadSpin says so -- `CRITICAL: ... the sample is biased`).

## 3. Decomposition: the jacobian is the whole of the spread

`w_mass = J x prod_k jac_BW_k x prod_k Z_hat_k`. Verified numerically:
`w / (J . jac_BW . Z_hat) = 1` to 1e-9 over all 375 715 PA trials.

| factor | mean | 99.9 % | max | max/mean |
|---|---|---|---|---|
| `J` (reshuffling jacobian) | 1.00040 | 1.7058 | 11.956 | **11.95** |
| `prod jac_BW . Z_hat` (the no-jac. weight) | 0.9551 | 0.9579 | 0.9579 | **1.002** |
| `w` (the product) | 0.9551 | 1.4534 | 6.4452 | 6.75 |

Plot: `plots/pa_decomposition.png`. The right panel is the upper tail only,
which is all an accept/reject sees: the no-jacobian weight stops dead at
1.002 x its mean, `J` runs to 12 x.

For `madspin` the same decomposition isolates the off-shell/on-shell
production matrix-element ratio `Tr(rho_off)/|M|^2_on`:

| `madspin` factor | mean | 99.9 % | max |
|---|---|---|---|
| `J` | 1.00017 | 1.7186 | 10.150 |
| `Z_hat` product | 1.00430 | 1.7086 | 2.774 |
| `Tr(rho_off)/M^2_on` | 0.99991 | 1.0978 | **1.261** |

**The amplitude effect expected to dominate `madspin` is the flattest factor in
the whole problem** (max/mean 1.26). `madspin` and `PA` carry the *same*
reshuffling jacobian, which is why their `eps_m` are the same to within 13 %;
`madspin` is marginally cheaper because its `Z_hat` table absorbs part of the
virtuality dependence.

## 4. Where the tail comes from

`plots/threshold_demo.png` -- deterministic, no Monte Carlo. One `g g > t t~`
put back to back at a chosen `sqrt(shat)`, both tops given the same fixed
off-shell mass, `Event.reshuffle_production` asked for the jacobian:

| both tops at | `J` at `sqrt(shat) = 346.1 GeV` | `J` at 1 TeV |
|---|---|---|
| 150.6 GeV (`-15 Gamma`) | **28.9** | 1.016 |
| 165.5 GeV (`-5 Gamma`) | 17.1 | 1.006 |
| 173.0 GeV (pole) | 1.000 | 1.000 |
| 195.4 GeV (`+15 Gamma`) | (mass set does not fit) | 0.982 |

The RAMBO map (`Event.mass_shuffle`) rescales every spatial momentum by a
common `chi` fixed by `sum_i sqrt(m_i^2 + chi^2 |p_i|^2) = sqrt(shat)` and
returns `J = chi^(3n-3) . prod_i (E_i/E'_i) . (sum |p_i|^2/E_i)/(sum |p'_i|^2/E'_i)`.
At the `t t~` threshold the on-shell tops are at rest, `|p_i| -> 0`, while an
off-shell configuration with lighter tops still needs a finite `|p'_i|`; so
`chi ~ |p'|/|p|` diverges and `J ~ 1/beta_t`. Physically: near threshold,
lowering the top virtuality really does open up a much larger phase space, and
the weight is telling the truth about it.

`plots/jacobian_vs_sqrts.png` and the slice table in `data/summary.json`
(`by_sqrts_slice`) locate it in the sample:

| `sqrt(shat)` [GeV] | fraction of sample | `<J>` | max `J` | max `w` |
|---|---|---|---|---|
| 346-350 | **0.55 %** | 1.209 | **11.96** | **6.45** |
| 350-355 | 1.2 % | 1.013 | 2.39 | 2.05 |
| 355-360 | 1.6 % | 0.993 | 1.90 | 1.65 |
| 370-400 | 13.4 % | 0.998 | 1.45 | 1.24 |
| 450-500 | 16.1 % | 1.000 | 1.14 | 0.96 |
| > 800 | 6.9 % | 1.000 | 1.02 | 1.06 |

The bound `C = 3.59` is set by the **0.55 % of production events that sit
within 4 GeV of the `t t~` threshold**. For the 84 % of the sample above
450 GeV, the largest weight ever drawn is 1.05 -- a bound of 1.05 would do, and
`eps_m` would be 1.1. The `3.7x` is entirely the price of covering the
threshold events with a single global bound.

## 5. Sensitivity to the requested event count -- yes, badly

`bound_bootstrap.py` replays the max-weight scan 30 times per setting, through
the shipped `_draw_mass_value` / `_production_jacobian_for` / `_combine_maxwgt`.
MadSpin derives both the probe size and the safety factor from the run card
(`N = max(75, 3 nevents^(1/3))`, `nb_sigma = max(4.5, log_7.7 nevents)`), so
both grow with the sample:

| run card `nevents` | `N` | `nb_sigma` | `C` (30 replays) | range | `eps_m = C/<w>` |
|---|---|---|---|---|---|
| 2 000 | 75 | 4.50 | 3.13 +- 1.17 | 1.69 - 6.05 | 3.27 +- 1.22 |
| 20 000 | 81 | 4.85 | 3.29 +- 1.55 | 1.71 - 7.58 | 3.44 +- 1.62 |
| 100 000 | 139 | 5.64 | 4.56 +- 1.90 | 2.05 - 9.67 | 4.77 +- 1.99 |
| 1 000 000 | 299 | 6.77 | 6.60 +- 2.46 | 2.94 - 12.97 | 6.91 +- 2.58 |

Plot: `plots/bound_stability.png`. The bound has a +-40 % run-to-run scatter and
climbs with the sample, exactly as `max of n draws ~ n^(1/3)` predicts. Two
independent measurements corroborate it:

* the campaign got `C = 3.36`, `eps_m = 3.50`; this reproduction, same code,
  same seed, same settings, freshly generated production sample, got
  `C = 3.593`, `eps_m = 3.76` -- a 7 % move from nothing but which events landed
  first in the file;
* over the same pair of runs the `PA (no jac.)` bound was `1.054` **both
  times**, and `1.0538` at 2 000 events. That weight has no tail, so its bound
  is reproducible to four digits.

`eps_m` for `PA` is therefore not a stable number at all. Comparing 3.50 with
3.31 as if they were measurements of the same precision is not meaningful: the
`madspin` and `PA` mass stages are the same cost within the noise.

## 6. Bug hypotheses, each ruled out

| hypothesis | test | result |
|---|---|---|
| the jacobian is applied twice | `set sequential_debug True`, which recomputes the joint weight for each accepted chain and compares | *chain weight / joint weight constant to 1.42e-07 over 2000 chains* (PA) and *1.5e-07* (madspin). A double `J` would move that ratio by up to a factor 12. `data/logs/PAdebug_2000.log` |
| the bound is probed on a different quantity from the one tested | `eps_m = C/<w>` reproduces MadSpin's own counter to 0.1 % in all three modes; and in the code the probe stores `w_mass_raw` and `_complete_upfront_probe` multiplies the same `Z_hat` back in | same quantity |
| the weight is mis-normalised | `<J> = 1.00040` over 375 715 draws; `w/(J . jac_BW . Z_hat) = 1` to 1e-9 | correctly normalised |
| a dimensionful / volume factor rides along | `J` is a ratio of phase-space volumes, dimensionless, and equals 1.000 identically at the pole mass (`threshold_demo`) | none |

## 7. What *is* worth flagging (not a bug, a limitation)

1. **The PA mass stage overflows its bound.** 11 of 375 715 weights (2.9e-5)
   exceeded `C` in this run; `madspin` 5; `PA (no jac.)` zero. MadSpin prints
   `CRITICAL ... the sample is biased`. With a tail index of 3.6, no finite
   probe removes this: it is structural, not bad luck.
2. **A global bound is the wrong shape for this weight.** The tail is a
   *per-production-event* property -- a threshold event has a large `J` for
   essentially every mass draw, every other event has `J = 1`. The max-weight
   scan already computes the per-production-event maxima (`per_event` in
   `get_sequential_maxwgt`) before `_combine_maxwgt` collapses them to one
   number. A per-event bound would take `eps_m` from 3.8 to ~1.1 for 99.4 % of
   the sample and simultaneously kill the overflows. That is a design change
   with its own validation burden, not a fix applied here.

---

## Reproducing

```bash
export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"    # MadSpin
python3 doc/madspin_pa_mass_stage/probe_pa_mass_stage.py \
    --nevents 100000 --workdir /scratch/w100k --outdir /scratch/out100k

export PATH="$HOME/.pyenv/versions/mg-3.12/bin:$PATH"    # matplotlib
python3 doc/madspin_pa_mass_stage/bound_bootstrap.py \
    --events /scratch/w100k/PROC/Events/run_01/unweighted_events.lhe.gz \
    --out doc/madspin_pa_mass_stage/data --pool 4000 --replicas 30 --ps-point 500
python3 doc/madspin_pa_mass_stage/threshold_demo.py \
    --events /scratch/w100k/PROC/Events/run_01/unweighted_events.lhe.gz \
    --out doc/madspin_pa_mass_stage/plots
cp doc/madspin_pa_mass_stage/data/bound_bootstrap.json /scratch/out100k/
python3 doc/madspin_pa_mass_stage/analyse.py \
    --data /scratch/out100k --plots doc/madspin_pa_mass_stage/plots \
    --out doc/madspin_pa_mass_stage/data
```

The per-trial streams are ~80 MB per mode and are not committed;
`data/summary.json` carries the quantiles of the full sample and
`data/arrays.npz` a 50 000-row uniform subsample per mode.
No shipped code was modified: `probe_launcher.py` wraps four methods at import
time and hands over to the same `MadSpinInterface`.

---

## Follow-up

`bound_design.md` (same directory) answers the question this left open -- is
`<w>` the same constant for every production event? -- and assesses the three
candidate fixes. Short version: it is **not** constant (flat to 0.1 % over 95 %
of the sample, `-5.8 %` at 349 GeV, divergent like `1/beta_t` at the `t t~`
threshold), the reshuffling jacobian is `|p'|/|p|` exactly for a `2 -> 2`
production so a per-event bound is analytic and free (`eps_m` 3.37 -> 1.36), and
the non-constancy causes a *between-event* normalisation bias that none of the
three options addresses.
