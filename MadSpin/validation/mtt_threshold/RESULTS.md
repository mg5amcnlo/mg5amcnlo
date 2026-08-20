# `m_tt` near `2 m_t`: where MadSpin comes back to the truth, and what it does below

`p p > t t~ j` at 13 TeV, LO, `mu_R = mu_F = m_t`, one light jet with
`pt > 20 GeV`, `|eta| < 5`. Truth: `p p > t t~ j, t > w+ b, t~ > w- b~` generated
by MG5 (doubly-resonant, both tops off shell out to `bwcutoff = 15` widths),
**5 000 000** events. MadSpin: **1 000 000** production events, decayed with
`t > w+ b` / `t~ > w- b~` and `BW_cut = 15`, once per spinmode, all three off the
same production sample and the same seed. `m_tt` is the mass of
`(W+ b) + (W- b~)` on both sides.

Figure: `plots/mtt_threshold.pdf` (MG7 style), `plots_userstyle/mtt_threshold.pdf`
(user style). Full per-bin table: `plots/numbers.txt`. Raw histograms:
`data/histograms.npz`, provenance `data/meta.json`.

---

## 0. The correction that has to come first

**The premise this work was set up under is wrong for `n >= 3`, and the figure
measures the consequence.**

The brief stated that the RAMBO reshuffle preserves the total `t t~`
four-momentum, so that the `m_tt` of a decayed event equals the `m_tt` of the
production event it came from, and MadSpin therefore has *exactly zero* support
below `2 m_t` whatever the spinmode.

That is true for a `2 -> 2` production and **false here**.
`Event.reshuffle_production` (`madgraph/various/lhe_parser.py`) collects **every
top-level final-state momentum** and hands the lot to `Event.mass_shuffle`. For
`p p > t t~ j` that is `t`, `t~` *and the recoiling jet*. RAMBO holds
`sqrt(shat)` fixed and scales all three spatial momenta by one common `chi`, so
in the partonic CM, for a massless recoil jet,

```
m_tt^2 (after)  = shat - 2 sqrt(shat) * chi * E_j
m_tt^2 (before) = shat - 2 sqrt(shat) *       E_j
```

A mass set lighter than the pole needs `chi > 1`, which pushes `m_tt` **down**,
through `2 m_t` and below. Measured, event by event, on 1 000 000 paired events:

| spinmode | mean `Delta m_tt` | rms | max abs | max abs `Delta sqrt(shat)` | events pushed below `2 m_t` |
|---|---|---|---|---|---|
| `madspin` | +0.137 GeV | **2.136 GeV** | 44.5 GeV | **0** | 1382 |
| `PA` | -0.028 GeV | **2.102 GeV** | 45.4 GeV | **0** | 1825 |
| `onshell` | 0.000 | **0.000** | **0.000** | **0** | 0 |

`Delta sqrt(shat) = 0` to the last bit in every event of every mode: that is what
proves the two event streams are really paired (MadSpin writes one decayed event
per production event, in order), and it is what makes the `Delta m_tt` column
mean something. So `sqrt(shat)` is preserved exactly, as the brief said -- and
`m_tt` is *not*, because the jet takes part in the reshuffle.

Only `onshell` behaves as the brief described, and it does so for a different
reason: it draws no virtuality and `_density_do_reshuffle` is False, so the
production momenta are never touched. Its `m_tt` histogram is **identical to the
undecayed production sample's, bin for bin** on the raw 0.25 GeV grid
(556 952 entries in range, maximum bin difference **0**). Its zero below `2 m_t`
is structural: **0 of 1 000 000 events**, and it would be 0 at any sample size.

The figure therefore has to make two different statements at once, and does:
below `2 m_t` the `onshell` ratio is drawn *as an exact zero*, with an open
marker on the axis, while `madspin` and `PA` carry real, measured points with
error bars. An empty sub-threshold bin of theirs would be a statement about the
sample size, so it would be drawn as a gap, never as a zero.

---

## 1. Where agreement returns

Scanned downwards from 420 GeV: the lowest bin edge above which **every** bin
agrees. "strict" uses the central ratio; "within errors" lets each bin spend its
own 1 sigma.

### 1a. Absolute normalisation -- the literal answer

| tolerance | `madspin` | `PA` | `onshell` |
|---|---|---|---|
| 5 %, strict | `m_tt >= 390 GeV` (1.037 +- 0.008) | `m_tt >= 376 GeV` (1.040 +- 0.014) | `m_tt >= 390 GeV` (1.032 +- 0.008) |
| 10 %, strict | `m_tt >= 345 GeV` (0.930 +- 0.052) | `m_tt >= 347 GeV` (1.060 +- 0.038) | `m_tt >= 360 GeV` (1.035 +- 0.015) |

**Read the 5 % row with care, because it is not what it looks like.** The two
samples do not share a total cross section: `truth / onshell = 0.9661`, i.e. the
absolute ratio sits on a plateau at **1.035**, not at 1. A 5 % band is therefore
only 1.5 % wide above that plateau, and the "390 GeV" is set by where the
per-bin error falls below 1.5 % -- by the sample size, not by the physics.
The 10 % row is a physics statement; the 5 % row, in absolute normalisation, is
half a statistics statement, and saying otherwise would be dishonest.

The offset has a known origin and is quantified: MG5's decay-chain truth
truncates each top's Breit-Wigner at `|m - m_t| < 15 Gamma_t` (`myamp.f`:
`abs(xmass - prmass) < bwcutoff * prwidth` -- the same convention MadSpin's
`BW_cut` uses, checked, not assumed), which removes 2.12 % per resonance and
4.20 % for the pair. MadSpin normalises to `sigma_production * BR` and takes no
such loss and no off-shell correction to the *rate* at all. Predicted
`truth/MadSpin = 0.9580`, measured **0.9661** for `PA` and `onshell` and
**0.9634** for `madspin` -- so the truncation explains it, with about +0.8 % of
genuine off-shell rate effect on top. This is the same normalisation question as
`doc/madspin_ae_normalisation/assessment.md`, seen from a different observable.

### 1b. Shape only -- the answer about the reshuffle

Each mode rescaled by its own 380-420 GeV offset (`madspin` 0.9650 +- 0.0027,
`PA` 0.9685 +- 0.0027, `onshell` 0.9662 +- 0.0027 -- all flat and all consistent
with the global one, so this really is a normalisation and not a shape).

| tolerance | `madspin` | `PA` | `onshell` |
|---|---|---|---|
| 5 %, strict | **`m_tt >= 353 GeV`** (0.980 +- 0.025) | **`m_tt >= 356 GeV`** (1.038 +- 0.016) | **`m_tt >= 360 GeV`** (1.000 +- 0.015) |
| 10 %, strict | `m_tt >= 346 GeV` (0.940 +- 0.042) | `m_tt >= 345 GeV` (1.014 +- 0.054) | `m_tt >= 350 GeV` (1.064 +- 0.029) |

**The number: shape agreement to 5 % returns at `2 m_t + 7 GeV` for `madspin`,
`2 m_t + 10 GeV` for `PA` and `2 m_t + 14 GeV` for `onshell`.** To 10 % it
returns essentially at threshold for the two off-shell modes and at
`2 m_t + 4 GeV` for `onshell`.

The ordering is the physics: the mode that samples the virtuality from the
matrix element recovers first, the mode that samples it from a fixed-width
Breit-Wigner second, and the mode that does not sample it at all recovers last.

---

## 2. Below threshold

`sigma(m_tt < 2 m_t)`, and what each mode does with it. The truth's
sub-threshold cross section is **1.0753 +- 0.0118 pb**, which is
**0.165 %** of its 651.61 pb total.

| | `sigma(m_tt < 2 m_t)` [pb] | events | ratio to truth | as a fraction of the total `sigma` |
|---|---|---|---|---|
| truth | 1.0753 +- 0.0118 | 8251 | 1 | 0.165 % |
| `madspin` | 0.9321 +- 0.0251 | 1382 | **0.867 +- 0.025** | misses 0.022 % of `sigma` |
| `PA` | 1.2309 +- 0.0288 | 1825 | **1.145 +- 0.030** | adds 0.024 % of `sigma` |
| `onshell` | **0 exactly** | **0** | **0** | **misses 0.165 % of `sigma`** |

So the honest size of the disagreement below threshold is:

* `onshell` misses **all** of it -- 100 %, structurally, which is 0.165 % of the
  total cross section;
* `madspin` **undershoots by 13.3 % +- 2.5 %**;
* `PA` **overshoots by 14.5 % +- 3.0 %**.

That the two off-shell modes get the *integrated* sub-threshold rate right to
about 15 % is not something the framework was designed to do -- it falls out of
the recoil-jet reshuffle -- and it is a good deal better than "structurally
absent".

### The window the brief asked for, `m_tt < 2 m_t + 5 GeV`

Truth: **4.6959 +- 0.0247 pb**, i.e. **0.721 %** of the total cross section.

| | `sigma` [pb] | difference vs truth | as a fraction of the total `sigma` |
|---|---|---|---|
| `madspin` | 4.5858 +- 0.0556 | **-2.3 % +- 1.3 %** | -0.017 % |
| `PA` | 5.0948 +- 0.0586 | **+8.5 % +- 1.4 %** | +0.061 % |
| `onshell` | 3.9340 +- 0.0515 | **-16.2 % +- 1.2 %** | **-0.117 %** |

`onshell` misses 0.117 % of the total cross section in that window; `madspin`
and `PA` misplace roughly 0.02-0.06 % of it. At `2 m_t + 10 GeV` the numbers are
-0.6 % +- 0.8 %, +4.8 % +- 0.9 % and -2.3 % +- 0.8 % respectively.

---

## 3. The spinmodes disagree with each other, and it is not small

Below `2 m_t`:

```
sigma(PA) / sigma(madspin) = 1.320 +- 0.047
```

**6.8 sigma from unity**, and the difference of the two integrals is
**7.8 sigma**. The two modes bracket the truth from opposite sides: `madspin`
low by 13 %, `PA` high by 15 %. It is systematic and not noise: over the nine
sub-threshold bins of `plots/numbers.txt`, `madspin/truth` never once goes above
1 (range 0.62 to 0.99) while `PA/truth` is above 1 in eight of the nine (range
0.96 to 1.32). Bin by bin the separation is 1.6 to 5.4 sigma -- individually
modest, uniform in sign, and 7.8 sigma once integrated.

The mechanism is visible in the `Delta m_tt` moments of section 0: the two modes
smear `m_tt` by almost exactly the same rms (2.14 vs 2.10 GeV) but with opposite
mean (+0.137 vs -0.028 GeV) and different tail shapes, and `PA` pushes 32 % more
events across the threshold than `madspin` does. `PA` draws each virtuality from
a fixed-width Breit-Wigner and lets the reshuffle sort out the kinematics;
`madspin` takes the virtuality from the off-shell density matrix, which already
knows that the production matrix element falls away as the pair mass approaches
the available `sqrt(shat)`. The truth says `madspin` is the closer of the two,
but neither is right.

`onshell` is not a third point on the same axis: it has no virtuality at all and
sits at exactly zero.

### The control: is the `madspin`/`PA` split the spinmode, or the scheme?

`unweighting = auto` sends `madspin` to `joint` and `PA`/`onshell` to
`sequential` here (section 5), so the two curves being compared did not run the
same accept/reject scheme. Every scheme is supposed to sample the same
distribution; that is an assumption, not a measurement, so `madspin` was rerun
with `unweighting` forced to `sequential`. 250 000 production events (a prefix
of the same sample) -- `madspin` under `sequential` is the slow corner
`_auto_unweighting_mode` exists to avoid, ~11x the CPU per event of the joint
default here (67 ms/event/core against 5.9, wall clock x cores on the same
machine, even though its *unweighting efficiency* is the better of the two,
0.201 against 0.180) -- and the control only has to resolve a scheme effect
against a 32 % spinmode effect.

```
sigma(m_tt < 2 m_t)   madspin      (joint,      1M)  0.9321 +- 0.0251 pb
                      madspin_seq  (sequential, 250k) 0.9523 +- 0.0507 pb
                      control / default = 1.022 +- 0.061   -> 0.4 sigma
```

and the mechanism moments agree too: mean `Delta m_tt` +0.129 vs +0.137 GeV,
rms 2.131 vs 2.136 GeV, 353/250 000 vs 1382/1 000 000 events pushed below
threshold (1412 vs 1382 per million). **The scheme is not what separates
`madspin` from `PA`.**

The control does settle one thing, though: `madspin`'s total cross section
under `sequential` is 674.451 pb against 676.346 pb under `joint`, so
`truth / madspin_seq = 0.96613` -- identical to `truth / PA = 0.96614` and
`truth / onshell = 0.96614`, where the `joint` run gave 0.96341. The 0.28 %
normalisation anomaly of `madspin` in section 1a is **entirely** the overweight
events the joint accept/reject emits (695 of them, largest factor 127), and none
of it is physics.

---

## 4. Above threshold, briefly

Between `2 m_t` and `2 m_t + 10 GeV` the three modes differ in shape in a way
that mirrors the sub-threshold picture. In the first GeV above threshold
(346-347 GeV) the ratios to truth are `madspin` 0.974 +- 0.043, `PA`
1.122 +- 0.047, `onshell` 0.857 +- 0.040: `onshell`'s turn-on is displaced,
because it has no way to put the pair anywhere but on the on-shell locus. By
356 GeV all three are within 6-10 % and the residual is the flat normalisation
offset of section 1a.

---

## 5. Statistics, and what the errors will and will not support

* truth 5 x 1 000 000 events (MG5 caps one `generate_events` at 1M --
  `madevent_interface.check_nb_events` -- so this is five independent runs with
  consecutive seeds, verified distinct, summed). 8251 of them below `2 m_t`.
* MadSpin 1 000 000 production events per spinmode, one shared production
  sample, one seed. 1382 (`madspin`) / 1825 (`PA`) events below `2 m_t`.
* No bias and no phase-space cut was used to concentrate events near threshold:
  the samples are the plain inclusive process. The sub-threshold region is
  0.165 % of the cross section and the statistics above are what 6M events buy.
* The deepest bin drawn, 316-326 GeV, holds **24** truth events. Its ratio
  (1.29 +- 0.59 for `madspin`, 1.08 +- 0.53 for `PA`) supports **no** conclusion
  and none is drawn from it; the next one up, 326-331 GeV, has 188 and is barely
  better (0.77 +- 0.16 and 1.21 +- 0.20). Everything claimed here rests either
  on the integrals of section 2 or on bins from 331 GeV up, where the truth has
  500 to 1000 events per bin and the ratios carry 5-12 % errors -- quoted with
  every number.
* The `madspin` decayed sample is **not** strictly unweighted. Its own log:
  695 of 1 000 000 written events (0.0695 %) carried a non-unit weight because a
  trial weight exceeded its accept/reject bound, largest factor 127.0, adding
  **+0.282 %** to the summed event weight. So `sum(w)/N` is 676.35 pb against a
  banner cross section of 674.44 pb. `PA` and `onshell` had **zero** such events
  and sit at 674.44 pb exactly. Every number here is computed from `sum(w)` and
  `sum(w^2)`, so the overweights are in both the central values and the errors --
  but the 0.28 % is a *rate* effect on `madspin` alone and is the whole of the
  normalisation difference quoted in section 1a (0.9634 vs 0.9661) -- see the
  control in section 3, where forcing `sequential` puts `madspin` at 0.96613,
  on top of the other two.
* **`unweighting = auto` does not resolve the same way for the three modes.**
  `madspin` ran under `joint`, `PA` and `onshell` under `sequential`; that is
  `_auto_unweighting_mode`'s documented rule (PA/onshell -> `sequential` at every
  multiplicity, madspin/full -> `joint` for up to two decaying particles), so
  each mode ran its own shipped default -- which is the right thing to compare --
  but it is an uncontrolled difference between the curves and the overweight
  events above occurred only in the `joint` run. Controlled for in section 3:
  forcing `madspin` to `sequential` moves the sub-threshold rate by
  0.4 sigma and removes the overweights.

---

## 6. What this does not cover

* **Only doubly-resonant diagrams.** The truth is the MG5 decay chain, so
  single-resonant and non-resonant `W+ b W- b~ j` contributions are absent from
  *both* sides. That is deliberate -- it isolates MadSpin's kinematic
  approximation instead of mixing it with a class of diagrams the framework
  never tried to reproduce -- but it means nothing here bounds the size of those
  contributions near threshold. They are a separate measurement.
* **On-shell `W`s throughout.** The `W` is a stable final-state particle on both
  sides. `W` virtuality is not probed and cannot be, in this setup.
* **One process, one scale, one PDF, LO.** No scale or PDF variation; a fixed
  `mu_R = mu_F = m_t` was chosen precisely so the two sides could not differ
  through the dynamical-scale definition. Nothing here is a statement about the
  scale uncertainty of the effect.
* **The accept/reject scheme is not scanned.** Each mode ran the scheme `auto`
  picks for it (recorded in `data/meta.json`), plus the one control of
  section 3. `sequential_global_retry` and `sequential_with_mass` are not
  exercised; `MadSpin/validation/mt_lineshape/` covers that axis.
* **The control is at a quarter of the statistics** (250 000 events), so it
  bounds a scheme effect on the sub-threshold rate at about 6 %, not better. It
  rules out the scheme as the explanation of a 32 % difference; it does not
  establish scheme agreement to better than 6 % on this observable.
* **`2 -> 3` only.** The `n >= 3` effect of section 0 is the whole story and it
  scales with how much momentum the reshuffle can take from the recoil. A
  `2 -> 4` production would move `m_tt` more; a `2 -> 2` production would not
  move it at all, and there the brief's original premise is exactly right. That
  multiplicity dependence is not measured here.
* **No replica.** Each spinmode was run once. The `madspin`/`PA` split of
  section 3 is 6.8 sigma on the *statistical* error, which is large enough that a
  seed replica would not change the conclusion, but the seed-to-seed noise floor
  was not measured on this observable.
* **`bwcutoff` is a parameter of the figure, not a background.** It sets where
  the truth stops (`2 (m_t - 15 Gamma_t) ~ 301 GeV`) and it is matched to
  MadSpin's `BW_cut`. A different value moves the sub-threshold normalisation on
  both sides and would change the section 2 numbers. Only 15 was run.
