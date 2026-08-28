# `m_tt` near `2 m_t`: where MadSpin comes back to the truth, and what it does below

`p p > t t~ j` at 13 TeV, LO, `mu_R = mu_F = m_t`, one light jet with
`pt > 20 GeV`, `|eta| < 5`. Truth: `p p > t t~ j, t > w+ b, t~ > w- b~` generated
by MG5 (doubly-resonant, both tops off shell out to `bwcutoff = 15` widths),
**5 000 000** events. MadSpin: **1 000 000** production events, decayed with
`t > w+ b` / `t~ > w- b~` and `BW_cut = 15`, once per spinmode, all four off the
same production sample and the same seed. `m_tt` is the mass of
`(W+ b) + (W- b~)` on both sides.

Figure: `plots/mtt_threshold.pdf` (MG7 style), `plots_userstyle/mtt_threshold.pdf`
(user style). Full per-bin table: `plots/numbers.txt`. Raw histograms:
`data/histograms.npz`, provenance `data/meta.json`.

**The figure is a shape comparison.** It plots `1/sigma dsigma/dm_tt`, each
curve divided by **its own total cross section over the full `m_tt` range** --
`sum(w)/N` of the whole sample, `meta.json` `runs[key].sumw`, *not* the integral
of the plotted 316-420 GeV window, which holds a few percent of the rate and
would define away part of the region under study. The 3.4 % normalisation
difference of section 1a therefore cancels and the lower pane sits on 1.

Two things follow, and neither is allowed to go quiet:

* **absolute statements cannot be read off the figure any more.** "`onshell`
  misses 16.2 % of the cross section below `2 m_t + 5 GeV`" is a rate statement;
  it lives in section 2 below and in `plots/numbers.txt`, which keeps the
  per-bin table in pb/GeV *and* a second copy in the figure's normalisation, so
  any drawn point can be checked. The agreement thresholds quoted **for the
  figure** are the shape ones of section 1b.
* **`onshell`'s zero below `2 m_t` is still zero.** Dividing by a total cross
  section cannot make a structural zero non-zero. In the ratio pane it carries
  **no** arrow and, since the figure cleanup, no marker either -- the bins are
  counted in `plots/numbers.txt`. In the main pane it is drawn as a **fall**:
  a closing vertical at 346.00 GeV, `onshell`'s measured lowest populated
  `m_tt`, in its own colour and dash pattern. `madspin`, `PA` and `madspin_v1`
  get none, because all three still have entries in the leftmost plotted bin
  (316.00 GeV; 6, 5 and 1 events) -- where they stop is where the sample ran
  out, not where the physics stops. The pane is log-`y`, so the fall runs to the
  **axis floor**, which is a floor and not a zero.

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

| spinmode | mean `Delta m_tt` | rms | max abs | max abs `Delta sqrt(shat)` | `m_tt` unchanged | events pushed below `2 m_t` |
|---|---|---|---|---|---|---|
| `madspin` | +0.137 GeV | **2.136 GeV** | 44.5 GeV | **0** | 0.29 % | 1382 |
| `PA` | -0.028 GeV | **2.102 GeV** | 45.4 GeV | **0** | 0.29 % | 1825 |
| `onshell` | 0.000 | **0.000** | **0.000** | **0** | **100 %** | 0 |
| `madspin_v1` | +0.092 GeV | **2.645 GeV** | **140.6 GeV** | **0** | **54.22 %** | **458** |

`Delta sqrt(shat) = 0` to the last bit in every event of every mode, `madspin_v1`
included: that is what proves the two event streams are really paired (MadSpin
writes one decayed event per production event, in order -- 1 000 000/1 000 000
for every mode), and it is what makes the `Delta m_tt` column mean something. So
`sqrt(shat)` is preserved exactly, as the brief said -- and `m_tt` is *not*,
because the jet takes part in the reshuffle.

"`m_tt` unchanged" is `|Delta m_tt| / m_tt < 1e-6`, i.e. `m_tt` came back the same
to the precision the LHE text carries. That threshold and not a bit-exact one:
`onshell` provably never touches the momenta and is still only **4.45 %**
bit-exact, because the LHE round-trips momenta as decimal text. 4.45 % is the
precision floor; the 1e-6 column sits above it.

**`madspin_v1` is not on the same axis as the other two off-shell modes**, and
that column is why. It has the **largest** rms of the four and the **longest**
tail by a factor of three (140.6 GeV against 45), yet it pushes the **fewest**
events across the threshold -- a third of `madspin`, a quarter of `PA`. The two
facts are not in tension: it holds `m_tt` **exactly fixed in 54 % of events**
(25.2 % of them bit for bit, against 0.00 % for `madspin` and `PA`) and moves the
remainder further than anything else does. The density modes' RAMBO
`mass_shuffle` scales every momentum by one common `chi`, so `m_tt` moves in
*every* event by O(1 GeV); the legacy path instead regenerates the phase-space
point from the decay-chain topology while holding the production tree's
invariants at the values it extracted from the production event
(`generate_momenta_conf` / `keep_inv` / `fixedinv` in `MadSpin/src/driver.f`), and
whenever the `t t~` invariant is one of the ones it holds, `m_tt` **cannot** move
at all. Half the sample is in that position. It is a narrow core with fat tails,
where `madspin` and `PA` are a broad single peak, and the threshold-crossing
count is set by the core.

Only `onshell` behaves as the brief described, and it does so for a different
reason: it draws no virtuality and `_density_do_reshuffle` is False, so the
production momenta are never touched. Its `m_tt` histogram is **identical to the
undecayed production sample's, bin for bin** on the raw 0.25 GeV grid
(556 952 entries in range, maximum bin difference **0**). Its zero below `2 m_t`
is structural: **0 of 1 000 000 events**, and it would be 0 at any sample size.

The figure therefore has to make two different statements at once, and does:
below `2 m_t` the `onshell` ratio is drawn *as an exact zero*, with an open
marker on the axis, while `madspin`, `PA` and `madspin_v1` carry real, measured
points with error bars. An empty sub-threshold bin of theirs would be a
statement about the sample size, so it would be drawn as a gap, never as a zero.

### The ratio pane is clipped to +-20 %, and nothing vanishes

The lower pane is capped at 0.8-1.2. **Fourteen** measured shape ratios live
outside that window and each carries an **arrow** at the boundary it left
through -- four `madspin`, one `PA`, nine `madspin_v1`. (It was fifteen when the
pane was in absolute normalisation, four/two/nine; the `PA` 326-331 GeV bin came
back inside once the +3.5 % plateau was divided out.) `onshell`'s exact zero
carries **no** arrow and, since the figure cleanup, no marker of its own: it is
a structural zero, its step leaves the window, and it is still a zero under this
normalisation because dividing by a total cross section cannot make it anything
else. The y-axis label says the pane is clipped and compares shapes, and
`plots/numbers.txt` lists every off-scale ratio with its value and its error, so
the clipping hides no number.

---

## 1. Where agreement returns

Scanned downwards from 420 GeV: the lowest bin edge above which **every** bin
agrees. "strict" uses the central ratio; "within errors" lets each bin spend its
own 1 sigma.

### 1a. Absolute normalisation -- the literal answer, **not** the figure's

The figure is normalised to shape (see the header and section 1b); this
subsection is the absolute answer, kept because the rate difference is itself a
result and because the numbers below would otherwise exist nowhere.

| tolerance | `madspin` | `PA` | `onshell` | `madspin_v1` |
|---|---|---|---|---|
| 5 %, strict | `m_tt >= 390 GeV` (1.037 +- 0.008) | `m_tt >= 376 GeV` (1.040 +- 0.014) | `m_tt >= 390 GeV` (1.032 +- 0.008) | `m_tt >= 390 GeV` (1.030 +- 0.008) |
| 10 %, strict | `m_tt >= 345 GeV` (0.930 +- 0.052) | `m_tt >= 347 GeV` (1.060 +- 0.038) | `m_tt >= 360 GeV` (1.035 +- 0.015) | `m_tt >= 347 GeV` (1.002 +- 0.037) |

**Read the 5 % row with care, because it is not what it looks like.** The two
samples do not share a total cross section: `truth / onshell = 0.9661`, i.e. the
absolute ratio sits on a plateau at **1.035**, not at 1. A 5 % band is therefore
only 1.5 % wide above that plateau, and the "390 GeV" is set by where the
per-bin error falls below 1.5 % -- by the sample size, not by the physics.
The 10 % row is a physics statement; the 5 % row, in absolute normalisation, is
half a statistics statement, and saying otherwise would be dishonest.

### Where the 3.4 % comes from -- checked, and only partly settled

The four measured ratios, each from the sample's own `sum(w)/N`:

| | `truth/mode` |
|---|---|
| `madspin` | 0.96342 |
| `PA` | 0.96614 |
| `onshell` | 0.96614 |
| `madspin_v1` | 0.96614 |
| `madspin_seq` (control) | 0.96613 |

with totals 651.608 pb (truth, 5M events), 676.346 pb (`madspin`) and
674.4446 pb (`PA`, `onshell`, `madspin_v1`, and the undecayed production sample,
all identical). `madspin` is off the common value by its `joint` overweights and
by nothing else -- forcing `sequential` puts it at 0.96613, on top of the other
three (section 3) -- so there is **one** number to explain, 0.9661.

**Three things it is not.** It is not statistics: MadEvent quotes
`651.8 +- 0.2185 pb` for the truth and `674.4 +- 0.2081 pb` for the production
(0.03 % each, now recorded in `meta.json` as `mg5_integration_pb`) and the five
truth runs agree to 0.011 %, so 3.4 % is of order 100 sigma. It is not a shape
effect: the `truth/onshell` ratio in 20 GeV slices from 380 GeV up is flat at
0.966, and dividing out the global total gives the *same* agreement thresholds
as dividing out a local 380-420 GeV anchor (both tables are in `numbers.txt`).
And it is not a branching-ratio mismatch -- the obvious candidate, since MadSpin
normalises to `sigma_prod * BR` while the truth's propagator is normalised by
the param card's `WT`: the model's own LO `Gamma(t -> W b)` at `m_t = 173`,
`m_b = 4.7`, `MW = 80.4185` is **1.49148 GeV** against `WT = 1.4915`, i.e.
`BR = 0.99998`. 0.002 %, not 0.8 %.

**What it is: the truth's Breit-Wigner truncation, and that is confirmed.**
MG5's decay-chain generation rejects every phase-space point with
`|m - m_t| >= 15 Gamma_t` -- `myamp.f`, and specifically the `gForceBW = 1`
branch, where a failed test is not on-shell *bookkeeping* but `cut_bw = .true.`,
so the point is thrown away and the truncation is in the integrated cross
section. Same convention as MadSpin's `BW_cut`, checked in the source rather
than assumed. MadSpin takes no such loss. The estimate reproduces:

| evaluation of the same NWA-normalised integral | per resonance | pair |
|---|---|---|
| non-relativistic BW, flat numerator: `1 - 2 arctan(2*15)/pi` | 0.97879 | **0.95802** |
| fixed-width relativistic BW (`p^2 - M(M - i Gamma)`, as generated), flat numerator | 0.97870 | **0.95785** |
| ... plus the decay numerator `m Gamma(m) / (m_t Gamma_t)` | 0.98106 | **0.96249** |
| ... plus `d ln sigma_prod/dm_t = -1.5 %/GeV` per top (an *input*) | 0.97824 | **0.95695** |
| **measured** | | **0.96614** |

The 4.20 % / 0.9580 of the original estimate is the first row, and it survives
one check it might not have: using the relativistic fixed-width propagator MG5
actually generates changes it by 0.02 %, because the corrections to the two
tails very nearly cancel inside the arctan. So **the truncation is confirmed as
the dominant term** -- it removes about 4 % where the whole difference is 3.4 %,
which leaves no room for anything else to be large.

**The residual is not 0.8 %, or rather, it is not measured.** That number is
what is left over after an estimate that holds the *numerator* flat across a
`+-22.4 GeV` window, and the numerator is not flat: the decay side alone,
`m Gamma(m)/(m_t Gamma_t)`, runs from 0.52 to 1.71 across that window and moves
the pair prediction from 0.9579 to 0.9625 -- shrinking the residual to +0.4 %.
The production side pulls the other way, and by a comparable amount: a slope of
the size `m_t` variations usually show (`-1.5 %/GeV` per top) puts the
prediction at 0.9570 and the residual at +1.0 %. This study never varied `m_t`,
so that slope is an input it does not have.

So: **the residual is +0.4 % to +1.0 %, positive in every variant tried, and of
the right size for a finite-width correction (`Gamma_t/m_t = 0.86 %`).** Calling
it "about 0.8 % of genuine off-shell rate" was the right ballpark reached by a
cancellation between two neglected effects of about half a percent each, and it
should be quoted as a range. Pinning it down needs a truth run at a second
`bwcutoff`; only 15 was run. **Nothing unaccounted-for is hiding in the 3.4 %**
-- the truncation dominates it and the three alternative explanations above are
each excluded at the sub-0.05 % level -- but the last sub-percent of it is
bounded, not determined.

`plot_mtt_threshold.normalisation_report` computes every number in this
subsection, so it can be re-run and disagreed with. This is the same
normalisation question as `doc/madspin_ae_normalisation/assessment.md`, seen from
a different observable.

### 1b. Shape -- the answer about the reshuffle, and the figure's numbers

**These are the thresholds to quote for the figure**, because this is the
figure's normalisation: each side divided by its own **total** cross section
over the full `m_tt` range (`truth/mode` = 0.9634, 0.9661, 0.9661, 0.9661).

| tolerance | `madspin` | `PA` | `onshell` | `madspin_v1` |
|---|---|---|---|---|
| 5 %, strict | **`m_tt >= 353 GeV`** (0.978 +- 0.025) | **`m_tt >= 356 GeV`** (1.035 +- 0.016) | **`m_tt >= 360 GeV`** (1.000 +- 0.015) | **`m_tt >= 360 GeV`** (0.994 +- 0.015) |
| 10 %, strict | `m_tt >= 346 GeV` (0.939 +- 0.042) | `m_tt >= 345 GeV` (1.012 +- 0.054) | `m_tt >= 350 GeV` (1.064 +- 0.029) | `m_tt >= 347 GeV` (0.968 +- 0.036) |

Rescaling instead by each mode's local 380-420 GeV offset (`madspin`
0.9650 +- 0.0027, `PA` 0.9685 +- 0.0027, `onshell` 0.9662 +- 0.0027,
`madspin_v1` 0.9663 +- 0.0027) gives **the same edge in all eight cells**, the
central ratios moving by 0.002 or less. Two independent ways of dividing out the
offset agreeing edge for edge is the evidence that it really is a flat rate
offset and not a shape; both tables are in `plots/numbers.txt`.

**The number: shape agreement to 5 % returns at `2 m_t + 7 GeV` for `madspin`,
`2 m_t + 10 GeV` for `PA` and `2 m_t + 14 GeV` for both `onshell` and
`madspin_v1`.** To 10 % it returns essentially at threshold for the two density
off-shell modes, at `2 m_t + 1 GeV` for `madspin_v1` and at `2 m_t + 4 GeV` for
`onshell`.

The ordering is the physics: the mode that samples the virtuality from the
matrix element recovers first, the mode that samples it from a fixed-width
Breit-Wigner second, and the modes that put little or nothing below threshold
recover last. **`madspin_v1` recovers with `onshell`, not with the modes it
shares a virtuality with** -- in shape it is the density modes' equal from
`2 m_t + 14 GeV` up, and below that it behaves like a mode that barely populates
the sub-threshold region, which is exactly what section 2 measures it to be.

The 10 % absolute row is the one place `madspin_v1` looks best of the four
(`m_tt >= 347 GeV`, ratio 1.002 +- 0.037). That is a coincidence of two errors
cancelling -- its deficit below threshold and the +3.5 % normalisation plateau --
and not a statement that it models the turn-on better. In shape, where the
plateau is divided out, it is last.

---

## 2. Below threshold

**Everything in this section is an absolute rate, and none of it can be read off
the figure**, which draws shapes. That is the price of the shape normalisation
and it is paid here rather than hidden: these numbers, and the per-bin table in
pb/GeV, are in `plots/numbers.txt` too.

`sigma(m_tt < 2 m_t)`, and what each mode does with it. The truth's
sub-threshold cross section is **1.0753 +- 0.0118 pb**, which is
**0.165 %** of its 651.61 pb total.

| | `sigma(m_tt < 2 m_t)` [pb] | events | ratio to truth | as a fraction of the total `sigma` |
|---|---|---|---|---|
| truth | 1.0753 +- 0.0118 | 8251 | 1 | 0.165 % |
| `madspin` | 0.9321 +- 0.0251 | 1382 | **0.867 +- 0.025** | misses 0.022 % of `sigma` |
| `PA` | 1.2309 +- 0.0288 | 1825 | **1.145 +- 0.030** | adds 0.024 % of `sigma` |
| `onshell` | **0 exactly** | **0** | **0** | **misses 0.165 % of `sigma`** |
| `madspin_v1` | 0.3089 +- 0.0144 | **458** | **0.287 +- 0.014** | misses 0.118 % of `sigma` |

So the honest size of the disagreement below threshold is:

* `onshell` misses **all** of it -- 100 %, structurally, which is 0.165 % of the
  total cross section;
* `madspin` **undershoots by 13.3 % +- 2.5 %**;
* `PA` **overshoots by 14.5 % +- 3.0 %**;
* `madspin_v1` **undershoots by 71.3 % +- 1.4 %**.

That the two *density* off-shell modes get the *integrated* sub-threshold rate
right to about 15 % is not something the framework was designed to do -- it falls
out of the recoil-jet reshuffle -- and it is a good deal better than
"structurally absent".

### `madspin_v1` is not between them; it is a fourth answer, and it is the worst

The question this run was set up to ask was whether the legacy path lands nearer
`madspin` or nearer `PA`. **It lands near neither.** At 0.287 +- 0.014 of the
truth it is **21.5 sigma** from `madspin`, **28.6 sigma** from `PA` and
**41.2 sigma** from the truth itself. On the axis that runs from `onshell`'s
structural 0 to `PA`'s 1.145, `madspin_v1` sits at 0.29 -- a third of the way to
`madspin`, and much closer to the mode that has *no* virtuality at all than to
either mode that has one.

That is worth stating plainly because it is the opposite of what the ordering of
the modes would suggest. `madspin_v1` **does** draw a virtuality -- it is not
`onshell`, its `Delta m_tt` is non-zero for half the sample and its tail is the
longest of the four. What it does not do is let that virtuality move the `t t~`
system: in 54 % of events `m_tt` is held fixed by construction (section 0), and
an event whose `m_tt` cannot move can never cross the threshold however far off
shell its tops go. The sub-threshold rate is therefore governed by the reshuffle
and not by the virtuality model, and the legacy reshuffle is the one that moves
`m_tt` least often.

The same ordering shows up in the windows just above threshold, where
`madspin_v1` tracks `onshell` rather than the density off-shell modes:

| window | `madspin` | `PA` | `onshell` | `madspin_v1` |
|---|---|---|---|---|
| first GeV above `2 m_t` (346-347) | 0.974 +- 0.043 | 1.122 +- 0.047 | 0.857 +- 0.040 | **0.830 +- 0.039** |
| `m_tt < 2 m_t + 5 GeV` | -2.3 % +- 1.3 % | +8.5 % +- 1.4 % | -16.2 % +- 1.2 % | **-12.7 % +- 1.2 %** |
| `m_tt < 2 m_t + 10 GeV` | -0.6 % +- 0.8 % | +4.8 % +- 0.9 % | -2.3 % +- 0.8 % | **-2.1 % +- 0.8 %** |

`madspin_v1`'s turn-on is displaced *further* than `onshell`'s in the first GeV
above threshold, and by `2 m_t + 10 GeV` the two are indistinguishable
(-2.1 % vs -2.3 %) while `madspin` is at -0.6 %.

**`madspin_v1`'s normalisation is clean, so none of this is a rate artefact.**
Its total is 674.4449 pb against a banner 674.4446 pb, and `truth / madspin_v1 =
0.9661`, on top of `PA` and `onshell`. Its accept/reject emitted **5** overweight
events in 1 000 000 (0.0005 %), adding **+312 pb-units to the summed weight, i.e.
+4.6e-05 % of the cross section** -- four orders of magnitude smaller than the
+0.282 % the `joint` run put into `madspin` (section 5). Nothing here is
confounded by an overweight-driven normalisation shift the way the
`madspin`-vs-`PA` comparison was.

### The window the brief asked for, `m_tt < 2 m_t + 5 GeV`

Truth: **4.6959 +- 0.0247 pb**, i.e. **0.721 %** of the total cross section.

| | `sigma` [pb] | difference vs truth | as a fraction of the total `sigma` |
|---|---|---|---|
| `madspin` | 4.5858 +- 0.0556 | **-2.3 % +- 1.3 %** | -0.017 % |
| `PA` | 5.0948 +- 0.0586 | **+8.5 % +- 1.4 %** | +0.061 % |
| `onshell` | 3.9340 +- 0.0515 | **-16.2 % +- 1.2 %** | **-0.117 %** |
| `madspin_v1` | 4.0979 +- 0.0526 | **-12.7 % +- 1.2 %** | **-0.092 %** |

`onshell` misses 0.117 % of the total cross section in that window and
`madspin_v1` 0.092 %; `madspin` and `PA` misplace roughly 0.02-0.06 % of it. At
`2 m_t + 10 GeV` the numbers are -0.6 % +- 0.8 %, +4.8 % +- 0.9 %,
-2.3 % +- 0.8 % and -2.1 % +- 0.8 % respectively.

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
sits at exactly zero. Neither is `madspin_v1` a fourth point on it: at
0.287 +- 0.014 it is 21.5 sigma below `madspin` and 28.6 sigma below `PA`, and
what separates it from both is not the virtuality model but the reshuffle -- it
holds `m_tt` fixed in 54 % of events where they hold it fixed in 0.3 %
(section 0). The `madspin`-vs-`PA` split is a virtuality question; the
`madspin_v1`-vs-both split is a kinematics question, and it is the larger of the
two by a factor of four.

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

Between `2 m_t` and `2 m_t + 10 GeV` the four modes differ in shape in a way
that mirrors the sub-threshold picture. In the first GeV above threshold
(346-347 GeV) the ratios to truth are `madspin` 0.974 +- 0.043, `PA`
1.122 +- 0.047, `onshell` 0.857 +- 0.040 and `madspin_v1` 0.830 +- 0.039:
`onshell`'s turn-on is displaced, because it has no way to put the pair anywhere
but on the on-shell locus, and `madspin_v1`'s is displaced slightly further
still. By 356 GeV all four are within 6-10 % and the residual is the flat
normalisation offset of section 1a.

---

## 5. Statistics, and what the errors will and will not support

* truth 5 x 1 000 000 events (MG5 caps one `generate_events` at 1M --
  `madevent_interface.check_nb_events` -- so this is five independent runs with
  consecutive seeds, verified distinct, summed). 8251 of them below `2 m_t`.
* MadSpin 1 000 000 production events per spinmode, one shared production
  sample, one seed. 1382 (`madspin`) / 1825 (`PA`) / 458 (`madspin_v1`) events
  below `2 m_t`. **458 is the number every `madspin_v1` sub-threshold statement
  rests on**; it is enough for the integral (3.1 % relative error, and the
  disagreement with `madspin` is 21.5 sigma) but the deepest per-bin ratios of
  section 2 carry 20-100 % errors and nothing is claimed from them.
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
* **`unweighting = auto` does not resolve the same way for the density modes.**
  `madspin` ran under `joint`, `PA` and `onshell` under `sequential`; that is
  `_auto_unweighting_mode`'s documented rule (PA/onshell -> `sequential` at every
  multiplicity, madspin/full -> `joint` for up to two decaying particles), so
  each mode ran its own shipped default -- which is the right thing to compare --
  but it is an uncontrolled difference between the curves and the overweight
  events above occurred only in the `joint` run. Controlled for in section 3:
  forcing `madspin` to `sequential` moves the sub-threshold rate by
  0.4 sigma and removes the overweights.
* **`madspin_v1` ran none of the four schemes, and its card could not have made
  it.** `set unweighting` never reaches `_unweighting_mode`, because that
  dispatcher lives inside `run_onshell` and `interface_madspin.do_launch` sends
  the legacy spinmode straight to `madspin.decay_all_events` instead. Its log
  carries no `MadSpin: unweighting = ...` line at all; `meta.json` records
  **`legacy`** for it, which is what it actually ran: one `max_weight` per decay
  channel, probed on `Nevents_for_max_weight x max_weight_ps_point` phase-space
  points before the event loop (`get_max_weight_from_event`, 299 points here),
  then a single test of the whole decay chain's weight against that bound, per
  trial, inside the Fortran driver. Two further legacy constraints, reported
  rather than worked around: **`nb_core` is ignored** (the legacy decay loop is
  single-process -- 4829 s of wall clock on one core against ~600 s on eight for
  the density modes) and **the run card is not read** (`interface_madspin`
  refuses `set run_card` under `madspin_v1`; nothing here needs it, the cuts live
  in the shared production sample). Efficiency 0.0862, i.e. 11.6 trial points per
  production event.
* **`madspin_v1`'s overweights are negligible, unlike `madspin`'s.** 5 of
  1 000 000 written events (0.0005 %) carried a non-unit weight, adding
  **+4.6e-05 %** to the sample's cross section -- against **+0.282 %** for the
  `joint` `madspin` run. Its total is 674.4449 pb against a banner 674.4446 pb.
  So the sub-threshold comparison against it is not confounded by a
  normalisation shift the way the original `madspin`-vs-`PA` one was.

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
* **The accept/reject scheme is not scanned.** Each density mode ran the scheme
  `auto` picks for it (recorded in `data/meta.json`), plus the one control of
  section 3. `sequential_global_retry` and `sequential_with_mass` are not
  exercised; `MadSpin/validation/mt_lineshape/` covers that axis. `madspin_v1`
  cannot be scanned on this axis at all -- the option does not reach it -- so
  there is no equivalent control for it, and its scheme is confounded with its
  spinmode by construction. What that does *not* leave open is whether the
  scheme explains its sub-threshold deficit: the deficit is a statement about
  which events can move `m_tt` at all (section 0), and an accept/reject scheme
  cannot make a held-fixed invariant move.
* **The 54 % held-fixed fraction is measured, not decomposed.** That `m_tt` comes
  back unchanged in half the events is measured event by event; *which*
  topologies do it is inferred from `keep_inv`/`fixedinv` in
  `MadSpin/src/driver.f` and is not broken down per production subprocess or per
  diagram here. The fraction will depend on the process and on which
  configuration the legacy path picks per event, and neither dependence is
  mapped.
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
  section 3 is 6.8 sigma on the *statistical* error and the `madspin_v1`-vs-both
  split is 21-29 sigma, which is large enough that a seed replica would not
  change either conclusion, but the seed-to-seed noise floor was not measured on
  this observable.
* **`bwcutoff` is a parameter of the figure, not a background.** It sets where
  the truth stops (`2 (m_t - 15 Gamma_t) ~ 301 GeV`) and it is matched to
  MadSpin's `BW_cut`. A different value moves the sub-threshold normalisation on
  both sides and would change the section 2 numbers. Only 15 was run.
* **The last sub-percent of the 3.4 % rate difference is bounded, not
  determined.** Section 1a excludes statistics, a shape effect and a
  branching-ratio mismatch, and confirms the Breit-Wigner truncation as the
  dominant term; what is left is +0.4 % to +1.0 % depending on
  `d ln sigma_prod/dm_t`, which this study cannot supply because it never varied
  `m_t`. Two runs would settle it -- a truth sample at a second `bwcutoff`
  (which separates the truncation from everything else directly) and a
  production sample at a shifted `m_t` (which measures the slope) -- and neither
  was done.
* **The figure's normalisation is a choice with a cost.** Dividing each curve by
  its own total cross section makes the turn-on readable and makes the rate
  difference invisible. The rate difference is real, is section 1a's subject,
  and is the reason section 2 exists in absolute units. A reader who only sees
  the figure will not know that MadSpin's total is 3.4 % above the truth's.
