# MadSpin polarisation weights on a showered NLO `p p > z z`

What this is: one streaming pass over each of **four** showered HepMC2 files
— the same `p p > z z` process, the same run card and the same 250 000 events,
differing only in MadSpin's `spinmode` — reduced to four ~20 MB `.npz`, and two
figures made off them. (Three at first; the fourth, `madspin_v1`, was added by
a later pass and appears on variant B only.) The `madspin` sample carries
`keep_weight_for_polarization_vector = [0, T]` weights for both `z` and is the
one the polarisation decomposition is of; the other two enter the distribution
pane of each figure as a mode comparison.

**This pass replaces the three files of the previous one.** The samples are now
`run_06` (`madspin`, the reference), `run_07` (`PA`) and `run_08` (`onshell`),
and their MadSpin card is **exclusive** — `decay z > e+ e-` and
`decay z > mu+ mu-` — where the previous three were the inclusive
`decay z > light light`. Everything in this file is re-derived from the new
files; nothing is carried over from the previous pass, and where a previous
number is quoted it is labelled as such and is there only for the contrast.

Base of this branch: `7e39b7b1b` (`MadSpin zz_pol_weights: the onshell and PA
spinmodes on the same figures`).

**A later pass added two figure variants** beside the original figures, off the
same cached `.npz` — no HepMC was read and nothing was generated. It is
described under *The two figure variants*; base of that pass `6bf26d66e`
(`MadSpin zz_pol_weights: the exclusive-decay samples, and bare figures`).

**A later pass still added a FOURTH sample**, `madspin_v1` (`run_10`), read
from its own showered HepMC by the same extractor. It is a fourth curve in the
distribution pane and a third in the ratio pane of **variant B only** — the
original figures and variant A are untouched and their PNGs are still
byte-identical to what this branch carried. It is described under *The fourth
sample: `madspin_v1`*; base of that pass `5d31068d8` (`MadSpin zz_pol_weights:
two figure variants off the cached weights`). Where a table below says "three"
and has since grown a fourth row, the fourth row is that sample.

**A later pass again added a FIFTH sample**, `LO` (`run_12`), and a **new
six-panel figure** off it. This one is not another spinmode: it is the same
`spinmode madspin` as the reference and differs in **order**, `order = LO`
against `order = NLO`. It is a fifth curve on variant B and the partner of the
reference on the new K-factor figure. It is described under *The fifth sample:
`LO`, and the K-factor figure*; base of that pass `bdbc78e75` (`MadSpin
zz_pol_weights: madspin_v1 as a fourth curve on variant B`). Only one HepMC
was read in that pass — `run_12_decayed_1`'s — and every other number in this
file comes off the caches that were already here.

**A later pass again** restructured variant A into a **six-panel figure of
ratios**, made a **second version of it carrying scale uncertainty**, and — to
do that — **re-read two HepMC files**, because the nine `MUR*_MUF*` columns the
scale band needs were *not* in the caches. It is described under *The six-panel
ratio figures, and the scale band*; base of that pass `aacedbf19` (`MadSpin
zz_pol_weights: an LO sample, and the NLO/LO K-factor figure`). **A pass after
that put a full-width distribution pane back on top of both of them**, so each
is now seven panes and not six; the six are unchanged and it is an addition
above them. See *The seventh pane* at the end of that section; base of that
pass `a333435e2` (`MadSpin zz_pol_weights: variant A as six panels, and the
scale band`), and it read no HepMC — every curve on the new pane comes off the
two `.npz` that were already here. The scale-band pass turns the
`+4.1 % −5.2 %` flag that this file has been carrying — "larger than the
differences and not cancelling in a controlled way" — into a measurement, and
the flag was **half right and half wrong**: see *Does the polarisation
dependence survive?*. This pass also found that **the `MUR`/`MUF` labels on the
`N` line of these files are transposed**, which changes no number here and is
documented because it changes the reading of any individual scale point.

```
data/weights.npz        250 000 rows: the weights and the observables (madspin)
data/meta.json          input path and size, event count, the 33 weight names,
                        the nominal-weight choice, the lepton selection, code
                        SHA, and the registry of all four samples
data/weights_onshell.npz, data/meta_onshell.json    the same, spinmode=onshell
data/weights_PA.npz,      data/meta_PA.json         the same, spinmode=PA
data/weights_madspin_v1.npz, data/meta_madspin_v1.json  the same, spinmode=
                        madspin_v1; 29 weight names, not 33 -- no ms_pol_*
data/weights_LO.npz,      data/meta_LO.json         the same, spinmode=madspin
                        but order = LO; 33 weight names, ms_pol_* present
extract_hepmc_pol.py    the one pass over a HepMC file, plain or gzipped
pol_analysis.py         the weight algebra, the selection, the ratio statistics
plot_zz_pol.py          the MG7-style figures, and numbers.txt
plot_zz_pol_userstyle.py  the same figures in the user style
plots/, plots_userstyle/  PDF and PNG
plots/variant_A_madspin_only/, plots_userstyle/variant_A_madspin_only/
plots/variant_B_shape_ratio/,  plots_userstyle/variant_B_shape_ratio/
plots/kfactor_LO_NLO/,         plots_userstyle/kfactor_LO_NLO/
plots/variant_A6_ratios/,      plots_userstyle/variant_A6_ratios/
plots/variant_A6_ratios_scale/, plots_userstyle/variant_A6_ratios_scale/
plots/variant_A2_ratios/,      plots_userstyle/variant_A2_ratios/
plots/variant_A2_ratios_scale/, plots_userstyle/variant_A2_ratios_scale/
numbers.txt             every number quoted below, and the per-bin tables
```

The HepMC files are **not** committed and were never copied or decompressed.

---

## Nothing is written on the figures any more

The figures now carry **their axis labels, their tick labels and their legend,
and nothing else**. Five things were on them and are not:

| what left the figure | where it is now |
|---|---|
| the plot title (process, energy, event count, decay card) | `numbers.txt`, section *WHAT IS NOT ON THE FIGURES* → *figure caption*; the string itself is `pol_analysis.CAPTION` |
| `integrated: R ± σ` in the sum pane | `numbers.txt`, *integrated polarisation ratios*; and the table below |
| the four per-pane `integrated R` in the 2×2 legends | the same table. **The dashed line in each small pane still is that value** — only the number left |
| `polarisation interference` over the sum pane | this file, and the sum pane's own axis label already names the quantity |
| `bands: ±2 %, ±5 %` (user style only) | `numbers.txt`; the two bands are still drawn, only the words are gone. Variant B's ratio pane carries the same two graphics at ±1 % and ±3 %, also unlabelled |
| `spinmode ... not drawn: too few events` | `numbers.txt` prints `on the figure: ...` per observable. It does not trigger on these samples — every mode is drawn on both figures |

`numbers.txt` opens with a *WHAT IS NOT ON THE FIGURES* section listing exactly
this, so a reader who has only the figure and the text file can recover
everything. The layout is otherwise untouched: same pane stack, same three
gridspecs and gaps, same `Z_0Z_0` naming, both styles, PDF **and** PNG,
`--check-minus` on the MG7 figures only (2/2 PDFs carry `/minus`).

One thing did change and it is not a style choice: the log-scale headroom on
`M(e+ μ+)`. The exclusive samples resolve the `Z_0Z_0` tail down to 1e-08 where
the sparse samples had empty bins there, which stretches that pane to six
decades and pushed the legend back onto the total. 60× → 250× (MG7) and
400× → 4000× (user style); checked on the rendered PNGs.

---

## The pass

| spinmode | run | on disk | inflated | lines | events | **one pass** | output |
|---|---|---|---|---|---|---|---|
| `madspin` (reference) | `run_06_decayed_1` | 3.449 GB `.gz` | 10.562 GB | 109 455 942 | 250 000 | **27.8 s** | `weights.npz`, 23.2 MB |
| `PA` | `run_07_decayed_1` | 3.462 GB `.gz` | 10.599 GB | 109 867 054 | 250 000 | **28.9 s** | `weights_PA.npz`, 23.2 MB |
| `onshell` | `run_08_decayed_1` | 3.449 GB `.gz` | 10.574 GB | 109 618 102 | 250 000 | **29.0 s** | `weights_onshell.npz`, 21.7 MB |
| `madspin_v1` *(later pass)* | `run_10_decayed_1` | 3.443 GB `.gz` | 10.540 GB | 109 701 181 | 250 000 | **30.7 s** *(serial)* | `weights_madspin_v1.npz`, 19.5 MB |
| `madspin`, **`order = LO`** *(later pass)* | `run_12_decayed_1` | 3.315 GB `.gz` | 10.206 GB | 106 124 632 | 250 000 | **28.0 s** *(serial)* | `weights_LO.npz`, 23.0 MB |

All are HepMC2 `IO_GenEvent`, `HepMC::Version 2.06.09`, from
`.../PROCNLO_loop_sm_7/Events/`. **The first three passes were run
concurrently**, on separate cores of an 18-core machine; their three times are
therefore within a second of one another and none of them is a serial
measurement. The fourth was run alone in a later pass, so its 30.7 s **is** a
serial time. They are
smaller and faster than the previous pass's files (4.6 GB `.gz` / 14.1 GB
inflated / 142 M lines / ~34 s) for the obvious reason: an exclusive leptonic
`Z` decay makes no hadrons, so there is far less event record per event.

All are streamed through the system `gzip -dc` in a child process — never a
temporary on disk, never the `gzip` module. The child inflates on its own
core while this process parses, and the parser is the bottleneck either way.
The progress line for a `.gz` deliberately prints no ETA: `os.path.getsize`
gives the *compressed* size, which is not the target the inflated byte count is
walking towards.

Read in binary, line by line, with a 4 MB buffer; nothing but the current
event's leptons and photons is ever in memory. `P` lines are split with
`maxsplit=9`, which stops the split as soon as the status field is in hand and
never touches the colour-flow tail.

The `E` line's weight block is found by walking *forward* through the
random-state count (`E <10 fields> n_random <random...> n_weights <weights...>`),
not by slicing back from the end of the line. All four files have
`n_random = 0`. The count found is asserted against the `N` line every event,
and the `N` line's names are compared against that file's own first event's
names on **every one of the 250 000 events of each of the four files**, not
sampled. Note that the count itself is not constant across files: 33 for the
first three, **29** for `madspin_v1` — see *The fourth sample*.

---

## The decay card is exclusive — measured, not assumed

The brief says the card is exclusive. It is, in intent:

```
decay z > e+ e-
decay z > mu+ mu-
```

(The same card also carries `decay t`, `decay w+`, `decay w-` lines, which are
inert: there is no top and no `W` in `p p > z z`.)

Whether it is exclusive **in effect** is a different question and it decides
what these figures mean, so it was read out of the event record — mark each
`z`'s end vertex, follow the chain of status-44 copies, take the particles born
at the final vertex — for every event of every file:

| spinmode | events | one `z → e+e-` **and** the other `z → μ+μ-` | anything else |
|---|---|---|---|
| `madspin` | 250 000 | **250 000 (100.0000 %)** | 0 |
| `PA` | 250 000 | **250 000 (100.0000 %)** | 0 |
| `onshell` | 250 000 | **250 000 (100.0000 %)** | 0 |
| `LO` *(later pass)* | 250 000 | **250 000 (100.0000 %)** | 0 |

**It is exclusive in effect, exactly, on all three files.** Not "essentially 1"
— 250 000 of 250 000, with no other channel present at all. For contrast, the
previous inclusive `decay z > light light` samples gave 572 (0.229 %).

The consequence is the one the brief predicted. `M(e+ μ+)` keeps **118 521**
events after the fiducial cuts where it kept 312, and `Δφ(e+e-)` keeps
**164 111** where it kept 11 984. Both clear `MIN_SEL_TO_DRAW = 2000` by a
factor of ~60 for every mode, so **all three spinmodes are drawn on both
figures** and no pane suppresses anything.

### What each cross section is a σ *of*

**This is where the absolute normalisation stops meaning what it meant.**
Because the card is exclusive, each file's σ is

```
sigma(p p > z z)  x  BR(z -> e+e-)  x  BR(z -> mu+mu-)  x  2
```

(the 2 being the two ways of assigning the two channels to the two `z`), and
**not** a `p p > z z` cross section. The previous samples' ~13.1 pb was
essentially the production cross section, because `decay z > light light` has a
branching fraction of ~1. These are smaller by 2.354e-03, which is that product
of branching fractions (2 × 0.03434² = 2.359e-03). **The two passes' rates are
not comparable; only the ratios between the three modes are.**

| spinmode | σ (events) | banner `XSECUP` | agreement |
|---|---|---|---|
| `madspin` | **0.030874 ± 0.000070 pb** | 0.030906 ± 0.000073 | 0.3σ |
| `PA` | **0.031116 ± 0.000070 pb** | 0.031137 ± 0.000075 | 0.2σ |
| `onshell` | **0.032415 ± 0.000073 pb** | 0.032367 ± 0.000073 | 0.5σ |

Each is `sigma(p p > z z, z → e+e-, z → μ+μ-)` at 13 TeV, MC@NLO + MadSpin +
Pythia8, and each reproduces its own banner.

---

## Which weight is the full, re-derived per file

Not carried over from the previous pass. Re-derived from each new file's own
`"0"` / `"Weight"` / `C` lines:

| spinmode | `"0"×N/"Weight"` min/max | `sum("0")` = `mean("Weight")` | `C` last event | `C` first event |
|---|---|---|---|---|
| `madspin` | 1.000000000000000 / 1.000000000000000 | 0.030873887204 pb | 0.030873887204 pb | 1.394e-07 pb |
| `PA` | 1.000000000000000 / 1.000000000000000 | 0.031115558336 pb | 0.031115558336 pb | 1.403e-07 pb |
| `onshell` | 1.000000000000000 / 1.000000000000000 | 0.032414610032 pb | 0.032414610032 pb | 1.460e-07 pb |

**Every part of it came out as before.**

* `"0" × N_events == "Weight"` **to the last bit, on every event of every
  file**. So `sum("0")` is the cross section and `mean("Weight")` is the same
  number, and every ratio in this study is identical whichever is used — only
  the vertical scale of the distributions differs.
* Both reproduce that file's **last** `C` line to 1e-11.
* The **first** `C` line is Pythia's running estimate after one event and is low
  by **2.2e5** in all three files — the same trap, three more times.
* The four `ms_pol_*` weights are in the `"Weight"` normalisation
  (median|ΣLL..TT| / median|`Weight`| = 0.984, i.e. of order 1 not of order
  1e-7). Dividing them by `"0"` would inflate every ratio by 250 000. **This is
  the decisive check**, and it is why the ratios use `"Weight"` while the
  absolute `dσ/dx` uses `"0"`.
* `sum("MUR1.0_MUF1.0")/sum("Weight") − 1` is −3.0e-05, 6.6e-06 and −1.6e-05 —
  the LHE `rwgt` block's rounding, not an independent candidate.

### The `N` lines

(A fourth file, `madspin_v1`, was added by a later pass and names **29**, not
33 — see *The fourth sample*. The three of this pass:)

All three files name the **same 33 weights in the same order** — `"0"`, the
eighteen `1010`–`1027` alternative-PDF members, the nine `MUR*_MUF*`,
`"Weight"`, and the four `ms_pol_23.*_23.*`. **Identical to the previous
pass's list.** Constant within each file, checked on every one of the 250 000
`N` lines of each file against that file's own first, not sampled.

So all three files carry polarisation weights, which they need not have: a
different `spinmode` is a different reweighting — and the fourth sample added
later shows exactly that, carrying none. That these three do is recorded and
**nothing here is built on it** — decomposing `onshell` and `PA` is a different
study, and the ratio panes belong to the sample whose decomposition they are.

### Negative weights and the `|Weight|` values

Negative-weight fractions 5.709 % / 5.640 % / 5.585 % (`madspin` / `PA` /
`onshell`), all consistent with the previous pass's ~5.5 %.

Each file's dominant `|Weight|` is its banner `XMAXUP` exactly. `PA` and
`onshell` carry **exactly one** value each (0.035071 and 0.036490). The
reference `run_06` carries **five**: 0.034853 on 249 996 events and four
single-event oddities (0.035835, 0.039692, 0.052436, 0.108678). Together those
four are worth 9.5e-07 pb in a 0.030874 pb total. This is the same phenomenon
seen once in the earlier `run_02` (one anomalous event); it recurs here, four
times, again only in the `madspin` run, and again it has no consequence for
anything in this study. It is recorded rather than smoothed over.

---

## The lepton selection

Unchanged: **highest-pT final-state (status 1) lepton per flavour and charge**,
dressed with every final-state photon within `ΔR < 0.1` (the run card's own
`rphreco`), then `pT > 10 GeV` and `|η| < 2.5`.

The selection is *much* cleaner on these samples than on the inclusive ones,
and the reason is structural: there are no hadronic `Z` and so no `b`/`c`
decays making extra leptons.

**Ambiguity** — the fraction of *selected* events carrying a **second**
same-flavour same-sign lepton:

| selection | flavour | above 3 GeV | above 10 GeV | (previous pass, inclusive) |
|---|---|---|---|---|
| `M(e+ μ+)` (118 521 ev) | e+ | 0.469 % | 0.112 % | 0.64 % / 0.00 % |
| | e− | 0.459 % | 0.120 % | 0.00 % / 0.00 % |
| | μ+ | 0.313 % | 0.066 % | 0.96 % / 0.64 % |
| | μ− | 0.327 % | 0.073 % | 0.96 % / 0.64 % |
| `Δφ(e+e-)` (164 111 ev) | e+ | 0.462 % | 0.108 % | **4.61 % / 2.15 %** |
| | e− | 0.461 % | 0.119 % | **4.35 % / 2.24 %** |

The `Δφ` ambiguity falls from 4.6 % to 0.46 %, an order of magnitude, exactly
as the disappearance of the hadronic `Z` predicts. The highest-pT choice is now
essentially never a genuine choice.

**Truth matching** is now trivial and says so: purity is **1.000** for both
observables *by construction*, since every event is in the channel. Only the
efficiency carries information — 0.4741 for `M(e+ μ+)` (all four leptons in the
fiducial) and 0.6564 for `Δφ(e+e-)` (two). It is still computed from the event
record rather than asserted, and that computation is what establishes the
exclusivity in the first place.

**Dressing** moves `M(e+ μ+)` by **+2.05 GeV** on average (rms 10.21), 15.7 %
of events by more than 1 %; `Δφ(e+e-)` by +6.6e-07 rad (rms 2.4e-03), 0.52 % by
more than 1 %. Same picture as before: it matters for the mass and not for the
angle.

---

## The three spinmodes compared

Same lepton selection, same bin edges, same absolute normalisation. Errors are
the **plain quadrature sum** — the opposite of what every ratio pane uses, and
correctly so: those divide two weights of the *same* events and must keep the
covariance, these are three independent runs with no covariance to keep.

**Two χ² are quoted and only the pair is the result.** One compares the
distributions as drawn, rate included; the other rescales to a common fiducial
rate first and so tests the shape alone, on one fewer degree of freedom. **On
this pass the pair separates the two modes cleanly, and in opposite
directions** — which is the whole reason both are computed.

**Inclusive cross section, all 250 000 events, no selection:**

| spinmode | σ | vs `madspin` |
|---|---|---|
| `madspin` | 0.030874 ± 0.000070 pb | — |
| `PA` | 0.031116 ± 0.000070 pb | +0.78 % |
| `onshell` | **0.032415 ± 0.000073 pb** | **+4.99 %** |

`onshell` sits 5 % above the other two, as it did on the previous samples
(+4.21 % there). The obvious explanation is that the on-shell mode puts both
`Z` exactly on their pole and so does not pay the off-shell/Breit–Wigner
truncation the other two do; **that is an inference from the numbers, not
something this study verified.**

### The rate is now visible, because there is no channel fraction to hide it

On the previous inclusive samples, `onshell`'s +4.2 % inclusive excess was
cancelled almost exactly by a 5.1 % *lower* `z → e+e-` fraction, and the
fiducial rate difference was channel bookkeeping. **With an exclusive card
there is no channel fraction**: it is 1 in every mode by construction. So the
inclusive excess propagates straight through:

| spinmode | σ_tot | *f*(`z→e+e-`) | product | σ_fid (`Δφ`) | acceptance |
|---|---|---|---|---|---|
| `madspin` | 0.030874 | 1.000000 | 0.030874 | 0.020400 | 0.6608 |
| `onshell` | 0.032415 | 1.000000 | 0.032415 | 0.021407 | 0.6604 |
| `PA` | 0.031116 | 1.000000 | 0.031116 | 0.020509 | 0.6591 |

**The acceptance is the same in all three to 0.3 %.** The selection treats the
three modes identically; what differs is the cross section they start from.

### `Δφ(e+e-)` — 164 111 / 163 838 / 163 674 survivors

| spinmode | N | fiducial σ | rate ratio | χ²/ndf as drawn | χ²/ndf shape only |
|---|---|---|---|---|---|
| `madspin` | 164 111 | 0.020400 ± 0.000056 pb | — | — | — |
| `onshell` | 163 838 | 0.021407 ± 0.000059 pb | **1.0494 ± 0.0041 (12.0σ)** | 173.09/12 = 14.42 (p 1e-30) | **22.28/11 = 2.03 (p 0.022)** |
| `PA` | 163 674 | 0.020509 ± 0.000057 pb | 1.0053 ± 0.0039 (1.4σ) | 59.26/12 = 4.94 (p 3e-08) | **57.61/11 = 5.24 (p 3e-08)** |

**This is the trap the χ² pair exists to catch, and this time it catches it
both ways round.**

* **`onshell` differs in RATE, not shape.** Its "as drawn" χ² is 173.09/12 —
  which alone would read as a violent disagreement. Rescaling to a common
  fiducial rate takes it to 22.28/11 = 2.03. Almost all of that 173 was the
  4.9 % normalisation offset. Quoting the first number alone would have claimed
  a shape effect that is largely not there. (The residual 2.03, p = 0.022, is
  not nothing and is discussed below; it is not what the raw χ² was saying.)
* **`PA` differs in SHAPE, not rate.** Its rate ratio is 1.0053 ± 0.0039 —
  1.4σ, i.e. consistent — and yet its "as drawn" χ² of 59.26/12 **does not
  shrink** when rescaled: 57.61/11 = 5.24, p = 3e-08. The disagreement survives
  the rescaling because it was never a normalisation.

  Where: the shape ratio `PA / madspin` runs +2.6 %, −3.4 %, +2.6 %, +2.6 %,
  +4.0 %, +2.7 %, +2.1 %, +3.6 %, +2.5 %, −1.3 %, −3.9 %, −3.0 % across the
  twelve bins. **`PA` moves rate out of the back-to-back region
  (`Δφ → π`) and into intermediate `Δφ`** — which is precisely the region the
  polarisation observable lives in. A pole-approximation mode dropping the
  off-shell and interference pieces that populate `Δφ → π` is the natural
  reading, and it is a reading, not a measurement this study made.

  **This is the reverse of the previous pass**, where `PA` differed in rate
  (0.9484, 3.7σ) and its shape χ² was 1.16. That difference was channel
  bookkeeping in an inclusive card. With that removed, what is left is a shape
  effect, and it was invisible at 11 984 events.

### `M(e+ μ+)` — 118 521 / 118 080 / 117 447 survivors, and now drawn

| spinmode | N | fiducial σ | rate ratio | χ²/ndf as drawn | χ²/ndf shape only |
|---|---|---|---|---|---|
| `madspin` | 118 521 | 0.014796 ± 0.000048 pb | — | — | — |
| `onshell` | 118 080 | 0.015477 ± 0.000050 pb | **1.0460 ± 0.0048 (9.6σ)** | 104.70/7 = 14.96 (p 1e-19) | 10.98/6 = 1.83 (p 0.089) |
| `PA` | 117 447 | 0.014784 ± 0.000048 pb | 0.9992 ± 0.0046 (0.2σ) | 2.52/7 = 0.36 (p 0.93) | 2.46/6 = 0.41 (p 0.87) |

The same story, cleaner. `onshell` is +4.6 % in rate and its shape χ² is 1.83
(p = 0.089) — marginal, and 105 → 11 is the rate being taken out. `PA` agrees
with `madspin` in this observable in both rate (0.2σ) and shape (0.41):
**`PA`'s shape difference is specific to `Δφ`, and is not a global rescaling of
the four-lepton mass.** That is consistent with it being an angular effect.

Note that this figure was **not drawn at all** for the extra modes on the
previous pass (306 and 317 survivors, below `MIN_SEL_TO_DRAW`). At 118 000 it
is drawn, and it is a real comparison.

---

## The four ratios, and their sum against 1

Integrated. Errors are the correlated (delta-method) ones.

| selection | LL | TT | TL | LT | **sum** | σ from 1 |
|---|---|---|---|---|---|---|
| all events | 0.05831 | 0.69708 | 0.12244 | 0.12233 | **1.00017 ± 0.00046** | **0.36** |
| `M(e+ μ+)` | 0.05921 | 0.67456 | 0.12648 | 0.12703 | **0.98729 ± 0.00055** | **23.1** |
| `Δφ(e+e-)` | 0.05797 | 0.68936 | 0.12027 | 0.12420 | **0.99180 ± 0.00050** | **16.4** |

### What the sum being 1 means, and what its departure means

Integrated over the **whole** sample the sum is 1 to 0.017 % and 0.36σ. That is
not the absence of interference. The interference between different
polarisations of the *same* `z` carries a non-trivial dependence on the decay
angles and **integrates to zero over the full decay phase space**, so an
unrestricted sum *must* come back to 1. That it does to 2e-04 is a check on the
weights themselves: MadSpin's four `ms_pol` weights and its nominal weight are
mutually consistent and correctly normalised. The previous pass got 0.99971 ±
0.00169; this one is four times more precise and says the same thing.

**What is new is that the departure under selection is now overwhelming rather
than suggestive.** Both selected samples sit *below* 1: 0.98729 at **23.1σ**
and 0.99180 at **16.4σ**, where the previous pass had 0.9926 ± 0.0125 (0.59σ)
and 0.9965 ± 0.0043 (0.82σ) — i.e. nothing. Every lepton selection restricts
the decay angles, the interference no longer integrates away, and with 118 k
and 164 k events instead of 312 and 11 984 it is resolved.

### The sum as a function of each observable

`Δφ(e+e-)` — a large, significant and strongly *shaped* effect. Full table in
`numbers.txt`:

| Δφ [rad] | N | sum/full | error | σ from 1 |
|---|---|---|---|---|
| 0.0 – 0.3 | 3 547 | 1.0166 | 0.0044 | 3.8 |
| 1.0 – 1.3 | 8 255 | **1.0156** | 0.0025 | 6.2 |
| 1.6 – 1.8 | 13 238 | 1.0122 | 0.0016 | 7.4 |
| 2.1 – 2.4 | 19 743 | 1.0005 | 0.0014 | 0.4 |
| 2.6 – 2.9 | 26 413 | 0.9735 | 0.0012 | 21.6 |
| 2.9 – 3.1 | 27 210 | **0.9542** | 0.0012 | **38.5** |

**+1.7 % at small and intermediate `Δφ`, −4.6 % at `Δφ → π`, crossing 1 at
`Δφ ≈ 2.2`, and the two cancel on integration.** The last bin alone is 38.5σ
from 1 (it was 5.6σ before). This is the polarisation interference — the
quantity the `pure_interference` machinery in this repository exists to isolate
— measured differentially and behaving exactly as an interference term must: it
moves rate around without changing the total.

`M(e+ μ+)` — **now significant in every bin**, where the previous pass could
say nothing about it at all:

| M(e+ μ+) [GeV] | N | sum/full | error | σ from 1 |
|---|---|---|---|---|
| 0 – 45 | 19 789 | 0.9772 | 0.0014 | 16.6 |
| 45 – 70 | 23 238 | 0.9788 | 0.0013 | 15.8 |
| 70 – 95 | 21 986 | 0.9909 | 0.0013 | 7.0 |
| 95 – 125 | 19 652 | 0.9970 | 0.0013 | 2.3 |
| 125 – 175 | 17 507 | 0.9923 | 0.0014 | 5.5 |
| 175 – 260 | 10 699 | 0.9898 | 0.0016 | 6.3 |
| 260 – 450 | 4 770 | 0.9893 | 0.0025 | 4.3 |

The interference deficit is **largest at low cross-pair mass** (−2.3 %) and
smallest near 95–125 GeV (−0.3 %). The previous pass's version of this pane
"wandered between 0.95 and 1.02 with 1.4σ excursions at most" and was
explicitly not to be read as structure. **This one is structure**, and it is
the second figure finally saying something — which is the point of the
exclusive card.

### What the components themselves show

* `TT/full` runs from 0.834 at `Δφ → 0` down to **0.549** at `Δφ → π`, while
  `LL/full` runs the other way, 0.021 → 0.080 (peaking at `Δφ ≈ 2.75`).
  Back-to-back leptons in the transverse plane are where the longitudinal `Z`
  lives; the polarisation observable is reproduced cleanly and now with tiny
  errors.
* In `M(e+ μ+)`, `LL/full` peaks at **0.0935** in 70–95 GeV and falls to 0.0045
  above 260 GeV while `TT/full` rises to 0.893 — the high-mass cross-pair tail
  is transverse, as expected for a high-`p_T` `ZZ` system.

### `TL` versus `LT`: a check that only works because the statistics arrived

The previous pass said `TL` and `LT` "agree with each other everywhere within
errors, which they must by the symmetry of the two identical `Z`". That is only
half true and this pass can see the other half. Taking the difference with the
*same* delta-method error (not by subtracting two correlated rows):

| selection | `(TL − LT)/full` | σ from **zero** |
|---|---|---|
| all events (no selection) | +0.00011 ± 0.00041 | **0.3** |
| `M(e+ μ+)` (all four leptons cut) | −0.00055 ± 0.00057 | **1.0** |
| `Δφ(e+e-)` (only the electrons cut) | −0.00393 ± 0.00049 | **8.0** |

The symmetry holds where the selection respects it — unselected, and under the
four-lepton fiducial which cuts on both `Z` alike — and **breaks by 8σ under
`Δφ(e+e-)`, which cuts on the electron `Z` only and leaves the muon `Z`
unconstrained.** That is what an asymmetric selection is supposed to do to two
weights that differ by which `z` is longitudinal. Where the symmetry is intact
the agreement is a check that the weight-to-name mapping was not transposed;
where it is broken, it is the selection and not the weights. This is in
`numbers.txt` as a `(TL - LT) / full` row under each selection.

---

## The ratio errors

Numerator and denominator are sums over the **same events**, and an event's
polarisation weight is strongly correlated with its nominal weight. The error
used everywhere is the linearised (delta-method) one:

```
R = N/D,  N = sum_i n_i,  D = sum_i d_i
var(R) = sum_i (n_i - R d_i)^2 / D^2
```

which is algebraically the jackknife over events, and is the naive
independent-samples formula minus the covariance term. When `n_i = c d_i` event
by event it correctly returns **zero**, which the naive formula does not.

How much it matters, on the sum where the correlation is strongest:

| sample | correct error | naive error | factor |
|---|---|---|---|
| all events | 0.00046 | 0.00323 | 7.0× too large |
| `Δφ(e+e-)` | 0.00050 | 0.00392 | 7.8× too large |
| `M(e+ μ+)` | 0.00055 | 0.00456 | 8.3× too large |

With naive bars the 38.5σ last `Δφ` bin would read as about 5σ and the 23.1σ
`M(e+ μ+)` deficit as about 2.8σ — still visible, but the differential
structure of the mass pane would not be.

---

## Figures

`plots/` (MG7 style, usetex, `--check-minus` passes: **6/6** PDFs carry
`/minus` — the two original figures and the four variant ones) and
`plots_userstyle/` (stock rcParams), each with `m_epmup` and `dphi_ee` in
PDF and PNG. Each figure is three tiers:

1. the distribution — the full (unpolarised) `dσ/dx` at absolute normalisation
   with `LL`, `TT`, `TL`, `LT` overlaid in the same binning, **and the full
   (unpolarised) `dσ/dx` of the `onshell` and `PA` spinmodes**;
2. a **full-width pane for `(LL+TT+TL+LT)/full`**, on its own vertical scale;
3. a 2 × 2 breakdown, `LL/full`, `TT/full`, `TL/full`, `LT/full`.

Tiers 1 and 2 are a stacked **pair** sharing one x axis: the tick labels and
the axis name go under the sum pane only, with a tight gap between the two. The
2 × 2 is a separate block with its own labels, set off by a modest gap. Three
different vertical gaps means three gridspecs rather than one `hspace`.

**The two extra spinmodes enter tier 1 and nowhere else.** Tiers 2 and 3 are
the polarisation decomposition of the `madspin` (`run_06`) sample; a second
sample's numerator over a first sample's denominator is not a quantity. They
are drawn as full curves — steps with error bars, in the two hues the four
components leave free (orange and cyan, dashed and dash-dot-dot) — because that
is what they are.

Tier 1 draws an extra spinmode only where the selection leaves it at least
`pol_analysis.MIN_SEL_TO_DRAW = 2000` events. **On these samples every mode
clears it on both observables** (118 080 / 117 447 and 163 838 / 163 674
against 2 000), so all three are on both figures and nothing is suppressed. The
threshold and the machinery around it are kept unchanged for the next sample
that does not clear it.

The curves carry physics names only — `Z_0 Z_0` / `Z_0 Z_T` / `Z_T Z_0` /
`Z_T Z_T` — except that the three full curves name their spinmode, because
three unlabelled "full" curves on one pane would be unreadable:

| figure name | short key | weight column |
|---|---|---|
| `Z_0 Z_0` | LL | `ms_pol_23.0_23.0` |
| `Z_T Z_T` | TT | `ms_pol_23.T_23.T` |
| `Z_T Z_0` | TL | `ms_pol_23.T_23.0` |
| `Z_0 Z_T` | LT | `ms_pol_23.0_23.T` |
| `full (unpolarised), spinmode = madspin` | full | `Weight` of `weights.npz` |
| `full, spinmode = onshell` | — | `Weight` of `weights_onshell.npz` |
| `full, spinmode = PA` | — | `Weight` of `weights_PA.npz` |

On the `M(e+ μ+)` figure the `PA` curve sits under the black `madspin` one for
most of the range and is hard to pick out. That is the result, not a drawing
fault: `PA / madspin` is 0.9992 ± 0.0046 in that observable and agrees bin by
bin. The numbers are in `numbers.txt`.

Tier 2 does not share a scale with anything, because it is the physics and a
shared window would squash it. The four small panes do not share one either:
`TT/full ≈ 0.70` and `LL/full ≈ 0.06`, so a single window would clip the first
or flatten the rest. Each is autoscaled around its own integrated value, drawn
as a dashed line — with, now, no number printed beside it.

---

## The two figure variants

Added in a later pass, off the cached `.npz` alone — **no HepMC file was read
and nothing was generated**. They are written **alongside** the figures above,
into subdirectories of the same two style directories, and the originals are
untouched: their PNGs are byte-identical to the ones this branch already
carried.

| | MG7 style | user style |
|---|---|---|
| the original three-tier figure | `plots/` | `plots_userstyle/` |
| **A** — the same, `onshell` and `PA` dropped | `plots/variant_A_madspin_only/` | `plots_userstyle/variant_A_madspin_only/` |
| **B** — distribution + one shape-ratio pane | `plots/variant_B_shape_ratio/` | `plots_userstyle/variant_B_shape_ratio/` |

Each is `m_epmup` and `dphi_ee`, PDF **and** PNG, same binning, same `Z_0Z_0`
naming, nothing on the canvas beyond axis labels, tick labels and the legend,
and `--check-minus` on the MG7 side only (6/6). Both are drawn by the same two
scripts; `--no-variants` suppresses them.

**A** is the reference sample's polarisation decomposition on its own: the same
three tiers, the same sum pane, the same 2 × 2, with the two extra spinmode
curves gone from the distribution. Only the log-pane legend headroom changed
with them (250× → 60× in the MG7 style, 4000× → 400× in the user style), because
the legend lost two rows.

**B** replaces the sum pane *and* the 2 × 2 with a single ratio pane carrying

```
(1/sigma dsigma/dX)_Y  /  (1/sigma dsigma/dX)_madspin
                        for Y = onshell, PA, and later madspin_v1
```

all curves together. The distribution pane keeps every mode. Dividing
each mode by its own cross section first takes the rate difference out, which
is the whole point here: `onshell` differs from `madspin` mostly in **rate**
(+4.99 % inclusive, 12.0σ on the `Δφ` fiducial) and `PA` agrees in rate
(1.4σ) and differs in **shape**. A ratio that still carried the normalisation
would show the first as a large flat offset and bury the second under it.

### Decision 1 — which σ normalises each curve

**The integral of the drawn histogram** (`pol_analysis.SHAPE_NORM = 'inrange'`),
not the whole selected fiducial cross section.

The reason is that it makes the pane an exact statement about what is on the
canvas: with the in-range integral the cross-section-weighted mean of the drawn
ratio over the drawn bins is 1 by construction, so **every visible departure
from 1 is paid for by another visible bin**. Normalising by the whole selected
σ lets the out-of-range rate — which is mode dependent — slide the whole pane
by an amount whose cause is off the canvas.

`Δφ` is binned over its entire physical range `[0, π]` and has no outside at
all, so there the two are the same number to the last bit. Only `M(e+ μ+)`,
whose last edge is 450 GeV while the observable reaches 4.2 TeV, distinguishes
them:

| observable | mode | selected events outside | σ outside / σ selected |
|---|---|---|---|
| `M(e+ μ+)` | `madspin` | 880 | 0.686 % |
| | `onshell` | 924 | 0.730 % |
| | `PA` | 889 | 0.720 % |
| | `madspin_v1` | 887 | 0.702 % |
| `Δφ(e+e-)` | all four | 0 | 0 |

`numbers.txt` prints **both** normalisations bin by bin so the choice is
auditable. The largest difference between them in any bin of either observable
is **0.0005** — about a twentieth of the smallest error bar on the pane. The
choice is made for the reason above and not because it changes an answer.

**This corrects a statement made earlier in this file.** *Not covered* below
quoted 3 120 / 3 088 / 3 134 events above 450 GeV. Those are events with a
**finite** `M(e+ μ+)`, i.e. before the fiducial `pT` and `|η|` cuts. The
**selected** overflow — the events actually inside `σ_fid` and missing from the
histogram — is 880 / 924 / 889, a factor 3.5 smaller. The qualitative point
(the drawn curve does not integrate to the quoted σ) stands; the number did
not.

### Decision 2 — the error, and whether the three samples are paired

**They are not paired, and the plain quadrature bar is correct here rather
than merely conservative.**

It was established, not assumed. If the three runs had decayed one common set
of production events the mode-to-mode ratio would be correlated and a paired
error would be both right and much smaller — the factor of 1.7 to 3.5 the
sibling `m_tt` work saw. Two tests, from the cached `.npz` alone:

| spinmode | events | `n(w < 0)` | row-by-row `corr(m_4l)` |
|---|---|---|---|
| `madspin` | 250 000 | **14 273** | — |
| `onshell` | 250 000 | **13 962** | −0.00028 ± 0.00203 |
| `PA` | 250 000 | **14 099** | +0.00041 ± 0.00203 |

(A later pass added a fourth row, `madspin_v1` at **14 179** — a fourth
distinct value, so the argument below extends to it unchanged. See *The fourth
sample*.)

`n(w < 0)` is the decisive one and it is **permutation invariant**. Neither
MadSpin nor Pythia can change the sign of an event weight — MadSpin multiplies
by a positive decay factor, Pythia passes the LHE weight through — so the
number of negative-weight events is a property of the *production* sample alone
and cannot depend on the spinmode or on the order the events are written in.
Three decays of one production sample would give the same count **exactly**.
They differ by up to 311 events. **A common production sample is excluded
outright, in any order**, not merely left unproven. Consistently with that,
the three runs also carry three different `XMAXUP` (0.034853 / 0.036490 /
0.035071) and three different cross sections, and `run_06` / `run_07` /
`run_08` are three separate generations.

`corr(m_4l)` corroborates it and is the nearest thing these cached columns have
to the `sqrt(shat)` the `m_tt` study paired on: for a `2 → 2` production event
the four-lepton invariant mass *is* the production `m(ZZ)`, so two decays of
one production event would agree in it to the dressing. Row by row it is
**zero** (0.2σ on 243 000 common rows) where pairing would give ~1.

The `.npz` carries no production-level column at all — no `sqrt(shat)`, no
event number, no production kinematics — so the literal
`max |Δ sqrt(shat)| = 0` test cannot be run on this cache. It does not need to
be: the negative-weight count is a *stronger* test in the direction that
matters, because it is permutation invariant, and it fails.

So between samples the two relative errors add in quadrature. **Within** a
sample they do not: a bin's content and the σ that normalises it are sums over
the *same* events (the bin is a subset of the normalisation), so
`shape_density` uses the same delta-method error the polarisation ratios use.
A plain `sqrt(Σw²)` would overstate the bar by `1/sqrt(1 − 2 p_b)` for a bin
holding a fraction `p_b` of the rate — 8 % on a twelfth of the `Δφ` rate — and
would understate the χ² by 14 %.

### What the pane shows

Both normalised by the in-range integral; the χ² loses one degree of freedom to
that normalisation.

| observable | Y | χ²/ndf | p | max &#124;pull&#124; | slope of a weighted straight line |
|---|---|---|---|---|---|
| `Δφ(e+e-)` | `PA` | **65.90/11 = 5.99** | **7e-10** | 4.56 | **−2.71 ± 0.49 % per rad (5.5σ)** |
| | `onshell` | 24.18/11 = 2.20 | 0.012 | 2.45 | −1.08 ± 0.50 % per rad (2.2σ) |
| | `madspin_v1` | 10.27/11 = 0.93 | 0.51 | 1.64 | −0.88 ± 0.49 % per rad (1.8σ) |
| `M(e+ μ+)` | `PA` | 2.92/6 = 0.49 | 0.82 | 1.06 | +0.03 ± 0.06 % per 10 GeV (0.5σ) |
| | `onshell` | 13.25/6 = 2.21 | 0.039 | 2.98 | −0.02 ± 0.06 % per 10 GeV (0.3σ) |
| | `madspin_v1` | 7.00/6 = 1.17 | 0.32 | 1.73 | −0.10 ± 0.06 % per 10 GeV (1.7σ) |

* **`PA` on `Δφ` is the result the pane exists for.** The ratio runs
  `+2.6, −3.4, +2.6, +2.6, +4.0, +2.7, +2.1, +3.6, +2.5, −1.3, −3.9, −3.0 %`
  across the twelve bins: **about +2.5 % at low and intermediate `Δφ` falling
  to −3.9 % as `Δφ → π`**, a 5.5σ downward slope. `PA` moves rate out of the
  back-to-back region and into intermediate `Δφ`, exactly the region the
  polarisation observable lives in. Removing the linear trend leaves 35.19/10,
  so it is not *only* a slope, but the slope is its dominant feature.
* **`onshell` on `Δφ` is not the null the rate story suggests.** Its residual
  shape χ² is 24.18/11, p = 0.012, with a −1.1 % per radian slope at 2.2σ. It
  is 2.7× smaller in χ² and 2.5× smaller in slope than `PA`'s and it is the
  *right* order of magnitude smaller, but calling it "shape agreement" would
  overstate it. **The honest statement is that `onshell`'s difference is
  dominated by rate and `PA`'s is entirely shape, not that `onshell` has no
  shape difference.**
* **`M(e+ μ+)` separates them the other way and much less sharply.** `PA` is a
  flat 1 (p = 0.82). `onshell`'s 13.25/6 (p = 0.039) has no trend at all and is
  carried by one bin: 95–125 GeV at **−3.0 ± 1.0 %**, 2.98σ. With seven bins
  and this p-value that is a bin to watch and not a measurement.

**The χ² above is not the "shape only" χ² quoted under *The three spinmodes
compared*, and the two are not interchangeable.** That one rescales the other
mode to the reference's *selected* fiducial rate and keeps a plain `sqrt(Σw²)`
per-bin bar; this one uses the in-range normalisation and the delta-method
within-sample bar. This one's bars are the smaller and its χ² the larger:
`Δφ`/`PA` 65.90/11 here against 57.61/11 there, `Δφ`/`onshell` 24.18/11 against
22.28/11. The pane draws the one computed here; both are in `numbers.txt`.

### The y-window of the B pane

`_ratio_ylim`, which every other ratio pane uses, takes the extreme of
`point ± bar` over every bin. On this pane that is the wrong rule: the thinly
populated edge bins carry bars two to three times the median and would set a
window in which the few-percent structure is a flat line. `_shape_ylim` instead
takes the window from the bins whose bar is at most 3× the median — the ones
that are a measurement — and then widens it to contain the *central* value of
every bin, so no drawn point is off the canvas even where its bar runs past the
frame. The padding is asymmetric because this pane carries a legend at its
upper left. The y major locator is forced to five ticks: a few-percent window
otherwise gets two, which is not a readable scale for a pane whose entire
content is a few-percent excursion.

---

## The fourth sample: `madspin_v1`

A later pass added a fourth spinmode, `run_10` / `madspin_v1`, as a fourth
curve in variant B's distribution pane and a **third curve in its single ratio
pane**. Variant A and the two original figures are untouched — not "regenerated
identically" but literally untouched: their PDFs were restored from the commit
after the re-run, and their PNGs (which matplotlib writes deterministically)
were byte-identical anyway.

`madspin_v1` is MadSpin's **legacy** spin-correlation path; the reference
`madspin` sample is the current **density** method. So this curve is
legacy-against-density, and it is the comparison the variant was extended for.

### The input, and a correction to the brief this pass was given

The brief for this pass named the **LHE** of `run_10_decayed_1` as the input,
and spent most of its length on the resulting confound: an LHE is parton level,
the other three samples are Pythia8-showered HepMC, so a difference would be
shower-versus-parton confounded with v1-versus-density. It then asked whether
LHE siblings of `run_06/07/08` existed so the confound could be broken.

**They do — every `run_0X_decayed_1/` carries both an `events.lhe.gz` and an
`events_PYTHIA8_0.hepmc.gz` — but the question does not arise, because
`run_10_decayed_1` carries a showered HepMC too.** Using it removes the
confound entirely rather than measuring around it, and that is what this pass
did. The LHE was not used. All four samples are now the same kind of object:

| | |
|---|---|
| input | `run_10_decayed_1/events_PYTHIA8_0.hepmc.gz`, HepMC2 `IO_GenEvent` |
| same extractor | `extract_hepmc_pol.py`, streamed through a `gzip -dc` child, never unpacked to disk |
| same selection | highest-pT status-1 lepton per flavour and charge, dressed within `ΔR < 0.1`, then `pT > 10 GeV`, `|η| < 2.5` |
| same binning, same in-range normalisation | `SHAPE_NORM = 'inrange'` |

**A difference on this figure is therefore a spinmode difference.** No shower
effect was measured and none needed to be; the parton-level comparison the
brief asked for was not performed and is not needed for this figure.

### The pass

| | |
|---|---|
| on disk / inflated | 3.443 GB `.gz` → 10.540 GB |
| lines | 109 701 181 |
| events | **250 000** |
| one pass | **30.7 s** |
| output | `data/weights_madspin_v1.npz`, 19.5 MB |

Unlike the first three — which were run concurrently on separate cores and
whose quoted times are therefore not serial measurements — **this pass was run
alone, so its 30.7 s is a genuine serial time** and the only number in that
column that can be read as a rate.

### The `N` line: 29 names, not 33 — established, not assumed

The other three files name 33 weights. This one names **29**:

```
N 29 "0" "1010" ... "1027" "MUR0.5_MUF0.5" ... "MUR2.0_MUF2.0" "Weight"
```

Exactly the four `ms_pol_23.*_23.*` are missing; nothing else differs and the
order is otherwise identical. The `N` line is **present and well formed** — it
is not absent, and the file is not malformed. Checked on every one of the
250 000 `N` lines against that file's own first, as for the other three.

This is handled **explicitly** and not by a silent fallback. The extractor's
`REQUIRED_WEIGHTS` are `"0"` and `"Weight"` alone — without those there is no
nominal and it exits — and everything else is kept if present and reported if
not. It printed

```
not on the N line, not kept: ms_pol_23.0_23.0, ms_pol_23.0_23.T,
                             ms_pol_23.T_23.0, ms_pol_23.T_23.T
```

as it read the line, and `data/meta_madspin_v1.json` records
`"has_pol_weights": false` and lists the four in `weight_names_absent`.
`Data.has_pol` is `False` for this sample and every consumer of `Data.pol`
already asks first, so it can never reach a polarisation pane — and it does
not: variant B's ratio pane is a ratio of **nominal** distributions and asks no
sample but the reference for a polarisation weight.

The legacy path simply does not emit them. That is a fact about the mode, it
was read off the file rather than assumed, and **nothing in this study is built
on it**.

### The nominal, identified with the same care as the others

Same answer, re-derived rather than carried over:

| | |
|---|---|
| `"0" × N / "Weight"`, min / max over all 250 000 | 1.000000000000000 / 1.000000000000000 |
| `sum("0")` = `mean("Weight")` | **0.030918986 pb** |
| `C` line, **last** event | 0.030918986 pb — agrees to 1e-11 |
| `C` line, **first** event | 1.395e-07 pb — **low by 2.2e5** |
| banner `XSECUP ± XERRUP` | 3.090971e-02 ± 6.382e-05 → **0.15σ** |
| `|Weight|` distinct values | **one**: 0.034874918 = banner `XMAXUP` exactly |
| `sum("MUR1.0_MUF1.0")/sum("Weight") − 1` | +1.8e-05, the `rwgt` block's rounding |

So the nominal is `"Weight"` (equivalently `"0"`, which is `"Weight"/N`), the
same quantity as for the other three, confirmed against this file's **own last
`C` line**. **The first `C` line is Pythia's running estimate after one event
and is low by 2.2e5 — the same trap, a fourth time.**

The absence of the `ms_pol_*` block changes nothing about which weight is
nominal: those were never candidates for it. The one check that used them —
that they sit in the `"Weight"` normalisation and not the `"0"` one, which was
called "the decisive check" for the other three — simply does not arise for a
file that has none.

One incidental contrast: `run_06` carries **five** distinct `|Weight|` values
(one dominant plus four single-event oddities); `run_10` carries exactly
**one**, like `run_07` and `run_08`. The oddity remains `run_06`'s alone.

### The decay card is exclusive in effect, on this file too

Read off the event record the same way — mark each `z`'s end vertex, follow the
chain of status-44 copies, take the particles born at the final vertex — for
every event:

| spinmode | events | one `z → e+e-` **and** the other `z → μ+μ-` | anything else |
|---|---|---|---|
| `madspin_v1` | 250 000 | **250 000 (100.0000 %)** | 0 |

Exactly exclusive, like the other three. Survivors after the fiducial cuts:

| observable | `madspin_v1` | `madspin` (reference) |
|---|---|---|
| `M(e+ μ+)` | **117 802** | 118 521 |
| `Δφ(e+e-)` | **163 630** | 164 111 |

Both clear `MIN_SEL_TO_DRAW = 2000` by ~60×, so the mode is drawn on both
figures and nothing is suppressed.

### It is a fourth independent generation, so quadrature still holds

The same decisive test, extended:

| spinmode | events | `n(w < 0)` | row-by-row `corr(m_4l)` |
|---|---|---|---|
| `madspin` | 250 000 | **14 273** | — |
| `onshell` | 250 000 | **13 962** | −0.00028 ± 0.00203 |
| `PA` | 250 000 | **14 099** | +0.00041 ± 0.00203 |
| `madspin_v1` | 250 000 | **14 179** | **−0.00005 ± 0.00203** |

`n(w < 0)` is permutation invariant and a property of the production sample
alone, and 14 179 is a fourth distinct value. **`run_10` is a fourth separate
generation**, `corr(m_4l)` is zero as for the others, and the plain quadrature
bar between modes remains correct rather than merely conservative.

### What the ratio pane now shows for `madspin_v1` — a null

| observable | χ²/ndf | p | max &#124;pull&#124; | slope of a weighted straight line |
|---|---|---|---|---|
| `Δφ(e+e-)` | **10.27/11 = 0.93** | 0.51 | 1.64 | −0.88 ± 0.49 % per rad (1.8σ) |
| `M(e+ μ+)` | **7.00/6 = 1.17** | 0.32 | 1.73 | −0.10 ± 0.06 % per 10 GeV (1.7σ) |

Per-bin, `madspin_v1 / madspin` runs

* `Δφ(e+e-)`: `−0.6, −2.2, +0.5, +3.4, +1.4, +0.1, +1.2, +1.5, +0.3, −0.4,
  −1.1, −1.2 %` across the twelve bins;
* `M(e+ μ+)`: `+1.0, −0.7, +1.4, −0.6, −0.0, −0.4, −3.9 %` across the seven.

**No bin departs from 1 by as much as 2σ on either observable, and both χ² per
degree of freedom are ≈ 1.** In rate the two also agree: the inclusive cross
sections are 0.030919 against 0.030874 pb, **+0.15 %**, and the fiducial rate
ratios are 0.9949 ± 0.0046 (1.1σ) on `M(e+ μ+)` and 0.9986 ± 0.0039 (0.4σ) on
`Δφ`.

**So the legacy path and the density method agree, in both rate and shape, on
both observables, at these statistics.** That is a different answer from either
of the other two curves on the same pane — `onshell` differs in rate, `PA`
differs in shape, `madspin_v1` differs in neither — which is what makes it
worth drawing there.

Two honest qualifications, because a null is only worth its power:

* **What this pane could have seen.** The `madspin_v1` bars are ~1 % per bin on
  `M(e+ μ+)` and 1–3 % on `Δφ`. A shape effect of the size `PA` shows (up to
  4 %, 5.99 χ²/ndf) would have been seen comfortably; one well below a percent
  would not. This is agreement at the percent level, not agreement in
  principle.
* **The `Δφ` slope is not quite flat.** −0.88 ± 0.49 % per rad is 1.8σ — not
  significant, and the χ² of 10.27/11 gives no reason to claim a trend — but it
  carries the *same sign* as `onshell`'s (−1.08 ± 0.50, 2.2σ) and `PA`'s
  (−2.71 ± 0.49, 5.5σ). Three modes sloping the same way at 1.8σ, 2.2σ and
  5.5σ is worth writing down and is **not** worth calling an effect on this
  statistics. It would be settled by more events, not by more analysis.

### What this pass did not do

* **The LHE was not read**, and no parton-level comparison was made. Early in
  the pass, before the input was corrected, the `run_10` LHE was inspected far
  enough to establish two things that are recorded here because they are facts
  about the files and cost nothing: its `<rwgt>` block carries `1001`–`1027`
  and **no `ms_pol_*`** (consistent with the HepMC's 29-name `N` line), while
  `run_06`'s LHE **does** carry them, spelled `ms_pol_23:0_23:0` with a colon
  where the HepMC has a dot. Nothing in this study is built on either
  observation and no LHE-derived number appears in any figure or table.
* **The shower effect was not measured.** It would need the parton-level pass
  the confound-breaking plan called for, and with a showered `run_10` in hand
  that plan became unnecessary rather than merely unfinished. The LHE siblings
  do exist for all four runs if anyone wants it later.
* **No polarisation decomposition of `madspin_v1`** — impossible from this
  file, which carries no `ms_pol_*` weights at all. This is the one comparison
  the fourth sample structurally cannot join.
* **The bin edges were not re-chosen** for the fourth curve either, for the
  same reason as before: the four samples' per-bin tables are read against the
  same edges.
* **The three earlier `meta.json` were edited in place**, and only their
  `sample_registry` field, so that all four metas list all four samples. That
  is a pure JSON edit — no HepMC was re-read and no `.npz` was regenerated, so
  `weights.npz`, `weights_onshell.npz` and `weights_PA.npz` are bit-for-bit the
  files this branch already carried.

---

## The fifth sample: `LO`, and the K-factor figure

A later pass added `run_12` / **`LO`** and a **new six-panel figure**, the
K-factor `NLO/LO` per polarisation component. One HepMC was read — the new
one. Everything else came off the caches already in `data/`.

`LO` is not a fifth spinmode. Its MadSpin card says `set spinmode madspin`,
the same as the reference's. What differs is the **order**:

```
run_06  <run_settings>  order = NLO   fixed_order = OFF  shower = PYTHIA8  madspin = ON
run_12  <run_settings>  order = LO    fixed_order = OFF  shower = PYTHIA8  madspin = ON
```

and the production cross sections in the two `summary.txt` follow: **13.66 pb**
against **10.59 pb**, a ratio of 1.290.

### It carries the four `ms_pol_*`, so the decomposition is real

Its `N` line names **33** weights — the reference's list exactly, including
`ms_pol_23.0_23.0`, `ms_pol_23.0_23.T`, `ms_pol_23.T_23.0`,
`ms_pol_23.T_23.T`. (This was the condition the brief put on the figure: had it
named 29 like `madspin_v1`, only the unpolarised curve of panel 6 could have
been drawn and panels 2–5 not at all.) It does, so all six panels are real.

### The pass, and the numbers established for it

| | |
|---|---|
| input | `run_12_decayed_1/events_PYTHIA8_0.hepmc.gz` |
| on disk / inflated | 3.315 GB `.gz` → 10.206 GB, 106 124 632 lines |
| **one pass** | **28.0 s**, serial, one `gzip -dc` child, nothing unpacked to disk |
| output | `data/weights_LO.npz`, 23.0 MB, and `data/meta_LO.json` |
| events | **250 000** |
| `N`-line names | **33**, identical to the reference's — `ms_pol_*` present |
| nominal | `"0" × N == "Weight"` to the last bit (min = max = 1.0, as on all four others) |
| σ from the last `C` line | **0.023 949 678 pb**, equal to `Σ w_"0"` to 12 digits |
| σ against the banner | 0.023 949 678 / 10.59 = 2.2615e-3 = 2·BR(Z→ee)·BR(Z→μμ); the reference gives 2.2602e-3 off 13.66. **Same branching factor, so it cancels in K** |
| the first `C` line | 9.5798e-08, low by a factor 2.5e5 — the same running-estimate artefact as the other four (2.2e5 there); the **last** is right |
| `n(w < 0)` | **0** |
| survivors | `M(e+ μ+)` **119 765**, `Δφ(e+e-)` **165 206** (reference: 118 521 / 164 111) |
| exclusive decay | **250 000 / 250 000 (100.0000 %)**, one `z → e+e-` and one `z → μ+μ-`, read off the event record |

One bookkeeping consequence: `data/meta_LO.json` carries a **five**-entry
`sample_registry` and the four older metas still carry their four-entry one,
because they were written before this sample existed and re-writing them would
have meant re-reading four HepMC files to change a list. `extract_hepmc_pol.py`
now names all five, so the registry in the newest meta is the complete one.

**`n(w < 0) = 0` is not a null result and it is not evidence of a shared
production sample.** An LO matrix element for `p p > z z` is positive definite,
so the count *must* be zero; it is 14 273 for the reference. The pairing test
of *The ratio errors* therefore still separates the two samples, and in any
case they cannot be paired: they are different orders of a different run.

### One MadSpin card line differs beyond the order, and it had to be checked

The banners of `run_06` and `run_12` differ in exactly **two** places, and one
of them is a polarisation setting:

```
run_06:   set frame_id 24
run_12:  #set frame_id 24        <- commented out, so the default 6 applies
```

`frame_id` is the bitmask `sum(2**n for n in me_frame)`
(`madgraph/various/banner.py`), so **24 = legs 3+4** — the two `z`, i.e. the
`ZZ` rest frame — and the default **6 = legs 1+2**, the initial partons. Those
are different frames in general, the polarisation basis is frame dependent, and
a K-factor between two different bases would be meaningless.

**They are the same frame on this sample.** `run_12`'s LHE is `2 → 2` in every
event — `NUP = 4` throughout — so `p1 + p2 = p3 + p4` exactly and legs 1+2 and
legs 3+4 define one boost. `run_06`'s LHE carries a fifth leg in about a fifth
of its events, which is why *it* needs `frame_id 24` stated explicitly. Both
samples therefore quantise along the `ZZ`-rest-frame axis and the components
being divided are the same components. Had `run_12` been a real-emission
sample this figure could not have been drawn.

### The figure

`plots/kfactor_LO_NLO/` and `plots_userstyle/kfactor_LO_NLO/`, PDF and PNG,
both observables, both styles, `--check-minus` on the MG7 pair. The whole run
is now **8/8 applicable PDFs carry `/minus`** — two originals, four variants
and the two new ones, every one of which has a minus somewhere in its tick
labels.

Three rows of two. Panels 1–5 are one component each — unpolarised, `Z_0Z_0`,
`Z_0Z_T`, `Z_TZ_0`, `Z_TZ_T`, in that order — carrying that component's **LO
and NLO** `dσ/dX`, same colour, solid for NLO and dashed for LO. Panel 6 is
`K = NLO/LO` with all five curves on one pane.

### The two figures use different normalisations, and why

**Variant B divides every curve by its own σ. This figure divides nothing.**
That is not an inconsistency; it is the difference between the two questions.

* A K-factor **is** a rate ratio. Normalising each side by its own σ would set
  every curve on panel 6 to 1 by construction and delete the entire answer.
  Panels 1–5 are therefore absolute pb per unit of the observable and panel 6
  is their ratio.
* Variant B asks whether the **shape** moves. At these cuts the rate moves 29 %
  and the shape a few percent, so an un-normalised variant-B pane would be
  four curves sitting near 0.78 with the few-percent structure invisible under
  them. That pane exists to remove exactly what this figure exists to show.

### Errors, per curve

| curve | bar | why |
|---|---|---|
| panels 1–5, LO and NLO alike | plain MC, `sqrt(Σ w²)` per bin | each is a single weighted sum, not a ratio. There is no numerator/denominator covariance for a delta-method bar to keep |
| panel 6, all five | the two relative errors in quadrature | numerator and denominator are sums over two **independent** samples (`run_06` vs `run_12`, different order), so there is no covariance to subtract either |

**The delta-method bar the other figures' ratio panes use is on nothing drawn
here, and that is not an oversight.** It belongs to a ratio whose two sums run
over *one* set of events — a component against its own sample's total — and no
quantity on this canvas is one. The brief asked for it "where that applies";
among the six panels it applies nowhere. Where it *does* apply to this
comparison is the component-**fraction** double ratio `f_NLO / f_LO`, which is
algebraically `K_comp / K_full` with the within-sample correlation kept on each
side. `numbers.txt` prints it beside the quadrature version precisely so that
the two treatments of the same statement can be read against each other; its
bars are two to four times smaller, as they should be.

### The answer: yes, the polarisations have different K-factors

Inclusive, all 250 000 events of each sample, no lepton selection:

| component | σ_NLO [pb] | σ_LO [pb] | **K = NLO/LO** | `K − K_unpol` | `f_NLO/f_LO` (delta method) |
|---|---|---|---|---|---|
| unpolarised | 0.0308739 | 0.0239497 | **1.2891 ± 0.0039** | — | — |
| `Z_0Z_0` | 0.0018002 | 0.0014012 | **1.2847 ± 0.0070** | −0.0044 ± 0.0080 (0.5σ) | 0.9966 ± 0.0046 (0.7σ) |
| `Z_0Z_T` | 0.0037769 | 0.0027755 | **1.3608 ± 0.0063** | +0.0717 ± 0.0074 (9.7σ) | 1.0556 ± 0.0037 (**14.8σ**) |
| `Z_TZ_0` | 0.0037802 | 0.0027795 | **1.3601 ± 0.0063** | +0.0709 ± 0.0074 (9.6σ) | 1.0550 ± 0.0037 (**14.7σ**) |
| `Z_TZ_T` | 0.0215217 | 0.0170104 | **1.2652 ± 0.0041** | −0.0239 ± 0.0057 (4.2σ) | 0.9815 ± 0.0011 (**16.3σ**) |

The two **mixed** components take the largest NLO enhancement, ~1.36 against
1.289 for the total; `Z_TZ_T`, which is 70 % of the rate and therefore drags
the total with it, takes the smallest at 1.265; `Z_0Z_0` is compatible with the
unpolarised K. The component fractions move accordingly: `Z_TZ_T` falls from
71.03 % of the rate at LO to 69.71 % at NLO while `Z_0Z_T` and `Z_TZ_0` each
rise from 11.59/11.61 % to 12.23/12.24 %.

**Every bar above is MC statistics only.** No scale or PDF uncertainty is in
any of them, and the scale envelope on each sample alone is `+4.1 % −5.2 %`,
far larger than these differences and not cancelling in the ratio in any
controlled way. The significances are statements about *these two samples*,
not a theory uncertainty on K.

### `Z_0Z_T` and `Z_TZ_0` are not the same curve on `Δφ(e+e-)`

`z1_ch = 11` and `z2_ch = 13` on **all 250 000 events of both samples**, so the
first index is always the `z` that went to `e+e-`: `ms_pol_23.0_23.T` is
(electron-`z` longitudinal, muon-`z` transverse). `Δφ(e+e-)` is built from the
electron `z` alone and is therefore sensitive to which index is which, and the
two components have genuinely different shapes in it *within one sample* —
last-bin/first-bin ratio of the normalised spectrum 15.7 (`Z_0Z_T`) against
49.1 (`Z_TZ_0`) at LO, 14.0 against 20.6 at NLO. `M(e+ μ+)` takes one lepton
from each `z` and is nearly symmetric between them, which is why their
K-factors agree there (1.360 / 1.348) and not here (1.378 / 1.354).

That asymmetry is the whole of the largest excursion on either panel 6:
`K(Z_TZ_0) = 2.94 ± 0.18` in the lowest `Δφ` bin against ~1.3 for everything
else, on 3 547 NLO and 3 308 LO selected events. It is not noise, and it is not
a large NLO prediction: with the electron `z` transverse the **LO** spectrum is
very nearly empty at small `Δφ` — that needs a boosted `z`, which `2 → 2`
kinematics supplies only in the tail — and real radiation fills a region that
started almost empty. A large K on a small denominator at the edge of LO phase
space says the LO prediction is unreliable there. The same mechanism, milder,
is the rise of `K` to 2.3 in the last `M(e+ μ+)` bin.

## `LO` on variant B — and there is no LO triplet

The brief asked for `LO` "on top of the NLO one in every pane" of variant B,
and flagged the problem itself: the ratio pane is a *mode-to-mode* ratio among
NLO samples and there is only one LO sample, so `onshell_LO / madspin_LO`-style
curves cannot exist. It asked first whether sibling LO runs exist.

### They do not. `run_12` is the only showered LO sample

Every run directory was checked, not just the `run_11`/`run_12`/`run_13`
neighbourhood the brief guessed at:

* `order = LO` appears in **exactly two** banners, `run_11` and `run_12`.
  Every other run — `run_01` … `run_10` — says `order = NLO`. There is no
  `run_13` at all.
* `run_11` has **no** `run_11_decayed_1/`. It was never decayed or showered,
  so there is no HepMC to read and it cannot be a curve on anything.
* `run_11` and `run_12`'s banners are **identical apart from the run tag** — same
  process, same run card, same `spinmode madspin`, same `frame_id` line. `run_11`
  is not a second LO spinmode; it is the same configuration at a different
  seed, and its σ (10.63 pb) agrees with `run_12`'s (10.59 pb).

So there is exactly one LO sample and it is `spinmode = madspin`.

### What was done, and why

`LO` was added as a **fourth `Y` curve in the existing ratio pane**, and the
pane's definition was **not** changed. It is still
`(1/σ dσ/dX)_Y / (1/σ dσ/dX)_madspin`; `LO` is simply another `Y`. This is the
brief's first honest option — an LO/NLO curve against the reference mode — and
it happens to need no redefinition at all, because the reference mode *is* the
NLO denominator the pane already divides by.

**What that curve means is not what the other three mean, and the figure cannot
say so**, because nothing is written on it. So it is said here, in
`numbers.txt` and in the code:

* `onshell`, `PA`, `madspin_v1`: same order, different spinmode → a **spinmode**
  effect.
* `LO`: same spinmode, different order → an **order** effect.

The alternative — `LO` in the distribution pane only, ratio pane untouched —
was rejected because the shape information is the interesting half of it and
would have been thrown away for a labelling problem that a sentence fixes.

Note also that this pane divides each curve by its own σ, so the thing an order
change is *most* visible in — the rate, the K-factor — is exactly what it
removes. That is what the new figure is for, and the two are complementary
rather than redundant. And it shows a real shape effect. `χ²/ndf` of
the four `Y` curves against 1:

| `Y` | `M(e+ μ+)` | `Δφ(e+e-)` |
|---|---|---|
| `onshell` | 13.25/6 = 2.21 | 24.18/11 = 2.20 |
| `PA` | 2.92/6 = 0.49 | **65.90/11 = 5.99** |
| `madspin_v1` | 7.00/6 = 1.17 | 10.27/11 = 0.93 |
| **`LO`** | **51.77/6 = 8.63** | 27.85/11 = 2.53 |

So the order change is the largest shape effect on `M(e+ μ+)` by a factor of
four, and on `Δφ(e+e-)` it is *not* the largest — `PA`'s spinmode effect is
more than twice it there. The two observables are sensitive to different
things and neither curve dominates both. Bin by bin on `M(e+ μ+)` `LO` is
+2.4 %, +1.4 %, +2.5 % over the first three bins and then −2.1 %, −0.1 %,
−6.1 %, −6.6 % — **not** a monotone slope but a step down above ~125 GeV: NLO
moves shape into the tail, and it does so abruptly rather than gradually at
these seven edges. Whether the step is real or an artefact of a binning chosen
for a 312-event sample is exactly the question the *bin edges were not
re-chosen* bullet is about.

**Which files this pass actually rewrote.** Only variant B's four PNGs and
four PDFs, plus the eight new K-factor files, `numbers.txt`, the two logs and
the three scripts. The two original figures and variant A's four were *not*
rewritten: re-running the scripts regenerated their PDFs with a different
matplotlib Type1 font-subset tag and byte-identical PNGs — i.e. identical
content — so the PDFs were restored to the bytes this branch already carried,
exactly as the `madspin_v1` pass did before it. `git status` after the pass
lists no original or variant-A figure at all.

The legend on variant B's distribution pane now has **nine** rows (MG7) and
**twelve** (user style), so the log-pane headroom was raised — 2500× → 12000×
and 50000× → 120000× — and the linear one 1.78 → 2.00 and 2.10 → 2.45. Checked
on the rendered PNGs; at the old values the `Z_TZ_T` legend entry sat on the
black total around `M = 100` GeV. **The original figures and variant A are
untouched**, as they were by the fourth sample.

---

## The six-panel ratio figures, and the scale band

Base of this pass `aacedbf19`. Two things were asked for: variant A
restructured as a six-panel figure of ratios, and a second version of it
carrying scale uncertainty. Both are written **alongside** everything that was
already here — variant A, variant B, the K-factor figure and the originals are
untouched and their PNGs are still byte-for-byte what this branch carried.

**Both have since grown a seventh pane**, a full-width distribution above the
3 × 2, which is *The seventh pane* at the end of this section. Everything
between here and there describes the six, and the six are unchanged by it. The
section keeps its name because the six panels are still its subject.

```
plots/variant_A6_ratios/            m_epmup.{pdf,png}  dphi_ee.{pdf,png}
plots/variant_A6_ratios_scale/      m_epmup.{pdf,png}  dphi_ee.{pdf,png}
plots_userstyle/variant_A6_ratios/         the same two, user style
plots_userstyle/variant_A6_ratios_scale/   the same two, user style
```

### Version 1 — the six panels

Variant A was a distribution pane, a full-width `(Z_0Z_0 + Z_TZ_T + Z_TZ_0 +
Z_0Z_T)/full` pane and a 2 × 2 of the individual polarisation ratios, on the
NLO reference alone. Version 1 is a 3 × 2 of **ratios**:

| | left | right |
|---|---|---|
| **top** | `(Z_0Z_0 + Z_TZ_T + Z_TZ_0 + Z_0Z_T)/full` | `K = NLO/LO`, five curves |
| **middle** | `Z_0Z_0/full` | `Z_0Z_T/full` |
| **bottom** | `Z_TZ_0/full` | `Z_TZ_T/full` |

This pass dropped the distribution pane, on the argument that it is still the
top pane of the original figures, of variant B and of five of the six
K-factor panels and that this figure is about ratios. **A later pass put a
full-width distribution pane back above the 3 × 2 and that argument did not
survive it** — see *The seventh pane* below. The six panels themselves are
exactly as this section describes them and were not touched.

**Both orders on five of the six.** The figure now spans NLO and LO, so every
panel that admits two orders draws two — the sum pane and the four fraction
panes each carry the NLO reference (solid, filled markers) and the LO sample
(dashed, open markers), on the `LS_ORDER` styles the K-factor figure already
uses. The K panel is itself the ratio *between* the orders and can only be one
curve per component.

The four fractions run `LL, LT, TL, TT` — the K-factor figure's component order
— and not variant A's `LL, TT, TL, LT`, because they sit under a K panel whose
five-entry legend is in the former and reading the two against each other
should not need re-sorting.

Errors are unchanged from the figures these panels are built out of, and it is
deliberately **not one rule for the whole canvas**: the five ratio panes take
the delta-method / jackknife bar (numerator and denominator are sums over the
*same* events of *one* sample, so their covariance is real and is kept), and
the K panel takes the two relative MC errors in quadrature (two independent
samples, nothing to subtract).

### The scale columns were NOT cached, so two HepMC were re-read

This file previously said, under *Not covered*, that a scale uncertainty on `K`
"needs no new event: it is a re-sum of columns already cached". **That was
wrong and is corrected here.** `data/weights.npz` and `data/weights_LO.npz`
carried `w_MUR1.0_MUF1.0` and *no other scale point*; the extractor's
`KEEP_WEIGHTS` was `['0', 'Weight', 'MUR1.0_MUF1.0'] + POL_NAMES`. The eight
remaining columns were not there to be summed.

So `KEEP_WEIGHTS` was widened to all nine and the two samples the scale study
needs were re-read from their HepMC — **32.1 s** for `run_06_decayed_1` and
**31.9 s** for `run_12_decayed_1`, one streaming pass each, run concurrently.
**Every column that was already there came back bit-for-bit identical**, and
the two `.npz` grew by the eight new columns and by nothing else (23.2 → 28.2
MB, 23.0 → 26.3 MB). That is checked, not assumed, and it is why every figure
and every number of the earlier passes is unchanged. The three other spinmode
samples were **not** re-read: nothing on these two figures uses them, so their
`.npz` still carry the central point alone and `pol_analysis.Data.has_scale`
is how anything downstream asks.

The eight new columns are `float32`; `MUR1.0_MUF1.0` stays `float64` so that
the pre-existing column is bit-identical and the `"0"`/`"Weight"` nominal
argument keeps its precision. They are only ever used as a ratio to the central
point, where `float32`'s 1e-7 is four orders below the smallest band drawn.

### The two labels are transposed — measured, and it changes no number

**The weight the `N` line calls `MURa_MUFb` is the one generated at
`muR = b`, `muF = a`.** The two *factors* are right; which scale each belongs
to is swapped. Three independent checks, all on these files:

1. **`p p > z z` at `order = LO` is the Born, `O(αs^0)`**, so its cross section
   is *exactly* `μR`-independent and can only move with `μF`. In
   `weights_LO.npz` the weight is degenerate in the **second** label on all
   250 000 events — `MUR0.5_MUF0.5`, `MUR0.5_MUF1.0` and `MUR0.5_MUF2.0` are
   the same float — and varies only with the **first**. So the first label is
   the factorisation scale.
2. **Directly, on event 1 of `run_12_decayed_1`.** Its LHE `<rwgt>` gives ids
   1001–1003 (`μR = 1, 2, 0.5` at `μF = 1`) the same value `2.3950370e-02`,
   ids 1004–1006 (`μF = 2`) `2.3914174e-02` and ids 1007–1009 (`μF = 0.5`)
   `2.3803324e-02`. The HepMC of the same event carries `2.3803324e-02` under
   all three `MUR0.5_*` names, `2.3950370e-02` under `MUR1.0_*` and
   `2.3914174e-02` under `MUR2.0_*`. The first name label tracks the LHE's
   `μF`, value for value.
3. **The mechanism.** The LHE header writes the nine with `μF` **outer** and
   `μR` **inner**, each cycling `(1.0, 2.0, 0.5)`; the name generator assumes
   `μR` outer and `μF` inner with the same cycle. Composing the two is exactly
   a transposition, and it reproduces all nine assignments including the fixed
   points `(1,1)`, `(0.5,0.5)` and `(2,2)`.

**It changes no number in this study.** An envelope is a min and a max over a
*set*, and a permutation of a set moves neither. The 7-point subset is defined
by which two points it *drops*, so it is not automatically safe — but it is:
the pair dropped, `(μR, μF) = (0.5, 2.0)` and `(2.0, 0.5)`, is the
anti-diagonal, which the transposition maps onto itself. Dropping the two
*names* `MUR0.5_MUF2.0` and `MUR2.0_MUF0.5` therefore drops the two correct
points. What it does change is the reading of any **individual** point, so
every per-point table in `numbers.txt` is printed under the true `μR`/`μF`.

Read under the true labels, both samples behave as they must: `σ_NLO` *falls*
with `μR` (a positive `O(αs)` correction) and *rises* with `μF`; `σ_LO` has no
`μR` dependence at all and rises with `μF` by more than NLO does, which is what
an NLO calculation compensating its LO `μF` dependence looks like.

### Decision 1 — which envelope: 7-point

**Drawn: the 7-point envelope** — the nine minus the anti-diagonal pair
`(μR, μF) = (0.5, 2.0)` and `(2.0, 0.5)`. Those two put a factor of four
between the scales, manufacturing a large `log(μR/μF)` the fixed-order
calculation was never asked to resum; they inflate the envelope without probing
a missing higher order. It is a convention, so `numbers.txt` prints the 9-point
envelope beside it everywhere.

On these samples the choice matters in exactly one of the two places, and both
are worth stating:

| quantity | 7-point | 9-point |
|---|---|---|
| `σ_NLO` | **+2.33 % / −1.92 %** | **+3.14 % / −3.59 %** |
| `σ_LO` | +4.14 % / −5.24 % | +4.14 % / −5.24 % |
| `K` (unpolarised, correlated) | `[1.2297, 1.3769]` | `[1.2297, 1.3769]` |

On the **cross section** the two dropped points *are* the two extremes and the
9-point envelope is nearly twice the 7-point one. On the **K-factor** they land
in the interior and the two envelopes agree to four decimals: `K` is largest at
`(μR, μF) = (0.5, 0.5)` and smallest at `(2.0, 2.0)`, both on the main diagonal
and both kept. Measured, not assumed.

(The `+4.1 % −5.2 %` this file has been quoting from the banner is the **LO**
sample's envelope, and it is a pure `μF` effect — as it must be at `O(αs^0)`.)

### Decision 2 — correlation between numerator and denominator

**Fractions (component / full inside one sample): no choice, and the code
offers none.** The same event weights appear top and bottom, so the envelope is
taken of the **ratio recomputed scale by scale**,

```
band = [ min_s f(s), max_s f(s) ],   f(s) = Σ_i w_c,i r_s,i / Σ_i w_full,i r_s,i
```

and **not** as the ratio of two separately built envelopes. The latter would be
describing a configuration that cannot occur — the numerator at its high scale
while the denominator sits at its low one, on the same events. Almost all the
variation cancels this way; what survives is only the reweighting of the
phase-space *mix* inside the bin, which is why the fraction bands on the figure
are sub-percent where a single cross section moves by several. This is the
whole reason the four fraction panels look almost band-free and the K panel
does not.

**K-factor: a real choice, both defensible, and they differ.** The **drawn band
is the correlated one** — the same `μR`/`μF` point in both samples, envelope of
the per-scale ratio. Reasons: the two samples are the same process, the same
PDF set, the same functional scale choice and the same run card, so a scale is
a *physical setting shared between them* rather than two independent nuisance
parameters; and the `μF`/parton-luminosity part of the variation, which has
nothing to do with the order, then largely cancels, leaving a band about the
missing higher order. The **uncorrelated** alternative — independent envelopes
combined as `[num_min/den_max, num_max/den_min]` — is about a third wider and
is quoted in every `numbers.txt` table:

| component | `K` | correlated (drawn) | uncorrelated (quoted) |
|---|---|---|---|
| unpolarised | 1.2891 | `+6.81 % −4.61 %` | `+7.99 % −5.82 %` |
| `Z_0Z_0` | 1.2847 | `+7.81 % −5.32 %` | `+9.69 % −7.19 %` |
| `Z_0Z_T` | 1.3608 | `+8.38 % −5.78 %` | `+9.73 % −7.07 %` |
| `Z_TZ_0` | 1.3601 | `+8.37 % −5.78 %` | `+9.75 % −7.08 %` |
| `Z_TZ_T` | 1.2652 | `+6.24 % −4.17 %` | `+7.28 % −5.30 %` |

### Decision 3 — how a polarised component is scale-varied at all

MadSpin writes the four `ms_pol_*` weights at the **central scale and at no
other**, so there is no cached `ms_pol_23.0_23.0` at `μR = 2`. What is done —
and it is the only thing the cached weights support — is to carry the full
weight's per-event scale ratio over to the component:

```
r_s(i) = w_s(i) / w_central(i)          w_c^s(i) = w_c(i) · r_s(i)
```

i.e. the per-event polarisation *fraction* is held fixed and only the event's
overall weight moves. `s = central` reproduces the nominal exactly.

**Exact at LO**, where the `μF` and `αs` dependence multiplies the whole
squared matrix element at a fixed phase-space point, identically for the
polarised projection and for the total. **An approximation at NLO**: the
virtual and real pieces are not decomposed by polarisation, and this study's
own result is that the components have *different* K-factors, which is the same
statement as their `αs` dependence differing. So the NLO component bands are
the total's band transported onto the component, not a first-principles
polarised scale variation. Only a MadSpin run emitting `ms_pol_*` per scale
point could do better; no cached weight can, and this is stated rather than
buried.

187 of the NLO sample's 250 000 events have `|w_s/w_central| > 3` for some `s`
— MC@NLO weights that very nearly cancel. They are **not** clipped: together
they move the reweighted total by 0.39 %, and the reweighted total agrees with
the directly summed `w_s` to 3e-05, the same 3e-05 by which
`sum(MUR1.0_MUF1.0)` and `sum(Weight)` already differ. The LO sample has none;
its extreme ratio is 1.246.

### Does the polarisation dependence survive?

**The flag this file has been carrying was half right and half wrong.** Right:
the scale band on `K` itself is `−4.6 % / +6.8 %`, and the spread *between*
components is about 7.5 %, so on the raw `K` the band and the effect really are
the same size. Wrong: that is not the comparison the physics question asks for.
The scale variation moves all five `K` **together** — it is very nearly a
common multiplicative shift, as the per-scale table in `numbers.txt` shows — so
the quantity to test is

```
D = K_component / K_full     ( ≡ f_NLO / f_LO, the double ratio )
```

taken **scale by scale**, so the common movement cancels. `D = 1` means "this
component has the same K-factor as the total".

| component | `K` | ± stat | scale band on `K` | `D` | ± stat | scale band on `D` | `\|D−1\|` / half-band |
|---|---|---|---|---|---|---|---|
| unpolarised | **1.2891** | 0.0039 | `[1.2297, 1.3769]` | — | — | — | — |
| `Z_0Z_0` | **1.2847** | 0.0070 | `[1.2164, 1.3851]` | 0.9966 | 0.0046 | `+0.94 % −0.78 %` | 0.4 |
| `Z_0Z_T` | **1.3608** | 0.0063 | `[1.2821, 1.4748]` | 1.0556 | 0.0037 | `+1.47 % −1.23 %` | **3.9** |
| `Z_TZ_0` | **1.3601** | 0.0063 | `[1.2815, 1.4738]` | 1.0550 | 0.0037 | `+1.45 % −1.22 %` | **3.9** |
| `Z_TZ_T` | **1.2652** | 0.0041 | `[1.2125, 1.3441]` | 0.9815 | 0.0011 | `+0.46 % −0.54 %` | **3.8** |

Inclusive, all 250 000 events of each sample, no lepton selection; 7-point
envelope, correlated.

**The answer is yes.** Three of the four components sit 3.8–3.9 half-bands away
from `D = 1` — the scale band on `D` is a factor of three or four smaller than
the effect, where the band on `K` alone was comparable to it. `Z_0Z_0` is
compatible with the unpolarised K on both uncertainties, as it already was on
statistics alone.

The comparison the brief names directly:

```
K(mixed average) / K(Z_T Z_T) = 1.0753,  7-point envelope [1.0572, 1.0969]
K(mixed average) − K(Z_T Z_T) = 0.0952,  7-point envelope [0.0693, 0.1302]
```

The envelope **never reaches 1** (never reaches 0 for the difference). The two
mixed components take a larger NLO enhancement than `Z_TZ_T` at *every one of
the seven scale points*, by at least **5.7 %**. The mixed-versus-`Z_TZ_T`
difference is therefore still significant with the band included, and the 9-
point envelope gives the same interval to four decimals.

What the scale band does **not** rescue is any statement about the *absolute*
K-factor of a component: `K(Z_0Z_T) = 1.3608` has a `[1.2821, 1.4748]` band
around it, and quoting it as `1.361 ± 0.006` was, as this file warned, a
statement about these two samples and not a theory uncertainty.

### The seventh pane — the distribution, put back on top

Base of this pass `a333435e2`. **Both** versions gained a **full-width
distribution pane above the 3 × 2**, so each figure is now one wide pane over
six. The six are untouched: same content, same order, same conventions, same
errors, same bands. Nothing outside `variant_A6_ratios*` was rewritten, and no
HepMC was read — every curve comes off `data/weights.npz` and
`data/weights_LO.npz`.

**What is on it.** `dσ/dx` at **absolute** normalisation, same binning as the
panes below: the unpolarised total and the four polarised components, **at NLO
and at LO** — ten curves. Colour is the component, exactly the colour it
carries everywhere else in this study; the line is the **order**, on the same
`LS_ORDER` (solid NLO, dashed LO) the six panes below and the K-factor figure
use, so one rule reads across the whole canvas. **LO is drawn over NLO**, which
is the reverse of the six panes below and is asked for: `K ≈ 1.29` is a hair's
breadth on a log pane, whichever curve is underneath disappears, and LO is the
thinner, dashed, non-reference one. Errors are the plain MC `√Σw²` of
`component_histogram` — each curve is a single weighted sum and not a ratio, so
there is no covariance for a delta-method bar to keep, the same rule panels 1–5
of the K-factor figure follow.

**Why it is back.** The section above dropped it because "this figure is about
ratios". Six ratio panes with no absolute scale anywhere say how the components
divide the rate up and never say what the rate *is*, and a reader who wants the
`K` pane to mean something has to go and fetch the two cross sections off
another figure. That is the gap this pane closes.

**Geometry.** A nested gridspec, exactly as the other stacked layouts here do
it: an outer `(2, 1)` with `height_ratios = [2.9, 10.2]` and `hspace = 0.09`,
and the 3 × 2 built as a `subgridspec` of the lower cell keeping its own
`hspace = 0.07`, `wspace = 0.30` unchanged. Two vertical rhythms need two
gridspecs — the outer gap has to clear the wide pane's own tick labels and axis
name, and any single `hspace` over four rows is either tight enough to crush
that or wide enough to tear the grid's rows apart. The figure grew from
`9.0 × 10.2` to `9.0 × 13.4` in (MG7) and `9.4 × 10.4` to `9.4 × 13.6` in
(user style). Checked on the rendered PNGs.

**Log y, and it is measured rather than inherited.** Every other distribution
pane in this study takes its y scale from `pol_analysis.LOGY`, which is a
property of the *observable*. That is right for a pane carrying five curves.
This one carries ten, so it asks the data:

| observable | span of the ten curves | drawn |
|---|---|---|
| `M(e+ μ+)` | `7.74e−09 … 1.17e−04`, **4.18 decades** | log |
| `Δφ(e+e−)` | `2.11e−05 … 1.31e−02`, **2.79 decades** | log |

Both clear the two-decade threshold, so both are log. `Δφ` is the one this
changes — `LOGY` has it linear, and drawn linear **four of the ten curves**
(`Z_0Z_0` and `Z_0Z_T` at both orders) lie on top of one another within a few
pixels of the frame and cannot be read at all. `LOGY` itself is not touched and
every other figure in the study is unchanged. The legend needed `14×` headroom
(MG7) / `26×` (user style, opaque frame) to clear the data; checked on the
PNGs, not on the axis limits.

**One thing the ten curves needed.** `Z_0Z_T` and `Z_TZ_0` are equal to 1–2 %
bin by bin — they must be, `ZZ` is symmetric and the two differ only by which
`Z` the projection is taken on — so on a log pane they are the same curve to
within the line width, and drawn flat the purple `Z_0Z_T` was simply **not on
the figure** while both of its legend entries were. The components are now
drawn with the line weight **stepping down**, so the later, thinner curve sits
inside the earlier, thicker one and leaves a halo of it showing. Nothing is
offset and no value is touched: two coincident curves are drawn coincident and
both are visible.

**The band, on the scale version.** Drawn on the **unpolarised pair only**, NLO
and LO, and not on the four components. Ten translucent bands behind ten curves
is a pane in which neither the bands nor the curves can be attributed, and the
unpolarised pair is what this pane exists to state. It is
`pol_analysis.RATIO6_TOP_BAND_KEYS`, a list and not a boolean, so widening it
to all five is one edit.

**This applies to the new pane only.** The six panes below band every curve
they carry, exactly as *Version 2* describes, and were not changed. Reading the
instruction as applying to the whole figure would strip the bands off the
`Z_0Z_0`, `Z_0Z_T`, `Z_TZ_0` and `Z_TZ_T` panes, which are the panes the scale
version exists for; that reading was not taken, and if it is wanted it is
`with_band` on `ratio_pane` and nothing else.

**And the band is small on a log pane, which is worth saying plainly.** A few
percent is 0.017 of a decade; on a pane four decades tall that is under two
pixels. The unpolarised band on this pane is therefore at the line width and
cannot be measured off the canvas — it is drawn to scale and not exaggerated.
The trade is deliberate: a linear `Δφ` pane makes the band legible and four
curves illegible, and the band is *tabulated* while the curves are not drawn
anywhere else on the figure. `numbers.txt` prints the per-bin envelope for
**all five components at both orders**, so nothing the pane does not draw is
lost. The six ratio panes remain where this figure's bands are read.

### The reduced variation — three panes, all full width

`variant_A2_ratios/` and `variant_A2_ratios_scale/` are the **same figure with
the four polarised-fraction panes dropped**: the distribution pane, then the
sum consistency, then the K-factor, stacked, all three full width. The digit
counts ratio panes exactly as `A6` does — six there, two here — so the two
names cannot be read for one another.

```
plots/variant_A2_ratios/            m_epmup.{pdf,png}  dphi_ee.{pdf,png}
plots/variant_A2_ratios_scale/      m_epmup.{pdf,png}  dphi_ee.{pdf,png}
plots_userstyle/variant_A2_ratios/         the same two, user style
plots_userstyle/variant_A2_ratios_scale/   the same two, user style
```

Nothing is recomputed and nothing is restyled. The three panes are drawn by
the very same painters the seven-pane figure uses, off the same
`ratio6_curves` objects, so every curve, every error bar, every colour, every
line style and every label is the one above — which is why `numbers.txt` is
**unchanged by this figure's existence**, the fractions' per-bin tables
included. The seven-pane figures are untouched and their PNGs are still
byte-for-byte what this branch carried.

**All three panes share one x axis**, and that is what the reduction buys. On
the seven-pane figure the distribution pane sits above a 3 × 2 and is a block
of its own, so it writes its own x tick labels and its own axis name and the
outer gap exists to hold them. Here every pane is full width over the same
binning, the x axis is written **once at the bottom**, and both gaps hold
nothing but frame — 0.136 in on the MG7 style, 0.140 in on the user style,
which is what the seven-pane figure's 3 × 2 puts between its own rows. The
gaps are held in **inches**: `hspace` is a fraction of the mean row height and
these rows are not those, so the fraction is solved for rather than carried
over.

**The long rotated axis name is still what sets the ratio panes' height, and
it does not care that they are now twice as wide.**
`(Z_0Z_0+Z_TZ_T+Z_TZ_0+Z_0Z_T)/full` is 1.90 in of type at fontsize 9 and is
measured against the pane's *height*. So the ratio panes keep **2.31 in**
each, the seven-pane figure's top-row height, clearing by **0.208 in** at each
end; the user style's plain-text version of the same name is 2.08 in, so its
panes keep **2.49 in** and clear by **0.205 in**. The distribution pane takes
**4.62 in** in both styles, exactly twice an MG7 ratio pane: it has to stay
dominant and on this figure its dominance is height alone.

Axes blocks 6.975 × 9.512 in (MG7) and 7.285 × 9.880 in (user style); the
width is the seven-pane figure's, deliberately, so the two read at one scale.
Canvases 7.875 × 10.172 in and 8.275 × 10.530 in, declared rather than cropped
on the `R6_AXES` construction, so all four figures of a style come out one size
— 1574 × 2034 px and 2482 × 3159 px. Margins are this figure's own, measured
worst case over both observables and both variants: the **left** one is now set
by the sum pane's rotated name (0.569 in MG7, 0.630 in user style) where on the
seven-pane figure it was set by `Z_0Z_0/full`'s `0.000 / 0.025 / 0.050` tick
column, a pane this figure does not draw. Minimum ink clearance on the rendered
PNGs 19 px (MG7) and 30 px (user style) — nothing clipped. `--check-minus` goes
12/12 to **16/16**: the four new PDFs all carry the distribution pane's log
decade ticks.

### One thing that had to be fixed to draw them

`--check-minus` reported the two `M(e+ μ+)` six-panel PDFs as failures. They
were not. `wants_minus` asks every tick label the *locator* produced whether it
carries a minus, and `MaxNLocator` routinely offers one tick beyond each end of
the view that matplotlib never draws: the `Z_0Z_0/full` pane's locator offers
`−0.03` on a pane whose window starts at `−0.0056`. The figure was declared to
"want" a minus that is not on it, the check then looked for a `/minus` that
could not be there, and a correct figure was reported as broken.

`plot_zz_pol.py` now defines a **wrapper** that filters *tick* labels by their
axis's view interval and still asks axis labels, titles and legend entries
unconditionally — narrowing those could hide a genuinely eaten sign, which is
the one outcome the check exists to prevent. The sibling module in
`MadSpin/validation/zz_nlo/` is **not** edited. The count went 10/12 with two
spurious failures to **10/10**, and the two `M(e+ μ+)` figures were then
correctly reported as "no minus sign in this figure, check not applicable".

**The seventh pane changed that count, and the wrapper is what makes the new
count trustworthy.** Both `variant_A6_ratios*` figures now carry a log y axis
on the new pane, whose `10^−8` … `10^−3` tick labels are real minus signs that
are really drawn, so all twelve PDFs are now *applicable* and the run reports
**12/12**. The wrapper is unchanged and is still doing the work: it now says
"applicable" because the minus is inside the view, having said "not
applicable" when the only candidate was a tick the locator offered outside it.

---

## Not covered

* **The bin edges were not re-chosen.** `M(e+ μ+)`'s seven edges
  (`0, 45, 70, 95, 125, 175, 260, 450`) were picked for the 312-event inclusive
  sample and are kept unchanged so that the two passes' figures and per-bin
  tables are read against the same edges. At 118 521 events they are far
  coarser than the statistics now support — several tens of bins would still
  give a few percent per bin — and re-binning is the single cheapest
  improvement available, at no cost beyond re-running `plot_zz_pol.py`. The
  same holds for `Δφ`'s twelve.
* **The `M(e+ μ+)` overflow is not drawn.** Events above the last edge at 450
  GeV (the observable reaches 4.2 TeV) are counted in the fiducial σ and in the
  integrated ratios but fall outside the histogram, so the drawn curve does not
  integrate to the quoted σ. This was true of the previous pass too.
  **Correction:** this bullet used to say 3 120 / 3 088 / 3 134 *selected*
  events. Those are events with a finite `M(e+ μ+)`, i.e. before the fiducial
  cuts; the selected overflow is **880 / 924 / 889 / 887** (`madspin` /
  `onshell` / `PA` / `madspin_v1`), 0.686 / 0.730 / 0.720 / 0.702 % of each
  mode's selected σ. See *The two figure variants* → *Decision 1*, which is
  where it mattered.
* **`PA`'s `Δφ` shape difference is measured, not explained.** 57.61/11 with
  the rate removed, moving ~3 % of the rate out of `Δφ → π`, is the headline
  result of this comparison and no attempt was made to attribute it to a
  mechanism inside MadSpin. The decayed LHE files would answer it without any
  showering at all, and they sit beside the HepMC in each run directory (they
  are not committed either). **Confirmed since:** every `run_0X_decayed_1/`
  does carry both an `events.lhe.gz` and an `events_PYTHIA8_0.hepmc.gz`, so
  that route is available; it was still not taken.
* **No polarisation decomposition of `onshell` or `PA`.** Those three files
  carry the four `ms_pol_*` weights and this study deliberately uses only the
  reference's. (The fourth sample, `madspin_v1`, carries none at all and so
  could not join this even if it were attempted.) Whether `Z_0Z_0/full` is the
  same in those three modes is now the obvious next question — especially given `PA`'s `Δφ` shape effect — and it is
  entirely answerable from the three committed `.npz` without touching a HepMC
  file again.
* **`onshell`'s 5 % cross section is inferred, not verified.** The pole
  explanation is a reading of the numbers.
* **No third and fourth spinmode.** ~~`none`, `full`, `madspin_v1` and
  `onshell_v1` exist and were not run.~~ **Superseded in part:**
  `madspin_v1` has since been run (`run_10`) and is a fourth curve on variant
  B — see *The fourth sample*. `none`, `full` and `onshell_v1` still were not
  run.
* **No systematic on the polarisation frame.** The reference card sets
  `frame_id 24`. **Refined by the LO pass:** the LO card leaves it at the
  default 6, which is a *different* setting but the *same frame* on a `2 → 2`
  event, and that equality was checked against the LHE multiplicity rather
  than assumed — see *The fifth sample*. What is still not done is varying the
  frame on the NLO sample, where 6 and 24 genuinely differ; that would need a
  new MadSpin run and no cached weight can answer it.
* **No PDF variation of the polarisation fractions.** All 33 weights are
  summed and recorded in `data/meta.json`; the eighteen `1010`–`1027`
  alternative-PDF members are still only summed and never divided.
  **Superseded in part:** the *scale* variation of the fractions and of `K`
  was built by a later pass and is *The six-panel ratio figures, and the scale
  band* above. This bullet used to say "no scale **or** PDF".
* **No comparison against a `pure_interference` calculation.**
* **Bare vs dressed is quoted, not plotted.** Both are in the `.npz`; only the
  dressed ones are drawn.
* **Detector effects, isolation and lepton efficiency** are all absent; this is
  a generator-level fiducial only.
* **The three passes were timed concurrently**, so the per-file times are not
  independent serial measurements. They are within a second of one another and
  nothing in this study depends on them. (The fourth sample's 30.7 s **is** a
  serial measurement — it was run alone — and is the only figure in that column
  that can be read as a rate.)

### The K-factor figure specifically

* **One LO sample, so no LO spinmode comparison.** `onshell` and `PA` have no
  LO counterparts (there is no `run_11_decayed_1` and no `run_13`), so whether
  the K-factor differences reported here are a property of the process or
  partly of `spinmode madspin` is untestable on what exists. It would take two
  more LO runs and two more showers.
* **The K-factor is LO+PS against NLO+PS, not a fixed-order K.** Both sides are
  showered by the same Pythia8, so the shower is common and largely cancels,
  but "largely" is not "exactly" and no fixed-order cross-check was made.
* **~~No scale or PDF uncertainty on K.~~ DONE for scale, and this bullet was
  wrong about the cost.** The correlated envelope of the ratio was built by a
  later pass and is *The six-panel ratio figures, and the scale band* above.
  Two corrections to what this bullet said. (1) "All 33 weights of both samples
  are in the `.npz`" — they were **not**: only `MUR1.0_MUF1.0` of the nine was
  cached, so the pass had to re-read two HepMC files, ~32 s each. (2)
  "`+4.1 % −5.2 %`" is the **LO** sample's envelope, not both samples'; the NLO
  one is `+2.33 % / −1.92 %` over seven points. What remains undone here is the
  **PDF** uncertainty: the eighteen `1010`–`1027` members are cached only as
  sums in `meta.json`, not as per-event columns, so a PDF band would need a
  third HepMC pass. Every bar on the K-factor figure itself is still MC
  statistics only — the bands are on the two new `variant_A6_ratios*` figures,
  which are written beside it and do not touch it.
* **The `Z_TZ_0` low-`Δφ` excursion is explained qualitatively, not
  quantitatively.** "The LO spectrum is nearly empty there" is read off the
  shape ratios, not off a phase-space calculation.
* **The polarisation interference is not decomposed at LO.** `Σ/full` is
  1.00070 for LO against 1.00017 for NLO inclusively — both within a per-mille
  of 1 — and no per-bin comparison of the interference between the two orders
  was made, although the cached weights would support one.
* **`run_11` was not extracted.** It is a second, statistically independent LO
  sample at the same settings and would have halved the LO error on every
  number above, but it was never showered, so using it would have meant
  generating rather than reading — outside this pass's brief of one new HepMC.

### The two six-panel ratio figures specifically

* **No PDF uncertainty anywhere.** The eighteen `1010`–`1027` members are in
  `meta.json` as sums only, never as per-event columns, so a PDF band would
  need a third HepMC pass. It was not run and is not approximated.
* **The NLO polarised scale variation is a transported band, not a
  first-principles one.** `ms_pol_*` exists at the central scale only; see
  *Decision 3*. Exact at LO, an approximation at NLO, and the size of what it
  costs is bounded by the very component-to-component `K` spread the figure
  measures — a few percent, on a band of several. Closing it needs a MadSpin
  run that emits `ms_pol_*` per scale point, not a re-read.
* **`αs(M_Z)` and the shower scale are not varied.** `shower_scale_factor` is
  1.0 in both run cards and no variation of it was requested or reweighted, so
  the shower is common to the two orders and treated as cancelling. "Largely"
  is not "exactly", as the K-factor section already says.
* **The transposed `MUR`/`MUF` labels are diagnosed but not fixed upstream.**
  This pass established the permutation and worked under the true labels; it
  did not touch whatever writes the names. Anyone reading a `MUR*_MUF*` weight
  out of a Pythia8-showered HepMC from this MG5_aMC should check the ordering
  on their own file rather than trust this one — the check in *The two labels
  are transposed* takes one LO event and two minutes.
* **The band is not propagated into the χ² or the significances elsewhere in
  this file.** Every σ quoted in the earlier sections is still MC statistics
  only, and only the K-factor and the double ratio have been given a scale
  band. The mode-to-mode comparisons (`onshell`, `PA`, `madspin_v1`) are all at
  one order and their samples were not re-read, so no band exists for them.
* **The two dropped 7-point corners were not investigated per bin.** They are
  interior on the *integrated* `K` and that was checked; whether they stay
  interior in every bin of every component was not.
* **No re-binning.** These figures inherit the same seven and twelve edges as
  everything else here, for the same reason and with the same reservation.

### The seventh pane specifically

* **The band on it is at the line width and cannot be read off the canvas.** A
  few-percent envelope on a four-decade log axis is under two pixels. The
  numbers are in `numbers.txt` for all five components at both orders, and the
  six ratio panes below are where this figure's bands are read. Nothing was
  exaggerated to make it visible and nothing should be.
* **The components' bands are computed but not drawn.** `numbers.txt` prints
  them; the pane draws the unpolarised pair only. Widening it is one edit
  (`pol_analysis.RATIO6_TOP_BAND_KEYS`), and the pane was not tried with all
  ten bands beyond confirming that they overlap into illegibility.
* **`Z_0Z_T` and `Z_TZ_0` are still coincident.** The line-weight ladder makes
  both *visible*; it does not make them *separable*, because they are equal to
  1–2 % and no drawing can separate curves that are the same curve. A reader
  who needs them apart has the two dedicated fraction panes below and the
  per-bin table.
* **The overflow is still not drawn**, so the `M(e+ μ+)` pane's curves do not
  integrate to the σ printed beside them in `numbers.txt` — the same 0.69 %
  the *Not covered* bullet on the overflow describes, now visible on a pane
  that quotes an absolute normalisation. It was not fixed here.
* **No spinmode curves on it.** `onshell`, `PA` and `madspin_v1` carry no
  `MUR*_MUF*` columns and no LO partner, and this figure is the NLO/LO pair;
  they stay on the original figures and on variant B.

### The variants specifically

* **`onshell`'s residual `Δφ` shape difference was measured and not chased.**
  24.18/11, p = 0.012, a −1.1 % per radian slope at 2.2σ. It is real enough to
  be worth a sentence and too weak to interpret at twelve bins; finer binning
  would settle whether it is the same downward trend as `PA`'s at a quarter of
  the size or something else.
* **The `M(e+ μ+)` 95–125 GeV bin of `onshell` (−3.0 %, 2.98σ) was not
  followed up.** It is one bin out of seven, its p-value over the pane is 0.039
  and the coarse binning is inherited. It is flagged, not claimed.
* **No shape ratio of the polarisation components between modes.** The pane
  compares the three modes' *totals*. Whether `Z_0Z_0/full` differs between
  modes is the separate study *No polarisation decomposition of `onshell` or
  `PA`* above, and it is still not done.
* **The bin edges were not re-chosen for the variants either**, for the same
  reason and with the same cost: the shape pane is the place where the coarse
  `M(e+ μ+)` binning hurts most, because seven bins over a 4 TeV-wide
  observable cannot resolve a trend.
* **No paired error bar exists to compare against.** The pairing test came back
  negative, so the correlated treatment was never computed and the factor it
  would have bought here is unknown and, on these samples, unbuyable. The
  fourth sample came back negative on the same test, at a fourth distinct
  `n(w < 0)`.
* **The two variants' PDFs are timestamped from a later run than the two
  original PDFs.** The originals' *content* is unchanged — their PNGs are
  byte-identical and their PDF streams differ only in matplotlib's random Type1
  font-subset tag — and the two original PDFs were left at the bytes this
  branch already carried rather than rewritten for a tag change.
