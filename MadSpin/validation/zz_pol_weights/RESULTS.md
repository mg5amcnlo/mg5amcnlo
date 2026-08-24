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

```
data/weights.npz        250 000 rows: the weights and the observables (madspin)
data/meta.json          input path and size, event count, the 33 weight names,
                        the nominal-weight choice, the lepton selection, code
                        SHA, and the registry of all four samples
data/weights_onshell.npz, data/meta_onshell.json    the same, spinmode=onshell
data/weights_PA.npz,      data/meta_PA.json         the same, spinmode=PA
data/weights_madspin_v1.npz, data/meta_madspin_v1.json  the same, spinmode=
                        madspin_v1; 29 weight names, not 33 -- no ms_pol_*
extract_hepmc_pol.py    the one pass over a HepMC file, plain or gzipped
pol_analysis.py         the weight algebra, the selection, the ratio statistics
plot_zz_pol.py          the MG7-style figures, and numbers.txt
plot_zz_pol_userstyle.py  the same figures in the user style
plots/, plots_userstyle/  PDF and PNG
plots/variant_A_madspin_only/, plots_userstyle/variant_A_madspin_only/
plots/variant_B_shape_ratio/,  plots_userstyle/variant_B_shape_ratio/
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
* **No systematic on the polarisation frame.** The card sets `frame_id 24`.
* **No scale or PDF variation of the polarisation fractions.** All 33 weights
  are summed and recorded in `data/meta.json`; the study divides only the
  central ones.
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
