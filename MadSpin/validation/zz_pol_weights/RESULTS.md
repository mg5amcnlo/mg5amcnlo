# MadSpin polarisation weights on a showered NLO `p p > z z`

What this is: one streaming pass over each of **three** showered HepMC2 files
— the same `p p > z z` process, the same run card and the same 250 000 events,
differing only in MadSpin's `spinmode` — reduced to three 22 MB `.npz`, and two
figures made off them. The `madspin` sample carries
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

```
data/weights.npz        250 000 rows: the weights and the observables (madspin)
data/meta.json          input path and size, event count, the 33 weight names,
                        the nominal-weight choice, the lepton selection, code
                        SHA, and the registry of all three samples
data/weights_onshell.npz, data/meta_onshell.json    the same, spinmode=onshell
data/weights_PA.npz,      data/meta_PA.json         the same, spinmode=PA
extract_hepmc_pol.py    the one pass over a HepMC file, plain or gzipped
pol_analysis.py         the weight algebra, the selection, the ratio statistics
plot_zz_pol.py          the MG7-style figures, and numbers.txt
plot_zz_pol_userstyle.py  the same figures in the user style
plots/, plots_userstyle/  PDF and PNG
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
| `bands: ±2 %, ±5 %` (user style only) | `numbers.txt`; the two bands are still drawn, only the words are gone |
| `spinmode ... not drawn: too few events` | `numbers.txt` prints `on the figure: ...` per observable. It does not trigger on these samples — all three modes are drawn on both figures |

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

All three are HepMC2 `IO_GenEvent`, `HepMC::Version 2.06.09`, from
`.../PROCNLO_loop_sm_7/Events/`. **The three passes were run concurrently**, on
separate cores of an 18-core machine; the times above are therefore within a
second of one another and none of them is a serial measurement. They are
smaller and faster than the previous pass's files (4.6 GB `.gz` / 14.1 GB
inflated / 142 M lines / ~34 s) for the obvious reason: an exclusive leptonic
`Z` decay makes no hadrons, so there is far less event record per event.

All three are streamed through the system `gzip -dc` in a child process — never
a temporary on disk, never the `gzip` module. The child inflates on its own
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
not by slicing back from the end of the line. All three files have
`n_random = 0`. The count found is asserted against the `N` line every event,
and the `N` line's names are compared against that file's own first event's
names on **every one of the 250 000 events of each of the three files**, not
sampled.

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

All three files name the **same 33 weights in the same order** — `"0"`, the
eighteen `1010`–`1027` alternative-PDF members, the nine `MUR*_MUF*`,
`"Weight"`, and the four `ms_pol_23.*_23.*`. **Identical to the previous
pass's list.** Constant within each file, checked on every one of the 250 000
`N` lines of each file against that file's own first, not sampled.

So all three files carry polarisation weights, which they need not have: a
different `spinmode` is a different reweighting. That they do is recorded and
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

`plots/` (MG7 style, usetex, `--check-minus` passes: 2/2 PDFs carry `/minus`)
and `plots_userstyle/` (stock rcParams), each with `m_epmup` and `dphi_ee` in
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

## Not covered

* **The bin edges were not re-chosen.** `M(e+ μ+)`'s seven edges
  (`0, 45, 70, 95, 125, 175, 260, 450`) were picked for the 312-event inclusive
  sample and are kept unchanged so that the two passes' figures and per-bin
  tables are read against the same edges. At 118 521 events they are far
  coarser than the statistics now support — several tens of bins would still
  give a few percent per bin — and re-binning is the single cheapest
  improvement available, at no cost beyond re-running `plot_zz_pol.py`. The
  same holds for `Δφ`'s twelve.
* **The `M(e+ μ+)` overflow is not drawn.** 3 120 / 3 088 / 3 134 selected
  events lie above the last edge at 450 GeV (the observable reaches 4.2 TeV).
  They are counted in the fiducial σ and in the integrated ratios but fall
  outside the histogram, so the drawn curve does not integrate to the quoted σ.
  This was true of the previous pass too and is stated here for the first time.
* **`PA`'s `Δφ` shape difference is measured, not explained.** 57.61/11 with
  the rate removed, moving ~3 % of the rate out of `Δφ → π`, is the headline
  result of this comparison and no attempt was made to attribute it to a
  mechanism inside MadSpin. The decayed LHE files would answer it without any
  showering at all, and they sit beside the HepMC in each run directory (they
  are not committed either).
* **No polarisation decomposition of `onshell` or `PA`.** All three files carry
  the four `ms_pol_*` weights and this study deliberately uses only the
  reference's. Whether `Z_0Z_0/full` is the same in the three modes is now the
  obvious next question — especially given `PA`'s `Δφ` shape effect — and it is
  entirely answerable from the three committed `.npz` without touching a HepMC
  file again.
* **`onshell`'s 5 % cross section is inferred, not verified.** The pole
  explanation is a reading of the numbers.
* **No third and fourth spinmode.** `none`, `full`, `madspin_v1` and
  `onshell_v1` exist and were not run.
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
  nothing in this study depends on them.
