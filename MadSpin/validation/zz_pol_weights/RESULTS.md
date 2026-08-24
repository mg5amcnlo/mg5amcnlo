# MadSpin polarisation weights on a showered NLO `p p > z z`

What this is: one streaming pass over each of **three** showered HepMC2 files
— the same `p p > z z` process, the same run card and the same 250 000 events,
differing only in MadSpin's `spinmode` — reduced to three 7 MB `.npz`, and two
figures made off them. The default (`madspin`) sample carries
`keep_weight_for_polarization_vector = [0, T]` weights for both `z` and is the
one the polarisation decomposition is of; the other two enter the distribution
pane of each figure as a mode comparison.

Base of this branch: `633f9c6fa` (`MadSpin zz_pol_weights: three more layout
edits, no numbers touched`), itself on `b0e4472bc`.

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

## The pass

| spinmode | run | on disk | inflated | lines | events | **one pass** | output |
|---|---|---|---|---|---|---|---|
| `madspin` (default) | `run_02_decayed_1` | 14.061 GB | — | 142 420 114 | 250 000 | **34.0 s** | `weights.npz`, 7.0 MB |
| `onshell` | `run_05_decayed_1` | **4.643 GB** `.gz` | 14.061 GB | 142 596 393 | 250 000 | **33.7 s** | `weights_onshell.npz`, 6.9 MB |
| `PA` | `run_03_decayed_1` | **4.658 GB** `.gz` | 14.086 GB | 142 633 550 | 250 000 | **34.6 s** | `weights_PA.npz`, 7.0 MB |

All three are HepMC2 `IO_GenEvent`, `HepMC::Version 2.06.09`, from
`.../PROCNLO_loop_sm_7/Events/`.

The two gzipped files are streamed through the system `gzip -dc` in a child
process — never a temporary on disk, never the `gzip` module. **The compressed
pass costs nothing**: 33.7 s and 34.6 s against the plain file's 34.0 s, because
the child inflates on its own core while this process parses, and the parser is
the bottleneck either way. A serial `gzip`-module read would have added the
full ~14 s of inflation to each pass. The progress line for a `.gz` deliberately
prints no ETA: `os.path.getsize` gives the *compressed* size, which is not the
target the inflated byte count is walking towards.

Read in binary, line by line, with a 4 MB buffer; nothing but the current
event's leptons and photons is ever in memory. `P` lines are split with
`maxsplit=9`, which stops the split as soon as the status field is in hand and
never touches the colour-flow tail — that is most of the 34 s saved.

The `E` line's weight block is found by walking *forward* through the
random-state count (`E <10 fields> n_random <random...> n_weights <weights...>`),
not by slicing back from the end of the line. All three files have
`n_random = 0`, so the two agree here; a file that did not would have silently
shifted the polarisation weights onto the scale variations. The count found is
asserted against the `N` line every event, and the `N` line's names are compared
against that file's own first event's names on **every one of the 250 000 events
of each of the three files**, not sampled. They are constant within each file,
and identical across the three; see the spinmode section.

---

## Which weight is the full

Below is the `madspin` (reference) sample. The same question is re-derived from
scratch for the other two further down, and comes out the same way.

**`"Weight"`.** But the more useful statement is that the question has no
consequence for the ratios, because `"0"` and `"Weight"` are the *same weight in
two normalisations*:

| evidence | value |
|---|---|
| `"0" × N_events / "Weight"`, min and max over 250 000 events | `1.000000000000000` / `1.000000000000000` |
| `sum("0")` | 13.115048182696 pb |
| `mean("Weight")` | 13.115048182696 pb |
| `C` line of the **last** event | 13.115048182689 pb |
| `C` line of the **first** event | 5.891066e-05 pb |
| `sum("MUR1.0_MUF1.0") / sum("Weight") − 1` | 4.4e-06 |
| median&#124;LL+TT+TL+LT&#124; / median&#124;`"Weight"`&#124; | 0.899 |

Reading that table:

* `"0" × 250000 == "Weight"` **to the last bit, every event**. So `sum("0")` is
  the cross section and `mean("Weight")` is the same number. Both reproduce the
  `C` line of the last event to 1e-11. Every ratio in this study is identical
  whichever is used; only the vertical scale of the distributions differs.
* The `C` line of the *first* event (5.9e-05) is Pythia's running cross-section
  estimate after one event, not the sample's cross section. Anchoring on it —
  the obvious mistake, since it is the first `C` line a streaming reader meets —
  would have been wrong by a factor of 2.2e5.
* The four `ms_pol_*` weights are of order 10, i.e. in the `"Weight"`
  normalisation. Dividing them by `"0"` would have inflated every ratio by
  250 000. **This is the decisive check**, and it is why the ratios use
  `"Weight"` while the absolute `dσ/dx` of the distributions uses `"0"`.
* `"MUR1.0_MUF1.0"` is not an independent candidate: it is Pythia's rename of
  the LHE `<wgt id='1001'>`, the same central scale, agreeing with `"Weight"` to
  the rounding of the LHE `rwgt` block.

Sanity: **13 687 of 250 000 events (5.475 %) carry a negative weight**. The banner's `<init>` quotes `XSECUP = 13.139183 ± 0.036376` pb;
the event sample gives 13.115048 pb, 0.7σ away.

**Correcting the first pass on `|Weight|`.** That pass reported "`|Weight|`
takes two values, 14.727665 and 15.625049 (MC@NLO)", which reads as two MC@NLO
branches. Counted: **249 999 events carry 14.727665 — the banner's `XWGTUP`
exactly — and exactly one event (index 185 691) carries 15.625049.** It is a
single-event anomaly worth 6.2e-05 pb in a 13.1 pb total, not a second branch.
The `onshell` and `PA` files each carry exactly **one** `|Weight|` value, equal
to their own banner `XWGTUP` (15.359319 and 14.684339). Whatever produced the
odd event in `run_02` did not recur in the other two runs; it has no
consequence for anything in this study, and it is recorded because the earlier
description of it was wrong.

---

## Contradicting the brief: this is not an `e+e- mu+mu-` sample

The brief says "with `z > e+ e-` and `z > mu+ mu-`". It is not. The MadSpin
card in the banner is

```
define light = 1 2 3 4 5 -1 -2 -3 -4 -5 11 12 13 14 15 16 -11 -12 -13 -14 -15 -16
decay z > light light
```

— every fermion but the top, i.e. an **inclusive `Z` decay**. Reading each `z`'s
channel out of the event record (mark the `z`'s end vertex, follow the chain of
status-44 copies, take the particles born at the final vertex), of 250 000
events:

* **572 (0.229 %)** have one `z → e+e-` **and** the other `z → μ+μ-`;
* 17 677 (7.07 %) have at least one `z → e+e-`.

Cross-checked against the decayed LHE, which gives exactly 572 and 17 677.

The consequence is not cosmetic:

* **`M(e+ μ+)`, the cross-pair mass, exists for two events in a thousand.**
  After the fiducial cuts, **312 events**. That figure is statistics-limited by
  the decay card and by nothing else; its ratio panes are honest but they are
  mostly noise, and nothing in it is a 3σ statement.
* `Δφ(e+e-)` needs only one `z`, so it keeps 11 984 events and is the figure
  that actually resolves the physics.

If a follow-up wants the `M(e+ μ+)` figure to mean something, the fix is a
MadSpin card with `decay z > e+ e-` and `decay z > mu+ mu-`, not more showering
of this sample.

A second note on the event record, for whoever reads the next HepMC file: this
writer maps *final* particles to status 1 and everything else to
`abs(pythia_status)`. A hard-process lepton that never branches is therefore
status **1**, not status 23, and counting status-23 leptons undercounts the hard
leptons by a third. Truth matching here goes through the vertex chain instead.

---

## The lepton selection

**Highest-pT final-state (status 1) lepton per flavour and charge**, dressed
with every final-state photon within `ΔR < 0.1`, then `pT > 10 GeV` and
`|η| < 2.5`.

* *Highest-pT per flavour and charge* is the obvious default and it is what is
  used. The sample is showered and hadronised, so `e+` is not a unique object:
  a hadronic `Z` makes more of them in `b` and `c` decays.
* *Dressing* is done, at the run card's own `rphreco = 0.1` — the same
  recombination radius the fixed-order calculation used to define its leptons,
  so the showered lepton is the object the matrix element was integrated for.
  It matters for the mass and not for the angle: `M(e+ μ+)` moves by **+2.49 GeV
  on average** (rms 9.06), with 16.0 % of selected events moving by more than
  1 %; `Δφ(e+e-)` moves by −4.8e-05 rad (rms 2.7e-03), 1.4 % of events by more
  than 1 %.
* *`pT > 10 GeV`, `|η| < 2.5`* is the standard four-lepton fiducial. Dropping
  the `η` cut would take the four-lepton sample from 312 to 558 events, but the
  extra "leptons" reach `|η| = 8` and correspond to no measurement, and the
  truth-matched purity does not improve enough to pay for that.

**How often the choice is ambiguous**, measured rather than asserted — the
fraction of *selected* events carrying a **second** same-flavour same-sign
lepton:

| selection | flavour | above 3 GeV | above 10 GeV |
|---|---|---|---|
| `M(e+ μ+)` (312 ev) | e+ | 0.64 % | 0.00 % |
| | e− | 0.00 % | 0.00 % |
| | μ+ | 0.96 % | 0.64 % |
| | μ− | 0.96 % | 0.64 % |
| `Δφ(e+e-)` (11 984 ev) | e+ | **4.61 %** | **2.15 %** |
| | e− | **4.35 %** | **2.24 %** |

The `Δφ` number is the one that matters and it is not negligible: **4.6 % of the
selected events carry a second `e+` above 3 GeV, and 2.2 % carry one above
10 GeV.** The reason is structural — in this inclusive sample the *other* `Z`
usually went hadronic, and `b`/`c` decays inside it make extra electrons. In the
four-lepton selection the other `Z` went to muons and the ambiguity all but
disappears.

Truth-matched, the selections come out at

| | selected | truth events | purity | efficiency |
|---|---|---|---|---|
| `M(e+ μ+)` | 312 | 572 | 0.904 | 0.493 |
| `Δφ(e+e-)` | 11 984 | 17 677 | 0.964 | 0.654 |

Truth is used **only** for these diagnostics. Every plotted observable is built
from final-state particles alone.

---

## The other two spinmodes

`run_05` is `set spinmode onshell`, `run_03` is `set spinmode PA`; `run_02` sets
no `spinmode` at all, which is MadSpin's default, `madspin`. Everything else in
the three cards is identical — same process, same `decay z > light light`, same
`frame_id 24`, same `keep_weight_for_polarization_vector = [0, T]`, same 250 000
events.

### What the `N` lines actually say

Checked per file the way the first was checked: every one of the 250 000 `N`
lines compared against that file's own first, not sampled. **Constant within
each file**, and the three files name the **same 33 weights in the same
order** — `"0"`, the eighteen `1010`–`1027` alternative-PDF members, the nine
`MUR*_MUF*`, `"Weight"`, and the four `ms_pol_23.*_23.*`.

So both new files **do** carry polarisation weights, which they need not have:
they are separate MadSpin runs and a different `spinmode` is a different
reweighting. That they carry them is recorded and **nothing here is built on
it.** Decomposing `onshell` and `PA` into `Z_0Z_0 … Z_TZ_T` is a different
study, and the ratio panes of these figures belong to the sample whose
decomposition they are. The extractor was nonetheless changed to *tolerate*
their absence — `"0"` and `"Weight"` are required, the rest are kept if named
and reported as absent if not — so that the next file with a different weight
set produces a note rather than a crash.

### The nominal, re-derived per file

| spinmode | `"0"×N/"Weight"` min/max | `sum("0")` = `mean("Weight")` | `C` last event | `C` first event |
|---|---|---|---|---|
| `madspin` | 1.000000000000000 / 1.000000000000000 | 13.115048183 pb | 13.115048183 pb | 5.891e-05 pb |
| `onshell` | 1.000000000000000 / 1.000000000000000 | 13.667705043 pb | 13.667705043 pb | 6.144e-05 pb |
| `PA` | 1.000000000000000 / 1.000000000000000 | 13.078107263 pb | 13.078107263 pb | 5.874e-05 pb |

**It came out the same, on all three, and it came out the same because it was
re-derived and not because it was carried over.** `"0" × N_events == "Weight"`
to the last bit on every event of every file; `sum("0") == mean("Weight")`
equals that file's **last** `C` line to 1e-11; and the **first** `C` line is
Pythia's running estimate after one event, low by 2.2e5 in every file — the same
trap, three times. Each sample's distributions are scaled by its **own**
`1/n_events`.

`sum("MUR1.0_MUF1.0")/sum("Weight") − 1` is 4.4e-06, −2.1e-05 and 2.4e-05 — the
LHE `rwgt` block's rounding in each case. Negative-weight fractions agree:
5.475 %, 5.507 %, 5.469 %.

### The comparison

Same lepton selection, same bin edges, same absolute normalisation. Two notes
on the statistics, both of which change the reading:

* the error here is the **plain quadrature sum**, the opposite of what every
  ratio pane in this study uses. Those divide two weights of the *same* events
  and must keep the covariance; these are three independent runs, independently
  showered, with no covariance to keep.
* **two χ² are quoted and only the pair is the result.** One compares the
  distributions as drawn, rate included; the other rescales to a common
  fiducial rate first and so tests the shape alone, on one fewer degree of
  freedom. A mode that differs only in normalisation gives a large first χ²
  and a small second one, and calling that a shape difference would be wrong.
  It is exactly what happens below.

**Inclusive cross section, all 250 000 events, no selection:**

| spinmode | σ | vs `madspin` | banner `XSECUP` |
|---|---|---|---|
| `madspin` | 13.115048 ± 0.029455 pb | — | 13.139183 ± 0.036376 |
| `onshell` | **13.667705 ± 0.030719 pb** | **+4.21 %** | 13.634223 ± 0.028026 |
| `PA` | 13.078107 ± 0.029369 pb | −0.28 % | 13.033217 ± 0.029759 |

Each sample reproduces its own banner (0.5σ, 0.8σ, 1.1σ), so these are the
three runs' own cross sections and not a reading error. **`onshell` sits 4.2 %
above the other two** — 13 σ on the statistical errors, which are tiny here —
while `PA` and `madspin` agree to 0.3 %, which is 0.9σ. The obvious explanation
is that the on-shell mode puts both `Z` exactly on their pole and so does not
pay the off-shell/Breit-Wigner truncation the other two do; **that is an
inference from the numbers, not something this study verified**, and it is the
kind of claim the `MadSpin/validation/zz_loopinduced` comparison against a
fully off-shell calculation is the right place to settle.

**`Δφ(e+e-)` — 11 984 / 11 366 / 11 478 survivors:**

| spinmode | N | fiducial σ | rate ratio | χ²/ndf as drawn | χ²/ndf shape only |
|---|---|---|---|---|---|
| `madspin` | 11 984 | 0.636824 ± 0.006449 pb | — | — | — |
| `onshell` | 11 366 | 0.626169 ± 0.006550 pb | 0.9833 ± 0.0143 (1.2σ) | 14.69/12 = 1.22 (p 0.26) | 13.37/11 = 1.22 (p 0.27) |
| `PA` | 11 478 | 0.603937 ± 0.006293 pb | **0.9484 ± 0.0138 (3.7σ)** | 26.05/12 = 2.17 (p **0.011**) | 12.81/11 = 1.16 (p 0.31) |

This is the figure that carries the comparison, and what it shows is:

* **The three modes agree on the *shape* of `Δφ(e+e-)`.** Rescaled to a common
  fiducial rate, both shape χ² are at their number of degrees of freedom
  (1.22 and 1.16, p = 0.27 and 0.31). There is no shape difference in this
  observable at 11 k events.
* **They disagree on the *rate*.** `PA` is 5.2 % below `madspin`, 3.7σ;
  `onshell` is 1.7 % below, 1.2σ. `PA`'s "as drawn" χ² of 26.05/12 — the one
  that looks like a 2.5σ disagreement — is **entirely** that normalisation
  offset: removing it takes the χ² from 26.05 to 12.81 on eleven degrees of
  freedom. Quoting the first χ² alone would have claimed a shape effect this
  data does not contain.
* **The rate difference is channel bookkeeping, not acceptance.** Writing the
  fiducial cross section as σ_tot × *f*(`z→e+e-`) × acceptance:

  | spinmode | σ_tot | *f*(`z→e+e-`) | product | fiducial σ | acceptance |
  |---|---|---|---|---|---|
  | `madspin` | 13.115048 | 0.070708 | 0.927339 | 0.636824 | 0.6867 |
  | `onshell` | 13.667705 | 0.067104 | 0.917158 | 0.626169 | 0.6827 |
  | `PA` | 13.078107 | 0.067448 | 0.882092 | 0.603937 | 0.6847 |

  **The acceptance is the same in all three to 0.6 %.** `onshell`'s 4.2 %
  higher inclusive cross section is cancelled almost exactly by its 5.1 % lower
  `z→e+e-` fraction, which is why its 4.2 % never reaches the figure. `PA`'s
  deficit is its 4.6 % lower `z→e+e-` fraction, essentially undiluted. So the
  whole of the visible difference between the modes in this observable is
  (i) the inclusive cross section and (ii) how often the `z` went to electrons
  — see the next subsection — and none of it is the selection.

**`M(e+ μ+)` — 312 / 306 / 317 survivors — cannot compare the modes at this
statistics, and is not drawn.**

| spinmode | N | fiducial σ | rate ratio | χ²/ndf vs `madspin` |
|---|---|---|---|---|
| `madspin` | 312 | 0.016848 ± 0.001041 pb | — | — |
| `onshell` | 306 | 0.017202 ± 0.001075 pb | 1.021 ± 0.090 (0.2σ) | 4.50/7 = 0.64 |
| `PA` | 317 | 0.016858 ± 0.001046 pb | 1.001 ± 0.088 (0.0σ) | 5.05/7 = 0.72 |

Both χ² are *below* their number of degrees of freedom (shape only: 0.74 and
0.84), and both rate ratios are 1 within 0.2σ. The per-bin errors are 20–50 %, and the per-bin ratios wander
between 0.62 and 1.47 with no bin more than 1.7σ from the reference. Three such
curves laid over one another would show gaps of 20–40 % that the eye reads as a
mode difference and that are entirely the noise of ~45 events per bin. **So the
two extra curves are not drawn on that figure**; the pane says so in as many
words, and the numbers above are the whole of what can honestly be said about
it. The rule is `pol_analysis.MIN_SEL_TO_DRAW = 2000`, applied per sample per
observable.

This is the `decay z > light light` card again. More showering would fix it
only at an absurd price: `M(e+ μ+)` keeps 312 events where `Δφ(e+e-)` keeps
11 984, so matching the latter's statistics needs **9.6 million showered events
per mode, about 540 GB of HepMC each**. The fix is a MadSpin card with
`decay z > e+ e-` and `decay z > mu+ mu-`, as the reference sample's own section
above already says.

### A difference in the decay channels

Read off the event record (truth, used to categorise only):

| spinmode | one `z→e+e-` **and** one `z→μ+μ-` | at least one `z→e+e-` |
|---|---|---|
| `madspin` | 572 (0.2288 %) | 17 677 (7.071 %) |
| `onshell` | 604 (0.2416 %) | 16 776 (6.710 %) |
| `PA` | 637 (0.2548 %) | 16 862 (6.745 %) |

The `z→e+e-` fractions are **not** the same across modes: 17 677 against 16 776
is a 5.1 % difference and 4.8σ on Poisson errors. The decay table is identical
in the three cards, so this is the accept/reject: the channel and the kinematics
are chosen together, and a mode that weights the production/decay correlation
differently ends up with different channel proportions among the events it
keeps. **It is the whole of why the fiducial `Δφ` rates differ** — the
acceptance table above shows the selection itself treats the three identically —
and it is reported here as an observation: this study did not trace it to a
specific line of MadSpin and does not claim to have. It is the natural thing to
chase next, and it is cheap to chase, because the decayed LHE files answer it
without any showering at all.

---

## The four ratios, and their sum against 1

Integrated. Errors are the correlated ones (next section).

**All 250 000 events, no selection:**

| | ratio | error |
|---|---|---|
| LL / full | 0.05840 | 0.00034 |
| TT / full | 0.69610 | 0.00108 |
| TL / full | 0.12281 | 0.00056 |
| LT / full | 0.12240 | 0.00057 |
| **(LL+TT+TL+LT) / full** | **0.99971** | **0.00169** |

**0.17σ from 1.**

| selection | LL | TT | TL | LT | **sum** | σ from 1 |
|---|---|---|---|---|---|---|
| all events | 0.0584 | 0.6961 | 0.1228 | 0.1224 | **0.99971 ± 0.00169** | 0.17 |
| `M(e+ μ+)` | 0.0619 | 0.6871 | 0.1177 | 0.1258 | **0.99261 ± 0.01252** | 0.59 |
| `Δφ(e+e-)` | 0.0582 | 0.6943 | 0.1197 | 0.1244 | **0.99650 ± 0.00429** | 0.82 |

### What the sum being 1 means

The brief expected the sum *not* to equal the full and said that if it did, that
was a finding worth checking rather than a relief. It is a finding, and it is
the expected one.

Integrated over the **whole** sample the sum is 1 to 0.03 % and 0.2σ. That is
not the absence of interference. The interference between different
polarisations of the *same* `z` carries a non-trivial dependence on the decay
angles — it is the term that is odd under the azimuthal and polar structure the
four diagonal weights are not — and it **integrates to zero over the full decay
phase space**. So an unrestricted sum *must* come back to 1, and the fact that
it does to 3e-04 is a check on the weights themselves: it says MadSpin's four
`ms_pol` weights and its nominal weight are mutually consistent and correctly
normalised.

The interference is real and is visible the moment you look differentially, or
apply any cut that restricts the decay angles — which every lepton selection
does. Both selected samples sit slightly **below** 1 (0.9926 and 0.9965) for
exactly that reason, though neither departure is individually significant.

### The sum as a function of each observable

`Δφ(e+e-)` — this is where the interference shows up, and it is a large,
significant, and *shaped* effect (full table in `numbers.txt`):

| Δφ [rad] | N | sum/full | error | σ from 1 |
|---|---|---|---|---|
| 0.0 – 0.3 | 399 | 1.0033 | 0.0171 | 0.2 |
| 1.0 – 1.3 | 638 | **1.0390** | 0.0191 | 2.1 |
| 1.6 – 1.8 | 940 | **1.0359** | 0.0185 | 1.9 |
| 2.1 – 2.4 | 1362 | 1.0018 | 0.0131 | 0.1 |
| 2.6 – 2.9 | 1902 | 0.9718 | 0.0106 | 2.7 |
| 2.9 – 3.1 | 1918 | **0.9460** | 0.0097 | **5.6** |

**+4 % at intermediate `Δφ`, −5.4 % at `Δφ → π`, and the two cancel on
integration.** The last bin alone is 5.6σ from 1. This is the polarisation
interference — the same quantity the `pure_interference` machinery in this
repository exists to isolate — measured differentially and behaving exactly as
an interference term must: it moves rate around without changing the total.

`M(e+ μ+)` — 312 events, so the bin-by-bin sum wanders between 0.95 and 1.02
with 1.4σ excursions at most. **Nothing in that pane is significant** and it
should not be read as structure. It is shown because it is what the data
supports, not because it says anything.

### What the components themselves show

* `TT/full` runs from 0.84 at `Δφ ≈ 0.4` down to **0.54** at `Δφ → π`, while
  `LL/full` runs the other way, 0.021 → 0.082. Back-to-back leptons in the
  transverse plane are where the longitudinal `Z` lives; that is the whole basis
  of the polarisation observable and it is reproduced cleanly.
* `TL` and `LT` agree with each other everywhere within errors (0.1228 vs
  0.1224 integrated), which they must by the symmetry of the two identical
  `Z`, and is a further check that the weight-to-name mapping was not
  transposed.
* In `M(e+ μ+)`, `LL/full` peaks near the `Z` mass region (0.093 at 70–125 GeV)
  and falls to 0.004 above 260 GeV while `TT/full` rises to 0.88 — the
  high-mass cross-pair tail is transverse, as expected for a high-`p_T`
  `ZZ` system.

---

## The ratio errors

Numerator and denominator are sums over the **same events**, and an event's
polarisation weight is strongly correlated with its nominal weight. The error
used everywhere is the linearised (delta-method) one for a ratio of two sums
over a common sample:

```
R = N/D,  N = sum_i n_i,  D = sum_i d_i
var(R) = sum_i (n_i - R d_i)^2 / D^2
```

which is algebraically the jackknife over events, and is the naive
independent-samples formula minus the covariance term. When `n_i = c d_i` event
by event it correctly returns **zero**, which the naive formula does not.

How much this matters, on the sum where the correlation is strongest:

| sample | correct error | naive error | factor |
|---|---|---|---|
| all events | 0.00169 | 0.00360 | 2.1× too large |
| `Δφ(e+e-)` | 0.00429 | 0.01490 | 3.5× too large |
| `M(e+ μ+)` | 0.01252 | 0.08763 | **7.0× too large** |

With naive bars, the 5.6σ last `Δφ` bin would have read as roughly 1.6σ and the
central result of this study would have been invisible.

The distributions in the top pane carry the usual `sqrt(sum w^2)` per bin; those
are genuinely independent between bins and need no such treatment.

---

## Figures

`plots/` (MG7 style, usetex, `--check-minus` passes: 2/2 PDFs carry `/minus`)
and `plots_userstyle/` (stock rcParams), each with `m_epmup` and `dphi_ee` in
PDF and PNG. Each figure is three tiers:

1. the distribution — the full (unpolarised) `dσ/dx` at absolute normalisation
   with `LL`, `TT`, `TL`, `LT` overlaid in the same binning, **and the full
   (unpolarised) `dσ/dx` of the `onshell` and `PA` spinmodes** where the
   statistics support it;
2. a **full-width pane for `(LL+TT+TL+LT)/full`**, on its own vertical scale,
   with the integrated value printed in it (the significance is quoted here and
   in `numbers.txt`, not on the figure);
3. a 2 × 2 breakdown, `LL/full`, `TT/full`, `TL/full`, `LT/full`.

Tiers 1 and 2 are a stacked **pair** sharing one x axis: the tick labels and
the axis name go under the sum pane only, with a tight gap between the two. The
2 x 2 is a separate block with its own labels, set off by a modest gap. Three
different vertical gaps means three gridspecs rather than one `hspace` -- tight
enough for the pair leaves the sum pane's axis name sitting on the 2 x 2, and
wide enough to clear it tears the pair apart.

**The two extra spinmodes enter tier 1 and nowhere else.** Tiers 2 and 3 are
the polarisation decomposition of the `madspin` sample; a second sample's
numerator over a first sample's denominator is not a quantity, and each extra
sample's own decomposition is a different study. They are drawn as full curves
— steps with error bars, in the two hues the four components leave free
(orange and cyan, dashed and dash-dot-dot) — rather than in a component's
style, because that is what they are.

Tier 1 draws an extra spinmode only where the selection leaves it at least
`pol_analysis.MIN_SEL_TO_DRAW = 2000` events. On `Δφ(e+e-)` (11 366 and 11 478)
both are drawn. On `M(e+ μ+)` (306 and 317) **neither is**, and the pane says
so: *"spinmode onshell (N=306) and PA (N=317) not drawn: too few events for a
comparison at this selection"*. A figure that silently omitted two of its three
samples would imply a comparison it never made; one that drew them would let
the eye read three noise realisations of ~45 events per bin as a mode
difference. The threshold is not a taste: at those numbers the per-bin errors
are 20–50 % and both χ² come in *below* their degrees of freedom.

The curves carry physics names only — `Z_0 Z_0` / `Z_0 Z_T` / `Z_T Z_0` /
`Z_T Z_T`, in the ratio pane labels as well as in the legend — except that the
three full curves now name their spinmode, because three unlabelled "full"
curves on one pane would be unreadable. The reference is
`full (unpolarised), spinmode = madspin`; the others are `full, spinmode =
onshell` and `full, spinmode = PA`. Which weight column each sum is is a fact
about the file and lives in `numbers.txt` (which prints the mapping table
explicitly), in `data/meta.json`'s `pol_map`, and in the table below:

| figure name | short key | weight column |
|---|---|---|
| `Z_0 Z_0` | LL | `ms_pol_23.0_23.0` |
| `Z_T Z_T` | TT | `ms_pol_23.T_23.T` |
| `Z_T Z_0` | TL | `ms_pol_23.T_23.0` |
| `Z_0 Z_T` | LT | `ms_pol_23.0_23.T` |
| `full (unpolarised), spinmode = madspin` | full | `Weight` of `weights.npz` |
| `full, spinmode = onshell` | — | `Weight` of `weights_onshell.npz` |
| `full, spinmode = PA` | — | `Weight` of `weights_PA.npz` |

Tier 2 does not share a scale with anything, because it is the physics and a
shared window would squash it. The four small panes do not share one either,
and for a different reason: `TT/full ≈ 0.70` and `LL/full ≈ 0.06`, so a single
window would either clip the first or flatten the rest into a line. Each is
autoscaled around its own integrated value, drawn as a dashed line.

---

## Not covered

* **No second observable with real four-lepton statistics.** The 312-event
  `M(e+ μ+)` figure is the ceiling this sample allows; see the decay-card note.
  With three samples this is now the binding limit on the mode comparison too:
  `M(e+ μ+)` cannot separate the spinmodes at 306–317 events and its extra
  curves are not drawn.
* **No polarisation decomposition of `onshell` or `PA`.** Both files carry the
  four `ms_pol_*` weights and this study deliberately does not use them. Whether
  `Z_0Z_0/full` is the same in the three modes is the obvious next question and
  it is entirely answerable from the three committed `.npz` without touching a
  HepMC file again.
* **The `z→e+e-` fraction difference between the modes is measured, not
  explained.** 7.071 % / 6.710 % / 6.745 % is a 4.8σ effect and it is the whole
  of the fiducial rate difference; no attempt was made here to attribute it to
  a mechanism inside MadSpin, and the decayed LHE files would answer it without
  any showering.
* **No third and fourth spinmode.** `none`, `full`, `madspin_v1` and
  `onshell_v1` exist and were not run; only `run_02`, `run_03` and `run_05`
  were available.
* **No systematic on the polarisation frame.** The card sets `frame_id 24`; the
  weights are what MadSpin produced in that frame and no alternative frame was
  generated, so the frame dependence of the four fractions is not measured here.
* **No scale or PDF variation of the polarisation fractions.** All 33 weights
  are summed and recorded in `data/meta.json`, but the study divides only the
  central ones; the `1010`–`1027` groups are alternative-PDF scale variations
  and were not propagated to the ratios.
* **No comparison against a `pure_interference` calculation.** The sum's
  departure from 1 is *identified* as the interference on physics grounds and by
  its integrating to zero; it was not checked against an independent computation
  of that term in this repository.
* **Bare vs dressed is quoted, not plotted.** Both are in the `.npz`
  (`m_epmup`/`m_epmup_dr`, `dphi_ee`/`dphi_ee_dr`); only the dressed ones are
  drawn.
* **Detector effects, isolation, and lepton efficiency** are all absent; this is
  a generator-level fiducial only.
