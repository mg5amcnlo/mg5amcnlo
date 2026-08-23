# MadSpin polarisation weights on a showered NLO `p p > z z`

What this is: one streaming pass over a 14 GB showered HepMC2 file that carries
MadSpin's `keep_weight_for_polarization_vector = [0, T]` weights for both `z`,
reduced to a 7 MB `.npz`, and two figures made off that `.npz`.

Base of this branch: `b0e4472bc` (`MadSpin: validate the NLO path on
p p > z z, and stack the gg box on top`).

```
data/weights.npz        250 000 rows: the weights and the observables
data/meta.json          input path and size, event count, the 33 weight names,
                        the nominal-weight choice, the lepton selection, code SHA
extract_hepmc_pol.py    the one pass over the HepMC file
pol_analysis.py         the weight algebra, the selection, the ratio statistics
plot_zz_pol.py          the MG7-style figures, and numbers.txt
plot_zz_pol_userstyle.py  the same figures in the user style
plots/, plots_userstyle/  PDF and PNG
numbers.txt             every number quoted below, and the per-bin tables
```

The HepMC file is **not** committed and was never copied.

---

## The pass

| | |
|---|---|
| input | `.../PROCNLO_loop_sm_7/Events/run_02_decayed_1/events_PYTHIA8_0.hepmc` |
| size | 14.061 GB, HepMC2 `IO_GenEvent`, `HepMC::Version 2.06.09` |
| lines | 142 420 114 |
| events | 250 000 |
| **wall time, one pass** | **34.0 s** |
| output | `data/weights.npz`, 7.0 MB |

Read in binary, line by line, with a 4 MB buffer; nothing but the current
event's leptons and photons is ever in memory. `P` lines are split with
`maxsplit=9`, which stops the split as soon as the status field is in hand and
never touches the colour-flow tail — that is most of the 34 s saved.

The `E` line's weight block is found by walking *forward* through the
random-state count (`E <10 fields> n_random <random...> n_weights <weights...>`),
not by slicing back from the end of the line. This file has `n_random = 0`, so
the two agree here; a file that did not would have silently shifted the
polarisation weights onto the scale variations. The count found is asserted
against the `N` line every event, and the `N` line's names are compared against
the first event's names on **every one of the 250 000 events**, not sampled.
They are constant.

---

## Which weight is the full

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

Sanity: `|Weight|` takes two values, 14.727665 and 15.625049 (MC@NLO), and
**13 687 of 250 000 events (5.475 %) carry a negative one**. The banner's
`<init>` quotes `XSECUP = 13.139183 ± 0.036376` pb; the event sample gives
13.115048 pb, 0.7σ away.

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
   with `LL`, `TT`, `TL`, `LT` overlaid in the same binning;
2. a **full-width pane for `(LL+TT+TL+LT)/full`**, on its own vertical scale,
   with the integrated value printed in it (the significance is quoted here and
   in `numbers.txt`, not on the figure);
3. a 2 × 2 breakdown, `LL/full`, `TT/full`, `TL/full`, `LT/full`.

The 2 x 2 block is a separate block from the two full-width panes, so those two
carry their own x tick labels and axis name rather than borrowing the ones at
the foot of the figure; the vertical gap between blocks is opened with a nested
gridspec so that the 2 x 2 itself stays tight.  The nominal curve is labelled
`full (unpolarised)` -- for what it is, not for the weight column it was summed
from, which is a fact about the file and belongs above rather than on the plot.

Tier 2 does not share a scale with anything, because it is the physics and a
shared window would squash it. The four small panes do not share one either,
and for a different reason: `TT/full ≈ 0.70` and `LL/full ≈ 0.06`, so a single
window would either clip the first or flatten the rest into a line. Each is
autoscaled around its own integrated value, drawn as a dashed line.

---

## Not covered

* **No second observable with real four-lepton statistics.** The 312-event
  `M(e+ μ+)` figure is the ceiling this sample allows; see the decay-card note.
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
