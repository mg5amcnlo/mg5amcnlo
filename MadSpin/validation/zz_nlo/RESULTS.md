# `p p > Z Z` at NLO + MadSpin, and the `g g` contribution on top of it

> **Frame correction, 2026-08-27.** The angular observables of this study
> (`cos_theta1`, `cos_theta2`, `cos1cos2`, and the `f_0` / `f_00` / `f_TT` /
> `C_kk` coefficients built from them) were harvested with a boost composition
> that is not the helicity frame — the axis in the four-lepton frame, the `l+`
> boosted into its pair's rest frame straight from the lab. That composition
> carries a Wigner rotation and damps every rank-2 moment towards `1/3`.
> `observables.py` is fixed and guarded by a self-test; the numbers and plots
> here are **not** re-harvested, because the samples no longer exist on disk.
> Sample-to-sample comparisons (MadSpin vs truth, `chi2/ndf`, ratio panels) are
> unaffected — both sides were harvested identically. Absolute polarisation
> coefficients are biased towards isotropy. Full diagnosis:
> [`POLWEIGHT_CLOSURE_DIAGNOSIS.md`](../zz_loopinduced/POLWEIGHT_CLOSURE_DIAGNOSIS.md).

Setup, cards and re-run instructions are in [README.md](README.md). Raw numbers
are in [`data/numbers.txt`](data/numbers.txt) (Part 1) and
[`data/numbers_stack.txt`](data/numbers_stack.txt) (Part 2); everything below is
read off those and off [`data/meta.json`](data/meta.json).

Base: **`36e3a8b78`** — the loop-induced `g g > z z` study, whose parent is
`178e9542d` on `origin/madspin_density`, i.e. this branch carries PRs #375,
#376, #377 and #379. In particular MadSpin's reported cross section already
carries the Breit-Wigner truncation (#379); one normalisation is quoted
throughout, the reported one, and nothing is corrected by hand.

---

## 0. Summary

**Seven things this study found, three of which overturn the framing it started
from** (the first, the third and the fourth).

* **An NLO off-shell truth was cheap, not expensive.** `p p > e+ e- mu+ mu- / a
  [QCD]`, 50 000 events, cost **88 s** on 14 cores. It is in the study and every
  Part 1 ratio divides by it. §1.
* **All six spinmodes ran**, including the two the loop-induced study could not.
  Nothing was refused. `fixed_order` was off and is not needed for MC@NLO
  events. §3.
* **`ptheavy` does not exist on the NLO run card.** The `pt(Z) > 1 GeV` of this
  study is applied through `pt_min_pdg`, which exists on both card classes, and
  the two are measured to be the same cut. §2.
* **The NLO card's default lepton cuts are inert here** — the loop-induced
  study's 46 % trap does not repeat, and that is measured, not assumed. §2c.
* **`madspin`, `PA` and `madspin_v1` reproduce the off-shell NLO truth to
  `+0.37 ± 0.53 %`**; `onshell`, `none` and `onshell_v1` are `+4.96 %`, which is
  exactly `1/f²` more. §5.
* **The one place MadSpin visibly fails at NLO is `m_4l` below `2 m_Z`.** The
  truth puts 1.26 % of its rate there; `madspin` recovers 12 % of that, `PA`
  18 %, `madspin_v1` 5 %, and the on-shell modes 0 % by construction. This is
  invisible in the loop-induced study and is a consequence of the reshuffle
  conserving `sqrt(shat)`. §6.
* **The `gg` channel is `+12.8 %` of NLO and `+17.5 %` of LO** — 47 % of the
  whole `NLO - LO` correction, on its own. It is strongly concentrated at low `m(ZZ)`
  and central rapidity, and it does **not** double count with the NLO real
  emission: neither sample carries a `g g` initial state anywhere near the
  other's. §7, §8.

---

## 1. The four samples

| | NLO | LO | LI | truth |
|---|---|---|---|---|
| process | `p p > z z [QCD]` | `p p > z z` | `g g > z z [noborn=QCD]` | `p p > e+ e- mu+ mu- / a [QCD]` |
| run card | `RunCardNLO` | `RunCardLO` | `RunCardLO` | `RunCardNLO` |
| **sigma [pb]** | **12.66 ± 0.061** | **9.2694 ± 0.00781** | **1.6253 ± 0.00129** | **0.02859 ± 6.0e-05** |
| events | 50 000 | 50 000 | 50 000 | 50 000 |
| seed actually used | 4321 | 4321 | 4321 | 4321 |
| wall time | **54 s** | **21 s** | **105 s** | **88 s** |

All four at 13 TeV, `pdlabel = nn23lo1` / `lhaid = 230000`, fixed
`muR = muF = m_Z = 91.1880 GeV`, `pt(Z) > 1 GeV`, `bwcutoff = 15`, every
per-lepton cut off. The `output` of all four processes together took 42 s
(CutTools and IREGI were already built in the source tree).

**The NLO truth was affordable and the brief expected it not to be.** That is
worth stating plainly, because the alternative — dropping the reference and
comparing the spinmodes against each other — would have changed what every Part 1
figure can claim. The cost was measured before committing, as asked: a
**2 000-event pilot** — no custom cuts yet, `req_acc = 0.05`, 8 cores — took
**6.5 min**, almost all of it the one-off MadLoop compilation and the pole
check ([`logs/run_truth_pilot_2000events.log.txt`](logs/run_truth_pilot_2000events.log.txt),
run in a scratch directory before the study proper; the number it reports,
0.02934 pb, is the *uncut* rate and is not used anywhere else). The production
run of 50 000 events with the full cuts then took **88 s**, i.e. about **570
events per wall second** on 14 cores. `p p > e+ e- mu+ mu- [QCD]` is a one-loop six-point calculation, but it
is a *tree*-level-Born one with 48 Born and 96 virtual diagrams, and MadLoop
handles it comfortably. It is roughly **37 times cheaper** than the
loop-induced four-lepton sample of the previous study (54 min for the same
50 000 events), which is the sample the "may be far too expensive" expectation
came from.

`p p > z z [QCD]` is generated with the Born at `QED^2<=4 QCD=0`, i.e.
`q q~ -> Z Z`, 8 Born subprocesses, 16 Born / 64 virtual / 144 real diagrams.

### On the PDF

`nn23lo1` — an **LO** PDF set — is used at NLO as well. That is not the
conventional choice for an NLO calculation and it is deliberate: the figure's
two ratio panes are meant to show a change of *order*, and the loop-induced
sample this study stacks on top was generated with `nn23lo1` in the companion
study. Using one PDF everywhere removes a difference that would otherwise sit
inside the K factor. It does inflate both the NLO correction and the `gg`
contribution somewhat, since `nn23lo1` carries `alpha_s(m_Z) = 0.130`. Nothing
here is a prediction for a measurement; it is a controlled comparison.

### On the scale

Fixed `muR = muF = m_Z` on all four samples, as in the loop-induced study, so
that no dynamical scale can differ between a `2 -> 2`, a `2 -> 3` and a
four-body final state. `dynamical_scale_choice` is therefore inert everywhere
(MG5 rewrites it to `-2` in the card *after* the run, which is why the banner and
the leftover card disagree; the banner is what §2 quotes). No scale variation
was run — see §10.

### On MC@NLO at parton level

The NLO events are **MC@NLO** events stopped at the parton level
(`launch aMC@NLO -p`), with `parton_shower = PYTHIA8`. Their *total* is the NLO
cross section — the MC counterterms integrate to the shower's own `O(alpha_s)`
contribution and cancel — and that is checked directly:

```
fixed order (launch NLO)   12.67  ± 0.076 pb
MC@NLO event sample        12.7056 ± 0.061 pb      ratio 1.0028 ± 0.0077
```

Their *distributions* at parton level carry the shower subtraction and are not
the fixed-order NLO ones; that is a property of MC@NLO and not a defect here,
but it is the reason `pt(ZZ)` is not plotted (§7) and it should be remembered
when reading the K factor at large `pt(Z)`.

---

## 2. Every cut, from the cards that were actually used

Read out of the `<MGRunCard>` block of each **generated event file's banner** —
the card as used, not a template. An entry that does not appear is one this
process hides.

| entry | NLO | LO | LI | truth | no-cut value |
|---|---|---|---|---|---|
| `pt_min_pdg` | **{23: 1.0}** | **{23: 1.0}** | **{23: 1.0}** | **{23: 1.0}** (inert; read by the custom cut) | {} |
| `pt_max_pdg` / `mxx_min_pdg` | {} | {} | {} | {} | {} |
| `eta_min_pdg` / `eta_max_pdg` | *(NLO card has none)* | {} | {} | *(none)* | {} |
| `ptheavy` | *(RunCardNLO has none)* | **0.0** | **0.0** | *(none)* | 0.0 |
| `ptl` | **0.0** | *(no leptons)* | *(no leptons)* | **0.0** | 0.0 |
| `etal` | **-1.0** | *(absent)* | *(absent)* | **-1.0** | -1.0 |
| `drll` | **0.0** | *(absent)* | *(absent)* | **0.0** | 0.0 |
| `drll_sf` | **0.0** | *(RunCardLO has none)* | *(none)* | **0.0** | 0.0 |
| `mll` | **0.0** | *(absent)* | *(absent)* | **0.0** | 0.0 |
| `mll_sf` | **0.0** | *(RunCardLO has none)* | *(none)* | **0.0** | 0.0 (default **30.0**) |
| `ptj` / `etaj` | 10.0 / -1.0 (inert, see below) | *(none)* | *(none)* | 10.0 / -1.0 (inert) | — |
| `ptgmin` | 20.0 (no photon) | *(none)* | *(none)* | 20.0 (`/a`, no photon) | — |
| `dsqrt_shat` / `dsqrt_shatmax` | *(none)* | 0.0 / -1 | 0.0 / -1 | *(none)* | 0.0 / -1 |
| `bwcutoff` | 15.0 (inert) | 15.0 (inert) | 15.0 (inert) | 15.0 (read by the custom cut) | — |
| `folding` | 1, 1, 1 | *(none)* | *(none)* | 1, 1, 1 | — |
| `ickkw` | 0 | *(LO: 0)* | *(LO: 0)* | 0 | 0 |
| `dynamical_scale_choice` | -1 (inert) | -1 (inert) | -1 (inert) | -1 (inert) | — |
| `nhel` | *(NLO card has none)* | **0** | **1** (loop-induced default) | *(none)* | — |
| `req_acc` | -1.0 (auto from `nevents`) | *(none)* | *(none)* | -1.0 | — |

Five entries need a word.

**(a) `drll_sf` and `mll_sf` exist only on the NLO card.** They are declared in
`RunCardNLO.__init__` (`madgraph/various/banner.py`) and have no `RunCardLO`
counterpart; the LO card carries `drll` / `drllmax`, which cover all lepton
pairs. Conversely `ptheavy`, `use_syst`, `nhel` and `sde_strategy` exist only on
the LO card. The loop-induced study noted the `drll_sf` half of this; the other
half is what forced the change of cut parameter in §2b.

**(b) `ptheavy` is not available at NLO, so the cut is `pt_min_pdg`.** On both
card classes `pt_min_pdg` is written by `setcuts.f` into the per-particle
`etmin` array, i.e. it is an **AND** over the particles carrying that PDG, while
`ptheavy` is an OR over the heavy final states. At `2 -> 2` the two `Z` have
identical `pt` and the two are the same cut. Measured, not argued — the control
run `lo_ptheavy` is the LO sample with `ptheavy = 1.0` and `pt_min_pdg = {}`:

| | sigma [pb] |
|---|---|
| LO, `pt_min_pdg = {23: 1.0}` | 9.2694 ± 0.00781 |
| LO, `ptheavy = 1.0` | **9.2694 ± 0.00781** |

the same number to every printed digit, at the same seed. And the loop-induced
sample, regenerated here with `pt_min_pdg`, gives **1.6253 ± 0.00129 pb**
against the previous study's **1.624 ± 0.001424 pb** with `ptheavy` — 0.7 sigma.

**(c) The NLO card's default lepton cuts are inert here, and that is measured.**
This is where the loop-induced study's largest trap was: MG5 writes
`ptl = 10`, `etal = 2.5`, `drll = 0.4` into an **LO** four-lepton card by
default, and they remove 46 % of the rate. The **NLO** card's defaults are a
different set — `ptl = 0`, `etal = -1`, `drll = 0`, `drll_sf = 0`, `mll = 0`,
`mll_sf = 30` — of which only `mll_sf` is a cut at all, and the study's own
`|m_ll - m_Z| < 15 Gamma_Z` window already requires `m_ll > 54.57 GeV`, so
`mll_sf = 30` can never fire. The control `truth_default_lepton_cuts` restores
it:

| | sigma [pb] |
|---|---|
| truth as used (`mll_sf = 0`) | 0.02859 ± 0.00006 |
| truth with MG5's default `mll_sf = 30` | **0.02858 ± 0.00006** |

**0.03 % apart, i.e. identical.** The 46 % trap does *not* repeat at NLO. The
cuts were turned off anyway, on every sample, because relying on a window to
mask a cut is not the same thing as not having the cut.

**(d) The loop-induced sample integrates with `nhel = 1` and the LO one with
`nhel = 0`.** That is MG5's own per-process default (helicity sampling is on by
default for a loop-induced process) and was left alone: it is a choice of
integration method, not a cut, and both give the same cross section. It is in
the audit because it is the one setting that differs between the two `RunCardLO`
samples.

**(e) `ptj = 10` is inert.** `passcuts_jets` in `Template/NLO/SubProcesses/cuts.f`
applies the jet cut only `if (ptj.gt.0d0 .and. nQCD.gt.1)`. `p p > z z [QCD]`
has at most one final-state parton, so `nQCD <= 1` always and the block is
skipped — which is what makes the NLO calculation infrared safe with a non-zero
`ptj` sitting in the card. Measured on the events: 11 094 of the 50 000 carry a
final-state parton, **9.8 % of those partons have `pt < 10 GeV`** and the softest
is at **0.45 GeV**. A live `ptj = 10` would have removed every one of them.

### What the pt cut removes

The loop-induced study measured `pt(Z) > 1 GeV` to remove `0.307 ± 0.102 %` of
its rate, 3.0 sigma from nothing. The same control at NLO — `pt_min_pdg = {}`
and nothing else changed:

| | sigma [pb] |
|---|---|
| NLO, `pt_min_pdg = {23: 1.0}` | 12.66 ± 0.061 |
| NLO, `pt_min_pdg = {}` | 12.70 ± 0.053 |

**0.3 ± 0.6 %**, i.e. the same central value and **not resolved** at this
statistics. The NLO integration is only 0.48 % accurate, against 0.09 % for the
loop-induced `2 -> 2`, so a 0.3 % effect cannot be seen. What the control does
establish is that the cut is not doing something large and unintended: it is
bounded to below 1.5 % at two sigma.

### The one place the pt cut leaks, on both sides equally

`pt(Z) > 1 GeV` is exact on every LO and every loop-induced event (`2 -> 2`,
`pt(Z1) = pt(Z2)`; measured minimum 1.0050 GeV). On the NLO samples it is
respected by all but a handful:

| sample | events below the cut | weight fraction |
|---|---|---|
| NLO `p p > z z` | 26 of 50 000 | -0.052 % |
| truth `p p > 4l` | 25 of 50 000 | -0.054 % |

All of them are H events (the `#aMCatNLO` type-2 flag) and all carry negative
weight; the softer `Z` reaches down to 0.18 GeV. The mechanism was **not
traced** — see §10 — but the two samples leak by the same amount with the same
sign, so it cancels in the Part 1 comparison, and 0.05 % is an order of
magnitude below the NLO integration error of 0.48 %. Two truth events also sit
just outside the mass window (by 0.04 GeV out of a 73 GeV window), from the same
effect.

---

## 3. Which spinmodes ran, which were refused, and why

**All six ran. None was refused.**

| mode | ran | wall time |
|---|---|---|
| `none` | yes | 88 s |
| `madspin` | yes | 220 s |
| `onshell` | yes | 142 s |
| `PA` | yes | 157 s |
| `madspin_v1` | yes | 95 s |
| `onshell_v1` | yes | 164 s |

The loop-induced study could run only four: `MadSpinInterface.do_launch` refuses
`madspin_v1` and `onshell_v1` when the banner's `generate` line contains
`noborn`. `p p > z z [QCD]` does not, so `process_LI` is false and the two
legacy modes are available. Running them is half the reason to do this at NLO,
and they are in every Part 1 figure.

### `fixed_order` is off, and MC@NLO events do not need it

`fixed_order` defaults to `False` and was left there. It is the option for a
**fixed-order** LHE, in which one production point is written as a *group* — a
born event plus its counter-events — that has to be decayed once, together, or
the subtraction stops cancelling. MC@NLO events are not that. They are
individual events, each with its own kinematics and its own (possibly negative)
weight; nothing in the file pairs one event with another.

That distinction decides which modes are usable, because
`_check_fixed_order_spinmode` refuses `fixed_order` in
`FIXED_ORDER_RESHUFFLING_SPINMODES = ('PA', 'madspin')`:

> `fixed_order is not available in spinmode=%s: that mode reshuffles the
> production onto sampled virtualities, and how an event group's counter-events
> follow the born event through that reshuffling is not defined`

Had these events needed `fixed_order`, only `onshell` and `onshell_v1` would
have been usable — and those are precisely the two modes §4 measures to be
5 % high and §5 measures to have no lineshape at all. They do not need it, so
the comparison is the full six.

---

## 4. Negative weights, and what MadSpin did with them

The production sample carries **3 268 negative-weight events out of 50 000
(6.54 %)**, and `sum|w| / sum w = 1.15038`. Every event has the same `|w|`
(14.616222 pb, `event_norm = average`), so the sample is unweighted up to the
sign. 11 094 events (22.2 %) carry an extra final-state parton.

**Every decayed sample carries exactly 3 268 negative-weight events.** All six
modes:

```
madspin      N(w<0) =   3268   MATCHES production
PA           N(w<0) =   3268   MATCHES production
onshell      N(w<0) =   3268   MATCHES production
none         N(w<0) =   3268   MATCHES production
madspin_v1   N(w<0) =   3268   MATCHES production
onshell_v1   N(w<0) =   3268   MATCHES production
```

That is what PR #375 predicts and it is the first end-to-end NLO use of that
path in this project. Its argument is that every accept/reject tests a
matrix-element weight — positive by construction — so the factor MadSpin
multiplies by is unsigned and the event's own sign rides through untouched. The
count above is the direct consequence, on a real MC@NLO sample rather than on
weights flipped by hand in a test.

The overweight safety net also fired, and correctly: `spinmode = madspin`
reports

```
3/50000 written events (0.006%) carried a non-unit weight ... (largest factor
1.4503) ... Carrying it added +0.00435749 to the summed event weight, i.e.
+0.000303% of the sample's cross-section
```

i.e. the weight-aware accounting of #375 running on a sample where `sum w` and
`sum |w|` genuinely differ (by 15 %).

**Where the negative weights do change the analysis** is the error on the total.
For the loop-induced study every decayed file was unweighted and
`sqrt(sum w²)/N` collapsed to a meaningless `sigma/sqrt(N)`. Here it does not:
the signed weights make it a real statistical error on the mean, and it comes
out at 0.000148 pb — within 7 % of MadSpin's own 0.000139 pb. `numbers.txt`
prints both. The ratios below use MadSpin's.

---

## 5. Part 1 — the six spinmodes against the off-shell NLO truth

### The totals

| sample | sigma [pb] | error | ratio to truth |
|---|---|---|---|
| **truth** | 0.02863706 | 6.0e-05 | — |
| `madspin` | 0.02874288 | 1.39e-04 | **1.00370 ± 0.00529** (+0.37 %, 0.7σ) |
| `PA` | 0.02874279 | 1.39e-04 | **1.00369 ± 0.00529** (+0.37 %, 0.7σ) |
| `madspin_v1` | 0.02873448 | 1.39e-04 | **1.00340 ± 0.00529** (+0.34 %, 0.6σ) |
| `onshell` | 0.03005676 | 1.45e-04 | 1.04958 ± 0.00553 (+4.96 %, 9.0σ) |
| `none` | 0.03005711 | 1.45e-04 | 1.04959 ± 0.00553 (+4.96 %, 9.0σ) |
| `onshell_v1` | 0.03005676 | 1.45e-04 | 1.04958 ± 0.00553 (+4.96 %, 9.0σ) |

`1.04958 / 1.00370 = 1.04572 = 1/f²` — the Breit-Wigner truncation factor of
#379, `f = bw_retained_fraction(91.1880, 2.441404, 15) = 0.9778977`. The modes
that draw a virtuality carry it; the modes that do not, correctly, do not. The
same structure as the loop-induced study, on a different process and a different
order, and now with `madspin_v1` and `onshell_v1` landing on the correct side of
it as well.

**The three off-shell modes agree with the NLO truth to 0.7 sigma.** The
loop-induced study measured `+0.71 ± 0.16 %` (4.6σ) for the same comparison; here
it is `+0.37 ± 0.53 %`, consistent with that and with zero. The difference is
statistical reach, not physics: the NLO truth's error is 0.21 % against the
loop-induced one's 0.13 %, and MadSpin's carried error is four times larger here
because the NLO production integration is only 0.48 % accurate.

### What separates the modes

Angular moments (weighted mean ± error of the mean, 50 000 events each):

| sample | `<cos θ₁>` | `<cos²θ₁>` | `<cos θ₁ cos θ₂>` | `<cos 2Φ>` | `<m(e+μ-)>` [GeV] |
|---|---|---|---|---|---|
| truth | −0.00594 ± 0.00265 | 0.35241 ± 0.00135 | **−0.00812 ± 0.00158** | +0.00581 ± 0.00317 | 115.074 ± 0.434 |
| `madspin` | +0.00187 ± 0.00266 | 0.35315 ± 0.00136 | −0.00598 ± 0.00158 | +0.00354 ± 0.00317 | 115.251 ± 0.429 |
| `PA` | −0.00261 ± 0.00267 | 0.35521 ± 0.00136 | −0.00545 ± 0.00160 | +0.01101 ± 0.00316 | 114.945 ± 0.430 |
| `onshell` | −0.00527 ± 0.00267 | 0.35606 ± 0.00136 | −0.00597 ± 0.00159 | +0.00857 ± 0.00315 | 114.461 ± 0.436 |
| `madspin_v1` | −0.00305 ± 0.00267 | 0.35621 ± 0.00136 | −0.00801 ± 0.00159 | +0.00278 ± 0.00317 | 115.204 ± 0.435 |
| `onshell_v1` | +0.00083 ± 0.00267 | 0.35656 ± 0.00136 | −0.00425 ± 0.00160 | +0.00567 ± 0.00317 | 114.830 ± 0.431 |
| `none` | +0.00479 ± 0.00258 | **0.33392 ± 0.00134** | **+0.00349 ± 0.00150** | −0.00136 ± 0.00315 | **116.526 ± 0.421** |

and the binned shape test — `chi2/ndf` of the per-bin ratio to the truth against
the best-fit flat line, so a pure normalisation offset does not enter:

| observable | `madspin` | `PA` | `onshell` | `none` | `madspin_v1` | `onshell_v1` |
|---|---|---|---|---|---|---|
| `cos1cos2` | 0.75 | 1.35 | 0.69 | **5.11** | 0.82 | 0.87 |
| `cos θ₁` | 1.28 | 1.10 | 1.11 | **3.07** | 0.95 | 1.23 |
| `cos θ₂` | 1.06 | 1.23 | 0.71 | **2.34** | 1.09 | 1.19 |
| `m(e+ mu+)` | 0.84 | 0.92 | 0.97 | 1.80 | 1.08 | 1.10 |
| `m(e+ mu-)` | 1.29 | 1.04 | 1.34 | 1.71 | 1.41 | 0.98 |
| `Delta phi(e+, mu-)` | 0.96 | 1.03 | 0.78 | 1.25 | 1.91 | 0.86 |
| `Phi` (decay planes) | 0.94 | 1.02 | 1.02 | 1.16 | 1.17 | 0.64 |
| `pt(e+e-)` | 1.25 | 1.46 | 0.97 | 1.03 | 1.80 | 0.97 |
| `m(e+e-)` | **1.98** | **4.13** | delta fn | delta fn | **4.39** | delta fn |
| `m(mu+mu-)` | **1.90** | **4.17** | delta fn | delta fn | **2.68** | delta fn |
| `m_4l` | **31.8** | **14.0** | 1.60 | 1.60 | **88.5** | 1.60 |
| `m_4l >= 2 m_Z` | 1.51 | 1.71 | 1.57 | 1.57 | 1.61 | 1.57 |

**Separates: `cos θ₁`, and `cos θ₁ cos θ₂` more sharply.** `spinmode = none`
gives `<cos²θ₁> = 0.33392 ± 0.00134`, i.e. 1/3 — a flat decay angle, an
unpolarised `Z`. Every other mode and the truth sit at 0.352–0.357. The gap is
`0.0185 ± 0.0019`, **9.7 sigma**, and on the binned shape test it is 3.07
against 0.95–1.28 for the correlated modes.

**`<cos θ₁ cos θ₂>` separates at NLO, and it did not in the loop-induced study.**
This is the observable that isolates the correlation *between* the two decays:
each factor is separately odd under `l+ <-> l-` for an unpolarised parent, so its
mean is exactly zero for any scheme that decays the two `Z` independently. The
truth measures `−0.00812 ± 0.00158`, `none` measures `+0.00349 ± 0.00150`, and
the difference is `0.0116 ± 0.0022`, **5.3 sigma**. The loop-induced study
measured the same separation at 1.1 sigma and reported explicitly that "the
correlation between the two decays is present in the truth at the sub-percent
level and is not resolved here". At NLO on `p p > Z Z` it *is* resolved: the
truth's value is 5.1 sigma from zero on its own, and every spin-correlated mode
tracks it (`madspin` −0.00598, `madspin_v1` −0.00801, `PA` −0.00545). The reason
is the process, not the order: `q q~ -> Z Z` is a `t`-channel exchange between
two initial-state fermions with definite chirality, which produces a much more
strongly correlated `ZZ` helicity state than the `gg` box does.

**Does not separate: `Phi`.** As in the loop-induced study, all seven curves are
flat and mutually consistent — `<cos 2Φ>` is `+0.0058 ± 0.0032` for the truth and
within 1.5 sigma of that for every mode, `chi2/ndf` between 0.6 and 1.2. Same
conclusion for the same reason: `Z Z` production is a continuum, and integrated
over `m_4l` and the production angle the interference modulation averages away.

**Does not separate: `Delta phi(e+, mu-)`, `pt(e+e-)`, the cross-pair masses.**
All of them sit between 0.8 and 1.9 for every mode including `none`, i.e. no
mode is distinguishable on them at 50 000 events. `m(e+ mu-)` separated `none` at
4.1 in the loop-induced study and does not here (1.71).

**The `_v1` modes are not worse.** `madspin_v1` tracks `madspin` on every
observable that has sensitivity — normalisation `+0.34 %` against `+0.37 %`,
`<cos²θ₁>` 0.35621 against 0.35315, `cos1cos2` `−0.00801` against `−0.00598`
where the truth is `−0.00812`. `onshell_v1` reports the *same* branching ratio
as `onshell` to every digit (0.002365632493) and gives an indistinguishable
sample. On the lineshape `madspin_v1` is worse than `madspin` (4.39 against
1.98) and comparable to `PA`. This is the first side-by-side measurement of the
legacy modes against a modern one on the same events.

### `m(l+l-)`: the lineshape, and `madspin` still wins

`onshell`, `none` and `onshell_v1` draw no virtuality, so their pair mass is a
delta function at `m_Z` (measured range 91.18799…91.18800 GeV) and the shape test
is undefined; `numbers.txt` says so rather than printing a large meaningless
number, and the ratio panes mark those bins with open circles rather than
plotting a measured zero.

Among the three that do draw one, `madspin` is best on both pairs — 1.98 / 1.90
against `PA`'s 4.13 / 4.17 and `madspin_v1`'s 4.39 / 2.68. Same ordering as the
loop-induced study found, now with the legacy mode placed: it behaves like `PA`
on the lineshape, not like `madspin`.

---

## 6. The one clear MadSpin failure at NLO: `m_4l` below `2 m_Z`

This is the finding that does not exist in the loop-induced study, and it exists
here only because an NLO sample has recoil.

MadSpin's reshuffle redistributes virtuality between the two resonances at
**fixed `sqrt(shat)`**. For a `2 -> 2` production event `sqrt(shat)` *is* `m(ZZ)`,
so `m_4l` cannot move at all — which is exactly what the loop-induced study
measured (`max |Delta m_4l| = 1.1e-05 GeV`, the LHE write precision). At NLO,
22.2 % of the events carry an extra parton, and for those the reshuffle can push
`m(ZZ)` below the on-shell threshold `2 m_Z = 182.376 GeV` while `sqrt(shat)`
stays put. The other 77.8 % cannot.

The truth has no such restriction:

| sample | events with `m_4l < 2 m_Z` | weight fraction | fraction of the truth's |
|---|---|---|---|
| **truth** | 644 | **1.257e-02** | — |
| `PA` | 148 | 2.209e-03 | **0.18** |
| `madspin` | 88 | 1.565e-03 | **0.12** |
| `madspin_v1` | 30 | 5.982e-04 | **0.05** |
| `onshell` / `none` / `onshell_v1` | 0 | 0 | **0** (structural) |

**1.26 % of the truth's rate inside a ±15-width window sits below the on-shell
`ZZ` threshold, and no MadSpin mode reproduces more than a fifth of it.** In the
total that is a ~1 % effect, which is why §5's normalisation agreement survives
it — but on the `m_4l` spectrum it is the whole story below 182 GeV, and it is
what the unrestricted `m_4l` chi2 of 14–89 is measuring. Restricted to bins
above the threshold, every mode is at 1.5–1.7 and indistinguishable.

That the on-shell modes give exactly zero there is a **structural** zero and is
marked as such in the figures — leaving those bins in the chi2 turns 1.60 into
more than 30 for a reason that has nothing to do with the mode's quality.

---

## 7. Part 2 — the stacked figure

Absolute normalisation in pb, same cuts, same scale, same PDF, 50 000 events
each.

```
LO    p p > z z              9.2694  ± 0.00781  pb
NLO   p p > z z [QCD]       12.66    ± 0.061    pb
LI    g g > z z (loop ind.)  1.6253  ± 0.00129  pb
```

```
K = NLO / LO         = 1.3707 ± 0.0067
(NLO + LI) / LO      = 1.5460 ± 0.0067
LI / NLO             = 0.1279 ± 0.0006      (+12.8 %)
LI / LO              = 0.1753 ± 0.0002      (+17.5 %)
```

**The `gg` channel is 12.8 % of the NLO cross section and 17.5 % of the LO one.**
The whole NLO correction is `NLO - LO = 3.436 pb`; the `gg` contribution is
`1.625 pb`, i.e. **47 % of it**. That is what makes it
worth a figure: it is formally an NNLO contribution, and it is numerically the
same order as terms nobody would drop.

The quoted `sigma` is the **sum of the event weights**, because that is what
normalises the histograms; the quoted error is the **integration** error, because
that is the uncertainty on the prediction. For the two unweighted samples the
two sigmas are identical; for the MC@NLO one the unweighting makes them differ
by `+0.36 %`, i.e. 0.75 of the integration error. `numbers_stack.txt` prints
both.

### Where the `gg` contribution sits

| observable | `LI / NLO` runs from | to | separates? |
|---|---|---|---|
| `m(ZZ)` | **17.1 %** at 180–190 GeV | **3.9 %** at 1100–1400 GeV | **yes, strongly** |
| `abs(y(ZZ))` | **18.4 %** at &#124;y&#124; < 0.17 | **1.8 %** at &#124;y&#124; ≈ 3.6 | **yes, strongly** |
| `pt(Z_lead)` | 57.8 % at `pt < 5 GeV` | 4.0 % at 400–600 GeV | partly (see below) |
| `abs(cos θ*)` | 13.3 % | 14.3 % | **no — flat** |

`m(ZZ)` and the `ZZ` rapidity are the two observables that make the point. The
`gg` luminosity falls faster with `x` than the `q q~` one, so the box
contribution is concentrated at low `m(ZZ)` and central rapidity and dies away in
both tails — a factor of four across `m(ZZ)` and a factor of ten across
`|y(ZZ)|`. The production angle `|cos θ*|` is flat to within 8 % of itself
across its whole range and separates nothing; it is plotted because a
non-separating observable is a result too.

`pt(Z_lead)` is the observable where the NLO correction itself is largest —
`NLO / LO` climbs from 1.00 in the first bin to **5.28** at 400–600 GeV, because at
LO the two `Z` are back to back and a hard `Z` costs a hard partonic collision,
while at NLO it can simply recoil against a jet. The first bin's 58 % `LI/NLO`
is that same effect seen from the other side and is not a `gg` enhancement: it is
the NLO curve being *suppressed* at `pt(Z) -> 0` relative to a `2 -> 2` sample.
Read it as "LO and LI are `2 -> 2` here and NLO is not" rather than as physics
about the box.

### Two observables deliberately not stacked

`pt(ZZ)` and `Delta phi(Z, Z)` are exactly 0 and exactly `pi` for **every** LO and
**every** loop-induced event — both are `2 -> 2` with no initial-state
transverse momentum, measured `pt(ZZ)_max = 0.0` on both. Only the NLO sample
has recoil (up to 920 GeV). A stack of one distribution against two delta
functions is not a figure. `pt(ZZ)` is still harvested and is in
`histograms.npz`.

---

## 8. The double-counting check

**They do not double count, and this is measured rather than asserted.**

The argument: `p p > z z [QCD]` has a Born of `q q~ -> Z Z` at `O(alpha^2)` and
real emission `q q~ -> Z Z g`, `q g -> Z Z q`, `g q~ -> Z Z q~`. `g g -> Z Z`
proceeds through a closed quark loop with no Born to attach to; its amplitude is
`O(alpha_s alpha)`, so its rate is `O(alpha_s^2)` relative to the LO cross
section and it first enters at NNLO. The two therefore live at different orders
and cannot overlap.

The measurement, on the initial state of every written event of each sample,
summed by weight:

| sample | initial states carrying the weight | `g g` present |
|---|---|---|
| **LO** | `q q~` only: `d d~` 52.1 %, `u u~` 37.3 %, `s s~` 7.8 %, `c c~` 2.7 % | **no** |
| **NLO** | `q q~` 87.2 %, `q g` + `g q~` 12.8 % (`d g` 4.5 %, `u g` 4.2 %, `d~ g` 1.4 %, …) | **no** |
| **LI** | `g g` 100.0 % | **yes** |

`(21, 21)` does not appear once in 50 000 NLO events. The same conclusion is
visible one level up, in the generated process list: `initial_states_map.dat` of
all four `P0_*` FKS directories lists twelve initial states, every one of them
`q q~`, `q g` or `g q~`, and no `g g`; and MG5 fixes the Born orders to
`QED^2<=4 QCD=0`, which excludes a gluon-initiated Born by construction.

The `qg` channel that does appear in the NLO sample carries **12.8 %** of its
weight — numerically almost identical to the `gg` contribution's 12.8 % of NLO,
which is a coincidence but a confusing one, so it is worth being explicit: those
are two different things. `qg -> ZZ q` is inside the NLO number; `gg -> ZZ` is
the band drawn on top of it.

What this check would *not* catch is a double counting that is not visible in
the initial state — for instance if the loop-induced sample had been generated
with `[QCD]` rather than `[noborn=QCD]` and had silently included a `q q~` Born.
The LI sample's initial state is 100 % `g g`, which rules that out here.

---

## 9. Wall times

| step | wall | notes |
|---|---|---|
| `output` of all four processes | 42 s | serial on purpose; CutTools/IREGI already built |
| NLO sample, 50 000 events | **54 s** | 14 cores |
| LO sample, 50 000 events | **21 s** | 14 cores |
| loop-induced sample, 50 000 events | **105 s** | 14 cores |
| **NLO four-lepton truth, 50 000 events** | **88 s** | 14 cores — the one this study expected to be unaffordable |
| truth pilot, 2 000 events | 6.5 min | almost all of it the one-off MadLoop build |
| MadSpin `none` | 88 s | 3 cores |
| MadSpin `madspin_v1` | 95 s | 3 cores |
| MadSpin `onshell` | 142 s | 3 cores |
| MadSpin `PA` | 157 s | 3 cores |
| MadSpin `onshell_v1` | 164 s | 3 cores |
| MadSpin `madspin` | **220 s** | 3 cores; the most expensive mode |
| control: NLO, no pt cut | 26 s | |
| control: LO with `ptheavy` | 19 s | |
| control: truth with default lepton cuts | 65 s | |
| cross-check: fixed-order NLO | 13 s | |

The whole study is about **25 minutes** of wall time on one 18-core machine, the
six MadSpin modes run in parallel. Nothing had its statistics reduced.

---

## 10. Not covered, and things that were left uncertain

* **The pt-cut leak was measured, not explained.** 26 NLO and 25 truth events sit
  below `pt(Z) = 1 GeV`; all are negative-weight H events. `setcuts.f` writes
  `etmin` for both `Z` of the `(n+1)`-body configuration and `passcuts_pdgs`
  tests both, so on the face of it the written kinematics should satisfy the cut.
  Why a small set of H events does not was not traced through the FKS bookkeeping.
  It is 0.05 % of the rate, it has the same size and sign on both sides of the
  Part 1 comparison, and it is stated rather than swept up.
* **The MC@NLO sample is parton level.** Its *total* is checked against the
  fixed-order NLO result (§1) and agrees to 0.3 %, but its distributions carry
  the shower subtraction terms of `parton_shower = PYTHIA8`. Nothing was
  showered, and no second shower choice was tried, so the shower dependence of
  the parton-level shapes is unmeasured.
* **One phase-space point.** 13 TeV, `nn23lo1`, fixed `mu = m_Z`, `BW_cut = 15`,
  `pt(Z) > 1 GeV`, one seed. No scale variation, no PDF uncertainty, no
  `BW_cut` scan, no second seed. The stacked figure's `K` factor in particular
  would move under a scale variation, and the `gg` contribution — which is
  `O(alpha_s^2)` and has no compensating scale dependence at this order — would
  move more.
* **An LO PDF is used at NLO** (§1). Choosing `nn23nlo` for the NLO and `gg`
  samples and keeping `nn23lo1` for LO would be the conventional set-up and was
  not run; it would change `K` and the `gg` fraction by several percent.
* **No `4e` / `4mu` channel** and no identical-lepton interference: the
  positional decay rule gives exactly one `e+e-` and one `mu+mu-` pair, and the
  truth is generated as `e+ e- mu+ mu-` only.
* **The photon is excluded** (`/a`) on the truth side, so `Z/gamma*`
  interference is not tested. Inside a 15-width window it is small but not zero.
* **`gg -> ZZ` interference with the `q q~ -> ZZ` amplitude is zero and was not
  checked**, because it vanishes by colour: the box amplitude carries
  `delta^{ab}` and the `q q~` one carries the identity in colour space, so the
  interference is `O(alpha_s)` relative and appears at a different order again.
  The stack adds the two rates, which is what that implies, but nothing here
  measures it.
* **`gg -> ZZ` at NLO, and its interference with `gg -> H -> ZZ`**, are outside
  this study entirely. The loop-induced sample is LO in its own right.
* **`m_4l` below `2 m_Z` (§6) was measured, not fixed.** Whether MadSpin could
  do better — a reshuffle that also moves the recoil, or an acceptance that lets
  a `2 -> 2` event's `sqrt(shat)` change — was not investigated.
* **`ms_dir` reuse across spinmodes was not used**, for the same reason the
  loop-induced study gives: `run_from_pickle` restores the pickled option
  object, so a reused directory carries the first run's `spinmode` into every
  lookup through `decay_all_events.options`.
