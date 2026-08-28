# `g g > z z` (loop induced) + MadSpin against the full off-shell `g g > 4l`

> **Support correction, 2026-08-28 — read this together with the frame note
> below.** The truth carries **2.09 % of its cross section below
> `m_4l = 2 m_Z`**, and **every** MadSpin spinmode has *exactly zero* support
> there: `g g > z z` puts both `z` on shell and the RAMBO reshuffle holds
> `sqrt(shat)` fixed. That region is entirely at low `pt` and it is the whole of
> `PA`'s `pt(e+ e-)` `chi2/ndf = 3.32`. On the support the two samples share,
> `PA` is **1.59** and is the *best* of the three spin-correlated modes
> (`madspin` 2.15, `onshell` 2.41); `onshell`'s 1.08 on the full support is the
> `1/f^2` normalisation error cancelling the same hole. Two further statements
> of section 5 below are superseded: `madspin` does **not** miss the `m(l+ l-)`
> low tail (1.020, not 0.773, on the reachable support, and shape `chi2/ndf`
> **0.89**), and "`madspin` is better than `PA` on every observable with any
> sensitivity" is not true on the matched support. The evidence, and what the
> validation should say instead, are in
> [PA_LOWPT_DIAGNOSIS.md](PA_LOWPT_DIAGNOSIS.md).

> **Frame correction, 2026-08-27 — applied, and the study re-run. Read this
> before any number below.** The angular observables of this study
> (`cos_theta1`, `cos_theta2`, `cos1cos2`, and the `f_0` / `f_00` / `f_TT` /
> `C_kk` coefficients built from them) were first harvested with a boost
> composition that is not the helicity frame — the axis in the four-lepton
> frame, the `l+` boosted into its pair's rest frame straight from the lab. That
> composition carries a Wigner rotation and damps every rank-2 moment towards
> `1/3`. `observables.py` is fixed and guarded by a self-test.
>
> **The study has since been regenerated from scratch with the fix in place, at
> 200 000 events per sample instead of 50 000.** `data/histograms.npz`,
> `data/meta.json`, `data/numbers.txt`, `plots/` and `plots_userstyle/` are all
> the post-fix 200 000-event set, and the corrected coefficient tables are in
> [SPIN_COEFFICIENTS.md](SPIN_COEFFICIENTS.md).
>
> **The prose of THIS file has not been rewritten and still describes the
> 50 000-event first pass.** Its structural conclusions stand — the four modes
> work, the `+0.71 %` / `+5.32 %` normalisation split, `spinmode = none` being
> the only mode that separates — but every *number* in it is superseded by
> `data/numbers.txt`. Where the two disagree, `data/numbers.txt` is right. Two
> conclusions moved enough to name here: `none`'s separation on `f_0` grew from
> `22 sigma` to `77 sigma`, and `PA` developed a real `pt(e+ e-)` shape
> deviation (`chi2/ndf = 3.32`) that 50 000 events could not resolve — **which
> the support note above then explained: it is the missing `m_4l < 2 m_Z`
> region, not a property of `PA`.**

Setup, cards and re-run instructions are in [README.md](README.md). Raw numbers
are in [`data/numbers.txt`](data/numbers.txt); everything below is read off it
and off [`data/meta.json`](data/meta.json).

Base: **`178e9542d`** on `origin/madspin_density` — the merge of PR **#379**, so
this branch **already carries** the Breit-Wigner truncation of MadSpin's
reported cross section (`MadSpin/decay.py` defines `bw_retained_fraction`). One
normalisation is quoted throughout, the reported one, and it carries the
truncation. Nothing is corrected by hand on top of it.

---

## 0. Summary

* The loop-induced density-matrix path **works** on `g g > z z`, in all four
  spinmodes MadSpin allows for a loop-induced process. `madspin_v1` and
  `onshell_v1` are refused by MadSpin itself, so this is a four-mode study.
* **Normalisation.** The modes that draw a virtuality (`madspin`, `PA`) are
  **+0.71 %** above the off-shell truth — inside the residual PR #379 documents
  for its own approximation, and now measured directly (§3). The modes that do
  not (`onshell`, `none`) are **+5.32 %**, which is exactly `1/f²` more, as they
  must be.
* **What separates the modes.** `cos θ₁` separates `spinmode = none` from
  everything else by **22 sigma** on a single moment (`<cos²θ₁>`), and by
  `chi2/ndf = 13.9` on the binned shape. `m(e+ mu-)` separates it at 4.5 sigma.
  `madspin`, `PA` and `onshell` all track the truth.
* **What does not.** The **decay-plane angle `Φ` separates nothing** — all five
  curves are flat and lie on top of each other. Neither does `m_4l`, and that
  one is exact: MadSpin's reshuffle conserves `sqrt(shat)` event by event, so
  every mode's `m_4l` is the production sample's, bit for bit.
* **`madspin` beats `PA`** on every observable with sensitivity. The `Z`
  lineshape is where they part company: `PA` has a monotonic tilt from x2.8 at
  62 GeV to x0.42 at 126 GeV, while `madspin` holds 0.95-1.03 across the region
  that carries the rate.
  *(2026-08-28: the lineshape half of this stands and is sharper than stated —
  `madspin` 0.89, `PA` 17.13 on the matched support — but "on every observable"
  does not: `PA` beats `madspin` on `pt(e+ e-)`, 1.59 against 2.15. See
  [PA_LOWPT_DIAGNOSIS.md](PA_LOWPT_DIAGNOSIS.md).)*

---

## 1. Stage 1 — the two production samples

| | sample A | sample B (reference) |
|---|---|---|
| process | `g g > z z [noborn=QCD]` | `g g > e+ e- mu+ mu- / a [noborn=QCD]` |
| loop diagrams | 28 | 60 |
| **cross section** | **1.624 ± 0.001424 pb** | **0.003648 ± 4.681e-06 pb** |
| events | 50 000 | 50 000 |
| seed actually used | 4321 | 30 |
| wall time | **1 min 34 s** (16 cores, 17m39s CPU) | **54 min** (18 cores, 13h32m39s CPU) |

Both loop-induced processes use MG5's **LO** run card (not the NLO one), so
`ptheavy` is available — see §7 for the one place that was not obvious.

Run-card settings that differ from the process's own default, applied to
**both**:

```
fixed_ren_scale = True      fixed_fac_scale = True
scale = dsqrt_q2fact1 = dsqrt_q2fact2 = 91.188      (= m_Z)
ptheavy  = 1.0        bwcutoff = 15.0
nevents  = 50000      use_syst = False
```

and, on **sample B only**, turning the standard MadEvent lepton cuts off:

```
ptl = 0.0        etal = -1.0        drll = 0.0        mmll = 0.0
custom_fcts = zz_equivalent_cuts.f
```

Those four defaults are `ptl = 10`, `etal = 2.5`, `drll = 0.4`, `mmll = 0` —
and sample A has no lepton for them to act on. Leaving them in would have
compared a cut four-lepton sample against an uncut one and would have shown up
as a large MadSpin discrepancy.

Everything else is the MG5 default and identical on both sides: 13 TeV
(`ebeam = 6500`), `pdlabel = nn23lo1` / `lhaid = 230000`, `nhel = 1`,
`sde_strategy = 1`, `event_norm = average`.

**On the seeds.** `iseed = 4321` was requested for both. Sample A used it.
Sample B did **not**: a 2 000-event pilot in the same directory consumed 4321,
and MadEvent rewrites `iseed` to 0 at the end of a run so the next one is
auto-seeded — sample B's banner records **30**. That is recorded rather than
re-run: the two samples are different processes with different phase-space
channels, so a shared seed would not correlate them in any case, and the seed
matters here only for reproducibility. `meta.json` carries the *actual* seed of
every sample, read back out of its banner, not the requested one.

`m_Z` and `Gamma_Z` as taken from the param card:

```
Block mass    23   9.118800e+01   # MZ
DECAY         23   2.441404e+00   # WZ
```

so the 15-width window is `|m_ll - 91.1880| < 36.62106`, i.e.
**54.56694 < m_ll < 127.80906 GeV**.

### How sample B's cuts were applied

Through the run card's supported `custom_fcts` hook, which replaces
`dummy_cuts` in `SubProcesses/dummy_fct.f`. The file is
[`zz_equivalent_cuts.f`](zz_equivalent_cuts.f). It hard-codes **no number**: it
reads `ptheavy` and `bwcutoff` from this run's own `run_card.dat` (`cuts.inc`,
`run.inc`) and `M_Z`, `Gamma_Z` from its own `param_card.dat` (`coupl.inc`), and
re-derives the lepton positions from `leshouche.inc` at run time rather than
trusting the process ordering.

The run card's own `mmll` / `mmllmax` were deliberately **not** used for the
mass window. The card itself warns that "for four lepton final state mmll cut
require to have different lepton masses for each flavor" — with massless `e`
and `mu` it cannot tell a same-flavour pair from `(e+ mu-)`, and would cut the
wrong combinations.

### Every cut parameter, from the cards that were actually used

Sample A has no lepton at generation time, so MadSpin's decayed events are
subject to **no** per-lepton cut whatsoever. Sample B must therefore have every
per-lepton cut off, or the comparison is a cut sample against an uncut one and
every distribution disagrees for a trivial reason. This is the audit, read out
of the `<MGRunCard>` block of each **generated event file's banner** — the card
as used, not a template. Entries not listed do not exist in this version's LO
run card for these processes.

| entry | sample A | sample B | no-cut value |
|---|---|---|---|
| `ptl` | *(absent: no leptons)* | **0.0** | 0.0 |
| `ptlmax` | *(absent)* | **-1.0** | -1.0 |
| `etal` | *(absent)* | **-1.0** | -1.0 |
| `etalmin` | *(absent)* | **0.0** | 0.0 |
| `drll` | *(absent)* | **0.0** | 0.0 |
| `drllmax` | *(absent)* | **-1.0** | -1.0 |
| `mmll` | *(absent)* | **0.0** | 0.0 |
| `mmllmax` | *(absent)* | **-1.0** | -1.0 |
| `mmnl` / `mmnlmax` | *(absent)* | **0.0 / -1.0** | 0.0 / -1.0 |
| `ptllmin` / `ptllmax` | *(absent)* | **0.0 / -1.0** | 0.0 / -1.0 |
| `xptl` | *(absent)* | **0.0** | 0.0 |
| `ptl1min` … `ptl4min` | *(absent)* | **0.0** (all four) | 0.0 |
| `ptl1max` … `ptl4max` | *(absent)* | **-1.0** (all four) | -1.0 |
| `pt_min_pdg` / `pt_max_pdg` | **{}** | **{}** | {} |
| `eta_min_pdg` / `eta_max_pdg` | **{}** | **{}** | {} |
| `mxx_min_pdg` | **{}** | **{}** | {} |
| `dsqrt_shat` / `dsqrt_shatmax` | **0.0 / -1** | **0.0 / -1** | 0.0 / -1 |
| `ptheavy` | **1.0** (acts on the two `z`) | **1.0** (natively inert; read by the custom cut) | 0.0 |
| `bwcutoff` | 15.0 (inert) | 15.0 (read by the custom cut) | — |

Three entries need a word:

* **`ptl = 10`, `etal = 2.5`, `drll = 0.4` are what MG5 put in sample B's card
  by default.** They were turned off before *any* sample B run, including the
  pilot — the banner above is the proof, and no sample had to be regenerated.
  §2(e) measures what they would have cost.
* **`misset` / `missetmax` are hidden** from sample B's card (the process has no
  neutrino) and default to `0.0 / -1.0` in `RunCardLO`, i.e. no cut.
* **`drll_sf` does not exist here.** It is a parameter of `RunCardNLO`
  (`banner.py:5853`), and loop-induced processes use the **LO** run card. The
  LO card carries only `drll` / `drllmax`, which cover all lepton pairs.

The only cuts acting on sample B are therefore the two the study asked for, both
from `zz_equivalent_cuts.f`.

None of the four MadSpin cards sets the `run_card` option, which would have
applied cuts to the decayed events after the fact and reintroduced exactly this
asymmetry from the other side. Each card is five `set` lines
(`seed`, `BW_cut`, `spinmode`, `nb_core`, `ms_dir`), two `decay` lines and
`launch`; they are archived verbatim in [`logs/`](logs/).

---

## 2. Do the cuts actually fire?

Four independent pieces of evidence, because a silently-ignored cut looks
exactly like a cut that fires and removes nothing — and would show up in this
study as *better* agreement, not worse.

**(a) The cut announces itself, with the values it read.** In every one of the
72 `SubProcesses/P0_gg_llll/G*/log.txt`:

```
 zz_equivalent_cuts ACTIVE: e+ e- at   3  4   mu+ mu- at   5  6
 zz_equivalent_cuts: pt(ll) >  1.0000000000000000  GeV ;
                     |m(ll) -  91.188000000000002 | <  36.621060000000000
```

`36.62106 = 15 x 2.441404` — i.e. it really did read `bwcutoff` and `WZ` out of
the cards, and it really did locate `e+ e- mu+ mu-` at legs 3, 4, 5, 6.

**(b) The events respect the boundaries.** Over the 50 000 written events,
`m(e+e-)` and `m(mu+mu-)` both lie strictly inside 54.56694 … 127.80906 GeV,
and `pt` of **both** reconstructed pairs is above 1 GeV. The two pair `pt` also
agree with each other to `9.1e-09 GeV`, which is the measurement behind the
claim that `ptheavy`'s "at least one heavy particle" and "both pairs" are the
same cut at `2 -> 2`.

**(c) The pt cut, measured against a control.** Sample A run again with
`ptheavy = 0` and nothing else changed:

| | sigma [pb] |
|---|---|
| sample A, `ptheavy = 1` | 1.624 ± 0.001424 |
| sample A, `ptheavy = 0` | 1.629 ± 0.0008527 |

The cut removes **0.307 ± 0.102 %** — 3.0 sigma from doing nothing. Small, as a
1 GeV cut on a 91 GeV particle should be, but resolved.

**(d) The mass window, from the integration cost — a control that was
CANCELLED.** Sample B was started again with `custom_fcts` pointed at
[`zz_ptonly_cuts.f`](zz_ptonly_cuts.f) — the same pt cut, no mass window — with
the intention of measuring the retained fraction. **It was stopped before it
finished** and is not a measurement. What it did show, before it was killed, is
worth stating on its own:

| | events asked for | state when stopped |
|---|---|---|
| sample B, window on | 50 000 | **finished in 52 min 56 s**, 473 refine jobs, 13h32m CPU |
| sample B, window off | 20 000 | **still refining at 1 h 39 min**, 213 jobs, never finished |

Fewer than half the events, more than twice the wall time, and not done — at
least a factor of four per event, on the same 18-core machine (though not with
identical concurrent load, so treat the factor as indicative rather than
measured).

**What that supports:** the `|m_ll - m_Z| < 15 Gamma_Z` window is doing
substantial work. With it removed the integrator spends its time in the far
off-shell region — which is precisely the region MadSpin's `BW_cut = 15`
excludes by construction — and the phase space it has to reach grows enough to
change the cost of the job by a large factor. A cut that were being silently
ignored could not do that.

**What it does NOT support:** any number for how much cross section lies outside
the window. That integral was never completed and none is quoted. The `f²` of §3
is MadSpin's own analytic propagator factor, not a measurement from this run,
and the "+0.71 %" residual is measured against sample B directly, which does not
need this control at all.

Nothing from the cancelled run reaches the results: no event file was ever
written (its `Events/` directory holds only `.keep`), `log_cross` refuses a log
that has no "Results Summary" block, and `meta.json` carries only
`controls.A_no_ptcut`. The cancelled log is archived as
[`logs/control_B_no_masswindow_CANCELLED.log.txt`](logs/control_B_no_masswindow_CANCELLED.log.txt)
under a name that cannot be mistaken for a result.

**(e) What the default lepton cuts would have cost.** The audit above says they
were off. This measures what would have happened if they had not been: sample B
run again with `ptl = 10`, `etal = 2.5`, `drll = 0.4` restored and everything
else — including both custom cuts — identical.

| | sigma [pb] |
|---|---|
| sample B as used (`ptl = 0`, `etal = -1`, `drll = 0`) | 0.003648 ± 0.0000047 |
| sample B with MG5's default lepton cuts | 0.001967 ± 0.0000215 (2 000 events) |

The defaults remove **46.1 ± 0.6 %** of the rate. Sample A has no lepton at
generation time and MadSpin's decays are unrestricted, so had this been missed
the study would have reported MadSpin as roughly a factor of two *high* against
truth — and the +0.7 % result of §3 would have been invisible underneath it.
This is the single largest trap in the setup, and it is a silent one: the
defaults are what MG5 writes into the card for you.

---

## 3. Normalisation, and the Breit-Wigner truncation

`MadSpin.decay.bw_retained_fraction(91.1880, 2.441404, 15)` returns

```
f     = 0.977897722284
f^2   = 0.956283955248        (two resonances)
1/f^2 = 1.045714501966
```

**MadSpin applies exactly this**, and applies it to the right modes. The
reported branching ratios are the direct check — no fitting, no tolerance:

```
BR(PA) / BR(onshell) = 0.956283955247   vs f^2 = 0.956283955248
                                        difference  -4.8e-13
```

`onshell` and `none` sample no virtuality and correctly get **no** factor; they
report the same BR as each other to `1.2e-05`, which is the Monte-Carlo noise on
MadSpin's own integration of the `z > l+ l-` partial widths. This is the merged
#379 behaviour verified on a process it had not been run on.

### The totals

| sample | sigma [pb] | error | ratio to truth |
|---|---|---|---|
| **truth** (sample B) | 0.003648442 | 4.68e-06 | — |
| `madspin` | 0.003674518 | 3.22e-06 | **1.00715 ± 0.00157**  (+0.71 %, 4.6σ) |
| `PA` | 0.003674518 | 3.22e-06 | **1.00715 ± 0.00157**  (+0.71 %, 4.6σ) |
| `onshell` | 0.003842497 | 3.37e-06 | 1.05319 ± 0.00164  (+5.32 %, 32.5σ) |
| `none` | 0.003842543 | 3.37e-06 | 1.05320 ± 0.00164  (+5.32 %, 32.5σ) |

`madspin` and `PA` report the **same** total to every digit, and that is right
rather than suspicious: MadSpin's reported sigma is
`sigma_prod x BR x truncation`, and the two modes share all three. They differ
only in the kinematics they hand back, which is what §5 measures.

`1.05319 / 1.00715 = 1.04571 = 1/f²` to five digits. The untruncated modes are
high by precisely the factor the truncated ones carry, which is what "no
virtuality drawn, so no truncation" means quantitatively.

The error quoted is the **integration** error, not the spread of the unweighted
events. These files carry one weight value, so `sqrt(sum w²)/N` collapses to
`sigma/sqrt(N)` = 0.45 % — a number that is not an uncertainty on sigma at all
(sigma is fixed by the integration, not by counting the events written out) and
that would have hidden the 4.6σ result under a factor-of-three inflation. The
per-bin histogram errors, where `sqrt(sum w²)` *is* right, are unaffected.

### The residual

The +0.71 % is not noise: it is 4.6 sigma, and #379's own commit message
predicts it. The factor `f²` is the retained fraction of the **propagator**,
while the rate integrand is the propagator times the matrix element times phase
space, and those also vary across a ±15-width window. #379 documents the
leftover as "+0.4 % to +1.0 % for a `t t~` pair at `BW_cut = 15`". This study
measures **+0.71 ± 0.16 %** for a `Z` pair, i.e. the same sign and the same size
on a different process, a different resonance and a loop-induced production
mechanism.

That is a direct measurement against sample B and needs no control run. A
separate control — sample B with the window removed, whose weight fraction
inside the window would isolate the retained fraction of the *full* integrand
rather than of the propagator alone — was started and **cancelled** (§2d); no
number from it is quoted.

---

## 4. The decay assignment is positional, and it was checked

Two `z` in the event and two `decay z >` lines is MadSpin's **positional** rule:
the i-th particle of that pdg takes the i-th line
([`doc/madspin_decay_groups.md`](../../../doc/madspin_decay_groups.md) §2). Not
assumed — counted, on all 50 000 events of every mode:

```
none      N=50000   distinct final states: 1    (-13, -11, 11, 13)  50000
onshell   N=50000   distinct final states: 1    (-13, -11, 11, 13)  50000
PA        N=50000   distinct final states: 1    (-13, -11, 11, 13)  50000
madspin   N=50000   distinct final states: 1    (-13, -11, 11, 13)  50000
```

Exactly one `e+e-` pair and one `mu+mu-` pair in every event, no `4e`, no `4mu`.
The harvester enforces the same thing and raises rather than averaging if it
ever sees two same-pdg leptons in one event, so a future regression cannot pass
silently.

The branching ratio confirms the other half of the rule. `none`'s untruncated
`BR = 0.0023656607`, and `2 x BR(Z->ee) x BR(Z->mumu)` is what the positional
formula gives — the factor 2 compensating for generating only one of the `2!`
assignments of two distinct channels to two identical parents.

---

## 5. What separates the modes, and what does not

Angular moments (weighted mean ± error of the mean, 50 000 events each):

| sample | `<cos θ₁>` | `<cos²θ₁>` | `<cos θ₁ cos θ₂>` | `<cos 2Φ>` | `<m(e+μ-)>` [GeV] |
|---|---|---|---|---|---|
| truth   | +0.00421 ± 0.00275 | 0.37767 ± 0.00138 | +0.00686 ± 0.00170 | +0.00264 ± 0.00317 | 101.960 ± 0.339 |
| `madspin` | +0.00178 ± 0.00275 | 0.37839 ± 0.00138 | +0.00386 ± 0.00171 | +0.00359 ± 0.00317 | 102.559 ± 0.342 |
| `PA`      | −0.00180 ± 0.00275 | 0.37888 ± 0.00138 | +0.00500 ± 0.00171 | −0.00288 ± 0.00316 | 102.356 ± 0.343 |
| `onshell` | +0.00137 ± 0.00276 | 0.38057 ± 0.00139 | +0.00611 ± 0.00170 | +0.00385 ± 0.00316 | 102.568 ± 0.344 |
| `none`    | +0.00084 ± 0.00259 | **0.33577 ± 0.00134** | +0.00427 ± 0.00150 | +0.00436 ± 0.00316 | **104.114 ± 0.331** |

And the binned shape test — `chi2/ndf` of the per-bin ratio against the
best-fit flat line, so a pure normalisation offset does not enter it and only a
genuine shape difference does:

| observable | `madspin` | `PA` | `onshell` | `none` |
|---|---|---|---|---|
| `cos θ₁ cos θ₂` | 1.01 | 0.80 | 0.73 | **28.6** |
| `cos θ₂` | 1.38 | 1.20 | 0.72 | **15.6** |
| `cos θ₁` | 0.75 | 0.88 | 0.81 | **13.9** |
| `m(e+ mu-)` | 1.11 | 0.88 | 1.19 | **4.08** |
| `m(e+ mu+)` | 0.99 | 1.11 | 1.10 | **3.79** |
| `Delta phi(e+, mu-)` | 0.80 | 0.95 | 0.90 | 1.44 |
| `Phi` (decay planes) | 0.57 | 1.38 | 1.25 | 1.04 |
| `m_4l` | 1.30 | 1.30 | 1.30 | 1.30 |
| `pt(e+e-)` | 1.10 | **1.76** | 1.10 | 1.10 |
| `m(e+e-)` | **3.54** | **5.18** | delta fn | delta fn |
| `m(mu+mu-)` | **2.31** | **3.87** | delta fn | delta fn |

Read down the `none` column and across the `m(l+l-)` rows; those are the two
places anything happens.

### Separates: `cos θ₁`

`spinmode = none` gives `<cos²θ₁> = 0.33577 ± 0.00134`, i.e. **1/3** — a flat
decay angle, an unpolarised `Z` decaying isotropically, which is what "no spin
correlation" means. Every other mode and the truth sit at 0.378–0.381. The gap
is `0.0419 ± 0.0019`, **22 sigma**, and on the binned shape test it is
`chi2/ndf = 13.9` against 0.8–0.9 for the correlated modes. This is the
headline: the figure shows a clean U-shape for truth/`madspin`/`PA`/`onshell`
and a flat line for `none`, deviating by +40 % in the middle and −25 % at the
edges.

### Separates: `m(e+ mu-)`

The cross-pair mass, the observable that mixes one lepton from each decay.
`none` is `+2.15 ± 0.47 GeV` higher in the mean and fails the shape test at
`chi2/ndf = 4.1` (against 0.9–1.2 for the others) — isotropic decays are wider.
The separation is real but an order of magnitude weaker than `cos θ₁`.

### Does NOT separate: the decay-plane angle `Φ`

All five curves are flat and mutually consistent: `<cos 2Φ>` is
`+0.0026 ± 0.0032` for the truth and within one sigma of that for every mode,
and the binned shape test gives `chi2/ndf` of 1.0–1.4 for all of them. **A plot
where four modes lie on top of each other is the result here.**

This is a measurement about the process, not a defect of the observable.
`Φ` is generated by interference between the `ZZ` helicity amplitudes, and it is
*the* correlation observable of the `H -> ZZ -> 4l` literature — but a Higgs is
a spin-0 resonance with a fixed `m_4l`, while `g g > Z Z` is a continuum.
Integrated over the production angle and over `m_4l`, the modulation averages
away. `spinmode = none` is flat because it must be; everything else is flat
because the physics is.

### `cos θ₁ cos θ₂`: the best discriminator, but not for the reason it was picked

Its *distribution* is the sharpest separator in the study — `chi2/ndf = 28.6`
for `none`, against 13.9 and 15.6 for the two single-`Z` angles it is built
from. That is worth having, and it is the figure to look at.

But it does not measure what it was chosen to measure. The **mean**
`<cos θ₁ cos θ₂>` is the part that is purely inter-decay correlation: each
factor is separately odd under `l+ <-> l-` for an unpolarised parent, so the
mean is exactly zero for any scheme that decays the two `z` independently,
however well it gets each one's own polarisation. The truth measures
`+0.00686 ± 0.00170` — 4 sigma from zero, so the correlation **is** there — and
`none` measures `+0.00427 ± 0.00150`, only **1.1 sigma** below it. On the mean,
this observable cannot separate the modes at 50 000 events.

So the `chi2` of 28.6 comes from the *width* of the distribution, i.e. from the
two marginals' `<cos²θ>` multiplying together, not from the correlation between
them. Being precise about this matters: **for `g g > Z Z` integrated
inclusively, what MadSpin's density matrix demonstrably buys you is the
single-`Z` polarisation. The correlation between the two decays is present in
the truth at the sub-percent level and is not resolved here.**

**What "sub-percent" is worth, undiluted.** A `Z` is a weak spin analyser: the
parity-violating `cos θ` term of `Z → l+ l-` carries
`eta_l = 2 g_V g_A / (g_V² + g_A²) = 0.2193` in this calculation's on-shell EW
scheme, and it enters `<cos θ₁ cos θ₂>` squared. Dividing it out,
`C_kk = 4 <cos θ₁ cos θ₂> / eta_l² = +0.570 ± 0.141` — the correlation between
the two `Z` helicity projections, against a ceiling of `f_TT = 0.83`. The
correlation is **large**; the moment is small only because of the `eta_l²/4 =
1/83` dilution. The same figure's single-`Z` moment gives the longitudinal
fraction `f_0 = 2 - 5 <cos²θ> = 0.112 ± 0.007`, reproduced by all three
spin-correlated modes and replaced by the isotropic `1/3` by `none`. The
derivations, the literature names, the polarised-matrix-element cross-check and
the verdict on which of these belongs in the paper are in
[SPIN_COEFFICIENTS.md](SPIN_COEFFICIENTS.md).

### Does NOT separate, exactly: `m_4l`

Every MadSpin mode gives the same `m_4l` as the production sample, event by
event:

```
max |Delta m_4l|  onshell vs none : 3.3e-06 GeV   (2e-08 relative)
max |Delta m_4l|  PA      vs none : 1.1e-05 GeV   (2e-08 relative)
max |Delta m_4l|  production m(ZZ) vs decayed m_4l : 1.1e-05 GeV
```

That residual is the LHE's write precision, not a difference. MadSpin's
reshuffle redistributes virtuality between the two resonances at **fixed
`sqrt(shat)`**, so `m_4l` is untouched by construction and cannot discriminate
between modes. It is still worth plotting: against the truth it tests whether
the off-shell four-body phase space distributes `m_4l` the way an on-shell
`2 -> 2` plus decays does, and it does, at `chi2/ndf = 1.3`.

> **Read that `chi2` with the correction of 2026-08-28.** Fixing `sqrt(shat)` is
> also what puts a hard floor at `m_4l = 2 m_Z` on every mode, and the shape test
> could not see it: `plot_zz_loopinduced.py` dropped from its `chi2` every bin
> where a mode has no support (a zero numerator gives a zero error), which is
> **3 bins carrying 1.22 %** of the truth over the histogram's range, with a
> further 0.16 % below its 150 GeV lower edge. Those bins are now named in
> `data/numbers.txt`. See [PA_LOWPT_DIAGNOSIS.md](PA_LOWPT_DIAGNOSIS.md).

### The lineshape: the one place `madspin` and `PA` part company

`m(e+e-)` and `m(mu+mu-)` were meant as the sanity check that the window and the
lineshape match. They turn out to be the only observables where the two
off-shell modes differ from each other, and by a lot. `onshell` and `none` draw
no virtuality at all, so their pair mass is a delta function at `m_Z` and the
shape test is undefined for them — `numbers.txt` says so rather than printing a
large meaningless chi2.

Ratio to truth, bin by bin (each bin is ~1 GeV wide; the "fraction of sigma"
column is what the bin is worth, so the far tails are a fraction of a percent
and carry correspondingly large errors):

| `m(e+e-)` | fraction of sigma | `madspin` / truth | `PA` / truth |
|---|---|---|---|
| 58.0 | 5.4e-04 | 0.336 | 0.485 |
| 61.9 | 2.8e-04 | 1.007 | **2.806** |
| 69.7 | 6.6e-04 | 0.763 | **1.831** |
| 80.4 | 3.3e-03 | 0.946 | 1.253 |
| 88.3 | 3.9e-02 | 1.000 | 1.066 |
| **91.2** | **2.5e-01** | **1.027** | **1.014** |
| 94.1 | 4.0e-02 | 1.012 | 0.888 |
| 100.0 | 5.0e-03 | 0.951 | 0.811 |
| 109.7 | 9.8e-04 | 1.028 | 0.884 |
| 120.5 | 3.8e-04 | 1.060 | **0.424** |
| 126.3 | 2.4e-04 | 1.343 | **0.420** |

`PA` has a clean monotonic **tilt**: too hard below the pole, too soft above it,
running from roughly ×2.8 at 62 GeV to ×0.42 at 126 GeV. That is what a
per-particle fixed-width Breit-Wigner draw against an *on-shell* density looks
like when the truth's `m`-dependence comes from an off-shell matrix element and
a running width.

`madspin` — the off-shell density, which is the mode that exists to get this
right — sits at 0.95–1.03 across the whole 80–110 GeV region that carries
essentially all the rate, and only departs in the far tails. Integrated:

```
fraction of sigma below 80 GeV:  truth 2.268 %   madspin 1.752 %   PA 3.124 %
```

so `madspin` is **0.773 ± 0.035** of the truth's low tail (6.5 sigma low) and
`PA` is **1.377 ± 0.043** (8.7 sigma high). Both miss, in opposite directions,
and `madspin` misses by about half as much. Its overall shape `chi2/ndf` is 3.54
against `PA`'s 5.18 on `m(e+e-)`, and 2.31 against 3.87 on `m(mu+mu-)`.

`madspin` is better than `PA` on **every** observable with any sensitivity —
lineshape, `pt(e+e-)` (1.10 against 1.76) and `Phi` (0.57 against 1.38) — and
the two are indistinguishable everywhere else. That ordering is the expected
one, and it is the first time it has been measured for a loop-induced process.

> **Superseded, 2026-08-28.** Both halves of this paragraph are artefacts of the
> support mismatch, and at 200 000 events on the *reachable* support the picture
> is: `madspin` 0.89 and `PA` 17.13 on `m(e+ e-)` (so the lineshape ordering is
> right and much sharper than stated), but `PA` 1.59 and `madspin` 2.15 on
> `pt(e+ e-)` — the other way round. The "`madspin` is 0.773 of the truth's low
> tail, 6.5 sigma low" number above is 1.020 once the truth is restricted to
> `m_4l >= 2 m_Z`; the truth's low-mass tail that `madspin` appeared to miss is
> the sub-threshold region *no* mode can produce. See
> [PA_LOWPT_DIAGNOSIS.md](PA_LOWPT_DIAGNOSIS.md) section 4.

---

## 6. Wall times

| step | wall | notes |
|---|---|---|
| `output` of sample B | 3 min 27 s | includes the one-off CutTools + IREGI builds |
| `output` of sample A | 4 s | builds already done |
| sample A, 50 000 events | **1 min 34 s** | 16 cores, 17m39s CPU |
| sample B, 50 000 events | **54 min** | 18 cores, **13h32m** CPU |
| MadSpin `none` | 12 min 26 s | 4 cores |
| MadSpin `onshell` | 22 min 37 s | 4 cores |
| MadSpin `PA` | 22 min 57 s | 4 cores |
| MadSpin `madspin` | **85 min 12 s** | 6 cores; the most expensive mode -- off-shell MEs in the density, ~4x PA's per-event cost |
| control: sample A, no pt cut | 8 min 16 s | |
| control: sample B, no mass window | **cancelled at 1 h 39 min** | 20 000 events, never finished -- see §2d; this is itself the evidence |
| control: sample B, default lepton cuts | ~5 min | 2 000 events |

50 000 events was affordable on both sides and no statistics were reduced.
Loop-induced four-lepton is indeed the expensive one — 13.5 CPU-hours against
0.3 for the `2 -> 2` — but at roughly **15 events per CPU-second** it is a
one-hour job on 18 cores, not a blocker. Everything ran concurrently on one
18-core machine.

---

## 7. Things that cost time, for whoever does this next

* **`ptheavy` is an LO run-card parameter and loop-induced uses the LO run
  card.** `Template/NLO/Cards/run_card.dat` has no `ptheavy`, which is
  misleading if you reason from the `[noborn=QCD]` syntax; the generated
  directory is a MadEvent one.
* **`ptheavy` is hidden** from a run card whose process has no heavy final
  state, so it must be written into sample B's card by hand before it can be
  set. It stays natively inert there — `setcuts.f` flags a particle heavy only
  above 10 GeV — which is exactly what makes it safe to reuse as the custom
  cut's threshold, so both sides read one number.
* **`custom_fcts` matches function names case-sensitively** against a lowercase
  table. `LOGICAL FUNCTION DUMMY_CUTS` is rejected; `logical function
  dummy_cuts` is accepted. The rejection message is
  `function %s is not designed for overwritting` — with a literal, unformatted
  `%s`, so it does not even tell you which function it rejected.
  (`madgraph/various/banner.py`, `edit_dummy_fct_from_file`.)
* **`run.inc` needs `../../Source/vector.inc` included before it**, or the
  custom cut fails to compile with `Automatic object 's_qpdf' cannot appear in
  COMMON`.
* **Two loop-induced `output` commands must not run concurrently.** MG5 builds
  CutTools and IREGI inside the *source* tree the first time, and two outputs
  race there; one dies with `cp: includects/avh_olo.f90: No such file or
  directory`.
* **MadEvent resets `iseed` to 0 after every run.** A second run in the same
  directory is auto-seeded unless you set it again. Read the seed back out of
  the banner; the card lies.

---

## 8. Not covered

* **Only one phase-space point of the parameter space**: 13 TeV, `nn23lo1`, a
  fixed scale at `m_Z`, `BW_cut = 15`, `pt(Z) > 1 GeV`. No scale variation, no
  `BW_cut` scan, no second seed for a statistical cross-check of either sample.
* **No `4e` / `4mu` channel**, and no interference between them: the positional
  rule was used precisely to avoid it, so the identical-lepton interference that
  a real `g g > 4e` calculation carries is outside this comparison. Sample B
  excludes it by construction too (`e+ e- mu+ mu-` only).
* **The photon is excluded** (`/a`) on the truth side, so `Z/gamma*`
  interference is not tested. With a 15-width window around the `Z` pole the
  photon contribution would be small but not zero.
* **`madspin_v1` and `onshell_v1` are untested here** because MadSpin refuses
  them for loop-induced processes. That refusal was confirmed to be the reason,
  not worked around.
* **The no-window control of sample B was not run to completion.** It exists as
  an idea -- run sample B with the pt cut only, then take the fraction of its own
  weight inside the 15-width window, which is a binomial ratio within one sample
  and would pin the retained fraction of the full integrand to about 0.2 % at
  20 000 events. It would sharpen §3's residual from "consistent with what #379
  documents" to "equal to it". It was stopped at 1 h 39 min because the run's
  own slowness had already made the point the cut needed to make, and because
  the residual is measured directly against sample B anyway. Whoever picks it up
  should budget more than 2 hours for 20 000 events.
* **`Phi` was measured to be flat, not proven to be.** A differential
  measurement — `Phi` in bins of `m_4l` or of the production angle, where the
  interference does not average away — could still separate the modes and was
  not attempted.
* The **`spinmode = none` value of `<cos θ₁ cos θ₂>`** is `+0.00427 ± 0.00150`
  where the exact answer is 0. That is 2.8 sigma, stable across both halves of
  the sample, and with three such moments tested it is not alarming — but it was
  not chased down.
* **`ms_dir` reuse across spinmodes was not used**, and the reason is a hazard
  rather than a measurement: `run_from_pickle` restores the *pickled* option
  object, so a reused directory carries the first run's `spinmode` and `BW_cut`
  into every lookup through `decay_all_events.options`. The kinematics were
  verified to follow the new spinmode; the normalisation path was not audited.
