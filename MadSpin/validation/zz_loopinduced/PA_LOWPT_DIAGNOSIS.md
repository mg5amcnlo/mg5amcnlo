# The `PA` low-`pt` deficit: what it is

`RESULTS.md` reports, for `pt(e+ e-)` at 200 000 events,

```
madspin  shape chi2/ndf = 1.23      PA  shape chi2/ndf = 3.32      onshell  1.08
```

with `PA` coherently low below 20 GeV — every bin pulling −3 to −6 sigma, about
−6 % of the rate in that region — and the three other modes not.  The obvious
reading is that something `PA` alone does is wrong, and the obvious suspect is
that `PA` is the only mode that reshuffles the production while taking the
density matrix on shell.

**That reading is wrong, and so is the suspect.**  The deficit is real,
reproducible and correctly measured; it is not a property of `PA`.  It is the
part of the truth that **no** spinmode can produce — `m_4l < 2 m_Z` — showing up
in `PA` and not in `onshell` only because `PA`'s normalisation is right and
`onshell`'s is 5.2 % high, which is very nearly the size needed to fill the same
hole.  Two errors of opposite sign in `onshell` cancel on this observable; one
error in `PA` does not.

Everything below is produced by [`pa_lowpt_diagnosis.py`](pa_lowpt_diagnosis.py)
from the per-event columns of the same 200 000-event run
(`data/meta.json`, `runs/<mode>/event_columns`).  Its `--selftest` asserts the
load-bearing claims; the derived histograms are cached in
`data/pa_lowpt_diagnosis.npz` (11 kB) so the figure
[`plots/pa_lowpt_diagnosis.pdf`](plots/pa_lowpt_diagnosis.pdf) and every table
here redraw without the 110 MB of columns.  **Nothing was regenerated.**

---

## 0. The answer in one paragraph

`g g > z z` puts both `z` on shell, so every production event has
`m_4l = sqrt(shat) >= 2 m_Z = 182.376 GeV`, and the RAMBO reshuffle holds
`sqrt(shat)` fixed — so at `2 -> 2` **every MadSpin mode has exactly zero
support below `2 m_Z`**.  The truth has **2.0915 %** of its cross section there,
and that rate is **entirely at low `pt`**: 9.5 % of the truth's `pt < 5 GeV`
bin, 8.8 % of `5-10`, 7.1 % of `10-15`, 4.9 % of `15-20`, 0.2 % by `40-50` and
nothing at all above 60 GeV.  Charge that hole to the mode and you get exactly
the observed deficit.  Restrict the truth to `m_4l >= 2 m_Z` — the support
MadSpin can reach — and `PA`'s `pt(e+ e-)` ratio below 20 GeV becomes
**0.995, 1.018, 1.013, 1.021**, flat within 1.7 sigma of its own normalisation,
and its shape `chi2/ndf` falls from **3.32 to 1.59** — the **best** of the three
spin-correlated modes, `madspin` 2.15 and `onshell` 2.41.  The reshuffling is
**not** responsible: dividing the reshuffling jacobian out of the `PA` sample
event by event leaves the deficit against the full truth in place
(`0.949 / 0.938 / 0.952` at 5-20 GeV).  It is responsible for something else —
the `m(l+ l-)` tilt of `RESULTS.md` section 5 — and that is a genuine and
quantified property of the production approximation, not a bug.

---

## 1. What was excluded, and how

| confounder | verdict | evidence |
|---|---|---|
| mass window differs between the two sides | **no** | truth `m(e+e-)` spans 54.567 – 127.776, `PA` 54.568 – 127.798; the window is `m_Z +- 15 Gamma_Z` = 54.567 – 127.809 on both |
| the deficit is a normalisation, not a shape | **no** | it survives normalisation: the study's `chi2` is already against each mode's own flat line, and the per-bin *shape* pulls are −6.4, −6.2, −5.5, −2.9 |
| the truth's own MC error is not propagated | **no** | every ratio error here is `sqrt((e_mode/y_mode)^2 + (e_truth/y_truth)^2)` from `sum w^2` on both sides; the truth carries a single weight, so its per-bin error is `1/sqrt(N_bin)` and it is included |
| the total rate is not preserved | **partly, and it matters** | see section 3 |
| the `pt(ll) > 1 GeV` cut is applied differently on the two sides | **yes, but small and of the opposite sign** | section 5 |
| the Wigner-rotation bug that T112 fixed | **no** | `pt(e+ e-)` builds no decay frame; and the fix is already in this branch |
| **the two samples do not have the same support** | **THIS** | sections 2–4 |

---

## 2. The support mismatch

`g g > z z` is a `2 -> 2` production of two on-shell `z`.  Its `sqrt(shat)`
therefore starts at `2 m_Z`.  `Event.reshuffle_production` boosts to the
partonic CM, scales all final-state three-momenta by one common `chi` and
solves for `chi` at **fixed** `sqrt(shat)` — so `m_4l` is untouched.  Measured
rather than assumed, over the paired 200 000 events:

```
max |m_4l(mode) - m_4l(none)|   1.63e-05 GeV      (the LHE write precision)
```

so:

| | `sigma` [fb] | ratio to the full truth |
|---|---|---|
| **truth**, whole window | 731.10 | 1 |
| truth, `m_4l >= 2 m_Z` | **715.81** | 0.97909 |
| truth, `m_4l <  2 m_Z` | **15.29** | **0.02092** |
| `PA` | 735.45 | 1.00595 |
| `madspin` | 736.61 | 1.00754 |
| `onshell` / `none` | 769.07 | 1.05194 |

`PA`, `madspin`, `onshell` and `none` all have **zero** events below `2 m_Z`, and
would at any sample size.  The truth's 2.09 % is not a tail of a distribution
MadSpin gets slightly wrong; it is a region of phase space the method does not
have.

Where it sits in `pt(e+ e-)` — the column `f_sub` is the share of each truth
bin that lies below `2 m_Z`:

| `pt(e+e-)` [GeV] | truth [fb] | `f_sub` | `PA` / truth (all) | `PA` / truth (`m_4l >= 2 m_Z`) |
|---|---|---|---|---|
| 0 – 5 | 22.89 | **9.53 %** | 0.9002 ± 0.0166 (−6.4 σ) | **0.9951 ± 0.0187 (−1.7 σ)** |
| 5 – 10 | 41.84 | **8.82 %** | 0.9282 ± 0.0125 (−6.2 σ) | **1.0179 ± 0.0140 (−0.7 σ)** |
| 10 – 15 | 47.32 | **7.12 %** | 0.9404 ± 0.0119 (−5.5 σ) | **1.0125 ± 0.0130 (−1.2 σ)** |
| 15 – 20 | 48.18 | **4.87 %** | 0.9711 ± 0.0121 (−2.9 σ) | **1.0209 ± 0.0128 (−0.5 σ)** |
| 20 – 25 | 46.61 | 3.48 % | 1.0227 (+1.3 σ) | 1.0596 (+2.4 σ) |
| 25 – 30 | 44.86 | 2.08 % | 1.0346 (+2.2 σ) | 1.0566 (+2.2 σ) |
| 30 – 35 | 42.77 | 1.51 % | 1.0366 (+2.3 σ) | 1.0525 (+1.8 σ) |
| 35 – 40 | 41.33 | 0.80 % | 1.0226 (+1.2 σ) | 1.0309 (+0.3 σ) |
| 40 – 50 | 74.07 | 0.22 % | 1.0452 (+3.8 σ) | 1.0474 (+1.9 σ) |
| 50 – 60 | 63.00 | 0.01 % | 1.0435 (+3.4 σ) | 1.0436 (+1.5 σ) |
| 60 – 600 | 258.1 | 0.00 % | 1.008 | 1.008 |

(sigmas are shape pulls, i.e. against each comparison's own flat line.)  The
monotonic fall of `f_sub` from 9.5 % to zero and the monotonic recovery of the
`PA` ratio from 0.900 to 1.02 are the same curve.  On the reachable support the
whole `pt < 20 GeV` region is `PA / truth = 1.0140`, against `1.0274` for the
sample as a whole: `PA` is *slightly low* there, by 1.3 %, not 6 % low.

The full shape test, both supports, over the study's own 2 GeV grid:

| `pt(e+ e-)` | full truth support | truth `m_4l >= 2 m_Z` |
|---|---|---|
| `PA` | rate 1.0057, **chi2/ndf 3.32** | rate 1.0272, **chi2/ndf 1.59** |
| `madspin` | rate 1.0073, chi2/ndf 1.23 | rate 1.0289, chi2/ndf **2.15** |
| `onshell` | rate 1.0517, chi2/ndf 1.08 | rate 1.0742, chi2/ndf **2.41** |

**The ranking inverts.**  On the support the two samples share, `PA` is the best
of the three on this observable and `onshell` the worst.

### Why only `PA` shows it

`onshell` and `none` draw no virtuality, so they take no Breit-Wigner truncation
and their cross section is `1/f^2 = 1.0457` too big (`RESULTS.md` section 3 —
correct behaviour, verified there to twelve digits).  In the `pt < 5 GeV` bin
`onshell` sits at 1.1819 against the reachable truth and the hole is −9.53 %;
`1.1819 x (1 - 0.0953) = 1.0692`, which is `onshell`'s measured ratio against
the full truth to four digits.  Over the whole spectrum the +5.2 % and the
−2.1 % nearly cancel and the shape looks flat.
`PA`'s normalisation is right (+0.6 % on the total), so nothing cancels the
hole and it appears as a shape.  `madspin`'s low-`pt` bins are separately
inflated — partly by the `pt > 1 GeV` cut asymmetry of section 5 — which does
the same job less completely.

**`onshell`'s `chi2/ndf = 1.08` on `pt(e+ e-)` is not evidence that `onshell`
describes this observable.  It is two errors of opposite sign.**

---

## 3. Where the rate goes

The total is **not** preserved on the reachable support, and this is the more
serious of the two statements.

MadSpin normalises to `sigma_prod x BR x f^2`, where `sigma_prod` is the
**on-shell** `g g > z z` cross section.  Nothing in that product knows about the
`m_4l < 2 m_Z` region, so the whole of it lands above threshold:

* `PA` / truth over the whole window: **+0.60 %**
* `PA` / truth over `m_4l >= 2 m_Z`: **+2.74 %**

i.e. the frequently quoted "+0.6 %, 14 sigma" agreement of the totals is itself
a partial cancellation of **+2.7 % above threshold** against **−2.1 % below**.

Integrated in `pt`, against the reachable truth:

| `pt(e+ e-)` [GeV] | share of the truth | `PA` / truth (all) | `PA` / truth (reachable) |
|---|---|---|---|
| 0 – 20 | 21.9 % | 0.9407 | **1.0140** |
| **20 – 60** | **42.8 %** | **1.0358** | **1.0482** |
| 60 – 130 | 28.1 % | 1.0094 | 1.0094 |
| 130 – 600 | 7.2 % | 1.0135 | 1.0135 |

**The compensating region is `20 < pt < 60 GeV`**, which carries 43 % of the
cross section and is +4.8 % high.  Two things live there: the +2.7 %
normalisation just described, and a genuine `PA` shape excess of about +2 %
which *is* the reshuffling (section 4) — `PA` with the jacobian removed is
+2.2 % there instead of +4.8 %, against a total of +1.7 % instead of +2.7 %.

---

## 4. The reshuffling: what it does and what it does not do

### The reshuffle is fully characterised, event by event, with no new run

RAMBO scales all CM three-momenta by one `chi` and leaves directions alone, so
for a `2 -> 2` production

```
pt(after) / pt(before) = chi = lambda^(1/2)(shat, m1^2, m2^2)
                               / lambda^(1/2)(shat, mZ^2, mZ^2)
```

`onshell` draws no virtuality and never reshuffles, so `pt(onshell)` **is** the
before-reshuffle `pt` of the same production event.  Measured:

```
max | chi - pt(PA)/pt(onshell) |        1.5e-05
max | chi - pt(madspin)/pt(onshell) |   4.0e-05
```

to the LHE write precision, on all 200 000 paired events.  This is the
before/after evidence the brief asked for, and it needed no new sample.

| `chi` | min | p5 | median | mean | p95 | max | fraction > 1 |
|---|---|---|---|---|---|---|---|
| `PA` | 0.029 | 0.876 | 1.001 | **1.047** | 1.269 | 22.9 | 0.538 |
| `madspin` | 0.004 | 0.851 | 1.000 | **1.021** | 1.191 | 25.3 | 0.487 |

`PA` inflates `pt` by 4.7 % on average, `madspin` by 2.1 %.  The tail to
`chi = 23` is the `m1 + m2 -> sqrt(shat)` edge and is 0.26 % of `PA`'s events
below `chi = 0.5`.

### It is not the cause of the low-`pt` deficit

`PA` with `density_keep_jacobian = False` reshuffles *after* the accept/reject,
so its accepted mass sets are distributed without the `chi` factor.  That is
exactly the shipped `PA` sample reweighted by `1/chi` — no rerun needed.  Doing
it, against the **full** truth:

| `pt(e+e-)` [GeV] | `PA` | `PA` with `1/chi` |
|---|---|---|
| 0 – 5 | 0.9002 | 0.9750 |
| 5 – 10 | 0.9282 | **0.9494** |
| 10 – 15 | 0.9404 | **0.9376** |
| 15 – 20 | 0.9711 | **0.9516** |

The deficit does not go away; in two of the four bins it gets *deeper*.  The
shape `chi2/ndf` falls (3.32 → 1.93) only because removing `chi` also removes
the compensating 20–60 GeV excess, which lowers the flat line the shape test
measures against.  **Removing the reshuffling jacobian does not remove the
deficit; restricting the truth to the reachable support does, completely.**

### What the reshuffling *is* responsible for

The `m(l+ l-)` tilt of `RESULTS.md` section 5.  `chi > 1` for mass sets lighter
than the pole, so folding `chi` into the accept/reject pulls `PA`'s virtualities
light — `<m(e+e-)> = 90.749` for `PA` against 91.155 for the truth and 91.330
for `madspin`.  Shape `chi2/ndf` on `m(e+ e-)`:

| | full truth | truth `m_4l >= 2 m_Z` |
|---|---|---|
| `PA` | 11.73 | **17.13** |
| `PA` with `1/chi` | 2.55 | 4.34 |
| `madspin` | 8.15 | **0.89** |

Two results here, and the second is a correction to `RESULTS.md`:

1. `PA`'s lineshape tilt is real, is **not** an artefact of the support, and is
   four times smaller with the jacobian removed.  It is the content of the
   production approximation: `PA` gets the phase-space factor right (`chi` is
   the correct `2 -> 2` reshuffling jacobian) and the matrix element wrong
   (on-shell), and the truth's `|M_off|^2 / |M_on|^2` happens to run roughly
   like `1/chi`, so *dropping* a formally correct factor makes `PA` look better
   on this process.  That is a coincidence of `g g > ZZ`, not a reason to
   change the default — see section 6.
2. **`madspin` does not miss the lineshape at all.**  `RESULTS.md` reports it as
   "0.773 ± 0.035 of the truth's low tail, 6.5 sigma low" on
   `sigma(m(e+e-) < 80 GeV)`.  That whole deficit is the support mismatch: the
   sub-threshold truth events are, necessarily, events where both pairs are
   light.  On the reachable support the fractions are truth **1.806 %**,
   `madspin` **1.842 %** — a ratio of **1.020**, not 0.773.  And `madspin`'s
   `m(e+ e-)` shape `chi2/ndf` is **0.89 over 75 bins**: the off-shell density
   reproduces the truth lineshape essentially perfectly.  `PA` moves the other
   way, 1.299 → **1.699**.

---

## 5. A second, smaller acceptance mismatch, of the opposite sign

`README.md` applies `pt(z) > 1 GeV` on sample A through the run card's
`ptheavy`, and the same cut to the *reconstructed* pairs of the truth.  At
`2 -> 2` `pt(z1) = pt(z2)` exactly, so the OR and the AND coincide — that part
is right.  What the argument does not cover is that in sample A the cut acts on
the **on-shell `z`, before MadSpin reshuffles**, while on the truth it acts on
the final reconstructed pair.  `chi < 1` events therefore migrate below 1 GeV
after the cut has already been passed:

| mode | share of `sigma` with `pt(e+e-) < 1 GeV` |
|---|---|
| `PA` | 0.0155 % |
| `madspin` | **0.1891 %** |
| `onshell` / `none` | 0 |
| truth | 0 (hard cut) |

Small overall, but it is 1.39 fb dropped into a `0 < pt < 5 GeV` bin whose
reachable-truth content is 20.7 fb — so it accounts for roughly half of
`madspin`'s +13 % in that bin.  Both samples' first `pt` bin is therefore
contaminated, in opposite directions, and neither contamination is about spin
correlations.

---

## 6. Verdict

**Not a bug — in MadSpin or in `PA`.**  There is nothing to fix in
`MadSpin/` or in `madgraph/various/lhe_parser.py`, and nothing here that should
change what `PA` output is.

* The `2 m_Z` support mismatch is the defining, documented limitation of
  decaying an on-shell production sample.  At `2 -> 2` it is a hard zero for
  every spinmode.  `MadSpin/validation/mtt_threshold/` measured the same thing
  for `p p > t t~ j` and found the sub-threshold rate is 0.165 % of `sigma`
  there (and *non*-zero for `madspin`/`PA`, because a recoil jet joins the
  reshuffle and lets `m_tt` cross the threshold — a `2 -> 3` effect that does
  not exist here).  This study's 2.09 % is much larger because `15 Gamma_Z` is
  ±41 % of `m_Z` while `15 Gamma_t` is ±12 % of `m_t`.  The same signature is
  visible in `zz_nlo`, where all four modes are low in the first `pt(e+ e-)`
  bins and the `m_4l < 2 m_Z` region is populated only partially.
* The `chi` jacobian is the correct `2 -> 2` phase-space factor and `PA` is
  right to carry it.  That it makes the `g g > ZZ` lineshape worse is a
  statement about `|M_off|^2 / |M_on|^2` for this process, i.e. about the
  production approximation, which is what `PA` is.  **`density_keep_jacobian`
  should not be flipped on this evidence**: one process, one observable, and
  `doc/madspin_ae_normalisation/assessment.md` section 2 measures the no-jacobian
  variant as the *worst* of the three modes near the `t t~` threshold (`A_e`
  48 % below its plateau, against `PA`'s +23 %).
* T57's dropped `A_e` is **not** this.  `A_e` would bias *towards* low
  `sqrt(shat)`, i.e. towards low `pt` — the opposite sign to what is observed —
  and it is a sub-percent effect where this one is 2.09 % of `sigma` with a hard
  zero underneath it.

### Can the paper keep quoting `PA` alongside the other modes?

**Yes, and it should.**  On the support the two samples share, `PA` is the best
of the three spin-correlated modes on `pt(e+ e-)` (1.59 against `madspin` 2.15
and `onshell` 2.41) and indistinguishable from them on all seven angular and
cross-pair observables.  Its one real weakness is the `m(l+ l-)` lineshape,
which is worse than reported once the support is matched (17.13 against
`madspin`'s 0.89) and which `RESULTS.md` already identifies and explains.

What must **not** be quoted:

* `pt(e+ e-)` `chi2/ndf = 3.32` as a defect of `PA`.  It is the support
  mismatch, and `onshell`'s 1.08 on the same observable is a cancellation of two
  errors and is worth less than `PA`'s 3.32.
* "`madspin` is 6.5 sigma low on the `m(l+ l-)` low tail".  It is 1.020 ± 0.02
  on the reachable support.
* Any statement of the form "`madspin` is better than `PA` on every observable
  with any sensitivity" (`RESULTS.md` section 5).  On the matched support it is
  better on the lineshape and worse on `pt`.

### What the validation should say instead

1. State the support mismatch as a headline number: **the truth carries 2.09 %
   of its cross section below `2 m_Z`, and every MadSpin mode has exactly zero
   there**; the `+0.60 %` agreement of the totals is `+2.74 %` above threshold
   against `−2.09 %` below.
2. Quote `pt(e+ e-)` (and any other `m_4l`-correlated observable) on both
   supports, or at least annotate the full-support number.  The two same-`Z`
   azimuthal separations added on 2026-08-28, `dphi_ee` and `dphi_mumu`, are
   exactly such observables — `Delta phi` between two leptons of one `Z` is
   anti-correlated with that `Z`'s `pt` at −0.65, so the same hole reappears
   there at *large* `Delta phi` and the same ranking inversion follows
   (`PA` 3.33 / 3.45 → 1.19 / 1.34).  See [DPHI_PAIRS.md](DPHI_PAIRS.md).
3. `plot_zz_loopinduced.py` **silently dropped from its shape `chi2`** every bin
   where a mode has no support against a truth that does — the `re_ > 0` test
   in `ratio()` removes them, because a zero numerator gives a zero error.  For
   `m_4l` that is 3 bins carrying 1.22 % of the truth over the histogram's
   range, plus a further 0.16 % below the histogram's 150 GeV lower edge and a
   180–190 GeV bin that straddles `2 m_Z`.  That is how a hard, infinitely
   significant zero was reported as `chi2/ndf = 2.03`.  **Fixed here**: those
   bins are now named and their share of the truth quoted, in `data/numbers.txt`.
4. Note that sample A's `pt > 1 GeV` acts before the reshuffle and the truth's
   after it (section 5), or apply the cut to the reconstructed pairs on both
   sides.

---

## 7. Reproducing

```
python3 pa_lowpt_diagnosis.py             # from the event columns; rewrites the cache
python3 pa_lowpt_diagnosis.py --from-cache --selftest
```

`--selftest` asserts the four claims this note stands on: the deficit is there
against the full truth, it is not there against the reachable truth, removing
the reshuffling jacobian does not remove it, and the samples are paired event by
event to the LHE write precision.  Full output:
[`plots/pa_lowpt_diagnosis.txt`](plots/pa_lowpt_diagnosis.txt).
