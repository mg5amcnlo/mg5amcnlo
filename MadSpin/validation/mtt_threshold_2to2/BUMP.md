# The bump just above `2 m_t`, and whether it compensates the empty region

A reading of the figure in [`RESULTS.md`](RESULTS.md)
(`plots/mtt_threshold.pdf`), asked and answered from the committed histograms.
**No events were generated for this.** Script: `analyse_bump.py`. Full numeric
report: `plots/bump_numbers.txt` (identical copy in `plots_userstyle/`).
Figure: `plots/mtt_bump.pdf` / `.png` (MG7 style),
`plots_userstyle/mtt_bump.pdf` / `.png` (user style).

The two questions:

1. There is a bump in the shape ratio just above `2 m_t`, peaking at
   **1.122 +- 0.030** in the 349-350 GeV bin. What is it?
2. Are the total cross sections identical -- does the bump compensate the
   region below threshold, where every spinmode is exactly 0?

---

## 0. The answer

**The bump is not MadSpin having too many events. It is the truth having too
few, because the truth's `m_tt` is the on-shell production spectrum smeared by
the two top virtualities and the smearing carries rate *out* of the turn-on --
into the sub-threshold region and into the first GeV above threshold.** MadSpin
is the same production sample unsmeared, so the ratio dips below 1 in the first
bin (0.812) and rises above it (1.12) over the next 10 GeV. Smearing the
committed production histogram with the truth's own Breit-Wigner reproduces the
turn-on to **1.6 %** where the unsmeared sample is off by **19 %**, and recovers
**93 %** of the sub-threshold rate where MadSpin has none.

**Whether it compensates depends on which normalisation the question is asked
in, and the two answers are different.**

| normalisation | excess over the truth, `2 m_t` to 370 GeV | truth's sub-threshold rate | balance |
|---|---|---|---|
| self-normalised (**the figure**) | `+0.17230 % +- 0.02683 %` of `sigma` | `0.17456 % +- 0.00187 %` | **0.99 +- 0.15** |
| rate-matched (mode scaled by `sigma_truth/sigma_mode`) | `+0.9129 +- 0.1421 pb` | `0.9249 +- 0.0099 pb` | **0.99 +- 0.15** |
| absolute pb | `+2.4230 pb` in the window, `+24.380 pb` over everything above threshold | `0.9249 pb` | **2.6 : 1, and 26 : 1 -- no** |

The first two rows are the same statement. The third is a different statement
and it is the one that answers "are the total cross sections identical": **they
are not.** Truth 529.824 pb against 553.279 pb, `truth/mode = 0.9576`. That
4.24 % is the Breit-Wigner truncation, a global rate loss over the whole `m_tt`
spectrum (`RESULTS.md` section 3), and it swamps the 0.175 % near threshold by a
factor 26. Nothing about the bump compensates it, and nothing was ever going to.

Numbers quoted from `onshell`, which took no overweight events; see section 4.

---

## 1. The bump is not the self-normalisation

The figure divides every curve by its own total cross section. A mode with zero
sub-threshold rate therefore has its whole above-threshold curve scaled up
relative to the truth's shape by

```
1 / (1 - f_below) - 1 = +0.17487 %,     f_below = 0.17456 % +- 0.00187 %
```

and that is **flat** -- the same in every bin from `2 m_t` to the end of the
spectrum. The bump is `+12.2 %` at its peak, **70 times** that. So the bump is a
redistribution of rate inside the spectrum, not an artefact of dividing by a
number.

## 2. Where the excess actually sits

`plots/numbers.txt` records shape ratios of `0.842 / 0.900 / 0.812 / 0.812` in
the first bin above threshold: **MadSpin is below the truth there.** So the
excess is not "MadSpin piles up where the truth spilled below". Disjoint
windows, so these add and their errors are independent:

| window [GeV] | mode | truth | excess | |
|---|---|---|---|---|
| `[346, 347)` | 0.05760 % | 0.07090 % | **-0.01330 % +- 0.00268 %** | -5.0 sigma |
| `[347, 352)` | 0.82030 % | 0.74394 % | **+0.07636 % +- 0.00980 %** | +7.8 sigma |
| `[352, 356)` | 0.97430 % | 0.92790 % | `+0.04640 % +- 0.01072 %` | +4.3 sigma |
| `[356, 362)` | 1.78830 % | 1.74906 % | `+0.03924 % +- 0.01449 %` | +2.7 sigma |
| `[362, 370)` | 2.79760 % | 2.77400 % | `+0.02360 % +- 0.01805 %` | +1.3 sigma |
| `[370, 380)` | 3.92500 % | 3.90312 % | `+0.02188 % +- 0.02126 %` | +1.0 sigma |
| `[380, 420)` | 17.00110 % | 16.89584 % | `+0.10526 % +- 0.04113 %` | +2.6 sigma |
| `[420, 470)` | 18.76130 % | 18.78202 % | `-0.02072 % +- 0.04277 %` | -0.5 sigma |
| `[470, 520)` | 14.26530 % | 14.29606 % | `-0.03076 % +- 0.03832 %` | -0.8 sigma |

A 5 sigma **deficit** in the first bin, then an excess that is half spent by
352 GeV and effectively finished by 370 GeV. The running total crosses zero at
348.25 GeV (`onshell` / `madspin_v1`; 348.00 for `madspin`, 347.50 for `PA` --
the difference is the overweights).

## 3. The integral, against the sub-threshold rate

The cumulative excess `F_mode(x) - F_truth(x)` from `2 m_t` upwards is the top
pane of `plots/mtt_bump.pdf`. Errors are binomial on the CDF -- the LHE carries
`event_norm = average`, so the events are equal-weight bar the overweights, and
a bin-by-bin Poisson sum would badly overstate the error at the far end.

| `x` [GeV] | `onshell` | vs `f_below` |
|---|---|---|
| 350 | `+0.02948 % +- 0.00753 %` | -19.3 sigma |
| 356 | `+0.10946 % +- 0.01470 %` | -4.4 sigma |
| 362 | `+0.14870 % +- 0.02045 %` | -1.3 sigma |
| **370** | **`+0.17230 % +- 0.02683 %`** | **-0.1 sigma** |
| 380 | `+0.19418 % +- 0.03334 %` | +0.6 sigma |
| 420 | `+0.29944 % +- 0.04881 %` | +2.6 sigma |
| 520 | `+0.24796 % +- 0.05359 %` | +1.4 sigma |

**The excess reaches the truth's entire sub-threshold rate at `2 m_t + 24 GeV`,
over a window holding 6.27 % of the cross section**, and from there to the top
of the histogram it adds `+0.07566 % +- 0.05460 %`, 1.4 sigma from nothing.

Two honesties about that number.

* **Part of it is an identity, and part of it is a measurement.** Over the
  *full* `m_tt` range the self-normalised excess above threshold is *forced* to
  equal `f_below`, because both curves integrate to 1 and the mode has nothing
  below. What is measured is not that the total comes out right -- it is **where
  it comes out**: all of it inside 24 GeV of threshold. The committed histogram
  stops at 520 GeV and holds only 60.3 % of `sigma`, so the identity is not
  available to the numbers above as a shortcut.
* **The normalisation systematic is larger than the statistical error.**
  Dividing each side by its own *total* `sigma` assumes the 4.24 % truncation
  loss is flat in `m_tt`. It is not quite: the parent study measures
  `truth/mode = 0.9576` on the totals against `0.9517 +- 0.0025` on the
  380-420 GeV anchor, a 0.6 % difference, and that same 0.6 % is what the
  cumulative picks up as `+0.105 % +- 0.041 %` over `[380, 420)`. Renormalising
  on the anchor instead moves the answer to `+0.13244 %`, i.e.
  **0.76 x `f_below`** rather than 0.99 x.

So: **`0.76` to `1.00` times the sub-threshold rate**, depending on which of two
defensible normalisations is used, against a statistical error of `+-0.15`. The
normalisation choice, not the statistics, is what limits how sharply the
compensation can be stated -- and the thing limiting it is a broad-band effect
the parent study already records, not anything to do with the threshold.

## 4. The overweights do not contaminate it

`RESULTS.md` section 2: `madspin` (joint) carries 21.11 event-equivalents of
overweight excess and `PA` (sequential) 69.53, all of it within ~1 GeV of
`2 m_t`; `onshell` and `madspin_v1` carry none and are the production sample
weight for weight. Measured on the same cumulative:

| mode | shift vs `onshell` at 370 GeV | of the measured excess | of its statistical error |
|---|---|---|---|
| `madspin` | `+0.00198 %` | 1.1 % | 7 % |
| `PA` | `+0.00650 %` | 3.8 % | 24 % |

**At most a quarter of the statistical error, so no.** They *are* the whole of
the visible spread between the four curves in the 346-347 GeV bin
(`0.812 / 0.842 / 0.900`), which is why that bin defines nothing here and why
the integration window starts at `2 m_t` rather than inside it. Every headline
number above is `onshell`'s.

## 5. The mechanism, constructed rather than asserted

**Hypothesis.** At virtualities `m1`, `m2` the pair threshold is `m1 + m2`, not
`2 m_t`. The production spectrum is exactly zero below `2 m_t` and rises like
`sqrt(M - 2 m_t)` above it -- **concave** -- so smearing it (i) fills the region
below threshold, (ii) *raises* the first GeV above it, where the sharp spectrum
is still near zero, and (iii) *lowers* the concave rise that follows. MadSpin's
`m_tt` is the unsmeared spectrum, because `m_tt = sqrt(shat)` at this
multiplicity and the reshuffle holds it fixed. Predicted: `mode/truth < 1` in
the first bin, `> 1` over the next ~10 GeV, `= 1` thereafter.

**Test.** Smear the committed production histogram with the Breit-Wigner the
truth used -- fixed-width relativistic, truncated at `15 Gamma_t`, times the
decay numerator `m Gamma(m) / (m_t Gamma_t)` -- with two independent
approximations:

* **A**, threshold shift: `M -> M + (m1 + m2 - 2 m_t)`. Needs only the
  production histogram.
* **B**, velocity factor: write `dsigma_prod/dM = Phi(M) beta_0(M)`, fit the
  smooth `Phi` over 346.5-460 GeV and extrapolate it below `2 m_t`, then rebuild
  with the exact `beta(M; m1, m2)`.

Neither is tuned to the truth. Largest `|ratio - 1|` against the truth, by
region:

| region [GeV] | model A | model B | MadSpin |
|---|---|---|---|
| deep tail 316-336 | 39.5 % | 24.8 % | 0 exact (no support) |
| below `2 m_t` 336-346 | 18.1 % | 9.3 % | 0 exact (no support) |
| **the bump 346-356** | **4.4 %** | **1.6 %** | **18.8 %** |
| recovery 356-380 | 2.9 % | 2.0 % | 3.3 % |
| flat 380-420 | 0.9 % | 1.2 % | 1.9 % |

and the sub-threshold rate, which is the thing MadSpin misses entirely:

| | `sigma(m_tt < 2 m_t)` / `sigma` | of the truth's |
|---|---|---|
| truth | 0.17456 % | -- |
| model A | 0.15155 % | 87 % |
| model B | 0.16228 % | **93 %** |
| MadSpin | **0 exactly** | 0 % |

and the cumulative excess of section 3 with the model in place of MadSpin --
if the smearing *is* the mechanism, the model must not have the excess:

| `x` [GeV] | MadSpin | model A | model B |
|---|---|---|---|
| 356 | `+0.10946 % +- 0.01470 %` | `-0.03345 %` | `-0.01359 %` |
| 370 | `+0.17230 % +- 0.02683 %` | `-0.11862 %` | `-0.04737 %` |

**Verdict: the smearing accounts for the whole of the bump, and slightly
overshoots it.** The model's residual is a roughly uniform ~1 % normalisation
offset against the truth across the drawn window, not a failure of the turn-on
shape. The model is leading order -- the matrix element is taken to depend on
the virtualities only through the phase space -- and is not a substitute for the
truth; its deep tail, where it is off by 25-40 %, holds 0.001 % of `sigma` and
nothing here rests on it.

---

## 6. What this does not settle

* **The model's ~1 % normalisation offset is not resolved into its parts.**
  Candidates: the 4.24 % truncation is not perfectly flat in `m_tt` (the 0.6 %
  of section 3), and the model's neglect of the virtuality dependence of the
  matrix element. Separating them needs a truth run at a second `bwcutoff`,
  which does not exist.
* **Which normalisation is "right" for the compensation question is not
  settled**, and it is what moves the answer from 0.76 to 1.00. Both are
  defensible; the figure uses the total.
* **`bwcutoff = 15` only.** How far below threshold the truth reaches -- and so
  the size of everything measured here -- is a parameter, not a prediction.
* **Doubly-resonant only, on both sides.** Single- and non-resonant
  `W+ b W- b~` diagrams are absent from the truth as well as from MadSpin, so
  nothing here bounds their contribution near threshold.
* **One seed per mode.** The overweight numbers of section 4 are one draw;
  nothing else depends on them.
* **`m_tt` only.** That the virtualities are invisible in `m_tt` at this
  multiplicity is not a statement that the spinmodes agree -- see
  `RESULTS.md` section 5.
