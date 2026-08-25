# `Z_0Z_T` against `Z_TZ_0`: why their K-factors differ, and whether that is real

Base of this pass `3cc1579bb`. **No HepMC was read and nothing was generated.**
Every number below comes off `data/weights.npz` (NLO, `run_06_decayed_1`) and
`data/weights_LO.npz` (LO, `run_12_decayed_1`), through
`lt_tl_analysis.py`, which is the script that produced all of it and the two
figures.

---

## The question, and the answer

The K-factor panel of `plots*/kfactor_LO_NLO/dphi_ee.*` shows

```
K(Z_TZ_0) = 2.94 +- 0.18      against      K(Z_0Z_T) = 1.47 +- 0.06
```

in the lowest `Δφ(e+e-)` bin, while the two components' **integrated**
K-factors agree to 0.0007 on 0.0063 bars. Is the split an analysis artefact?

**It is physics.** Not an artefact, and not in any part an index-mapping bug.
Three independent measurements say so, and the second of them is the one the
brief nominated as decisive.

---

## The discriminator, and what it returned

`ZZ` is symmetric under exchanging the two `Z`, and the MadSpin card lists
`decay z > e+ e-` **before** `decay z > mu+ mu-` (checked in the banner of both
runs, `run_06_decayed_1_tag_1_banner.txt` lines 636-637, identical in
`run_12`), so MadSpin's positional rule makes the first weight index the
electron-`Z`. The
exchange `e <-> mu` therefore maps `Z_0Z_T` onto `Z_TZ_0`.

* **`M(e+ mu+)` is exactly invariant under that exchange**, and so is its
  four-lepton fiducial, which cuts all four leptons alike. The two components
  **must** agree bin by bin.
* **`Δφ(e+e-)` is built from the electron-`Z` alone.** It is blind to the
  muon-`Z` and is *not* invariant. The two components are under no obligation
  to agree.

Measured, with the delta-method bar (the two sums run over the **same** events,
so the covariance is kept — this is `pol_analysis.ratio`'s error, not the naive
one, and here it is 3–8× smaller):

| quantity | `M(e+ mu+)` — symmetric | `Δφ(e+e-)` — asymmetric |
|---|---|---|
| `χ²` of `Z_TZ_0/Z_0Z_T = 1`, **LO** | **7.6 / 7** | **5328.8 / 12** |
| `χ²` of `Z_TZ_0/Z_0Z_T = 1`, **NLO** | **3.0 / 7** | **265.7 / 12** |
| `χ²` of `K(Z_TZ_0)/K(Z_0Z_T) = 1` | **3.7 / 7** | **329.5 / 12** |
| worst bin, `K(Z_TZ_0)/K(Z_0Z_T)` | 0.945 ± 0.055 (1.0σ) | 2.001 ± 0.130 (**7.7σ**) |
| range of `Z_TZ_0/Z_0Z_T` across bins, LO | 0.991 … 1.028 | **0.348 … 1.089** |

**Agreement on the symmetric observable, disagreement on the asymmetric one.**
That is precisely the signature the brief specified for real physics. Had the
weight-index-to-`Z` mapping been transposed, or had anything else in the
analysis broken the two components apart, `M(e+ mu+)` would have broken too.
It does not: seven bins, all within 1.7σ, best constant `0.9921 ± 0.0056`.

Note that `M(e+ mu+)` holds even in its **own** large-K bin: at
260–450 GeV `K` itself reaches 2.2–2.3, and the ratio there is still
0.945 ± 0.055. A large K-factor is not by itself a symmetry violation.

---

## The index mapping is the way round everyone assumed — measured, not read

Tested physically rather than by reading code, as asked. For `Z -> l+ l-` the
helicity-0 angular distribution is `sin²θ*` (`<cos²θ*> = 1/5`) and the sum over
the two transverse helicities is `1 + cos²θ*` (`<cos²θ*> = 2/5`).
`lt_tl_analysis.cos_star` rebuilds `cos θ*` of the `e+` in the `e+e-` rest
frame and of the `mu+` in the `mu+mu-` rest frame from the cached `pT`, `eta`
and `Δφ` / `m`, and weights it by each polarisation column.

`madspin` (NLO), 118 521 events in the four-lepton fiducial:

| column | electron side | muon side | reading |
|---|---|---|---|
| `Z_0Z_0` | 0.2564 ± 0.0015 | 0.2498 ± 0.0015 | long. electron, long. muon |
| `Z_0Z_T` | **0.2567 ± 0.0012** | **0.3219 ± 0.0014** | **long. electron, transv. muon** |
| `Z_TZ_0` | **0.3258 ± 0.0014** | **0.2503 ± 0.0012** | **transv. electron, long. muon** |
| `Z_TZ_T` | 0.3464 ± 0.0010 | 0.3438 ± 0.0010 | transv. electron, transv. muon |

The `LO` sample gives the same 2 × 2 to three decimals. **The first index is
the electron-`Z`.** The `Z_0Z_T` and `Z_TZ_0` rows are not swapped, and the
diagonal rows come out on the correct sides, which they would not if the
columns had been mislabelled in any permutation.

The absolute values sit between 0.20 and 0.40 rather than on them, and that is
expected: the fiducial cuts, the use of the lab-frame `Z` direction as the
helicity axis rather than the `ZZ`-frame one that `frame_id 24` quantises
along, and the `T` column summing both transverse helicities all dilute the
contrast. The test is the **contrast and the 2 × 2 pattern**, both of which are
unambiguous: the electron side moves 0.2567 -> 0.3258 between the two
columns and the muon side 0.3219 -> 0.2503, each a ~37σ swing on bars of
0.0012–0.0014 (and that treats the two as independent, which they are not —
they are reweightings of the same events, so the real separation is larger).

---

## The mirror test — the sharpest confirmation

`Δφ(mu+mu-)` is the `e <-> mu` image of `Δφ(e+e-)`. If the first index really
is the electron-`Z`, then `Z_TZ_0/Z_0Z_T` measured on `Δφ(mu+mu-)` must equal
**`1/(Z_TZ_0/Z_0Z_T)` measured on `Δφ(e+e-)`**, bin by bin. Nothing in the
weight columns enforces that, and the two sides are not even the same event set
— each keeps the two leptons its own observable is built from.

`Δφ(mu+mu-)` is not a cached column. It is reconstructed exactly from `m_mumu`
and the muon `pT`/`eta` via `m² = 2 pT₁ pT₂ (cosh Δη − cos Δφ)`. The same
identity run **forwards** on the electrons, where `Δφ` *is* cached, reproduces
`m_ee` to `1.8e-4` GeV (NLO) and `2.6e-4` GeV (LO) — float32 rounding, so the
inversion is trustworthy.

| sample | `χ²` of measured `Δφ(mu+mu-)` against the `Δφ(e+e-)` prediction | range predicted |
|---|---|---|
| `LO` | **17.1 / 12** | 0.918 … 2.870 |
| `madspin` (NLO) | **15.9 / 12** | 0.974 … 1.434 |

Every pull is within 2.1σ, over a predicted range spanning a factor of **3.1**
at LO. The muon side reproduces a factor-of-three swing predicted from the
electron side and from nothing else. That is not what a mislabelled column
does.

---

## Why the physics goes this way

For a boosted `Z -> l+l-`, `cos θ* ≈ 0` (which is where a **longitudinal** `Z`
puts its leptons) shares the energy evenly and collimates both leptons along
the boost — **small `Δφ`**. `|cos θ*| ≈ 1` (**transverse**) gives one hard
lepton along the boost and one soft one, which can land anywhere in azimuth —
**large `Δφ`**. So with the electron-`Z` longitudinal, `Δφ(e+e-)` is pushed to
small values; with it transverse, to large ones. The LO ratio runs
`0.348 -> 1.089` from the first bin to the last, monotonically, exactly that.

At NLO the same trend survives but is **compressed**, `0.697 -> 1.027`. Real
radiation gives the `ZZ` system transverse momentum, so the `Z` bosons are
boosted by production kinematics as well as by their own decay angle, and the
small-`Δφ` region is populated by events whose `Δφ` says little about the
polarisation. The polarisation discrimination is diluted.

`K(Z_TZ_0)/K(Z_0Z_T)` is the ratio of those two curves, and the whole of the
K-factor split is that compression:

```
K(Z_TZ_0)/K(Z_0Z_T)  =  (0.697 / 0.348)  =  2.00 +- 0.13   in the lowest bin
```

---

## Which bins can support a conclusion

The raw event count overstates what a polarisation column can support: the
column is a fraction of each event's weight and is far from uniform across the
bin. `lt_tl_analysis.py` prints the effective population
`N_eff = (Σw)² / Σw²` for both columns.

| `Δφ` bin | `N_LO` | `N_eff(Z_0Z_T)` | `N_eff(Z_TZ_0)` | `K(Z_TZ_0)/K(Z_0Z_T)` |
|---|---|---|---|---|
| 0.00–0.26 | 3 308 | 1 285 | **574** | 2.001 ± 0.130 (7.7σ) |
| 0.26–0.52 | 3 824 | 1 502 | **787** | 1.728 ± 0.099 (7.4σ) |
| 0.52–0.79 | 4 814 | 1 908 | 1 359 | 1.454 ± 0.065 (7.0σ) |
| … | | | | |
| 2.62–2.88 | 26 765 | 12 258 | 13 914 | 0.920 ± 0.011 (7.3σ) |
| 2.88–3.14 | 28 283 | 11 948 | 15 378 | 0.943 ± 0.011 (5.3σ) |

**The two lowest bins are thin** — `N_eff(Z_TZ_0) = 574` and `787` — and the
previous pass was right that the LO denominator there is at the edge of `2 -> 2`
phase space and that a K-factor on it is not a reliable NLO prediction. Those
two bins should be quoted with that caveat.

**But the conclusion does not rest on them.** The effect is a smooth,
monotonic trend across all twelve bins with a crossing near `Δφ ≈ 1.7`, and it
is significant at both ends:

* dropping the two lowest bins: `χ²(K ratio = 1) = 105.4 / 10`
* the **upper half only** (`Δφ > 1.57`, 13 000–28 000 LO events per bin,
  `N_eff` 5 800–15 400): `χ² = 44.1 / 6`
* against its own best **constant** rather than against 1:
  `χ² = 299.7 / 11`, i.e. it is a **trend**, not an offset and not noise.

So: every bin can support the statement that the two components differ; the
two lowest bins cannot support a quantitative K-factor. On `M(e+ mu+)` all
seven bins support the null, including the thinnest.

---

## Reconciling "not interchangeable" with "degenerate to 1–2 %"

The two earlier statements are about **different observables**, and the second
one was over-generalised. Measured on the absolute `dσ/dx` the seventh
(distribution) pane draws, the largest bin-by-bin departure of `Z_TZ_0` from
`Z_0Z_T` is:

| observable | order | min | max | max &#124;dev&#124; |
|---|---|---|---|---|
| `M(e+ mu+)` | LO | 0.991 | 1.028 | **2.8 %** |
| `M(e+ mu+)` | NLO | 0.971 | 1.006 | **2.9 %** |
| `Δφ(e+e-)` | LO | 0.348 | 1.089 | **65.2 %** |
| `Δφ(e+e-)` | NLO | 0.697 | 1.027 | **30.3 %** |

**`RESULTS.md` is wrong on one point and it is worth correcting.** Its
seventh-pane section says the two "are equal to 1–2 % bin by bin — they must
be, `ZZ` is symmetric and the two differ only by which `Z` the projection is
taken on". That measurement is right for the **`M(e+ mu+)`** pane and wrong for
the **`Δφ(e+e-)`** one, where they differ by up to a factor 2.9. And the reason
given is wrong in general: `ZZ` symmetry forces the two components to agree
only on observables that **respect the exchange**, which `Δφ(e+e-)` does not.
The consequence for the figure is small but real — the two curves are *not*
coincident on the `Δφ` top pane, so the line-weight stepping introduced to make
two coincident curves both visible is unnecessary there (harmless, but the
stated justification does not apply). The `M(e+ mu+)` pane is unaffected.

The earlier "not interchangeable on `Δφ`" statement is correct and is the same
physics as this pass measured. The only thing that needed correcting was its
reading of the lowest bin as an isolated edge effect: it is the extreme point
of a trend that spans the full range.

`RESULTS.md` and the existing figures are **left untouched** by this pass; this
file records the correction.

---

## What this pass did not settle

* **Scale and PDF uncertainties are absent from every bar here**, exactly as
  in the K-factor section of `RESULTS.md`. The scale envelope on either
  sample alone is `+4.1 % −5.2 %`, which is larger than the `M(e+ mu+)` null's
  precision and does not cancel in these ratios in any controlled way. What
  *does* partially cancel is the `Z_TZ_0/Z_0Z_T` ratio **within** one sample,
  since the two columns are reweightings of the same events — but that was not
  measured here, and the nine scale columns are cached in both `.npz` if
  someone wants it.
* **The helicity axis used in the mapping test is the lab-frame `Z`
  direction, not the `ZZ` rest-frame axis `frame_id 24` quantises along.** The
  full four-momenta are recoverable from the cached columns (`m_epmup` fixes
  the last relative azimuth up to discrete signs, and `m_4l` resolves those),
  so a sharper test is possible. It was not needed: the 2 × 2 pattern is
  already unambiguous and the mirror test is independent of the axis choice
  entirely.
* **Whether the same split appears in the other three spinmode samples**
  (`onshell`, `PA`, `madspin_v1`). Those `.npz` carry the four `ms_pol_*`
  columns, so `lt_tl_analysis.symmetry_test` would run on them unchanged, but
  none of them has an LO partner, so only the within-sample `Z_TZ_0/Z_0Z_T`
  half of the test is available there.
* **Finer binning.** The seven `M(e+ mu+)` edges were chosen for a pass with
  312 surviving events and are kept for continuity; with 118 000 the null could
  be tested much more finely, and a null on seven wide bins is a weaker null
  than one on twenty narrow ones.

---

## Deliverables

| what | path |
|---|---|
| the script, and every number above | `MadSpin/validation/zz_pol_weights/lt_tl_analysis.py` |
| figure, MG7 paper style | `MadSpin/validation/zz_pol_weights/plots/lt_tl/lt_tl.pdf` / `.png` |
| figure, user style | `MadSpin/validation/zz_pol_weights/plots_userstyle/lt_tl/lt_tl.pdf` / `.png` |

Run with

```
python3 lt_tl_analysis.py --check-minus
```

`--check-minus` covers the MG7-style PDF (the user style renders without
usetex and cannot be bitten by the Type1 subsetting bug); it reports
`1/1 applicable PDFs carry /minus`.

**The figure**, four rows of two, read across:

1. `cos θ*` under `Z_0Z_T` and `Z_TZ_0`, electron side (left) and muon side
   (right) — the mapping, measured. The two curves swap between the panes.
2. `Z_TZ_0 / Z_0Z_T` per bin, LO and NLO, on `M(e+ mu+)` (left, flat) and
   `Δφ(e+e-)` (right, a factor 3 at LO).
3. `K(Z_TZ_0) / K(Z_0Z_T)` per bin, same two columns — the row the question was
   asked about.
4. The mirror test, `LO` (left) and `madspin` (right).

Colours, line styles and the `NLO` solid / `LO` dashed rule are
`plot_zz_pol.COLOR`, `.LS` and `.LS_ORDER`, i.e. the same objects the rest of
the study draws with, so `Z_0Z_T` is the same purple and `Z_TZ_0` the same green
here as on every other canvas. Nothing is written on the figure but axis
labels, tick labels, legends and the per-pane `χ²`.
