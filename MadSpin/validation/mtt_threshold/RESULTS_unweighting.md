# The unweighting scheme and the tabulated `Z_k`

Companion of `RESULTS.md`. That study varies the **spinmode**; this one holds the
spinmode fixed and varies the **accept/reject scheme**, so what is left is the
`Z_k` question.

Everything here runs off `data/histograms_unweighting.npz` and
`data/meta_unweighting.json`, harvested from seven MadSpin runs that all decay
the same production sample. `data/histograms.npz` is the *other* study's file and
is not touched.

---

## 1. The question, and why it has a sharp answer

`Z_k(m)` is the tabulated look-ahead factor in the mass-stage weight — the
fraction of the decay pool that can reach virtuality `m`, fitted during the
max-weight scan. The three schemes treat it differently, and the differences are
structural rather than a matter of degree:

| scheme | mass stage | what happens to `Z_k` |
|---|---|---|
| `joint` | none | never built, never read |
| `sequential_global_retry` | yes, but a rejected decay discards the whole mass set and restarts | the per-angle stage stops normalising and `Z_k` **cancels identically** |
| `sequential` | yes | trusts the tabulated `Z_hat`; residual bias is exactly `Z_hat / Z` |
| `sequential_with_mass` (`PA` only) | mass drawn inside each slot's own accept/reject | no `Z_k` arises at all |

So the null hypothesis is sharp:

> **`joint` and `sequential_global_retry` must agree within statistical error**,
> neither of them depending on the table. A departure of `sequential` from both
> is the `Z_k` residual.

This is not a statement about how well the table is fitted. Neither scheme in the
null pair can be biased by the table *at all*, so a difference between them has
to come from somewhere else — and section 6 shows that in one cell it does.

## 2. The cells, and the fact that they are not all the same size

Seven cells: the two off-shell spinmodes crossed with the schemes each can
actually run. Every cell sets `unweighting` explicitly — `auto` is never used —
and the scheme is parsed back out of the log, so a silent fallback cannot
masquerade as a measurement. All seven ran the scheme they were asked for.

| cell | spinmode | scheme (asked = ran) | events | eff. | wall | `sum(w)/N` [pb] |
|---|---|---|---|---|---|---|
| `PA_joint` | `PA` | `joint` | 1 000 000 | 0.0864 | 3647 s | 674.4446 |
| `PA_seq` | `PA` | `sequential` | 1 000 000 | 0.3752 | 1525 s | 674.4446 |
| `PA_globalretry` | `PA` | `sequential_global_retry` | 1 000 000 | 0.3197 | 1845 s | 674.4446 |
| `PA_withmass` | `PA` | `sequential_with_mass` | 1 000 000 | 0.0543 | 3601 s | 674.4446 |
| `ms_joint` | `madspin` | `joint` | 1 000 000 | 0.1732 | 2674 s | **676.4383** |
| `ms_seq` | `madspin` | `sequential` | **500 000** | 0.2027 | 7050 s | 674.4471 |
| `ms_globalretry` | `madspin` | `sequential_global_retry` | **500 000** | 0.3178 | 11679 s | 674.4461 |

The banner cross section is 674.4446 pb in every cell — MadSpin normalises to
`sigma_production * BR`, so it is identical across schemes by construction. The
`sum(w)/N` column is not, and the one cell that departs from it is `ms_joint`;
that is section 6.

**The 1M/500k split is real and is handled explicitly.** The two `madspin`
sequential cells were the two slowest (7050 s and 11679 s against 1525–3647 s for
the rest) and were run at 500k. `prod_500000.lhe.gz` is a *front truncation* of
the same 1M production sample, so the smaller cells decay the first 500 000 of
the same events in the same order. Consequently:

* every spectrum is a **per-event** quantity (`density` divides by that cell's own
  `nevents`), so the split cannot leak into a normalisation — it survives only in
  the error bars, which is where it belongs;
* the **paired** comparisons `zip` the two files and therefore use the common
  500k prefix, so a 1M cell and a 500k cell are compared over exactly the same
  production events. The pairing is verified, not assumed: `max |Δ√ŝ| = 0` to the
  last bit in all seven pairings;
* the **unpaired** sensitivity uses `sqrt((1-f)/(f N_a) + (1-f)/(f N_b))`, not the
  symmetric `sqrt(2/(fN))`, so a 1M-vs-500k pair is correctly reported as worth
  barely more than 500k vs 500k.

## 3. Sensitivity — quoted before any comparison is read

The failure mode of this study is an underpowered null reported as agreement, so
the resolution comes first.

| window | `PA` pair (1M v 1M) unpaired / paired | `madspin` pair (1M v 500k) unpaired / paired |
|---|---|---|
| below `2m_t` | 3.351 % / 3.059 % | 4.653 % / 4.987 % |
| `2m_t+5` | 1.624 % / 1.040 % | 2.101 % / 1.527 % |
| `2m_t+10` | 1.064 % / 0.508 % | 1.345 % / 0.753 % |
| 330–370 | 0.573 % / 0.173 % | 0.713 % / 0.252 % |
| below 380 | 0.445 % / 0.109 % | 0.550 % / 0.156 % |
| full 290–520 | 0.126 % / 0.010 % | 0.155 % / 0.015 % |

The sub-threshold window holds ~0.165 % of the cross section. Even at 1M per cell
that is a **3.4 % resolution**, while the `Z_k` residual expected from the shipped
table is sub-per-cent. **The sub-threshold ratio therefore cannot support a claim
of agreement in either direction** — any agreement read off it alone is a
statement about the sample size, not about `Z_k`. That is why the wider windows
and the top virtuality exist.

Note the one place pairing does *not* help: below `2m_t` the two schemes agree on
only ~280 of ~1780 events (~96 of ~690 in the `madspin` row). The sub-threshold
population is almost entirely reshuffled independently between schemes, so the
discordant count is nearly the full count and the paired error is barely better
than the unpaired one (54.4 vs 59.3 events). Pairing buys a lot at 380 GeV
(0.109 % vs 0.445 %) and almost nothing below threshold.

## 4. The tabulated `Z_k`, as the runs logged it

The runs log the table and it parses back:

| cell | tables built | `Z` span | bin/fit deviation |
|---|---|---|---|
| `PA_joint` | **0 — none built** | — | — |
| `PA_withmass` | **0 — none built** | — | — |
| `PA_seq`, `PA_globalretry` | 2 | 0.912 → 1.059 (16 %) | ≤ 0.5 % |
| `ms_joint` | **0 — none built** | — | — |
| `ms_seq`, `ms_globalretry` | 2 | **0.525 → 1.705 (factor 3.2)** | ≤ 0.5 % |

That `joint` builds no table at all is the check that it cannot be biased by one
— it is a structural fact read out of the log, not an assumption.

This sharpens the prediction considerably. In the `madspin` row the table spans a
factor of 3.2 and is responsible for a large part of the lineshape; in the `PA`
row it spans 16 % and is responsible for very little. **A `Z_k` residual should
therefore show up in the `madspin` row and be nearly invisible in the `PA` row.**
`doc/madspin_sequential_plan.md` puts the lineshape sensitivity at 0.25 GeV per
unit fractional slope error with the table good to ~0.5 %, i.e. an expected shift
of **~0.001 GeV** — below what 1M events resolve. This was always going to be a
bound, not a detection.

## 5. `m(W b)`: the quantity `Z_k` distorts directly

`m_tt` only sees `Z_k` after the production reshuffle has smeared it. The
reconstructed top virtuality is what the mass stage acts on directly. Both tops
of every event enter, so `n = 2N`.

| cell | ⟨m_top⟩ [GeV] | ± | rms |
|---|---|---|---|
| `PA_joint` | 172.958803 | 0.002262 | 3.1996 |
| `PA_seq` | 172.960721 | 0.002265 | 3.2038 |
| `PA_globalretry` | 172.958179 | 0.002265 | 3.2030 |
| `PA_withmass` | 172.954496 | 0.002263 | 3.1998 |
| `ms_joint` | 173.188281 | 0.002310 | 3.2671 |
| `ms_seq` | 173.172570 | 0.003327 | 3.3268 |
| `ms_globalretry` | 173.171083 | 0.003323 | 3.3228 |

The `PA`/`madspin` offset (~0.21 GeV) is a **spinmode** difference and is not the
subject here; the comparisons below are always within a row.

### The paired differences

| pair | Δ⟨m_top⟩ [GeV] | significance | what it tests |
|---|---|---|---|
| `PA_joint` − `PA_globalretry` | **+0.000624 ± 0.003200** | 0.2 σ | **the null** |
| `ms_joint` − `ms_globalretry` | **+0.017031 ± 0.004655** | **3.7 σ** | **the null** |
| `PA_seq` − `PA_globalretry` | +0.002542 ± 0.003201 | 0.8 σ | the `Z_k` residual |
| `ms_seq` − `ms_globalretry` | +0.001486 ± 0.004683 | 0.3 σ | the `Z_k` residual |
| `PA_seq` − `PA_joint` | +0.001918 ± 0.003200 | 0.6 σ | `Z_hat` vs no mass stage |
| `PA_withmass` − `PA_joint` | −0.004307 ± 0.003196 | 1.3 σ | no `Z_k` at all |

The null holds in the `PA` row at 0.2 σ. In the `madspin` row it appears to fail
at 3.7 σ — but that cell is precisely the one carrying overweights, and section 6
shows the failure is theirs, not `Z_k`'s.

The same null in `m_tt`, with its sensitivity beside it:

| window | `PA_joint` / `PA_globalretry` | `ms_joint` / `ms_globalretry` |
|---|---|---|
| below `2m_t` | 1.02008 ± 0.03121 (0.6 σ) | 1.00145 ± 0.04994 (0.03 σ) |
| `2m_t+5` | 1.01252 ± 0.01053 (1.2 σ) | 1.00867 ± 0.01541 (0.6 σ) |
| `2m_t+10` | 1.00208 ± 0.00509 (0.4 σ) | 0.99792 ± 0.00751 (0.3 σ) |
| 330–370 | 1.00124 ± 0.00174 (0.7 σ) | 0.99724 ± 0.00251 (1.1 σ) |
| below 380 | 1.00004 ± 0.00109 (0.04 σ) | 1.00027 ± 0.00156 (0.2 σ) |
| full 290–520 | 1.00000 ± 0.00010 (0.04 σ) | 0.99987 ± 0.00015 (0.9 σ) |

In `m_tt` the null holds everywhere in both rows — the largest pull is 1.2 σ. The
`madspin` failure shows up only in ⟨m_top⟩, which is the more sensitive observable
and the one the overweights bias most directly.

## 6. Overweights, and why the `madspin` null failure is not a `Z_k` effect

The overweight safety net writes an event with a weight above 1 rather than
rejecting it when its weight exceeds the bound. Per cell:

| cell | carrying | of | largest factor | excess `sum(w)` | shift on σ |
|---|---|---|---|---|---|
| `PA_joint` | **0** | 1 000 000 | 1.0000 | 0 | 0.0000 % |
| `PA_seq` | **0** | 1 000 000 | 1.0000 | 0 | 0.0000 % |
| `PA_globalretry` | **0** | 1 000 000 | 1.0000 | 0 | 0.0000 % |
| `PA_withmass` | **0** | 1 000 000 | 1.0000 | 0 | 0.0000 % |
| `ms_joint` | **754** | 1 000 000 | **82.998** | 1.994e+06 | **+0.2960 %** |
| `ms_seq` | 4 | 500 000 | 2.209 | 1261.6 | +0.0004 % |
| `ms_globalretry` | 5 | 500 000 | 1.390 | 776.4 | +0.0002 % |

**`ms_joint` is the only cell whose normalisation is moved** (+0.296 %, visible
directly as `sum(w)/N` = 676.438 pb against a banner of 674.445 pb). Every `PA`
cell is exactly unit-weight; the two `madspin` sequential cells are unit-weight to
4 significant figures.

Because those 754 events are written once but represent ~83 events' worth of
cross section, the *unweighted* mean of section 5 is a biased estimator of the
physical mean for that cell alone. Recomputing both means on identical binned
support isolates the weights:

| cell | ⟨m_top⟩ unweighted | ⟨m_top⟩ weighted | difference | Kish `deff` |
|---|---|---|---|---|
| `PA_joint` | 172.958805 | 172.958805 | −0.000000 | 1.00000 |
| `PA_seq` | 172.960724 | 172.960724 | −0.000000 | 1.00000 |
| `PA_globalretry` | 172.958176 | 172.958176 | −0.000000 | 1.00000 |
| `PA_withmass` | 172.954496 | 172.954496 | +0.000000 | 1.00000 |
| `ms_joint` | 173.188286 | 173.163575 | **−0.024711** | **1.02814** |
| `ms_seq` | 173.172576 | 173.172538 | −0.000038 | 1.00000 |
| `ms_globalretry` | 173.171078 | 173.171065 | −0.000013 | 1.00000 |

The design effect `deff = N·Σw²/(Σw)²` is exactly 1 for every clean cell — an
independent readout, not going through the log line at all — and 1.028 for
`ms_joint`.

The overweight carry moves `ms_joint`'s ⟨m_top⟩ by **−0.0247 GeV**, which is
*larger than the +0.0170 GeV anomaly itself*. Redone on the weighted means, which
are the physical ones:

| pair (weighted, unpaired) | Δ⟨m_top⟩ [GeV] | significance |
|---|---|---|
| `ms_joint` − `ms_globalretry` | −0.007490 ± 0.004065 | 1.8 σ |
| `ms_seq` − `ms_globalretry` | +0.001473 ± 0.004702 | 0.3 σ |
| `PA_joint` − `PA_globalretry` | +0.000629 ± 0.003201 | 0.2 σ |

Correcting for the overweights takes the `madspin` null-pair discrepancy from
**3.7 σ to 1.8 σ**, and flips its sign — the overweights over-explain it. The
structural argument is stronger still: `joint` builds **no `Z_k` table**, so a
`joint`-vs-`global_retry` difference *cannot* be a `Z_k` effect whatever its size.
It is a rate artefact of the accept/reject machinery and must not be read as one.

Caveat, stated rather than hidden: the weighted comparison is **unpaired**, because
the stored per-event pairing carries no weights. A paired weighted difference is
not available from this harvest and would be the cleaner test.

## 7. Does `sequential` depart?

**No.** The `Z_k` residual test is `sequential` against `sequential_global_retry`
— both have a mass stage, both are essentially unit-weight (4 and 5 overweight
events), and they differ *only* in whether the tabulated `Z_k` is trusted or
cancels:

* `PA`: **+0.0025 ± 0.0032 GeV (0.8 σ)** — no residual.
  95 % upper limit |Δ⟨m_top⟩| < 0.0088 GeV (central + 1.96 σ).
* `madspin`: **+0.0015 ± 0.0047 GeV (0.3 σ)** — no residual.
  95 % upper limit |Δ⟨m_top⟩| < 0.0107 GeV.

Crucially the `madspin` row is where the table spans a factor of 3.2, so the row
in which a residual was *predicted* to be visible shows none.

In `m_tt` the same conclusion holds window by window. Over the two `Z_k` tests
(`seq` vs `global_retry`, both rows, both cells clean) the largest pull across all
seven windows is **1.82 σ** — `PA_seq`/`PA_globalretry` below 380 GeV, ratio
0.99804 ± 0.00108 — with everything else below 1.4 σ:

| window | `PA_seq` / `PA_globalretry` | `ms_seq` / `ms_globalretry` |
|---|---|---|
| below `2m_t` | 1.02008 ± 0.03109 (0.7 σ) | 1.06657 ± 0.05071 (1.3 σ) |
| `2m_t+5` | 1.00579 ± 0.01050 (0.6 σ) | 1.01524 ± 0.01541 (1.0 σ) |
| `2m_t+10` | 1.00404 ± 0.00512 (0.8 σ) | 0.99645 ± 0.00752 (0.5 σ) |
| 330–370 | 0.99976 ± 0.00174 (0.1 σ) | 1.00312 ± 0.00252 (1.2 σ) |
| below 380 | 0.99804 ± 0.00108 (1.8 σ) | 1.00111 ± 0.00157 (0.7 σ) |
| full 290–520 | 0.99997 ± 0.00010 (0.3 σ) | 0.99991 ± 0.00015 (0.6 σ) |

Note the sub-threshold entries: `1.02 ± 0.031` and `1.067 ± 0.051` are perfectly
consistent with 1, but at a 3.1 % and 5.1 % resolution against a sub-per-cent
expected effect they say almost nothing — exactly the underpowered null section 3
warns about. The `below 380` and `full` rows, at 0.11 % and 0.015 %, are what
carries the conclusion.

The one larger pull anywhere in the study is 2.33 σ, `ms_seq` vs `ms_joint` in
330–370 GeV (1.00589 ± 0.00253). It involves `ms_joint`, whose event-level
distribution is distorted by the 754 overweights of section 6 — window counts are
unweighted event counts and so are sensitive to exactly that distortion. It is
not a `Z_k` test and is not read as one.

**This is a bound, not a detection, and it should be reported as one.** The
expected residual from the shipped table is ~0.001 GeV; the measurement resolves
~0.003 GeV in the `PA` row and ~0.005 GeV in the `madspin` row. The measurement is
therefore consistent with the expectation but is a factor of a few short of being
able to *see* it. Saying "`sequential` agrees" would overstate the result; the
honest statement is "no residual is resolved at a sensitivity 3–5× coarser than
the effect predicted".

## 8. Figures

Two figures per style, one per spinmode row. The upper pane is the schemes over
the truth in **`1/σ dσ/dm_tt`** — each curve normalised by its own total cross
section over the full `m_tt` range, so the rate difference (including
`ms_joint`'s overweight carry) divides out. No in-plot prose, no `m_tt`
definition in the header or the axis caption, and no sample-size line: the cells
are *not* all the same size and the counts, with the sensitivity each one buys,
are in `numbers.txt` where they carry their errors. The upper pane's own error
bars are still the harvest's `sqrt(Σw²)/σ` and were deliberately left alone;
against the delta-method bar the ratio pane now uses they are 0.24 % too large,
which is invisible at this scale but is stated here rather than left implicit.

### The ratio pane divides by `joint`, and the truth is not in it

`joint` builds no `Z_k` table; `sequential_global_retry` builds one and cancels
it identically. **Those two agreeing is the null hypothesis**, so the pane puts
it on the line: `joint` *is* the flat reference at 1, drawn in its own colour and
dash, and every other scheme is drawn over it. `sequential` — the only scheme
that trusts the tabulated `Ẑ` — leaving the other two is the residual `Ẑ/Z`, and
it is now read directly instead of by subtracting two large common shape
differences from the truth by eye. The truth is deliberately absent from the
pane; it is still the black curve above, where the absolute lineshape is what is
being shown.

### What the pane draws is **per-curve statistics**, and what to quote is the paired error

The fixed ±5 % / ±10 % reference bands are gone. In their place:

* the **band around 1** is `joint`'s **own** statistical error, bin by bin;
* every **other curve carries its own** statistical error.

Both come from `UData.own_shape_err`. Because these are **shape** ratios — each
curve divided by its own total σ — a bin is a *subset* of the normalisation it
is divided by, and the correct within-sample bar is the **delta-method**
(linearised / jackknife) one this project already uses for exactly this case in
`zz_pol_weights/pol_analysis.ratio`:

```
R = N/D,  N = Σ nᵢ,  D = Σ dᵢ
dR = (dN − R dD)/D
var(R) = Σ (nᵢ − R dᵢ)² / D²
```

It returns **zero** when `nᵢ ∝ dᵢ`, which a plain `sqrt(Σw²)` does not; the
naive form is **0.23–0.24 % too large** on these bins (median over the pane,
printed in `numbers.txt`). For a unit-weight cell it reduces exactly to the
binomial `R²(1−n/N)/n` that the paired estimator uses term by term — the two
error treatments on this figure are the same statistics seen from two sides.
The weighted form is kept because `ms_joint` is *not* unit weight
(`deff = 1.028`) and its band has to know that.

> **Do not add the band and a bar in quadrature.** That is the wrong error on
> the difference between a scheme and `joint`, and it is **too large**.

The reason is the pairing, which has not gone away and is still measured. The
cells are not independent of one another: the four `PA_*` cells all come off one
1M production sample, and `ms_seq` / `ms_globalretry` off a 500k front
truncation of the `ms_joint` sample; every one of them decays the same
production events in the same order (`max |Δ√ŝ| = 0` across a row, asserted at
harvest). So a curve and `joint` fluctuate **together**, and the error on their
difference is smaller than their two own errors combined. Quadrature discards
that correlation and reproduces the `(unpaired)` column of `numbers.txt`
exactly — inflating the error by a median factor **1.16** in the `PA` row and
**1.10** in `madspin`, which would shrink every per-bin χ² by 26 % and 18 % and
could hide a real departure.

**The paired numbers remain the right ones for any significance statement.**
`run_mtt_unweighting.py --stage paired-bins` re-reads the decayed LHE files (it
re-runs no MadSpin and generates nothing) and counts, per plot bin, how many
production events land in *that* bin under a cell and under its row's `joint`;
`UData.paired_ratio` turns those into the covariance the difference's error
needs. Per-window coincidences — which is all the harvest had stored — cannot do
this, because an event can be in one window under both schemes and in a
*different bin* under each. The stage asserts its own window sums against the
harvest's stored coincidences before the numbers are used. Every per-bin row of
`numbers.txt` now prints **all three** errors side by side —
`ratio ± own [paired] (unpaired)`, with `joint`'s own band in its own column —
so nothing that the figure stopped drawing has stopped being available.

Per curve, per pane:

| pane | curve | denominator | drawn on the figure | quoted for significance |
|---|---|---|---|---|
| `PA` | `sequential`, `sequential_global_retry`, `sequential_with_mass` | `PA_joint` | its own delta-method error | **fully paired**, 1 000 000 shared events, gain 0.98→0.59 in σ from the sub-threshold tail to the 5 GeV bins above 380 GeV |
| `PA` | `joint` | itself | exactly 1, no bar — its own error is the **band** | it *is* the denominator |
| `madspin` | `sequential`, `sequential_global_retry` | `ms_joint` | its own delta-method error | **partly paired**: the 500k cells are a front truncation of `ms_joint`'s 1M, so the coincidences exist over that prefix and the unshared half of `ms_joint` enters as an independent error. Gain 1.00→0.75 |
| `madspin` | `joint` | itself | the **band**, as above | it *is* the denominator |

No pair had to be left unpaired. The pairing is weaker in the `madspin` row for
a structural reason, not a missing measurement: half of the denominator is not
shared with either numerator.

One consequence of drawing `joint`'s own error as a band rather than folding it
into every bar: the band is very wide in the deep sub-threshold bins, where
`joint` holds a handful of events — ±100 % in the leftmost `madspin` bin
(316–326 GeV, `n = 1`), ±33 % in the same `PA` bin (`n = 9`), ±18 % at
326–331 GeV — so it fills the pane there. That is what the statistics of those
bins are, and the previous fixed bands said nothing about them.

Errors are computed from **counts**, values from **weights**. For the four `PA`
cells the two are identical to the last digit (zero overweights,
`deff = 1.00000`), so the pane there is exact. In the `madspin` row `ms_joint`
carries +0.296 % of its σ in 754 overweight events, and the pane's weighted
ratio sits **+0.00295 above** the counted one in most bins — a normalisation
offset of the whole pane, which moves both curves together and leaves the
`sequential` vs `sequential_global_retry` difference untouched. It is *not*
uniform: in the few bins where an overweight event actually lands (374–378 GeV,
350–351 GeV) the offset falls to −0.0002, i.e. those bins carry an extra
weight-driven shift of up to 0.32 % on top of it. That is a rate artefact of the
accept/reject machinery, not a `Z_k` effect, and it is well inside those bins'
paired error (1.9 % at 374–378 GeV) — but it is why the `madspin` pane's central
values and its error bars are not quite the same measurement, and section 6's
caveat about `ms_joint` applies to the pane too.

### Clipping

Still **clipped to 0.8–1.2**, with off-scale points drawn on the boundary as
triangles pointing the way they went. **The new bars change nothing here**: the
clip is applied to the point's central value, which is untouched by the change
of error treatment, so it is still **6 points in the `PA` pane and 5 in the
`madspin` pane** — the same bins as before, all of them below 342 GeV where the
bins hold between 1 and 161 events, the deep sub-threshold tail, ~0.02 % of σ.
From 342 GeV up, nothing leaves the pane in either row. What *did* change is
that the band around 1 now runs off the top and bottom of the pane in the
leftmost bins, where `joint`'s own error is larger than the clip; it is a band,
not a point, so it is simply cut by the frame and carries no arrow. Its value
in every bin is the `band` column of `numbers.txt`.
Every off-scale value is in `numbers.txt`, flagged with a `*`; the
clip is no longer annotated on the figure and there is no footnote under it.
No bin on this figure is a *structural* zero — every cell draws a virtuality and
reshuffles, so all of them can reach below `2m_t` — so an empty bin is drawn as
a gap and the open-circle marker the companion figure uses for `onshell` never
appears here.

### What the pane shows

The numbers below are **paired** and come from `numbers.txt`, not from the bars
on the figure. The figure's bars are per-curve statistics and cannot give a
significance; reading one off them by eye — band and bar overlapping or not — is
the uncorrelated, too-large answer.

**Does the visual reading of `sequential` against `sequential_global_retry`
change?** Not in its verdict, and only mildly in appearance. The two curves'
bars were nearly identical to each other before (paired, 0.0177 and 0.0177
median in `PA`) and are nearly identical to each other now (own, 0.0146 and
0.0146); both shrank by the same factor ≈ 0.82 in `PA` and ≈ 0.90 in `madspin`.
So the *relative* picture — two curves, comparable bars, tracking each other
across the pane with `sequential` sitting a fraction of a bar low above
350 GeV — is the same picture, drawn with uniformly shorter bars. What is new is
the band, and a reader who folds band and bar together lands on a *larger* error
than the old bars showed. Both readings are wrong as significances; the paired
window numbers below are the measurement and they have not moved by a digit.

In `PA`: `sequential_global_retry`/`joint` = **0.99996 ± 0.00109** below 380 GeV
(0.04 σ — the null holding), while `sequential`/`joint` = **0.99800 ± 0.00108**
(1.85 σ). The two curves are separated by **0.99804 ± 0.00108, 1.82 σ**, the
largest pull in the study. In `madspin`, with half the statistics, both sit on
1: retry/joint = 0.99973 ± 0.00156 (0.17 σ), seq/joint = 1.00084 ± 0.00157
(0.54 σ), seq/retry = 1.00111 ± 0.00157 (0.71 σ). A ~2 σ hint in one row and
nothing in the other is a hint, not a detection — see section 7.

A per-bin χ² against the flat reference is in `numbers.txt`. In the `PA` row it
is elevated for **all three** curves together (1.22, 1.45, 1.26), which is a
statement about the shared denominator `PA_joint`, not about any one scheme: the
three siblings agree with each other to 0.3 % in the bins where all three depart
from `joint` by 1.5 %. That is exactly why the reading above is a comparison of
two curves rather than of a curve with the line.

* `plots_unweighting/` — MG7 paper style (usetex), PDF + PNG
* `plots_unweighting_userstyle/` — user style, PDF + PNG

Both carry `numbers.txt` with everything above unclipped and per bin.

## 9. The differential picture: the `m(W b)` lineshape figure

Section 5 read the virtuality through its **mean**. This section reads it
through its **shape**, and the shape turns out to carry three to four times more
information about the same effect.

Figures: `plots_unweighting/mtop_unweighting_{PA,madspin}.pdf/.png` and the same
two in `plots_unweighting_userstyle/`. They are written **alongside** the `m_tt`
figures, which are unchanged. `numbers.txt` in both directories carries a
top-virtuality section appended after the `m_tt` one.

### What it is built from, and the three things it cannot do

No MadSpin was re-run and no LHE file was re-read: the virtuality histogram
(`mtop_bins`, `<cell>_mtop_sumw`, `<cell>_mtop_cnt`) was already in
`data/histograms_unweighting.npz`, harvested in the same pass as `m_tt`. That
is just as well, because the decayed LHE files
(`/tmp/mtt_unweighting_work/MS/mode_*/events_decayed.lhe.gz`) **no longer
exist** — the directories survive and are empty. Three limits follow:

* **`m(t)` and `m(t~)` cannot be separated.** `harvest_cell` fills `m(W+ b)` and
  `m(W- b~)` into the *same* array, deliberately. Every number here is the two
  tops together, `n = 2N`. A per-top split is a real check — the `m_tt` study
  found single-resonance shifts that reversed sign between the two, so only a
  same-sign move on both is evidence — and it is one this harvest cannot supply
  and the LHE files can no longer be re-read for.
* **There is no `truth` curve.** `truth_mtop_*` was never stored. The upper pane
  has only the scheme curves and nothing is drawn in its place.
* **`Σw²` was not stored per virtuality bin.** It is reconstructed from `Σw` and
  the count on the fine grid — exact for a bin with no overweight, and an upper
  bound (`Σ_j x_j² ≤ x²`) otherwise. The reconstruction is *scored* at run time
  against the `m_tt` histogram, where `Σw²` is stored: median 1.00000 for all
  seven cells, worst bin 1.166 and only in `ms_joint`, always upwards.

### The binning

A Breit-Wigner needs a different grid from a threshold. The plotted range is
**exactly** the `±15 Γ_t` cut window, 150.6275–195.3725 GeV (the harvest grid is
`m_t ± 16 Γ_t` and holds nothing outside the cut — checked per cell), and the
width grows outwards:

| \|m − m_t\| | bin width | |
|---|---|---|
| 0–2 Γ_t | Γ_t/5 = 0.29830 GeV | five bins across the FWHM |
| 2–5 Γ_t | Γ_t/3 = 0.49717 GeV | |
| 5–9 Γ_t | 2Γ_t/3 = 0.99433 GeV | |
| 9–15 Γ_t | Γ_t = 1.49150 GeV | |

62 bins, every edge on a harvested fine-bin edge (the harvest grid is exactly
`Γ_t/75`, so the rebinning is exact and is asserted). Per-bin precision stays
inside 0.19–2.7 % instead of the 0.16–4.8 % a uniform grid of the same bin count
gives, and it costs nothing: the Fisher error on a rigid shift of the peak is
0.00077 GeV on this grid and 0.00077 GeV on a 150-bin uniform `Γ_t/5` grid.
Tail bins carry no information about where the peak is.

### The pairing comes out differently here, and that is measured

On `m_tt` the paired error is 10–16 % smaller than band-and-bar in quadrature,
and section 8 says loudly that the quadrature reading is wrong there. On the
virtuality it is not: each scheme re-draws its own masses and the shared
production kinematics barely constrains `m(W b)`. The stored paired moments say
so directly — paired σ divided by unpaired quadrature at the same `n`:

| pair | paired | unpaired | ratio |
|---|---|---|---|
| `PA_joint` vs `PA_globalretry` | 0.0031998 | 0.0032013 | 0.99951 |
| `PA_seq` vs `PA_globalretry` | 0.0032007 | 0.0032034 | 0.99916 |
| `ms_seq` vs `ms_globalretry` | 0.0046831 | 0.0047019 | 0.99600 |

Pairing buys between 0.01 % and 0.4 % here. The per-bin coincidence counts were
never produced for the virtuality and cannot be now, but they would move these
errors by less than half a per cent — so the quadrature error *is* the paired
error on this observable. **That is a property of `m(W b)`, not a general
licence**: the same construction on `m_tt` is 10–16 % too large.

### The pane is clipped at ±10 %, not ±20 %

Same *criterion* as the `m_tt` figure — wide enough to hold what the schemes do
to each other, narrow enough that a per-cent difference is visible — applied to
a different observable. Nothing leaves ±12 % and the bars run 0.19–3.6 %, so a
±20 % pane would assert agreement at a scale twenty times coarser than the
sensitivity. The convention is unchanged: two points go off in the `madspin` row
and are drawn on the boundary as triangles, with their unclipped values starred
in `numbers.txt`. Both other deviations from the `m_tt` figure's geometry are
cosmetic and forced: the `m_t` label rides above the peak rather than at the
foot of its rule (the pole is mid-axis, where a lower-right legend is), and the
band caption sits at the bottom right rather than the top right (the `madspin`
curves reach the top of a ±10 % pane).

### Sensitivity, before any ratio is read

Per bin, the 1 σ resolution on the shape ratio to `joint` is 0.27 % (best),
1.6 % (median), 3.7 % (worst) in `PA`; 0.33 / 1.9 / 8.2 % in `madspin`.

Projected on a **rigid shift** of the lineshape — the same direction ⟨m(W b)⟩
projects on, so the two are directly comparable:

| pair | σ(δ), lineshape | σ, moment | gain |
|---|---|---|---|
| `PA_seq` / `PA_globalretry` | **0.00106 GeV** | 0.00320 GeV | 3.0× |
| `ms_seq` / `ms_globalretry` | **0.00150 GeV** | 0.00468 GeV | 3.1× |

⟨m(W b)⟩ of a Breit-Wigner is dominated by tails that carry no information about
where the peak sits, so the sample mean is a badly inefficient estimator of a
shift. The figure therefore **improves** the bound on record rather than
repeating it.

### The reading

| row | pair | lineshape χ²/ndf | rigid shift δ | 95 % UL |
|---|---|---|---|---|
| `PA` | `seq` / `globalretry` | 43.3/62 = 0.698 | **+0.00060 ± 0.00106** (0.6 σ) | \|δ\| < 0.00267 GeV |
| `madspin` | `seq` / `globalretry` | 43.7/62 = 0.705 | **−0.00118 ± 0.00150** (0.8 σ) | \|δ\| < 0.00412 GeV |

**A bound, not a detection.** No `Z_k` effect is seen in either row, the two
rows do not even agree in sign, and both are consistent with zero. Against the
~0.001 GeV residual the shipped table is expected to produce, σ(δ) ≈ 0.0011 GeV
is a *one-sigma* sensitivity to the expected effect: this excludes an effect a
few times larger than predicted and does not touch the prediction itself. It is
nonetheless the closest this study gets — the 95 % upper limits are 3.3× and
2.6× tighter than the ⟨m(W b)⟩ limits of section 5 (0.00882 and 0.01067 GeV).

The `madspin` row's ratio pane must be read with section 6 in hand: **both**
coloured curves leave 1 together there, which is a statement about the
denominator `ms_joint` — whose counted and weighted ⟨m_top⟩ differ by
−0.0247 GeV against < 4·10⁻⁵ for every other cell — and not about `Z_k`. The
`Z_k` statement in that row is `ms_seq` against `ms_globalretry`, in which
`ms_joint` does not appear at all.

### One caveat that is itself a measurement

Every error here treats an event's two tops as independent (`n = 2N`). They are
not: the `PA` row is a genuine null and its per-bin χ²/ndf comes out **below 1**
on every binning tried (0.56–0.88, over grids from 10 to 150 bins). The two
virtualities of an event compete for the same `√ŝ`, so they are anti-correlated
and a histogram of `2N` tops fluctuates less than `2N` independent draws would.

Which way this cuts matters: the quoted per-bin errors are **conservative**, so
the agreement looks *better* than it is. If the true error is `sqrt(0.7) = 0.84`
of the quoted one, every departure is 1.19× more significant than the bars
suggest and σ(δ) is 0.84× the values above. The same assumption is already in
the ⟨m(W b)⟩ moment errors of section 5 (`rms/sqrt(2N)`), so it is inherited,
not introduced. Settling it needs the per-event *pair* of virtualities, which
the harvest did not keep; a future harvest should store their per-event mean.

## 10. What this does not cover

* **A paired *weighted* comparison** — see the caveat in section 6.
* **The error on a *difference*, on the figure.** The ratio pane now draws
  per-curve statistics only. There is no drawn quantity from which a
  scheme-versus-`joint` significance can be read; that number lives in the
  `[paired]` column and the window table of `numbers.txt` and nowhere else.
* **`Σw²` over the full `m_tt` range.** The harvest kept it only inside the
  histogram's 290–520 GeV grid, which holds 55.7 % of σ. The delta-method
  denominator needs it over the whole file, so the missing part is reconstructed
  from the in-range design effect (`UData._sumw2_total`). It enters `var(R)`
  multiplied by `R²`, so a 3 % error on it — the entire spread of `deff` across
  these cells — moves a bar by 4·10⁻⁵ of itself, and six of the seven cells are
  unit weight (`deff = 1.00000`) where the reconstruction is exact. It is the
  only approximation on the figure's error bars.
* **The upper pane's error bars**, still `sqrt(Σw²)/σ` and 0.24 % too large
  against the delta-method form; left unchanged deliberately.
* **`onshell`** — no virtuality is drawn, `_spinmode_has_density` is False and
  `_unweighting_mode` forces `joint` whatever the card says. There is no scheme
  axis to scan.
* **`sequential_with_mass` in the `madspin` row** — not run; it is a `PA`-only
  scheme here.
* **Why `ms_joint` generates 754 overweights when `PA_joint` generates none.**
  Established as fact and quantified, not explained. It is a property of the
  `madspin` spinmode's weight distribution against the joint bound, and it
  deserves its own study.
* **The `Z_hat/Z` residual at the sensitivity the prediction demands.** Reaching
  ~0.001 GeV on ⟨m_top⟩ needs roughly an order of magnitude more events, or a
  paired weighted estimator.
* **Bin-to-bin covariance in the ratio pane.** `--stage paired-bins` harvests
  the *diagonal* — how many production events land in the same bin under two
  cells. An event that lands in *different* bins under the two schemes
  correlates those two bins, and that off-diagonal block was not counted. Each
  bin's error bar is therefore right, and so is the expectation of the per-bin
  χ², but the χ²'s spread is not `sqrt(2 ndf)` and it must not be read as a
  p-value. The window numbers, which are exact, carry every significance
  statement in this document.
* **A paired ratio pane with weights.** The coincidence counts are counts; a
  weighted paired estimator would need the per-event weights kept alongside the
  pairing, which this harvest does not store. It matters only in the `madspin`
  row, and only at the 0.3 % level described in section 8.
* **`m(t)` and `m(t~)` separately.** The harvest fills both tops into one
  histogram by design and the decayed LHE files it would take to split them are
  gone. The `m_tt` study found single-resonance shifts that reversed sign
  between the two resonances, so a same-sign move on both is the only real
  evidence and this study cannot produce that test. **This is the largest single
  gap in section 9.**
* **A `truth` curve for `m(W b)`.** `truth_mtop_*` was never harvested. The
  virtuality figure is scheme-versus-scheme only, which is what isolates `Z_k`
  but says nothing about whether MadSpin's lineshape matches the full off-shell
  matrix element.
* **A per-bin *paired* error on the virtuality figure.** The coincidence counts
  were never produced for `m(W b)` and cannot be now. It costs at most 0.4 %,
  measured off the stored paired moments (section 9), and is the one place in
  this document where the paired error is quoted from an argument rather than a
  count.
* **Whether an event's two tops are independent.** They are not — the `PA` null
  χ²/ndf is 0.56–0.88 — but by how much cannot be measured from what was
  stored. The per-bin and moment errors are conservative as a result.
