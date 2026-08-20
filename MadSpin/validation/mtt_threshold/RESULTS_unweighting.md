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

* `PA`: **+0.0025 ± 0.0032 GeV (0.8 σ)** — no residual. Bound |Δ⟨m_top⟩| < 0.0064 GeV at 2 σ.
* `madspin`: **+0.0015 ± 0.0047 GeV (0.3 σ)** — no residual. Bound |Δ⟨m_top⟩| < 0.0094 GeV at 2 σ.

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

Two figures per style, one per spinmode row, each with its schemes over the truth
and a ratio pane beneath. Both follow the current house style: **`1/σ dσ/dm_tt`**
(each curve normalised by its own total cross section over the full `m_tt` range,
so the rate difference — including `ms_joint`'s overweight carry — divides out and
the pane below is a pure shape ratio), ratio pane **clipped to 0.8–1.2** with
off-scale points drawn as boundary triangles, and no in-plot prose or `m_tt`
definition in either the header or the axis caption.

* `plots_unweighting/` — MG7 paper style (usetex), PDF + PNG
* `plots_unweighting_userstyle/` — user style, PDF + PNG

Both carry `numbers.txt` with everything above unclipped and per bin.

## 9. What this does not cover

* **A paired *weighted* comparison** — see the caveat in section 6.
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
