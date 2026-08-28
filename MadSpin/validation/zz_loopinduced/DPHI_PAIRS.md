# The two same-`Z` `Delta phi`: what they show, and why they are a `pt` plot

`Delta phi(e+ e-)` and `Delta phi(mu+ mu-)` were added to the study on
2026-08-28. **No events were regenerated** — the two columns were harvested from
the LHE of the same durable 200 000-event samples, and every array that was
already in `data/histograms.npz` and in the per-event column files is
byte-identical afterwards.

Figures: [`plots/dphi_ee.pdf`](plots/dphi_ee.pdf),
[`plots/dphi_mumu.pdf`](plots/dphi_mumu.pdf) and the same two under
`plots_userstyle/`.

---

## 0. In one paragraph

These are **intra-pair** angles — both leptons out of the *same* `Z` — so unlike
`cos1cos2` they are **not** an inter-decay correlation. `Delta phi` between the
two leptons of one `Z` is tightly anti-correlated with that `Z`'s transverse
momentum (Pearson **−0.65** on the truth, **−0.71** on ranks): a boosted `Z`
collimates its decay products, a `Z` at rest gives them back to back at
`Delta phi = pi`. They are therefore a re-reading of `pt(e+ e-)` and inherit its
support problem exactly. **The `PA` shape deficit of
[PA_LOWPT_DIAGNOSIS.md](PA_LOWPT_DIAGNOSIS.md) reappears in both, at large
`Delta phi`, with the same size and the same resolution**: `chi2/ndf` 3.33 and
3.45 against the full truth, 1.19 and 1.34 against the reachable support
`m_4l >= 2 m_Z` — against `pt(e+ e-)`'s 3.32 → 1.59. The ranking inverts in the
same direction. This is the support mismatch appearing where the `pt`
correlation says it must, and it must **not** be reported as a new `PA` anomaly.
The one genuinely new thing the two observables add is that they are a
**single-`Z` polarisation** handle: `spinmode = none` fails them at
`chi2/ndf` 5.89 / 5.59 on the full truth and **4.72 / 4.03 on the matched
support**, where nothing about the support can explain it.

---

## 1. Definition

Same convention as the existing `dphi_epmum`, and deliberately so — the three
`Delta phi` of this study are folded by one function, `observables._dphi`, as

```
Delta phi = abs((phi_a - phi_b + pi) % (2 pi) - pi)      in [0, pi]
```

`_selftest_dphi` runs on import and pins the fold to `arccos(cos(phi_a - phi_b))`
and to invariance under a rotation about the beam. The fold matters more here
than it does for the cross-pair one: a wrong branch leaves a step at `+-pi`, and
both intra-pair distributions **peak** at `pi`, so the step would land exactly on
the interesting bin.

Binning is the `dphi_epmum` grid, 32 bins over `[0, pi]`, so the three are read
off the same axis.

## 2. They are a `pt(ll)` plot

On the truth, per event:

| | `corr(Delta phi, pt)` | rank correlation |
|---|---|---|
| `(e+ e-)` | **−0.6475** | −0.7109 |
| `(mu+ mu-)` | −0.6461 | −0.7087 |

and the mean `pt` of the pair falls monotonically across the `Delta phi` range,
from 129 GeV in the first eighth to 21 GeV in the last:

| `Delta phi(e+ e-)` | truth [fb] | `<pt(e+e-)>` [GeV] | `f_sub` |
|---|---|---|---|
| 0.000 – 0.785 | 54.22 | 129.4 | 0.26 % |
| 0.785 – 1.571 | 99.37 | 101.4 | 0.20 % |
| 1.571 – 2.094 | 113.18 | 69.8 | 0.42 % |
| 2.094 – 2.400 | 91.75 | 52.5 | 0.84 % |
| 2.400 – 2.600 | 73.64 | 42.7 | 1.77 % |
| 2.600 – 2.800 | 89.35 | 35.2 | 2.83 % |
| 2.800 – 2.900 | 53.12 | 29.9 | 3.83 % |
| 2.900 – 3.000 | 58.63 | 26.1 | 4.35 % |
| 3.000 – 3.050 | 32.56 | 23.4 | 5.11 % |
| 3.050 – 3.142 | 65.28 | 21.2 | **5.55 %** |

`f_sub` is the share of that truth bin lying below `m_4l = 2 m_Z` — the region
**no** spinmode can reach. It climbs monotonically with `Delta phi`, by a factor
of 21 from the first bin to the last. Put the other way round: **64.6 %** of the
truth's whole sub-threshold rate sits above `Delta phi = 2.8`, against **28.7 %**
of the full truth. The hole is more than twice as concentrated at large
`Delta phi` as the cross section is.

## 3. `PA` against both supports

The prediction is that `PA` — the one mode whose normalisation is right, so that
nothing cancels the hole — should be pulled low at large `Delta phi` and recover
on the reachable support. It is, bin by bin:

| `Delta phi(e+ e-)` | `PA` / truth (all) | `PA` / truth (`m_4l >= 2 m_Z`) |
|---|---|---|
| 0.000 – 0.785 | 1.0171 ± 0.0118 (+1.0 σ) | 1.0198 ± 0.0118 (−0.7 σ) |
| 0.785 – 1.571 | 1.0275 ± 0.0088 (+2.5 σ) | 1.0296 ± 0.0088 (+0.3 σ) |
| 1.571 – 2.094 | 1.0433 ± 0.0083 (+4.5 σ) | 1.0476 ± 0.0084 (+2.4 σ) |
| 2.094 – 2.400 | 1.0209 ± 0.0091 (+1.7 σ) | 1.0296 ± 0.0092 (+0.2 σ) |
| 2.400 – 2.600 | 1.0073 ± 0.0100 (+0.1 σ) | 1.0255 ± 0.0103 (−0.2 σ) |
| 2.600 – 2.800 | 0.9964 ± 0.0090 (−1.1 σ) | 1.0254 ± 0.0094 (−0.2 σ) |
| 2.800 – 2.900 | 0.9799 ± 0.0116 (−2.3 σ) | 1.0190 ± 0.0122 (−0.7 σ) |
| 2.900 – 3.000 | 0.9827 ± 0.0110 (−2.1 σ) | 1.0274 ± 0.0117 (−0.0 σ) |
| 3.000 – 3.050 | 0.9765 ± 0.0147 (−2.0 σ) | 1.0290 ± 0.0157 (+0.1 σ) |
| 3.050 – 3.142 | **0.9462 ± 0.0102 (−5.9 σ)** | **1.0019 ± 0.0109 (−2.3 σ)** |

(pulls are shape pulls, against each comparison's own flat line.) `(mu+ mu-)` is
the same table to within its errors; its last bin is 0.9441 ± 0.0101 (−6.1 σ) /
0.9995 ± 0.0108 (−2.6 σ).

The whole shape test, from `pa_lowpt_diagnosis.py`'s two-support table:

| observable | mode | full truth | truth `m_4l >= 2 m_Z` |
|---|---|---|---|
| `dphi_ee` | `PA` | rate 1.0059, **chi2/ndf 3.33** | rate 1.0274, **1.19** |
| | `madspin` | 1.0075, 0.91 | 1.0291, **2.19** |
| | `onshell` | 1.0519, 0.92 | 1.0744, **1.99** |
| | `none` | 1.0519, **5.89** | 1.0744, **4.72** |
| `dphi_mumu` | `PA` | 1.0059, **3.45** | 1.0274, **1.34** |
| | `madspin` | 1.0075, 1.15 | 1.0291, **1.95** |
| | `onshell` | 1.0519, 1.02 | 1.0744, **2.19** |
| | `none` | 1.0519, **5.59** | 1.0744, **4.03** |
| `pt_ee` (for comparison) | `PA` | 1.0057, 3.32 | 1.0272, 1.59 |
| | `madspin` | 1.0073, 1.23 | 1.0289, 2.15 |
| | `onshell` | 1.0517, 1.08 | 1.0742, 2.41 |

**The `dphi_ee` / `dphi_mumu` rows are the `pt_ee` row.** `PA` is worst of the
three spin-correlated modes on the full truth and best of them on the support
the two samples actually share; `onshell`'s flatness on the full truth is the
same cancellation of its `1/f^2` normalisation error against the same hole that
PA_LOWPT_DIAGNOSIS.md documents for `pt`. The rates track it too: `+0.60 %`
over the full truth is `+2.74 %` over the reachable one, exactly the split of
section 3 there.

The same conclusion off a single number, `<Delta phi>`, which is
normalisation-free:

| sample | `<Delta phi(e+e-)>` | vs full truth | vs reachable truth |
|---|---|---|---|
| truth, whole window | 2.20399 ± 0.00176 | — | — |
| truth, `m_4l >= 2 m_Z` | 2.19145 ± 0.00178 | — | — |
| `PA` | 2.18892 ± 0.00176 | **−6.1 σ** | **−1.0 σ** |
| `madspin` | 2.20772 ± 0.00175 | +1.5 σ | +6.5 σ |
| `onshell` | 2.20262 ± 0.00176 | −0.6 σ | +4.5 σ |
| `none` | 2.17654 ± 0.00179 | **−10.9 σ** | **−5.9 σ** |

The sub-threshold events raise the truth's own `<Delta phi>` by 0.0125 rad, which
is 7 sigma of the mean — the support decides the comparison here, not the mode.

## 4. What is genuinely new: `none` fails these

Everything above is the `pt` story re-read. The part that is not is the `none`
row: `chi2/ndf` **5.89 / 5.59** on the full truth and **4.72 / 4.03 on the
matched support**, and `−5.9 / −6.5 sigma` on `<Delta phi>` against the reachable
truth. Nothing about the support explains that. An intra-pair `Delta phi` is a
**single-`Z` polarisation** observable — the same family as `cos_theta1`, not the
same family as `cos1cos2` — and `none` is the mode that has no polarisation.

The comparison that isolates it is `none` against `onshell`, not against the
truth. Both draw no virtuality and neither reshuffles, so their `Z` momenta are
the *same numbers*: `max |pt(e+e-)_onshell − pt(e+e-)_none| = 8.7e-06 GeV`, the
LHE write precision. They differ in the decay orientation and in nothing else,
and `onshell` is `chi2/ndf` 0.92 / 1.02 where `none` is 5.89 / 5.59.

The direction is the one the kinematics predict, and it is worth stating because
it is the opposite of the naive guess. At fixed `Z` momentum, a pair emitted
*along* the `Z`'s flight axis (`cos^2 theta -> 1`, the configuration a
**transverse** `Z` favours through `1 + cos^2 theta`) stays close to back to
back in the lab as long as the `Z` is not strongly boosted — and it is not here:
the median `pt(e+ e-)` is 43.8 GeV and **83 %** of the truth has
`pt(e+ e-) < m_Z`. A pair emitted *across* the axis (`cos^2 theta -> 0`, favoured
by a **longitudinal** `Z`) opens to intermediate `Delta phi`. Measured on the
truth, in bins of `cos^2 theta_1` whose mean `pt(e+ e-)` is flat at 56 GeV
throughout, so this is not the `pt` correlation of section 2 in disguise:

| `cos^2 theta_1` | `<Delta phi(e+e-)>` | fraction above 3.0 | `<pt(e+e-)>` [GeV] |
|---|---|---|---|
| 0.00 – 0.10 | 2.0802 | 0.121 | 56.8 |
| 0.10 – 0.25 | 2.1019 | 0.118 | 56.6 |
| 0.25 – 0.50 | 2.1515 | 0.118 | 56.9 |
| 0.50 – 0.75 | 2.2578 | 0.123 | 56.3 |
| 0.75 – 0.90 | 2.3877 | 0.146 | 56.2 |
| 0.90 – 1.00 | **2.6371** | **0.277** | 55.8 |

The truth's `Z` are strongly transverse — `f_0 = 0.0669 ± 0.0025`, against the
isotropic `1/3` — and `none`'s are unpolarised at `f_0 = 0.3313 ± 0.0024`. So
`none` under-populates the back-to-back region and over-populates the collimated
one, which is exactly what the figures show: relative to its own normalisation
`none` runs **+5 to +14 %** high over the first bins of `dphi_ee` (+5 to +22 % on
`dphi_mumu`) and **2 to 3 % low** at `Delta phi -> pi`, while `onshell` on the
same events stays within a couple of percent of flat.

This is a weaker handle than `cos_theta1`, which puts `none` at `chi2/ndf` 86.6,
and it measures the same physics. Its interest is that it is a **pure lab-frame**
observable: no rest-frame reconstruction, no boost composition, and therefore
none of the Wigner-rotation exposure that POLWEIGHT_CLOSURE_DIAGNOSIS.md is
about.

The cross-pair `dphi_epmum` puts `none` at 4.94 / 4.06, so the three
`Delta phi` are comparably sensitive to it and none of them approaches
`cos_theta1`.

## 5. Do `(e+ e-)` and `(mu+ mu-)` agree?

They must, by construction: the two `Z` are equivalent and only the positional
decay rule decides which pair the study calls first. They do. The right test is
the **paired** one — the two observables are measured on the *same* events, so
independent errors would be wrong — and the per-event difference gives

| sample | `<dphi_ee − dphi_mumu>` |
|---|---|
| truth | −0.00105 ± 0.00178 (−0.6 σ) |
| `PA` | −0.00098 ± 0.00179 (−0.5 σ) |
| `madspin` | +0.00300 ± 0.00177 (+1.7 σ) |
| `onshell` | −0.00205 ± 0.00177 (−1.2 σ) |
| `none` | +0.00037 ± 0.00172 (+0.2 σ) |

All five consistent with zero; the largest, `madspin`'s +1.7 σ, is one of five
tests and is not a finding. The shape `chi2/ndf` values also pair up
(3.33/3.45, 0.91/1.15, 0.92/1.02, 5.89/5.59) within the ~0.25 spread a
`chi2/ndf` at 31 degrees of freedom has. **No `e`/`mu` asymmetry.**

## 6. What to quote

* Quote **both** supports, as PA_LOWPT_DIAGNOSIS.md requires for every
  `m_4l`-correlated observable. `dphi_ee` and `dphi_mumu` are as
  `m_4l`-correlated as `pt_ee` is.
* Do **not** quote `PA`'s 3.33 / 3.45 as a defect of `PA`, and do not quote
  `onshell`'s 0.92 / 1.02 as evidence that `onshell` describes these
  distributions. Both statements are about the support, in opposite directions.
* The `none` numbers **are** quotable on the matched support, and they are the
  reason to have these two observables at all rather than only `pt_ee`.

## 7. Reproducing

```
export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
python3 run_zz_loopinduced.py --stage harvest \
    --basedir ~/Documents/madspin_validation_samples/t118_zz_loopinduced \
    --events-out ~/Documents/madspin_validation_samples/t118_zz_loopinduced/event_columns
python3 plot_zz_loopinduced.py --check-minus
python3 plot_zz_loopinduced_userstyle.py
python3 pa_lowpt_diagnosis.py --selftest     # the two-support table, all observables
```

The two-support table of section 3 comes out of `pa_lowpt_diagnosis.py`
unchanged — it already loops over `meta['observables']`, so adding an observable
to `observables.BINS` is enough for it to be measured on both supports. The
per-bin tables of sections 2, 3 and 5 are read off the per-event columns
directly.
