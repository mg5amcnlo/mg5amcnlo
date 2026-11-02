# Complete WZ prescription comparison plots

[Download the 15-page plot book](full_comparison.pdf) · [Shower spectra only](all_variants.pdf) · [Source scales](source_scales/source_scales.pdf)

All four available one-million-event samples are overlaid: original hard H scale, previous geometric mean, fresh geometric mean with backward ISR off, and fresh geometric mean with backward ISR on. Columns show the merged result, @0 and @1 separately.

The middle row uses fresh OFF as a common denominator. The bottom row separates the two controlled comparisons: historical geometric/hard scales, and fresh backward ISR on/off. Historical and fresh epochs have different charged-resonance QED steering; cross-epoch ratios therefore do not isolate one source change.

## Complete coverage

| PDF page | Observable | View | Image |
| ---: | --- | --- | --- |
| 1 | total rate | overview | [PNG](rates.png) |
| 2 | pT W+Z [GeV] | full | [PNG](pt_wz_full.png) |
| 3 | pT W+Z [GeV] | low | [PNG](pt_wz_low.png) |
| 4 | pT W+Z [GeV] | tail | [PNG](pt_wz_tail.png) |
| 5 | log10 pT W+Z/GeV | full | [PNG](log_pt_wz_full.png) |
| 6 | pT j1 [GeV] | full | [PNG](pt_j1_full.png) |
| 7 | pT j1 [GeV] | low | [PNG](pt_j1_low.png) |
| 8 | pT j1 [GeV] | tail | [PNG](pt_j1_tail.png) |
| 9 | log10 pT j1/GeV | full | [PNG](log_pt_j1_full.png) |
| 10 | sqrt d01 [GeV] | full | [PNG](djr01_full.png) |
| 11 | sqrt d01 [GeV] | low | [PNG](djr01_low.png) |
| 12 | sqrt d01 [GeV] | tail | [PNG](djr01_tail.png) |
| 13 | log10 sqrt d01/GeV | full | [PNG](log_djr01_full.png) |
| 14 | S-event source SCALUP | Full source support | [PNG](source_scales/scalup_s.png) |
| 15 | H-event source SCALUP | Full source support | [PNG](source_scales/scalup_h.png) |

Linear spectra retain their full 0–1000 GeV range. The low-scale views use native 10 GeV bins over 0–150 GeV. Tail views cover 400–1000 GeV with 50 GeV bins; all original tail bins remain visible in the full-range pages. Logarithmic spectra retain log10(x/GeV) from −3 to 3. Signed or cancelled bins use symlog axes, and empty component ranges are marked explicitly.

## Findings from the expanded range

| Merged tail, 400–1000 GeV | Fresh OFF (pb) | Fresh ON (pb) | ON/OFF change |
| --- | ---: | ---: | ---: |
| pT(WZ) | 0.223458 ± 0.004752 | 0.235208 ± 0.004841 | +5.26% |
| Leading-jet pT | 0.191743 ± 0.004344 | 0.207839 ± 0.004484 | +8.39% |
| sqrt(d01) | 0.246385 ± 0.004930 | 0.266334 ± 0.005070 | +8.10% |

These central tail changes were not shown by the earlier 0–400 GeV figures. The three observables are correlated; their tail shifts are not independent evidence. Conditional sampling errors, separate source integration uncertainty and unavailable cross-run covariance preclude a significance claim.

| Very soft pT(WZ), 0.001–0.1 GeV | Rate (pb) |
| --- | ---: |
| Original hard scale (historical) | 0.0086565 ± 0.0008357 |
| Geometric mean (historical) | 0.0075840 ± 0.0007699 |
| Geometric mean / ISR off (fresh) | 0.0027512 ± 0.0004711 |
| Geometric mean / ISR on (fresh) | 0.0028366 ± 0.0004663 |

This tiny absolute-rate region shows a relative historical/fresh difference. Fresh ON/OFF are close here. Different historical W-boson QED steering prevents assigning the epoch shift to the H scale or backward-ISR prescription alone.

## Normalization and uncertainty

All curves retain absolute cross sections per bin; no unit-area rescaling is applied. Log spectra exclude exactly zero and underflow values below 0.001 GeV. Leading-jet spectra omit events without an anti-kt R=0.4 jet with pT>20 GeV and |eta|<4.5; their integrals therefore do not equal the total cross section.

The saved HwU errors are conditional event/shower errors. Source integration-normalization errors are separate. Rebinning combines bin errors in quadrature without bin covariance. Ratio bands show denominator errors and bars numerator error divided by the denominator; these are not combined ratio confidence intervals. Ratios require a positive denominator above three sampling errors. The source-scale companion documents its fixed-N sampling-error estimator separately.

[Original numerical comparison](../comparison.md) · [All native histogram bins](../histogram_comparison.csv) · [Plot ratios](ratios.csv) · [Plot manifest](manifest.json) · [Source-scale bin table](source_scales/source_scales.csv)

## Reproduction

```sh
python3 fxfx_diagnostics/plot_wz_source_scales.py
python3 fxfx_diagnostics/plot_wz_variants.py
```

The source-scale script reads the retained compressed LHE files. The shower script checks and reads the four retained HwU files. If `pdfunite` is available, their two books are combined into `full_comparison.pdf`.
