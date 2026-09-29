# Differential validation of resonance recoil

This repeats the fixed-order comparisons in [resonance_recoil.md](resonance_recoil.md)
with `req_acc_FO = 0.001` and a differential analysis. The integrated precision
target is 0.1%; it is not a per-bin precision requirement.

## Calculation and provenance

The comparison uses `u b > d b e+ ve QCD=0 [QCD]`, `loop_sm-no_b_mass`, and the
complex-mass scheme. The top, W and Z widths are finite. The 13 TeV beams,
nn23nlo PDFs, fixed 173 GeV renormalization and factorization scales, and
jet algorithm and thresholds match the earlier comparison. The flavour-jet
requirements have been corrected after finding unsubtracted collinear limits
that invalidated the earlier rates.

The baseline uses the production templates from `ccc098c3b`. Both local-recoil
runs use `6ed084212`. The cutoff variation changes
`(xicut, deltaO, xiScut, deltaS)` from `(0.5, 1, 0.5, 1)` to
`(0.1, 0.2, 0.2, 0.3)`. Each run has four workers. Independent random seeds are
420101 for the baseline, 420203 for local recoil, and 420307 for varied cutoffs.

Generated processes, logs and a manifest with source and card hashes are in
`/tmp/mg5_resonance_precision_finite_bafs7rd5`. The subdirectories `baseline`, `local`
and `cutcheck` contain the three calculations. The integration is currently
running; no high-precision result is claimed in this document yet. The first
attempt in `/tmp/mg5_resonance_precision_bafs7rd5` was stopped after exposing
the cut problem; its results must not be combined with the corrected runs.

## Why the cuts changed

The old requirement of two resolved jets including any nonzero bottom tag
still admits two extra initial-state collinear limits. They involve underlying
Born flavours outside the explicitly generated `u b` process:

| Real channel | Unresolved leg | Other resolved jets accepted by old cuts |
| --- | --- | --- |
| `g b > d b e+ ve u~` | d parallel to incoming g | b and u~ |
| `u g > d b e+ ve b~` | b parallel to incoming g | d and b~ |

A direct probe lowers the unresolved pT from 1 to 0.00316 GeV at fixed 1 TeV
partonic energy, with physical on-shell momenta conserving four-momentum.
In both channels the old cut passes, `Sij=1`, and `pT^2 |M|^2` approaches a
nonzero constant (approximately 7.161e-8 and 3.081e-8 in the probe's matrix-element
normalization). The real contribution therefore has an unsubtracted collinear
pole. Matrix-element limit checks for the selected FKS pairs do not detect it.

The corrected cuts require separate jets with positive net bottom and positive
net down flavour. The probe is retained as
[`check_resonance_validation_limits.f`](../tests/input_files/check_resonance_validation_limits.f).
It checks both rejected limits and corresponding resolved configurations.
The original diagnostic and output are in the stopped run's `corner_probe.f`
and `corner_probe.log`.

## Analysis

The analysis is
[`analysis_HwU_resonance_recoil.f`](../tests/input_files/analysis_HwU_resonance_recoil.f).
Copy it to the generated process's `FixedOrderAnalysis` directory and set

```
FO_ANALYSIS_FORMAT = HwU
FO_ANALYSE = analysis_HwU_resonance_recoil.o
```

in `Cards/FO_analyse_card.dat`. Use the existing
[`resonance_recoil_cuts.f`](../tests/input_files/resonance_recoil_cuts.f)
user cut: at least two anti-kt R=0.4 jets with pT above 20 GeV, including separate
positive-net-bottom and positive-net-down jets. This validation is specific to
the explicitly flavour-labelled process; it is not a generic bottom-tagging
analysis for grouped subprocesses.

The leading jet with positive net bottom flavour is combined with the positron
and neutrino to reconstruct the top. The recoil jet is the hardest separate
jet with positive net down flavour. Incoming partons and the zero-momentum padding
leg in Born/counterevent configurations are excluded from clustering.

| Observable | Bins | Range |
| --- | ---: | --- |
| Reconstructed top mass, broad | 40 | 0--400 GeV |
| Reconstructed top mass, peak | 50 | 150--200 GeV |
| Positron-neutrino invariant mass | 40 | 70--90 GeV |
| Leading bottom-jet pT | 20 | 20--220 GeV |
| Leading bottom-jet pseudorapidity | 20 | -5--5 |
| Recoil-jet pT | 20 | 20--220 GeV |
| Recoil-jet pseudorapidity | 24 | -6--6 |
| Positron pT | 25 | 0--200 GeV |
| Positron pseudorapidity | 20 | -5--5 |
| Reconstructed top pT | 30 | 0--300 GeV |
| Bottom-jet--positron delta R | 24 | 0--6 |
| Resolved jet multiplicity | 2 | 2 or 3 jets |

Every spectrum includes underflow and overflow in its first and last bins.
Values are cross sections per bin. Total NLO and Born rates are booked
separately. All spectra include the real, counterevent, virtual and Born
contributions. HwU combines correlated contributions from each integration
point before estimating statistical errors.

## Comparison and uncertainty treatment

[`compare_resonance_recoil.py`](../tests/input_files/compare_resonance_recoil.py)
reads the three final `MADatNLO.HwU` files and writes a multipage PDF, a PNG
overview, a CSV containing every bin, and JSON/Markdown summaries. For example:

```
python3 tests/input_files/compare_resonance_recoil.py \
  --baseline /path/to/baseline/MADatNLO.HwU \
  --local /path/to/local/MADatNLO.HwU \
  --cutcheck /path/to/cutcheck/MADatNLO.HwU \
  --output /path/to/comparison
```

For independent runs, a bin's pull is the difference divided by the quadrature
sum of its two Monte Carlo errors. The comparisons use absolute cross sections;
they do not normalize away a possible rate difference. The ratio plots show
the candidate's uncertainty as error bars and the baseline's uncertainty as a
separate band. Sparse bins remain in the numerical tables; ratios with a
baseline significance below three are omitted from the plots.

The summary includes Bonferroni-adjusted marginal Gaussian tests across all
bins and all three pairs of runs. This controls the multiple-comparison rate
without assuming that bins are independent. No diagonal chi-square p-value is
used, because NLO counterevents can correlate different bins. The Gaussian
approximation still needs adequate sampling, especially in sparse tails.

The sum of each spectrum is checked against the total rate. Small deviations
from exact closure can arise from HwU's iteration weighting when a sparse bin
is empty in some iterations. The allowed discrepancy is at most 1% of the
total Monte Carlo error, or the text-output rounding tolerance.

## Technical checks

The 60 existing focused regression tests pass. Three additional tests cover
the reconstruction and jet cuts in soft/collinear limits, signed counterevent
weights, the comparison's uncertainty propagation, and rejection of incomplete
histogram files. The reconstruction fixture also passes using external FastJet.

All three generated runs pass their matrix-element tests and all 20 virtual
pole tests at tolerance 1e-5. The current local-recoil templates also pass 331
Herwig7 and 358 Pythia8 nonzero subtraction-limit checks in the leptonic process.
These are subtraction checks; they do not run a parton shower.
