# Couplings, PDFs and shower settings in MC@NLO-Delta with MAcNLOPS

28 October 2026. This note collects which QCD inputs enter each stage of an
MC@NLO-Delta source matched with MAcNLOPS, which of them must agree, and
the option of running the one-step shower with the settings that generated
the Sudakov table. Numbers refer to the DY and ttbar checks in
`/export/slow1/rikkert/macnlops_transfer_check_20261027` (NNPDF4.0 NLO,
13 TeV, 10^5 events); see also `docs/macnlops_derivation.md` and
`docs/audits/delta_noemission/investigation_report.md`.

## Summary

* Four stages use QCD inputs: the NLO hard process with its MC
  counterterms, the MC@NLO-Delta source factor `Delta_src`, the MAcNLOPS
  one-step shower with its S veto, and the continuation shower.
* Three couplings meet in the matching: `alpha_s^ME` in the hard matrix
  elements and MC counterterms, `alpha_s^tab` baked into the Sudakov table,
  and `alpha_s^PS` of the one-step shower. The continuation has its own.
* Kernels, maps, phase-space boundaries, masses and `pT0` must agree at the
  first power of the coupling. The coupling values, their running and the
  PDF scale choices may differ at the next order, but they decide whether
  the MAcNLOPS transfer balances and how many negative weights remain.
* The archive behaves as a table generated with PYTHIA's default shower
  coupling (0.1365, one loop, no CMW), `pT0Ref = 0`, for every entry type
  (quark and gluon II, FF and FI). Running the one-step
  shower with these settings removes the coupling mismatch without
  regenerating anything. For incoming quarks it also improves the
  forward-table approximation with NNPDF4.0 (below). The continuation can
  then use any settings: only the first emission is matched.
* The S veto must convert the counterterm coupling to the shower's at each
  emission; it already does (`K_sh`), whichever shower coupling is chosen.

## Where the inputs enter

| Stage | What it computes | QCD inputs |
| --- | --- | --- |
| Hard process | B, V, R and the MC counterterms (`K`, dead zones, `D(mu)` starting scales) | `alpha_s^ME(mu_R)`, hard PDFs at `mu_F`, model masses, MC kernels and maps |
| Source `Delta_src` | no-emission probability between `D(mu)` start and the reconstructed stopping scale, per Born leg | table (`alpha_s^tab`, its running, `pT0`, masses, recoil and z bounds, forward PDF mode) times the PDF ratio of the hard PDF; optional backward correction |
| One-step shower | first emission, or its absence, of each S event | `alpha_s^PS`, running, CMW, `pT0Ref`, `pTmin`, shower PDF, recoil, masses, nfmax |
| S veto | `v = min(u,K)/(beta_c K_sh)`, `K_sh = K alpha_s^PS(pT_evol)/alpha_s^K` | `alpha_s^K` = `alpha_s^ME` at the real point's `mu_R`; `alpha_s^PS` recorded at the emission; `beta_c = sigma_S/sigma_B` |
| Continuation | the rest of the shower from the handoff | anything, within the handoff constraints below |

## Current values

| Input | Hard process | Table (archive) | One-step shower |
| --- | --- | --- | --- |
| alpha_s(MZ), order | PDF value 0.1180024, NLO | 0.1365, one loop | PDF `AlphaS_MZ`, `AlphaS_OrderQCD`+1: 0.118, two loops |
| CMW | - | off | off |
| ISR `pT0Ref` | - | 0 (forced by the generation wrapper) | 0 (forced) |
| `StandardModel:alphaSnfmax` | - | 6 (default) | 5 (forced) |
| Light / c / b masses | model (5FS: massless) | 0 / 0 / 4.8 GeV | 0 / 0 / 0 (5FS) |
| ISR recoil | - | global (dipoleRecoil off) | global |
| FSR recoil | - | PYTHIA 8.2 default, `globalRecoil = off` | `globalRecoil = on`, `nMaxGlobalBranch = 1` |
| PDF | NNPDF40_nlo_as_01180 at `mu_F` | none, forward mode 1; caller multiplies by the hard PDF ratio | same set, backward (`pdfMode 0`) at `pT` |
| Version | - | patched 8.225 | 8.318 |

Sources: the one-step profile is `configure()` in
`Template/NLO/MCatNLO/srcPythia8/MacNLOPS.cc` and `shower_parameters()` in
`madgraph/various/macnlops.py`, which takes the coupling from the PDF
metadata. `alpha_s^K` is `g**2/(4*pi)` after `set_alphaS` at the real
point (`driver_macnlops.f`). The table's generation settings are in the
delta_noemission report. Its quark II entries equal a fresh mode-1
forward kernel with one-loop 0.1365 to 0.2% above 5 GeV over the DY
sample (0.9% at 3 GeV); the wrapper's declared ISR value 0.130 does not
fit. 0.1365 is PYTHIA's
default ISR and FSR value, while 0.130 is its MPI default.

## Consistency conditions

### NLO accuracy

The MC counterterm must be the O(alpha_s) expansion of the shower's first
emission. Kernels, Jacobians, recoil maps, phase-space boundaries
(including the II bound `Z(t)`), heavy-quark masses, matrix-element
corrections and ISR `pT0` screening must therefore agree.
`docs/macnlops_derivation.md` ("A leading-order shower density is
sufficient for NLO matching") shows that the coupling value, its running,
CMW and PDF-scale choices only change the density at the next order. The
first-emission couplings `alpha_s^ME`, `alpha_s^tab` and `alpha_s^PS` can
therefore differ without losing NLO accuracy.

### Transfer balance

MAcNLOPS removes S weight `X` where the shower emits and adds H weight `Y`
from the source. With the v2 veto the two balance when

\[
 \beta_c\,\Delta_{\rm PS}\,K_{\rm sh}=\Delta_{\rm src}\,K,
\qquad K_{\rm sh}=K\,\frac{\alpha_s^{\rm PS}(p_{T,\rm evol})}{\alpha_s^{K}}.
\]

`K_sh` uses the coupling recorded by the one-step shower, so the coupling
conversion follows whatever shower coupling is chosen. `Delta_src` has to
equal the one-step shower's no-emission probability `Delta_PS` to all
orders in the logarithms that matter. Every difference between the table
and the one-step shower appears directly as a transfer imbalance:

* ttbar with the old table: anomalous gluon entries, X/Y = 3.6 (net -8%);
  with the per-colour-line PDFmode-1 table, X/Y = 1.00.
* DY with the new table: X/Y = 1.75. The source Delta is below the shower's.
  At common cuts of 10 and 20 GeV (two legs) the factors are:

| Cause | 10 GeV | 20 GeV |
| --- | --- | --- |
| Table coupling 0.1365 (one loop) vs one-step 0.118 (two loops) | 0.918 | 0.969 |
| Forward kernel x PDF ratio vs backward shower (z < Z(t) bound vs DGLAP to z = 1) | 0.767 | 0.860 |
| Grouped flavour PDF ratio of the caller (u+c, d+s+b), all flavours | 0.911 | 0.962 |

The PDF ratio also carries a PDF-set dependence. If the one-step shower
used a different PDF from the hard process, `K_sh` would need the ratio
of the shower's and the counterterm's PDF factors. It is omitted now
because both use the same set (and was found to be about 1 for DY).

### Continuation

The shower is a Markov chain. Once the first emission, or its absence
down to the cutoff, has been generated with the matched one-step
settings, the continuation may use any coupling, tune, `pT0`, MPI and
hadronization. The change of settings at the first emission scale is
beyond NLO. The handoff in `docs/macnlops_derivation.md` ("Continuing the
retained LHE endpoint") imposes the remaining constraints: start from the
recorded ceilings, never re-draw a first emission, and keep the infrared
cutoffs at or above the one-step values (0.5 GeV for ISR; the probed FSR
value), since an "exhausted" S event uses a sentinel scale below them.

## Aligning the one-step shower with the table

Regenerating a table is expensive. The alternative is to give the
one-step shower the archive's generation settings:

```
SpaceShower:alphaSvalue = 0.1365     TimeShower:alphaSvalue = 0.1365
SpaceShower:alphaSorder = 1          TimeShower:alphaSorder = 1
SpaceShower:alphaSuseCMW = off       TimeShower:alphaSuseCMW = off
SpaceShower:pT0Ref = 0
```

Today the coupling comes from the PDF metadata (`shower_parameters()`) and
would need a profile entry for the table-aligned choice. CMW, `pT0Ref`, ME
corrections, rapidity ordering, azimuthal asymmetries and ISR recoil
already agree. The remaining generation differences are accepted or cannot
be changed: the archive's `alphaSnfmax = 6` (the one-step shower forces 5;
this only matters above the top mass), its dummy-dipole FSR recoil (the
one-step shower needs global recoil for MC@NLO) and its 4.8 GeV b mass.

### Coupling of every entry type

Fresh entries from the generation API (`generateSingleSudakov` through
`vendor/SudGen/pythia_interface_for_sudgen.cc`, PYTHIA 8.318, PDF mode 1,
the generation masses and flags) compared with the raw PDFmode-1 archive,
before the per-colour-line roots. Archive divided by fresh, at 0.1365 and
at 0.130, both one-loop:

| Entries (points) | 0.1365 | 0.130 |
| --- | --- | --- |
| gluon II (8: m = 91 to 1000 GeV, q = 5 to 50 GeV) | 1.000-1.005 | 0.89-0.998 |
| quark II (2) | 1.002 | 0.98 |
| FF quark, gluon, top (15) | 0.98-1.04 | 0.84-0.996 |
| FI quark, gluon, top (15) | 0.99-1.013 | 0.89-0.996 |

Every type fits 0.1365 at one loop; 0.130 is excluded everywhere. The FSR
scatter is the archive's own generation noise. It used 10^4 trials per
FSR grid node (5x10^5 for ISR). Along q, at exact grid nodes, the residual
changes sign about every other node, with rms 1.0% (FF quark) and 2.0%
(FI gluon), close to the expected binomial noise. The dummy FSR API treats
id 6 as massless, so the top entries equal the quark entries; this tests
the coupling, not massive radiation. Scripts and results:
`/export/slow1/rikkert/macnlops_transfer_check_20261027/table_fingerprint`.

### Incoming legs

The table is PDF-free (forward mode 1). The caller supplies the PDF ratio,
and the product only reproduces a backward shower up to two effects: the
missing PDF evolution from Z(t) < z < 1 (the study's boundary term C), and
the difference between the PDF's own evolution and the shower's LO
kernels with its coupling. For light-quark DY legs (per leg, NNPDF4.0,
damped starts) the forward product divided by the one-step shower's
backward probability B is:

| q [GeV] | F R / B at 0.118 (2-loop) | F R e^C / B at 0.118 (2-loop) | F R / B at 0.1365 (1-loop) | F R e^C / B at 0.1365 (1-loop) |
| --- | --- | --- | --- | --- |
| 3 | 0.784 | 0.865 | 0.812 | 0.918 |
| 5 | 0.830 | 0.911 | 0.849 | 0.951 |
| 10 | 0.882 | 0.951 | 0.891 | 0.975 |
| 20 | 0.934 | 0.979 | 0.937 | 0.990 |

Here F is the forward kernel at the shower's coupling: the table itself
when the coupling is aligned. The larger one-loop coupling brings the LO
backward evolution closer to NNPDF4.0's NLO evolution, as LO PDF fits
prefer about 0.13. Alignment therefore removes the table's coupling
mismatch and halves the remaining PDF-evolution error. The boundary term
C must still be added, evaluated with the shower's coupling
(`DeltaBackwardISR`, below). With an LO PDF that evolves with exactly the
shower's coupling the identity is exact (3e-4 at the study's PDF13 point),
but that would give the one-step shower a PDF different from the hard
process.

### Final-state legs

FSR from tops (ttbar FI/FF dipoles) involves no PDFs, so alignment makes
those table entries consistent with the one-step coupling, within the 1-2%
node noise above.

## Backward correction for incoming quarks (prototype)

`#DeltaBackwardISR` (default off, `FKS_params.dat`) changes the factor of
an incoming quark connected to the other incoming parton to `F R exp(C)`.
F replaces the table by the analytic mode-1 forward kernel at
`#DeltaBackwardAlphaS` with PYTHIA's running of order
`#DeltaBackwardAlphaSOrder` (1 or 2, matching PYTHIA's `AlphaStrong` to
2e-9); R is the caller's grouped PDF ratio; C is the boundary term from
three ln(t) by two z Gauss nodes (`delta_backward_quark` in
`montecarlocounter.f`). In DY at 0.118 (second order) it brings X/Y from
1.75 to 1.41 (net -1.09% to -0.66%) without measurable cost. Gluons and
quarks connected to final-state partons keep the table.

With an aligned one-step shower, set `#DeltaBackwardAlphaS 0.1365d0` and
`#DeltaBackwardAlphaSOrder 1`. F then equals the table's quark entries
(within 0.2% above 5 GeV), and C is evaluated at the shower's coupling.
Both values must equal the one-step shower's; reading them from the same
profile as the shower is still to be done.

## Decisions

1. Choose the one-step coupling policy: tied to the PDF (now), aligned to
   the table, or regenerated tables at the PDF coupling. Alignment needs
   only shower settings (every table entry fits 0.1365 at one loop); the
   continuation is unaffected.
2. Make `alpha_s^PS` a single recorded profile value that the shower, the
   backward correction and the provenance all read.
3. Keep `alpha_s^K` as the hard coupling at the real point, and add the
   PDF factor to `K_sh` if the one-step shower ever uses a different PDF
   set.
4. Extend the backward correction to gluons, and decide how to treat the
   grouped flavour sums, which cost 4-9% at 10-20 GeV in DY.
