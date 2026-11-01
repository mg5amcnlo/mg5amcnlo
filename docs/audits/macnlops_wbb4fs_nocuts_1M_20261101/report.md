# W+bbbar, four-flavour, 1M events without cuts

The 1,000,000-event validation is complete with **no generation or analysis
acceptance cuts**. The signed H-removal/S-veto cancellation has a residual
at the first emission. In the 5–20 GeV recoil interval, subsequent showering
adds no statistically resolved ratio drift:
**-0.00154 ± 0.00754** (full minus first).
The native event accounting and scale checks pass. Native Pythia threshold
and trial-envelope diagnostics remain and are documented below.

## First emission and full shower

Recoil means **pT(W+b+bbar)** using the stable W+ and bare massive bottom
quarks. The interval 5–20 GeV is a diagnostic range, not an acceptance cut.
S is the signed net S veto, including its negative-S contribution; H is
the removed negative-H contribution. H/S = 1 would indicate cancellation.
All four distributions use original signed weights in absolute cross-section units.

| Stage | Source [pb] | Matched [pb] | S veto [pb] | H removal [pb] | H/S | Matched/source − 1 |
|---|---:|---:|---:|---:|---:|---:|
| First emission | 217.767945 | 216.918896 | 9.947201 | 9.098152 | 0.91464 ± 0.01218 | -0.3899% ± 0.0585% |
| Full shower | 202.876432 | 202.087891 | 9.074335 | 8.285794 | 0.91310 ± 0.01281 | -0.3887% ± 0.0598% |

The corresponding H−S differences are
**-0.849049 ± 0.127967 pb**
initially and **-0.788541 ± 0.122215 pb**
after showering. The range contains 486,511 initial and
453,439 final events; migration is allowed, and neither
sample is normalized to the surviving or displayed events.

Inclusive source and matched cross sections are
**475.431551 and 473.689680 pb**,
a change of **-0.366377% ± 0.029911%**.
They are exactly preserved by full continuation because every input event
is showered and there is no acceptance selection. Inclusive S and H
transfers are 17.158649 and 15.416778 pb.

Errors use the same 50 paired contiguous 20k-event source/shower chunks.
These are conditional sampling estimates; hard-integration, channel-beta,
PDF and native-envelope uncertainties are separate. Averaging over the
independent matching-decision draw gives first/full H/S =
0.91114/0.91967;
the residual is therefore also present in the expected-transfer control.
The initial population is 746,444 S (106,705 negative) and 253,556 H
(24,003 negative). Class weights and all 19 observables are retained in
[comparison.json](results/validation_1M/comparison.json).

## Physical setup

The process is `p p > w+ b b~ QCD=2 QED=1 [QCD]` in `loop_sm`, pp at 13 TeV,
with a stable W+ and mb = 4.7 GeV. Both proton and jet multiparticle
labels contain only g,u,d,s,c and antiparticles; the PDF is
`NNPDF40_nlo_as_01180_nf_4/0` (LHAPDF 334700). The actual PDF coupling is
alpha_s(91.1876 GeV) = 0.1126177040. Hard scales use H_T/2.
Every run-card generation cut is disabled, with empty custom and PDG-cut
lists. Jets use anti-kT R=0.5 with **no pT or rapidity threshold**.
Radiation jets exclude the two stable bottom quarks; all-parton jets
include them. Missing jets have zero pT, and missing coordinates do not
reject an event. No coordinate is missing in either 1M-event sample.

The source is the current `delta_native_ir` common-continuation-scale
construction. First emission retains the shipped Sudakov convention
alpha_s(MZ)=0.1365; continuation uses 0.118. Both are one-loop, without
CMW rescaling. Continuation uses local recoil. Both stages explicitly use
four incoming and gluon-splitting flavours and hard-model particle masses.
The stock `StandardModel:alphaSnfmax=5` is retained: this is a four-flavour
hard/PDF/species validation, rather than a modified strictly nf4-running
shower. The evolution cutoff is 0.5 GeV. MPI, hadronization, primordial kT,
QED showering and W/top/Higgs decays are off. Pythia source and installed
library are unchanged. Cards and measured native initialization settings
are retained in this directory.

## Required PDF-support correction

An initial deterministic matching failure exposed an in-grid negative
charm PDF near t=1.703 GeV. Native Pythia clips the Born/daughter PDF to
zero but may still generate a branch using its internal PDF floor. The
ordinary shower/kernel PDF ratio then divides by zero. This is separate
from the later native threshold diagnostics.

Default v4 now checks the canonical Born flavours, x values and emission
scale on **both incoming QCD legs** for both S and H. On nonpositive native
support it disables transfer and retains both original contributions;
nonfinite values remain fatal. This protected exactly **three S events**
and no H events in the production sample. All events, source weights,
source Delta factors and shower endpoints are retained. Positive-support
decisions and legacy v2/v3 definitions are unchanged. This is no generation
or analysis cut and is not a derivation of Pythia's PDF-floor branch law.
Tiny positive PDFs and support changes along a Sudakov path are not
certified by this endpoint guard.

The same failed 9,973-event shard was replayed with the original seed:
all source records/endpoints were unchanged, the initial 5,293 output events
were byte-identical, and the other 9,972 decisions were identical. Original
hard events were not regenerated. The pre-guard generation epoch and
post-guard matching epoch are preserved separately. All **119 relevant
regression tests** pass, including eight new PDF-support tests, with no
skips. The [implementation snapshot](implementation/matching_epoch.json),
patch and focused failure probes are retained.

## Accounting, scales and diagnostic recovery

All 1M first endpoints pass mass, flavour, probability and written-scale
checks. Native initialization and continuation were probed on **1,754**
stratified weighted-transfer contexts using the actual production card:
zero global, production or dipole-scale mismatches, missing FSR pairs,
above-cap emissions or photons. These tests verify scale interpretation;
they do not prove equality of the analytic and every native branch law,
or reconstruct the complete state of an uninterrupted shower.

All 50 full jobs have exactly 20,000 tried, selected and accepted events;
no source event is dropped or externally retried. Native statistics contain
**201 first-stage and 9,345 continuation ISR weight-above-unity warnings**,
and **eight continuation `pT2nearThreshold: xPDF = 0` diagnostics** across
four jobs. Stock code returns no candidate for the affected dipole and
continues the competition; no event abort occurs. The clean-native-error
statistics gate therefore fails even though event accounting passes.
Occurrence counts alone do not bound their distribution effect or native
envelope bias. See the separate original and replay warning audits.

The original analyzer incorrectly closed its per-event stream at periodic
PYAEND checkpoints. Histograms covered all 1M events, while the binary
streams retained only the first 5,001 rows/job and `fort.97` the last 4,999.
This was an output-lifecycle bug. Changing only `close(97)` to `flush(97)`
and replaying the identical ordered inputs, job cards and random seeds in
isolated directories gives all 20,000 rows/job. **Every original HwU file
is byte-identical**, all original retained event rows agree byte-for-byte,
and native warning/statistics tables are exactly equal. Original incomplete
outputs and their failure log remain archived; final results use the
complete replay. The original and replay analysis epochs are both pinned.

The final check compares every event's four weights/order to the initial
records and independently reconstructs **425,000 HwU bin/weight entries**
within their printed rounding. A separate scalar/covariance calculation
reproduces the first/full results and paired drift. Eight independent
analysis geometry/Fortran-ABI checks also pass.

## Outputs and reproduction

The [three-page PDF](plots/wbb_nocuts_validation.pdf) shows the four-weight
first/full recoil, radiation-jet, bottom-pair and hard-object distributions.
PNG previews are beside it. The compact record includes cards, scripts,
matching and scale audits, native statistics, replay proofs, regression
logs and numerical histograms. `snapshot.json` pins these files and the
external event/array records by SHA256.

Complete generated processes, event files, isolated builds, per-job streams
and original failed diagnostics remain at:
`/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101`.
The exact matching recovery is in `resume_matching.py`; the diagnostic-only
full replay is in `analysis/replay_full_streams.py`. Final postprocessing:

```bash
python analysis/analyse.py \
  --first results/run_1M_wbb4fs_nocuts_lhe.npz \
  --full-root results/full_1M_replay \
  --out results/validation_1M
```

The channel-averaged S/B correction and native shower approximations remain
possible sources of the residual. This validation finds no additional
scale-import error and does not uniquely attribute the first-emission
imbalance to one approximation.
