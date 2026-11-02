# Fresh native-t_c validation

This workflow runs eight fresh NLO+PS samples with the
`mcatnlo-delta-native-tc-v1` source and the native Born-scale correction.
The installed Pythia8 is used without modification. Historical native-IR
samples belong to a different source epoch and cannot be adopted here.

The default campaign directory is
`/export/slow1/rikkert/macnlops_native_tc_1M_20261101`.
Process keys are `EE2j DY H ttbar Wj EE3j EE4j Wbb`. Stages run in the fixed
order `export cards generate diag shower analyse`; name only the stages needed.

```bash
# Export and configure a fresh sample; generation is an explicit stage.
python docs/audits/native_tc_validation/run.py \
  --process EE2j DY --run run_1M_native_tc --events 1000000 --cores 32 \
  export cards

# Generate and match the configured source, retaining zero-weight rejected events.
python docs/audits/native_tc_validation/run.py \
  --process EE2j DY --run run_1M_native_tc --events 1000000 --cores 32 \
  generate diag shower analyse

# Adopt the already generated pilot without changing its hard-generation cards.
python docs/audits/native_tc_validation/run.py \
  --process EE2j DY --run run_pilot_native_tc --events 2000 \
  diag shower analyse
```

`--base`, `--events`, `--cores`, `--run`, and a single-process `--seed` are
configurable. Existing valid exports are reused. Existing collected events
skip generation. The `cards` stage refuses to overwrite cards for a collected
run. Never run `cards` while a process is generating or showering. The `shower`
stage changes the continuation/analysis configuration only; it leaves hard
generation, FKS and parameter cards unchanged.

The e+e− energy is 91.188 GeV, with bare lepton beams and all hard scales fixed
at that energy. EE3j/EE4j have a Durham E-scheme generation resolution
`ycut=0.01`. EE2j has no additional generation cut. Hadronic runs use 13 TeV,
hard PDF 331700, and the native H_T/2 scale. DY requires m(e+e−)>60 GeV. Wj uses
anti-kT R=0.5 with a 20 GeV generation cut; its analysis includes both the
inclusive sample and a separate leading accepted jet pT>40 GeV, |y|<4.5
fiducial sample. Wbb uses the massive-bottom 4FS model, mb=4.7 GeV,
PDF 334700, no incoming bottom and no generation or analysis cuts.

First-emission evolution uses the native matching profile, alpha_s(MZ)=0.1365.
Continuation uses alpha_s(MZ)=0.118 at order one. MPI, hadronization,
primordial kT, QED radiation and W/top/H decays are disabled. Physical flavour
species are four for Wbb and five otherwise; stock Pythia alphaSnfmax remains
five. These settings distinguish the massive hard 4FS model from Pythia's
stock trial-running convention. Actual native settings and warning tables are
recorded for each completed shower job. The continuation retains the prior
validation protocol's `pdfcode=1`, which selects the corresponding NLO LHAPDF
set for hadronic runs. Bare-lepton EE cards contain inactive `PDF:pSet=13`
with `PDF:lepton=off`.

`diag` reads the collected LHE once, preserves every original signed nominal
event, and adds diagnostic weights by atomic replacement of the canonical
compressed file. The original content hash is retained; a second event backup
is not created. It saves only histogram aggregates and class/rule counts.
The four comparison weights are the source, S-veto only, H-removal only and
combined matching. Two additional signed source class weights identify S/H
without relying on final-state multiplicity. All share the same physical
endpoint and shower continuation. The identity
`veto + removal = source + matched` is checked per event.

The first and full shower use the same compiled observable helper. The primary
observable is the Durham N→N+1 transition scale for EE Nj and the hard-system
transverse recoil for DY, H, ttbar and Wbb. Wj uses pT(W+leading jet).
Secondary distributions include masses, rapidities, object/jet momenta, Durham
transitions and the C parameter. Signed source normalization is retained:
survivors, histogram ranges and fiducial selections never rescale the sample.

The Fortran hook writes per-job HwU tables and nine aggregate counts. It
replaces these files at Pythia checkpoints and creates no event streams.
At 1M events, fifty native jobs supply paired block errors; smaller divisible
pilots use four jobs. First-stage block errors use contiguous source chunks.
Full-stage errors use independent native shower jobs with their corresponding
fixed source chunks. They do not include integration or stock-shower model
uncertainties. Inclusive weight closure is checked with a tolerance allowing
the native HwU print precision.

Results live in `results/KEY/RUN/`: `first.json`, `full.json`, compressed
histogram aggregates, `comparison.json`, `shape_summary.json`, `shapes.pdf`,
and `native_accounting.json`. Cards, exact commands, source/workflow hashes,
git revision/diff hash, seeds and installed Pythia library hashes are recorded
under `cards/`, `commands/`, and `validation/`. Canonical LHE files stay in
the generated process `Events/RUN/` directory. There is no automatic pruning;
remove redundant source shards only after a separate successful audit.

The initial EE2j/DY diagnostic epoch recorded a gzip container hash only.
Native showering decompresses and recompresses that file, so the original
resume guard could falsely flag an unchanged payload. The metadata-only
follow-up records the uncompressed diagnostic payload hash for new runs and
uses it for resumes; gzip container hashes remain encoding provenance.
Historical `first.json` files stay immutable. For these older completed runs,
`encoding_audit.py` must first bind the current payload digest and ordered
six-weight sequence digest to the exact historical summary, verify every
event's diagnostic identity, and reproduce all signed class counts and weight
sums. The resume guard accepts that bound audit or rejects with an explanation.
No pre-shower uncompressed hash was recorded in the initial epoch, so this
legacy adoption is not represented as a historical byte-sequence proof.

Reports/accounting can be regenerated without showering:

```bash
python docs/audits/native_tc_validation/report.py \
  --process EE2j DY --run run_pilot_native_tc
python docs/audits/native_tc_validation/audit.py \
  --process EE2j DY --run run_pilot_native_tc --events 2000
python docs/audits/native_tc_validation/closure.py \
  --process EE2j DY --run run_pilot_native_tc
python docs/audits/native_tc_validation/epochs.py \
  --process EE2j DY --run run_pilot_native_tc
```

`closure.py` independently compares every first-stage chunk total with its
corresponding full-shower job using half the spacing of each actual printed
HwU token, with a separate small accumulation allowance. Original/retained
signed S/H counters must also agree exactly. `epochs.py` imports frozen
historical numbers without modifying their source-epoch labels or errors;
independent samples are compared as independent estimates.

Native warning occurrences and native event accounting are separate results.
The known threshold `xPDF = 0` diagnostic returns no candidate for one dipole;
it is recorded explicitly even when every input event is accepted. Warnings
are never used to filter, retry or reseed events.

The campaign summary is refreshed without reading events:

```bash
python docs/audits/native_tc_validation/campaign_report.py
python docs/audits/native_tc_validation/final_profile_audit.py
python docs/audits/native_tc_validation/fiducial_compare.py
```

`campaign_report.py` keeps incomplete samples explicit until all individual
event, profile, native-accounting, payload and token-precision closure audits
pass. Conditional H/S transfer fluctuations and shower-induced shifts remain
reported measurements; they do not determine the structural audit gate.
`final_profile_audit.py` checks actual initialized first-emission support and
the last case-insensitive assignment of each full-shower card key, including
model masses, flavour species, couplings and hadronic LHAPDF choices.
`fiducial_compare.py` adds differential Wj inclusive/fiducial40 plots and
qualified historical comparisons using the existing histogram aggregates.

The Wj 1M run exposed a nonconvergent global-gluon quadrature. The matching-only
repair evaluates the dead-cone algebra stably, splits at exact native floor
boundaries and uses the cosine endpoint map for the massive-recoil case.
The analytic kernel and requested integration tolerance are unchanged.
Focused tests, high-precision references and the complete same-seed failed
shard validate the numerical repair before cohort recovery.

`resume_matching.py` reads the exact original nonzero
`SubProcesses/nevents_unweighted` order and all recorded normalization files.
Its default action writes a plan. Execution requires an explicitly reviewed
matcher SHA matching both the repository and the selected exported provider:

```bash
python docs/audits/native_tc_validation/resume_matching.py --process Wj
python docs/audits/native_tc_validation/resume_matching.py \
  --process Wj --execute --matcher-sha REVIEWED_SHA256
```

The recovery replays every original channel with the original individual seed
and retains all hard points, source weights and normalization. For this Wj
epoch that means all 104 channels / 1M events, including 73 channels that had
already completed, with base seed 4320501. Compact old summaries/logs and
deduplicated settings are archived before rebuilding. The old complete
endpoint sequences and interrupted prefixes are compared exactly outside the
intentional decision/reweight tags. Old bulky matched files are removed only
after the replacement cohort, native collection, counts, source hashes and
signed normalization verify. No source generation or success-based shard
selection occurs. The exact execution, hashes and checks live under
`BASE/recovery/Wj/run_1M_native_tc_global_gluon_quadrature/`.

Matching numerical epochs are bound to the actual exported matcher source,
object and provider executable hashes in the campaign evidence. A repository
snapshot taken later can contain the repair while a previously exported live
provider still uses the earlier numerical implementation. Wj recovery uses
the reviewed repair; the other live providers retain their frozen epochs.
Installed Pythia files and physical shower settings are unchanged.

The Wbb all-event audit retains its original strict failure report for 12
boosted inverse reconstructions differing from their source scales by at most
`6.7e-11` GeV. Independent native reconstruction reproduces every decision
scale and encloses every source scale within combined one-ULP input-momentum
controls (1,524 evaluations). The optional `--roundtrip-controls` argument of
`../native_tc_event_audit.py` accepts these specific cases only for the exact
uncompressed input SHA recorded in
`BASE/validation/Wbb_H_scale_roundoff/roundtrip_controls.json`. Its default
tolerance remains unchanged. Exact source-to-continuation cap checks and
deliberately wrong-cap mutation controls remain independent requirements.
Two Wj inverse reconstructions use the same input-bound protocol, with
independently tested one- and two-ULP envelopes respectively (206 native
evaluations). Their controls and original strict report are retained under
`BASE/validation/Wj_H_scale_roundoff/`; the default audit tolerance remains
unchanged for every other event.
