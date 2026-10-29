# Sampled-component MAcNLOPS campaign

Since 29 October a `macnlops = True` launch matches every channel right
after generation and collects the result as `Events/<run>/events.lhe.gz`;
`shower <run>` then runs the full shower (`docs/macnlops_derivation.md`,
first section). The runner described here predates that: it expects the
source from `launch` and the matching from `shower`, so it needs updating
before new samples are produced with it.

The production entry point is `docs/audits/run_macnlops_campaign.py`. The local
`production_1M_20261023/campaign.py` delegates to it. Its default output is the
fresh `production_1M_sampled/` directory: the completed global-sum production
and plots in the `20261023` directories are historical inputs, not checkpoints
for these new runs.

The six processes are ZZ, Drell–Yan, ttbar, W+bbbar (4FS), t-channel single top,
and W+jet. Each gets one million common MC@NLO source events for ordinary and
Delta matching. The source LHE is the MC@NLO reference; MAcNLOPS retains a subset
with unchanged signed weights. The production stops after the first emission
and H removal / S veto, writing the randomized combined
`Events/<run>/macnlops/events.lhe.gz` alongside the unchanged source
`Events/<run>/events.lhe.gz`.

Every source explicitly sets:

```text
#MCExplicitKLSum
.false.
#MCSubtractionAtFixedFlow
.true.
#FKSExplicitSum
.false.
```

Thus kl and ij are sampled, with MC subtraction evaluated at the selected
colour flow. The runner verifies that every event carries both a sampled
hard-event annotation and its component packet, and checks the collected veto
prescription: `component-source-v1` ordinarily, and
`component-undamped-delta-v1` for Delta. Fresh exports ensure the current
component registries and evaluator are present. Source fingerprints, per-sample
contracts, and LHE hashes guard against reusing older global-sum checkpoints.

The scheduler runs two complete process pipelines concurrently, each confined
to a disjoint set of up to six CPUs (twelve total on this machine). A pipeline
exports the process, then runs ordinary and Delta sources sequentially, since
they share mutable process cards. Its slot is reused only after it finishes.
OpenMP, BLAS and nested make parallelism are restricted; MG5 receives the slot's
core count. Fewer available CPUs reduce the allocation automatically when
preparing a new campaign. Failures are recorded and return a nonzero status;
completed samples are hash-checked on resume. Partial matching outputs require
inspection rather than being overwritten.

Prepare without exporting or generating events:

```bash
python docs/audits/run_macnlops_campaign.py prepare
python docs/audits/analyse_macnlops_campaign.py plan
```

The prepared configuration is `production_1M_sampled/config.json`; the analysis
plan, including all observable bins, is
`production_1M_sampled/validation/analysis_plan.json`.

Launch production when wanted:

```bash
python docs/audits/run_macnlops_campaign.py run
python docs/audits/run_macnlops_campaign.py status
```

Use `--campaign-dir <fresh-directory>` with both tools to choose another output
location. `prepare` also accepts `--events`, `--parallel-processes` (maximum 2)
and `--cores-per-process` (maximum 6). Existing configurations are immutable;
prepare a new directory to change the source code or campaign definition.

The explicit restart for the numerical matching corrections is:

```bash
python docs/audits/run_macnlops_campaign.py rematch
```

This archives the original configuration, checkpoints and card snapshots under
`history/matching_fixes_20261025/`. Each existing `macnlops` directory is renamed
`macnlops_before_matching_fixes_20261025`; corrected outputs use the usual
`macnlops/events.lhe.gz` path. Every channel is rematched, including channels
which previously succeeded. Saved first-emission proposals and their original
seeds are reused; incomplete proposal streams are regenerated with the corrected
stock-Pythia driver. Original source LHE files and generation manifests stay
unchanged. Separate native build receipts pin the new driver/counter objects
linked to the original generated amplitudes. The saved source mode is restored
explicitly when the process cards already describe the later Delta run.

Before collection, an order-independent hash of every original hard packet,
sampled component and signed weight checks that the proposals reproduce the
entire common source, including repeated events. New checkpoints are written
only after that check and validation of the collected counts and weight sum.
Existing source-generation provenance is retained separately from the new
matching implementation. The three missing Delta sources are then generated
with the corrected code. The same two slots of six cores bound rematching and
generation; `run` resumes this restart using its saved configuration.

## W+jet cuts and observables

The 5FS process is `p p > w+ j [QCD]`, with a stable W+. Generation sets
`jetalgo=-1` (anti-kt), `jetradius=0.5`, `ptj=10` GeV and `etaj=-1` (unrestricted).
Other generation cuts are removed. Single top also retains its previously
requested absence of generation cuts. Drell–Yan retains its 60 GeV same-flavour
dilepton mass threshold.

The W+jet analysis clusters final-state light quarks (including b) and gluons
using anti-kt R=0.5. It requires at least one jet with **pT > 25 GeV and |y| < 4.5**,
selecting the hardest jet satisfying both conditions. Rapidity is computed from
the jet four-momentum. The selection is applied independently to each analyzed
endpoint, including the uninterrupted reference shower, to retain migrations.
All other processes keep their existing R=0.4 analysis definitions.

W+jet plots include W and leading fiducial jet pT and rapidity; W+jet invariant
mass, recoil pT and rapidity; their azimuthal and rapidity separations; and the
second central jet pT and rapidity. Additional-jet pT is inclusive below 25 GeV
(zero if absent) to expose soft radiation; its rapidity plot requires that
second jet to exceed 25 GeV. Soft-region recoil and additional-jet plots use
uniform 0.5 GeV bins over 0–50 GeV. Top-pair soft plots keep uniform 1 GeV bins
over 0–150 GeV.

## Streamed validation

After production, register completed samples and run the analysis separately:

```bash
python docs/audits/analyse_macnlops_campaign.py prepare
python docs/audits/analyse_macnlops_campaign.py run
```

In a separate shell, update combined vector PDF books as datasets finish:

```bash
python docs/audits/make_macnlops_campaign_books.py --watch
```

The books are `validation/first/validation_summary.pdf` and
`validation/full/validation_summary.pdf`, with four selected observables per
ordinary/Delta dataset. Each level also has a `rates.csv` and
`available_summary.json` recording the completed datasets, page count, input
panel hashes and whether all twelve datasets are present. W+jet rates use its
analysis jet selection. The per-process galleries retain all observables and
the separate removal/veto controls; every ratio uses MAcNLOPS as denominator.

`run` builds the small C++ histogram consumer if necessary. The explicit
`build` action can prepare that binary beforehand. It links the stock Pythia8
library, fjcore, zlib and OpenSSL; no Pythia patches or HepMC output are used.
Two analysis jobs run at once using the campaign's CPU slots (the streaming
consumer itself is serial).

Both first-emission and full-shower plots compare MC@NLO, H removal only,
S veto only and combined MAcNLOPS. Full-shower plots also show the unrejected
first-emission sample continued from LHE. The reference uses an uninterrupted
MC@NLO shower with global recoil mode 2; the staged shower imports the written
continuation scales and uses local recoil. In particular W+jet sets
`TimeShower:nPartonsInBorn=1` for the reference. Both showers disable decays,
hadronisation, MPI, primordial kT, colour reconnection, QED and resonance
radiation. Beam remnants are retained for partonic colour/momentum closure.

Decisions are reused, not regenerated during validation. Histograms are filled
on the fly; no showered event files are written. All ratios use combined
MAcNLOPS as denominator. Signed sumw/sumw2 and paired covariances are retained,
including migrations into and out of the W+jet fiducial region. Pairing uses
the original hard event and its sampled component. Raw totals check the full
input population; separate `fiducial_totals` give selected cross sections.
Neither survivors nor plotted ranges are renormalized. Results are in
`production_1M_sampled/validation/{first,full}/<process>_<mode>/` as PDF, PNG,
CSV, and compressed NumPy histograms with covariance; raw streamed histogram
JSON lives in `validation/jobs/`.

Lightweight checks (no generation or showering):

```bash
python -m unittest tests.unit_tests.various.test_macnlops_campaign -v
```

These exercise the two-slot queue, all twelve ordinary/Delta card combinations,
legacy checkpoint rejection, sampled event packets, and the actual C++ W+jet
clustering, selection, signed histogram weights and migration covariance.
