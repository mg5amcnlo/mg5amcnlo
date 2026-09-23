# Complete-history MC@NLO performance

The measured optimizations reduce subprocess CPU time by 33% for Drell–Yan
and 40% for stable top-pair production with fixed scales, and by 30% for
top-pair production with dynamic scales, while retaining the
complete native-history sum. These measurements compare the optimized branch
with commit `c2366e492`, the complete-history implementation immediately before
this work. They do **not** measure a speedup relative to upstream `origin/3.x`.

The benchmark artifacts are in
`/export/tmp/rikkert/mg5_perf_c2366e492`. The `baseline_*` outputs come from the
detached `baseline_code` checkout at `c2366e492`; `revised_*` outputs include the
optimizations and the real-helicity correction described below. The
`refined_ttbar` output additionally contains the certified helicity-mask
rescaling refinement and supplies the final dynamic-scale measurement. Earlier
`candidate_*` and `final_*` exports are intermediate results and are excluded
from the timing tables.

## Calculation retained

For a fixed outer FKS sector \(a=(ij)\), every required labelled native sector
\(b=(kl)\) contributes to

\[
  \widehat H_a=S_a\sum_b H_b,
  \qquad H_b=P_b\bigl(S_bR-M_b\bigr).
\]

Here \(M_b\) includes the native MC subtraction and its prescribed G
replacements. Each term retains its native Born projection, damping, cuts,
PDFs, coupling orders and physical counterevent-to-real measure ratio. The
outer integration weight is applied once, and the outer context owns the
event kinematics, colours and shower scales. No history is sampled or omitted
as a performance approximation.

The initial outer evaluation now retains S records only, because its H records
would be replaced by this sum. Inner evaluations compute H records only. They
avoid ordinary S counterterms and degenerate remnants; zero-P real terms and
zero G replacements return before unnecessary matrix-element work. The
ordinary matching configuration still uses the existing `bogus_probne_fun`
mode 2, with P rising from zero below 0.5 GeV to one above 10 GeV. Setting
`mcatnlo_delta=False` therefore does not justify assuming P=1 everywhere.

Only the full real amplitude can be shared between histories:

\[
 r_b=\xi_b^2(1-y_b)R(\pi_b p;\theta_b).
\]

The cache key includes evaluator identity, all labelled momenta and the full
model state θ. The driver scopes reuse to one physical real point; explicit
permutations π map its particle labels. Analytic soft/collinear limits retain
their native Born projections. Polarization restrictions retain the native
frame when a summed boost-invariant amplitude cannot be assumed.

The shared Born provider reuses matching amplitudes for correlations and
retains allocated result/model buffers. PDF reuse requires matching native
flavours, factorization state and beam data. Contribution storage grows
geometrically and group membership is packed, avoiding the former dense
quadratic allocation. Group lookup still scans earlier records; this change
does not implement a hash table or remove that search cost.

Equal Born momenta alone do not imply equal model state: the native MC term
calls `set_alphaS` on the real event, whereas its FKS counterterms call it on
the counterevent. Dynamic scales can consequently give different couplings
for the same Born projection. Full-amplitude caches must distinguish those
states. The lifetime and invalidation of learned zero-helicity masks is a
separate question from amplitude-cache validity.

The subsequent mask refinement certifies homogeneity from the actual UFO
couplings and HELAS diagrams for each squared-order component q:

\[
  B_{h,q}(\lambda g_s)=\lambda^{D_q}B_{h,q}(g_s),\qquad\lambda>0.
\]

Under a consistent nonzero strong-coupling rescaling, zero helicities remain
zero. The learned mask can therefore survive that change while the numerical
amplitudes are recomputed. The signature compares masses, widths and normalized
couplings exactly; unsupported expressions, nonhomogeneous components,
nonfinite values and changes from zero to nonzero use the conservative reset.
This is not an approximate amplitude-cache hit. The fresh refined ttbar export
certifies `homogeneous_G` in `Source/BornSupport/helicity_signature.json`.

## Real-helicity correction

Physical point comparisons exposed an existing `T_IDENT` shortcut that inferred
equal helicity amplitudes from equality at one training point. In
`g g > W+ d u~`, this produced a roughly 0.14% disagreement under a longitudinal
boost even with real-amplitude caching disabled. The optimized real evaluators
remove this inference and retain only zero-helicity filtering. The W+jet test
checks all 25 native histories per context at three points, boosted frames and
changed model states, against a full-helicity reference at relative tolerance
`1e-11`. This is a correctness change, so raw matrix-dispatch counts alone
overstate the performance gain; the tables also count actual helicity
amplitude evaluations.

## Reproducible settings and timing

All timed runs use `loop_sm-no_b_mass`, 13 TeV pp collisions, built-in `nn23nlo`
(NNPDF23nlo_as_0119_qed_mem0, LHE PDF ID 244800), ordinary PYTHIA8 matching
without showering, `mcatnlo_delta=False`, folding `(1,1,1)`, and `ickkw=0`.
Each requests 1000 events and 3% integration accuracy. The actual integration
uncertainties are smaller. Scale reweighting uses the nine combinations of
factors `(1,2,0.5)` for each coupling tag, giving 27 weights per event; PDF
reweighting and stored native reweight information are off.

| Context | Process and cuts | Scale setting | Seed |
| --- | --- | --- | --- |
| DY, fixed | `p p > e+ e- [QCD]`, dilepton mass above 60 GeV, no lepton pT/eta cut | μR=μF=91.188 GeV | 732101 |
| ttbar, fixed | `p p > t t~ [QCD]`, stable 173 GeV tops, no jet cut | μR=μF=173 GeV | 732201 |
| ttbar, dynamic | Same top-pair settings | Both fixed-scale flags false, `dynamical_scale_choice=-1` | 732211 |

For each pair the model and run cards are byte-identical. Frozen cards, source
hashes, integration results and per-channel logs live in the corresponding
`*_controlled_record` and `*_dynamic_record` directories. Source hashes cover
the generated real matrices, provider library and native runtime. Later named
runs can replace logs in live subprocess directories; the comparison scripts
use the frozen copies.

The machine is an Intel Core i7-8700K at 3.70 GHz with six physical cores,
32 GB RAM, Linux x86_64 and GNU Fortran 13.3.0. Baseline and revised jobs run
simultaneously on disjoint physical cores: `taskset -c 0,1,2` and
`taskset -c 3,4,5`, each with three integration workers. No other jobs from this
validation ran during the paired measurements. Executables were precompiled
and passed their matrix-element, MC and pole checks. Dynamic run cards were
recompiled before timing. Launches use `--nocompile`.

The primary metric is the sum of the existing subprocess CPU `Total` over
grid construction, envelope integration and event generation. Launch wall
time is reported separately. Neither includes a performance claim about
export or compilation. Both outputs carry the same lightweight counters in
their generated sources; this instrumentation is absent from production
templates. These are individual controlled pairs, not repeated timing trials.

## CPU and operation counts

The dynamic row uses the final certified strong-coupling mask refinement. The
fixed-scale rows precede that refinement; their model state does not vary, so
the new rescaling branch is inactive there. An earlier dynamic run with the
conservative full-state mask reset is retained separately for comparison.

| Context | Baseline CPU [s] | Revised CPU [s] | Speedup | CPU reduction | Precompiled launch wall [s], baseline → revised |
| --- | ---: | ---: | ---: | ---: | ---: |
| DY, fixed | 112.224 | 75.346 | 1.489× | 32.9% | 40.58 → 27.89 |
| ttbar, fixed | 137.721 | 82.087 | 1.678× | 40.4% | 54.34 → 33.31 |
| ttbar, dynamic, refined mask | 138.713 | 96.721 | 1.434× | 30.3% | 54.37 → 38.87 |

| Context | Grid CPU [s] | Envelope CPU [s] | Event CPU [s] |
| --- | ---: | ---: | ---: |
| DY, fixed | 35.450 → 23.508 | 74.720 → 50.060 | 2.054 → 1.778 |
| ttbar, fixed | 42.799 → 25.772 | 87.540 → 51.498 | 7.382 → 4.818 |
| ttbar, dynamic, refined mask | 42.564 → 29.783 | 88.238 → 60.940 | 7.910 → 5.998 |

Envelope-stage operation counts, baseline → revised:

| Operation | DY, fixed | ttbar, fixed | ttbar, dynamic, refined mask |
| --- | ---: | ---: | ---: |
| Full real matrix dispatches | 1,000,488 → 400,196 | 755,894 → 194,104 | 754,893 → 382,061 |
| Actual real helicity amplitudes | 8,004,480 → 3,202,720 | 8,130,552 → 3,939,408 | 8,122,624 → 8,444,480 |
| Degenerate-remnant calls | 831,759 → 339,682 | 570,016 → 211,312 | 565,267 → 207,963 |
| Retained contribution records | 3,206,215 → 2,490,436 | 2,440,845 → 1,984,207 | 2,428,848 → 1,976,417 |
| Native weight evaluations | 1,041,600 → 1,041,600 | 755,966 → 755,966 | 754,976 → 755,582 |
| Inner history rows / inverse attempts | 624,960 → 624,960 | 561,844 → 561,844 | 561,184 → 561,588 |
| Native Born requests | 6,240,885 → 5,090,368 | 4,328,719 → 3,465,026 | 4,315,406 → 3,455,868 |

The fixed-scale pairs have identical history, inverse-attempt and native-weight
counts in every integration stage. The reductions therefore come from doing
less redundant work for those histories. Dynamic scales reduce exact
real-amplitude cache reuse. The conservative full-state mask reset also caused
32 helicity amplitudes to be recomputed per revised dispatch: 12,223,136
envelope helicity amplitudes in total, versus 8,122,624 in the baseline. The
certified rescaling refinement reduces this to **8,444,480**, 30.9% below that
conservative implementation and 4.0% above the original baseline, while
retaining the `T_IDENT` correctness fix. The historical conservative pair took
139.560 → 99.870 CPU seconds (28.4% reduction). The final pair gives the 30.3%
reduction above. This distinguishes model-dependent amplitude-cache misses
from the avoidable cost of relearning zero-helicity masks.

## Physics and event checks

The signed integration values below retain the precision of `res_1.txt`.
The quoted difference uses
\(\lvert\sigma_1-\sigma_0\rvert/\sqrt{\delta\sigma_1^2+\delta\sigma_0^2}\)
as a scale for comparison; common random numbers mean the pair is correlated.

| Context | Baseline [pb] | Revised [pb] | Difference / combined uncertainty | Negative events / 1000, baseline → revised | Logged generation efficiency |
| --- | ---: | ---: | ---: | ---: | ---: |
| DY, fixed | 1900.13221 ± 5.2203 | 1900.13273 ± 5.2203 | 0.0000704 | 55 → 55 | 33.5122% → 33.5122% |
| ttbar, fixed | 750.662931 ± 4.6713 | 750.663026 ± 4.6713 | 0.0000144 | 196 → 196 | 17.1798% → 17.1798% |
| ttbar, dynamic, refined mask | 683.141049 ± 4.3945 | 683.516669 ± 4.3959 | 0.0604 | 209 → 211 | 16.3194% → 16.1027% |

The efficiency is the ratio of the summed logged `events generated, novi`
counts to logged generation attempts, not 1000 divided by that attempt count.
Every LHE file contains 1000 events and 27,000 finite scale weights. Momentum
conservation residuals are below `3e-15` relative to the summed particle
energies. In DY the abbreviated post-generation summary omits a channel that
received zero events in this small sample. This occurs for both versions;
`res_1.txt` and the LHE initialization retain the complete integral used here.

The event streams are not bit-identical. For fixed-scale DY/ttbar, the largest
relative change in a central event weight is respectively `4.6e-8`/`7.8e-8`;
the largest momentum-component change normalized by the largest particle
energy in that event (with a 1 GeV floor) is `6.6e-7`/`1.1e-6`.
Two ttbar events select a different light-quark flavour;
their colour, mother and status assignments agree. Scale-weight changes can
be larger in cancellations: the largest absolute scale-weight difference
divided by the central event weight is `5.1e-4`/`2.0e-3`. With dynamic scales,
adaptive integration and unweighting produce different event selections.
Indexed dynamic events therefore do not supply a pointwise equivalence test.
The 404 extra
envelope history rows reflect changed sampling, not dropped histories.

A compiled Fortran regression compares the full native evaluator, filtered to
H or S, with the specialized paths in 216 combinations of P, G replacements,
cuts, massive/massless radiation and QCD/QED/mixed orders at `1e-12`. It also
checks that poisoned S-remnant storage is untouched by H-only evaluations and
that zero-H paths avoid unnecessary real/counterterm work. Provider and real
tests separately interleave momenta, model states, providers and correlations;
the full-helicity W+jet comparison guards the real-frame reuse. The 1000-event
benchmark runs are integration and performance regressions, not a replacement
for high-statistics distribution comparisons.

The final source passes 70 focused regression tests (25 provider tests, seven
mask-refinement tests and 38 native-history/PDF/storage and related checks),
plus eight exact export IO comparisons. Additional fresh legacy HEFT and
`HC_NLO_X0_UFO-heft` exports certify homogeneous strong-coupling scaling and
pass standalone/library comparisons of amplitudes, correlations, helicities,
coupling orders and colour-flow weights, including coherent rescaling and
coupling activation checks.

The final refined ttbar code also completed a 100-event MC@NLO-Delta run with
seed 732202, fixed scales, stored native reweight information and 27 scale
weights per event. It obtained **751.785953 ± 3.9540 pb**, consistent with the
previous matching fixed-order check `753.8 ± 3.6 pb` (0.38 combined
uncertainties). Nine events have negative weights. All colour connections and
dipole-scale indices pass the audit; dipole scales range from 30.513002 to
428.74518 GeV, with unequal scales in 92 events. The relative momentum residual
is below `4.64e-16`. All 100 stored reweight blocks are present, containing
2502 contribution rows with valid native provider/context/history and outer
event-owner IDs. Every LHE scale weight is finite.

Pythia8.313, with both production-scale and separate-dipole-scale reading
enabled, tried, selected and accepted all 100 events without a shower error.
The supplied `HwU.o py8an_HwU_rates.o` analysis produced its central rate
histogram in `refined_ttbar/MCatNLO/RUN_PYTHIA8_1/MADatNLO.HwU`. Its event loop
stops before analyzing the last accepted event, so 99 events enter the
analysis. The HwU scale-variation columns were not populated correctly, and
the launcher did not copy the advertised histogram to the Events directory;
the earlier `candidate_ttbar` output has the same invalid variation columns.
This check establishes shower acceptance and finite LHE reweights, not
validated showered scale-variation histograms. The detailed final audits are
`refined_ttbar_delta_lhe_audit.json` and
`refined_ttbar_delta_shower_reweight_audit.json`.

An additional W+jet MC@NLO-Delta run with the corrected real evaluator produced
100 events, including 17 negative weights. Its signed integral was
approximately `17280 ± 140 pb`, compared with the previous matching fixed-order
check `16950 ± 110 pb` (1.85 combined uncertainties). Every event passed colour
conservation and colour-connected dipole-index checks; all 27 scale weights
were finite. Dipole scales ranged from 2.3513402 to 202.72549 GeV, with unequal
dipole scales in 84 events. The largest relative momentum residual was
`4.24e-15`. Pythia8 accepted all 100 attempted events without an error; its log
contains the usual end-of-file message and one shower weight-above-unity
warning. This fixed-scale check precedes the subsequent helicity-mask
rescaling refinement. The audit is `revised_wjet_delta_lhe_audit.json` and the
shower log is `revised_wjet_shower.log` in the artifact directory.

A separate fixed-order DY NLO check requested 1% accuracy and obtained
`1909 ± 8.8 pb`, with scale variation `+3.8%/-6.7%`. It used the earlier
optimization snapshot before the real-helicity correction, so it checks the
common Born/PDF/storage changes but is not an additional final-source
acceptance run. Its summary is
`candidate_dy/Events/optimized_fo/summary.txt`.

## Artifacts and replay

Within `/export/tmp/rikkert/mg5_perf_c2366e492`, the principal machine-readable
results are `revised_controlled_comparison.json` and
`refined_dynamic_certified_comparison.json`. They include all three stage profiles,
per-channel signed integrals, event audits and operation counts. Frozen inputs
and logs are in `baseline_dy_controlled_record`,
`revised_dy_controlled_record`, `baseline_ttbar_controlled_record`,
`revised_ttbar_controlled_record`, `baseline_ttbar_dynamic_certified_record`
and `refined_ttbar_dynamic_certified_record`. Events remain under each output's
`Events/controlled` or `Events/dynamic_certified` directory. The earlier
conservative-mask measurement is in `revised_dynamic_comparison.json` and the
`*_dynamic_record` directories. A second matched final-code pair at actual seed
732212 is preserved in `refined_dynamic_final_comparison.json`; it gives a
30.5% CPU reduction and a cross-section difference of 0.0314 combined
uncertainties. The tables use the requested seed 732211, confirmed in both
integration logs and `randinit`.

`generate_baseline.mg5`, `generate_revised.mg5` and `generate_refined.mg5` specify exports.
`prepare_benchmark.py` installs the physics cards, while
`instrument_benchmark.py OUTPUT` adds the identical nine counters to a fresh
output. `prepare_repeat.py VARIANT PROCESS RUNNAME` restores the fixed-scale
cards and creates a named launch command; for dynamic scales use the frozen
dynamic run card and recompile before timing. Compile both outputs before
starting the pair, then run their `bin/aMCatNLO` command files under the CPU
affinities above with `/usr/bin/time -v`. Freeze each completed run immediately
using `freeze_run.py VARIANT PROCESS RUNNAME`. The
`summarize_controlled.py`/`compare_controlled.py` and
`summarize_dynamic_certified.py`/`compare_dynamic_certified.py` pairs regenerate
the final JSON reports.

In this environment compilation checks, integration and showering require
the transitive FastJet libraries on the loader path:

```sh
export LD_LIBRARY_PATH=/export/tmp/rikkert/FastJet/lib:/export/tmp/rikkert/HepMC/lib:/export/tmp/rikkert/Pythia/pythia8313/lib:/export/tmp/rikkert/lhapdf/lib
```

Without this setting `libfastjetplugins` cannot load its `libsiscone` dependencies.
The failed first refined launches are preserved with the suffix
`_missing_fastjet`; they are excluded from measurements. Check the integration
logs and event files as well as the launcher exit status: the command interface
can return zero after printing a subcommand failure. Confirm that pole checks
actually tried and passed 20 points, rather than accepting a zero-point report.
Restore the frozen run card before each replay, including after a failed
launch: a positive `iseed` is reset to zero during launch, so an unchecked
retry can increment `randinit`. Use a fresh named run after a failed launch;
reusing an incompletely registered run name can fail in the results bookkeeping.

These small scripts and raw outputs are external validation artifacts, not
installed generator components. The repository regression can be replayed with
`python -m unittest tests.unit_tests.fks.test_mc_history_weights -v`.
