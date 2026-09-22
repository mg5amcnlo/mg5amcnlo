# Global MC history guard validation

Base revision: `ae5422c5e3e17ecaecac5acd58e8ca17ca912f1a`.
Validation date: 2026-09-22, Linux, gfortran.

The guard now checks the output-wide history registry. The runtime evaluates
foreign histories with their native Born provider, mapping, colour flow, cuts,
PDFs and coupling orders. Missing or ambiguous required histories remain errors.
The implemented sum is `Hhat_a = S_a sum_b P_b (S_b R - M_b)`.

## Regression checks

```sh
python -W ignore::ResourceWarning -m unittest \
  tests.unit_tests.iolibs.test_born_support \
  tests.unit_tests.fks.test_momentum_maps \
  tests.unit_tests.fks.test_mc_dead_zones \
  tests.unit_tests.fks.test_soft_col_limits \
  tests.unit_tests.iolibs.test_export_fks.TestFKSOutput.test_w_nlo_gen_qcd
```

**38 tests passed in 59.604 seconds**, recorded in `targeted_tests.log`.
These freshly export and compile providers and native contexts. Provider
comparisons with the standalone implementations use `1e-12` tolerance and
array-bound checking. They cover DY, both ttbar Born families, W+jet, helicities,
spin/colour/charge correlations, diagram/flow weights, coupling orders, extra
Born terms, model-state changes and interleaved provider/correlated calls.

Registry tests cover grouped-flavour ordering, momentum permutations, identical
orientations, unique ownership, missing histories and serial/parallel exports.
Build tests cover incremental compilation, private symbols, relocation and a
colocated worker library. The NLO LHE parser's existing regression also passes
(`lhe_parser_test.log`; its test harness requires `unittest.debug=False`).

The numerical regressions include two real ttbar points found during event
validation: an almost stationary recoil and a point near coalescing massive-map
solutions. Both pass the existing `1e-7` momentum check. The auxiliary massive
map uses the radiator momentum and opening angle, avoiding the unstable
quadratic inversion. A separate test compares its counterevent/real measure
ratio against the original parameterization at `1e-6`; tolerances were not
relaxed.

## Physics checks

Fresh exports use `loop_sm-no_b_mass`, five light quark flavours, 13 TeV beams,
PYTHIA8 matching, `mcatnlo_delta=False`, no shower, and folding `(1,1,1)`.
Fixed scales are 91.188, 173 and 80.419 GeV for DY, ttbar and W+jet respectively.
DY has `mll_sf=60`; W+jet has `ptj=30`, jet radius 0.4.
Both fixed-order and MC@NLO runs request accuracy `0.003`.
All subprocess ME, MC and pole checks pass with the existing tolerances,
including pole cancellation at 20/20 points per subprocess with tolerance `1e-5`.

Values below are rounded as in the saved run summaries, in pb. The MC integrator
applies the requested accuracy to the absolute-weight integral, so its signed
cross-section uncertainty can exceed 0.3%.

| Process / PDF | Fixed order | Unshowered MC@NLO | Difference / combined error |
| --- | ---: | ---: | ---: |
| DY / built-in nn23nlo | 1906 ± 4.6 | 1905 ± 5.2 | 0.14 |
| ttbar / built-in nn23nlo | 756.5 ± 1.4 | 760.0 ± 2.6 | 1.19 |
| W+jet / built-in nn23nlo | 16740 ± 45 | 16680 ± 81 | 0.65 |
| ttbar / LHAPDF 244600, final massive map | 744.9 ± 1.5 | 742.8 ± 2.7 | 0.68 |

Every comparison passes the three-combined-standard-deviation requirement.
The first three pairs were completed before the final stabilization of the
auxiliary massive map. The final-source LHAPDF pair additionally validates that
change with a complete integration and event generation. Run cards, parameter
cards and seeds are preserved in the archived banners; `results.json` records
the comparisons. `generate.mg5` and each process's command files record the
export/run commands.

Explicit foreign-sector limit checks cover both directions between gg and
quark-antiquark ttbar Born families and the foreign W+jet aliases in
`P0_gdx_wpux`. Each uses 20 soft and 20 collinear points for every valid native
Born configuration, for both MC/ME-limit and ME/ME-limit comparisons. All pass;
logs are in `foreign_limits/`.

## Mapping independence

The instrumented driver evaluates all valid native Born configurations at each
fixed real point and compares the complete H coefficients, grouped by
contribution type, including G replacements. It removes native sampling and
channel factors and retains counterevent/real measure ratios.

All three audits pass 1,000 comparisons at `1e-6`. The ttbar audit includes
1,000 comparisons in foreign gg contexts; W+jet includes 326 foreign-context
comparisons. For ttbar, the flow sampling quantile is held at 0.25 so both
configurations use the same conditional flow. DY and W+jet have one Born flow.
The frozen audit routine is `mapping_audit.f`; the logs are each process's
`mapping_audit.log`. Exit code 78 is its intentional success stop after 1,000
comparisons. The production driver retains its inversion and forward checks
and selects the first native configuration that passes both.

## Events and reweighting

`ttbar_events.lhe.gz` contains 50 unshowered events generated with the final
source and LHAPDF 244600. The offline scale/PDF reweighter completed, producing
128 finite weights per event. Checks can be repeated with:

```sh
python validation/history_guard_ae5422c5/check_events.py
```

The sample passes momentum conservation, colour closure and finite nonnegative
shower-scale checks (32.79–270.71 GeV). All 1,199 contribution records retain
native provenance through parsing/serialization. The 106 native-history
records resolve to unique outer owners in the global registry, including three
records from foreign Born contexts. The central reweight agrees with the event
weight to the LHE printing precision (maximum relative difference `4.57e-5`).
Results are saved in `event_checks.json`.

LHAPDF's Python binding is unavailable in this environment. The Fortran
reweighter successfully writes all PDF member weights; the run summary cannot
combine those members into a PDF uncertainty.

## Scope

New outputs require regeneration to receive the registry, native tables and
wrappers. There is no new run-card option. The Linux build and worker staging
are tested; macOS rules have not been executed here. Physical-Delta and showered
comparisons, and currently unsupported splitting modes or resonance inversions,
are outside this validation. Source hashes are in `tested_sources.sha256`.
