# Born support library validation

This records the earlier provider-only implementation. The native-history guard
fix and its subsequent validation are documented in
[`../history_guard_ae5422c5/report.md`](../history_guard_ae5422c5/report.md).

Base revision: `ae5422c5e3e17ecaecac5acd58e8ca17ca912f1a`.
Validation date: 2026-09-22, Linux, gfortran.

**The full shared-library/native-history plan has not passed acceptance.**
The provider library, global registry, local wrappers and build/staging changes
are implemented. Foreign native-context activation, native measure/channel
normalization and provider/history provenance in contribution records are not.
The original incomplete-history guard is retained. No unavailable history is
silently omitted and no local redistribution prescription is substituted.

## Targeted regression checks

Run from the repository root:

```sh
python -W ignore::ResourceWarning -m unittest \
  tests.unit_tests.iolibs.test_born_support \
  tests.unit_tests.fks.test_momentum_maps \
  tests.unit_tests.fks.test_mc_dead_zones \
  tests.unit_tests.fks.test_soft_col_limits \
  tests.unit_tests.iolibs.test_export_fks.TestFKSOutput.test_w_nlo_gen_qcd
```

**33 tests passed in 53.720 seconds.** The log is archived as
`targeted_tests.log`. These tests freshly generate their
outputs from the final source, independently compile the original standalone
Born implementations, and compare them with the DSO with array-bound checking
enabled. They cover DY, both ttbar Born families, W+jet channels, QED charge
correlations, LO-only amplitudes, and an existing mixed gluon/photon extra-Born
process. Comparisons use relative tolerance
`1e-12`, without loosening existing tolerances.

The comparisons include summed and split Born terms, helicities, spin
counterterms, colour/charge correlations, flow/diagram weights and extra Born terms.
They alternate providers and change couplings and momenta, including transverse
momentum changes that leave the former energy/longitudinal cache key unchanged.
Two separately numbered process contexts are also checked against one shared
provider. The registry tests exercise flavour/order maps, identical-particle
orientations, tags, incoming ordering and missing/ambiguous ownership.
The registry also resolves ten identical final-state gluons to their 90
ordered emitter orientations without enumerating spectator permutations.

Build checks cover private dynamic symbols, individual-provider incremental
rebuilds, HELAS dependency refresh, restoration of a removed public module file,
execution after relocating the output, and executable-plus-DSO scratch staging.
Serial/parallel global registries are compared directly, alongside the existing
export regression. The macOS build rules have not been run here.

The full `tests.unit_tests.iolibs.test_export_fks` run completed 12 tests with
five passes and seven setup errors (`export_regression.log`). Six IO tests
have no reference files in this checkout. The GoSam test attempts the
explicitly unsupported Python 3/low-memory/OLP combination. No reference
outputs were generated or replaced to turn those errors into passes. The five
passing tests cover serial/parallel W and Z QCD/QED and W+jet LO-only exports.

LO-only contexts preserve their amplitudes, helicities and flow/diagram weights.
Their unsupported correlated requests now return an explicit API status, since
the original routines have neither native FKS entries nor the required NLO
order slots. Metadata queries return only native Born flavours, initialize
unused topology entries, and retain the topology's Fortran index bounds.

The provider comparisons exposed an uninitialized massive-father spin buffer
in the original Born template. That buffer is now initialized at each
helicity, so changing sectors cannot retain an unused spin-interference value.

## Development physics runs

These runs used fresh full-NLO exports generated during development, before
the final metadata separation, process-number deduplication and unused spin
buffer correction. They are smoke results, not final physics acceptance for
the complete plan. `generate.mg5`, `generation.log`, launch commands, banners,
integration results and subprocess test logs are archived beside this report.

Settings match the earlier fixed-order reference in
[`mcatnlo_xsections_f81ea6917/report.md`](../mcatnlo_xsections_f81ea6917/report.md):
13 TeV proton beams; five massless flavours; built-in `nn23nlo`; fixed scales
91.188 GeV (DY), 173 GeV (ttbar), 80.419 GeV (W+jet); stable top and W;
`ickkw=0`; `mcatnlo_delta=False`; PYTHIA8 matching without showering; folding
1; requested accuracy 0.003. DY requires dilepton mass above 60 GeV. W+jet uses
anti-kT R=0.4 and jet pT above 30 GeV. Full cuts are recorded in the banners.
The fixed-order seed is 41121 and the MC@NLO seed is 51121.

| Process | Fixed-order result [pb] | Unshowered MC@NLO [pb] |
| --- | ---: | ---: |
| DY | 1898.5779783 ± 4.7717054 | 1911.9403936 ± 5.2854993 |
| ttbar | 756.648466 ± 1.4032 | Failed: incomplete native histories |
| W+jet | 16837.9149 ± 44.166 | Failed: incomplete native histories |

All three fixed-order runs attained the requested 0.3% accuracy and agree with
the archived standalone fixed-order references within three combined standard
deviations. The DY MC@NLO and fixed-order results differ by 1.88 combined
standard deviations. Standard matrix-element, pole and MC checks passed for
the subprocesses before integration. The unit soft/collinear checks are also
included in the targeted suite above; foreign-history limits are untested.

The ttbar failure is recorded in
`ttbar/P0_gg_ttx_GF1.0_log_MINT0.txt` for FKS sector 7. W+jet fails in
`wjet/P0_ug_wpd_GF1.0_log_MINT0.txt` (sector 2) and
`wjet/P0_gdx_wpux_GF1.0_log_MINT0.txt` (sector 3). These are the retained
local-runtime completeness errors. Their integration controllers were stopped
after the workers failed; no cross section is reported for either process.

The local FastJet installation's default config returned a stale include
prefix. Runs used the existing `fastjet-config` wrapper in the historical
validation directory, which invokes the installed config with `--guess-prefix`.
`LD_LIBRARY_PATH` included `/export/tmp/rikkert/FastJet/lib`. This changes the
dependency location, not any physics tolerance.

## Unshowered DY event sample

`dy_events.lhe.gz` contains 50 events, seed 61121, with stored scale/PDF weights.
PDF reweighting used LHAPDF set 244600 (`NNPDF23_nlo_as_0119_qed`) through the
installed `/nfs/home/rikkert/LHAPDF` library. This differs from the built-in PDF
backend of the integral-only runs above. The banner records the exact settings.

Reproduce the structural checks with:

```sh
python validation/born_support_ae5422c5/check_events.py
```

All events conserve four-momentum within `1e-5` GeV, have closed colour tags,
finite nonnegative shower scales (21.956503–219.80828 GeV), and 128 finite
reweight entries each. Results are in `dy_events_checks.json`. This establishes
file/weight structure only: numerical scale/PDF reweight equivalence and foreign
event ownership remain unvalidated.

The sample run reports 1880.522337 ± 5.277537 pb, about 4.2 combined standard
deviations below the integral-only DY MC run with the other PDF backend. That
discrepancy has not been isolated and is not claimed as an accepted physics
comparison.

## Remaining acceptance work

The driver must consume global histories only after native state activation
and restoration cover mapping configurations, masses, colours, statistical
factors, order maps, PDF bookkeeping, scales and topology-dependent caches.
Inner contributions must remove their native channel/sampling factors and
retain physical counterevent-to-real measure ratios. Native provenance and
outer event ownership must be tracked separately through reweighting.

Required checks still include independence of the auxiliary native mapping,
complete H sums and foreign limits; final-source 0.3% integrations for all
three processes; and ttbar/W+jet event ownership, colour, scale and reweight
validation. The library infrastructure alone does not satisfy those criteria.
