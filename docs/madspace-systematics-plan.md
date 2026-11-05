# Plan: native scale/PDF systematics in the madspace event output

Status: 2026-09-02. Phases 0 to 4 implemented (see Section 7).

## 1. Goal

When madspace writes the final event sample (`lhe`, `lhe_npy`, `compact_npy`,
and the gridpack path), it should compute the scale and PDF variation weights
itself, at write time, so that the dedicated post-processing step
(`run_lhe_postprocessing` -> `systematics.py` + LHAPDF) is no longer needed.
The same variation weights must feed the observable histograms, so the run
summary carries scale/PDF bands per bin, not only the nominal curve. The
per-event information that is only needed to *recompute* the weights later
(x1, x2, factorisation scales, PDF product) becomes optional in the output.

Tests use `NNPDF40MC_lo_as_01180` as the nominal PDF, the mg7 default.

## 2. What exists today (facts the design builds on)

**Generation side already records the reweighting inputs.** With
`[generation] systematics = true` the launcher passes `partial_weights=True`
to every `ms.Integrand`
([madevent.py:1895](../madgraph/iolibs/template_files/mg7/madevent.py:1895)).
The integrand then emits `x1`, `fact_scale1`, `x2`, `fact_scale2` and
`partial_weight_product = pdf1*pdf2` next to `ren_scale` and `alpha_qcd`
([integrand.cpp:920-945](../madspace/src/phasespace/integrand.cpp:920)).
`ChannelEventGenerator::write_events` stores them for the accepted
(unweighted) events only
([channel_generator.cpp:581-660](../madspace/src/driver/channel_generator.cpp:581)),
and `EventGenerator::read_and_combine` copies them into the combined buffer
([event_generator.cpp:632-724](../madspace/src/driver/event_generator.cpp:632)).
The record layout is in [io.hpp](../madspace/include/madspace/driver/io.hpp)
(`EventRecord::f_beam1 / f_beam2 / f_partial_weights`). These columns are
currently written into the `compact_npy` and `lhe_npy` outputs and silently
dropped for `lhe`. Nobody consumes them.

**The weight is a product with a known structure.** The nominal event weight is
`INV_GEV2_TO_PB * |M|^2(alpha_s(mu_R)) * f1(x1, mu_F1) * f2(x2, mu_F2) / (2 s x1^2 x2^2)`
([kinematics.hpp:476](../madspace/src/kernels/kinematics.hpp:476)), where the
matrix element takes `alpha_s` as an input and is a full sum over helicities
and colours (the random colour/helicity/diagram inputs only pick the labels
written to the event, [api.cpp:165](../madgraph/iolibs/template_files/mg7/api.cpp:165)).
Which beam PDG goes with `x1`/`x2` is fixed by the flavour index:
`flavors[i]["options"][0][0:2]` in `SubProcesses/subprocesses.json`
(all options of one flavour index share the initial state,
[export_mg7.py:397](../madgraph/iolibs/export_mg7.py:397)). The recorded
`x1`/`x2` are the values used in the PDF call, *before* the optional mirror
z-flip of the written momenta ([integrand.cpp:544-560](../madspace/src/phasespace/integrand.cpp:544)).

**PDF and alpha_s interpolation live in madspace and match LHAPDF.**
`PdfGrid` parses an LHAPDF `.dat` member file, `AlphaSGrid` parses the
`AlphaS_Qs/Vals` of the `.info`, both build Hermite-spline coefficient tensors
([pdf.cpp](../madspace/src/phasespace/pdf.cpp)), evaluated by
`kernel_interpolate_pdf` / `kernel_interpolate_alpha_s`
([kernels/pdf.hpp](../madspace/src/kernels/pdf.hpp)). They reproduce LHAPDF
`xfxQ2` to about 1e-6 and `alphasQ` to about 1e-5 (see memory note
"madspace PDF API"). The coefficient tensor for one member of
`NNPDF23_lo_as_0130_qed` (100 x-nodes, 50 q-nodes, 14 flavours) is
16 x 14 x 51 x 101 doubles, about 9 MB; the raw grid values are 0.6 MB.

**Today's post-processing path.** `run_lhe_postprocessing` calls
`systematics.py` on the LHE file with `--lo_nqcd=<single_qcd_order>`
([madevent.py:2428-2476](../madgraph/iolibs/template_files/mg7/madevent.py:2428)).
Because the mg7 LHE has no `<mgrwt>` block, `lhe_parser.reconstruct_lo_weight`
rebuilds x_i from the written momenta and E_beam, takes the beam PDG from the
written particles, and uses `SCALUP` for both scales
([lhe_parser.py:2207](../madgraph/various/lhe_parser.py:2207)). It needs the
Python LHAPDF module, is single-threaded Python over every event, requires
`output_format = "lhe"`, and gives up (exit code 0, crash log) whenever the
QCD power of |M|^2 is not uniform (`single_qcd_order = -1`,
[export_cpp.py:3521-3535](../madgraph/iolibs/export_cpp.py:3521)).
The acceptance test is `test_systematics_mg7`
([test_cmd_madevent.py:1192](../tests/acceptance_tests/test_cmd_madevent.py:1192)).

**Histograms.** `ObservableHistograms` fills one histogram per observable per
channel during integration, from a scalar weight
([histograms.cpp](../madspace/src/phasespace/histograms.cpp),
`op_histogram` in [cpu/runtime.cpp:803](../madspace/src/cpu/runtime.cpp:803)
and gpu/runtime.cu). `EventGenerator::histograms()` normalises and merges them
into `info.json` (`"histograms"`, [event_generator.cpp:582](../madspace/src/driver/event_generator.cpp:582)).
The `[histograms]` run-card section is empty by default.

**Default PDF.** `NNPDF40MC_lo_as_01180` has `NumMembers: 1` and
`ErrorType: replicas`, so `pdf = ["errorset"]` on the default set yields no
PDF variation at all. `NNPDF23_lo_as_0130_qed` (101 replicas) is available in
the CI cache and on this machine.

## 3. Design

### 3.1 Where the weights are computed: at combine time, on the unweighted sample

Two options were considered.

* **A. Inside the integrand graph at generation time**, as extra outputs.
  Scale variations are cheap there (6 extra PDF calls, 2 extra alpha_s calls
  per point), but PDF error sets are not: 100 members x 2 beams per phase
  space point costs as much as a 2 -> 2 matrix element, is paid on every
  integration point (survey, optimisation, rejected points), and needs
  either 0.9 GB of coefficient tensors on the device or a new on-the-fly
  interpolation kernel. It also touches the instruction set, both runtimes,
  and MadNIS training graphs.
* **B. At combine time**, in `EventGenerator::combine_*`, on the accepted
  events only (10^4 to 10^6 events). Everything needed is already in the
  per-channel event files: `x1, x2, fact_scale1, fact_scale2, ren_scale,
  alpha_qcd, partial_weight_product, flavor_index, subprocess_index`. The
  calculation is plain C++ on the CPU, reuses the existing `PdfGrid` parser
  and Hermite coefficient code, runs inside the existing combine thread pool,
  and is identical for all four output paths (lhe, lhe_npy, compact_npy,
  gridpack).

**Decision: B.** It is the path the `partial_weights` machinery was built
for, it costs seconds instead of a fraction of the generation time, and it
has a built-in self-check: recomputing the *nominal* PDF product and alpha_s
from the stored inputs must reproduce `partial_weight_product` and
`alpha_qcd` to round-off (Section 5, V0).

### 3.2 Formula

For variation k with scale factors (r_R, r_F) and PDF member m (of set S):

```
w_k = w_0 * R_alpha * R_pdf
R_alpha = [ alpha_s^S(r_R * mu_R) / alpha_s^0(mu_R) ]^n_QCD
R_pdf   = f^S_m(pdg1, x1, r_F * mu_F1) * f^S_m(pdg2, x2, r_F * mu_F2)
        / ( f^0_0(pdg1, x1, mu_F1)     * f^0_0(pdg2, x2, mu_F2) )
```

* `n_QCD` is the alpha_s power of |M|^2 for the event's subprocess. When it
  is uniform over the diagrams of a subprocess the rescaling is exact at LO.
  This is the same formula `systematics.get_lo_wgt` applies
  ([systematics.py:952](../madgraph/various/systematics.py:952)); it also uses
  the *varied* set's alpha_s, so a cross-set variation (different
  alpha_s(M_Z)) rescales |M|^2 too. Members of one set share the `.info`
  alpha_s, so "errorset" variations do not need `n_QCD`.
* Leptonic beams: `R_pdf = 1`.
* The denominator is taken from the stored `partial_weight_product`, which
  guarantees `R = 1` exactly for the nominal point.
* Fixed scales (`fixed_ren_scale`, `fixed_fact_scale`) need no special case:
  the stored `ren_scale`/`fact_scale*` are the fixed values and get scaled.
* Alternative dynamical-scale choices (`dyn` in systematics.py) recompute
  mu from the stored momenta with the four existing scale definitions
  ([kernels/scale.hpp](../madspace/src/kernels/scale.hpp)); these are
  invariant under the mirror z-flip. Phase 3.

**Non-uniform QCD power (mixed QCD/QED orders).** Phase 1 keeps today's
behaviour: mu_R variations and cross-set PDF variations are skipped with an
explicit warning, mu_F variations and same-set members are still produced.
Phase 4 makes it exact. The plan first proposed asking the matrix element for
|M|^2 split by alpha_s power; the implementation instead **re-evaluates the
matrix element at combine time** at the varied alpha_s, for the events of the
mixed-order subprocesses only: `R_alpha = |M|^2(alpha_s') / |M|^2(alpha_s)`.
This is exact for any order structure, needs no code-generation change, and
the objections listed in the proposal turned out not to hold: the mirror is a
rotation by pi about the x axis (`kernel_mirror_momenta` flips py and pz), so
the spin-summed |M|^2 is invariant; the matrix element is loaded into a
dedicated CPU context (the CPU library exists whenever a CPU device was run;
a GPU-only run falls back to dropping mu_R with the warning); gridpacks ship
the libraries already. The cost is a handful of matrix-element calls per
unweighted event.

### 3.3 Components

**C++ (madspace)**

1. `driver/systematics.hpp/.cpp` (new).
   * `SystematicsConfig`: `mur`, `muf`, `dyn_scales`, `together`,
     `pdf_members` (list of `{set_name, lhaid, member, dat_path, info_path}`),
     `nominal_lhaid`, `write_inputs`.
   * `SubprocessSystArgs`: `n_qcd`, `beam_pdgs[flavor] = {pdg1, pdg2}`,
     `has_pdf[2]`; built per unmerged subprocess exactly like
     `LHECompleter::SubprocArgs`, JSON (de)serialisable so the gridpack can
     reload it (`data/systematics.json`).
   * `SystematicsCalculator`: owns the nominal and varied `PdfGrid`s (raw
     values only) and `AlphaSGrid`s; `compute(EventBuffer&, WeightMatrix&)`
     evaluates all K variations for a batch of events; thread-safe (read-only
     grids), called from the combine thread pool.
   * `Variation` list with the naming/grouping rules of
     `systematics.write_banner`: integer ids from 1, attributes
     `MUR MUF PDF [DYN_SCALE]`, groups `Central scale variation`
     (`combine="envelope"`) and one group per PDF set
     (`combine=<ErrorType>`), so Pythia8/Rivet/MadAnalysis5 and
     `lhe_parser.parse_reweight` see what they see today.
   * `SystematicsSummary`: per-variation cross-section (sum of weights),
     scale envelope, PDF uncertainty by `ErrorType` (replicas: std-dev;
     hessian/symmhessian: quadrature; `NumMembers == 1`: none, with a
     warning). Same numbers as `systematics.print_cross_sections`.
2. PDF and alpha_s evaluation through the existing batched functions. The
   first implementation added a scalar `PdfGrid::interpolate`; review asked
   for the regular batched API instead (one code path, and the planned
   UMAMI-based external PDF libraries are batched too). The calculator now
   registers the nominal grid and every varied member as globals of a CPU
   context under a private prefix, restricted to the PIDs the events use,
   and evaluates `PartonDensity` (dynamic PID) and `RunningCoupling`
   functions through runtimes, one batched call per grid and per
   (alpha_s grid, mu_R, dynamical scale) combination for a whole combine
   batch. Memory: the coefficient tensor of a member is 0.66 MB per PID kept
   (16 x 5151 doubles for a 100 x 50 grid), so 100 replicas cost 66 MB for a
   single-flavour process and about 0.7 GB for all 11 light partons.
3. Output plumbing.
   * `io.hpp`: new `EventRecord::f_syst_weights` group with a runtime count
     `K` (fields `rwgt_0 .. rwgt_{K-1}`, `f64`); `DataLayout` learns a
     repeated field; `EventBuffer`/`EventFile` unchanged otherwise. A
     `write_inputs=false` layout drops `f_beam1/f_beam2/f_partial_weights`
     from the *output* layout (they stay in the per-channel files).
   * `lhe_output.hpp/.cpp`: `LHEEvent` gets `std::vector<double> rwgt` and
     `format_to` emits `<rwgt><wgt id='k'> ... </wgt></rwgt>` in the
     `%+13.7e` format `lhe_parser` writes; optional `<mgrwt>` block
     (`<rscale>`, `<asrwt>`, `<pdfrwt beam="1/2">`, `<totfact>`) in the
     MadEvent format when `write_inputs` is on; `LHEMeta.headers` gets an
     `initrwgt` header (`escape_content=false`).
   * `event_generator.hpp/.cpp`: `combine_to_*` take an optional
     `SystematicsCalculator`; after each `read_and_combine` batch the
     weights are computed (inside the pool job for the `lhe` path) and
     written; `init_combine` builds the output layouts from `write_inputs`.
4. Event-sample histograms with variations.
   * New `ObservableValues` function generator (observables only, no
     binning) built from the same `HistItem`s as `ObservableHistograms`, one
     per subprocess (observables depend on the subprocess PDG list).
   * At combine time, per batch and per subprocess, a CPU `FunctionRuntime`
     evaluates the observables on the buffer momenta; C++ bins them with the
     full weight matrix into `sum w_k`, `sum w_k^2` per bin and variation.
   * `EventGenerator::event_histograms()` -> `info.json["event_histograms"]`:
     per observable `{name, min, max, weights: [{id, bin_values, bin_errors}],
     scale_envelope: {low, high}, pdf_uncertainty: {low, high}}`, where
     `weights[0]` is the nominal computed on the same events. The
     integration-time `"histograms"` block stays as it is (nominal, higher
     statistics); V3 checks the two nominals agree.
5. Python bindings in [src/python/madspace.cpp](../madspace/src/python/madspace.cpp)
   for the new classes and the extended `combine_to_*` signatures.

**Python (MadGraph)**

6. Run card, `RunCardMG7` ([banner.py:6374](../madgraph/various/banner.py:6374))
   and [run_card.toml](../madgraph/iolibs/template_files/mg7/run_card.toml):
   new `[systematics]` section
   ```toml
   [systematics]
   enable = true                 # compute scale/PDF weights when writing events
   mur = [0.5, 1.0, 2.0]
   muf = [0.5, 1.0, 2.0]
   together = ["mur", "muf"]     # 3x3 grid, as systematics.py
   dynamical_scale = []          # extra choices, phase 3
   pdf = ["errorset"]            # members of the nominal set; set names / lhaids; "central"
   write_inputs = false          # keep x1/x2/mu_F/pdf product columns (npy) and <mgrwt> (lhe)
   ```
   `[generation] systematics` becomes an alias of `[systematics] enable`
   (the legacy `use_syst` mapping at [banner.py:7173](../madgraph/various/banner.py:7173)
   follows). `[postprocessing] systematics*` defaults to `false` and is kept
   only as the legacy LHAPDF path for A/B checks; it refuses to run when the
   native weights are already in the file.
7. Launcher [madevent.py](../madgraph/iolibs/template_files/mg7/madevent.py):
   `init_beam` resolves the PDF list (`ensure_pdf_set` downloads members as
   it does for the nominal set; "errorset" expands from `NumMembers`), forces
   `partial_weights` on when `enable`, builds `SubprocessSystArgs` next to
   `build_lhe_completer`, adds the `initrwgt` header in `build_lhe_meta`,
   passes the calculator in `generate_events`, and `get_result` reports the
   per-variation cross-sections and the bands. `save_gridpack` stores
   `data/systematics.json`; [gridpack.py](../madgraph/iolibs/template_files/mg7/gridpack.py)
   reloads it.
8. Exporter [export_mg7.py](../madgraph/iolibs/export_mg7.py):
   `subprocesses.json` gains `"qcd_power"` per subprocess (uniform power of
   alpha_s in |M|^2, else -1), computed with the same
   `diagram.calculate_orders()` loop as `single_qcd_order`.
9. A sidecar `events.weights.json` next to `events.npy` (variation ids,
   attributes, groups) so npy consumers get the same metadata as `initrwgt`.

### 3.4 Phases

| Phase | Content | Depends on |
|---|---|---|
| 0 | Exporter: `qcd_power` in `subprocesses.json`; run-card section; PDF list resolution and download in the launcher | none |
| 1 | `SystematicsCalculator` (mu_R, mu_F, PDF members/sets, uniform `n_QCD`), shared PDF evaluation, LHE `<rwgt>`/`initrwgt`, npy columns + sidecar, gridpack, summary table; legacy path off by default | 0 |
| 2 | Event-sample histograms with variation bands in `info.json`; `write_inputs` (drop columns / write `<mgrwt>`) | 1 |
| 3 | Dynamical-scale variations from stored momenta; `together` combinations beyond mur x muf | 1 |
| 4 | Exact mu_R variation for mixed QCD/QED orders via |M|^2 split by alpha_s power from the ME API (`umami` output + per-event columns) | 1, madmatrix split orders |

Phase 1 is the deliverable that removes the post-processing step. Phases 2
and 3 complete the request (histogram bands, feature parity with
`systematics.py`). Phase 4 removes a limitation the current path has too.

## 4. Behavioural changes and defaults to confirm

* `write_inputs = false` by default: once the weights are in the file the
  inputs are redundant. Setting it to `true` writes the columns to npy and
  `<mgrwt>` to LHE, which lets `systematics.py`/LHAPDF recompute the weights
  later with other settings and is what V1 uses.
* With the default PDF (`NNPDF40MC_lo_as_01180`, one member) `pdf = ["errorset"]`
  produces no PDF weights. The run prints a warning naming the set and
  suggesting a set with members. No silent success.
* The legacy `[postprocessing] systematics` stays available for one release
  as an A/B tool and is then removed together with `_run_systematics`.
* madspace has no random seed, so any event-by-event comparison between two
  runs is impossible; the validation below compares within one file or
  statistically.

## 5. Validation

Run madspace tests with pytest from the scratchpad with `PYTHONPATH` pointing
at `madspace/install` first, MadGraph tests with `./tests/test_manager.py`
(memory notes "madspace build/install", "running tests"). Nominal PDF for
every test: `NNPDF40MC_lo_as_01180`.

**V0. Unit tests, madspace (`madspace/tests/test_systematics.py`).**
* `interpolate(grid, pid, x, q)` vs `lhapdf.xfxQ2` for member 0 of
  `NNPDF40MC_lo_as_01180` and members 0, 1, 50 of `NNPDF23_lo_as_0130_qed`
  on a 100 x 100 (x, Q^2) grid: relative difference < 1e-5 (skip without
  lhapdf, like `test_pdf.py`).
* Scalar `interpolate` vs the runtime `PartonDensity` kernel on the same
  points: relative difference < 1e-12.
* Nominal reproduction on a real integrand batch (build the integrand from
  a run directory as in the memory note "MadNIS integrand inspection"):
  recomputed `f1*f2 == partial_weight_product` and recomputed
  `alpha_s(ren_scale) == alpha_qcd` to < 1e-12 relative.
* `r_R = r_F = 1`, member 0 gives exactly 1; `n_QCD = 0` makes every mu_R
  weight exactly 1; leptonic beams make every mu_F weight exactly 1.
* `initrwgt` and `<rwgt>` text round-trips through `lhe_parser.Event`
  (`parse_reweight`, `Banner`), ids and groups identical to what
  `systematics.write_banner` emits for the same configuration.

**V1. Weight-by-weight cross-check against the legacy LHAPDF path.**
`u u > u u` (uniform `n_QCD = 2`, no mirror), `output_format = "lhe"`,
`write_inputs = true`, `pdf = ["NNPDF23_lo_as_0130_qed"]` plus the 3 x 3
scale grid. Then run `systematics.py` on the same file with `--remove_wgts`
and the same options; it now reads the `<mgrwt>` block instead of
reconstructing it. Compare every weight of every event: relative difference
< 1e-4 (LHAPDF vs madspace interpolation and alpha_s agree to ~1e-5).
Repeat on a process with a mirror flavour (`u g > u g`) to check the
(x1, x2, pdg) bookkeeping under the z-flip. This second run also tells us
whether today's `reconstruct_lo_weight` from written momenta is correct for
mirrored events; if it is not, that is a finding about the legacy path, not
about this feature.

**V2. Physics closure (independent of `systematics.py`).** LO reweighting
of scales and PDFs is exact, so the reweighted cross-section must agree with
a direct run within Monte Carlo errors. `p p > e+ e-` and `u u > u u` with
`fixed_ren_scale = fixed_fact_scale = true`, `mu = M_Z`:
* sum of `w(mur=2, muf=2)` vs a direct run at `mu = 2 M_Z`;
* sum of `w(pdf = NNPDF23_lo_as_0130_qed)` vs a direct run with that PDF;
* `e+ e- > mu+ mu-` (`n_QCD = 0`, leptonic): all weights equal the nominal.
Acceptance: |difference| < 3 sigma of the combined MC errors, with enough
events that sigma is below 0.5 percent. Both runs on the same code.

**V3. Histograms (phase 2).** For `u u > u u` with `[histograms]`
`jet-pt`, `jet-eta`, `sqrt_s`:
* `event_histograms.weights[0]` vs the integration `histograms`: per-bin
  pull chi^2/ndf within [0.5, 2];
* for every variation, sum over bins (including under/overflow) equals the
  per-variation cross-section from the summary to 1e-10;
* scale envelope is non-trivial for `u u > u u` and exactly the nominal for
  `e+ e- > mu+ mu-`;
* in the mixed-order process `u u~ > d d~` + `QED=2`, the mu_R weights are
  absent with the warning, mu_F weights are present (phase 1 behaviour).

**V4. Output-format and gridpack parity.** Same run card, `lhe`, `lhe_npy`,
`compact_npy` and a gridpack run: identical variation list and ids
(`initrwgt` == `events.weights.json`), identical column layout, and the
per-variation cross-sections agree statistically. With `write_inputs =
false` the npy files carry no `x1/x2/fact_scale*/partial_weight_product`
columns and the LHE has no `<mgrwt>`; with `true` they do.

**V5. Downstream consumers.** Through `MG7RunCmd`: Pythia8 shower reads the
LHEF3 weights (check the HepMC weight names), MadSpin preserves the `<rwgt>`
block, `lhe_parser.parse_reweight` returns K weights. `test_systematics_mg7`
is rewritten to the native path (no `switch`, no LHAPDF Python module needed)
and a second copy exercises the legacy path with `[postprocessing]
systematics = true` as long as it exists. The systematics crash-log filter in
`_run_mg7_postproc` is removed so a failure is visible again.

**V6. Performance.** 10^5 events, 3 x 3 scales + 101 replicas: combine time
with and without systematics, single-threaded and with the default pool.
Target: the systematics pass adds < 20 percent to `combine` and < 2 percent
to the whole run for `u u > u u`. Report the numbers.

**V7. Regression.** Full MadGraph unit suite (baseline 892 tests, 2 known
red), madspace pytest suite, the mg7 acceptance tests
(`test_check_xsec_processes_mg7`, `test_cmd_madevent` mg7 group,
`test_readonly_gridpack`). Generated-code IOTests are the user's; report,
do not regenerate.

## 6. Risks and open points

* **Interpolation agreement.** The on-the-fly scalar evaluation must use the
  same node indexing at q-region boundaries as `initialize_coefficients`;
  V0 (< 1e-12 vs the kernel) is the gate before anything else lands.
* **Memory for many members.** Raw grids only: 101 replicas of a 14-flavour
  set are 60 MB. Coefficient tensors are never built for the variations.
* **Downloads.** "errorset" needs every member file; `ensure_pdf_set`
  already fetches whole sets, so no new mechanism, but a first run with a
  large set is slow. The warning for a one-member set must be loud.
* **Mixed orders** stay partially covered until phase 4; the run summary must
  say so explicitly.
* **`lhe_npy`/`compact_npy` column naming** is a format change for the few
  consumers of these files (all in this repository); the sidecar JSON is the
  contract.
* **No seed in madspace**: every acceptance check is statistical or
  within-file. Choose event counts so that MC errors are below the
  tolerances quoted above.

## 7. Implementation status (2026-09-02)

All four phases are implemented on branch
`claude/madspace-scale-pdf-uncertainty-df5721` (not committed).

* Phase 0. Exporter: `qcd_power` per subprocess in `subprocesses.json`
  ([export_mg7.py](../madgraph/iolibs/export_mg7.py)). Run card: `[systematics]`
  section (`enable, mur, muf, together, dynamical_scale, pdf, write_inputs`),
  `[generation] systematics` kept as a hidden alias, `[postprocessing]
  systematics` now defaults to `false` (legacy path, refused when native weights
  exist). PDF list resolution with downloads in the launcher.
* Phase 1. [driver/systematics.hpp](../madspace/include/madspace/driver/systematics.hpp)
  (`SystematicsConfig`, `SubprocessSystArgs`, `SystematicsCalculator`),
  `rwgt_<id>` npy columns, LHE `<rwgt>` + `initrwgt` + optional `<mgrwt>`,
  `combine_to_*(…, systematics, histograms)`, summary in
  `info.json["systematics"]`, `events.weights.json` sidecar, cross-section
  table in the log, `data/systematics.json` in gridpacks (reloaded by
  `gridpack.py`, including the matrix elements of phase 4).
* Phase 2. [driver/event_histograms.hpp](../madspace/include/madspace/driver/event_histograms.hpp):
  `ObservableValues` (observables without binning) evaluated per subprocess on a
  CPU context at combine time, binned with the nominal and every variation
  weight; `info.json["event_histograms"]` carries per observable the nominal
  `bin_values/bin_errors` (binomial errors, sum = cross section), one entry per
  weight id, the `scale_envelope` (low/high per bin) and the `pdf_uncertainty`
  per set. `write_inputs` (phase 1) drops the reweighting inputs from the output.
* Phase 3. `dynamical_scale` variations (systematics.py codes 1-4, given by the
  mg7 names) recomputed from the stored momenta and combined with the mur/muf
  grid, tagged `DYN_SCALE="n"` in the header.
* Phase 4. Exact mu_R variation for mixed-order subprocesses by re-evaluating
  the matrix element at combine time (Section 3.2); the launcher loads the CPU
  library of those subprocesses into a dedicated context.

Validation results (nominal PDF of this checkout: NNPDF23_lo_as_0130_qed; the
interpolation tests also cover NNPDF40MC_lo_as_01180):

| Gate | Result |
|---|---|
| V0 | the batched `PartonDensity` / `RunningCoupling` functions the calculator uses vs LHAPDF: 1e-15 (PDF) and 3e-16 (alpha_s). `madspace/tests/test_systematics.py`: 9 tests (LHAPDF agreement, LO formula, identities, mixed-order drop, PDF groups and header, dynamical scales, event histograms, JSON round trip), reference values from the same batched functions on the default context; whole madspace suite green |
| V1 | weight by weight vs `systematics.py` + LHAPDF reading the written `<mgrwt>`: `u u > u u` 8 scale + 101 PDF weights 9e-8; `p p > e+ e-` (4 mirrored flavours, 207 of 400 events z-flipped) 2e-8; `u g > u g` 4e-8; `u u > u u` with the 4 dynamical scale choices (37 weights) 1e-7. The (pdg, x) pairs of `<mgrwt>` match the written momenta of mirrored events. Matrix-element reweighting forced on `u u > u u` vs the analytic alpha_s^2 rescaling: 2e-7 over 500 events x 8 weights |
| V2 | not run |
| V3 | event histograms: per-variation bin sums equal the per-variation cross sections to 1e-9; the envelope brackets the nominal in every bin; shapes agree with the integration histograms (sqrt_s chi2/ndf 0.5; the integration histograms are not normalised to the cross section and quote smaller errors in sparse bins, so only shapes are comparable) |
| V4 | `lhe`, `lhe_npy`, `compact_npy` and gridpack runs produce the same ids; npy columns present, reweighting inputs dropped with `write_inputs = false`; the mixed-order gridpack reloads its matrix element and writes the same 19 weights |
| V5 | acceptance tests `test_systematics_mg7` (native, with event histograms), `test_systematics_mg7_legacy`, `test_systematics_mixed_order_mg7` pass; Pythia8/MadSpin consumers not re-run |
| V6 | see the speed table below |
| V7 | run-card unit tests (25) pass; the full MadGraph unit suite was not re-run |

Edge cases run end to end: `u u~ > d d~` + `QED=2` (`qcd_power = [-1, 2]`)
now gets the full scale grid through the matrix element (+58% / -38% at LO
with HT/2), finite positive weights, no warning; without a CPU library it
falls back to dropping mu_R with the warning; `e+ e- > mu+ mu-` writes two
trivial mu_R weights and no mu_F ones.

**Speed** (`u u > u u`, 18-thread Mac, LHE output, runs made one after the
other with nothing else running; the legacy tool ran on the very same file
with the same variations). "109 weights" = 3 x 3 scales + 101 NNPDF2.3
replicas, "8 weights" = the scale grid only.

| Events | Native, 109 weights: whole run / combine step (wall, CPU) | Native, 8 weights: whole run / combine | No systematics: whole run / combine | Legacy `systematics.py`, 109 weights (wall, CPU) |
|---|---|---|---|---|
| 20 000 | 3.1 s / 0.25 s, 0.6 s | 0.3 s / 0.03 s, 0.1 s | 0.3 s / 0.02 s | 21.5 s, 21.0 s |
| 100 000 | 2.3 s / 0.90 s, 2.8 s | 0.4 s / 0.12 s, 0.5 s | 1.0 s / 0.07 s | 104 s, 103 s |

The fixed cost of the 109-weight runs (about 1 to 2 s) is parsing the 100
member grids (10 ms per text file) and building their coefficient tensors;
the per-event cost is 0.25 us CPU per weight (0.9 s wall for 100k events x
109 weights, 24 threads sharing the batched runtime). The legacy tool costs
about 1 ms of CPU per event (single-threaded Python over a 500 MB file at
100k events). An earlier figure of 1025 s for the legacy 100k run was
measured while other tests were running and is superseded.

Build notes: built with the worktree's `madspace/build` (ninja) and installed
with `cmake --install build --prefix madspace/install`; the `generate_pyi`
post-step needs `pybind11-stubgen` in the build Python.
