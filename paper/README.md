# MC@NLO matching with Pythia8 with local recoil (paper)

The existing MC@NLO samples and their figures require regeneration. On
25 October 2026, inclusive closure exposed an uninitialized coloured core
mother momentum in `cluster.f`, which corrupted the shower damping scales.
The source and prepared matching executables are now fixed. Existing event
files are preserved and marked by `NORMALIZATION_FIX.json`; their rates and
shapes must not be used as validated predictions. The fixed-order and POWHEG
results are unaffected. See `../docs/audits/singletop_normalization_mcexplicit.json`.
The local-recoil numerical repairs are also installed in the export template:
stable beam-axis angles, CM velocities, null recoil components and collinear
energy fractions. The deployment and controlled integration checks are recorded
in `../docs/audits/central_C_LOC_numerical_deployment.json` and
`../docs/audits/central_C_LOC_numerical_fix_validation.json`. A complete temporary
2,000-event generation and eight-worker shower test, including the plots, is
recorded in `../docs/audits/central_C_LOC_fresh_pipeline_validation.json`.
Exports made before these numerical repairs are preserved under
`../campaign_archive/before_numerical_fix_20261025_091850/`.
From the repository root, regenerate all 22 affected MC@NLO samples with

```sh
python3 paper/run_all.py --cores 64
```

This runs four central samples, four G-width variations, two common-G
comparisons and twelve shower-scale variations sequentially. Each sample
has one million events and is showered with the same stable-top,
perturbative-QCD setup as before. The fixed-order and POWHEG runs are
reused. At completion the script updates the comparison PDFs, manuscript
figures, numerical report and generated tables. It does not edit the
manuscript's prose or remove its superseded-sample notice automatically.

`--cores` controls both integration and Pythia shower jobs (default: 64).
For each million-event sample, the shower card uses `nevents = -1`,
`nsplit_jobs = 64` and `combine_td = T`: 64 workers shower 15,625 events
each, with distinct reproducible seeds. MG5 merges their histograms into
the usual HwU/PDF outputs, keeping all hard-scale weights. The process
configuration uses `run_mode = 2` and `nb_core = 64`. For another core
count, the runner chooses the largest exact divisor of the event count
within that budget and MG5's limit of 99 shower jobs. `--cores 1` gives
a serial shower.

Fresh exports go to `central_runs_fixed/`, `g_runs_fixed/` and
`scale_runs_fixed/`. The central run names are `C_GLB`, `C_LOC`, `C_Q` and
`C_QT`, with explicit generation/shower seeds in `proposed_runs.csv`.
Old event files and integration grids are not copied into these exports.
Each new process records its input cards and fixed source hashes, including
the phase-space helpers, radiation maps, initial recoil and FKS sector
functions. The `numerical_fix` field distinguishes the repaired local
geometry and endpoint calculations from earlier exports. Completion
records use schema 3; older prepared or completed exports cannot be
reused as if they contained these repairs.
Verified completed runs are skipped on a subsequent invocation. A failed
run stops the sequence and its log is retained; an incomplete event run
requires inspection before retrying.

The subsequent `C_LOC` generation failure (`Invalid resonance subtraction
scales`) came from the legacy dot product rounding small positive invariants
to zero. The scale conversion now uses the unrounded phase-space helper.
See `../docs/audits/central_C_LOC_resonance_scale_failure.json`.
The existing `C_LOC` integration is retained with an audited
`campaign_generation_pending.json` checkpoint. Running the same command
archives partial event chunks and restarts event generation with
`--only_generation`, without repeating the integration. The runner checks
the saved job list, event count and integration-grid hashes before doing so.
It does not automatically adopt other incomplete runs.

The completed `C_GLB` sample is retained: this local scale routine is not
called with its recoil settings. Its `campaign_source_migration.json`
records that compatibility check, while the completion record keeps the
source hashes actually used to generate the sample in `generation_inputs`.

The existing MG5 Pythia driver analyses one fewer event per worker than
it showers. Thus these 64-way samples have 999,936 analysis calls for
one million accepted events. Reporting uses that count for the inferred
negative-weight fraction; the histogram normalization convention is
unchanged. See `../docs/audits/singletop_parallel_showers_validation.json`
for the small shower test and prepared-card checks.

To inspect the list or prepare the exports without starting integrations:

```sh
python3 paper/run_all.py --dry-run
python3 paper/run_all.py --prepare-only --cores 64
```

`--output-root PATH` changes the parent of the three fresh campaign
directories; `--skip-analysis` omits the final comparison stage. The
generation scripts check that the configured stock and private Pythia
installations exist and that their XML paths match the shower cards.
They keep the same Pythia builds and do not patch the stock installation.

Build the manuscript from this directory with

```sh
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

The bibliography, `references.bib`, uses INSPIRE citation keys; the entries were
downloaded from inspirehep.net.

The figures in `figures/` read the four fresh central exports with

```sh
python3 paper/figures/make_plots.py
```

The matched-run tables and validation figures can be regenerated separately with

```sh
python3 paper/analyse_matched_runs.py --require-complete
```

This requires all 22 fresh runs. `--allow-partial` explicitly permits a
report of finished subsets, omitting unavailable comparisons and envelopes.
It rejects superseded samples rather than falling back to the old directories.
The numbers in the manuscript's prose must be updated from the new report.
The fixed-order validation in section 3.5 uses the twelve completed integrations
under `../fo_runs/`; `fixed_order_results.json` records their full-precision
results, input checks, seed combinations and pulls. The other validation
numbers are from `../docs/single_top_local_recoil.tex`.

The numerical campaigns in `proposed_runs.csv` have sequential runners:

```sh
python3 paper/run_central.py --cores 64
python3 paper/run_g_variations.py --cores 64
python3 paper/run_shower_scales.py --cores 64
```

Run these commands from the repository root. Their defaults are the fresh
directories listed above; the scale campaign retains all nine hard-scale
weights. Each supports `--dry-run`, `--prepare-only`, `--output-dir PATH`
and `--only RUN_ID ...`. For example, `--only C_LOC C_Q` selects two
central samples. The separate `run_fixed_order.py` remains available for
the twelve completed fixed-order definitions, but is not invoked by the
rerun command. See `TODO_runs.txt` for the campaign definitions.
