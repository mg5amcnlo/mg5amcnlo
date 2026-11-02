# MadtRex

MadtRex is MadGraph7's event-reweighting layer. It connects madspace's
UMAMI matrix-element libraries (`madspace.MatrixElementApi`) to the
Rex/teaRex LHE parsing and reweighting library, and provides:

- `TupperWare` / `WareHouse`: turn loaded `MatrixElementApi` objects into
  `REX::tea` weightors and build `procReweightor`s from them;
- `Driver`: a JSON/card-driven front end for parameter reweighting
  (`ParamHandler` reads the reweight cards);
- `PdfDriver`: PDF and scale-variation reweighting;
- readers/writers for madspace's binary (`.npy`) event format.

MadGraph7 uses it for the reweighting step after event generation
(`madgraph/iolibs/template_files/mg7/madtrex.py`), as one of two
reweighting backends (see "Choosing a backend").

## Layout

```
madtrex/
  include/madtrex/   C++ headers
  src/               C++ sources
  src/python/        pybind11 bindings (the _madtrex_py module)
  madtrex/           Python package
  Rex/               bundled fork of Rex (Python package: pyrex)
  install.py         installer
```

`Rex/` is a plain copy of https://github.com/zeniheisser/Rex (commit
94b4ed435a4bddce2f041c210e0455f2eff01379), tracked as ordinary files in this
repository; it no longer follows upstream. Rex is dual licensed: LGPL v3 (or
later) as upstream, or, as part of MadGraph7, the MadGraph7 license (`LICENSE`
at the top of MadGraph7).

## Requirements

- **madspace**, installed first (normally `python madspace/install.py`, which
  installs into `madspace/install/`). MadtRex builds against its installed
  headers and `libmadspace`, and imports it at runtime.
- **pyrex**, the Python package of the bundled `Rex/` fork. `install.py`
  builds and installs it for you.
- A C++20 compiler, CMake >= 3.15 and Python >= 3.11. pybind11 and
  nlohmann_json are fetched by CMake at the same versions madspace uses
  (required for the two extension modules to share types).

MadSpace does **not** depend on MadtRex: madspace builds, installs and
imports without it, and without Rex/pyrex.

## Installing

```bash
python madspace/install.py      # if madspace is not installed yet
python madtrex/install.py
```

or, from the MadGraph prompt, `install madspace` followed by
`install madtrex` (which forwards `nb_core` as the job count).

This installs `pyrex` and `madtrex` into `madtrex/install/`. Options:
`--yes` (no prompts), `-j N` (parallel jobs), `--clean` (rebuild from
scratch), `--system` (install into the current Python environment, where
madspace must then also be importable). To use it outside MadGraph:

```bash
export PYTHONPATH=madtrex/install:madspace/install
python -c "import madtrex; print(madtrex.TupperWare)"
```

## Reweighting drivers

- `madgraph/iolibs/template_files/mg7/madtrex_driver.py` is the standalone
  driver: it needs only madspace and madtrex (and its three
  `*_reweight_card.py` modules next to it), not MadGraph. It runs against an
  MG7 process directory or a gridpack:

  ```bash
  python madgraph/iolibs/template_files/mg7/madtrex_driver.py \
      --process-dir PROC --rwgt-card PROC/Cards/reweight_card.dat
  ```

  It adds the weights to the events file itself (see "Choosing a backend"),
  or writes them to `--output PATH`. It finds `madspace/install` and
  `madtrex/install` by walking up from its own location, or uses whatever
  `madspace`/`madtrex` are already importable.
  It may need `make`, to compile a matrix element at another precision when
  the sources are there. It cannot handle `change model` cards.
- `madgraph/iolibs/template_files/mg7/madtrex.py` is the MadGraph side, used
  by MG7's reweighting step: the same driver, plus `change model` support
  (regenerating the process in the new model with `bin/madgraph`). Run it
  as `python -m madgraph.iolibs.template_files.mg7.madtrex`.

## Choosing a backend

An MG7 output can be reweighted by MadtRex or by MadGraph's classic
reweighting module (`madgraph/interface/reweight_interface.py`, f2py-compiled
matrix elements). Both read the same `Cards/reweight_card.dat`, with one
vocabulary (the classic one): `change <option> <value>` for options of the
reweighting, `set ...` for physics parameters. Options both backends have use
the classic name and default.

Who reweights is decided in this order (`madgraph/various/reweight_backend.py`):

1. `generate_events --reweight-backend=classic|madtrex|auto`;
2. the reweight switch of the launch question: `classic`, `madtrex`,
   `density` or a plugin (classic). `ON` is auto;
3. the card's `change backend classic|madtrex|auto` line;
4. auto: a card with directives only the classic module understands
   (`change mode NLO`, `change output`, density-matrix options, ...) goes to
   the classic module, one with MadtRex-only directives (`change mode pdf`,
   `change fptype`, ...) to MadtRex (both kinds at once is an error). An
   output other than MG7, or NLO events, use the classic module. Otherwise
   MadtRex reweights when it is installed and the run's `output_format` is
   `lhe` or `lhe_npy`, else the classic module (when f2py is available).

When two explicit choices disagree, the higher one wins, with a warning
naming both. A choice that cannot run (MadtRex on a madevent output or on
`compact_npy` events, the classic module without f2py, a card with the other
backend's directives) stops the launch before the events are generated, with
every reason. The run logs one line, e.g.
`Reweighting with madtrex (auto: MG7 output, madtrex installed, no
classic-only directives in reweight_card.dat)`, and records the decision in
`Events/<run>/reweighting.json`.

Both backends add their weights to the run's own event file, after the
weights already there (the systematics', an earlier reweighting's):
`Events/<run>/events.lhe.gz` (or `events.lhe`), or `Events/<run>/events.npy`
for an `lhe_npy` run reweighted by MadtRex. The file is rewritten next to
itself and replaced atomically (gzipped again if it was).

- In an LHE file the weights are `<wgt>` entries of the `<rwgt>` block, in
  the `mg_reweighting` weight group (classic module) or `tearex_reweighting`
  (MadtRex) of `<initrwgt>`.
- In `events.npy` each weight is a column `rwgt_<name>`, after the
  systematics' `rwgt_<n>` columns, and `header.lhe` lists it in a
  `tearex_reweighting` weight group. `npy_to_lhe` writes those weights into
  the LHE file it makes.
- The madevent tools' copy of the events (`unweighted_events.lhe.gz`, which
  MadSpin and the shower read) is kept identical to the run's event file.
- An `output_format = "lhe"` run that MadtRex reweights (also when `auto`
  chooses it, and when a shower or MadSpin switched the run to `lhe`) is
  generated as `lhe_npy`: MadtRex reads and extends `events.npy` much faster
  than it parses and rewrites an LHE file. Once reweighted, `npy_to_lhe`
  writes the requested `events.lhe.gz` from it, with every weight (the
  systematics', MadtRex's), and `events.npy`/`header.lhe` are removed; the
  user gets the same LHE file as before, in the format of an LHE run that is
  not reweighted. When the reweighting fails, the LHE file is still written,
  without its weights (and the failure reported, as for any run). A run
  with an LHE-only step before the reweighting (`[postprocessing]`
  `time_of_flight`, or the legacy `systematics`) is generated as LHE, as
  before.
- `reweighting.json` records the file's path (`events`).

The standalone driver does the same unless `--output PATH` asks for the
reweighted events (or, for npy input, the weight files) to be written
elsewhere.

Reweighting runs once per generation, after the systematics.
`generate_events --reweight=CARD` is a shortcut for switching reweighting on
with CARD as `Cards/reweight_card.dat` (a different card already there is
kept as `reweight_card.dat.bak`).

The card vocabulary:

| line | backends |
|---|---|
| `change backend classic\|madtrex\|auto` | both (default auto) |
| `change helicity True\|False` | both (default True) |
| `change mode LO` | both: parameter reweighting, the default |
| `change model NAME`, `change process DEF` | both (MadtRex needs both) |
| `launch [--rwgt_name=NAME]`, `set BLOCK ID VALUE`, `set NAME VALUE`, `scan:[...]` | both |
| `change mode pdf`, then `set PDF_SET\|BEAM1_PDF_SET\|BEAM2_PDF_SET\|MUF\|MUR\|MUF_FACTOR\|MUR_FACTOR` | MadtRex |
| `change fptype`, `change verbose\|max_batch\|max_deferred_iterations\|streaming\|stream_batch_size`, `change nb_core\|min_batch` (see "Threads"), `change alpha_s_power\|scale_variation_output` | MadtRex |
| `change mode NLO\|NLO_tree\|LO+NLO`, `change output`, `rwgt_dir`, `tree_path`, `virtual_path`, `systematics`, `multicore`, `boost`, `change process ... --add`, a param_card path, density-matrix options, plugins | classic |

A card section starts with `change mode LO` or `change mode pdf`; lines before
the first one are LO. The former MadtRex spellings (`set
helicity_reweighting`, `set use_helicity`, `set FPTYPE`, `set mode
parameter|pdf`, `set VERBOSE`, ...) are still accepted, with a deprecation
warning. The standalone driver (`bin/reweight`) is MadtRex only: it accepts
`change backend madtrex|auto` and stops on a card for the classic module,
which needs MadGraph.

## Threads

MadtRex reweights on as many threads as the event generation uses: the
`[run] cpu_thread_pool_size` of the run card (all available cores for -1,
the default). The events are classified into their subprocesses in parallel,
and each batch of events (one per subprocess, or a part of one) is
reweighted on its own thread, the normalisation as well as each launch.

A sample dominated by one subprocess would keep a single thread busy, so a
subprocess is split into batches when it is large: with `N` events and `T`
threads, every subprocess with more than `max(min_batch, N / T)` events is cut
into batches of that size. A small sample, or a small subprocess, is not
spread over all threads: below `min_batch` events (1000 by default) a batch
costs more to set up -- its own matrix element instance, loaded from the
library and the param_card, its own sorting -- than another thread saves,
and a sample of fewer than `2 * min_batch` events is not split at all.

Both are reweight-card options of the LO part, before the first `launch`:

```
change nb_core 4        # threads (auto: as the generation, the default)
change min_batch 5000   # fewest events per batch when splitting
```

An explicit `change max_batch N` splits at N events instead, and `change
nb_core 1` runs on one thread. From Python, `Driver.set_threads()`/
`set_min_batch()` (or the `WareHouse` ones, and `WareHouse.batch_size_for()`
for the resulting batch size); PDF reweighting (`PdfDriver`) runs as one
batch.

A batch split off a subprocess evaluates its own instance of the matrix
element library. MadMatrix libraries keep the model parameters in one
singleton per process directory (in the common library), which loading an
instance resets to the param_card: MadtRex therefore loads them one at a
time, while no matrix element is evaluated, and sets the parameters of the
current launch on them again.

## Helicity reweighting

By default MadtRex reweights each event at its own helicity: the matrix
elements (normaliser and reweighted alike) are evaluated for the helicity
combination in the event's LHE `SPINUP` column, through MadMatrix's
per-event helicity input (`UMAMI_IN_HELICITY_INDEX`), rather than summed
over helicities. Set `change helicity False` (the classic reweighting
module's option, with its default True) for helicity-summed matrix elements
instead:

- in a reweight card (in its LO part, before the first `launch`):
  ```
  change helicity False
  ```
  (`set helicity_reweighting false` and `set use_helicity false` are its
  deprecated former spellings). It has no meaning for PDF reweighting, and a
  PDF section that sets it is an error;
- on the command line of either driver (and `bin/reweight` in a gridpack):
  `--helicity-reweighting` / `--no-helicity-reweighting`, which override the
  card;
- from Python: `run_parameter_reweighting(..., helicity_reweighting=False)`,
  `Driver.set_helicity_reweighting(False)` or
  `TupperWare.set_helicity_reweighting(False)`.

Either way an unchanged parameter point reweights every event by exactly 1.

How an event gets its helicity index: the library enumerates its helicity
combinations in the order of the subprocess's helicity table in
`subprocesses.json` (the table MG7 also writes `SPINUP` from; UMAMI exposes
the library's helicity count, not the table itself). MadtRex matches each
event to a flavor channel of its subprocess by the positions of its legs
(incoming, then outgoing, in LHE order), reads the legs' `SPINUP` values in
that order and looks the combination up in the table. MG7 writes the events
of a mirrored flavor channel with their two incoming legs swapped (LHE puts
the +z beam first); those are put back into the library's leg order, the
momenta passed to the matrix element as well as the helicities.

Helicity reweighting needs all of this to work:

- a **matrix element library** that accepts per-event helicities
  (`UMAMI_IN_HELICITY_INDEX`), and a **helicity map** for its subprocess: a
  helicity table in `subprocesses.json` with one helicity per leg and as many
  combinations as the library reports (`UMAMI_META_HELICITY_COUNT`). If
  either is missing, that subprocess is reweighted with helicity summation,
  with a one-time warning on stderr;
- a **sample** with helicity information: a `SPINUP` of -1, 0 or +1 for
  every particle of every event, in a combination found in the table. An
  event with any other value (9, "unknown/summed", as MG7 writes for an
  interference sample generated with helicity-summed matrix elements), or
  with a combination not in the table, is an **error** naming the matrix
  element and the event; reweight such a sample with
  `change helicity False`.

`madtrex/tools/check_helicity_map.py PROC_DIR EVENTS.lhe` checks the map for
a sample against the matrix element libraries directly (through UMAMI,
independently of MadtRex): the per-helicity |M|^2 sum to the summed one, no
event's own helicity has |M|^2 = 0, and the events' helicities are
distributed as the |M|^2 says they should be.

## Gridpacks

With `[gridpack] include_madtrex = true` (the default) in `run_card.toml`, an
installed madtrex is copied into the gridpack as `madtrex/install/` (just
`madtrex`, `pyrex` and the Rex libraries), next to `madspace/install/`, and
the gridpack's `bin/generate_events` puts it on `sys.path`. If madtrex is not
installed it is skipped with a message. With madtrex included, the gridpack
also gets the standalone driver as `bin/reweight` (plus the reweight-card
modules and `SubProcesses/subprocesses.json`), so the events it generates can
be reweighted without MadGraph:

```bash
bin/generate_events --events 10000
bin/reweight --rwgt-card my_reweight_card.dat
```

Without `include_source`, only the precision the libraries were built with
is available, and `change model` cards need MadGraph.

`include_madtrex_source = true` copies the madtrex sources (including `Rex/`)
as well, and also gives the gridpack `bin/reweight`. On its first use,
`bin/reweight` builds madtrex from those sources (`madtrex/install.py`) when
the gridpack has no madtrex install, and rebuilds it when the install was
compiled against another madspace than the gridpack's (madtrex records the
madspace headers it was built against and refuses any other madspace). This
needs a madspace with its headers, i.e. built from source in the gridpack:
use it with `include_madspace = false` and `include_madspace_source = true`
when the gridpack runs on another machine. The madtrex module finds
`libmadspace` relative to itself in the gridpack layout, so the gridpack can
be moved.

Outside a gridpack, `import madtrex` likewise refuses a madspace other than
the one it was built against (e.g. after `install madspace`), with the
command to rebuild it, and MadGraph's automatic backend choice then uses the
classic module.

## Building by hand

When building by hand, CMake finds madspace by importing it with the build's
Python; pass `-DMADSPACE_DIR=/path/to/madspace` (the package directory, which
contains `include/` and `lib/`) to override.
