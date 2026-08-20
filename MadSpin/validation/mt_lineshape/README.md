# `m(t)` lineshape across every MadSpin spinmode and unweighting scheme

Baseline reference measurement on the **shipped** code, before any per-event
mass-bound work.  It answers one question: *is the resonance lineshape the same
in every cell that is supposed to sample the same distribution, and where is it
not?*

Everything in `data/` is raw measurement; everything in `plots*/` is
regenerated from it by a script that never touches MadSpin.

## What is measured

Process `p p > t t~` at 13 TeV, LO, both tops decayed fully leptonically
(`t > b w+, w+ > l+ vl` and the charge conjugate), 200 000 production events.
One madevent sample and one MadSpin seed are shared by every cell, so two cells
differ **only** by the spinmode / accept-reject scheme under test.

The observable is a **per-event** quantity: the invariant mass
`sqrt(E^2 - |p|^2)` of the status-2 resonance in the decayed LHE record, i.e.
the virtuality MadSpin assigned to that top in that event.  It is not a fitted
width and not a pole parameter.  Means and rms of it are single numbers per
cell and live in `RESULTS.md` / `plots/lineshape_numbers.txt`; the histograms
are of the quantity itself, and the axis labels say so.

## Which cells are real

The grid is `spinmode x unweighting`, but most of it is not 5 x 4 distinct code
paths.  What the runs themselves report (`meta.json` records it per run, and
`RESULTS.md` proves the collapses bin by bin off the raw histograms):

| spinmode | schemes that are really distinct | lineshape? |
|---|---|---|
| `madspin` | `joint`, `sequential`, `sequential_global_retry` | yes |
| `full` | — | **`full` is `madspin`**: `run_madspin` rewrites the name before anything reads it |
| `PA` | `joint`, `sequential`, `sequential_global_retry`, `sequential_with_mass` | yes |
| `onshell` | `joint`, `sequential`, `sequential_global_retry` | **no** — samples no virtuality, `m = M` exactly |
| `madspin_v1` | one cell; the `unweighting` entry is inert | yes |
| `none` | one cell; the `unweighting` entry is inert | **no** — `m = M` exactly |

`sequential_with_mass` needs a per-particle mass draw, so it is a real scheme
under `PA` only; under the offshell spinmodes `_unweighting_mode` announces
`sequential` instead and the run is bit-identical to the `sequential` cell.

A second axis, `density_keep_jacobian` (PA only, default `True`), is measured
as well: with it off the production reshuffle becomes a post-acceptance
kinematic dressing instead of part of the accept/reject weight, so it changes
what is being unweighted and can move the lineshape.  It has its own figure.

Two **replica** rows re-run one cell off the same production events with a
different MadSpin seed.  They are the noise floor: whatever chi2/dof two
replicas of one scheme give each other is what "agreement" costs here, and no
smaller difference between two schemes means anything.

## Files

| path | what |
|---|---|
| `run_lineshape.py` | runs the grid, writes `data/histograms.npz` + `data/meta.json` |
| `data/histograms.npz` | the raw histograms on a fine uniform grid (`Gamma/12`, 360 bins over the full `+-15 Gamma` support) |
| `data/meta.json` | runs, statistics, seeds, card options, pole/width, cross sections, code SHA |
| `data/logs/*.log` | each run's `madspin.log` |
| `plot_lineshape.py` | MG7-paper style figures + `lineshape_numbers.txt` |
| `plot_lineshape_userstyle.py` | the same figures in the user's own style |
| `plots/`, `plots_userstyle/` | PDF + PNG |
| `RESULTS.md` | the numbers and what they say |

A second campaign sits beside the baseline, taken after the per-event
mass-stage bound landed:

| path | what |
|---|---|
| `data_after_bound/` | the same measurement re-run on the per-event-bound code, plus a `decay_output = weighted` cell per offshell/PA family |
| `compare_bound.py` | before vs after: `chi2`/dof, two-sample KS, mean shifts, threshold slices, and the figures in both styles |
| `plots_bound/`, `plots_bound_userstyle/` | PDF + PNG, and `plots_bound/bound_numbers.txt` |

Two things about that pair are worth knowing before reading it. First, the
baseline was taken on `a311d2e64`, which predates **both** the per-event bound
and the overweight safety net that landed with it, so "before vs after" spans
two changes; `compare_bound.py` separates them by reporting entry counts and
weights as different questions. Second, `decay_output = weighted` makes no
accept/reject at all, so histogrammed with its event weights it is the target
density with no bound anywhere in it — the one reference that cannot move when
a bound does.

`run_lineshape.py` also slices the `m` histogram by the top's velocity in the
`t tbar` rest frame (`BETA_EDGES`, three bands). Near threshold is where the
per-event bound — evaluated at the low corner of the Breit-Wigner windows, and
bounding a ratio of `lambda^(1/2)` factors — sits furthest above the weight it
bounds. The baseline predates the slicing, so those keys are in
`data_after_bound/` only.

The `.npz` is deliberately finer than any binning worth plotting, so the
committed data stays the raw measurement and the binning can be revisited
without re-running MadSpin.  The plotting scripts group whole numbers of fine
bins into three zones (`Gamma/6` in the core, `Gamma/2` on the shoulders,
`2 Gamma` in the far tails), so no bin edge moves and nothing is interpolated.

## Reproducing

```sh
export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
python3 run_lineshape.py --nevents 200000 --outdir data \
        --basedir /some/scratch/tree --nb-core 8 --cross-check
python3 plot_lineshape.py           --data data --out plots
python3 plot_lineshape_userstyle.py --data data --out plots_userstyle
```

The two plotting steps need only numpy and matplotlib and run off the committed
`data/`; neither needs MadSpin, an LHE file, or the repository's LHE parser.

The after-bound campaign and its comparison:

```sh
python3 run_lineshape.py --nevents 200000 --outdir data_after_bound \
        --basedir /some/scratch/tree --nb-core 8 \
        --roles grid,extra,replica,weighted \
        --only madspin_joint,madspin_sequential,madspin_seqglobal,\
PA_joint,PA_sequential,PA_seqglobal,PA_seqwithmass,\
PAnojac_joint,PAnojac_sequential,PAnojac_seqglobal,PAnojac_seqwithmass,\
onshell_joint,onshell_sequential,onshell_seqglobal,\
madspin_joint_rep,PA_joint_rep,madspin_weighted,PA_weighted
python3 compare_bound.py --before data --after data_after_bound \
        --out plots_bound --out-userstyle plots_bound_userstyle
```

`compare_bound.py --selftest` checks its own `chi2` and Kolmogorov p-values
against their table values; scipy is not a dependency of this repository, so
both are implemented in the script.
