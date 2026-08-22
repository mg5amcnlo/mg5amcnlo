# `p p > z z` at NLO + MadSpin, and what the loop-induced `g g` adds on top

The NLO continuation of [`../zz_loopinduced`](../zz_loopinduced): the same
process, the same cuts, the same scale, the same observables, one order higher —
plus the physics figure the two studies together make possible. Findings are in
[RESULTS.md](RESULTS.md).

## The two halves

**Part 1 — MadSpin at NLO.** `p p > z z [QCD]` MC@NLO events, decayed by MadSpin
in **all six** spinmodes, against a directly generated, fully off-shell NLO
four-lepton sample.

| | |
|---|---|
| **sample A** | `p p > z z [QCD]`, then MadSpin with `decay z > e+ e-` and `decay z > mu+ mu-` |
| **truth** | `p p > e+ e- mu+ mu- / a [QCD]` — the reference every ratio divides by |

The loop-induced study could run only four modes: MadSpin refuses `madspin_v1`
and `onshell_v1` for a loop-induced process. This sample is **not** loop
induced, so all six run and are compared.

`fixed_order` is **off** throughout, and is not needed. It is the option for a
*fixed-order* LHE, whose events come in groups (a born event plus its
counter-events) that have to be decayed once, together. MC@NLO events are not
that: they are individual events carrying (possibly negative) weights. That
matters because `fixed_order` is *refused* in `PA` and `madspin`/`full` — had
these events needed it, only `onshell`/`onshell_v1` would have been usable.

**Part 2 — the stacked physics figure.** Three production samples with
identical cuts, scale, PDF and statistics:

| | |
|---|---|
| **NLO** | `p p > z z [QCD]` — `q q~` Born plus `q g` / `g q~` real emission |
| **LI** | `g g > z z [noborn=QCD]` — the loop-induced quark box, stacked on top |
| **LO** | `p p > z z` — the curve both ratio panes divide by |

with two ratio panes, `NLO/LO` and `(NLO+LI)/LO`.

## Making the samples the same physics

Same for all four samples:

* 13 TeV (`ebeam = 6500`), `pdlabel = nn23lo1` / `lhaid = 230000`;
* **fixed** `muR = muF = m_Z = 91.1880 GeV`;
* `pt(Z) > 1 GeV` on **both** `Z`, and `bwcutoff = 15`;
* every per-lepton cut off;
* 50 000 events, requested seed 4321.

### The cut is `pt_min_pdg`, not `ptheavy`

The loop-induced study applied `pt(Z) > 1 GeV` through the run card's
`ptheavy`. That parameter **does not exist on the NLO run card** — it is
`RunCardLO`-only. `pt_min_pdg` exists on *both* card classes, so it is what this
study uses, on all four samples.

The two are the same cut here, and that is measured rather than argued:
`ptheavy` is an OR over the heavy final states while `pt_min_pdg` writes a
per-particle `etmin` (an AND), but at `2 -> 2` the two `Z` have identical `pt`.
The control run `lo_ptheavy` — the LO sample with `ptheavy = 1` and
`pt_min_pdg = {}` — returns **9.2694 ± 0.00781 pb**, the production sample's
number to every printed digit; and the loop-induced sample reproduces the
`ptheavy` result of the previous study.

### The NLO run card is a different class from the LO one

Beyond `ptheavy`, the entries that exist on one and not the other:

| | `RunCardLO` (LO, loop induced) | `RunCardNLO` (NLO, truth) |
|---|---|---|
| fixed scale | `scale`, `dsqrt_q2fact1/2` | `muR_ref_fixed`, `muF_ref_fixed` |
| systematics | `use_syst` | `reweight_scale`, `reweight_PDF`, `store_rwgt_info` |
| lepton pairs | `drll`, `mmll` | `drll`, **`drll_sf`**, `mll`, **`mll_sf`** |
| other | `ptheavy`, `nhel`, `sde_strategy` | `parton_shower`, `folding`, `req_acc`, `ptj` |

`drll_sf` and `mll_sf` are `RunCardNLO`-only (`banner.py`, `RunCardNLO.__init__`)
and have no LO counterpart; the LO card carries only `drll` / `drllmax`, which
cover all lepton pairs. §2 of RESULTS.md audits every one of them off the card
each sample was actually run with.

### How the truth sample's cuts are applied

The truth has no `z` to cut on, so `pt_min_pdg = {23: 1}` is natively inert
there and is read instead by
[`zz_equivalent_cuts_nlo.f`](zz_equivalent_cuts_nlo.f), which applies it — and
the `|m_ll - m_Z| < 15 Gamma_Z` window — to the *reconstructed* `(e+ e-)` and
`(mu+ mu-)` systems. Same idea as the loop-induced study's
`zz_equivalent_cuts.f`, but a separate file, because the two run-card classes
give `dummy_cuts` different signatures:

```
LO / loop induced :  dummy_cuts(P)                  P(0:3, nexternal)
NLO               :  dummy_cuts(P, ISTATUS, IPDG)   P(0:4, nexternal)
```

The NLO one is the better signature to work with: `IPDG` arrives per event, so
the leptons are located from it directly instead of from `leshouche.inc` — which
matters more here, since an NLO process has several FKS subprocesses whose
orderings need not agree.

It hard-codes no number: `pt_min_pdg` comes from this run's own `run.inc`,
`bwcutoff` likewise, and `M_Z` / `Gamma_Z` from its own `coupl.inc`.

## Layout

```
observables_zz.py                  the production-level ZZ observables; imports
                                   ../zz_loopinduced/observables.py whole for the
                                   four-lepton ones and the (self-testing) boost
zz_equivalent_cuts_nlo.f           the truth sample's custom cuts
run_zz_nlo.py                      the driver: prod / madspin / controls / harvest
plot_zz_nlo.py                     Part 1 figures, MG7 paper style (+ --check-minus)
plot_zz_nlo_userstyle.py           Part 1 figures, the user's own style
plot_zz_stack.py                   Part 2 figure,  MG7 paper style (+ --check-minus)
plot_zz_stack_userstyle.py         Part 2 figure,  the user's own style
data/histograms.npz                the raw histograms, both halves
data/meta.json                     runs, statistics, seeds, card options, cuts,
                                   wall times, code SHA
data/numbers.txt                   the Part 1 numeric report
data/numbers_stack.txt             the Part 2 numeric report, incl. the
                                   double-counting check
plots/, plots_userstyle/           PDF and PNG
logs/                              run logs, copied as .log.txt
RESULTS.md                         the findings
```

## Re-running

```
export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"     # f2py is required
python3 run_zz_nlo.py --stage prod     --basedir /tmp/zz_nlo_work --nb-core 14
python3 run_zz_nlo.py --stage madspin  --basedir /tmp/zz_nlo_work --nb-core 3
python3 run_zz_nlo.py --stage controls --basedir /tmp/zz_nlo_work --nb-core 10
python3 run_zz_nlo.py --stage harvest  --basedir /tmp/zz_nlo_work
python3 plot_zz_nlo.py --check-minus ; python3 plot_zz_nlo_userstyle.py
python3 plot_zz_stack.py --check-minus ; python3 plot_zz_stack_userstyle.py
```

The whole thing is about 25 minutes of wall time on an 18-core machine. The four
plotting scripts need only `data/`; they import neither MadSpin nor MadGraph.

Notes for whoever runs it next:

* **f2py is required** and it has to be a working one. The bare `f2py` on a
  Homebrew PATH has a dead shebang and fails with exit 126.
* **The `output` commands are serial on purpose.** MG5 compiles CutTools/IREGI
  inside the *source* tree the first time an NLO or loop-induced output is made;
  two outputs started at once race there.
* **A control reuses its sample's process directory**, so it overwrites that
  directory's integration grids and, before this driver named its logs after the
  *run*, overwrote the production run's log as well. The cross sections in
  `meta.json` are therefore read from `Events/<run>/summary.txt` (aMC@NLO) or
  `HTML/<run>/results.html` (MadEvent), which live under the run's own name and
  survive.
* **MG5 rewrites a run card in lower case** once a run has used it
  (`muR_ref_fixed` becomes `mur_ref_fixed`). `set_run_card` matches
  case-insensitively for that reason; a case-sensitive match works on a fresh
  directory and fails on a reused one.
* **MadEvent and aMC@NLO reset `iseed` to 0 after every run.** Read the seed back
  out of the banner; the card lies.
* Logs are copied as `.log.txt`: the repository `.gitignore` carries a blanket
  `*.log`.
