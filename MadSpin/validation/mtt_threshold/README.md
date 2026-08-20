# `m_tt` near the `2 m_t` threshold

`dsigma/dm_tt` for `p p > t t~ j` around `2 m_t`, MadSpin's four spinmodes
(`madspin`, `PA`, `onshell`, `madspin_v1`) against a doubly-resonant off-shell
MadGraph truth, in absolute normalisation.

The question: **above what `m_tt` does MadSpin agree with the truth, and what
does it do below that.**

`RESULTS.md` has the answers and the numbers. This file is the how.

## What is in here

| file | what |
|---|---|
| `run_mtt_threshold.py` | generates everything and writes the raw histograms. The only file that needs MG5/MadSpin. |
| `plot_mtt_threshold.py` | the figure in the MG7 paper style, plus the full numeric report. |
| `plot_mtt_threshold_userstyle.py` | the same figure in the user's own matplotlib style. Same data, same numbers. |
| `data/histograms.npz` | the raw measurement: a uniform 0.25 GeV grid over 290-520 GeV, one `sumw`/`sumw2`/`cnt` triple per sample, plus the event-by-event `Delta m_tt` histograms. |
| `data/meta.json` | processes, cards, cuts, seeds, statistics, cross sections, resolved unweighting scheme, code SHA. |
| `data/logs/` | every MadSpin log and card, and the two MG5 logs and scripts. Copied as `.log.txt` because the repository's `.gitignore` has a blanket `*.log` rule. |
| `plots/`, `plots_userstyle/` | PDF + PNG + `numbers.txt` for each style. |

`plot_mtt_threshold.py --check-minus` (on by default) re-opens the MG7-style PDF
and asserts a math minus survived matplotlib's usetex Type1 subsetting. It is
discriminating -- `NO_MINUS_FIX=1` makes it report `False` -- and that is the
point of it. It must **not** be pointed at `plots_userstyle/mtt_threshold.pdf`:
that figure renders without usetex, never goes through the buggy path, and
carries `/minus` either way, so the check would pass vacuously.

Re-drawing needs nothing but numpy and matplotlib:

```
python3 plot_mtt_threshold.py            # -> plots/
python3 plot_mtt_threshold_userstyle.py  # -> plots_userstyle/
```

Re-measuring needs an f2py-capable python:

```
export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
python3 run_mtt_threshold.py --stage prod    --nevents-prod 1000000 --basedir /tmp/w --nb-core 8
python3 run_mtt_threshold.py --stage truth   --nevents-truth 1000000 --basedir /tmp/w --nb-core 8
python3 run_mtt_threshold.py --stage madspin --modes madspin,PA,onshell,madspin_v1 --basedir /tmp/w --nb-core 8
python3 run_mtt_threshold.py --stage harvest --basedir /tmp/w --cross-check \
        --truth-lhe <extra truth LHE files, comma separated>
```

The four MadSpin modes are independent runs off one shared production sample,
each in its own directory, so `--modes` can be used to launch them as four
concurrent processes. `--stage all` does the lot in one go.

### `madspin_v1`, and the three things it does not do

`spinmode = madspin_v1` is the legacy Fortran path. `interface_madspin.do_launch`
does not send it to `run_onshell` at all -- it falls through to
`madspin.decay_all_events` -- and three settings therefore behave differently
for it. They are reported rather than worked around, and the same card is used
for all four modes so that only the `spinmode` line differs:

* **`nb_core` is ignored.** The legacy decay loop is single-process. The
  `set nb_core` line the factory writes is inert here, and this mode is the
  slowest of the four in wall clock for that reason alone.
* **the run card is not read.** `interface_madspin` refuses `set run_card`
  edits under `madspin_v1` outright. Nothing in this study needs them: the cuts
  and the scales live in the production sample, which is shared.
* **`set unweighting` never reaches `_unweighting_mode`.** That dispatcher is
  inside `run_onshell`. The legacy path runs its own accept/reject -- one
  `max_weight` per decay channel, probed on
  `Nevents_for_max_weight x max_weight_ps_point` phase-space points before the
  event loop (`decay_all_events.get_max_weight_from_event`), then one test of
  the whole decay chain's weight against that bound per trial, inside the
  Fortran driver. Its log carries no `MadSpin: unweighting = ...` line at all,
  so `meta.json` records `legacy` for it -- **not** one of the four schemes the
  card can ask for, and not `auto` resolving to one of them.

`--modes madspin_seq` runs the **control**: `spinmode = madspin` with
`unweighting` forced to `sequential`. It exists because `auto` does *not*
resolve the same way for the density modes -- `_auto_unweighting_mode` sends
`PA`/`onshell` to `sequential` at every multiplicity but `madspin`/`full` to
`joint` for up to two decaying particles, which is exactly this setup. Each
curve on the figure therefore runs its own shipped default, and the control
checks that the `madspin`-vs-`PA` difference is the spinmode and not the
scheme. It is harvested if it is on disk and reported in `numbers.txt`, but it
is deliberately not drawn.

## The setup

Everything below is one dict at the top of `run_mtt_threshold.py`, used verbatim
by both sides, so "same model, PDF, scales and cuts" is a property of the code.

* model `sm`, LO, `p = j = g u c d s u~ c~ d~ s~` (no `b` in either, and the `b`
  is massive: `MB = 4.7`).
* 6.5 + 6.5 TeV, PDF `nn23lo1` (the shipped default, no LHAPDF needed).
* **fixed** `mu_R = mu_F = m_t = 173 GeV`. Deliberate: the default dynamical
  scale is CKKW back-clustering *of the generated final state*, and that final
  state is not the same on the two sides (3 particles vs 7), so a dynamical
  scale would inject a difference that has nothing to do with the physics under
  study.
* one light jet, `pt > 20 GeV`, `|eta| < 5`. No cut on the top decay products
  (`cut_decays` is off), so the truth sample's acceptance is the production
  sample's.
* `bwcutoff = 15` in the truth run card and `BW_cut = 15` in every MadSpin card.
  Matched on purpose, and in the same convention: MG5 tests
  `abs(xmass - prmass) < bwcutoff * prwidth` (`myamp.f`), i.e. 15 widths in the
  **mass**, which is what MadSpin's `_draw_mass_value` also allows. It sets how
  far below `2 m_t` the truth reaches, so it is a parameter of the figure.

## The truth sample

`p p > t t~ j, t > w+ b, t~ > w- b~`, generated by MG5 with the decay-chain
syntax. The matrix element carries both tops' Breit-Wigner propagators, so the
tops go off shell and the sample populates `m_tt < 2 m_t`.

It contains **only the doubly-resonant diagrams** -- exactly the diagram set
MadSpin also has. Single-resonant and non-resonant `W+ b W- b~ j` contributions
are **not** in it, by choice: including them would measure MadSpin's kinematic
approximation *plus* a class of diagrams the framework never tried to reproduce,
and the two would not be separable afterwards. What is measured here is the
kinematics alone.

The Ws are stable final-state particles on both sides, so the top virtuality is
the only thing that varies and the comparison is like for like.

## The observable

`m_tt` is reconstructed identically on both sides as the invariant mass of the
sum of the four status-1 particles with `|pid|` in `{24, 5}`, i.e.
`(W+ b) + (W- b~)`. The light jet is never a `b` and the initial state carries
no `b`, so that sum is unambiguous; the harvester counts the particles it found
per event and `meta.json` records that it was exactly four in every event of
every file. A `lhe_parser` cross-check on a slice of each file is recorded too.

## What the figure has to make unmistakable

Below `2 m_t` no on-shell `t t~` pair can land. `spinmode = onshell` draws no
virtuality and never reshuffles, so it inherits the production sample's `m_tt`
exactly and has **zero** support there -- structurally, for any sample size.
That zero is drawn as a zero, with an open marker on the ratio axis.

`madspin`, `PA` and `madspin_v1` are **not** in that position, and the figure
must not say they are. See `RESULTS.md`: for a `2 -> 3` production the density
modes' reshuffle rescales the recoil jet as well as the tops, so `m_tt` moves
and the sub-threshold region is populated; `madspin_v1` gets there by a
different route, regenerating the whole phase-space point from the decay-chain
topology with the new off-shell masses while holding `sqrt(shat)` and the
production tree's invariants fixed. An empty sub-threshold bin of any of the
three would be a statement about the sample size, so it is drawn as a gap
rather than as a zero.

## The ratio pane is clipped, and says so

The lower pane is capped at `0.8`-`1.2`. Several points genuinely live outside
that window, and the figure marks each of them:

* a **measured** ratio outside the window gets an **arrow** at the boundary it
  left through, pointing that way. A clipped point drawn as an ordinary marker
  sitting at `1.2` would be worse than not clipping at all.
* `onshell`'s exact zero below `2 m_t` keeps its **open circle**, on the lower
  boundary, and carries **no** arrow. It is a structural zero, not a point that
  ran off the pane, and the two have to stay distinguishable.

The y-axis label and an in-pane key both say the pane is clipped, and
`plots/numbers.txt` lists every off-scale ratio with its value and error, so
nothing the clipping hides is lost.
