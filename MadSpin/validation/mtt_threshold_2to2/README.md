# `m_tt` near the `2 m_t` threshold, for a `2 -> 2` production

`1/sigma dsigma/dm_tt` for **`p p > t t~`** -- no jet -- around `2 m_t`,
MadSpin's four spinmodes (`madspin`, `PA`, `onshell`, `madspin_v1`) against a
doubly-resonant off-shell MadGraph truth, **each curve divided by its own total
cross section** so the figure compares shapes.

This is the sibling of [`../mtt_threshold`](../mtt_threshold), which asks the
same question of `p p > t t~ j`. It exists to isolate one thing.

## The question

In the `t t~ j` study the off-shell spinmodes **do** populate `m_tt < 2 m_t`.
The mechanism is `Event.reshuffle_production`, which hands **every** top-level
final-state momentum to RAMBO's `mass_shuffle` at fixed `sqrt(shat)` -- for
`p p > t t~ j` that is `t`, `t~` *and the recoiling jet*. The jet is rescaled by
the same `chi` as the tops, so the `t t~` four-momentum is not preserved:

```
m_tt^2(after)  = shat - 2 sqrt(shat) * chi * E_j        (massless recoil)
m_tt^2(before) = shat - 2 sqrt(shat) *       E_j
```

A mass set below the pole gives `chi > 1` and pushes `m_tt` *down*, below
`2 m_t`. Measured there: `madspin` mean `Delta m_tt` +0.137 GeV with 1382 events
of 1M pushed below threshold, `PA` -0.028 / 1825, `madspin_v1` +0.092 / 458,
`onshell` exactly 0 / 0.

Take the jet away and that term goes with it. For `p p > t t~` the two tops are
the entire final state, so `m_tt = sqrt(shat)` **identically** -- and
`sqrt(shat)` is exactly what `mass_shuffle` holds fixed. So no spinmode should
be able to move `m_tt` at all, whatever virtualities it draws, and the
sub-threshold region should be structurally empty for **every** mode, not just
for `onshell`.

`RESULTS.md` has the answer and the numbers. This file is the how.
[`BUMP.md`](BUMP.md) answers a separate question about the same figure: what
the bump just above `2 m_t` is, and whether it compensates the empty region
below threshold.

## What is in here

| file | what |
|---|---|
| `run_mtt_threshold.py` | generates everything and writes the raw histograms. Imports the `t t~ j` driver and re-points it; the only file that needs MG5/MadSpin. |
| `plot_mtt_threshold.py` | the figure in the MG7 paper style, plus the full numeric report. Imports the `t t~ j` figure code and re-points it. |
| `plot_mtt_threshold_userstyle.py` | the same figure in the user's own matplotlib style. Same data, same numbers. |
| `analyse_bump.py` | a follow-up reading of the same histograms: what the bump just above `2 m_t` is, and whether it compensates the empty sub-threshold region. Writes `plots*/mtt_bump.pdf` + `.png` and `plots*/bump_numbers.txt`; answer in `BUMP.md`. Generates nothing. |
| `data/histograms.npz` | the raw measurement: a uniform 0.25 GeV grid over 290-520 GeV, one `sumw`/`sumw2`/`cnt` triple per sample, plus the event-by-event `Delta m_tt` histograms. |
| `data/meta.json` | processes, cards, cuts, seeds, statistics, cross sections, resolved unweighting scheme, code SHA. |
| `data/logs/` | every MadSpin log and card, and the MG5 logs and scripts. Copied as `.log.txt` because the repository's `.gitignore` has a blanket `*.log` rule. |
| `plots/`, `plots_userstyle/` | PDF + PNG + `numbers.txt` for each style. |

## Reuse, and what "the same figure" means here

Nothing about the measurement or the drawing is re-decided in this directory.

* `run_mtt_threshold.py` loads `../mtt_threshold/run_mtt_threshold.py` by path
  and rebinds three module globals: the production process, the truth process
  and the run card. The LHE harvester, the `m_tt` reconstruction, the
  event-by-event `Delta m_tt` pairing and its `sqrt(shat)` check, the histogram
  grid and the whole staged `main` are that module's own code.
* `plot_mtt_threshold.py` loads `../mtt_threshold/plot_mtt_threshold.py` the
  same way and rebinds the process label and the two legend tables. The panel
  geometry, the zone binning, the log scale, the shape normalisation, the
  clipped ratio pane, the arrows, the open-circle convention, the bands key and
  the minus-sign workaround are all that module's.

Two changes were made **in** `../mtt_threshold/plot_mtt_threshold.py` so that
this could work, and both were checked to leave that study's own output
bit-identical (`plots/numbers.txt` unchanged, `plots/mtt_threshold.png`
identical by MD5):

1. `structurally_empty` no longer tests `key == 'onshell'`. It asks
   `preserves_mtt(d, key)`: did the mode return every event's `m_tt` unchanged,
   as measured by `pair_delta`? For `p p > t t~ j` that is true of `onshell`
   and of nothing else, so the answer there is what it was. Which modes are
   structurally empty is a property of the *process*, which is why it is now
   read out of the data rather than hard-coded.
2. the open circles are drawn **concentric**, largest first, indexed by
   position among the modes that have any. With one such mode -- the `t t~ j`
   case -- that is the original single `ms=5` marker. With four it is the only
   way to show them: they coincide exactly, and the turn-on is binned at 1 GeV,
   which is about 4 pt of axis and cannot hold four 5 pt markers side by side.

The setup line above the curves reads `pp -> t t~`, the truth legend entry
matches, and everything else on the figure is unchanged: no in-plot prose, no
`m_tt` definition on the axis or above it, `m_tt [GeV]` on the x axis, the
`2 m_t` line and tag, the legend, the `+-5 %` / `+-10 %` bands key and the
arrow/circle key.

## Re-running

Re-drawing needs nothing but numpy and matplotlib:

```
python3 plot_mtt_threshold.py            # -> plots/
python3 plot_mtt_threshold_userstyle.py  # -> plots_userstyle/
python3 analyse_bump.py                  # -> both, as mtt_bump.* / bump_numbers.txt
```

`plot_mtt_threshold.py --check-minus` (on by default) re-opens the MG7-style
PDF and asserts a math minus survived matplotlib's usetex Type1 subsetting. It
is discriminating -- `NO_MINUS_FIX=1` makes it report `False` -- and that is the
point of it. It must **not** be pointed at
`plots_userstyle/mtt_threshold.pdf`: that figure renders without usetex, never
goes through the buggy path, and carries `/minus` either way, so the check would
pass vacuously. The user-style script carries an `assert` on
`rcParams['text.usetex']` guarding exactly that.

Re-measuring needs an f2py-capable python:

```
export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"
python3 run_mtt_threshold.py --stage prod    --nevents-prod 1000000 --basedir /tmp/w --nb-core 9
python3 run_mtt_threshold.py --stage truth   --nevents-truth 1000000 --truth-runs 5 --basedir /tmp/w --nb-core 9
python3 run_mtt_threshold.py --stage madspin --modes madspin,PA,onshell,madspin_v1 --basedir /tmp/w --nb-core 3
python3 run_mtt_threshold.py --stage harvest --truth-runs 5 --basedir /tmp/w --cross-check
```

The four MadSpin modes are independent runs off one shared production sample,
each in its own directory, so `--modes` can be used to launch them as four
concurrent processes -- which is how the committed data was made.

`--truth-runs N` is the one option this driver adds. MG5 refuses more than 1M
events in one `generate_events` (`madevent_interface.check_nb_events`), so more
truth statistics can only come from more runs; this launches `run_02..run_0N` on
the same output directory with consecutive seeds and hands them to the inherited
`--truth-lhe` path, so the pooling arithmetic is still `harvest_many`'s.

## The setup

Everything below is the `t t~ j` study's dict, imported verbatim, so "same
model, PDF, scales and cuts" is a property of the code.

* model `sm`, LO, `p = j = g u c d s u~ c~ d~ s~` (no `b` in either, and the `b`
  is massive: `MB = 4.7`).
* 6.5 + 6.5 TeV, PDF `nn23lo1` (the shipped default, no LHAPDF needed).
* **fixed** `mu_R = mu_F = m_t = 173 GeV`. Deliberate, for the same reason as in
  the `t t~ j` study: the default dynamical scale is CKKW back-clustering *of
  the generated final state*, and that final state is not the same on the two
  sides (2 particles vs 6), so a dynamical scale would inject a difference that
  has nothing to do with the physics under study.
* **no cuts.** This is the one deliberate departure. The `t t~ j` run card sets
  `ptj > 20 GeV`, `|eta_j| < 5` because the real emission needs them to be
  finite. There is no jet in this final state for them to act on, so carrying
  them over would record a cut in `meta.json` that applies to nothing. As
  before, `cut_decays` is off, so nothing cuts on the top decay products either
  and the truth sample's acceptance is the production sample's.
* `bwcutoff = 15` in the truth run card and `BW_cut = 15` in every MadSpin card.
  Matched on purpose and in the same convention: MG5 tests
  `abs(xmass - prmass) < bwcutoff * prwidth` (`myamp.f`), i.e. 15 widths in the
  **mass**, which is what MadSpin's `_draw_mass_value` also allows. It sets how
  far below `2 m_t` the truth reaches, so it is a parameter of the figure.

## The truth sample

`p p > t t~, t > w+ b, t~ > w- b~`, generated by MG5 with the decay-chain
syntax. The matrix element carries both tops' Breit-Wigner propagators, so the
tops go off shell and the sample populates `m_tt < 2 m_t` -- which is what makes
the comparison possible at all: the truth reaches a region no spinmode does.

It contains **only the doubly-resonant diagrams** -- exactly the diagram set
MadSpin also has. Single-resonant and non-resonant `W+ b W- b~` contributions
are **not** in it, by choice: including them would measure MadSpin's kinematic
approximation *plus* a class of diagrams the framework never tried to reproduce,
and the two would not be separable afterwards. What is measured here is the
kinematics alone.

The Ws are stable final-state particles on both sides, so the top virtuality is
the only thing that varies and the comparison is like for like.

## The observable

`m_tt` is reconstructed identically on both sides as the invariant mass of the
sum of the four status-1 particles with `|pid|` in `{24, 5}`, i.e.
`(W+ b) + (W- b~)`. There is no `b` anywhere else in either event, so that sum
is unambiguous; the harvester counts the particles it found per event and
`meta.json` records that it was exactly four in every event of every file. A
`lhe_parser` cross-check on a slice of each file is recorded too.

For this process that sum is also the whole final state, so `m_tt` is
`sqrt(shat)` on both sides. That identity is the reason the study has an answer
at all, and it is why `pair_delta`'s `max |Delta sqrt(shat)|` column does double
duty here: it is both the proof that the two event streams are paired and the
statement of the invariance being tested.

## The normalisation, and why it is shape

Every curve is normalised by **its own total cross section over the full `m_tt`
range**, `sum(w)/N` of the whole sample, not the integral of the plotted
316-420 GeV window -- that window holds a few percent of the rate, and
normalising to it would divide out part of the region under study.

The reason is the same as in the `t t~ j` study: MG5's decay-chain truth
truncates each top's Breit-Wigner at `bwcutoff` widths and MadSpin's
`sigma_production * BR` takes no such loss, so the two sides do not share a total
cross section. A change now in review makes MadSpin's reported cross section
carry that truncation; **this branch does not contain it**, so the raw
`truth/mode` ratios and each sample's total `sigma` are recorded in `RESULTS.md`
and in `numbers.txt` to be checked against the corrected code later. Plotting
`1/sigma dsigma/dm_tt` sidesteps the question for the figure.

## What the figure has to make unmistakable

That the sub-threshold region is empty for **all four** modes, and that this is
structural rather than a sample-size accident.

The convention is the `t t~ j` study's, applied to every mode that earns it: an
exact structural zero is drawn as an **open circle on the lower boundary of the
clipped ratio pane, with no arrow**, so it stays visually distinct from a
measured point that ran off the pane (which gets an arrow) and from an empty bin
that is just a statement about `N` (which is left as a gap). Here all four modes
get the circle, the circles coincide, and they are drawn concentric so that all
four are visible.

The emptiness is also stated as a **count** -- `0 of N` -- in `numbers.txt` and
in `RESULTS.md`, because an emptiness drawn is weaker than an emptiness counted.

## No prose on the figure

There is no annotation in the plot area. What the shaded region means, why no
mode reaches it, the counts -- all of that is in `RESULTS.md` and `numbers.txt`,
where it can carry its errors. What stays on the figure is one setup line above
the curves (process, energy, order, scales, `BW cut`), the `2 m_t` tag on the
threshold line, the legend, and the ratio-pane keys. The `m_tt` axis names the
variable and its unit and nothing else; what `m_tt` is built from is in
`RESULTS.md`, in `meta.json['observable']` and in "The observable" above.
