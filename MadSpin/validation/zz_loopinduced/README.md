# `g g > z z` (loop induced) + MadSpin, against the full off-shell four-lepton calculation

> **Frame correction, 2026-08-27 — applied, and the study re-run.** The angular
> observables of this study (`cos_theta1`, `cos_theta2`, `cos1cos2`, and the
> `f_0` / `f_00` / `f_TT` / `C_kk` coefficients built from them) were first
> harvested with a boost composition that is not the helicity frame — the axis
> in the four-lepton frame, the `l+` boosted into its pair's rest frame straight
> from the lab. That composition carries a Wigner rotation and damps every
> rank-2 moment towards `1/3`. `observables.py` is fixed and guarded by a
> self-test, and **the whole study has since been regenerated with the fix in
> place at 200 000 events per sample**, so `data/` and both plot directories are
> current. Full diagnosis:
> [`POLWEIGHT_CLOSURE_DIAGNOSIS.md`](POLWEIGHT_CLOSURE_DIAGNOSIS.md); the
> corrected coefficients are in
> [SPIN_COEFFICIENTS.md](SPIN_COEFFICIENTS.md).

What is here, and how to re-run it. The findings are in [RESULTS.md](RESULTS.md);
what the two angular figures measure, what the di-boson literature calls it, and
which coefficient is worth quoting are in
[SPIN_COEFFICIENTS.md](SPIN_COEFFICIENTS.md).

## The comparison

| | |
|---|---|
| **sample A** | `generate g g > z z [noborn=QCD]`, then MadSpin with `decay z > e+ e-` and `decay z > mu+ mu-` |
| **sample B** | `generate g g > e+ e- mu+ mu- / a [noborn=QCD]` — the reference |

Both are loop-induced: there is no Born, the whole rate comes from the quark
box. Sample A exercises the loop-induced density-matrix path, which is new in
MG5aMC — MadSpin has to build its spin-density matrix out of a MadLoop
amplitude rather than a tree one, and `HelicityFilterLevel` is forced to 0 for
exactly that reason.

MadSpin's four spinmodes are compared: `none`, `madspin`, `onshell`, `PA`.
`madspin_v1` and `onshell_v1` are *absent on purpose* — MadSpin refuses them for
a loop-induced process:

> The MadSpin modes 'madspin_v1' and 'onshell_v1' are are not compatible with
> loop-induced processes. Please choose a mode among 'none', 'PA', 'madspin' or
> 'onshell'.

## Making the two sides the same physics

Everything in the two run cards is identical except where the final states force
it apart:

* **fixed scales at `m_Z`** on both sides (`fixed_ren_scale = fixed_fac_scale =
  True`, `scale = dsqrt_q2fact1 = dsqrt_q2fact2 = 91.1880`), so no dynamical
  scale can differ between a two-body and a four-body final state;
* same PDF (`nn23lo1`), same beams (13 TeV), same seed policy, 50 000 events
  each;
* `pt(Z) > 1 GeV`. In sample A this is the run card's `ptheavy`, acting on the
  two `z` directly. Sample B has no `z`, so it is applied to the *reconstructed*
  `(e+ e-)` and `(mu+ mu-)` systems;
* `|m_ll - m_Z| < 15 Gamma_Z` on **both** pairs of sample B, matching
  `BW_cut = 15` in every MadSpin card.

`ptheavy` is "minimum pt for at least one heavy final state", i.e. an OR over
the heavy particles. At `2 -> 2` the initial state carries no transverse
momentum, so `pt(Z1) = pt(Z2)` exactly and the OR and the AND are the same cut.
Measured on the produced events: `max |pt(e+e-) - pt(mu+mu-)| = 9.1e-09 GeV`.

### How sample B's cuts are applied

Through the run card's supported `custom_fcts` hook, which replaces `dummy_cuts`
in `SubProcesses/dummy_fct.f`. The file is
[`zz_equivalent_cuts.f`](zz_equivalent_cuts.f) and it **hard-codes no number**:
it reads `ptheavy` and `bwcutoff` out of this run's own `run_card.dat` (via
`cuts.inc` / `run.inc`) and `M_Z`, `Gamma_Z` out of its own `param_card.dat`
(via `coupl.inc`), so the two sides cannot drift apart by someone editing one
card and not the other. It also re-derives the lepton positions from
`leshouche.inc` at run time instead of trusting the process ordering, and stops
the run if it cannot find one each of `e+ e- mu+ mu-`.

Two gotchas that cost time and are worth writing down:

* `ptheavy` is **hidden** from a run card whose process has no heavy final
  state, so it has to be written into sample B's card explicitly before it can
  be set. It stays natively inert there (`setcuts.f` flags a particle heavy only
  above 10 GeV, and every final state is a massless lepton) — which is precisely
  what makes it safe to reuse as the custom cut's threshold.
* `custom_fcts` matches function names **case-sensitively** against a lowercase
  table, so `LOGICAL FUNCTION DUMMY_CUTS` is rejected while
  `logical function dummy_cuts` is accepted. The rejection message is
  `function %s is not designed for overwritting` — with a literal, unformatted
  `%s`.

The run card's own `mmll` / `mmllmax` were **not** used for the mass window. The
card warns that "for four lepton final state mmll cut require to have different
lepton masses for each flavor", i.e. with massless `e` and `mu` it cannot tell a
same-flavour pair from `(e+ mu-)` and would cut the wrong combinations.

Sample B also needs the standard MadEvent lepton cuts turned **off** —
`ptl = 10`, `etal = 2.5`, `drll = 0.4` are defaults, and sample A has no lepton
for them to act on. Leaving them in would have compared a cut four-lepton sample
against an uncut one, and would have looked like a MadSpin discrepancy.

## The decay assignment

Two `z` in the event and two `decay z >` lines is MadSpin's **positional** rule:
the first `z` takes the first line, the second the second. That gives exactly
one `e+e-` pair and one `mu+mu-` pair per event, which is what makes sample A
comparable to sample B, and it is *not* a random draw over the two channels —
a random draw would produce `4e` and `4mu` events and break the comparison.
Confirmed on the events rather than assumed; see RESULTS.md.

## Layout

```
observables.py                     the event-level observables, shared by the
                                   harvester and both plotting scripts
zz_equivalent_cuts.f               sample B's custom cuts
run_zz_loopinduced.py              the driver: prod / madspin / harvest
plot_zz_loopinduced.py             figures in the MG7 paper style (+ --check-minus)
plot_zz_loopinduced_userstyle.py   the same figures in the user's own style
data/histograms.npz                the raw histograms
data/meta.json                     runs, statistics, seeds, card options, cuts, code SHA
data/numbers.txt                   the numeric report
plots/, plots_userstyle/           PDF and PNG
plots/m_mumu_refstyle.{pdf,png}    m(mu+ mu-) a SECOND time, in the layout of
                                   the user's own plot_matplotlib.py.  It
                                   carries no annotation, so read the section
                                   below before quoting it.  plots/m_mumu.* is
                                   unchanged and still the default rendering
logs/                              run logs, copied as .log.txt
RESULTS.md                         the findings
SPIN_COEFFICIENTS.md               what cos(theta1) and cos1cos2 measure: the
                                   polarisation fractions, C_kk, the literature
                                   names, and the verdict on what to quote
polweight_closure.py               the cross-check of that extraction against
                                   MadSpin's own ms_pol_* weights -- the only
                                   script here that runs on a DIFFERENT study's
                                   samples, because these ones carry no such
                                   weights
PA_LOWPT_DIAGNOSIS.md              why PA sits low below pt(ee) = 20 GeV: the
                                   truth's m_4l < 2 m_Z region, which no
                                   spinmode can reach.  Read it before quoting
                                   any pt(e+ e-) or m(l+ l-) number
pa_lowpt_diagnosis.py              its measurement, from the per-event columns;
                                   --from-cache works off data/, --selftest
                                   asserts the four claims
data/pa_lowpt_diagnosis.npz        the derived histograms, 11 kB
plots/pa_lowpt_diagnosis.{pdf,png,txt}
DPHI_PAIRS.md                      the two same-Z Delta phi (dphi_ee,
                                   dphi_mumu): they are a pt(ll) plot in
                                   disguise, so they inherit the support
                                   problem above -- and they are the one
                                   lab-frame handle on the single-Z
                                   polarisation
```

**One thing this comparison cannot do, and it matters.** `g g > z z` produces
two on-shell `z`, so `m_4l = sqrt(shat) >= 2 m_Z`, and the RAMBO reshuffle holds
`sqrt(shat)` fixed — every MadSpin mode has *exactly zero* support below
`2 m_Z`, at any sample size. Sample B has **2.09 %** of its cross section there,
all of it at low `pt`. Any bin-by-bin comparison over the full truth charges
that hole to whichever mode is normalised correctly. See
[PA_LOWPT_DIAGNOSIS.md](PA_LOWPT_DIAGNOSIS.md).

## Reading `plots/m_mumu_refstyle`

A second rendering of `m(mu+ mu-)` only, in the layout of the user's own
`plot_matplotlib.py`: the truth as a solid black step drawn *on top*, each
MadSpin mode as markers with capped per-bin error bars and a faint companion
step, and a ratio pane of `errorbar(fmt='o')` points against a dashed unity
line.

**It shows three curves — `truth`, `madspin`, `PA` — and shows the same three in
both panes**, bar the truth in the lower one, which cannot be a ratio against
itself. `onshell` and `none` are absent from the whole figure. Neither draws any
virtuality, so their pair mass is a delta function at `m_Z`: a single filled bin
in the distribution and, against an off-shell truth, a ratio that is a carpet of
structural zeros plus one off-scale spike rather than a measurement (see
`RATIO_MODES` in `plot_zz_loopinduced.py`). Both still appear in
`plots/m_mumu.*`, where the delta function is the point, and both still get
their rate and shape lines in `data/numbers.txt`.

That is the one structural difference from `plots/m_mumu.*`, and it is why this
figure needs no annotation: there, the distribution pane draws all five curves
while the ratio pane draws two, so the legend over-claims for the lower pane and
a note has to say so. Here the legend describes both panes exactly.

**Beyond that, this figure carries no text past its axis labels and legend.**
Two things it therefore does not say on the canvas, both of which a reader
needs:

* **The shaded green band is a MODELLING spread, not an uncertainty on the
  points.** It is the reference's `ratio_uncertainty`: `|madspin - PA| /
  madspin` per bin, drawn as `1 +- ` that, with a second darker tier at half
  width inside it. Note the denominator — the band is normalised to `madspin`
  while the points around it are normalised to `truth`. That asymmetry is the
  reference's own rule (`band_a_counts` when the ratio divides by the exact
  calculation) and is kept deliberately: the band answers "how far apart are
  the two spin treatments, relative to one of them". The statistical
  uncertainty is the error bars, and it is a separate object.
* **The band is wider than the pane in 12 of the 75 bins**, all of them below
  69 GeV, where `madspin` and `PA` differ by a factor of three and the half
  width reaches 2.32. Those bins are the ones where the green fill runs from
  the pane floor to its ceiling. The ratio window is `(0.0, 2.0)`, chosen from
  the points so that no measured point or error bar is clipped; sizing it to
  contain the band instead needs `(-2.0, 4.0)`, which squashes every point into
  the middle quarter of the pane and spends the lower half of the axis on
  negative ratios. `plot_zz_loopinduced.py` prints the count on every run and
  `draw_refstyle` returns it, so it never has to be re-derived by eye.

## Re-running

```
export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"     # f2py is required
python3 run_zz_loopinduced.py --stage all --basedir /tmp/zz_work --nb-core 6
python3 plot_zz_loopinduced.py --check-minus
python3 plot_zz_loopinduced_userstyle.py
python3 plot_zz_loopinduced.py --only-numbers    # numbers.txt alone, no figures
```

The two plotting scripts need only `data/`; they import neither MadSpin nor
MadGraph.

Notes for whoever runs it next:

* **f2py is required** (loop-induced *and* MadSpin), and it has to be a working
  one. The bare `f2py` on a Homebrew PATH has a dead shebang and fails with exit
  126.
* The two `output` commands are **serial on purpose**. MG5 compiles
  CutTools/IREGI inside the *source* tree the first time a loop-induced output is
  made; two outputs started at once race in that shared directory and one dies
  with `cp: includects/avh_olo.f90: No such file or directory`.
* Every mode gets its own `ms_dir`. Reuse across modes saves about 2.5 minutes of
  MadLoop compilation against a per-mode decay cost of tens of minutes, and
  `run_from_pickle` restores the *pickled* option object, so a reused directory
  carries the first run's `spinmode` and `BW_cut` into every lookup that goes
  through `decay_all_events.options`. Not worth the saving.
* Logs are copied as `.log.txt`: the repository `.gitignore` carries a blanket
  `*.log`.
