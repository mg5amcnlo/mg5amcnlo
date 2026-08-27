# Fig. 5 of the MadSpin2 paper — event generation

`Delta phi(e- e+)` between the electron and the positron of the `t t~` decay,
for the SMEFT `O_tG` **amplitude-level interference** term, with spin
correlations (`spinmode = onshell`) against without (`spinmode = none`).
Paper reference: `applications.tex`, Sec. `subsec:valid_interf`, figure
`fig:valid_interf`, currently `figures/dphi_pp_tt_SMEFT_INT_NEWMG_.pdf`.

**This directory generates events only.** It makes no plot. The plotting step
consumes `data/meta.json` and `data/histograms.npz`.

```
python3 run_smeft_fig5.py --workdir /tmp/smeft_fig5_work --fixed-order-probe
```

---

## HEALTH WARNING — the SM NLO sample is not sound

**`sm_nlo` was decayed with its spin-density matrices evaluated at the model
default parameters, not at the run's own card.** Task T113 found that MadSpin's
density path initialises its standalone matrix-element library from
`<madspin_me>/Cards/param_card.dat` — the card `output standalone` wrote from
the model defaults — and never from the run's card, whose fallback branch is
unreachable — the same mechanism described below under *Where the Wilson
coefficient is set, and why it is not the param_card*. Its audit found,
specifically for this directory's samples:

> `PROC_sm_nlo` ran with `MT 172.76 / WT 1.33 / WZ 2.4952 / WW 2.085` while its
> ME directories held `173 / 1.4915 / 2.4414 / 2.0476`.

So the SM NLO density matrices used a top mass 0.24 GeV too high and a top
width 12 % too large. **Every curve and every number involving `sm_nlo` — that
is, variations `C`, `D` and both NLO entries of `E` — is affected at an
unquantified level and must not be quoted as a result until the sample is
regenerated.** The figures are drawn anyway, because they are wanted.

**None of the figures says so on its face.** `C` and `D` predate the finding and
are left byte-identical; `E` carried the caveat in red under its x-axis until
the user asked for every free-floating annotation to be removed from the plot.
So **this section and `plots/numbers_E.txt` — which opens and closes with the
warning — are the only places it lives.** Anyone writing a caption from these
figures must carry it across by hand.

### The two LO samples are clear (audited here, T113's audit having stopped)

The status of `sm_lo` was left unresolved when the audit task was stopped. It is
settled here, cheaply, by running T113's own `audit_me_param_cards.py` over this
directory's `--workdir`. The result, for all six decayed runs:

| run | ME card against the run's card |
|---|---|
| `eft_int` / `onshell` | differs in `sminputs 3` (`alpha_s(M_Z)`) only: `0.1190025` against `0.1179` |
| `sm_lo` / `onshell` | the same, `alpha_s(M_Z)` only |
| `sm_nlo` / `onshell` | `MT`, `ymt`, `WT`, `WW`, `WZ` **and** `alpha_s` — the defect above |
| all three `none` runs | no `madspin_me` directory at all — `none` builds no density matrices, so it cannot be affected |

Note in particular that `MT` is **172.76 in both cards** for the two LO samples:
the top mass, the one parameter that would move the $\Delta\phi$ shape, is
right. The single parameter that does differ, `alpha_s(M_Z)`, is harmless here:
at LO the QCD production density matrix and the `O_tG` interference are both
proportional to $g_s^4$, one overall factor, and MadSpin normalises its
accept/reject weight on `Tr(rho_prod)`, so the factor cancels exactly. The
tree-level decay matrix elements carry no `alpha_s` at all.

**Conclusion: the LO curves — the whole of variation `B`, the whole upper pane
of `E`, and pane 1's denominator — are sound. Only the NLO ones are suspect.**
(`alpha_s` should still be fixed; it is simply not a physics error on these
samples.)

---

## The sign of `ctGRe`, and why it is not free

The paper's text says `ctGRe = 1`. **These samples use `ctGRe = -1`**, and the
plotted quantity is therefore `-2 Re(M_SM^* M_tG)` with respect to the caption
as it currently stands. Two independent reasons:

1. **Normalisation.** At `ctGRe = +1` the integrated interference is
   `sigma = -139.66 pb` (`Lambda = 1 TeV`). Normalising a distribution to unit
   area by dividing by a negative integral flips the curve as well as scaling
   it. At `ctGRe = -1` the integral is `+139.66 pb`, exactly opposite, and the
   normalisation is ordinary.

2. **It is the only sign that runs.** At `ctGRe = +1` the interference is
   negative not just in total but essentially everywhere in the sampled phase
   space: 100 % of the production events come out with a negative weight.
   MadSpin's density spinmodes normalise their accept/reject weight on
   `Tr(rho_prod)` and `_check_production_density` refuses any production event
   with `Tr(rho_prod) <= 0`, so `spinmode = onshell` aborts on the first event
   with

   ```
   MadSpinDegenerateWeight: ... the production spin-density matrix of process
   '21 21 > -6 6' is identically zero ... Tr(rho_prod) = -0.135019...
   ```

   The guard's own diagnostic misreads the situation ("identically zero", and a
   "the full production matrix element vanishes as well" branch) because it
   treats `<= 0` as `== 0`. For a pure amplitude-interference sample a negative
   `Tr(rho_prod)` is physical, not degenerate.

   At `ctGRe = -1` every sign flips and the run completes. About 0.02 % of the
   production events still carry a negative MadEvent weight, and those are *not*
   negative-matrix-element points: MadSpin evaluates `Tr(rho_prod) > 0` for all
   of them (a `spinmode = onshell` run on the negative-weight events alone
   completes normally). They are MadEvent unweighting residuals — the `sm_lo`
   sample, whose matrix element is positive-definite by construction, carries
   the same ~0.02 % negative fraction. MadSpin preserves their sign exactly: the
   negative fraction is identical in the production file and in both decayed
   files.

Everything about the two signs is otherwise identical, because the interference
is linear in `c_tG`. `data/meta.json` records the value used, and the sample
`eft_int_ctg_plus` is a production-only run at `+1` with the same seed that
demonstrates the exact mirror (`sigma = -139.65838 +- 0.3043437 pb` against
`+139.65838 +- 0.3043437 pb`).

## Where the Wilson coefficient is set, and why it is not the param_card

The coefficient lives in a **restriction card** (`restrict_fig5m.dat`,
`restrict_fig5p.dat`, both built from SMEFTsim's `restrict_spyros.dat` by the
driver), not in a `set ctGRe ...` at launch.

MadSpin evaluates its density matrices from a standalone matrix-element
directory (`madspin_me`, `madspin_decay`, written by `output standalone`), and
`initialise_f2py_module` initialises that library from
`<madspin_me>/Cards/param_card.dat` — the card `output standalone` wrote from
the **model defaults** — whenever that file exists, which it always does. The
run's own card, which MadSpin does write to `<path_me>/param_card.dat` from the
event banner, is never read on this path (its fallback branch is unreachable).
So a param_card override at launch reaches MadEvent but *not* the density
matrices: with `set ctGRe -1` the events were generated at `-1` while the
density matrices were evaluated at the model default. Putting the coefficient
in the restriction card makes the two agree by construction.

One further wrinkle: a restriction value of exactly `1` makes MG5 *fix* the
parameter (`fix_parameter_values` -> `rule_card.add_one`) and delete it from the
param_card altogether, which is why `restrict_spyros.dat` leaves `ctGRe`
unsettable. `-1` is an ordinary external parameter.

## The samples

All six decayed samples share model-independent settings: `sqrt(s) = 13 TeV`,
fixed `mu_R = mu_F = 173 GeV`, NNPDF2.3 LO (`lhaid 230000`,
`NNPDF23_lo_as_0130_qed`, `alpha_s(M_Z) = 0.130`), **no cuts**, `bwcutoff = 15`
matched by MadSpin `BW_cut = 15`, and the same decay lines.

| key | process | model | order |
|---|---|---|---|
| `eft_int` | `p p > t t~ NP=1 NP^2==1` | SMEFTsim, `ctGRe = -1` | LO |
| `sm_lo` | `p p > t t~ NP=0` | same model, EFT vertex off | LO |
| `sm_nlo` | `p p > t t~ [QCD]` | `loop_sm` | NLO (MC@NLO) |
| `eft_int_ctg_plus` | `p p > t t~ NP=1 NP^2==1` | SMEFTsim, `ctGRe = +1` | LO, production only |

`sm_nlo` is in `loop_sm` and not in SMEFTsim because SMEFTsim v3.0.2 is a
tree-level UFO — no `CT_vertices.py`, no `Fortran/` — so `[QCD]` cannot be
generated in it. `loop_sm` is set to the SMEFTsim top (`MT = 172.76`,
`WT = 1.33`, `ymt = 172.76`) so the three samples describe one top quark. The
NLO run deliberately keeps the **LO** PDF set, as the paper's NLO `ZZ` study
does, so an LO/NLO comparison shows a change of perturbative order and not a
change of PDF.

### Decay lines

```
decay t  > w+ b, w+ > e+ ve
decay t~ > w- b~, w- > e- ve~
```

One `decay` line per `t`-like particle, in the order the particles appear in the
production process — the multi-channel positional rule. Both tops decay to
*electrons* specifically, because the observable is `Delta phi(e- e+)`. The
harvest does not assume this worked: it counts the status-1 `e-` and `e+` of
every event and records the census in
`lepton_multiplicity_census` / `events_without_exactly_one_e_minus_and_one_e_plus`.

### Scale and PDF

The paper's Sec. 5 default is the CKKW back-clustering scale; every `t t~` study
of this series uses a fixed `mu_R = mu_F = 173 GeV` instead, and that is what is
used here. It matters more than usual for this figure: the whole point is a
ratio between two *decays* of one production sample, and a dynamical scale
computed on the generated final state is not the same object on a production
LHE and on a decayed one.

### Cuts

None. The paper states none for this figure, the process is a `2 -> 2` massive
final state so nothing needs a cut to be finite, and the leptons are left fully
inclusive (`cut_decays = False`, no `ptl`/`etal`/`drll`/`mmll`).

## `fixed_order`

Left off, and the driver *measures* rather than asserts what happens.
`--fixed-order-probe` runs `fixed_order = True` in `onshell`, `PA` and
`madspin` against a 2000-event slice of the NLO sample, plus a
`fixed_order = False` control in `onshell` on the same slice. Results
(`fixed_order_probe` in `data/meta.json`):

| run | verdict |
|---|---|
| `onshell`, `fixed_order = True` | **exited 0 but wrote an EMPTY decayed file — silent no-op** |
| `PA`, `fixed_order = True` | refused (`FIXED_ORDER_RESHUFFLING_SPINMODES`) |
| `madspin`, `fixed_order = True` | refused (same) |
| `onshell`, `fixed_order = False` | ran, 2000 events out, `sigma = 10.898 pb`, 22.9 % negative |

So the answer to "does the NLO sample need `fixed_order`?" is stronger than
"no": it must not use it. MC@NLO writes individual signed-weight events, not
the born/counter-event groups `fixed_order` exists to keep together, so the
grouping finds nothing — the log says

```
MadSpin: fixed_order is on, keeping the joint accept/reject (unweighting ignored)
MadSpin: unweighting = joint (auto, 0 decaying particle(s))
MadSpin: unweighting 500000 events on 12 cores
MadSpin unweight efficiency: 0.0000 (0 written / 0 trials, inf trials/event)
```

and MadSpin exits 0 with an empty output. (Note also `500000 events`: the count
comes from the production banner, not from the 2000 events actually in the
sliced file.) `PA` and `madspin` at least refuse loudly; `onshell` does not.

## Outputs

* `data/meta.json` — the manifest. Per sample: process, model and restriction,
  Wilson coefficients, scale, PDF, cuts, `BW_cut`, seeds, event counts,
  cross-section with its integration error, negative-weight fraction (both at
  production and after the decay), spinmode, the unweighting scheme the run
  actually resolved `auto` to, the overweight safety-net counters, and the
  lepton census.
* `data/histograms.npz` — `Delta phi(e-, e+)` on a uniform 720-bin grid over
  `[0, pi]`, per sample and spinmode: `sumw`, `sumw2`, `count`, `count_pos`,
  `count_neg`, plus the shared `edges`. Far finer than any plot needs, so the
  plotting step can rebin without regenerating.
* `logs/*.log.txt` — production and MadSpin logs (`.log.txt`, because the repo
  `.gitignore` has a blanket `*.log`).

The decayed LHE files are **not** committed: at the statistics used here they
are several GB. They stay under `--workdir`, whose absolute path is recorded in
`meta.json` under `setup.workdir` and per sample under
`spinmodes.<mode>.decayed_lhe`.

### Normalisation, for the plotting step

Two traps, both spelled out per sample in `meta.json` under
`spinmodes.<mode>.normalisation`.

1. **The weights are means, not a sum.** These LHE carry `IDWTUP = -4`:
   `XWGTUP` is the whole cross-section in pb on *every* event, with the event's
   sign attached. So

   ```
   sigma            = sum(sumw) / n_events_in_file          [pb]
   dsigma/dDeltaphi = sumw[i] / n_events_in_file / (pi/720) [pb/rad]
   error on that    = sqrt(sumw2[i]) / n_events_in_file / (pi/720)
   ```

   `mean_weight_over_init_xsec` is recorded and is 1 to rounding, so the
   convention can be checked rather than trusted.

2. **The decayed banner comment is stale.** MadSpin rewrites the `<init>`
   block's `XSECUP` to `sigma * BR` but leaves the production banner's
   `# Integrated weight (pb)` comment untouched, so the comment still shows the
   *production* cross-section — e.g. `139.78` where `<init>` says `1.9739`.
   `decayed_xsec_pb` is taken from `<init>`; both numbers are kept in
   `decayed_xsec_detail` so the discrepancy is visible rather than a trap.

`sumw` may be negative bin by bin for an interference sample. That is physical.

---

## Plotting

```
python3 plot_smeft_fig5.py            # MG7 paper style  -> plots/          (A-D)
python3 plot_smeft_fig5_userstyle.py  # user's own style -> plots_userstyle/ (A-D)
python3 plot_smeft_fig5_varE.py       # both styles, variation E only
```

Both run entirely off `data/histograms.npz` and `data/meta.json`. Neither needs
MadSpin, an LHE file, or `--workdir`: the decayed samples can be gone and the
figure still redraws. `plot_smeft_fig5_userstyle.py` imports its data handling,
error propagation and checks from `plot_smeft_fig5.py`, so there is one
implementation of the physics and two renderings of it.

Each writes PDF **and** PNG for every variation, the plotted curves as their own
`smeft_fig5_curves.npz` (rebinned and normalised, so a panel can be reproduced
without redoing either step), and `numbers.txt`. `plot_smeft_fig5.py` also runs
`--check-minus` on every PDF it writes: matplotlib's usetex Type1 subsetting has
silently eaten every math minus in this project before, and this figure states
its sign convention with one. Note that `pdftotext` extracts *nothing* from
these PDFs — subsetted fonts, no `ToUnicode` — so it cannot be used to verify
any label; the check is on the font encoding, and label text is verified by
looking at the PNG.

### The five variations

| tag | curves | ratio pane(s) | what it adds |
|---|---|---|---|
| `A` | EFT `onshell`, EFT `none`, SM LO `onshell` | `onshell/none` | the original figure plus an SM reference |
| `B` | `A` + SM LO `none` | `onshell/none` | **recommended** — the two `none` curves land on top of each other |
| `C` | EFT both + SM NLO both | `onshell/none` | the same at NLO |
| `D` | EFT both + SM LO both + SM NLO both | `onshell/none` | LO and NLO together |
| `E` | EFT `onshell`, SM LO `onshell`, SM NLO `onshell`, **one** `none` | **two**: shape ratios, then SM + interference | three orders on one pane, and what the operator does to the SM prediction; no text on the plot |

`A`–`D` come from `plot_smeft_fig5.py` and `plot_smeft_fig5_userstyle.py`; `E`
comes from `plot_smeft_fig5_varE.py`, which imports those two modules and
modifies neither, so `A`–`D` are byte-identical to what they were before `E`
existed.

`B` is the one to put in the paper. It is the only variation that shows the
result the extra samples were generated for: the SM and the interference term
have *the same* $\Delta\phi$ shape once the spin correlations are switched off
(they agree to 2 %, against a 0.7 % per-bin statistical error), so the whole
visible difference between the EFT and the SM in this observable — 18 % at
$\Delta\phi=\pi$ — is a spin-correlation effect and nothing else. `A` cannot
make that point (no SM `none` curve, and its ratio pane can only hold one
curve). `C` weakens it: the NLO `none` curve differs from the EFT `none` curve
by up to 8 % because the extra radiation changes the $t\bar t$ boost, so the
coincidence is blurred by something that has nothing to do with spin. `D` says
only that the spin-correlation effect is the same at LO and NLO, at the price of
six curves.

### Variation `E`

Three `onshell` curves — the SMEFT interference, the SM at LO and the SM at NLO
— plus **one** `spinmode = none` curve, over **two** ratio panes.
`plot_smeft_fig5_varE.py --help`; it writes `smeft_fig5_E.pdf`/`.png`,
`smeft_fig5_E_curves.npz` and `numbers_E.txt` into *both* plot directories, and
runs the same `--check-minus` on the MG7 PDF.

#### No text on the plot

`E` carries **axis labels and legends only**. There is no header block, no
parameter-point line, no note on either ratio pane, no title and no footnote,
and the legend entries are identifying rather than explaining. Everything that
used to be written on the figure now lives in this section and in
`numbers_E.txt`, which is why both are more detailed than they would otherwise
be. `numbers_E.txt` ends with an explicit list of what a caption has to carry.

#### Which `none` curve is drawn, and what it may claim

One `none` curve is drawn, the **SMEFT** one, and it covers the two **LO**
samples — the SMEFT interference and the SM at LO — and nothing else. Its legend
entry names exactly those two:

> SMEFT $\mathcal{O}_{tG}$ interference & SM, LO, `none`

The three pairwise agreements, measured on `histograms.npz` at the plotted
20-bin binning (all in `numbers_E.txt`):

| pair of `none` shapes | max \|ratio−1\| | chi2/ndf | per-bin error |
|---|---|---|---|
| SMEFT / SM LO | **2.1 %** | **1.9** | 0.7 % |
| SMEFT / SM NLO | **8.0 %** | **10.5** | 1.3 % |
| SM LO / SM NLO | **8.7 %** | **13.8** | 1.3 % |

So the two LO shapes really are one curve, and the NLO one really is not: it is
four times further away and its chi2/ndf is 10.5, not 1.9. This **confirms
T110's 8 %** rather than overturning it, and the gap does not close at finer
binning (8.1 % at 36 bins, 11.1 % at 72). The cause has nothing to do with spin
— the extra radiation changes the $t\bar t$ boost — so a single `none` curve
labelled as standing for all three samples would simply be false.

The resolution is to narrow the curve's claim, not to add a curve the user asked
not to have: **SM NLO appears on the pane with its `onshell` curve alone**,
exactly as SM LO did in variation `A`, and nothing in the legend invites the
reader to assume it has a dashed partner hiding under the blue one.

The SMEFT `none` is the one kept for the same reasons as before: `E`'s LO
content is then exactly the published figure's curve set; it keeps the blue
sample complete, solid *and* dashed, which is right for a figure whose subject
is the operator; and both LO `none` samples have 1 M events and the same 0.48 %
median per-bin error, so statistics does not decide it.

Everything else in the upper pane is `B` unchanged, unit-area normalisation
included.

#### Pane 1: `NLO/LO` and `SMEFT/LO`, and it is not a K-factor

Every curve in the upper pane is normalised to unit area, so the rates are
divided out *before* the ratio is taken; the pane shows only how the shape
moves. The y label says `shape ratio` and that is now the only thing on the
figure that says it. The number a reader might otherwise think they are looking
at — the LO→NLO K-factor of these samples, `10.900/7.175 = 1.52` — appears
nowhere on the figure and must go in the caption.

#### Pane 2: `(LO + SMEFT)/LO` and `(NLO + SMEFT)/NLO`, cross-section weighted

Adding two unit-area shapes needs a relative weight, and that weight sets the
whole size of the pane. The weight used is the ratio of the **decayed cross
sections**, which `meta.json` carries for every sample:

```
n_sum(x) = [sigma_SM n_SM(x) + sigma_int n_int(x)] / (sigma_SM + sigma_int)
```

with `sigma_int = 1.9739 pb`, `sigma_SM(LO) = 7.1753 pb`,
`sigma_SM(NLO) = 10.9004 pb`, all `onshell`, giving
`w = sigma_int/sigma_SM = 0.275` (LO) and `0.181` (NLO). That is the real
relative rate of the two contributions, so the pane is the physical statement
"what the operator does to this distribution". The alternative — averaging the
two unit-area shapes with equal weight — is well defined but corresponds to no
parameter point: it would assert the dimension-six interference is as large as
the SM, which at `ctGRe = -1`, `Lambda = 1 TeV` it is not.

Three things follow. **None of them is on the figure any more**, so all three
have to be in the caption:

1. **The pane's magnitude scales with `c_tG/Lambda^2`.** The samples are at
   `ctGRe = -1`, `Lambda = 1 TeV`. At `c_tG = +1` the interference flips sign
   and the two pane-2 curves *mirror about 1*. Without that statement the pane
   has no scale at all.

2. **Pane 2 carries no information beyond pane 1 and the two cross sections.**
   The shared `n_SM` cancels algebraically:

   ```
   (SM + int)/SM = 1 + [w/(1+w)] * (n_int/n_SM - 1)
   ```

   so pane 2's LO curve is pane 1's blue curve shrunk towards 1 by
   `w/(1+w) = 0.2157`, and its NLO curve is `SMEFT/NLO` shrunk by `0.1533`. This
   is worth stating rather than hiding — and it is also how the errors are
   computed, so the one `n_SM` measurement is not counted as two independent
   ones in numerator and denominator.

3. The NLO curve of pane 2 is **smaller** than the LO one for an arithmetic
   reason, not a physical one: `sigma_SM` grows by the 1.52 K-factor while the
   interference is only available at LO, so `w` falls from 0.275 to 0.181. Read
   it as "an LO interference against an NLO SM", which is what it is, not as
   "the operator matters less at NLO".

Both ratio panes use the `onshell` curves throughout, since that is the physical
prediction. **Both need `sm_nlo`, which is not sound** — see the health warning
at the top of this file.

The measured ends of the four ratio curves, on the plotted 20-bin binning:

| curve | first bin | last bin |
|---|---|---|
| `NLO / LO` (pane 1) | **+3.57 % ± 1.45** | **−6.53 % ± 1.04** |
| `SMEFT / LO` (pane 1) | **+11.54 % ± 0.79** | **−17.76 % ± 0.46** |
| `(LO + SMEFT) / LO` (pane 2) | **+2.49 % ± 0.17** | **−3.83 % ± 0.10** |
| `(NLO + SMEFT) / NLO` (pane 2) | **+1.18 % ± 0.23** | **−1.84 % ± 0.15** |

#### Caption for `E`

Everything the figure no longer says, in one place. `\OM{}` marks the two things
a referee would ask about.

```latex
\caption{$\Delta\phi(e^-e^+)$ of the electron and the positron from the
$t\bar{t}$ decay in $pp\to t\bar{t}$ at $\sqrt{s}=13$~TeV, NNPDF2.3LO, fixed
$\mu_R=\mu_F=173$~GeV and no cuts, with
$t\to W^+b\,(W^+\to e^+\nu_e)$ and $\bar t\to W^-\bar b\,(W^-\to e^-\bar\nu_e)$.
Solid: with spin correlations (\code{spinmode=onshell}), for the amplitude-level
interference of Eq.~(\ref{eq:CMDM_interf}) (blue), the SM at LO (black) and the
SM at NLO (red).  Dashed: the same observable without spin correlations
(\code{spinmode=none}); \emph{one} such curve is drawn because the interference
and the SM LO shapes coincide there to $2\%$ ($\chi^2/\mathrm{ndf}=1.9$ against a
$0.7\%$ per-bin error), so the entire separation of those two solid curves is a
spin-correlation effect.  The SM NLO \code{none} shape is \emph{not} on that
curve --- it differs by $8\%$, a radiation effect on the $t\bar t$ boost rather
than a spin effect --- and is therefore not drawn.  Every curve is normalised to
unit area.  The interference is generated at \code{ctGRe=-1} with
$\Lambda=1$~TeV, so the quantity plotted is
$-2\operatorname{Re}(\mcal^*_{\rm SM}\mcal_{tG})$.  Middle panel: ratios of the
unit-area \code{onshell} shapes to the SM LO one.  Because the rates are divided
out before the ratio is taken this is a \emph{shape} ratio and not a $K$-factor;
the LO$\to$NLO $K$-factor of these samples is $1.52$ and is not shown.  Lower
panel: the SM prediction with the interference added, $(\sigma_{\rm SM}n_{\rm
SM}+\sigma_{\rm int}n_{\rm int})/(\sigma_{\rm SM}+\sigma_{\rm int})$ divided by
$n_{\rm SM}$, i.e.\ weighted by the samples' own cross sections
($\sigma_{\rm int}/\sigma_{\rm SM}=0.275$ at LO and $0.181$ at NLO).  The size of
this panel is proportional to $c_{tG}/\Lambda^2$ and its two curves mirror about
unity for $c_{tG}>0$; the interference is leading order in both curves, so the
NLO one is the smaller only because $\sigma_{\rm SM}$ carries the $K$-factor.
\OM{The SM NLO sample was decayed with its spin-density matrices evaluated at
the model default $m_t=173$~GeV and $\Gamma_t=1.4915$~GeV rather than the run's
$172.76$ and $1.33$; the red curves are provisional until it is regenerated.}
\OM{Samples: $10^6$ events each at LO, $5\times10^5$ at NLO.}}
```

#### One style note

The user-style rendering extends the ratio-limit ladder it imports with
intermediate rungs (`±2 %`, `±5 %`, `±8 %`, `±20 %`), locally and in
`plot_smeft_fig5_varE.py` only: the user's own ladder jumps from `±1 %` to
`±15 %`, which was fine for `B`'s `±30 %` pane but would draw pane 2's `±2.5 %`
as a flat line.

### What is plotted, and the normalisation

`(1/sigma) dsigma/dDelta phi` in 1/rad, **every curve normalised to unit area**,
on 20 uniform bins over `[0, pi]` (`--nbins` must divide the stored 720). The
interference normalisation is arbitrary — it scales with `c_tG/Lambda^2` — so an
absolute vertical axis would carry no information.

The ratio pane is each sample divided by **itself** with the spin correlations
switched off, never one sample by another. For the interference term that is the
only ratio that is well defined without a convention, because the arbitrary
factor cancels between numerator and denominator.

Normalising to unit area costs nothing here, and `check_normalisation()` proves
it rather than asserting it: `onshell` and `none` share a total cross section to
`5e-4` (MadSpin writes `sigma_production * BR` in either mode), so the ratio of
the unit-area shapes *is* the absolute ratio to that accuracy.

`check_normalisation()` re-derives all three of the traps in the section above
from `histograms.npz` alone, and `main()` refuses to draw if any of them fails:

* `sigma = sum(sumw)/N` reproduces every decayed `<init>` XSECUP to `1.1e-4`;
* the banner `# Integrated weight (pb)` comment is the *production* cross
  section on all four LO samples — using it would be wrong by a factor ~70;
* `sumw2` errors are **1.81x** the naive `sqrt(N)` ones on the 22.9 %-negative
  NLO sample, and 1.00x on the LO ones. A `sqrt(N)` error bar on the NLO curve
  would be about half its true size.

### The size of the spin-correlation effect

On the plotted binning, `onshell/none`:

| sample | first bin | last bin |
|---|---|---|
| EFT `O_tG` interference | **+31.2 % ± 1.0** | **−29.8 % ± 0.4** |
| SM LO | +18.2 % ± 0.9 | −15.2 % ± 0.4 |
| SM NLO | +14.4 % ± 2.2 | −15.0 % ± 1.2 |

The paper's "an impact of up to 25 %" is **too small**: the interference term
moves by about 30 % at *both* ends, and the effect is monotonic in between, so
the ratio spans a factor 1.31/0.70 = 1.87 across the range. "up to 30 %" is the
correct statement. At finer binning the first bin drifts to +25 % (the ratio is
flat to within its errors below `0.2 pi`), so 30 % is the value at the
figure's own binning and should be quoted with it.

## The paper text

The caption and the paragraph below match what `plots/smeft_fig5_B.pdf` shows.
Three things in the current text have to change: the Wilson coefficient sign,
the 25 %, and the missing statement of the scale, which deviates from the Sec. 5
default announced in `validation.tex`.

```latex
% --- applications.tex, replacing "setting all Wilson coefficients ... importance."
setting all Wilson coefficients to zero except for the real part of the
$\mathcal{O}_{tG}$ operator, which we set to \code{ctGRe=-1} with
$\Lambda=1$~TeV.  The sign is not a free choice: at \code{ctGRe=+1} the
interference term of Eq.~(\ref{eq:CMDM_interf}) is negative over essentially the
whole sampled phase space, so that normalising the distribution to unit area
would flip its sign, and \newms\ rejects production events with a non-positive
$\mathrm{Tr}(\rho_{\rm prod})$; at \code{ctGRe=-1} every sign is reversed and the
generation is ordinary.  The quantity shown is therefore
$-2\operatorname{Re}(\mcal^*_{\rm SM}\mcal_{tG})$.  The samples are generated at
$\sqrt{s}=13$~TeV with the NNPDF2.3LO set, no cuts, and a \emph{fixed}
$\mu_R=\mu_F=173$~GeV rather than the CKKW scale used elsewhere in this
paper\footnote{The observable is a ratio between two \emph{decays} of one
production sample, and a dynamical scale recomputed on the generated final state
is not the same object on an undecayed and on a decayed event file.}
\OM{Deviation from the default announced in Sec.~\ref{sec:validation}: flag it
there too, as for the $m_{t\bar t}$ figures.}  For comparison we also generate
the SM contribution alone with the same settings.  The top decays are performed
in the \code{onshell} mode for the setup with spin correlations and in the
\code{none} mode for the setup without.  A comparison of the two simulations is
presented in Fig.~\ref{fig:valid_interf}, which shows the $\Delta\phi$
distribution between the two electrons from the $t\bar{t}$ decay, an observable
which is sensitive to spin-correlation effects (see
Sec.~\ref{subsec:validation-polarised} for more details).  For the interference
term the inclusion of spin correlations changes the shape of the distribution by
$+31\%$ at $\Delta\phi\to0$ and by $-30\%$ at $\Delta\phi\to\pi$, roughly twice
the corresponding effect in the SM ($+18\%$ and $-15\%$), and the correct
simulation of spin correlations for such measurements is therefore of critical
importance.
```

```latex
% --- applications.tex, the figure caption
\caption{$\Delta\phi(e^-e^+)$ of the electron and the positron from the
$t\bar{t}$ decay in the $pp\to t\bar{t}$ process, with spin correlations
(\code{spinmode=onshell}, solid) and without (\code{spinmode=none}, dashed), for
the amplitude-level interference of Eq.~(\ref{eq:CMDM_interf}) (blue) and for the
SM at LO (black).  Every curve is normalised to unit area, the normalisation of
the interference term being arbitrary; the lower panel gives, for each sample
separately, the ratio of its own \code{onshell} to its own \code{none}
prediction, i.e.\ the size of the spin-correlation effect.  The interference is
generated at \code{ctGRe=-1} ($\Lambda=1$~TeV), so the quantity plotted is
$-2\operatorname{Re}(\mcal^*_{\rm SM}\mcal_{tG})$.  Note that the two
\code{none} curves coincide to within $2\%$: without spin correlations this
observable retains almost no memory of the operator, so the entire separation
between the two solid curves is a spin-correlation effect.  Samples: $10^6$
events each, $\sqrt{s}=13$~TeV, NNPDF2.3LO, fixed $\mu_R=\mu_F=173$~GeV, no
cuts, $t\to W^+b\,(W^+\to e^+\nu_e)$ and $\bar t\to W^-\bar b\,(W^-\to
e^-\bar\nu_e)$.}
```

```latex
% --- validation.tex, after "...the default in \mgamc."
\OM{Two $t\bar t$ studies (Fig.~\ref{fig:valid_interf} and the $m_{t\bar t}$
panels) use a fixed $\mu_R=\mu_F=173$~GeV instead; this is stated where they
are described.}
```
