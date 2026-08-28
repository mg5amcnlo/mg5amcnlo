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

## The SM NLO sample: a real defect, fixed, and its size measured

**`sm_nlo` was once decayed with its spin-density matrices evaluated at the
model default parameters instead of the run's own card.** MadSpin's density path
initialised its standalone matrix-element library from
`<madspin_me>/Cards/param_card.dat` — the card `output standalone` wrote from
the model defaults — and never from the run's card (the same mechanism as *Where
the Wilson coefficient is set, and why it is not the param_card*, below). Task
T113 found, for `PROC_sm_nlo`:

> the run used `MT 172.76 / WT 1.33 / WZ 2.4952 / WW 2.085` while its ME
> directories held `173 / 1.4915 / 2.4414 / 2.0476`

— a top mass 0.24 GeV too high and a top width 12 % too large in the density
matrix.

**The defect is fixed** (`0a1007bc2`)**, and the sample committed here was
regenerated with the fix** (task T123, branch `claude/ms-smeft-fig5-nlo-redo`),
reusing the same 500k production events at the same MadSpin seeds — so the
old/new comparison is exact rather than Monte-Carlo noise. **The measured effect
on this observable is nil:**

| | old | new | change |
|---|---|---|---|
| cross section | `10.9004361 pb` | `10.9004235 pb` | `-0.0001 %`, against a `0.020 pb` integration error |
| `Delta phi` shape | | | below the reseeding noise floor |

"Below the noise floor" is a measurement and not a hope. Buggy against fixed
moves the $\Delta\phi$ forward–backward asymmetry by `+0.0076`; changing *only*
the MadSpin seed, 42 → 99, with the fixed code moves it by `-0.0050`; buggy
against that seed-99 run, by `+0.0025`. The 720-bin `chi2/ndf` is `1.002`,
`0.975` and `0.997` for the three pairings — all equally indistinguishable. The
defect's "signal" is the size of reseeding noise.

**Only `spinmode = onshell` was ever affected.** `none` builds no density matrix
— it has no `madspin_me` directory at all — so its histograms reproduced bit for
bit. On the paired figures (`F`, `G`–`N`) the dashed NLO curves did not move by
one bit; the solid ones moved by less than a reseed.

The full record — what was regenerated and what was not, the acceptance test,
and the control run — is under *Regeneration of `sm_nlo` (the param_card fix)*
at the foot of this file. This note also opens and closes every
`plots/numbers_E.txt` … `numbers_P.txt`. The figures themselves carry no text
beyond axis labels and legends, so a caption written from them must carry it
across by hand.

### The two LO samples are clear (audited here, T113's audit having stopped)

The status of `sm_lo` was left unresolved when the audit task was stopped. It is
settled here, cheaply, by running T113's own `audit_me_param_cards.py` over this
directory's `--workdir`. The result, for all six decayed runs **as originally
generated**:

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
of `E`, and pane 1's denominator — were never in question.** On the regenerated
`sm_nlo` the audit reports `MATCH` for every entry, `aS` included, so the
`alpha_s` mismatch is gone there too; it survives only on the two LO samples,
where it cancels.

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
python3 plot_smeft_fig5_varF.py       # both styles, variation F only
python3 plot_smeft_fig5_ctg_scan.py   # both styles, variations G-N (the c_tG scan)
python3 plot_smeft_fig5_final.py      # both styles, variations O and P (the FINAL figure)
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

### The sixteen variations

**`O` is the final figure.**  `A`–`N` are the record of how it got there.

| tag | curves | ratio pane(s) | what it adds |
|---|---|---|---|
| `A` | EFT `onshell`, EFT `none`, SM LO `onshell` | `onshell/none` | the original figure plus an SM reference |
| `B` | `A` + SM LO `none` | `onshell/none` | **recommended** — the two `none` curves land on top of each other |
| `C` | EFT both + SM NLO both | `onshell/none` | the same at NLO |
| `D` | EFT both + SM LO both + SM NLO both | `onshell/none` | LO and NLO together |
| `E` | EFT `onshell`, SM LO `onshell`, SM NLO `onshell`, **one** `none` | **two**: shape ratios, then SM + interference | three orders on one pane, and what the operator does to the SM prediction; no text on the plot |
| `F` | all six: three samples × both spinmodes | **two**, four curves each: `E`'s ratios in *both* spinmodes | `E` decomposed — the dashed curve is the non-spin part of each ratio and the solid/dashed gap is the spin-correlation effect |
| `G`–`J` | as `F` (identical upper pane) | `F`'s, `shape` convention, at `c_tG = -1, +1, +10, -10` | the Wilson-coefficient scan with the rate change divided out; pane 1 is `c_tG`-invariant and is the same in all four |
| `K`–`N` | as `F` (identical upper pane) | same four `c_tG`, `rate` convention: ratios of *unnormalised* `dsigma` | the same scan with the interference at its physical sign and size; **pane 1 scans here**, and `K`/`L` are the two pane-1 versions |
| **`O`** | **`H`'s upper pane and pane 1 unchanged; pane 2 `onshell` only, two curves** | **`shape` at `c_tG = +1`; `onshell` named once in the corner** | **the FINAL figure — `H` with pane 2 cut to the two physical curves, no units on the axes and decimal ticks** |
| `P` | as `O`, plus a third pane-2 curve | `O`'s, with `(NLO + k·int)/NLO`, `k = n_NLO/n_LO` | the proposal: the only definition of the K-factor whose curve is not already on the pane |

`A`–`D` come from `plot_smeft_fig5.py` and `plot_smeft_fig5_userstyle.py`; `E`
from `plot_smeft_fig5_varE.py`; `F` from `plot_smeft_fig5_varF.py`, which
imports all three and modifies none of them; `G`–`N` from
`plot_smeft_fig5_ctg_scan.py`, which imports all four and modifies none of them.  `O` and `P` come from `plot_smeft_fig5_final.py`, which imports all five and
modifies none of them, so `A`–`N` are byte-identical.
Each variation is an addition, none of the scripts modifying the ones it
imports. The earlier "`A`–`F` stay byte-identical" rule no longer holds and was
never about the scripts: the `sm_nlo` regeneration changed the histogram file,
so every figure carrying an NLO curve — `C`, `D`, `E`, `F` and all of `G`–`N` —
is redrawn and differs in its bytes. `A` and `B` carry no NLO curve and are
**pixel-identical** to what they were, in both styles. What the redraw changed
numerically is a per-bin shift of at most 3.5 % on the `sm_nlo` `onshell` shape
(median 1.0 %, `chi2/ndf` 0.91 on the plotted 20 bins) — see the section on the
SM NLO sample above.

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
prediction. Both need `sm_nlo`; that sample was regenerated with the
`param_card` fix and the change is immaterial — see *The SM NLO sample* at the
top of this file.

The measured ends of the four ratio curves, on the plotted 20-bin binning:

| curve | first bin | last bin |
|---|---|---|
| `NLO / LO` (pane 1) | **+4.11 % ± 1.45** | **−4.11 % ± 1.05** |
| `SMEFT / LO` (pane 1) | **+11.54 % ± 0.79** | **−17.76 % ± 0.46** |
| `(LO + SMEFT) / LO` (pane 2) | **+2.49 % ± 0.17** | **−3.83 % ± 0.10** |
| `(NLO + SMEFT) / NLO` (pane 2) | **+1.09 % ± 0.23** | **−2.18 % ± 0.15** |

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
\OM{Samples: $10^6$ events each at LO, $5\times10^5$ at NLO.}}
```

#### One style note

The user-style rendering extends the ratio-limit ladder it imports with
intermediate rungs (`±2 %`, `±5 %`, `±8 %`, `±20 %`), locally and in
`plot_smeft_fig5_varE.py` only: the user's own ladder jumps from `±1 %` to
`±15 %`, which was fine for `B`'s `±30 %` pane but would draw pane 2's `±2.5 %`
as a flat line.

### Variation `F` — `E`, decomposed

`E` with **every curve doubled**: every quantity is drawn once from the
`onshell` samples and once from the `none` ones.
`plot_smeft_fig5_varF.py --help`; same outputs as `E` under the `F` tag
(`smeft_fig5_F.pdf`/`.png`, `smeft_fig5_F_curves.npz`, `numbers_F.txt`, in both
plot directories) and the same `--check-minus` on the MG7 PDF. Same rule on
text: **axis labels and legends only.**

* Upper pane: all six curves. This undoes `E`'s single-`none` compromise, and it
  dissolves the problem `E` had to solve by narrowing its legend claim — with
  all three `none` curves drawn, nothing is implied about a sample that is not
  on the pane, so the legend entries are plain `sample, spinmode`.
* Pane 1: `NLO/LO` and `SMEFT/LO`, each in both spinmodes.
* Pane 2: `(LO+SMEFT)/LO` and `(NLO+SMEFT)/NLO`, each in both spinmodes.

#### Encoding

**Line style is the spinmode** (solid `onshell`, dashed `none`) and **colour is
the quantity** — the same rule in all three panes, and the one `A`–`E` already
use. In the upper pane the quantity is the sample (blue SMEFT, black SM LO, red
SM NLO, as in `D`); in the ratio panes it is the ratio, coloured by the sample
that is not the LO reference (pane 1) or by the SM sample being corrected (pane
2). So a reader pairs curves by colour and reads the vertical gap *within* a
colour as the spin-correlation effect. In the user style the marker fill carries
the spinmode instead (filled `onshell`, open `none`), as it does in `A`–`E`
there.

#### What it shows: the decomposition

Each ratio splits into the dashed curve — the part that is **not** a
spin-correlation effect — and the gap up to the solid curve, which is the
spin-correlation effect itself. The two pane-1 pairs then say opposite things,
and the contrast is the pane's whole point.

| curve | first bin | last bin | max \|dev\| | mean |
|---|---|---|---|---|
| `SMEFT/LO`, `onshell` | +11.54 % ± 0.79 | −17.76 % ± 0.46 | 17.76 % | +2.08 % |
| **`SMEFT/LO`, `none`** | **+0.53 % ± 0.80** | **−0.58 % ± 0.49** | **2.06 %** | **+0.24 %** |
| `NLO/LO`, `onshell` | +4.11 % ± 1.45 | −4.11 % ± 1.05 | 7.46 % | +0.92 % |
| **`NLO/LO`, `none`** | **+7.02 % ± 1.60** | **−6.75 % ± 0.95** | **9.59 %** | **+1.47 %** |
| `(LO+SMEFT)/LO`, `onshell` | +2.49 % ± 0.17 | −3.83 % ± 0.10 | 3.83 % | +0.45 % |
| **`(LO+SMEFT)/LO`, `none`** | **+0.11 % ± 0.17** | **−0.13 % ± 0.11** | **0.44 %** | **+0.05 %** |
| `(NLO+SMEFT)/NLO`, `onshell` | +1.09 % ± 0.23 | −2.18 % ± 0.15 | 2.18 % | +0.15 % |
| **`(NLO+SMEFT)/NLO`, `none`** | **−0.93 % ± 0.22** | **+1.01 % ± 0.17** | **1.22 %** | **−0.16 %** |

Both predictions the figure was drawn to test come out right:

1. **`SMEFT/LO` in the `none` case collapses onto 1** — max 2.1 %, mean 0.2 %,
   ends +0.5 % / −0.6 %, against an `onshell` curve running +11.5 % to −17.8 %.
   So **the entire `SMEFT/LO` structure in this observable is spin
   correlation.** This is the strongest form of what `B` was recommended for:
   `B` showed the two `none` curves lying on top of each other, `F` shows their
   *ratio* pinned flat at 1 underneath an `onshell` ratio that spans a factor
   1.36. Note this is the same measurement seen twice — the flat blue dashed
   curve *is* the ratio whose max deviation is the 2.1 % quoted in `E`'s
   section — so it is a consistency check, not new information.

2. **`NLO/LO` in the `none` case does not collapse** — 9.59 % max, which is the
   8.75 % of the `SM LO none / SM NLO none` comparison the other way round
   (`1/(1−0.0875) − 1 = 0.0959`). The LO/NLO difference is a radiation effect on
   the $t\bar t$ boost and survives switching the spin correlations off.

   A detail worth stating because it is the opposite of what one might guess:
   the `none` curve is **larger** than the `onshell` one (9.6 % against 7.5 %),
   so the spin correlations partly *mask* the LO/NLO shape difference rather
   than cause it.

Pane 2 inherits both statements, shrunk by `w/(1+w)`: its LO `none` curve is
flat to 0.4 %, while its NLO `none` curve keeps a 1.2 % structure of the
opposite sign to the `onshell` one.

#### The `none` curves' weights, measured rather than assumed

Pane 2 needs `w = sigma_int/sigma_SM`, and the `none` curves take it from the
`none` samples' own cross sections. `check_weights()` runs before anything is
drawn and refuses to draw if they disagree by more than `1e-3`:

| | `onshell` | `none` | ratio−1 |
|---|---|---|---|
| `sigma(eft_int)` | 1.973865 | 1.974740 | −4.427e−4 |
| `sigma(sm_lo)` | 7.175290 | 7.178468 | −4.428e−4 |
| `sigma(sm_nlo)` | 10.900424 | 10.905950 | −5.067e−4 |
| `w(sm_lo)` | 0.275092 | 0.275092 | **−6.3e−8** |
| `w(sm_nlo)` | 0.181082 | 0.181070 | **−6.4e−5** |

This confirms T110's `5e-4`, and the LO weight does better still: `eft_int` and
`sm_lo` move between the spinmodes by the *same* 4.428e−4, which cancels in
their ratio. Using the `onshell` weights throughout would have changed nothing
visible — but that would have been an assumption, and it is now a measurement.

#### Caption for `F`

Same content as `E`'s caption, with the decomposition added. The parameter point
are on the same footing as for `E`: they are **not** on the figure and must come
from here.

```latex
\caption{As Fig.~\ref{fig:valid_interf}, with every curve shown both with spin
correlations (\code{spinmode=onshell}, solid) and without (\code{spinmode=none},
dashed): the amplitude-level interference of Eq.~(\ref{eq:CMDM_interf}) (blue),
the SM at LO (black) and the SM at NLO (red).  Every curve is normalised to unit
area; the interference is generated at \code{ctGRe=-1} with $\Lambda=1$~TeV, so
the quantity plotted is $-2\operatorname{Re}(\mcal^*_{\rm SM}\mcal_{tG})$.
Middle panel: ratios of the unit-area shapes to the SM LO one --- a \emph{shape}
ratio, not a $K$-factor, the rates having been divided out (the LO$\to$NLO
$K$-factor is $1.52$ and is not shown).  Lower panel: the SM prediction with the
interference added, weighted by the samples' own cross sections
($\sigma_{\rm int}/\sigma_{\rm SM}=0.275$ at LO, $0.181$ at NLO), divided by the
SM alone; its size is proportional to $c_{tG}/\Lambda^2$ and all four curves
mirror about unity for $c_{tG}>0$.  In both lower panels the dashed curve is the
part of the ratio that is \emph{not} a spin-correlation effect and the gap up to
the solid curve of the same colour is the spin-correlation effect itself.  The
blue dashed curve is flat at unity to $2\%$ while the blue solid one spans
$+12\%$ to $-18\%$: the entire difference between the interference and the SM in
this observable is a spin-correlation effect.  The red pair behaves oppositely
--- the dashed curve carries a $10\%$ structure of its own --- because the
LO/NLO difference here is the extra radiation changing the $t\bar t$ boost, not
spin.}
```

> **Correction to one clause of `F`'s caption.** It says the pane-2 curves
> "mirror about unity for $c_{tG}>0$". The sign is right and the size is not.
> `F`'s pane 2 divides by the *total* of `SM + interference`, so its coefficient
> is `w/(1+w)`, not `w`: flipping `c_tG` flips `w` but also moves `1+w`, and the
> deviation at `+1` comes out **1.759x** (LO) and **1.442x** (NLO) the one at
> `-1`, not `1.000x`. The mirror is exact only in the `rate` convention of
> `K`–`N` below, whose coefficient is `w` itself. `F`'s own figure is drawn at
> `c_tG = -1` and is unaffected by the correction; only the extrapolation in
> that clause was wrong.

### Variations `G`–`N` — `F` scanned over the Wilson coefficient

`plot_smeft_fig5_ctg_scan.py --help`. Eight figures: `F`'s three panes at
`c_tG = -1, +1, +10, -10` in each of two ratio conventions. Both styles, PDF and
PNG, a `_curves.npz` and a `numbers_<TAG>.txt` per figure, `--check-minus` on
every MG7 PDF. It reads `data/histograms.npz` and `data/meta.json` and nothing
else — no cross section, weight or NLO number is hard-coded — so **regenerating
the SM NLO sample is a re-run of this script, not a rewrite of it**.

| tag | file stem | `c_tG` | convention |
|---|---|---|---|
| `G` | `smeft_fig5_G_ctg_m1_shape` | `-1` | `shape` |
| `H` | `smeft_fig5_H_ctg_p1_shape` | `+1` | `shape` |
| `I` | `smeft_fig5_I_ctg_p10_shape` | `+10` | `shape` |
| `J` | `smeft_fig5_J_ctg_m10_shape` | `-10` | `shape` |
| `K` | `smeft_fig5_K_ctg_m1_rate` | `-1` | `rate` |
| `L` | `smeft_fig5_L_ctg_p1_rate` | `+1` | `rate` |
| `M` | `smeft_fig5_M_ctg_p10_rate` | `+10` | `rate` |
| `N` | `smeft_fig5_N_ctg_m10_rate` | `-10` | `rate` |

The coefficient is **not written on any of them** — axis labels and legends
only, as everywhere else here. It is in the file name, in this table and at the
top *and* bottom of each `numbers_<TAG>.txt`. `G` reproduces `F` exactly.

#### Pane 1 cannot depend on `c_tG`, and that is why the eight are not what was asked for

The request was pane 2 at four coefficients crossed with **two versions of
pane 1**, one at `-1` and one at `+1`. That cross product does not exist.

The interference is linear in `c_tG`. `F`'s panes are built from unit-area
shapes, and

```
n_int = (dsigma_int/dphi) / sigma_int
```

carries the factor `c_tG/c_ref` in the numerator *and* in the denominator, where
it cancels — **sign included**, since a negative overall factor divides out of a
ratio. So `n_int` is exactly invariant. Consequences:

* the **upper pane is identical in all eight figures**, which is what was
  expected and is the reason it is;
* but `F`'s **pane 1 is identical too** — `n_NLO/n_LO` and `n_int/n_LO` contain
  no `c_tG` at all — and not merely at `+1` against `-1` but at *every* non-zero
  coefficient. Drawing it at `-1` and again at `+1` would have shipped one
  figure under two names.

`check_ctg_invariance()` measures this instead of asserting it: rescaling `sumw`
and `sumw2` by `c_tG/c_ref` and renormalising moves the upper-pane curves and
their error bars by at most `3.3e-16`, i.e. by nothing.

The coefficient enters the figure in exactly one place, the **signed weight**

```
w_SM(c_tG) = sigma_int(c_tG) / sigma_SM ,      w = -0.275092 c_tG  (LO)
                                               w = -0.181082 c_tG  (NLO)
```

(the minus is because the samples were generated at `ctGRe = -1`, where
`sigma_int` is *positive*). Any curve containing `w` scans; any curve not
containing `w` does not.

#### The convention

**The interference always enters a ratio pane with the physical sign and
magnitude it has at the stated `c_tG`, through `w`, and is never re-normalised
to positive unit area there.** `n_int` is unit-area in the *upper* pane only,
where the sign of its own integral divides out; that sign is not lost, it is
moved to the one place it changes an answer and carried there explicitly by `w`.
Both ratio panes of a given figure use the same convention.

`rate` (`K`–`N`) — every ratio between **unnormalised** differential cross
sections:

```
pane 1:  dsigma_NLO/dsigma_LO = K (n_NLO/n_LO),  K = sigma_NLO/sigma_LO = 1.519
         dsigma_int/dsigma_LO = w_LO(c_tG) (n_int/n_LO)
pane 2:  (dsigma_SM + dsigma_int)/dsigma_SM = 1 + w_SM(c_tG) rho_SM
```

No denominator can change sign, the rate change the operator makes is included,
and `c_tG = +1` **mirrors `c_tG = -1` exactly** about the no-interference line:
`w` flips and nothing else moves. Pane 1 scans here, so `K` and `L` *are* the
two genuinely different pane-1 versions that were wanted — pane 1's blue curve
runs `+0.226 … +0.308` at `-1` and `-0.308 … -0.226` at `+1`.

`shape` (`G`–`J`) — every ratio between **unit-area** curves, which is what the
upper pane's normalisation implies:

```
pane 1:  n_NLO/n_LO,  n_int/n_LO                    (c_tG-invariant)
pane 2:  n_(SM+int)/n_SM = 1 + [w/(1+w)] (rho_SM - 1)
```

This is `F`'s pane 2 exactly. It answers a different question — does the
*shape* move, with the rate change divided out — and it is the convention that
carries the sign trap (below).

Both use the same error identity, the one that stops the shared `n_SM` being
counted as two independent measurements. Writing either pane-2 curve as

```
baseline + k (rho_SM - 1),          rho_SM = n_int/n_SM
```

with `(baseline, k) = (1, w/(1+w))` for `shape` and `(1+w, w)` for `rate`, the
SM measurement appears once, inside `rho_SM`, and the error is
`|k| sigma(rho_SM)` in both. `rho_SM`'s own error adds `n_int`'s and `n_SM`'s in
quadrature; the samples are independent. `pane2_baseline_*` and `pane2_k_*` are
in each `_curves.npz` beside `pane2_w_*`.

#### The weights, and where the sum stops being physics

| `c_tG` | `w` (LO) | `1 + w` (LO) | `w` (NLO) | `1 + w` (NLO) | bins with `dsigma < 0` | verdict |
|---|---|---|---|---|---|---|
| `-1` | `+0.2751` | `1.2751` | `+0.1811` | `1.1811` | 0 / 20 | sound |
| `+1` | `-0.2751` | `0.7249` | `-0.1811` | `0.8189` | 0 / 20 | sound |
| `-10` | `+2.7509` | `3.7509` | `+1.8108` | `2.8108` | 0 / 20 | **interference bigger than the SM, outside EFT validity** |
| `+10` | `-2.7509` | `-1.7509` | `-1.8108` | `-0.8108` | **20 / 20** | **negative `dsigma`, outside EFT validity** |

(`onshell`; the `none` weights differ in the sixth digit and are tabulated in
each `numbers_*.txt`.) So:

* `G`, `H`, `K`, `L` are physics: a 28 % (LO) / 18 % (NLO) correction on the
  rate, of either sign.
* `J` and `N` (`c_tG = -10`) stay positive everywhere, so nothing *looks*
  broken — but the dimension-six interference is now 2.75 (LO) and 1.81 (NLO)
  times the entire SM cross section. A "correction" three times the size of what
  it corrects is not one; the dimension-six **squared** term that was dropped is
  of the same order or larger. `N`'s pane 2 sits at `3.26 … 4.08` (LO).
* `I` and `M` (`c_tG = +10`) have `SM + interference` **negative in all 20 bins,
  at both LO and NLO**. `M` draws that as it comes out — pane 2 lies entirely
  below zero, `-2.08 … -1.26` (LO) and `-0.95 … -0.59` (NLO) — with **no
  clipping and no floor**. That is the honest picture of a failed expansion.

#### The sign trap, which is `I`

The `shape` convention divides by `1 + w`, the *total* of `SM + interference`
relative to the SM. That vanishes and then changes sign at

```
c_tG = +3.635  (LO)        c_tG = +5.522  (NLO)
```

— the coefficients at which the interference exactly cancels the SM cross
section. **`c_tG = +10` is past both poles**, so `I`'s two pane-2 curves are
ratios of two *negative* densities. They come out positive, near 1, spanning
`0.72 … 1.19`: a picture indistinguishable from a modest 20 % shape distortion,
drawn from a differential cross section that is negative in every bin. This is
exactly the arbitrary-sign normalisation this figure has been avoiding since
`ctGRe = -1` was chosen over `+1`. **Read `M` instead at that coefficient**, and
treat `I` as the illustration of why the `shape` convention needs its pole
quoted. `numbers_I.txt` says so at the top and at the bottom.

#### Caption for `G`–`N`

`F`'s caption, with three substitutions and one addition. The `c_tG` value and
the convention are **not** on the figure and must come from here.

```latex
\caption{As Fig.~\ref{fig:valid_interf_decomposed} (variation \texttt{F}), drawn
at $c_{tG}=\VALUE$ with $\Lambda=1$~TeV.  The upper panel is independent of
$c_{tG}$: the interference is linear in the coefficient, so the factor cancels
between $\rd\sigma_{\rm int}$ and $\sigma_{\rm int}$ in the unit-area
normalisation, sign included.  Middle panel: \CONVENTIONSENTENCE.  Lower panel:
the SM prediction with the interference added at its physical sign and size,
$w=\sigma_{\rm int}(c_{tG})/\sigma_{\rm SM}=\WVALUES$ at LO and NLO, divided by
the SM alone.  In both lower panels the dashed curve is the part of the ratio
that is \emph{not} a spin-correlation effect and the gap up to the solid curve
of the same colour is the spin-correlation effect itself.}
```

with, for `I`, `J`, `M` and `N`, this sentence added and **not** optional:

```latex
At $|c_{tG}|=10$ the dimension-six interference exceeds the SM cross section it
corrects, so the dropped dimension-six squared contribution is of the same order
or larger and the linear truncation shown here is not a prediction; at
$c_{tG}=+10$ the sum is negative in every bin and is plotted unclipped.
```

### Variations `O` and `P` — the final figure

**`O` (`smeft_fig5_O_ctg_p1_shape_final`) is the figure for the paper.** It is
`H` — the `shape` convention at `c_tG = +1` — with three changes, all of them in
pane 2 or on the axes. `plot_smeft_fig5_final.py --help`; both styles, PDF and
PNG, a `_curves.npz` and a `numbers_<TAG>.txt` per figure, `--check-minus` on
the MG7 PDFs. It imports the five scripts that draw `A`–`N` and modifies none of
them, so those fourteen stay byte-identical.

| | `H` | `O` |
|---|---|---|
| upper pane | six curves | **unchanged, curve for curve** |
| pane 1 | four curves | **unchanged, curve for curve** |
| pane 2 | four curves, both spinmodes | **two curves, `onshell` only** |
| pane-2 legend | spinmode per entry | spinmode **once, in the corner** |
| `Delta phi` axis | `[rad]` | **no unit** |
| upper vertical axis | `[rad^-1]` | **no unit** |
| x ticks | `0, pi/4, ... pi` | **`0, 0.5, ... 3.0`, decimal, on an axis running to `pi`** |

#### Pane 2: the `none` curves are gone

Four lines become two, both `onshell`: `(LO + SMEFT)/LO` and
`(NLO + SMEFT)/NLO`. The `shape` convention and the `w/(1+w)` error identity are
`H`'s, unchanged.

| pane-2 curve | first bin | last bin |
|---|---|---|
| `(LO + SMEFT)/LO`, `onshell` | **−4.38 % ± 0.30** | **+6.74 % ± 0.18** |
| `(NLO + SMEFT)/NLO`, `onshell` | **−1.58 % ± 0.33** | **+3.15 % ± 0.21** |

(These are `G`'s numbers mirrored and amplified: `H` is at `c_tG = +1` where `w`
flips sign, and in the `shape` convention `1 + w` moves too, so the deviation is
1.759x (LO) and 1.442x (NLO) the one at `c_tG = -1` — see the correction to
`F`'s caption above.)

#### `onshell` in the corner: the one exception to "no text on the figure"

The series' standing rule is axis labels and legends only. `O` breaks it once,
deliberately and by request: with every curve in pane 2 in one spinmode,
repeating `onshell` in both legend entries was pure repetition, so it is written
once in the pane's top-left corner and taken out of both entries. The upper pane
and pane 1 still carry both spinmodes and still name them per entry. Nothing
else is on the figure: no coefficient, no convention, no header, no title.

#### Units off the axes, decimal ticks on

`[rad]` is off the `Delta phi` axis and `[rad^-1]` off the upper pane's vertical
axis; the quantities are still radians and inverse radians and a caption must
say so. The x ticks are `0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0` with minor ticks every
`0.1`, on an axis that still runs to `pi = 3.14159` — so the last label is `3`
and the frame is a little past it. This matches the paper's other figures. It is
a **consistency** choice and not an improvement: the `pi/4` ladder of `A`–`N` is
the better-looking one and was given up on purpose.

#### The third pane-2 curve that was asked for, and why it is not drawn

`O` was asked for with a third pane-2 curve, `(NLO + K·int)/NLO` with `K` the
bin-by-bin `NLO/LO` K-factor — an estimate of the missing NLO interference, the
`O_tG` interference being available at LO only. **It is a curve already on the
pane, and it is not drawn.**

With `K(x) = dsigma_NLO(x)/dsigma_LO(x)` the numerator factorises:

```
dsigma_NLO + K dsigma_int = dsigma_NLO * [1 + dsigma_int/dsigma_LO]
```

so `dsigma_NLO` cancels and the ratio to `dsigma_NLO` is `1 + dsigma_int/dsigma_LO`
— which *is* `(LO + int)/LO`. In the `rate` convention of `K`–`N` that is an
exact identity; `check_k_degeneracy()` measures the difference and gets `0.0`,
not a small number.

`O` is in the `shape` convention, where the numerator and `n_NLO` are both
renormalised to unit area, so the cancellation leaves **one multiplicative
constant**:

```
curve3(x)  = [1 + w_LO rho_LO(x)] / Z ,        Z = 1 + w_LO <rho_LO>_NLO
curveLO(x) = [1 + w_LO rho_LO(x)] / (1 + w_LO)
```

`curve3/curveLO = (1 + w_LO)/Z`, with **no `x` in it**. Measured at
`c_tG = +1`, with `w_LO = -0.27509208` and `Z = 0.72382273`:

| | measured |
|---|---|
| `curve3/curveLO` | `1.0014992469` in **every one** of the 20 bins |
| bin-to-bin spread of that ratio | `2e-16` — the last bit of double precision |
| max separation of the two curves | `0.0016`, i.e. **0.91 of the plotted 1σ error bar** |

So the third curve is the LO curve rigidly rescaled by `+0.15 %`, by nothing but
the renormalisation of the K-weighted sum. It carries **no** information about
the shape, and drawing it would put two lines on top of each other and invite a
reader to measure a vertical gap that is a normalisation artefact.

Note the brief's own guess — that the two would differ "through `w_NLO` against
`w_LO` in the prefactor" — is not what happens. Both curves carry `w_LO`; the
whole difference is the normalisation constant `Z` against `1 + w_LO`.

This is not a defect of the calculation, it is what the ansatz *says*. "The
interference receives the same bin-by-bin multiplicative NLO correction as the
SM" is exactly "the NLO correction does not change the operator's relative
effect", and pane 2's LO curve is already that statement. The useful consequence
is a **re-reading of the curve that is there**: `(LO + SMEFT)/LO` may be quoted
as the NLO estimate under a bin-by-bin K-factor ansatz. That belongs in the
caption, not in a third colour.

#### What definition of `K` would carry information — variation `P`

Only the **x-dependence** of `K` can. Split it,

```
K(x) = <K> k(x) ,    <K> = sigma_NLO/sigma_LO = 1.519 ,   <k> = 1
```

and the two factors do different things. `<K>` multiplies the interference's
*rate*, which is algebraically the same knob as `c_tG` — `G`–`N` already scan
it, and in the `shape` convention it divides straight back out. Only
`k(x) = n_NLO(x)/n_LO(x)`, the K-factor with its inclusive size divided out,
moves the curve relative to the two that are drawn.

`P` (`smeft_fig5_P_ctg_p1_shape_kshape`) is `O` with that third curve added:

```
(NLO + k int)/NLO = [1 + w_NLO rho_LO(x)] / Z' ,   Z' = 1 + w_NLO <rho_LO>_NLO
```

— the interference keeps its LO **rate** but is given the NLO **shape**
distortion. It is genuinely a third curve:

| | measured |
|---|---|
| ends | **−2.47 % ± 0.18** … **+4.02 % ± 0.10** |
| distance from `(NLO + SMEFT)/NLO` | **6.35 σ** at its furthest bin |
| distance from `(LO + SMEFT)/LO` | **15.54 σ** |
| between the two | in **17** of the 20 bins |

Pane 2 of `P` has one spinmode, so line style is free there and carries the K
ansatz instead (dash-dot in the MG7 style, open markers in the user style). That
is a local exception, stated in `numbers_P.txt`; `O` does not use it. **`P` is
the proposal, `O` is the figure.**

Both statements are measured by `check_k_degeneracy()`, which runs before
anything is drawn, refuses to draw if either fails, and writes its table into
`numbers_O.txt` and `numbers_P.txt`.

#### Caption for `O`

`H`'s caption, with the convention and coefficient carried across as always, and
two things added: the axes' missing units, and the K-factor re-reading of the LO
curve.

```latex
\caption{$\Delta\phi(e^-e^+)$ of the electron and the positron from the
$t\bar{t}$ decay in $pp\to t\bar{t}$ at $\sqrt{s}=13$~TeV, NNPDF2.3LO, fixed
$\mu_R=\mu_F=173$~GeV and no cuts, with $t\to W^+b\,(W^+\to e^+\nu_e)$ and
$\bar t\to W^-\bar b\,(W^-\to e^-\bar\nu_e)$.  $\Delta\phi$ is in radians and
the upper panel's ordinate in $\mathrm{rad}^{-1}$; the units are not repeated on
the axes.  Upper panel: every curve normalised to unit area, with spin
correlations (\code{spinmode=onshell}, solid) and without (\code{spinmode=none},
dashed), for the amplitude-level interference of Eq.~(\ref{eq:CMDM_interf})
(blue), the SM at LO (black) and the SM at NLO (red).  Middle panel: ratios of
the unit-area shapes to the SM LO one --- a \emph{shape} ratio and not a
$K$-factor, the rates having been divided out (the LO$\to$NLO $K$-factor of
these samples is $1.519$ and is not shown); the dashed curve is the part of the
ratio that is \emph{not} a spin-correlation effect and the gap up to the solid
curve of the same colour is the spin-correlation effect itself.  Lower panel:
the SM prediction with the interference added, weighted by the samples' own
cross sections, divided by the SM alone, at $c_{tG}=+1$ with $\Lambda=1$~TeV
($w=\sigma_{\rm int}/\sigma_{\rm SM}=-0.275$ at LO and $-0.181$ at NLO); it is
drawn from the \code{onshell} samples only, as marked in the panel, and its size
is proportional to $c_{tG}/\Lambda^2$.  Because the interference is available at
LO only, the LO curve of that panel is \emph{also} the NLO estimate obtained by
scaling the interference with the bin-by-bin $K$-factor
$\rd\sigma_{\rm NLO}/\rd\sigma_{\rm LO}$: that ansatz makes the multiplicative
NLO correction cancel between numerator and denominator, so it returns the LO
curve identically.  \OM{Samples: $10^6$ events each at LO, $5\times10^5$ at
NLO.}}
```

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
| SM NLO | +15.0 % ± 2.2 | −12.7 % ± 1.2 |

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

## Regeneration of `sm_nlo` (the param_card fix)

The `sm_nlo` sample committed here is **not** the one the first run produced.
It was regenerated on
`71ffba30a696f54a6939a59ce0b8fa6a4040a19e`, which merges
`0a1007bc25de8398a56a7fb25bbd8f92eaf88e3b` — *"MadSpin: evaluate the matrix
elements with the run's own `param_card`"* — into the sample definitions.

### What was wrong

MadSpin initialised its density matrix elements from the `param_card.dat` that
`output standalone` writes **from the model defaults**, not from the card the
input events carry. Every other sample here dodges this, because the SMEFT ones
put their Wilson coefficient in the model *restriction* precisely so that the
default and the run agree (see `setup.smeft_model.restriction_note`). The NLO
sample could not: it sets `MT`/`WT`/`ymt`/`WW`/`WZ` on the `loop_sm` defaults at
launch. So its ME directories held

| | events | matrix elements |
|---|---|---|
| `MT` | 172.76 | 173.0 |
| `WT` | 1.33 | 1.4915 |
| `WZ` | 2.4952 | 2.441404 |
| `WW` | 2.085 | 2.0476 |
| `ymt` | 172.76 | 173.0 |

Top mass 0.24 GeV high, top width 12 % large.

### What was regenerated, and what was not

The defect is entirely inside MadSpin. The production run was always launched
with the right numbers — the banner `<slha>` of `PROC_sm_nlo` carries them — so
**the 500k production LHE was reused unchanged** (md5
`9730e92f1433841f3dc2ebbfdb83e2e1`) and only the decay stage was rerun, in both
spinmodes, at the same seeds. That makes the old/new comparison exact: same
events, same weights, same cross-section, so any difference is the fix and not
Monte-Carlo noise in the production.

`spinmode = none` needs no density matrix, so MadSpin never builds
`madspin_me/` for it and the defective card was never read. Re-running it
reproduced the old histograms **bit for bit**. Only the `onshell` curve was ever
affected.

### Acceptance test

`audit_me_param_cards.py` compares every numeric SLHA entry of the card each
matrix-element directory is initialised from against the card the run carries.
Pointed at the *original* run it reproduces the table above; pointed at the
regenerated one every directory reports `MATCH`, `aS` included — the fix copies
the whole card, so not even the (inert, PDF-supplied) `aS` differs any more.

```
audit_me_param_cards.py --workdir <regeneration workdir>
```

### What changed numerically: nothing that matters

* **Cross-section**: `10.9004361 -> 10.9004235 pb`, a shift of `1.3e-05 pb`
  (`-0.0001 %`) against a `0.020 pb` integration error. The `<init>` `XSECUP`
  is `10.899373 +- 0.020274 pb` before *and* after, to every digit the banner
  carries. `none` is unchanged exactly.
* **Shape**: it does not move by anything 500k events can resolve. A control
  run — the fixed code, the same production events, only the MadSpin seed
  changed from 42 to 99 — moves the `Delta phi` forward–backward asymmetry by
  `-0.0050`, while the *defect* moves it by `+0.0076`. The two are the same
  size. Against the seed-99 run the buggy sample differs by only `+0.0025`. The
  720-bin shape `chi2/ndf` is 1.002 (buggy vs fixed), 0.975 (buggy vs fixed,
  other seed) and 0.997 (fixed vs fixed): all three pairings are equally
  indistinguishable.

So the defect was **real but immaterial** at this statistics — which is a
different statement from "we fixed it", and the one to make. A 0.24 GeV top
mass and a 12 % top width in the *density matrix* leave the `Delta phi`
spin-correlation shape where it was. The sample is replaced because it is now
sound, not because the figure changes.

The control run is kept under
`samples.sm_nlo.regeneration.control_run.path` and is deliberately **not** in
`histograms.npz`: it is a systematics control, not a curve on the figure.

### The redraw, and what it moved

All fourteen variations were re-run against the regenerated `histograms.npz`
(blob `2d596a26`, previously `998a1efd`). What moved, and by how much:

(`O` and `P` were added afterwards and were only ever drawn from the
regenerated file, so they have no `before` to move from. Their pane 2 is
`onshell` only and is therefore built entirely from curves the defect could in
principle have touched; the measured move on the `c_tG = +1` `shape` pane 2 is
`0.83 %` at most, `1.8 sigma`, below the reseeding noise floor. The provenance
note opens and closes `numbers_O.txt` and `numbers_P.txt` as it does
`numbers_E.txt` … `numbers_N.txt`; the figures themselves carry none of it.)

| figures | max per-bin relative change | in units of the plotted error |
|---|---|---|
| `A`, `B` | **0** — no NLO curve; PNG pixel-identical in both styles | — |
| `sm_nlo onshell` shape (the primitive, all NLO figures) | **3.53 %** max, 0.98 % median | 1.9 sigma max, `chi2/ndf` 0.91 on 20 bins |
| pane 1 `NLO/LO` (`E`, `F`, `G`–`N`) | 3.53 % | 1.8 sigma |
| pane 2 at `c_tG = -1` / `+1` (`E`, `F`, `G`, `H`, `K`, `L`) | 0.56 % / 0.83 % | 1.8 sigma |
| pane 2 at `c_tG = +10` / `-10` (`I`, `J`, `M`, `N`) | 6.99 % / 2.26 % | 1.8 sigma |
| cross section, every figure | `-0.0001 %` | 0.0006 of the integration error |

The larger percentage at `|c_tG| = 10` is the `1/(1+w)` amplification of the
*same* shift — the sigma column is flat across the whole table, which is the
point. Everything moved by less than reseeding the MadSpin run (T123's control:
`Delta A_FB = -0.0050` for a seed change alone against `+0.0076` here), so
nothing exceeds the noise floor.

**The earlier "keep `A`–`F` byte-identical" rule is retired.** It was about
adding variations without disturbing the ones already there. A new sample
legitimately moves every figure that carries an NLO curve, and `C`, `D`, `E`,
`F` and all of `G`–`N` differ in their bytes and (slightly) in their pixels.
`A` and `B` carry no NLO curve and are pixel-identical; their PDFs differ only
in the embedded timestamp and matplotlib's random font-subset tag.

### Where the samples live

`setup.workdir` is a single scalar that stopped describing the study when
`sm_nlo` was regenerated elsewhere. It has **not** been silently repointed —
that would make it wrong for the other three samples. Instead `meta.json` now
carries `setup.workdirs`, one entry per sample with an explicit `durable` flag,
and `setup.workdir_note` says the scalar is deprecated as a pointer.

| sample | workdir | durable |
|---|---|---|
| `eft_int`, `eft_int_ctg_plus`, `sm_lo` | `/private/tmp/smeft_fig5_work` (19 GB) | **no** — under `/tmp` |
| `sm_nlo` | `/Users/omattelaer/Documents/madspin_validation_samples/t_fig5_sm_nlo_redo` | yes |

Copying 19 GB out of `/tmp` is neither cheap nor useful. What *is* cheap and
does make the study repeatable is the steering, so the 124 kB that matters —
`mg5_*.dat`, `PROC_*/Cards/{run,param,proc}_card.dat` and every
`MS_*/madspin_card.dat` — is mirrored to a durable
`setup.cards_archive.path`. With those plus `run_smeft_fig5.py` the samples can
be remade; without them a reaped `/tmp` would take the exact cards with it.

None of this is needed to redraw: **no plotting script reads any workdir.**
`data/histograms.npz` and `data/meta.json` are the whole input, and every figure
redraws with all four workdirs gone. `meta.json`'s `figures` block records which
histogram blob each figure set was drawn from, with the plotting environment.
