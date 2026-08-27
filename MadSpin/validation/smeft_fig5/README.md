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
python3 plot_smeft_fig5.py            # MG7 paper style  -> plots/
python3 plot_smeft_fig5_userstyle.py  # user's own style -> plots_userstyle/
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

### The four variations

| tag | curves | what it adds |
|---|---|---|
| `A` | EFT `onshell`, EFT `none`, SM LO `onshell` | the original figure plus an SM reference |
| `B` | `A` + SM LO `none` | **recommended** — the two `none` curves land on top of each other |
| `C` | EFT both + SM NLO both | the same at NLO |
| `D` | EFT both + SM LO both + SM NLO both | LO and NLO together |

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
