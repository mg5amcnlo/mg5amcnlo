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
