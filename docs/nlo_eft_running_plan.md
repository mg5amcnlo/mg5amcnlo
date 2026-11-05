# Plan: running of EFT couplings at NLO, and Figure 4 of 2212.05067 at NLO

Status: 2026-10-19. Part A (NLO support) implemented and validated, see Section 2.
Part B was first understood as "Table 4" of the paper; the request was corrected to
**Figure 4** (Section 3). The Table 4 work was stopped, what exists is in Section 4.

## 1. Goal

1. Make the event-by-event running of the SMEFT Wilson coefficients (the
   `running.py` mechanism, supported in `output madevent` since PR #302) work in
   the aMC@NLO output (`output <dir>` of a `[QCD]` process): fixed EFT scale and
   dynamical EFT scale mu = H_T/2.
2. Redo Figure 4 of arXiv:2212.05067 (Aoude, Maltoni, Mattelaer, Severi,
   Vryonidou) at NLO QCD, in the style of the MG7 paper (`MG7_paper`, plot
   template `plotexample/dummyplot.py`, `plots/<name>/{reduce.py, make_plots.py,
   PROVENANCE.md, data/}`).

Figure 4 of that paper (`fig:mtt_scales`, `figs/mtt_DIM64F_4.png` and `_12.png`):
d sigma / d m_tt of the linear interference at 13 TeV for c_tq^(8) = 1 (colour
octet, top) and c_Qu^(1) = 1 (colour singlet, bottom) at mu0 = 2 TeV, Lambda = 2 TeV,
mu_R = mu_F = H_T/2, in the bins 300-380-470-620-820-1100-1500-2500 GeV, for no
running, mu = H_T/2 and mu = m_t, with a ratio panel (mu = H_T/2 over mu = m_t). LO QCD
and LO EW. For the colour singlet the EW-induced interference dominates.

## 2. What exists at NLO today (verified on this branch, 2026-10-19)

Verified: `import model SMEFTatNLO_running-<restriction>` (copy of
`~/Desktop/UFOMODEL/SMEFTatNLO_running`, restriction that keeps the 19
`DIM64F` coefficients at non-0/1 values and zeroes the other blocks),
`generate p p > t t~ QED=0 NP^2<=2 [QCD]`, `output <dir>` works in 40 s
(11 Born, 302 virtual diagrams; NLO only accepts `<=` squared-order
constraints, `NP^2==2` is refused). The directory contains `Source/RUNNING`,
`C_RUNNING_k` / `MP_C_RUNNING_k` in `Source/MODEL/couplings.f`.

**Result of the check: not supported.** The NLO output of a model with
`running.py` does not compile: the generated `C_RUNNING_k` / `MP_C_RUNNING_k`
(`template_running_*` in `export_v4.py`) include `../vector.inc`, which exists in a
madevent output only (`gfortran: Cannot open included file '../vector.inc'`
at `couplings.f` while building `libmodel`). Fixed (G0 below); everything
else below was found while making it run.

Already there (scaffolding, never exercised at NLO):

* the model-side design is right: the one-loop UV counterterms of
  `SMEFTatNLO_running` (`CT_couplings.py`) carry `reglog(muprime**2/MU_R**2)`
  times the anomalous dimension, i.e. the NLO amplitude is expressed with
  c(mu') defined at the EFT scale `muprime` (UFO parameter, `Block LOOP 2`),
  independently of the loop scale `MU_R`. Hence mu' and mu_R can be chosen
  independently, which is the setting of the paper (mu_R = H_T/2, mu' = m_t or
  H_T/2 or mu0);
* `export_v4.py` (`UFOModelConverter...`, around line 11850) writes
  `UPDATE_AS_PARAM` with the two branches (fixed: `MDL_MUPRIME = mue_ref_fixed`
  once, `Gother = g(alpha_s(mue_ref_fixed))`; dynamical: `MDL_MUPRIME =
  mue_over_ref*MODEL_SCALE`, `Gother = G`) and `export_fks.py:247` copies
  `Template/Running` into `Source/RUNNING`;
* `RunCardNLO` has `fixed_extra_scale` (hidden, **system=True, forced True**),
  `mue_ref_fixed` (hidden, shortcut mz/mw/mt/mh), `mue_over_ref` (hidden,
  **system=True, forced 1**). Comment in the code: "running from Ellis-Sexton
  scale not implemented". Only `mue_ref_fixed` is written in the run card.
* a disabled test `notest_eft_running_nlo` (`tests/acceptance_tests/test_cmd_amcatnlo.py:599`)
  whose model directory `SMEFTatNLO_running-NLO` does not exist: **fixed-scale
  running at NLO was never validated**.

Gaps found (status 2026-10-19; all changes uncommitted in this worktree):

| # | gap | where | status |
|---|-----|-------|--------|
| G0 | NLO build fails: `include '../vector.inc'` in the generated running routines (madevent-only file). Conditional on `self.vector_size`, madevent output unchanged. | `export_v4.py` `write_one_running_block` | done |
| G0b | ALOHA: for long (>= 50 characters) combined amplitude names `combine_name` hashed the string *with* the `%(tags)s` placeholder, so the call never matched the routine written by ALOHA (`ld: _aloha_m... not found`). Hits every NLO output with large combined Lorentz structures (plain SMEFT@NLO too); a regression of the `%(tags)s` fix of the FD gauge. | `aloha_writers.py` `myHash`; unit test `CombineNameHashTest` | done |
| G1 | `MODEL_SCALE` was never set outside `UPDATE_AS_PARAM2`. NLO: set in `set_ren_scale`; LO: in `update_scale_coupling_vec` and for a fixed scale in `cuts.f` (harmless before, `mue_over_ref != 1` was broken). | `setscales.f`, `Template/LO/SubProcesses/{reweight,cuts}.f` | done |
| G2 | `MP_UPDATE_AS_PARAM` (quadruple precision rescue) evolved with `G` whatever the extra-scale mode. Now follows the double-precision branch through the common `/to_gother_mp/`. | `export_v4.py` | done (not exercised: needs an unstable point) |
| G3 | Run card: `fixed_extra_scale` and `mue_over_ref` no longer `system`, the three keys are written. | `banner.py` | done |
| G3b | Two `set.union(...)` calls whose result was dropped (`import_ufo.py` `track_dependant`, `export_v4.py` `PS_dependent_key`) removed `MU_R` (and `aEWM1`) from the per-event dependency lists as soon as the model has `running_elements`: `MDL_MU_R__EXP__2 = MU_R**2` was evaluated once, at card-reading time, in all counterterm couplings `reglog(.../MU_R**2)`. | `import_ufo.py`, `export_v4.py` | done, to be validated by V6 |
| G3c | `MDL_MUPRIME__EXP__2` was stale too: its dependency key is the UFO name `muprime` while the exporter used `mdl_muprime`; and `UPDATE_AS_PARAM` set `MDL_MUPRIME` after `intparam_definition.inc`. Both names are now tracked and `MDL_MUPRIME` is set before the include. Without this the NLO counterterm saw mu' = 172.5 GeV whatever `mue_ref_fixed` (first mu' scan: no cancellation, 15% LO vs 18% NLO spread). | `export_v4.py` | done, V3 rerun |
| G4 | Ellis-Sexton scale / scale reweighting: the virtual is evaluated at `MU_R = QES` and converted to mu_R with alpha_s beta-function terms only. Exact when mu' is held fixed (the finite counterterm makes the mu_R dependence alpha_s-like), not when mu' is tied to mu_R (dynamical scenario). Central scales: nothing to do. `UPDATE_AS_PARAM2` (reweighting) makes mu' follow mu_R. | `reweight_xsec.f` | open: not used here (`reweight_scale = False`); to be refused or documented |
| G5 | Cached Born amplitudes / `gscale` rescaling assume integer powers of the coupling. | `Template/NLO/SubProcesses` | not exercised (runs at one scale per contribution give consistent LO/NLO numbers); to be checked |
| G6 | `FIRST` logic of the fixed branch: `C0` saved at the first call; one executable per run, fine. | `couplings.f` | no action |
| G7 | MC@NLO / shower modes. | - | not needed for the table, untested |

### Validation of the NLO support

* V1 LO consistency (done): the Born of the NLO output (`LO` fixed-order mode, interference only, `#SelectedCouplingOrders`) vs the madevent LO, cQu8 = 1 at mu0 = 2 TeV, 13 TeV, NNPDF4.0 NLO, mu_R = mu_F = H_T/2: no running 1.050(2) vs 1.052(1) pb, fixed mu' = m_t 1.212(2) vs 1.213(1), dynamical 1.178(2) vs 1.179(1).
* V2 closure of the running vs an independent RGE solution: covered at LO by the madevent comparison (the madevent running was closed against an independent RGE to 0.15% in the MG7 paper work).
* V3 mu' independence (physical check of the counterterm): NLO interference with c(mu') for mu' = 2 TeV, 1 TeV, 500, 172.5 GeV, and dynamical, for cQu8 (octet), cQu1 (singlet), cQq83; flat up to O(alpha_s^2) expected. First scan, *before* G3b/G3c: LO 1.045 -> 1.206 pb (+15%), NLO 1.296 -> 1.527 pb (+18%) for cQu8, no cancellation (the counterterm saw a frozen mu'). Rerun in progress.
* V4 dynamical scenario is consistent with the fixed ones (interpolates, as at LO).
* V5 acceptance test replacing `notest_eft_running_nlo` (`test_eft_running_nlo`: run-card keys, generated code, Born interference differs between extra scales), ~9 min; see the commit message for its status.
* V6 SM sector of the running model at NLO vs `loop_sm` (`p p > t t~ [QCD]`, same cards): checks the effect of G3b on the SM counterterms.

## 3. Figure 4 at NLO: definition, pipeline, result (done 2026-10-19)

* Processes (persistent run directory `~/Documents/git_workspace/eft_nlo_run/`, outside /tmp which is
  cleaned), model `SMEFTatNLO_running` restricted to the 19 four-fermion coefficients
  (`restrict_ttbar4F.dat`, Q = Lambda = 2 TeV):
  * QCD-induced interference, LO and NLO QCD: `generate p p > t t~ QED=0 NP^2<=4 [QCD]`
    (orders (2,2,0) + (2,4,0));
  * EW-induced interference, **LO only**: `generate p p > t t~ QED^2<=2 NP^2<=2 [QCD]` (order
    (2,0,2)). The NLO QCD correction (2,2,2) fails the pole check (single pole, cause not understood);
    one process with both is impossible (14 split orders > 10 grids of the integrator, now 16).
  NLO accepts only `<=` constraints; the interference alone is integrated with the
  `#SelectedCouplingOrders` entry of `Cards/FKS_params.dat` (without it the SM dominates the variance).
* Scenarios as in the paper: no running (`mue_ref_fixed = 2000`), fixed `172.5`, dynamical
  (`fixed_extra_scale = False`, `mue_over_ref = 1`); `dynamical_scale_choice = 3` (H_T/2) for mu_R and
  mu_F, NNPDF4.0 NLO at both orders, m_t = 172.5.
* Analysis `docs/eft_running_nlo/analysis_HwU_ttbar_eft.f` (5 GeV bins of m_tt); tail statistics from
  sliced runs (`mxx_min_pdg`, m_tt > 620, 820, 1100, 1500 GeV). 23 bins from 345 to 2500 GeV (the paper has 7).
* Result: NLO interference flat in the EFT scale (c_tq^(8): 2.04/2.07/2.05 pb for no running, m_t, H_T/2
  against 1.78/2.07/2.01 at LO; c_Qu^(1): 0.252/0.253/0.240 against 0.088/0.341/0.292 pb); ratio
  dynamical/fixed 0.97-1.05 (octet) at NLO against down to 0.89 at LO, 0.92-0.97 below 720 GeV (singlet)
  against 0.48 at large mass at LO.
* Files: `MG7_paper/.claude/worktrees/eft-running-nlo-table/plots/eft_running_nlo/` (`reduce.py`,
  `make_plots.py`, `generation/`, `data/fig4.json`, `PROVENANCE.md`) and `sub_bsm.tex` (figure
  `fig_bsm_running_nlo` and text; not committed).

## 4. What exists for Table 4 (stopped at the request of the user, 2026-10-19)

* Fit side (`MG7_paper/.claude/worktrees/eft-running-nlo-table/plots/eft_running_nlo/fit/`): 26 data
  points, chi2 and interval code, HwU converter, README with all sources. The public inputs do not
  reproduce the published "No running" intervals: the quadratic ones come out 1.7-1.9 times narrower,
  driven by the CMS 13 TeV dilepton m_tt tail; without that data set all quadratic edges are within 20%.
* NLO side: no NLO predictions for the 14 operators; the mu' scan of the QCD-induced interference
  (cQu8, cQu1, cQq83) shows the NLO flat to ~1% in mu' (3-7% for cQu1) against 8-15% (a factor 4 for cQu1) at LO.
* A faithful NLO Table 4 needs the EW-induced interference (dominant for the colour singlets) at NLO QCD,
  the A_C observables and the fit inputs of the authors.
