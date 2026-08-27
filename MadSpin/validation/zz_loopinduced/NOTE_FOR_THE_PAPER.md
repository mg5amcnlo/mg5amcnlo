# Note for the MadSpin-2 paper session — polarisation figures

**Short version: your `Z0Z0 / Z0ZT / ZTZ0 / ZTZT` figures are fine. One other
figure is not, and one sentence of prose needs a decision.**

## 1. The polarised-weight figures are validated, not suspect

An earlier note (section 9 of `SPIN_COEFFICIENTS.md`) claimed MadSpin's
`ms_pol_*` weights fail a closure test by 14-21 sigma. **That claim is
withdrawn.** The failure was in the validation study's own angular observable,
not in the weights. Re-checked on the same 250 000-event `p p > z z` sample:

```
reweight by ms_pol_23:0_23:0  ->  <cos^2 theta> = 0.2004 / 0.2005   (must be 1/5)
reweight by ms_pol_23:T_23:T  ->                  0.4008 / 0.3992   (must be 2/5)
             worst of the eight entries: 1.4 sigma
f_0 from the decay angles 0.17023 +- 0.00307   from the weights 0.17440 +- 0.00035
```

So the weights carry the decay correctly, project on the axis they claim
(`me_frame`, the `ZZ` rest frame on a `2 -> 2`), and the `[0, T]` partition sums
to the unpolarised total to `1.000705 +- 0.000395`. If you want a sentence for
the text, section 4 below has one.

`figures/NLO_Polarised/*` and Fig. `NLO_polarisations_application` in
`applications.tex` are built from these weights, against `M(e+ mu+)` and
`Delta phi(e+ e-)`. Neither observable constructs a decay-angle frame anywhere
(`pol_analysis.py` has no boost into a `Z` rest frame; the stored
`data/weights*.npz` carry no `cos_theta` array). **Nothing to change.**

## 2. `figures/LI_processes/cos1cos2.pdf` — the curves are damped

The `cos1cos2` figure of the loop-induced section (`applications.tex`,
Fig. `fig:cos1cos2_LI`) plots a variable that was built with the wrong boost
composition: the axis was taken in the four-lepton frame while the `l+` was
boosted into its pair's rest frame straight from the lab. Those two boosts are
not collinear and their composition carries a Wigner rotation — median 8
degrees on a comparable sample — which damps the distribution towards flat.

**What survives.** The figure's argument is untouched. Every curve on it, truth
included, was harvested with the same definition, so "the three spin-correlated
`spinmode`s sit on the reference and `none` does not" and every `chi2/ndf` in it
still hold. `spinmode = none` is exactly flat in either convention, because an
isotropic decay is isotropic about every axis.

**What does not.** The shape. On a comparable ZZ sample the corrected
distribution differs from the plotted one by up to 20 % in a bin (mean 5 %, 1.6 %
of the rate changes bin). And the caption's own words — "the angle between the
direction of the `e+` in the `(e+ e-)` rest frame and the direction of the
`(e+ e-)` system in the four-lepton rest frame" — read as the correct sequential
construction, so the figure does not currently implement its caption.

**What it costs to fix.** The `g g > z z` samples the figure was made on lived
in `/tmp` and have been swept; no copy survives anywhere on the machine. The
loop-induced study has to be re-run to regenerate it. `observables.py` is
already fixed and guarded, so a re-run produces the corrected figure with no
further work.

**If you do not want to re-run**: the figure is still a valid MadSpin-vs-truth
comparison, and the honest fix is to the caption — say that `theta_1` is the
polar angle of the `e+` in the `(e+e-)` rest frame *reached from the lab*, and
drop the words "helicity angle" if they appear. It is a well-defined,
discriminating, frame-dependent observable; it is just not the helicity angle,
and no polarisation fraction may be read off it.

## 3. Numbers you must not quote yet

`SPIN_COEFFICIENTS.md` recommended quoting `f_0 = 0.112 +- 0.007` for the `gg`
box, `C_kk = +0.57 +- 0.14` against `-0.68 +- 0.13`, and drafted a LaTeX
paragraph around them. **All of those are damped and none is currently
quotable.** They are biased towards isotropy: on the proxy sample `f_0` moves by
`-0.040` and `C_kk` by `-0.108`. The *physics* of that paragraph is unchanged —
`eta_l = 0.219` multiplies only the `P1` term, the rank-2 moment is undiluted,
`f_0 = 2 - 5 <cos^2 theta>`, the calibration is `4/eta_l^2 = 83.2` and not the
`9` of the spin-1/2 case, and the `gg` box and the `qq~` continuum correlate the
helicities with **opposite sign** (a rotation cannot flip a sign, so that
statement is robust). Only the numbers have to wait for a re-run.

Nothing in `applications.tex` or `validation.tex` currently quotes any of them,
so there is no correction to make — only a "do not paste these in yet".

## 4. A sentence you can use, if you want one

> The polarised weights were checked event by event against the decay angles:
> reweighting a 250\,000-event $pp\to ZZ$ sample by each
> $\{Z_0Z_0,Z_0Z_T,Z_TZ_0,Z_TZ_T\}$ weight reproduces
> $\langle\cos^2\theta\rangle = 1/5$ for a longitudinal and $2/5$ for a
> transverse $Z$, on all eight entries, to better than $1.4\sigma$.

---

Full diagnosis, evidence and the regression test:
`MadSpin/validation/zz_loopinduced/POLWEIGHT_CLOSURE_DIAGNOSIS.md`.
No MadSpin source was changed by any of this.
