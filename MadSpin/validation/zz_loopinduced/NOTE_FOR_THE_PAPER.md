# Note for the MadSpin-2 paper session — polarisation figures

**Short version: your `Z0Z0 / Z0ZT / ZTZ0 / ZTZT` figures are fine. The
loop-induced `cos1cos2` figure was damped by a frame bug and has now been
regenerated — replace it. Every coefficient the paper wanted from that study is
now measured and quotable, and the opposite-sign `C_kk` result survives.**

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

## 2. `figures/LI_processes/cos1cos2.pdf` — regenerated, please replace it

**The study has been re-run.** The frame fix is in, the samples are at 200 000
events instead of 50 000, and they now live outside `/tmp` at
`~/Documents/madspin_validation_samples/t118_zz_loopinduced/` with per-event
columns beside them, so this cannot happen a third time. Take the new figure
from

```
MadSpin/validation/zz_loopinduced/plots/cos1cos2.pdf            (MG7 style)
MadSpin/validation/zz_loopinduced/plots_userstyle/cos1cos2.pdf  (user style)
```

and likewise `pt_ee.pdf` from either directory. Both are PDF+PNG, `usetex`, and
the MG7 PDFs pass the minus-sign check.

**What changed in the two figures.**

* `cos1cos2`: the three spin-correlated modes still lie on truth
  (`chi2/ndf` 0.89 to 1.29). `spinmode = none` fails **much** harder —
  `chi2/ndf = 174.2/39`, against `28.6/39` before. Roughly `4x` of that is the
  four-fold statistics; the remaining `~55 %` is the frame fix sharpening a
  distribution that the Wigner rotation had smeared towards flat. Visually the
  `none` curve's central peak and depleted tails are now unmistakable.
* `pt_ee`: unchanged for `madspin` (1.10 -> 1.23), `onshell` and `none`
  (1.10 -> 1.08). **`PA` moved from `1.76` to `3.32`** — see section 5, where it
  is now diagnosed as the missing `m_4l < 2 m_Z` truth and not a `PA` defect;
  this is
  new and it is not a frame effect.

The caption is now correct as written: `theta_1` *is* the helicity angle, the
`e+` boosted `lab -> 4l frame -> (e+e-) rest frame` sequentially, and a
polarisation fraction may be read off it.

**Conclusions that did NOT move**: the four spinmodes all work on a
loop-induced process; the `+0.7 %` / `+5.2 %` normalisation split between the
modes that draw a virtuality and those that do not; `spinmode = none` being the
only mode the angular observables reject. Those are all still true and, where
statistics matter, more strongly true.

### The old section 2, kept for the record

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

**What it cost to fix.** A full re-run, which has now been done.

## 3. The numbers, now quotable

The "do not paste these in yet" of the previous version is lifted. Post-fix, on
200 000 events per sample:

| | pre-fix (50k) | **post-fix (200k)** |
|---|---|---|
| `f_0` (gg box, truth) | `0.112 +- 0.007` | **`0.067 +- 0.003`** |
| `f_TT` (gg box, truth) | `0.828 +- 0.011` | **`0.908 +- 0.007`** |
| `C_kk` (gg box) | `+0.57 +- 0.14` | **`+0.38 +- 0.07`** |
| `C_kk` (qq~ continuum) | `-0.68 +- 0.13` | **`-0.645 +- 0.080`** |
| `none` vs truth on `f_0` | `21.7 sigma` | **`77 sigma`** |

**The opposite-sign `C_kk` result survives** and is now a measurement rather
than a prediction: `+0.380 +- 0.072` against `-0.645 +- 0.080`, **9.5 sigma
apart**, both sides post-fix and both at 200 000 events. This was the one claim
genuinely at risk, because the correction pushed the `gg` value *down* by
`0.19`, towards the `qq~` sign. It did not cross: `+0.380` is `5.3 sigma` from
zero on its own.

The `gg` box is **more** transverse than the earlier numbers said — the
longitudinal fraction is `6.7 %`, not `11 %`, and `f_TT = 0.908`.

Quote `f_0` from the **`f_0 (both)`** column of `data/numbers.txt`, not from one
lepton pair. The two `Z` are equivalent, so the estimator is their per-event
average; it is a `sqrt(2)` improvement and it removes the one `+2.2 sigma`
scatter that the single-pair columns show for `PA`. All three physical modes
then reproduce truth to better than `1.7 sigma`.

The corrected LaTeX paragraph is at the end of `SPIN_COEFFICIENTS.md` and is
ready to paste.

**Caveat on one consistency check.** The pre-fix argument that `eta_l = 0.15`
must be wrong "because `C_kk` would come out `1.23`, outside its own bound" no
longer works: at the post-fix moment the wrong `eta_l` gives `0.82`, inside the
bound. The `eta_l = 0.219` value is still right — the derivation in section 3 of
`SPIN_COEFFICIENTS.md` settles it — but do not present the bound violation as
the evidence.

## 4. A sentence you can use, if you want one

> The polarised weights were checked event by event against the decay angles:
> reweighting a 250\,000-event $pp\to ZZ$ sample by each
> $\{Z_0Z_0,Z_0Z_T,Z_TZ_0,Z_TZ_T\}$ weight reproduces
> $\langle\cos^2\theta\rangle = 1/5$ for a longitudinal and $2/5$ for a
> transverse $Z$, on all eight entries, to better than $1.4\sigma$.

## 5. New, and unrelated to the frame: `PA` and `pt(e+ e-)`

At 200 000 events `spinmode = PA` shows a **coherent 7.5 % deficit below
`pt(e+ e-) = 20 GeV`**, recovering above it. Every low-`pt` bin pulls `-4` to
`-7 sigma`; `chi2/ndf = 3.32/69` against `madspin`'s `1.23` and `onshell`'s
`1.08`. `pt_ee` builds no decay-angle frame, so the frame fix cannot touch it —
this is purely the four-fold statistics resolving something that 50 000 events
left at a marginal `1.76`.

> **Diagnosed, 2026-08-28. The reshuffling hypothesis below is wrong, and so is
> the reading that this is a `PA` defect.** Full evidence:
> [PA_LOWPT_DIAGNOSIS.md](PA_LOWPT_DIAGNOSIS.md).
>
> The truth carries **2.09 % of its cross section below `m_4l = 2 m_Z`**, and
> **every** spinmode has *exactly zero* support there — `g g > z z` puts both
> `z` on shell, and the RAMBO reshuffle holds `sqrt(shat)` fixed, so `m_4l` is
> the production sample's, bit for bit. That missing region is entirely at low
> `pt`: 9.5 % of the truth's `pt < 5 GeV` bin, falling to nothing by 60 GeV.
> Restrict the truth to `m_4l >= 2 m_Z` and `PA`'s ratio below 20 GeV becomes
> `0.995 / 1.018 / 1.013 / 1.021`, flat, with `chi2/ndf` falling **3.32 -> 1.59**
> — the *best* of the three spin-correlated modes (`madspin` 2.15,
> `onshell` 2.41). `onshell`'s 1.08 on the full support is its `1/f^2 = +5.2 %`
> normalisation error cancelling the very same hole: in the first bin
> `1.1819 x (1 - 0.0953) = 1.069`, to four digits.
>
> The reshuffling is **not** the cause — dividing its jacobian out event by
> event leaves the deficit in place. It is the cause of the `m(l+ l-)` tilt,
> which is a real and quantified property of the production approximation.
>
> **Consequences for the paper.** `PA` can and should be quoted alongside the
> other modes. Do **not** quote `pt(e+ e-) chi2/ndf = 3.32` as a `PA` defect,
> and do not quote `onshell`'s 1.08 as agreement. If `pt_ee` is shown, `PA` will
> sit low in the first few bins of the ratio panel and the caption must say why:
> all four modes are missing the sub-`2 m_Z` truth, and only `PA` has a
> normalisation honest enough to show it.

`PA` is the only mode that reshuffles the production onto sampled virtualities
while evaluating the density matrix at **on-shell** momenta. `onshell` does not
reshuffle, `madspin` evaluates at the reshuffled momenta, and neither shows the
effect — so a mismatch between reshuffled kinematics and an on-shell density is
at least a coherent place to look.

~~**This is not yet understood and should not go in the paper as a result.**~~
(Superseded by the box above; the paragraph is kept so the note reads as the
record of what was thought when.)

---

Full diagnosis, evidence and the regression test:
`MadSpin/validation/zz_loopinduced/POLWEIGHT_CLOSURE_DIAGNOSIS.md` (the frame
bug) and `MadSpin/validation/zz_loopinduced/PA_LOWPT_DIAGNOSIS.md` (the `PA`
low-`pt` deficit).
No MadSpin source was changed by any of this.
