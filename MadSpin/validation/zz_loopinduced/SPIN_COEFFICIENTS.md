# What the two angular figures measure, and what it is called

> ## AMENDMENT (2026-08-27) — read this first
>
> **Section 9 of this file is withdrawn.** It reported that MadSpin's
> `ms_pol_*` weights fail a `1/5` / `2/5` closure by 14-21 sigma and called for
> an investigation of MadSpin. The investigation was done and the failure was
> **ours**: `observables.compute` built `cos_theta1/2` by taking the axis in the
> four-lepton frame while boosting the `l+` into its pair's rest frame straight
> from the **lab**. Those two boosts are not collinear, so their composition
> carries a Wigner rotation — median 8 degrees, mean 11, tail to 81 — which
> tilted the analysing frame off the axis and damped every rank-2 moment towards
> `1/3`. That, and nothing else, was the "0 <-> T leakage of 0.055".
>
> With the boosts composed sequentially (`lab -> 4l frame -> pair rest frame`)
> the eight entries close to **1.4 sigma at worst** on the same 250 000 events,
> and the two `f_0` extractions agree to **1.4 sigma** where they were 12 apart.
> **MadSpin's polarised weights are correct and nothing in MadSpin was changed.**
> The full diagnosis, the evidence, the fix and the regression test are in
> [`POLWEIGHT_CLOSURE_DIAGNOSIS.md`](POLWEIGHT_CLOSURE_DIAGNOSIS.md).
>
> **Consequence for the numbers below.** Every rank-2 angular quantity in this
> file — `f_0`, `f_00`, `f_TT`, `f_00 - f_0 f_0`, `C_kk`, and the `cos_theta`
> and `cos1cos2` histograms — was harvested with the pre-fix definition and is
> **biased towards the isotropic value**. On the proxy sample of section 9,
> `f_0` moves from `0.2105` to `0.1702` and `C_kk` from `-0.650` to `-0.758`.
> The `g g > z z` samples this study ran on have been swept from `/tmp` and no
> copy survives, so the corrected numbers require re-running the study; they are
> **not** recomputed here. The tables below are left as they were harvested,
> marked, rather than silently patched.
>
> **Not affected:** every MadSpin-versus-truth comparison (both sides were
> harvested identically, so the `chi2/ndf` and the ratio panels stand),
> `spinmode = none` being exactly `1/3` (an isotropic decay is isotropic about
> any axis), `phi_planes` (built entirely inside the four-lepton frame), and the
> `ms_pol_*` weights and the paper's `Z0Z0/Z0ZT/ZTZ0/ZTZT` figures built from
> them.

`cos(theta1)` and `cos1cos2` are the two figures of this study that carry spin
information. This file works out what named quantity sits behind each of them,
extracts it for every sample, closes the extraction against MadSpin's own
polarised matrix-element weights on an independent sample, gives the names the
di-boson literature uses, and ends with a verdict on which of them — if any —
belongs in the paper.

Short version:

| quantity | what it is | worth quoting? |
|---|---|---|
| `f_0 = 2 - 5 <cos^2 theta>` | the longitudinal polarisation fraction of one `Z` | **yes** |
| `C_kk = 4 <cos th1 cos th2> / eta_l^2` | the helicity-sign correlation between the two `Z` | **yes, as physics; no, as a MadSpin test** |
| `f_00 - f_0 f_0` | the rank-2 (alignment) correlation | no — 3 sigma at 50 000 events, and blind to the mode that fails |

The one substantive correction to the brief that prompted this: the analysing
power in **this** calculation is `eta_l = 0.2193`, not `0.15`. `0.15` is the
*effective* leptonic value, `sin^2 theta_eff = 0.2315`; the SM UFO runs the
on-shell scheme, `sw2 = 1 - MW^2/MZ^2 = 0.2222`, and `eta_l` is very steep in
`sw2`. The dilution is `eta_l^2/4 = 0.0120`, not `0.0056` — a factor 2.1 — and
that is the difference between `C_kk = 0.57` (allowed) and `C_kk = 1.23`
(impossible, since `|C_kk| <= 1`). The measured `<cos th1 cos th2>` is small
*entirely* because of the dilution; the underlying correlation is large.

---

## 1. The angles, read off the code

`observables.py` lines 244-262, not the description of them:

```python
zee_in4 = boost_to_rest(zee, four);  d1 = _unit(zee_in4[:, 1:4])
zmm_in4 = boost_to_rest(zmm, four);  d2 = _unit(zmm_in4[:, 1:4])
e1 = _unit(boost_to_rest(boost_to_rest(ep,  four), zee_in4)[:, 1:4])
e2 = _unit(boost_to_rest(boost_to_rest(mup, four), zmm_in4)[:, 1:4])
out['cos_theta1'] = clip(sum(d1 * e1)); out['cos_theta2'] = clip(sum(d2 * e2))
```

so `theta1` is the polar angle of the `e+` **in the `(e+ e-)` rest frame**,
measured against the `(e+ e-)` pair's own direction **in the four-lepton rest
frame**, and `theta2` the same for the `mu+` against the `(mu+ mu-)` direction.
Each `Z` is analysed on **its own** helicity axis, and in the four-lepton rest
frame those two axes are anti-parallel. That is the standard `H -> ZZ -> 4l`
helicity convention the module's docstring cites (Gao et al.,
[arXiv:0708.0458](https://arxiv.org/abs/0708.0458)), and it is what makes the
labels below mean what the literature means by them.

The **double** boost on the third and fourth lines is the whole content of the
amendment at the top of this file. Until 2026-08-27 they read
`boost_to_rest(ep, zee)` — the axis in the four-lepton frame, the lepton boosted
into the pair's rest frame straight from the lab. Two rest frames of the same
`Z`, differing by a Wigner rotation of median 8 degrees, and every number below
that was harvested before the fix carries its damping.

Two properties of *this* study's cuts matter for everything that follows. The
only cuts are `pt(l+ l-) > 1 GeV` on each reconstructed pair and
`|m(l+ l-) - MZ| < 15 Gamma_Z` on each pair (`zz_equivalent_cuts.f`), and
**neither is a function of a decay angle**. So the azimuths `phi1`, `phi2` are
integrated over their full range, and every statement below that relies on
"the off-diagonal density-matrix elements integrate away" is exact here rather
than approximate. A study with lepton `pT`/`eta` cuts could not say that.

## 2. The decay density matrix, and where the analysing power does and does not enter

For a `Z` of helicity `lambda` along its own flight direction decaying to
massless leptons, the `l+` polar angle is distributed as

```
W_{+1}(c) = (3/8)(1 + c^2 + 2 eta_l c)
W_{-1}(c) = (3/8)(1 + c^2 - 2 eta_l c)
W_{ 0}(c) = (3/4)(1 - c^2)
```

each normalised to 1 over `c in [-1, 1]`. Derivation, so that the weights are
not taken on trust: the two-body helicity amplitude is
`d^1_{lambda, h}(theta)` with `h = +-1` the lepton-pair helicity, so
`W_lambda = |A_+|^2 |d^1_{lambda,+1}|^2 + |A_-|^2 |d^1_{lambda,-1}|^2`. With
`d^1_{1,1} = (1+c)/2`, `d^1_{1,-1} = (1-c)/2`, `d^1_{0,+-1} = -+ sin(theta)/sqrt(2)`,
and writing `eta_l = (|A_+|^2 - |A_-|^2)/(|A_+|^2 + |A_-|^2)`,

```
W_{+1} ∝ (1+c)^2 (1+eta_l)/2 + (1-c)^2 (1-eta_l)/2 = 2(1 + c^2) + 4 eta_l c
W_{0}  ∝ sin^2(theta)
```

which normalise to the three lines above. In the SM
`eta_l = (g_L^2 - g_R^2)/(g_L^2 + g_R^2) = 2 g_V g_A/(g_V^2 + g_A^2)`, and with
`g_V/g_A = 1 - 4 sw2` that is

```
eta_l = (1 - 4 sw2) / (1 - 4 sw2 + 8 sw2^2)
```

— identical to the form Aguilar-Saavedra *et al.* use
([arXiv:2209.13441](https://arxiv.org/abs/2209.13441), where it is also called
`eta_l`), and the algebra `(1 + (1-4sw2)^2)/2 = 1 - 4sw2 + 8sw2^2` shows the two
spellings are the same function.

**The number for this study.** The samples ran the MG5 default `param_card`,
`aEWM1 = 132.507`, `Gf = 1.16639e-5`, `MZ = 91.188`, from which the SM UFO
builds `MW = 80.4190` and `sw2 = 1 - MW^2/MZ^2 = 0.222246`. Hence

```
eta_l = 0.21933          eta_l^2 / 4 = 0.012026          4 / eta_l^2 = 83.154
```

`observables.ETA_L` computes exactly this and a self-test on import checks that
the projector below returns `1/3` on an isotropic decay, `0` on a pure
transverse one, `1` on a pure longitudinal one, and is untouched by the `eta_l`
term.

Expanding the three shapes on Legendre polynomials makes the point of the whole
exercise visible in one line:

```
W_{+-1} = 1/2 + (1/4) P2(c) +- (3 eta_l/4) P1(c)
W_{ 0}  = 1/2 - (1/2) P2(c)
```

`eta_l` multiplies **only** `P1`. The rank-1 (vector polarisation) moments are
diluted; the rank-2 (alignment) moments are not diluted at all. That is claim 2
of the brief, and it is right.

## 3. The rank-2 moment: the longitudinal fraction `f_0`

Let `f_0 = P(lambda = 0)` be the longitudinal population of one `Z`. Averaging
the three shapes with the populations,

```
<cos^2 theta> = (2/5)(1 - f_0) + (1/5) f_0 = (2 - f_0)/5
```

(using `<c^2>_{+-1} = (3/8)(2/3 + 2/5) = 2/5` and
`<c^2>_0 = (3/4)(2/3 - 2/5) = 1/5`), so

> **`f_0 = 2 - 5 <cos^2 theta>`**

which is the brief's claim 2, re-derived rather than recalled, and it is
correct. `f_0 = 1/3` corresponds to `<cos^2 theta> = 1/3`, the isotropic value.

The useful way to carry it is as a **per-event estimator**,
`observables.POL_0(c) = 2 - 5 c^2`, whose mean is `f_0` with an ordinary error
of the mean. It is not a probability event by event — it runs over `[-3, 2]` —
but it is unbiased, and the product of the two `Z`'s estimators is the joint
fraction:

```
f_00 = < (2 - 5 cos^2 th1)(2 - 5 cos^2 th2) >
     = 25 <c1^2 c2^2> - 10 (<c1^2> + <c2^2>) + 4
```

(check: if the two `Z` are independent this collapses to `(2-f_0^1)(2-f_0^2)`
expanded, i.e. `f_0^1 f_0^2`, exactly). The other three joint fractions follow:
`f_0T = f_0^(1) - f_00`, `f_T0 = f_0^(2) - f_00`,
`f_TT = 1 - f_0^(1) - f_0^(2) + f_00`, and `f_TT = <(1-POL_0)(1-POL_0)>`.

## 4. The rank-1 correlation: what `<cos th1 cos th2>` actually is

After the azimuthal integration only the diagonal populations
`P(lambda1, lambda2)` survive, so

```
W(c1, c2) = sum_{l1,l2} P(l1,l2) W_{l1}(c1) W_{l2}(c2)
```

and since `<c>_{+-1} = +- eta_l/2` and `<c>_0 = 0`,

> **`<cos th1 cos th2> = (eta_l^2 / 4) C_kk`,
>  `C_kk = <S_k^(1) S_k^(2)> = P(++) + P(--) - P(+-) - P(-+)`**

with `S_k` the spin-1 projection on each `Z`'s **own** helicity axis (eigenvalues
`+1, 0, -1`; a longitudinal `Z` contributes zero, which is why this is a purely
transverse-sector object and why `|C_kk| <= f_TT`).

**The calibration constant is `4/eta_l^2`, and even at `eta_l = 1` it is 4, not
9.** The 9 of the `t t~` case is the spin-1/2 version of the same algebra: for a
decay `(1/2)(1 + kappa c)` one has `<c>_{+-} = +- kappa/3`, hence
`<c1 c2> = (kappa^2/9) C_nn` and `C_nn = 9 <c1 c2>` at `kappa = 1`. Running the
spin-1 algebra reproduces `4`, and running the spin-1/2 algebra reproduces `9`,
so the machinery here is calibrated against the number T54 established. Do not
carry `9` across.

Numerically `4/eta_l^2 = 83.154`, so a `C_kk` of order 1 shows up as a
`<cos th1 cos th2>` of order 0.012. **That is the whole explanation of why the
measured moment is 0.0069 and only 4 sigma from zero.** Inverting it:

```
gg -> ZZ  (this study, truth)         C_kk = +0.570 +- 0.141     [PRE-FIX]
qq~ -> ZZ (zz_nlo, truth)             C_kk = -0.675 +- 0.131     [PRE-FIX]
```

**[PRE-FIX]** — both numbers were harvested with the boost composition the
amendment at the top of this file withdraws, and both are damped towards zero.
The size of the damping on a sample where it can be measured is
`C_kk: -0.650 -> -0.758` (`run_12`, section 9). The *sign* difference between
the two mechanisms is robust — a rotation cannot flip a sign — but the
magnitudes and the `6.5 sigma` separation below are not, until both studies are
re-run.

(the second from the sibling study's own published moment,
`<c1 c2> = -0.00812 +- 0.00158`, same cuts, same window, same `eta_l`). The two
production mechanisms have **opposite-sign** helicity-sign correlations, 6.5
sigma apart, and both are large — `|C_kk| ~ 0.6` against a ceiling of
`f_TT ~ 0.83`. So the answer to "is the smallness the dilution or is the
production genuinely uncorrelated?" is: **it is the dilution, entirely.**

## 5. The rank-2 correlation: the undiluted analogue

Claim 3 of the brief asked for the undiluted correlation analogue. It is the
connected part of `f_00`. Expanding `W_{lambda}` on `P2` with coefficients
`b_{+-1} = 1/4`, `b_0 = -1/2`,

```
<P2(c1) P2(c2)> - <P2(c1)><P2(c2)> = (9/100) ( f_00 - f_0^(1) f_0^(2) )
```

and since `P2(c) = (3c^2-1)/2`, equivalently

```
f_00 - f_0^(1) f_0^(2) = 25 [ <c1^2 c2^2> - <c1^2><c2^2> ]
```

with **no `eta_l` anywhere**. So the brief's instinct was right about what the
object is. In the irreducible-tensor notation of section 7 it is
`C_{2020} - A^1_{20} A^2_{20}`, up to the factor `9/2` that convention carries.

It is now computed (`pol00`, and the `f_00 - f_0 f_0` column of
`data/numbers.txt`), and the verdict on it is in section 8: it is real but weak,
and it does not discriminate.

## 6. The numbers

From `data/numbers.txt`, regenerated by this pass. `f_0 (e+ e-)` is exact
(unbinned `<cos^2 theta1>`); the rest come from the committed histograms with
the binning bias removed by a local-quadratic within-bin reconstruction
(`plot_zz_loopinduced._binned_power`; validated to `1.6e-5` on `<cos^2 theta>`
and `3.3e-5` on `<(c1 c2)^2>` against the unbinned truth of an independent
250 000-event sample, an order of magnitude under the statistical bars).
Re-harvesting replaces the whole block with unbinned moments — the harvester now
stores `cos2_theta2`, `pol0_1`, `pol0_2`, `pol00`, `polTT`.

**[PRE-FIX] Every number in the two tables below is damped towards isotropy by
the Wigner rotation the amendment at the top of this file describes.** They are
left as harvested rather than silently patched; the `g g > z z` samples have
been swept from `/tmp` and no copy survives, so correcting them requires
re-running the study. Direction and rough size, from the proxy sample of
section 9: `f_0` down by about `0.04`, `f_TT` up by about `0.07`, `f_00` down by
about `0.009`, `C_kk` more negative by about `0.11`. The **pull** table that
follows is *not* affected in any interesting way — truth and the modes were
harvested identically — and neither is the `+21.7 sigma` on `none`.

| sample | `f_0` (e+e-) | `f_0` (mu+mu-) | `f_00` | `f_00 - f_0 f_0` | `f_TT` | `C_kk` |
|---|---|---|---|---|---|---|
| truth   | +0.1116 +- 0.0069 | +0.1067 +- 0.0069 | +0.0461 +- 0.0108 | +0.0342 +- 0.0107 | +0.8278 +- 0.0108 | +0.570 +- 0.141 |
| madspin | +0.1081 +- 0.0069 | +0.0935 +- 0.0069 | +0.0507 +- 0.0107 | +0.0406 +- 0.0107 | +0.8491 +- 0.0107 | +0.321 +- 0.142 |
| PA      | +0.1056 +- 0.0069 | +0.0860 +- 0.0069 | +0.0348 +- 0.0107 | +0.0257 +- 0.0107 | +0.8432 +- 0.0107 | +0.416 +- 0.142 |
| onshell | +0.0972 +- 0.0069 | +0.1051 +- 0.0069 | +0.0368 +- 0.0107 | +0.0266 +- 0.0107 | +0.8346 +- 0.0107 | +0.508 +- 0.142 |
| none    | +0.3211 +- 0.0067 | +0.3287 +- 0.0067 | +0.1145 +- 0.0105 | +0.0090 +- 0.0100 | +0.4647 +- 0.0105 | +0.355 +- 0.125 |

Pulls against truth (independent samples, so the bars add in quadrature):

| sample | `f_0` (e+e-) | `f_0` (mu+mu-) | `f_00` | `f_00 - f_0 f_0` | `C_kk` |
|---|---|---|---|---|---|
| madspin | -0.4 | -1.3 | +0.3 | +0.4 | -1.2 |
| PA      | -0.6 | -2.1 | -0.7 | -0.6 | -0.8 |
| onshell | -1.5 | -0.2 | -0.6 | -0.5 | -0.3 |
| none    | **+21.7** | **+23.1** | +4.6 | -1.7 | -1.1 |

### Is `spinmode = none` exactly isotropic? Measured, not assumed

It has to be, and for a reason worth writing down: decaying a `Z` with no
production density leaves it **unpolarised**, `P(lambda) = 1/3` for each
`lambda`, and `sum_lambda (1/3) W_lambda(c) = 1/2` identically — the `P2` pieces
cancel between `2 x (1/4)` and `-1/2`. So an unpolarised `Z` decays flat in
`cos(theta)` whether MadSpin draws from flat phase space or from the
spin-averaged decay matrix element; the two routes give the same thing.

Measured on the committed `none` histograms:

```
f_0(e+ e-)   = 0.3211 +- 0.0067      vs 1/3      -1.8 sigma
f_0(mu+ mu-) = 0.3287 +- 0.0067      vs 1/3      -0.7 sigma
f_00 - f_0 f_0 = +0.0090 +- 0.0100   vs 0        +0.9 sigma
```

so yes: isotropic to the statistics of 50 000 events, and the rank-2
*correlation* is compatible with zero, which is the null test that the joint
estimator has to pass for a mode that decays the two `Z` independently. The one
mild tension anywhere in this file is `none`'s `<cos th1 cos th2> = +0.00427 +-
0.00150`, 2.8 sigma from the zero that an independent decay must give; with five
samples and six moments that is unremarkable, and it is not a shape effect —
`none`'s `cos1cos2` **histogram** fails at `chi2/ndf = 28.6`.

## 7. What the literature calls these things

Two conventions are in use and they are **not** the same normalisation. Both
appear below with the mapping to what is measured here.

**(a) Fano / Gell-Mann, the entanglement literature.** Two spin-1 particles are
two qutrits, so the Bloch expansion runs over the eight generators per particle,

```
rho = (1/9) [ 1 (x) 1 + a_i (lambda_i (x) 1) + b_j (1 (x) lambda_j)
                      + c_ij (lambda_i (x) lambda_j) ]
```

with `tr(lambda_j lambda_k) = 2 delta_jk`. `a_i`, `b_j` are the single-particle
polarisation vectors and `c_ij` the **spin correlation matrix** — the direct
descendant of the `t t~` `C_ij`, but an 8x8 matrix, not 3x3, and with a
different numerical range. Ashby-Pickering, Barr and Wierzchucka,
[arXiv:2209.13990](https://arxiv.org/abs/2209.13990), set up the tomography with
Wigner P/Q symbols; the review by Barr, Fabbrichesi, Floreanini, Gabrielli and
Marzola, [arXiv:2402.07972](https://arxiv.org/abs/2402.07972), collects the
conventions; Fabbrichesi *et al.*,
[arXiv:2302.00683](https://arxiv.org/abs/2302.00683), apply them to weak
di-boson production. **Normalisation hazard:** whether the `1/9` is in front,
whether the generators are `lambda_i` or `lambda_i/2`, and whether `c_ij`
absorbs a factor `9/4` all vary between papers, and the numerical value of
`c_ij` changes with each. Quote the convention with the number, always.

**(b) Irreducible tensor operators, the `H -> ZZ` literature.**
Aguilar-Saavedra, Bernal, Casas and Moreno,
[arXiv:2209.13441](https://arxiv.org/abs/2209.13441), write

```
rho = (1/9) [ 1 (x) 1 + A^1_{LM} T^L_M (x) 1 + A^2_{LM} 1 (x) T^L_M
                      + C_{L1 M1 L2 M2} T^L1_M1 (x) T^L2_M2 ]
```

with `T^L_M` the rank-`L` multipole operators, `L = 1` the vector polarisation
and `L = 2` the alignment, and extract them by integrating the angular
distribution against spherical harmonics. **This is the notation that matches
the physics of this study**, because `L` is exactly the rank that decides
whether `eta_l` enters. Their companion `H -> WW` paper is
[arXiv:2209.14033](https://arxiv.org/abs/2209.14033).

The dictionary, for the basis ordered `(+1, 0, -1)` with
`T^1_0 = sqrt(3/2) S_z` and `T^2_0 = (3 S_z^2 - 2)/sqrt(2)`:

```
A_{20}  = (1 - 3 f_0)/sqrt(2) = 5 sqrt(2) <P2(cos theta)>      <-- what f_0 is
C_{2020} - A^1_{20} A^2_{20} = (9/2) ( f_00 - f_0^1 f_0^2 )    <-- section 5
C_{1010} = (3/2) C_kk = 6 <cos th1 cos th2> / eta_l^2          <-- section 4
```

The `A_{20} = 5 sqrt(2) <P2>` line was checked against their projector
`int Y^0_2 dsigma/sigma = (B_2/4pi) A_{20}` with `B_2 = sqrt(2pi/5)`; it agrees.
`S_z = (1/2) lambda_3 + (sqrt(3)/2) lambda_8` gives the (a)-convention
translation, so `C_kk` is a fixed combination of `c_33`, `c_88` and `c_38`
rather than a single Gell-Mann entry — one more reason to quote `C_kk` in the
spin-operator language and give the mapping.

**(c) The experimental name.** `f_0` is not exotic at all: it is the
**longitudinal polarisation fraction**, the standard di-boson polarisation
observable that ATLAS and CMS measure in `WZ`, `WW` and `ZZ`, and `f_00`,
`f_0T`, `f_T0`, `f_TT` are the **joint (doubly-polarised) fractions**. For
`gg -> ZZ` specifically the reference is Javurkova, Ruiz, Sabatini and
collaborators, *Polarized ZZ pairs in gluon fusion and vector boson fusion at
the LHC*, [arXiv:2401.17365](https://arxiv.org/abs/2401.17365) — same process,
same decomposition, and the natural paper to cite beside `f_0` here. The
matrix-element side of the same decomposition is
[arXiv:1912.01725](https://arxiv.org/abs/1912.01725) (Buarque Franzosi,
Mattelaer, Ruiz, Shil), which is what MadGraph's polarised matrix elements —
and MadSpin's `keep_weight_for_polarization_vector` weights — implement.

**Recommendation:** call it `f_0` and cite arXiv:2401.17365; mention the
Fano/`C_ij` language once, in the sentence that introduces `C_kk`, with
arXiv:2402.07972 and arXiv:2209.13441. Do not invent notation.

## 8. Does any of this discriminate better than the histogram chi2?

**No — and that is the answer, not a hedge.**

| | `none` vs truth, coefficient | `none` vs truth, shape chi2/ndf |
|---|---|---|
| `cos_theta1` / `f_0` | +21.7 sigma | 540.3/39 = 13.85 |
| `cos1cos2` / `C_kk` | -1.1 sigma | 1116.2/39 = **28.62** |
| `cos1cos2` / `f_00 - f_0 f_0` | -1.7 sigma | (same figure) |

The `cos1cos2` figure's discriminating power lives entirely in its **shape**,
and both coefficients built from it are blind to the very mode the figure
catches. The reason is structural: `spinmode = none` decays each `Z`
isotropically, so it gets `<cos th1 cos th2> = 0` — and the truth's value is
also nearly zero, because of the `eta_l^2` dilution. Two nearly-zero numbers do
not separate. The histogram separates because `none`'s `cos1cos2` distribution
is the product of two *flat* variables while the truth's is the product of two
`(1 + cos^2)`-shaped ones, and that is a shape difference of order 1 that no
first or second moment of the product captures.

`f_0` is the exception, but even there it does not *beat* the chi2 — 21.7 sigma
against 13.85 per degree of freedom over 39 bins is the same statement twice.
What `f_0` adds is not power, it is **meaning**: "MadSpin reproduces the
`cos(theta1)` histogram" and "the `gg -> ZZ` box produces
`(11.2 +- 0.7) %` longitudinal `Z`, MadSpin's density path gives
`(10.8 +- 0.7) %`, and switching the density off gives the unpolarised
`1/3`" are the same measurement, and only the second is a physics sentence.

`f_00 - f_0 f_0 = +0.0342 +- 0.0107` on the truth is a real 3.2-sigma
measurement of the rank-2 correlation, and it is the only *undiluted*
inter-decay object available — but 3.2 sigma at 50 000 events, blind to `none`,
and needing about `5e5` events for a 10-sigma statement. It is computed and
reported; it should not go in the paper.

## 9. The polarised-matrix-element cross-check — it ran, it failed, and the
failure was in this file

MadSpin can attach one extra weight per polarisation combination
(`set keep_weight_for_polarization_vector [0, T]`), giving `sigma_LL`,
`sigma_LT`, `sigma_TL`, `sigma_TT` on the very same events. Those weights exist
on the `p p > z z` samples of the `zz_pol_weights` study. **They do not exist on
this study's `g g > z z` samples**, so the check is not on these events — it is
a check of the *method*, on a sample where the geometry is exactly right.

### The sample chosen, and why

`Events/run_12_decayed_1` of `PROCNLO_loop_sm_7`: `p p > z z [QCD]` run at
`order = LO`, `spinmode = madspin`, `BW_cut = 15`, exclusive
`decay z > e+ e-` / `decay z > mu+ mu-`, 250 000 events, **no cuts**, read from
the Les Houches file, and every event has `NUP = 8` — a Born `2 -> 2`. That last
point is the reason for choosing it: MadSpin defines the `0`/`T` labels in the
`me_frame` (run-card `me_frame`, default `frame_id = 6`, i.e. the rest frame of
legs 1 and 2), and on a `2 -> 2` that frame **is** the `ZZ` rest frame, so the
matrix-element quantisation axis and this study's helicity axis coincide
exactly. Verified rather than assumed: `pt(ZZ) = 3.7e-8 GeV` over all 250 000
events, and computing the angles in the `me_frame` and in the four-lepton frame
gives bit-identical results.

### The result, with the boosts composed correctly

Reweighting the sample by each polarised weight must give `<cos^2 theta> = 1/5`
for a longitudinal `Z` and `2/5` for a transverse one. It does:

| weight | `<cos^2 th1>` | must be | `<cos^2 th2>` | must be |
|---|---|---|---|---|
| `LL` | 0.2004 +- 0.0006 | 0.2 | 0.2005 +- 0.0007 | 0.2 |
| `LT` | 0.2007 +- 0.0005 | 0.2 | 0.3998 +- 0.0009 | 0.4 |
| `TL` | 0.4009 +- 0.0010 | 0.4 | 0.2003 +- 0.0006 | 0.2 |
| `TT` | 0.4008 +- 0.0006 | 0.4 | 0.3992 +- 0.0008 | 0.4 |

Worst of the eight, **1.4 sigma** (jackknife, 20 blocks). The `LL` shape is a
pure `sin^2 theta` to `<P2> = -0.1994 +- 0.0008` (exact `-0.2`) and
`<P4> = -0.0004 +- 0.0010` (exact `0`), which is only true if `LL` really is the
single `(00),(00)` entry of the joint density matrix that
`decay.py::_restriction_rows` says it is.

And the two extractions of `f_0` — from the decay angles and from the polarised
matrix elements — agree:

| | angular moment | polarised ME | |
|---|---|---|---|
| `f_0` (e-side) | 0.17023 +- 0.00307 | `(LL+LT)/full` = 0.17440 +- 0.00035 | -1.4 sigma |
| `f_0` (mu-side) | 0.17720 +- 0.00306 | `(LL+TL)/full` = 0.17456 +- 0.00035 | +0.9 sigma |
| `f_00` | 0.05437 +- 0.00475 | `LL/full` = 0.05851 +- 0.00018 | -0.9 sigma |

The `0-T` interference the `{0, T}` partition drops contributes `+0.07 %` to the
total (`sum of the four / full = 1.000705 +- 0.000395`), unchanged by any of
this — it is a property of the weights alone.

### What the earlier version of this section reported, and why it was wrong

It reported the same eight entries as `0.2104 / 0.3893 / 0.2144 / 0.3890 / ...`,
14-21 sigma out, fitted by a symmetric `0 <-> T` leakage of `0.055`, and
concluded that "the `f_0` from the angular distribution and the `f_0` from
MadSpin's `ms_pol_*` weights are not the same number at the 5 % level". The
leakage was the Wigner rotation of the mixed boost composition described in
section 1 and diagnosed in
[`POLWEIGHT_CLOSURE_DIAGNOSIS.md`](POLWEIGHT_CLOSURE_DIAGNOSIS.md). Its
signature was already visible in the data and was not looked for: the deviation
**vanishes** when the `ZZ` system is at rest in the lab (`<P2>_LL = -0.2038 +-
0.0029` for `|y(ZZ)| < 0.15`) and when the `Z` flies along the beam
(`-0.1948 +- 0.0022` for `|cos theta*| > 0.95`), grows with the `Z` speed, and
is flat in the virtuality — the four defining properties of a Wigner rotation
between the lab and the `ZZ` frame, and none of them a property of anything in
MadSpin.

The exclusion list of the earlier version survives intact **except for its
first entry**. "The frame" was excluded by trying the axis in the lab (0.286)
and along the beam (0.293) and finding the `ZZ` axis (0.2104) closest. That scan
varied only the *axis* and left the lepton boosted from the lab in all three
variants, so every member of it was a mixed composition and the answer was not
in the family being scanned. The right question was not which axis but which
boost path.

`polweight_closure.py` in this directory reproduces every number above in one
pass over that Les Houches file, and `--mixed` reproduces the historical
failure.

For completeness, the same sample gives, from its (corrected) angular moments,
`f_0 = 0.1702 +- 0.0031` and `C_kk = -0.758 +- 0.061` for `qq~ -> ZZ` at LO. The
`C_kk = -0.675 +- 0.131` quoted for the sibling `zz_nlo` study in section 4 was
harvested with the pre-fix definition and will move by a comparable amount when
that study is re-run, so the "0.2 sigma agreement" claimed there is not a
statement that can currently be made either way.

## 10. Verdict

**0. The frame, which came first and was got wrong first.** `f_0` and `C_kk`
are moments about the `Z`'s **helicity** axis, and reaching that frame means
boosting to the four-lepton frame and then to the pair's rest frame, in that
order. Composing the two boosts the other way round — axis in one frame, lepton
boosted from another — is not a different convention, it is not a convention at
all: the resulting angle is not Lorentz invariant, and it damps every rank-2
moment towards `1/3`. This file's first pass did that, blamed the resulting
5 % closure failure on MadSpin, and was wrong. The three claims below are
unaffected in structure; their *numbers* are not.
   ([`POLWEIGHT_CLOSURE_DIAGNOSIS.md`](POLWEIGHT_CLOSURE_DIAGNOSIS.md))

1. **Claim 1 (dilution) — right in structure, wrong in the number, and the
   conclusion inverts.** `eta_l` enters squared, as claimed. But `eta_l = 0.219`
   here, not `0.15`, because the SM UFO is in the on-shell scheme; the dilution
   is `1/83`, not `1/180`. Undiluting, `C_kk = +0.570 +- 0.141` for `gg -> ZZ`
   against a ceiling of `f_TT = 0.83`. The production is **strongly**
   spin-correlated; the moment is small only because a `Z` is a poor spin
   analyser. Had `eta_l = 0.15` been used, `C_kk` would have come out `1.23`,
   outside its own bound — which is how the error shows up.
2. **Claim 2 (undiluted alignment) — right, including the weights and the
   formula.** `f_0 = 2 - 5 <cos^2 theta>` is confirmed by an independent
   derivation from the density matrix, `f_0 = 0.1116 +- 0.0069` for the truth,
   and `spinmode = none` gives `0.3211 +- 0.0067` and `0.3287 +- 0.0067`,
   consistent with the isotropic `1/3` that an unpolarised `Z` must give. The
   `none` numbers are the only ones in the file the frame error cannot touch —
   an isotropic decay is isotropic about every axis — which is also why they
   could not have caught it. `f_0 = 0.1116` **[PRE-FIX]**.
3. **Claim 3 (tensor-tensor analogue) — right about the object, wrong about it
   being worth adding.** The object is `f_00 - f_0^(1) f_0^(2)`, equivalently
   `25 [<c1^2 c2^2> - <c1^2><c2^2>]`, and it is genuinely `eta_l`-free. It is
   now computed. On the truth it is `+0.0342 +- 0.0107`, a 3.2-sigma
   measurement, and it separates `none` from truth by `-1.7 sigma`, i.e. not at
   all. It stays in `numbers.txt` and out of the paper.

**Quote `f_0` in the paper.** Mention `C_kk` in one sentence for the physics —
the opposite signs between the `gg` box and `qq~` are the interesting thing, and
they are 6.5 sigma apart. Leave the rank-2 correlation in `numbers.txt`.

---

## The paragraph for the paper

**Do not use the numbers in the LaTeX below.** Every `f_0` and `C_kk` in it was
harvested with the withdrawn boost composition and is damped towards isotropy;
the study has to be re-run before the paragraph can be quoted. The *physics* of
it — that `eta_l` multiplies only `P1`, that the rank-2 moment is undiluted,
that `f_0 = 2 - 5 <cos^2 theta>`, that the calibration is `4/eta_l^2 = 83.2` and
not the `9` of the spin-1/2 case, and that the `gg` box and the `qq~` continuum
correlate the helicities with opposite sign — is unchanged.

```latex
The two \texttt{cos}\,$\theta$ figures admit a named reading. For a $Z$ of
helicity $\lambda$ decaying to massless leptons the polar angle of the $\ell^+$
in the $Z$ rest frame follows $W_{\pm1}\propto 1+\cos^2\theta\pm2\eta_\ell
\cos\theta$ and $W_0\propto\sin^2\theta$, so the parity-violating term carries
the leptonic analysing power $\eta_\ell=2g_Vg_A/(g_V^2+g_A^2)$ --- $0.219$ in
the on-shell scheme of the default \textsc{MadGraph} parameter card --- while
the $(1+\cos^2\theta)$ versus $\sin^2\theta$ shape difference carries none. The
rank-two moment is therefore undiluted and gives the longitudinal polarisation
fraction of a single $Z$ directly, $f_0 = 2-5\langle\cos^2\theta_1\rangle$. On
the off-shell four-lepton reference we measure $f_0 = 0.112\pm0.007$, and the
three spin-correlated \textsc{MadSpin} modes reproduce it to better than
$1.5\sigma$: $0.108\pm0.007$ (\texttt{madspin}), $0.106\pm0.007$ (\texttt{PA})
and $0.097\pm0.007$ (\texttt{onshell}). Switching the production density off
gives $0.321\pm0.007$, the isotropic value $1/3$ that an unpolarised $Z$ must
produce, $21.7\sigma$ away. The inter-decay moment is the same statement one
rank lower, $\langle\cos\theta_1\cos\theta_2\rangle = (\eta_\ell^2/4)\,C_{kk}$
with $C_{kk}=\langle S_k^{(1)}S_k^{(2)}\rangle$ the correlation of the two
helicity projections; the calibration is $4/\eta_\ell^2 = 83.2$, not the $9$ of
the $t\bar t$ case, which is the spin-$1/2$ algebra. Undiluting the measured
moment gives $C_{kk} = +0.57\pm0.14$ for the $gg$ box against
$-0.68\pm0.13$ for the $q\bar q$ continuum of Sec.~\ref{sec:zznlo}: the two
mechanisms correlate the $Z$ helicities with opposite sign, and the smallness of
the raw moment is the $\eta_\ell^2$ dilution and nothing else. As a test of
\textsc{MadSpin} the coefficient adds nothing over the histogram --- it is blind
to \texttt{spinmode = none}, whose $\cos\theta_1\cos\theta_2$ distribution
nevertheless fails at $\chi^2/\mathrm{ndf} = 28.6$ --- so we quote it as
physics and keep the shape test as the test.
```
