# The polarised-weight closure failure was ours, not MadSpin's

Section 9 of `SPIN_COEFFICIENTS.md` reported that reweighting a MadSpin sample
by its `ms_pol_*` weights gave `<cos^2 theta> = 0.2104` where a longitudinal `Z`
must give `1/5`, and `0.3893` where a transverse one must give `2/5` — eight
entries, all 14-21 sigma out, with a single symmetric `0 <-> T` leakage of
`0.055` fitting all of them and independently explaining a 12-sigma gap in
`f_0`. It called for its own investigation.

**Verdict: the weights are right. The closure test was wrong.** The angular
observable it closed against was built from a boost composition that is not the
helicity frame, and the "leakage" is the Wigner rotation of that composition.
With the composition corrected the eight entries close to **1.4 sigma at worst**
on the same 250 000 events, and the two independent extractions of `f_0` — from
the decay angles and from the polarised matrix elements — agree to **1.4
sigma** where they used to be 12 apart.

No MadSpin source was changed. `MadSpin/validation/zz_loopinduced/observables.py`
was.

---

## 1. What was wrong

`observables.compute` built the two decay angles as

```python
zee_in4 = boost_to_rest(zee, four)          # axis: the Z direction in the 4l frame
d1 = _unit(zee_in4[:, 1:4])
e1 = _unit(boost_to_rest(ep, zee)[:, 1:4])  # lepton: boosted from the LAB
```

The axis `d1` is taken in the four-lepton rest frame. The lepton direction `e1`
is taken in the `(e+ e-)` rest frame reached by a **pure boost from the lab**.
Those are two different rest frames of the same `Z`.

A pure boost `B_1` from the lab to the `Z` rest frame, and the pair
`B_2 B_3` = (lab -> 4l frame) then (4l frame -> `Z` rest frame), both land on a
frame where the `Z` is at rest, but they are not the same Lorentz
transformation: they differ by a rotation,

```
B_1^-1 B_2 B_3  =  R(omega)                (the Wigner rotation)
```

and `omega` vanishes only when the two boosts are collinear — i.e. when the
four-lepton system is at rest in the lab, or when the `Z` flies along the
boost. Neither holds in general at a hadron collider. So the study was
measuring the polar angle of the lepton **in a frame rotated by `omega` away
from the axis it was measuring against**, and the observable was a function of
the observer: boosting the whole event along the beam changed it.

`omega` is not small. On `run_12` of `PROCNLO_loop_sm_7` (`p p > z z`, LO,
250 000 events, no cuts), measured directly as the angle between the two
constructions of the `l+` direction:

```
median 8.3 deg    <omega> = 11.2 deg    90th percentile 25.2 deg    max 81.1 deg
```

A rotation by `omega` damps every rank-2 moment towards the isotropic value,
which is exactly what "everything pulled towards `1/3`" meant.

## 2. The evidence, in the order it was found

### 2.1 The shapes are spin-1 shapes, so nothing structural is broken

A spin-1 two-body decay density is quadratic in `cos theta`, so
`<P3> = <P4> = 0` identically, whatever the polarisation and whatever the axis.
Measured under the nominal weight and under each of the four polarised weights,
all ten entries are consistent with zero (`|<P3>|, |<P4>| < 0.002`, errors
`0.0007-0.0013`). That rules out a broken mask, a wrong row, a mislabelled
basis or a contaminated weight: whatever is happening preserves the rank of the
distribution, which a rotation does and a bookkeeping error generally does not.

### 2.2 The leak has a pure kinematic signature

Binning the closure on `run_12`, `<P2(cos theta1)>` under the `LL` weight
(exactly `-0.2` if the weight means what it says):

| binned in | bin | `<P2>_LL` |
|---|---|---|
| `\|y(ZZ)\|` | `< 0.15` | **-0.2038 +- 0.0029** |
| | `0.5-0.8` | -0.1914 +- 0.0032 |
| | `> 1.8` | -0.1696 +- 0.0017 |
| `\|cos theta*\|` of `Z1` in the `ZZ` frame | `< 0.2` | -0.1775 +- 0.0031 |
| | `> 0.95` | **-0.1948 +- 0.0022** |
| `beta(Z1)` in the `ZZ` frame | `< 0.4` | **-0.1953 +- 0.0022** |
| | `> 0.92` | -0.1736 +- 0.0034 |
| `\|m(e+e-) - MZ\|` | `< 1 GeV` | -0.1844 +- 0.0014 |
| | `15-40 GeV` | -0.1812 +- 0.0070 |

The leak vanishes when the `ZZ` system is at rest in the lab, vanishes when the
`Z` flies along the beam, grows with the `Z` speed, and is flat in the
virtuality. Those are the four defining properties of a Wigner rotation between
the lab and the `ZZ` frame, and they are not the properties of anything in
MadSpin: the virtuality flatness in particular re-confirms T111's exclusion of
off-shellness.

### 2.3 Correcting the composition closes the test

Same file, same weights, same 250 000 events; the only change is that the
lepton is boosted `lab -> ZZ frame -> Z rest frame` instead of `lab -> Z rest
frame`:

| weight | `<cos^2 th1>` was | is | must be | `<cos^2 th2>` was | is | must be |
|---|---|---|---|---|---|---|
| `LL` | 0.2104 | **0.2004** | 0.2 | 0.2117 | **0.2005** | 0.2 |
| `LT` | 0.2144 | **0.2007** | 0.2 | 0.3876 | **0.3998** | 0.4 |
| `TL` | 0.3890 | **0.4009** | 0.4 | 0.2142 | **0.2003** | 0.2 |
| `TT` | 0.3893 | **0.4008** | 0.4 | 0.3875 | **0.3992** | 0.4 |

Worst of the eight: **1.4 sigma** (jackknife over 20 blocks), against 14-21
sigma before. The Legendre moments of the `LL` shape are
`<P1> = -0.0028 +- 0.0013`, `<P2> = -0.1994 +- 0.0008` (exact value `-0.2`),
`<P3> = +0.0018 +- 0.0009`, `<P4> = -0.0004 +- 0.0010` — a pure `sin^2 theta`,
which is what `rho_prod(00,00) rho_dec(0,0)` has to be.

And the 12-sigma `f_0` gap is gone:

```
f_0 (slot 1)  angular 0.17023 +- 0.00307   polarised ME 0.17440 +- 0.00035   -1.4 sigma
f_0 (slot 2)  angular 0.17720 +- 0.00306   polarised ME 0.17456 +- 0.00035   +0.9 sigma
```

### 2.4 The mechanism, checked as a number and not only as a trend

`<P2(cos omega)>` weighted by the `LL` weight is `0.9515`; the naive
factorised prediction for the damped moment is
`0.9515 x (-0.2) = -0.1903` against the measured `-0.1843`. The residual is the
correlation between `omega` and the decay angle, which the factorised estimate
drops. For the `TT` weight the naive prediction (`+0.0893`) misses the measured
`+0.0839` by more, and correctly so: the transverse block carries `m != 0`
multipoles — the `(+1,-1)` interference — that a rotation folds into `m = 0`
where a pure `m = 0` block like `LL` has none. That asymmetry between the `L`
and `T` damping (`0.921` against `0.839`) is why the "symmetric `0 <-> T`
leakage" of section 9 fitted well but not perfectly, and why it was never a
rotation *of the axis alone*.

## 3. What the weights actually are — read off the code, as asked

`MadSpin/interface_madspin.py`, `_polarization_ratios` (line ~8007) and its
block comment at line ~7755:

```
w_C = w_nominal * <rho_dec, rho_prod>_C / <rho_dec, rho_prod>
```

Both contractions are `density_dec.scalar_multiplication(density_prod)` on the
matrices built for the nominal weight, with `density_prod.hel_restriction` set
to the combination `C` for the numerator and left at the production restriction
for the denominator. The restriction constrains **both** the bra and the ket
of every slot (`decay.py::_restriction_rows`), and it is applied on the
*accepted* decay chain — in the joint path immediately after the nominal
contraction (`interface_madspin.py:10275`), in the sequential path on the
accepted chain explicitly (`interface_madspin.py:9768`).

So hypothesis 1 of the brief — "the weight may be a production-only ratio" — is
false at the level of the code, and would in any case have predicted
`<cos^2 theta>_LL = <cos^2 theta>_full = 0.366`, not `0.2104`.

Hypothesis 2 — "`L/T` interference does not vanish the way the closure assumes"
— is also false, and it is worth writing down why, because it is the one that
looked most likely to make this a non-bug. `w_LL ∝ rho_prod(00,00) |A_0(Om1)|^2
|A_0(Om2)|^2` factorises exactly, and `|A_0|^2 ∝ sin^2 theta` carries no
azimuthal dependence at all, so the `LL` marginal in `cos theta1` is `(3/4)(1 -
c^2)` for **any** production density whatever — there is nothing to interfere
with. For the `T` block the `(+1,-1)` interference does survive the restriction
(that is what makes `[0, T]` a partition), but it enters as `Re[rho_prod(+-,-+)
e^{2 i phi}]` and dies in the azimuthal integral that `<cos^2 theta>` performs
— provided the azimuth is the one about the same axis, which is precisely what
the mixed composition broke. The closure test's `1/5` and `2/5` were right;
they were being asked of the wrong angle.

Hypothesis 3 — a basis-ordering subtlety — is false: an ordering error cannot
switch itself off at `y(ZZ) = 0`.

## 4. Which of T111's exclusions survived, and which fell

| T111's exclusion | verdict |
|---|---|
| **the frame** ("the `ZZ` axis is the right one; lab gives 0.286, beam 0.293") | **fell.** The axis was right; the *lepton's* frame was not. T111 varied only `d1` and left `e1` boosted from the lab in all three variants, so all three were mixed compositions and the scan could not reach the answer. The `lab` variant is the one internally consistent member of that family (axis and lepton both from the lab) and it is a genuinely different, genuinely worse basis — which is why the scan looked conclusive. |
| the reshuffling (production/decay `Z` directions agree to 0.000 deg) | survived, and is not the issue. |
| off-shellness (flat in `\|m_ee - MZ\|`) | **re-verified** here (table in 2.2): `-0.1844` at the peak against `-0.1812` in the far tail. Flat, and it stays flat after the fix. |
| the spinmode (`onshell` shows the same) | survived — and is now *explained*: the Wigner rotation is kinematics, so every spinmode and every order shows it. |
| weight tails / statistics | survived; jackknife bars reproduced here. |
| the mask logic (`_restriction_rows` constrains bra and ket, `LL` is the single `(00),(00)` entry) | **re-verified** by reading, and now confirmed empirically: the `LL` weight's angular shape is a pure `sin^2` to `<P2> = -0.1994 +- 0.0008`, `<P4> = -0.0004 +- 0.0010`. That is only true if `LL` really is that one entry. |
| the sum rule (`sum of the four / full = 1.000705 +- 0.000395`) | unchanged by the fix — it is a property of the weights alone. |

## 5. The fix, and the test that keeps it fixed

`observables.compute` now boosts sequentially:

```python
zee_in4 = boost_to_rest(zee, four)
d1 = _unit(zee_in4[:, 1:4])
e1 = _unit(boost_to_rest(boost_to_rest(ep, four), zee_in4)[:, 1:4])
```

`observables._selftest_helicity_frame`, run on import beside the two self-tests
that were already there, checks two things on a synthetic `ZZ -> 4l` sample:

1. **Lorentz invariance.** `cos_theta1`, `cos_theta2`, `cos1cos2` and
   `phi_planes` are built from the four lepton momenta and from rest frames
   those momenta define, so boosting the whole event cannot move them. The test
   boosts by `beta_z = 0.90`, `beta_z = -0.55` and a non-collinear
   `(0.31, -0.18, 0.62)` and requires agreement to `1e-6`. The pre-fix
   definition moves `cos_theta1` by **0.94** under the first of those and fails
   immediately.
2. **Positive identification of the frame.** A decay generated as a pure
   `sin^2 theta` about the parent's own direction in the four-lepton frame has
   to come back with `<cos^2 theta> = 1/5` — the same closure, on a sample where
   the answer is known by construction rather than taken from MadSpin.

`polweight_closure.py` is fixed the same way and keeps a `--mixed` flag that
reproduces the old failure on demand.

## 6. What this costs the study

The correction is a real change to every rank-2 angular number, because a
rotation damps them. Measured on `run_12` (`p p > z z`, LO, no cuts, 250 000
events) — this is the only ZZ sample carrying `ms_pol_*` weights, and it is a
*proxy*: the size of the shift depends on the rapidity spectrum of the `ZZ`
system and must be re-measured per sample.

| | as shipped (mixed) | corrected | shift |
|---|---|---|---|
| `<cos^2 th1>` | 0.35790 +- 0.00061 | 0.36595 +- 0.00061 | +0.0081 (13.2 sd) |
| `f_0` (slot 1) | 0.21048 +- 0.00305 | **0.17023 +- 0.00307** | -0.0403 (-13.2 sd) |
| `f_0` (slot 2) | 0.21791 +- 0.00304 | **0.17720 +- 0.00306** | -0.0407 (-13.4 sd) |
| `f_00` | 0.06330 +- 0.00473 | 0.05437 +- 0.00475 | -0.0089 (-1.9 sd) |
| `f_TT` | 0.63491 +- 0.00576 | 0.70694 +- 0.00593 | +0.0720 (12.5 sd) |
| `<cos th1 cos th2>` | -0.00782 +- 0.00072 | -0.00911 +- 0.00073 | -0.0013 (-1.8 sd) |
| `C_kk` | -0.650 +- 0.060 | **-0.758 +- 0.061** | -0.108 |

At the histogram level, over 40 bins on `[-1, 1]`:

```
cos_theta1   max bin shift  5.6 %   mean |shift| 2.7 %   1.35 % of the rate moves bin
cos1cos2     max bin shift 19.7 %   mean |shift| 5.1 %   1.62 % of the rate moves bin
```

The bias always points **towards** the isotropic value, so a shipped `f_0`
below `1/3` is too high and a shipped `f_TT` is too low.

### What does NOT change

* **Every MadSpin-versus-truth comparison in this study.** Both sides were
  harvested with the same definition, so the `chi2/ndf` values, the ratio
  panels and the conclusion "the three spin-correlated modes sit on top of the
  off-shell truth, `none` does not" are all unaffected. The mixed variable is a
  perfectly good discriminating observable; it is just not the helicity angle.
* **`spinmode = none` is exactly `1/3` in either convention.** An isotropic
  decay is isotropic about every axis and a rotation cannot change that, so the
  `21.7 sigma` separation of `none` from truth stands.
* **`phi_planes`.** It is built entirely inside the four-lepton frame
  (`ep4, em4, mup4, mum4` are all `boost_to_rest(., four)`), so it never had the
  problem.
* **The `ms_pol_*` weights themselves, and everything built from them.**

## 7. The shipped figures

| artefact | affected? | what to do |
|---|---|---|
| `figures/NLO_Polarised/*` of `madspin2_paper` — the `Z0Z0 / Z0ZT / ZTZ0 / ZTZT` polarisation-fraction figures (`applications.tex`, Fig. `NLO_polarisations_application`), from the `zz_pol_weights` study | **no** | Nothing. They are built from the `ms_pol_*` weights, which this note validates to 1.4 sigma, and their observables are `M(e+ mu+)` and `Delta phi(e+ e-)` — no decay-angle frame is constructed anywhere in `pol_analysis.py`. The `data/weights*.npz` carry no `cos_theta` array at all. |
| `figures/LI_processes/cos1cos2.pdf` (`applications.tex`, Fig. `fig:cos1cos2_LI`) | **yes, the curves** | The plotted variable is the mixed one. The figure's *message* survives (both sides identical treatment), but the shape is damped: on the proxy sample the distribution moves by up to 20 % in a bin. The caption's prose — "the angle between the direction of the `e+` in the `(e+ e-)` rest frame and the direction of the `(e+ e-)` system in the four-lepton rest frame" — reads as the sequential definition, so the figure does not currently implement its own caption. |
| `plots/cos_theta1.*`, `plots/cos_theta2.*`, `plots/cos1cos2.*` of this study, and the same in `plots_userstyle/` | **yes, the curves** | Re-harvest. |
| `data/numbers.txt`, `data/histograms.npz`, and every angular number of `RESULTS.md` / `SPIN_COEFFICIENTS.md` | **yes** | Re-harvest. |
| the sibling `zz_nlo` study's `cos_theta1`, `cos2_theta1`, `cos1cos2` columns | **yes** | `observables_zz.leptonic()` returns this very module, so it inherits both the bug and the fix. Its production-level observables (`m_zz`, `pt_z`, `y_zz`, `cos_theta_star`) are untouched. |

**The re-harvest cannot be done from what is on disk.** The `g g > z z` samples
this study ran on lived under `/tmp/t75run/` and have been swept: the directory
tree is intact and every file in it is gone, and no other `g g > z z` or
`g g > e+ e- mu+ mu-` MadSpin-decayed event file exists on the machine. Both
studies have to be re-run to get corrected angular numbers. Until they are, the
`f_0` and `C_kk` values in `SPIN_COEFFICIENTS.md` and `data/numbers.txt` should
be read as biased towards isotropy by an amount of order the `run_12` shift.

For orientation only, propagating `run_12`'s damping
(`<P2>_L -> -0.1843`, `<P2>_T -> +0.0839`, so
`f_0^true = (<P2>_mixed - 0.0839) / (-0.2682)`) to this study's shipped
`f_0(truth) = 0.1116` gives `~0.065`. Treat that as an upper bound on the size
of the correction rather than as a number: `g g > z z` is more central than
`q q~ > z z`, the damping is milder at small `|y(ZZ)|` (`<P2>_LL = -0.2038` for
`|y| < 0.15` against `-0.1696` for `|y| > 1.8`), and this study also carries a
mass window and a `pt(Z) > 1 GeV` cut that `run_12` does not.

## 8. Reproducing all of it

```
cd MadSpin/validation/zz_loopinduced
python3 -c "import observables"            # the three self-tests, incl. the new one
python3 polweight_closure.py /path/to/run_12_decayed_1/events.lhe.gz
python3 polweight_closure.py /path/to/run_12_decayed_1/events.lhe.gz --mixed
```

The first prints the eight closing entries and the two agreeing `f_0`; the
second prints the historical failure. The sample is
`PROCNLO_loop_sm_7/Events/run_12_decayed_1` — `p p > z z [QCD]` at `order = LO`,
`spinmode = madspin`, `BW_cut = 15`, `set keep_weight_for_polarization_vector
[0, T]`, exclusive `decay z > e+ e-` / `decay z > mu+ mu-`, 250 000 events, no
cuts, every event `NUP = 8`.
