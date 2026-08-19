# Does the `blocks_cnn` pane show that the diagonal blocks "sit on zero"?

**No.  The plot is right and the prose is wrong.**  Two different objects are
being called `C_nn`: the *coefficient*, which does vanish on every diagonal
block, and the *distribution in picobarns* that the pane actually draws, which
emphatically does not.

Everything below is measured from the committed `data/histograms.npz`; nothing
re-runs MadSpin and nothing is fitted.  Reproduce with

    python3 check_cnn_claim.py            # -> plots/claim_cnn_numbers.txt
                                          #    plots/claim_cnn_vs_ckk.pdf

## 1. What the pane plots

`analyse_interference.py:286` defines the per-event observable

```python
o['cnn'] = (up * nhat).sum(1) * (um * nhat).sum(1)
```

i.e. `X = cos(theta^n_{l+}) * cos(theta^n_{l-})` -- a number between -1 and +1
for **every event**, not a coefficient.  `nhat` is built at line 274 in the
`t tbar` CM frame from `khat = unit(p_t)`,

```python
nhat = sgn[:, None] * np.cross(zhat, khat) / sinT[:, None]
```

with `sgn = sign(khat . zhat)`; `up`, `um` are the `l+` / `l-` directions in the
`t` / `tbar` rest frames.  Both legs are projected on the **same** `nhat`, with
no relative minus sign (unlike `cos_k_m` at line 281, which does carry one).

`blocks_figure` (`plot_interference.py:395`) then draws, for each of the nine
samples, at lines 417 and 423,

```python
ax.hist(x=ctr, weights=s / wid, histtype='step', ...)   # s = sum of weights
ax.set_ylabel(r'$d\sigma/dX$ [pb]')
```

So the y-axis is **`dsigma/dX` in pb**, and the x-axis is that per-event
product.  The coefficient is the *first moment* of this distribution, not the
distribution: `C_nn` is proportional to `<X>`, i.e. to `integral X dsigma /
integral dsigma`.  The pane's own x-axis label,

```python
('cnn', r'$\cos\theta^{n}_{\ell^+}\cos\theta^{n}_{\ell^-}$   ($C_{nn}$)', ...)
```

(`plot_interference.py:135`) puts `(C_nn)` next to the per-event product, which
is exactly the invitation to conflate the two.

## 2. The decisive table

`sum w` is the integral of the drawn curve; `sum w * X / sigma_9` is that
block's **additive contribution to `<X>`** over the full sample.  Errors are the
MC errors of the committed `sum w^2` moments.

| block | integral [pb] | contribution to `<X>` (`C_nn`) | sigmas | contribution to `<X>` (`C_kk`) | sigmas |
|---|---|---|---|---|---|
| `(D+,D+)` | 7.9314 +- 0.0355 | +0.00029 +- 0.00050 | +0.6 | +0.03616 +- 0.00050 | **+73.0** |
| `(D+,D-)` | 3.9385 +- 0.0176 | -0.00035 +- 0.00025 | -1.4 | -0.01874 +- 0.00025 | **-75.4** |
| `(D-,D+)` | 3.9386 +- 0.0176 | +0.00005 +- 0.00025 | +0.2 | -0.01820 +- 0.00025 | **-73.7** |
| `(D-,D-)` | 7.9465 +- 0.0355 | +0.00019 +- 0.00050 | +0.4 | +0.03665 +- 0.00050 | **+73.6** |
| `(I,D+)` | 0.0006 +- 0.0083 | +0.00001 +- 0.00009 | +0.1 | -0.00001 +- 0.00010 | -0.2 |
| `(I,D-)` | -0.0034 +- 0.0084 | -0.00006 +- 0.00009 | -0.7 | -0.00001 +- 0.00010 | -0.1 |
| `(D+,I)` | 0.0009 +- 0.0083 | +0.00006 +- 0.00008 | +0.7 | +0.00007 +- 0.00010 | +0.7 |
| `(D-,I)` | 0.0018 +- 0.0084 | +0.00006 +- 0.00009 | +0.7 | -0.00003 +- 0.00010 | -0.3 |
| **`(I,I)`** | **-0.0237 +- 0.0306** | **+0.03654 +- 0.00056** | **+64.9** | -0.00033 +- 0.00026 | -1.3 |
| 4 diagonal | **23.7551 +- 0.0560** | **+0.00018 +- 0.00079** | **+0.2** | +0.03587 +- 0.00079 | +45.7 |
| 4 single-`I` | -0.0001 +- 0.0167 | +0.00007 +- 0.00017 | +0.4 | +0.00001 +- 0.00020 | +0.1 |
| all 9 blocks | 23.7313 +- 0.0660 | +0.03679 +- 0.00098 | +37.4 | +0.03555 +- 0.00085 | +41.8 |
| unpolarised | 23.7651 | +0.03625 +- 0.00149 | | +0.03815 +- 0.00148 | |

Read it in two directions:

* **Down the "integral" column.**  The four diagonal blocks carry
  `23.7551 +- 0.0560` pb, i.e. **the entire cross section** (unpolarised:
  `23.7651` pb).  Nothing about them "sits on zero".  The peak bin of the
  `(D+,D+)` curve is `1.304 +- 0.014` pb.  The five interference blocks all
  integrate to zero, as they must.
* **Down the `C_nn` column.**  Every diagonal block's contribution is
  compatible with zero (largest pull `-1.4`), and summed they give
  `+0.00018 +- 0.00079` -- consistent with zero at `0.2 sigma`.  The four
  single-`I` blocks are zero at `0.4 sigma`.  `(I,I)` alone gives
  `+0.03654 +- 0.00056`, which is `99.3%` of the total `+0.03679 +- 0.00098`,
  and `64.9 sigma` from zero.

**The `C_kk` column is the control.**  Exactly the same nine samples, exactly
the same estimator, and the roles are swapped: the diagonal blocks are at
`73-75 sigma` each and `(I,I)` is at `-1.3 sigma`.  That is the double-flip
structure of `C_nn` versus the diagonal structure of `C_kk`, measured, on the
same events.

**How a non-zero histogram has a zero mean: symmetry.**  Mirror the `cnn`
histogram of a diagonal block about `X = 0` and compare bin by bin:

| block | `chi2`, `C_nn` | `chi2`, `C_kk` | peak bin of the `C_nn` curve |
|---|---|---|---|
| `(D+,D+)` | 10.8 / 10 | **5302.1 / 10** | 1.304 +- 0.014 pb |
| `(D+,D-)` | 9.4 / 10 | **5653.8 / 10** | 0.653 +- 0.007 pb |
| `(D-,D+)` | 9.6 / 10 | **5380.5 / 10** | 0.652 +- 0.007 pb |
| `(D-,D-)` | 15.8 / 10 | **5382.4 / 10** | 1.316 +- 0.014 pb |

All four diagonal blocks are symmetric in `X` for `C_nn` -- large everywhere,
zero first moment -- and grossly asymmetric for `C_kk`.  Same samples, same
binning.  This is visible on the pages themselves: compare the diagonal curves
in `plots/blocks_cnn.pdf` (peaked and symmetric about 0) with
`plots/blocks_ckk.pdf` (the same blocks, visibly skewed).

## 3. Why `C_nn` vanishes on a diagonal block -- the algebra

Write the production spin density matrix in the helicity basis
`{++, +-, -+, --}`, quantisation axis `z = k`, with `(x, y, z) = (r, n, k)`:

    rho = 1/4 [ 1 x 1 + B+.sigma x 1 + 1 x B-.sigma + C_ij sigma_i x sigma_j ],
    Tr rho = 1,     C_ij = Tr[ rho sigma_i x sigma_j ].

`sigma_k = sigma_z` is diagonal; `sigma_n = sigma_y` and `sigma_r = sigma_x` are
purely off-diagonal.  `check_cnn_claim.py` Part A does the `4x4` algebra
numerically and prints the matrices; the result is

    C_kk = + rho(++,++) - rho(+-,+-) - rho(-+,-+) + rho(--,--)     [0 flips]
    C_nn = - rho(++,--) + rho(+-,-+) + rho(-+,+-) - rho(--,++)     [2 flips]
    C_rr = + rho(++,--) + rho(+-,-+) + rho(-+,+-) + rho(--,++)     [2 flips]

i.e., using hermiticity,

    C_nn = 2 Re rho(+-,-+) - 2 Re rho(++,--)
    C_rr = 2 Re rho(+-,-+) + 2 Re rho(++,--)

confirming the identities recorded in `MADSPIN_SEQUENTIAL_PLAN.md` section
13.15 (`4 Re rho(++;--) = C_rr - C_nn`, `4 Re rho(+-;-+) = C_rr + C_nn`);
checked on 2000 random `(B+, B-, C_ij)`, worst residual `2.2e-16`.

`C_nn` and `C_rr` touch **only** entries in which the top *and* the antitop
helicity index differ between bra and ket -- the `(I,I)` block, and nothing
else.  `C_kk` touches only the diagonal entries.  On a definite-helicity block
`|h h'><h h'|`, a product of helicity eigenstates,

    (D+,D+): C_kk = +1   C_nn = 0   C_rr = 0   B+_n = 0   B-_n = 0
    (D+,D-): C_kk = -1   C_nn = 0   C_rr = 0   B+_n = 0   B-_n = 0
    (D-,D+): C_kk = -1   C_nn = 0   C_rr = 0   B+_n = 0   B-_n = 0
    (D-,D-): C_kk = +1   C_nn = 0   C_rr = 0   B+_n = 0   B-_n = 0

so `C_nn = <sigma_n> <sigma_n> = 0 * 0 = 0` for each of them -- and this is
exactly what the `C_nn` column of the table measures.  It says nothing about
their cross section.

**A free by-product: the null result also fixes MadSpin's quantisation axis.**
`C_ij = <sigma_i>_t <sigma_j>_tbar` on a product state, and a helicity
eigenstate along an axis `a` has `<sigma> = +-a`, so the diagonal blocks give
`C_nn = 0` only if `a . n = 0`.  Both transverse coefficients and the transverse
polarisation vanish on all four diagonal blocks:

| diagonal blocks, summed | value | pull |
|---|---|---|
| contribution to `<cos th^n_{l+} cos th^n_{l-}>` | +0.00018 +- 0.00079 | +0.2 |
| contribution to `<cos th^r_{l+} cos th^r_{l-}>` | -0.00045 +- 0.00079 | -0.6 |
| `<cos th^n_{l+}>` per block | <= 0.0048 +- 0.0026 | <= 1.8 |

so `a` is orthogonal to both `n` and `r`, i.e. `a = +-k` -- MadSpin quantises
along the same top direction in the `t tbar` CM frame that
`analyse_interference.py` uses (`me_frame = [1,2]`).  That was never a free
parameter of this test, but it is worth knowing it is measured and not assumed.

## 3b. The mean *is* the coefficient, up to the standard 1/9 -- calibrated here

The link between the plotted variable and the coefficient is the standard
factor `1/9` (`1/3` per leg, from the `(1 + cos)/2` lepton distribution, with
the leptonic spin-analysing power `kappa_l = 1`).  The four diagonal blocks
calibrate it directly, because for them the coefficients are known exactly:
`C_kk = +1, -1, -1, +1` and `B+_k = B-_k` of unit magnitude.

| block | `<cos th^k_{l+}>` | `<cos th^k_{l-}>` (own sign) | `<cos th^k_{l+} cos th^k_{l-}>` | expected |
|---|---|---|---|---|
| `(D+,D+)` | +0.33174 +- 0.00211 | +0.32914 +- 0.00211 | +0.10819 +- 0.00140 | `+1/9 = 0.11111` |
| `(D+,D-)` | +0.33324 +- 0.00211 | -0.33636 +- 0.00211 | -0.11294 +- 0.00141 | `-1/9` |
| `(D-,D+)` | -0.33090 +- 0.00211 | +0.33163 +- 0.00211 | -0.10964 +- 0.00140 | `-1/9` |
| `(D-,D-)` | -0.33331 +- 0.00211 | -0.33102 +- 0.00211 | +0.10946 +- 0.00140 | `+1/9` |

Every single-lepton mean is `1/3` and every product is `1/9` (mean of the four
magnitudes: `0.11006 +- 0.00070`, `1.5 sigma` from `1/9`).  So the plotted mean
*is* a coefficient, up to `1/9` and a sign -- and section 6 fixes the sign
against the literature.  The unpolarised sample reads

    9 <cos th^n_+ cos th^n_-> = +0.3263 +- 0.0133
    9 <cos th^k_+ cos th^k_-> = +0.3434 +- 0.0132
    9 <cos th^r_+ cos th^r_-> = +0.0354 +- 0.0135

which is the right size for `p p -> t t~` at 13 TeV.  Note that this whole
subsection is a *calibration of the estimator*, not part of the verdict: every
conclusion below is about which entries of `rho` a coefficient touches and about
whether a contribution is *zero*, and no sign convention can move a zero.

## 4. Verdict and corrected wording

The plot is correct and should not be touched.  Two sentences of the draft
need changing.

**Replace**

> It shows where the effect actually lives: the four diagonal blocks and the
> four mixed blocks with a single `I` index sit on zero, and the entire `C_nn`
> signal is carried by the `(I,I)` block, in which *both* helicity indices are
> off-diagonal.

**by**

> It shows where the effect actually lives.  The four diagonal blocks are the
> tall curves: they carry the whole cross section, `23.755 +- 0.056` pb of
> `23.765` pb, so on this pane they are the last thing that sits on zero.  What
> vanishes for them is not the distribution but its *first moment* -- each of
> their curves is symmetric about `cos theta^n_{l+} cos theta^n_{l-} = 0`, and
> summed they contribute `+0.00018 +- 0.00079` to
> `<cos theta^n_{l+} cos theta^n_{l-}>`, zero at `0.2 sigma`.  The four mixed
> blocks with a single `I` index are zero in the stronger sense, bin by bin, and
> the lower panel shows it.  The entire `C_nn` signal is therefore carried by
> the `(I,I)` block, in which *both* helicity indices are off-diagonal: it
> supplies `+0.03654 +- 0.00056` of the total `+0.03679 +- 0.00098`.  The
> control is `plots/blocks_ckk.pdf`, the same nine samples for `C_kk`, where the
> roles are exactly reversed.

**Replace**

> `C_nn` ... is built **entirely** from off-diagonal entries of the production
> density matrix.

**by**

> The *coefficient* `C_nn` is built entirely from the **doubly** off-diagonal
> entries of the production density matrix.  In the helicity basis `sigma_n` is
> transverse to the quantisation axis, so
> `C_nn = Tr[rho sigma_n x sigma_n] = 2 Re rho(+-,-+) - 2 Re rho(++,--)`, and
> both entries flip the helicity of the top *and* of the antitop; the diagonal
> entries do not enter.  `C_kk` is the opposite,
> `C_kk = rho(++,++) - rho(+-,+-) - rho(-+,-+) + rho(--,--)`, purely diagonal.
> This is a statement about the coefficient and not about the distribution: a
> block of definite helicities still has a cross section, and still populates
> the full range of `cos theta^n_{l+} cos theta^n_{l-}`.  What vanishes for it
> is the mean.

(Strictly, the `2 Re rho(+-,-+) - 2 Re rho(++,--)` form also assumes the
helicity spinors are phased so that `sigma_r = sigma_x` and `sigma_n =
sigma_y`; see section 5.2.  If that is a distraction in the paper, the
phase-robust version is `C_rr +- C_nn = 4 Re rho(+-,-+)` and
`4 Re rho(++,--)`, and the double-flip statement itself needs no convention at
all.)

If a citation is wanted for that paragraph, Baumgart and Tweedie
(`arXiv:1212.4888`, Sec. 2) is the published statement of it: they show the
transverse block of `C` enters only as azimuthal modulations "resulting from
the interference of different helicity channels", as against the "relative
probabilities" that `C_kk` measures.  The `(k, n, r)` basis itself is
Bernreuther, Heisler and Si (`arXiv:1508.05271`, Eq. (5) and Eq. (45)), which
is what this code implements.

**Optional, but it is what caused the trouble.**  The pane's x-axis label reads
`cos theta^n_{l+} cos theta^n_{l-}  (C_nn)`, which names a per-event product as
if it were the coefficient.  Either drop the parenthesis or make it
`(-> C_nn)`, and say in the caption that `C_nn` is the *mean* of the plotted
variable, not the histogram.  The same applies to `RESULTS.md`, which writes
`<C_nn>` throughout for `<cos theta^n_{l+} cos theta^n_{l-}>`.

## 4b. Where the error is, and where it is not

`RESULTS.md` section 4b is **correct as written**.  It says only

> For `C_nn` and `cos phi_ll` the four blocks with a *single* interference
> factor are flat at zero and the whole effect is the doubly-interfering
> `(I,I)` block

-- a statement about the five interference blocks, which is exactly what the
lower panel of the pane shows and exactly what the table above confirms.  The
draft section went one step further and extended "sit on zero" to the four
diagonal blocks as well.  That step is the error; nothing upstream of it is.

## 5. The literature: is `C_nn` really a pure double-flip quantity?

Yes, and the `(k, n, r)` axes in this code are the standard ones.  Everything in
this section was read from the arXiv sources; where something could not be
confirmed it is said so.

### 5.1 The basis -- an exact match, no convention gap

**W. Bernreuther, D. Heisler, Z.-G. Si, "A set of top quark spin correlation and
polarization observables for the LHC", JHEP 12 (2015) 026, `arXiv:1508.05271`.**
Eq. (5) defines

    {r, k, n}:   r = (1/r)(p - y k),   n = (1/r)(p x k),
                 y = k.p,   r = sqrt(1 - y^2)

with `k` the top direction in the `t tbar` ZMF and `p` the direction of one of
the incoming partons.  Eq. (45) is the LHC version with `p_p = (0,0,1)`, and
Table 5 supplies the Bose-symmetry factor `sign(y_p)`.  `analyse_interference.py`
lines 273-274 build exactly this:

```python
rhat = sgn[:, None] * (zhat - cosT[:, None] * khat) / sinT[:, None]
nhat = sgn[:, None] * np.cross(zhat, khat) / sinT[:, None]
```

with `sgn = sign(k.z)`.  **CMS, `arXiv:1907.03729` Sec. 2 and Eq. (4)** uses the
identical `n = (p x k)/sin(Theta)`, `r = (p - k cos Theta)/sin(Theta)` and the
identical `sign(cos Theta)` flip.  **Afik and Munoz de Nova,
`arXiv:2003.02280` Sec. II.2** write `n = r x k`, which is the same vector.
So there is **no** `n`-sign gap between this code and the standard references --
and in any case `sgn` cancels in `C_nn`, which is quadratic in `n`.

*(Caution as promised: **ATLAS `arXiv:1903.07570`** defines no basis of its own,
it delegates to `arXiv:1508.05271`.  **Mahlon and Parke `arXiv:1001.3422`** do
not use the `(k,n,r)` triad at all -- they parametrise by an angle `xi` from the
recoil direction, helicity being `cos xi = -1`.  **Uwer `hep-ph/0412097`** and
**Mahlon and Parke `hep-ph/9512264`** were not read beyond their metadata, so
nothing is attributed to them here.)*

### 5.2 The double-flip statement

The published statement closest to the claim is **M. Baumgart, B. Tweedie, "A New
Twist on Top Quark Spin Correlations", JHEP 03 (2013) 117, `arXiv:1212.4888`.**
Their Sec. 2 uses the same `1/4` Fano-Bloch form with `C^{i ibar} = <4 S^i
Sbar^ibar>`, and they show that the *transverse* block of `C` appears only as
azimuthal modulations in `phi -+ phibar`, describing them as

> the presence of spin correlations can be seen as sinusoidal modulations
> resulting from the interference of different helicity channels

and as sensitive to

> the interference between different spin configurations in a given basis,
> rather than their relative probabilities

while `C^{33} = C_kk` is the "relative probabilities" piece.  Their two
modulations map one-to-one onto the two double-flip entries: `phi - phibar`
carries `(C_rr + C_nn)/2`, i.e. `rho(+-,-+)`; `phi + phibar` carries
`(C_rr - C_nn)/2`, i.e. `rho(++,--)`.  That is the same decomposition as
`MADSPIN_SEQUENTIAL_PLAN.md` 13.15 and as section 3 above.

The algebra itself needs no citation and is done in `check_cnn_claim.py` Part A;
the cleanest way to see it is with ladder operators,

    sigma_x x sigma_x + sigma_y x sigma_y = 2( |+-><-+| + |-+><+-| )
    sigma_x x sigma_x - sigma_y x sigma_y = 2( |++><--| + |--><++| )

-- every term a *double* ladder operator.  `sigma_n` and `sigma_r` are transverse
to the quantisation axis, so each flips one unit of helicity on its own side and
the tensor product flips both.

**One caveat worth recording.**  The statement *"`C_nn` is purely
double-helicity-flip"* is free of any phase convention.  The finer statement
*"`C_nn = 2 Re rho(+-,-+) - 2 Re rho(++,--)`"* additionally assumes the helicity
spinors are phased so that `sigma_r = sigma_x` and `sigma_n = sigma_y`; a
rephasing `|-> -> e^{i alpha}|->` rotates `sigma_x` into `sigma_y` and so mixes
`C_rr` with `C_nn`.  The phase-robust objects are the combinations
`C_rr +- C_nn`.  Nothing in this note depends on the finer statement.

### 5.3 The sign of `C` -- named, because it is a real trap

There are **two** independent sign conventions in the literature, and they are
opposite:

* **Experimental / Bernreuther convention.**  `arXiv:1508.05271` Eq. (40) writes
  the double-differential distribution with an explicit **minus**,
  `1/4(1 + B_1 cos th_+ + B_2 cos th_- - C cos th_+ cos th_-)`, and its Table 5
  measures the *antitop* angle against `b = -a` (for all three axes: `(a,b) =
  (k, -k)`, `(sign(y_p) n_p, -sign(y_p) n_p)`, likewise for `r`).  CMS
  `arXiv:1907.03729` Sec. 2 states the motivation in words -- *"The negative sign
  in front of the matrix C is chosen to define same-helicity top quarks as having
  positive spin correlation"* -- and CMS `arXiv:2512.17557` Eq. (1) writes the
  `-C_ij` explicitly in the density matrix.
* **Quantum-information convention.**  Afik and Munoz de Nova `arXiv:2003.02280`
  Eq. (2) and ATLAS `arXiv:2311.07288` use `rho = 1/4[... + C_ij sigma_i x
  sigma_j]` with a **plus**, so `D = tr[C]/3` and entanglement means `D < -1/3`.
  In that convention `C_nn` is *negative* for `t tbar` (`-1` for the threshold
  `gg` singlet).

This code projects **both** leptons on the **same** `n`, i.e. it does *not* apply
Bernreuther's `b = -a` reversal.  Combining that with Eq. (40)'s explicit `-C`,
the two flips cancel and

    < (u_+ . n)(u_- . n) >  =  + C(n,n)_{ATLAS/CMS} / 9  =  - C_nn^{rho} / 9 .

So the code's `9 <cnn> = +0.3263 +- 0.0133` is directly comparable to the
*experimental* `C_nn`, and it should be *minus* the density-matrix `C_nn`.

### 5.4 The numerical cross-check that ties it together

| quantity | this run (LO, 13 TeV) | published |
|---|---|---|
| `C_nn` | **+0.3263 +- 0.0133** | `+0.329 +- 0.020` (CMS `arXiv:1907.03729`); `0.320 +- 0.002` (ATLAS `arXiv:1903.07570`) |
| `C_kk` | +0.3434 +- 0.0132 | `+0.300 +- 0.038` (CMS); `0.314 +- 0.002` (ATLAS) |
| `C_rr` | +0.0354 +- 0.0135 | `+0.081 +- 0.032` (CMS); `0.050 +- 0.002` (ATLAS) |
| `D = -(C_kk+C_rr+C_nn)/3` | **-0.2350 +- 0.0077** | **`-0.237 +- 0.011`** (CMS) |

`D` agrees at `0.15 sigma`.  `D = -3 <cos phi_ll>` is Bernreuther Eq. (55) /
CMS Eq. (12) (`dsigma/dcos phi = 1/2 (1 - D cos phi)`), and it is computed here
from the *same* `<cos phi_ll> = 0.07834 +- 0.00255` that `plot_interference.py`
already reports.  The individual `C_kk` and `C_rr` trade against each other
between LO and the unfolded measurement while their sum is preserved, which is
why `D` is the sharp comparison.

**Not verified, stated as such:** a fetch of Bernreuther Table 7 returned
`C(k,k) = 0.559` at 13 TeV, which does not sit well with either experiment's
`~0.31`; I could not resolve whether that is a mis-extraction or a different
quantity, so **no BHS Table 7 number is relied on here**.  The comparison above
uses only the two experimental papers, whose numbers were confirmed against
their own quoted `D`.

**None of section 5 changes the verdict.**  It establishes that the observable is
the standard one and that the coefficient really is a pure double-flip quantity.
The verdict in section 4 rests on section 2 (a measurement) and section 3 (an
identity), neither of which a sign convention can touch.

## 6. Files

* `check_cnn_claim.py` -- the algebra, the table and the figure.
* `plots/claim_cnn_numbers.txt` -- the full printed output.
* `plots/claim_cnn_vs_ckk.pdf`, `.png` -- three panels: (a) the integral of each
  block's curve on the `C_nn` pane, (b) each block's contribution to
  `<cos theta^n cos theta^n>`, (c) the same for `C_kk`.
