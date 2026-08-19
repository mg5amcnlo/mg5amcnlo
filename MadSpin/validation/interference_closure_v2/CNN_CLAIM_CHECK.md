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

`plot_interference.py:395` then draws, for each of the nine samples,

```python
ax.hist(x=ctr, weights=s / wid, ...)
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

| block | observable | `chi2` of `h(-X)` vs `h(+X)`, 10 pairs |
|---|---|---|
| `(D+,D+)` | `C_nn` | 10.8 / 10 |
| `(D-,D-)` | `C_nn` | 15.8 / 10 |
| `(D+,D+)` | `C_kk` | **5302.1 / 10** |

The diagonal blocks are symmetric in `X` for `C_nn` -- large everywhere, zero
first moment -- and grossly asymmetric for `C_kk`.  This is visible on the
pages themselves: compare the diagonal curves in `plots/blocks_cnn.pdf` (peaked
and symmetric about 0) with `plots/blocks_ckk.pdf` (the same blocks, visibly
skewed).

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

**Optional, but it is what caused the trouble.**  The pane's x-axis label reads
`cos theta^n_{l+} cos theta^n_{l-}  (C_nn)`, which names a per-event product as
if it were the coefficient.  Either drop the parenthesis or make it
`(-> C_nn)`, and say in the caption that `C_nn` is the *mean* of the plotted
variable, not the histogram.  The same applies to `RESULTS.md`, which writes
`<C_nn>` throughout for `<cos theta^n_{l+} cos theta^n_{l-}>`.

## 5. Files

* `check_cnn_claim.py` -- the algebra, the table and the figure.
* `plots/claim_cnn_numbers.txt` -- the full printed output.
* `plots/claim_cnn_vs_ckk.pdf`, `.png` -- three panels: (a) the integral of each
  block's curve on the `C_nn` pane, (b) each block's contribution to
  `<cos theta^n cos theta^n>`, (c) the same for `C_kk`.
