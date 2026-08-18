# Results

Base: `30e0596f7` (merge of #355 into `claude/ms-density-pol-restrict`, the
branch of PR #349), on top of `fa6df97e8` / `11e07a6c4`.  Nothing else merged in.

13 TeV, LO, `p p > t t~`, NNPDF23LO (`nn23lo1`), `me_frame = 1,2` (partonic CM).
MadSpin `spinmode = onshell` (a density spinmode), `BW_cut = 15`,
`max_weight_ps_point = 400`, `decay t > b w+, w+ > l+ vl` and the conjugate with
`l = e, mu`.  **50 000 unweighted events per sample, 250 000 in total.**
`iseed = 4321` and MadSpin `seed = 7777` for all five.

`tests/test_manager.py test_madspin -t0`: **150 tests, OK.**

## 1. Total-rate closure -- exact

The sum over a complete helicity basis of `|M_i|^2` *is* the unpolarised
`|M|^2`; no off-diagonal term enters a rate.  This must therefore close exactly,
and it does.

| sample | production `sigma` [pb] | after MadSpin [pb] |
|---|---|---|
| `p p > t{+} t~{+}` | 168.6357 +- 0.0860 | 7.93743 +- 0.00405 |
| `p p > t{+} t~{-}` |  83.6570 +- 0.0441 | 3.93749 +- 0.00207 |
| `p p > t{-} t~{+}` |  83.6580 +- 0.0483 | 3.93875 +- 0.00228 |
| `p p > t{-} t~{-}` | 168.6357 +- 0.0860 | 7.93743 +- 0.00405 |
| **sum of the four** | **504.5864 +- 0.1381** | **23.75109 +- 0.00650** |
| `p p > t t~` (unpolarised) | 504.5860 +- 0.2805 | 23.75670 +- 0.01321 |
| **sum / unpolarised** | **1.000001 +- 0.000620  (0.00 sigma)** | **0.999764 +- 0.000620  (-0.38 sigma)** |

Closure at the level of **6e-4** (the statistical precision of the comparison),
before and after the decay.  The agreement *after* MadSpin additionally shows
that the restricted trace `Tr(rho)|restricted` is what normalises the polarised
accept/reject: the four polarised samples reproduce the unpolarised branching
fraction to 2.4e-4.

`sigma(t{+}t~{+}) = sigma(t{-}t~{-})` and `sigma(t{+}t~{-}) = sigma(t{-}t~{+})`
to every printed digit, as parity requires.

## 2. Each polarised sample carries the polarisation it was asked for

Analytic check, independent of the closure: for a top of definite helicity `h`
the charged lepton (`alpha_l = 1`) follows `(1 + h cos theta)/2` in the top rest
frame about the parent direction **in `me_frame`**, so `<cos theta> = h/3`.

| sample | `<cos theta_k(l+)>` | `<cos theta_k(l-)>` | `<cos_k+ * cos_k->` | product of the two singles |
|---|---|---|---|---|
| `t{+} t~{+}` | +0.3329 +- 0.0021 | -0.3331 +- 0.0021 | +0.1124 +- 0.0014 | +0.1109 |
| `t{+} t~{-}` | +0.3310 +- 0.0021 | +0.3351 +- 0.0021 | -0.1114 +- 0.0014 | -0.1109 |
| `t{-} t~{+}` | -0.3344 +- 0.0021 | -0.3319 +- 0.0021 | -0.1106 +- 0.0014 | -0.1110 |
| `t{-} t~{-}` | -0.3349 +- 0.0021 | +0.3312 +- 0.0021 | +0.1110 +- 0.0014 | +0.1109 |
| unpolarised | +0.0060 +- 0.0026 | -0.0028 +- 0.0026 | +0.0383 +- 0.0015 | -- |

Every entry is within ~1 sigma of the analytic `+-1/3`, and inside each fully
polarised sample the two leptons **factorise** (last two columns agree), i.e.
no spin correlation survives once the production helicities are fixed -- which
is precisely the statement that a single diagonal entry of `rho` has been
selected.  The unpolarised sample has zero net polarisation, as QCD requires,
and keeps its `C_kk`.

## 3. Sum of the four vs. the unpolarised sample

`chi2` of the per-bin ratio against 1, 20 bins, statistical errors only.  The
error on the sum adds the two members of each parity pair linearly (see
README); the uncorrelated value is in `plots/closure_numbers.txt`.

| observable | `chi2`/20 | max abs(ratio-1) | mean stat. err. per bin | expected | plot |
|---|---|---|---|---|---|
| `cos theta_k(l+)` | **24.6** | 0.050 | 2.4% | closes | `closure_cos_k_p.png` |
| `cos theta_k(l-)` | **12.1** | 0.050 | 2.4% | closes | `closure_cos_k_m.png` |
| `C_kk` product | **8.4** | 0.109 | 4.0% | closes | `closure_ckk.png` |
| `cos theta_n(l+)` | **23.0** | 0.080 | 2.5% | closes (single-particle) | `closure_cos_n_p.png` |
| `C_rr` product | **11.4** | 0.067 | 3.8% | see below | `closure_crr.png` |
| `C_nn` product | **398.4** | **0.890** | 4.8% | **does not close** | `closure_cnn.png` |
| `cos phi_ll` | **159.4** | 0.150 | 2.5% | **does not close** | `closure_cos_phi.png` |
| `Delta phi(l+,l-)` lab | **245.0** | 0.142 | 2.5% | **does not close** | `closure_dphi_lab.png` |
| `pT(t)` (control) | **13.8** | 0.209 | 4.3% | closes | `closure_pt_t.png` |
| `m(t t~)` (control) | **17.6** | 0.195 | 4.9% | closes | `closure_m_tt.png` |

(`max abs(ratio-1)` for the two controls is a single sparse tail bin; the
`chi2` is the meaningful number there.)

Overview: `plots/closure_summary.png`.

### 3a. What closes

Every observable that is built only from the **diagonal** of `rho` is flat at 1
within statistics:

* the single-lepton helicity angles -- `chi2/ndf` of 1.2 and 0.6, per-bin
  agreement at the 2.4% level;
* the `k`-`k` spin correlation, `chi2/ndf = 0.4`;
* both **control** distributions, `pT(t)` and `m(t t~)`, `chi2/ndf` of 0.7 and
  0.9.  Nothing in the machinery -- the reweighting normalisation, the frame
  boost, the cross-section bookkeeping -- distorts the production kinematics.

Mean values:

    <cos theta_k(l+)>   sum -0.00126 +- 0.00157   unpol +0.00599 +- 0.00259
    <C_kk>              sum +0.03783 +- 0.00105   unpol +0.03829 +- 0.00148   (0.25 sigma)
    <pT(t)>             sum  120.539 +- 0.237     unpol  120.178 +- 0.356     (0.84 sigma)
    <m(t t~)>           sum  526.928 +- 0.560     unpol  525.944 +- 0.805     (1.00 sigma)

### 3b. What does not close, and why

    <C_nn>          sum +0.00135 +- 0.00111   unpol +0.03859 +- 0.00147   (20.2 sigma)
    <cos phi_ll>    sum +0.03945 +- 0.00189   unpol +0.07798 +- 0.00256   (12.1 sigma)
    <Delta phi>     sum  1.82111 +- 0.00295   unpol  1.74510 +- 0.00405   (15.2 sigma)

The whole effect is the **transverse (`n`-`n`) spin correlation**, and it is
100% removed by the restriction to the diagonal, exactly as it must be:

* `sigma_n (x) sigma_n` has no diagonal matrix element in the helicity basis.
  In terms of the `(h_t, h_tbar)` density matrix,
  `<sigma_n sigma_n> = 2 Re[rho(+-,-+)] - 2 Re[rho(++,--)]`, and both entries
  have `i != j`.  Fixing both helicities kills them, so the sum of the four
  polarised samples has `<C_nn>` compatible with **zero** (`0.0014 +- 0.0011`)
  while the unpolarised sample has `0.0386 +- 0.0015`.  The bin-by-bin ratio
  in `closure_cnn.png` is a clean monotone slope from ~1.9 at `-1` to ~0.7 at
  `+1` -- the shape of removing one correlation coefficient, not noise and not
  a normalisation shift (the integral of the ratio is 1 by section 1).
* `cos phi_ll = C_kk + C_rr + C_nn` term by term, and the numbers close on
  that identity:

      sum   0.03783 + 0.00026 + 0.00135 = 0.03944   (measured 0.03945)
      unpol 0.03829 + 0.00109 + 0.03859 = 0.07797   (measured 0.07798)

  so **every bit** of the `cos phi_ll` deficit is `C_nn`; `C_kk` and `C_rr`
  contribute `-0.0005` and `-0.0008`, both compatible with zero.
* `Delta phi(l+,l-)` in the lab is the same physics seen through the boost,
  and moves in the same direction.

`C_rr` is *also* a purely off-diagonal observable, and it *is* zeroed by the
restriction -- but the SM value it is being compared against is itself
compatible with zero after integrating over the whole phase space
(`0.00109 +- 0.00150`), because `C_rr` changes sign between threshold and high
`m(t t~)`.  So `C_rr` closes numerically without being a counter-example: it
simply has no interference left to remove at this level of inclusiveness.  It is
listed as a caution, not as evidence.

The single-particle transverse angle `cos theta_n(l+)` closes because a *net*
transverse polarisation of one top is zero in QCD at LO on both sides -- the
transverse density-matrix elements show up only in the two-particle
correlation, not in the one-particle marginal.

## 4. Verdict

**The closure test passes.**  What was tested, precisely:

* the total rate closes exactly, before (0.00 sigma) and after (0.38 sigma) the
  decay, to 6e-4;
* each of the four polarised samples carries the requested helicity to the
  analytic `+-1/3`, on the `me_frame` axis (i.e. #355's axis; on the lab axis
  these numbers would be diluted by roughly a third, see `fa6df97e8`);
* every observable built from the diagonal of `rho` -- both single-lepton
  helicity angles, `C_kk` -- agrees between the sum and the unpolarised sample
  at the 2-4% per-bin statistical precision, `chi2/ndf` of 1.2, 0.6, 0.4;
* the two production-level controls agree, `chi2/ndf` 0.7 and 0.9, so the
  deviations elsewhere are not a machinery artefact;
* the deviations that *are* there sit exactly where the off-diagonal `i != j`
  terms sit, are quantitatively accounted for by a single coefficient (`C_nn`),
  and satisfy the `cos phi = C_kk + C_rr + C_nn` identity to 1e-5.

There is **no** deviation that interference does not explain.  Nothing here
blocks #349.

The whole test was then **repeated in the offshell density mode**
(`spinmode madspin` + `unweighting sequential`, 5 x 50 000 events) and gives the
same answer -- section 6.

What was **not** tested:

* `spinmode = PA` was not closure-tested (it shares
  `calculate_matrix_element_from_density` and `_frame_boost` with the two modes
  that were, so the code under test is the same).
* the default `unweighting = auto` for `spinmode = madspin` with two decaying
  particles, i.e. `joint`, could not be run to completion at all -- see
  section 5.  Everything here used `sequential` (`onshell`'s own default, and
  set explicitly for `madspin`).
* LO only, one process, no polarised beams, no `{T}`/`{0}` (vector) braces, no
  brace on a particle MadSpin does not decay.
* statistical precision of the ratio is 2.4-4.9% per bin; a sub-percent
  interference effect in the "closing" observables would not be visible.

## 5. Side finding (does not block #349): the max-weight bound and polarisation

Accept/reject trials per written event, identical cards otherwise:

| production | `spinmode madspin` (offshell, `unweighting=joint`) | `spinmode onshell` (`unweighting=sequential`) |
|---|---|---|
| `p p > t t~` | 4.05 | 4.11 |
| `p p > t{+} t~{+}` | 213 | 4.43 |
| `p p > t{-} t~{-}` | 204 | 4.44 |
| `p p > t{+} t~{-}` | -- | 4.45 |
| `p p > t{-} t~{+}` | 5800 - 6300 | 4.47 |

The `t{-} t~{+}` figure was **1066** in an earlier `madspin` run that differed
from the 5800-6300 one *only in `nevents`*, i.e. only in the size of the
max-weight probe (`Nevents_for_max_weight = max(75, 3 nevents^(1/3))`,
`nb_sigma = max(4.5, log_7.7 nevents)`).  A bound that moves by a factor 6 with
the probe size is being set by a long tail, not by a stable maximum.  At
6 000 trials/event the run needs ~120 M decay-pool events per particle and does
not finish: this is what forced the switch to `onshell`.

Two things differ between those two columns -- `madspin` is offshell *and*
selects `unweighting = joint` for two decaying particles, while `onshell`
selects `unweighting = sequential` (both via `auto`).  **It is the unweighting
scheme, not the offshellness.**  Measured directly, `p p > t{-} t~{+}`,
2 000 events, `spinmode madspin` (still offshell) with an explicit
`set unweighting sequential`:

    MadSpin unweight efficiency: 0.1164 (2000 written / 17177 trials, 8.59 trials/event)

**8.59** against **5800-6300** for the same offshell mode with the joint
scheme -- a factor ~700 recovered by changing only the accept/reject
organisation.  The joint bound is one number covering the whole `(mass set) x
(decay of t) x (decay of t~)` space at once; the sequential scheme bounds each
decaying particle separately, so a long tail in one factor no longer inflates
the bound seen by the others.  Restricting `rho_prod` to a single helicity makes
the joint weight much more sharply peaked, and that is what the joint bound
cannot follow.

This is a **CPU cost, not a bias**: an over-estimated max-weight bound is always
safe, and no `nb_overflow` warning was emitted in any run.  The actionable
follow-up is small and belongs with #349: `unweighting = auto` should prefer
`sequential` when the production carries a polarisation brace, the way it
already does under `PA`/`onshell`.

Section 6 repeats the whole closure test in `spinmode madspin` +
`unweighting sequential`, i.e. in the default offshell density mode, to confirm
that none of the conclusions depend on the mode.

## 6. The same test in the offshell density mode

`spinmode madspin` with `set unweighting sequential`, 5 x 50 000 events, same
seeds, same cards otherwise.  Efficiency: 6.67 trials/event unpolarised,
9.14 / 9.00 / 9.26 / 9.45 for `t{+}t~{+}` / `t{+}t~{-}` / `t{-}t~{+}` /
`t{-}t~{-}` -- a factor 1.4, not a factor 1000.

Total-rate closure:

| | sum of the four | unpolarised | ratio |
|---|---|---|---|
| production | 504.3692 +- 0.1556 pb | 504.3240 +- 0.2830 pb | 1.000090 +- 0.000640 (0.14 sigma) |
| after MadSpin | 23.74478 +- 0.00732 pb | 23.73997 +- 0.01332 pb | 1.000203 +- 0.000641 (0.32 sigma) |

Per-bin ratio, `chi2` over 20 bins:

| observable | `chi2`/20 | onshell (section 3) |
|---|---|---|
| `cos theta_k(l+)` | 28.4 | 24.6 |
| `cos theta_k(l-)` | 21.2 | 12.1 |
| `C_kk` | 14.0 | 8.4 |
| `cos theta_n(l+)` | 15.1 | 23.0 |
| `C_rr` | 16.3 | 11.4 |
| **`C_nn`** | **363.8** | **398.4** |
| **`cos phi_ll`** | **174.0** | **159.4** |
| **`Delta phi(l+,l-)`** | **192.2** | **245.0** |
| `pT(t)` (control) | 17.3 | 13.8 |
| `m(t t~)` (control) | 25.7 | 17.6 |

`<C_nn>` is again removed in full -- `0.00079 +- 0.00111` for the sum against
`0.03574 +- 0.00149` unpolarised, 18.8 sigma -- and the
`cos phi = C_kk + C_rr + C_nn` identity again closes to 1e-5:

    sum   0.03783 - 0.00043 + 0.00079 = 0.03819   (measured 0.03819)
    unpol 0.03852 + 0.00408 + 0.03574 = 0.07834   (measured 0.07834)

Same conclusion, in the mode users get by default.  Plots in `plots_offshell/`,
numbers in `plots_offshell/closure_numbers.txt`.

### Caveat found in this pass: the sequential bound under-covers for polarised

MadSpin's own diagnostic fired in the *offshell + sequential* runs -- and only
there:

    p p > t t~        CRITICAL: MadSpin sequential:  2 weights exceeded their per-particle maximum
    p p > t{+} t~{+}  CRITICAL: MadSpin sequential: 91 weights exceeded ...
    p p > t{+} t~{-}  CRITICAL: MadSpin sequential: 82 weights exceeded ...
    p p > t{-} t~{+}  CRITICAL: MadSpin sequential: 86 weights exceeded ...
    p p > t{-} t~{-}  CRITICAL: MadSpin sequential: 86 weights exceeded ...

out of 50 000 written events each, i.e. **0.17% of events in the polarised
samples against 0.004% unpolarised** -- a factor 43.  The five `onshell` runs of
section 3 emitted **no** overflow at all.

This is the *same* underlying observation as section 5, seen from the other
side: the polarised weight distribution has a longer tail than the unpolarised
one, and the `mean + nb_sigma*sd` max-weight estimator does not track it.  Under
the joint scheme it over-covers by a factor 50-1500 (unusable CPU); under the
sequential scheme it under-covers on ~0.2% of events (small bias, correctly
flagged).  The size of the resulting bias is bounded by that 0.17% and is
invisible at the statistics here -- section 6 reproduces section 3 within
errors on every observable -- but the tuning of `nb_sigma` /
`Nevents_for_max_weight` for a polarised production is the concrete follow-up
this test identifies.

The section 3 result, which is the one quoted in the verdict, is free of it.
