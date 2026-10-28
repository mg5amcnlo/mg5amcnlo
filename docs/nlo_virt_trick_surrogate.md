# Fitted surrogate of the virtual for the virt-trick

In aMC@NLO the finite part of the one-loop virtual V is computed only at a
fraction f of the phase-space points (the *virt-trick*). Everywhere else a
cheap surrogate A replaces it, and the difference is reweighted:

    A + Θ(f - u) (V - A) / f        (u uniform in [0,1])

This is exact for any A. A better A gives a smaller variance in fixed order.
In MC@NLO event generation it also gives fewer negative events, because the
difference is carried by separate "virt" events, which are negative whenever
A > V.

Until now A was a grid of V/B in the Born phase space (or a polynomial fit,
`UsePolyVirtual`). This change adds a surrogate fitted per integration channel,
selected by `VirtTrickSurrogate` in `Cards/FKS_params.dat`:

| value | surrogate |
|---|---|
| 0 | the grid (or the polynomial fit), as before |
| 1 | colour-dipole surrogate |
| 2 | colour dipole + kinematic logarithms of all Born pairs (**default**) |

The per-channel fit uses the native Born interface and the PDF and bias weights
of each complete integration point. The same surrogate ratio is retained
through all folds. Born-spreading calibration uses the grid mean without
evaluating residual loops or preparing unused surrogate inputs.

The fit is stored as `VSU` rows in `mint_grids`. Fixed-order split jobs keep
one replica's coefficients and channel choices when combining grids, while
the grid or polynomial fallback continues to learn the exact virtual.
Older grids without `VSU` rows use that fallback. Start from grid setup when
changing surrogate options.

## The surrogate

With B the Born, B_kl the colour-linked Borns of the I_reg terms (already
computed in `bornsoftvirtual`), and L_kl = log(|2 p_k·p_l| / Q_ES²):

    A_1 = t1 B + t2 Σ_{k<l} B_kl L_kl + t3 Σ_{k<l} B_kl L_kl² + t4 Σ_k B_kk
    A_2 = A_1 + B Σ_{k<l} (a_kl L_kl + b_kl L_kl²)     (all Born pairs, coloured or not)

The kinematic logarithms of mode 2 matter when the Born has few colour dipoles.
For p p > W+ Z, for example, the dipole terms are functions of s-hat only.

The coefficients are fitted per channel during the grid setup by weighted least
squares. The weight is the squared integrand weight of the virtual. They are
refitted after every iteration, frozen within it, and written to `mint_grids`
for the later steps. The fit uses points where V is computed anyway and reuses
the colour-linked Borns of the I_reg terms. Separate virtual events also
evaluate those links through the native Born interface, since they skip the
integrated counterterms.

`virtual_surrogate.f90` owns the features, private fit state and `VSU` records.
It accumulates the three dipole features as each colour link is evaluated.
`fks_singular.f` supplies the native Born momenta and colour links.
`virtual_approximation.f` handles weighted observations. `mint_module.f90`
triggers refits and grid I/O at iteration boundaries.

## Fallback to the grid

The grid stays as a fallback. Once the fit and the grid have been evaluated on
the same points (out of sample for both), a channel uses the fit only if it
does better there:

- **Event generation:** the fit must have a smaller weight of negative residual,
  Σ |c| max(0, A − V). This is what produces the negative virt events.
- **Fixed order:** the fit must have a smaller residual variance.

A comparison of variances does not see the sign of the residual. In event
generation it was seen to keep the grid where the grid gave more negative
events.

## Effect

These measurements are from the [upstream implementation](https://github.com/MadGraphTeam/MadGraph7/commit/c9d4939a423af8652d63f5d4f1edcaad9678a5e6).
They have not been repeated with this branch's other integration changes.

The table gives the negative events that the trick itself adds,
E = V_ev (r_V − r_n), as a percentage of all events, at folding 1,1,1. The cost
column gives the MadLoop calls per unweighted event (grid setup, envelope and
events). It compares the previous default (grid) with the new default (mode 2).

| process | E grid → mode 2 [%] | MadLoop calls per event |
|---|---|---|
| p p > t t~ | 0.36 → 0.10 | 0.32 → 0.23 |
| p p > e+ e- j | 0.27 → 0.28 | 11.2 → 10.9 |
| p p > t t~ j | 0.36 → 0.04 | 25.8 → 18.0 |
| e+ e- > b b~ b b~ | 0.88 → 0.16 | 10.0 → 7.9 |
| p p > w+ z (4FS) | 1.87 → 0.49 | 1.96 → 0.42 |
| p p > z j j (4FS) | 0.40 → 0.09 (±0.15) | 423 → 477 |
| p p > b b~ (4FS) | 0.30 → 0.09 | 0.89 → 0.74 |

- **p p > w+ z:** the trick's negatives dominate. The total negative fraction
  drops from 0.053 to 0.041, and the CPU time per effective event improves by
  9–12%.
- **Other processes:** the trick's negatives are a small part of the total.
  The total negative fraction and the CPU per effective event change within
  errors.
- **Fixed order:** the figure of merit does not change within errors.
- **Cross sections:** they agree in every case, since the method is exact.

Samples: 3 × 50k events (t t~, W+Z, b b~), 2 × 10k (e+ e- j, t t~ j,
b b~ b b~), and 1 × 2k for Z j j.
