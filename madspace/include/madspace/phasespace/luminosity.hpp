#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/invariants.hpp"

namespace madspace {

/**
 * PDF-convolution mapping: parton momentum fractions from two random numbers.
 *
 * Trades the parton momentum fractions @f$x_1, x_2@f$ for the partonic
 * invariant @f$\hat s = x_1 x_2 s_\mathrm{lab}@f$ and one angular variable [1].
 * @f$\hat s@f$ is drawn in `[s_hat_min, s_hat_max]` with an @ref Invariant,
 * which forwards @p invariant_power, @p mass and @p width. The momentum
 * fractions then follow from logarithmic sampling,
 *
 * @f[
 *   x_1 = \tau^{\,r}, \qquad x_2 = \tau^{\,1-r},
 *   \qquad \tau = \hat s / s_\mathrm{lab},
 * @f]
 *
 * which cancels the @f$1/x_1@f$ of the convolution measure. Choosing
 * @p invariant_power @f$\simeq 1@f$ (the default) also absorbs the @f$1/\hat s@f$
 * flux factor. A nonzero @p width instead resolves an s-channel resonance in
 * @f$\hat s@f$ directly (Sec. 2.2.2 of [1]). This parametrization is specific
 * to hadron collisions.
 *
 * `batch` is the leading batch dimension.
 *
 * **Inputs**
 * - `r_s` – `float`, shape `(batch,)` – random number for @f$\hat s@f$.
 * - `r_x` – `float`, shape `(batch,)` – random number for the @f$x_1/x_2@f$
 *   split.
 *
 * **Conditions**
 * - None.
 *
 * **Outputs**
 * - `x1` – `float`, shape `(batch,)` – first parton momentum fraction.
 * - `x2` – `float`, shape `(batch,)` – second parton momentum fraction.
 * - `s_hat` – `float`, shape `(batch,)` – partonic invariant
 *   @f$\hat s = x_1 x_2 s_\mathrm{lab}@f$.
 *
 * In addition every mapping returns a `det` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2.2)
 */
class Luminosity : public Mapping {
public:
    /**
     * @param s_lab           Hadronic invariant @f$s_\mathrm{lab}@f$ (squared
     *                        collider energy).
     * @param s_hat_min       Lower bound on @f$\hat s@f$, set by the final-state
     *                        masses and analysis cuts.
     * @param s_hat_max       Upper bound on @f$\hat s@f$; `0` uses
     *                        @f$s_\mathrm{lab}@f$.
     * @param invariant_power Exponent of the @f$1/\hat s^{\,p}@f$ sampling of
     *                        @f$\hat s@f$; `1` absorbs the flux factor. See
     *                        @ref Invariant.
     * @param mass            Resonance mass for @f$\hat s@f$; see @ref Invariant.
     * @param width           Resonance width; a nonzero value selects
     *                        Breit-Wigner sampling of @f$\hat s@f$. See
     *                        @ref Invariant.
     */
    Luminosity(
        double s_lab,
        double s_hat_min,
        double s_hat_max = 0,
        double invariant_power = 1,
        double mass = 0,
        double width = 0
    );

private:
    Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;

    double _s_lab, _s_hat_min, _s_hat_max;
    Invariant _invariant;
};

} // namespace madspace
