#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * One-dimensional invertible mapping for a single phase-space invariant.
 *
 * Samples an invariant `invariant` in `[invariant_min, invariant_max]` so as to
 * flatten a propagator, a resonance, a threshold or a soft/collinear
 * enhancement [1]. The @p power / @p mass / @p width triple selects the
 * transformation @f$s = G(r)@f$.
 *
 * A nonzero @p width selects Breit-Wigner sampling around the pole @f$m^2@f$,
 *
 * @f[
 *   G_\mathrm{BW}(r) = m\Gamma\,\tan\!\big[u_1 + (u_2 - u_1)\,r\big] + m^2,
 *   \qquad u_{1/2} = \arctan\!\frac{x_{\min/\max} - m^2}{m\Gamma}.
 * @f]
 *
 * For @p width = 0 the invariant is sampled flat when @p power = 0, i.e.
 * @f$G(r) = x_{\min} + (x_{\max} - x_{\min})\,r@f$; with the logarithmic
 * @f$\nu \to 1@f$ limit when @p power = 1; and with the power law
 *
 * @f[
 *   G_\nu(r) = \Big[
 *     r\,(x_{\max} - m^2)^{1-\nu} + (1 - r)\,(x_{\min} - m^2)^{1-\nu}
 *   \Big]^{\frac{1}{1-\nu}} + m^2
 * @f]
 *
 * otherwise. A nonzero @p flat_window (with a nonzero @p width) flattens the
 * Breit-Wigner peak: the density is kept outside
 * @f$|\sqrt{s} - m| < w\,\Gamma@f$ and replaced inside by a constant, the
 * mean of its two edge values. This is the mapping of a `$`-excluded
 * propagator, whose matrix element vanishes in that window.
 *
 * In the power-law and logarithmic forms the pole is shifted to
 * @f$m^2 - 0.01@f$ (in the units of the invariant), which keeps them finite
 * at a boundary on the pole. The naive choice @f$\nu = 2@f$ is rarely
 * optimal; @ref PhaseSpaceMapping uses @f$\nu = 0.8@f$ by default.
 *
 * `batch` is the leading batch dimension.
 *
 * **Inputs**
 * - `random` – `float`, shape `(batch,)` – the unit-hypercube coordinate.
 *
 * **Conditions**
 * - `invariant_min` – `float`, shape `(batch,)` – lower integration bound.
 * - `invariant_max` – `float`, shape `(batch,)` – upper integration bound.
 *
 * **Outputs**
 * - `invariant` – `float`, shape `(batch,)` – the sampled invariant.
 *
 * In addition every mapping returns a `det` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2.1)
 */
class Invariant : public Mapping {
public:
    /**
     * @param power  Exponent @f$\nu@f$ of the @f$1/(s - m^2)^\nu@f$ power-law
     *               sampling: `0` is flat, `1` the logarithmic limit, any other
     *               value uses @f$G_\nu@f$. Ignored when @p width is nonzero.
     * @param mass   Pole / resonance mass @f$m@f$.
     * @param width  Resonance width @f$\Gamma@f$; a nonzero value selects
     *               Breit-Wigner sampling around @p mass.
     * @param flat_window  Half-width @f$w@f$, in units of @p width, of the
     *               window in which the Breit-Wigner density is flattened;
     *               `0` (the default) keeps the plain Breit-Wigner.
     */
    Invariant(
        double power = 0, double mass = 0, double width = 0, double flat_window = 0
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

    double _power, _mass, _width, _flat_window;
};

} // namespace madspace
