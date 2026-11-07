#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * Phase-space mapping based on the RAMBO algorithm.
 *
 * Generates the momenta of an @p n_particles final state from a unit hypercube
 * by recursively factorizing the @f$n@f$-body phase space into @f$1\to2@f$
 * decays, following RAMBO [2] and its invertible RAMBO-on-diet variant [3].
 * The RAMBO-on-diet step maps the mass variables @f$y_i@f$ to random numbers
 * @f$r_i@f$ as
 *
 * @f[
 *   r_i = (k+1)\,y_i - k\,y_i^{1+1/k}, \qquad k = n-i,
 * @f]
 *
 * which is exactly flat but has no closed-form inverse for @f$n>3@f$.
 * FastRambo replaces it by a one-parameter rational-quadratic function [4]
 *
 * @f[
 *   G_\mathrm{RQF}(y_i, c_k) =
 *   \frac{y_i^2 + c_k\,y_i(1-y_i)}{1 + (c_k-2)\,y_i(1-y_i)},
 * @f]
 *
 * The inverse and Jacobian of this map are closed-form, so the mapping stays
 * strictly invertible while being fully analytic and vectorizable. Each
 * @f$c_k@f$ is fitted once to approximate the RAMBO-on-diet step. For
 * @f$c_1 = 2@f$ the two agree exactly. The weight then acquires a small
 * non-uniform factor @f$w_n(r)@f$. Massive final states (@p massless false) are
 * obtained by reweighting a massless point rather than rescaling momenta. See
 * [1], Sec. 2.3, for the derivation.
 *
 * Entries marked with an index `i` are repeated for `i = 0 … n - 1`. `batch` is
 * the batch dimension.
 *
 * **Inputs**
 * - `random_i` – `float`, shape `(batch,)` – the `3 * n_particles - 4` uniform
 *   random numbers in [0, 1) consumed by the mapping.
 * - `com_momentum` – `float`, shape `(batch, 4)` – total incoming four-momentum.
 *   Present only when @p com is false.
 *
 * **Conditions**
 * - `com_energy` – `float`, shape `(batch,)` – total center-of-mass energy.
 * - `mass_i` – `float`, shape `(batch,)` – final-state masses. Present only
 *   when @p massless is false.
 *
 * **Outputs**
 * - `momentum_i` – `float`, shape `(batch, 4)` – the `n_particles` final-state
 *   four-momenta.
 *
 * In addition every mapping returns a `det` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] R. Kleiss, W. J. Stirling, S. D. Ellis, Comput. Phys. Commun. 40 (1986)
 *   359, https://doi.org/10.1016/0010-4655(86)90119-0
 * - [3] S. Plätzer, "RAMBO on diet", https://arxiv.org/abs/1308.2922
 * - [4] C. Durkan et al., "Neural spline flows",
 *   https://arxiv.org/abs/1906.04032
 */
class FastRamboMapping : public Mapping {
public:
    /**
     * Construct the mapping for a fixed multiplicity.
     *
     * @param n_particles Number of final-state particles; must be in [3, 12].
     * @param massless    If true all final-state particles are massless,
     *                    otherwise the masses are read from the conditions.
     * @param com         If true the momenta are generated in the
     *                    center-of-mass frame, otherwise the total incoming
     *                    momentum is taken from the `com_momentum` input.
     */
    FastRamboMapping(std::size_t n_particles, bool massless, bool com = true);

    /// Number of uniform random inputs consumed by the forward mapping,
    /// equal to `3 * n_particles - 4`.
    std::size_t random_dim() const { return 3 * _n_particles - 4; }

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

    std::size_t _n_particles;
    bool _massless;
    double _com;
};

} // namespace madspace
