#pragma once

#include <array>
#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/invariants.hpp"
#include "madspace/phasespace/topology.hpp"
#include "madspace/phasespace/two_particle.hpp"

namespace madspace {

/**
 * `t`-channel part of a phase-space mapping: a chain of @f$2 \to 2@f$ blocks.
 *
 * Builds the @f$\kappa@f$ space-like momentum transfers of a topology and the
 * associated outgoing momenta [1]. Restricting to orderings that grow inward
 * from the two incoming legs reduces the @f$\kappa!@f$ possible orders to
 * @f$2^{\kappa-1}@f$. The mapping first draws @f$\kappa-1@f$ time-like
 * invariants. It then walks @f$i = 1 \dots \kappa@f$, emitting one
 * @ref TwoToTwoParticleScattering block per step and updating the incoming
 * momentum on the chosen side by subtraction (Sec. 2.2.8 of [1], following
 * [2]).
 *
 * `batch` is the leading batch dimension. `kappa` is the number of `t`-channel
 * propagators, `len(integration_order)`.
 *
 * **Inputs**
 * - `random_i` – `float`, shape `(batch,)` – the `3 * kappa - 1` random numbers
 *   (`i = 0 … 3*kappa-2`).
 *
 * **Conditions**
 * - `com_energy` – `float`, shape `(batch,)` – total collision energy.
 * - `mass_i` – `float`, shape `(batch,)` – the `kappa + 1` outgoing masses.
 * - `x1`, `x2` – `float`, shape `(batch,)` – beam momentum fractions. Present
 *   only when a rapidity bound is given (see the constructor).
 *
 * **Outputs**
 * - `momentum_i` – `float`, shape `(batch, 4)` – the `kappa + 3` momenta
 *   (incoming and outgoing) of the `t`-channel chain.
 *
 * In addition every mapping returns a `det` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2.8)
 * - [2] O. Mattelaer, K. Ostrolenk, "Speeding up MadGraph5_aMC@NLO",
 *   https://arxiv.org/abs/2102.00773
 */
class TPropagatorMapping : public Mapping {
public:
    /**
     * @param integration_order Order in which the `t`-channel invariants are
     *                          sampled; 0-based, and each step must extend the
     *                          currently sampled range at its low or high end
     *                          (see @ref Topology).
     * @param invariant_power   Exponent of the `1/|t|^p` sampling of every
     *                          momentum transfer; see @ref Invariant.
     * @param pt_min            Per-outgoing-particle minimum transverse
     *                          momentum; empty disables the cut. See @ref Cuts.
     * @param y_max             Per-outgoing-particle bound on the absolute lab
     *                          rapidity, negative (or empty) for none. Only the
     *                          first scattering of the chain, between the two
     *                          beams, can use it: there |t| fixes both
     *                          light-cone components of the peeled particle,
     *                          so its bound and that of the recoil (the bound
     *                          of all other particles, if each has one) narrow
     *                          the |t| range exactly. The mapping then takes
     *                          the beam momentum fractions `x1`, `x2` (`float`,
     *                          shape `(batch,)`) as two further conditions.
     */
    TPropagatorMapping(
        const std::vector<std::size_t>& integration_order,
        double invariant_power = 0.8,
        const std::vector<double>& pt_min = {},
        const std::vector<double>& y_max = {}
    );
    /// Number of uniform random inputs consumed by the forward mapping,
    /// equal to `3 * len(integration_order) - 1`.
    std::size_t random_dim() const { return 3 * _integration_order.size() - 1; }
    /// Whether the first scattering is narrowed by rapidity bounds (and the
    /// mapping therefore takes `x1`, `x2` as conditions).
    bool has_rapidity_window() const { return _rapidity_window; }

private:
    // The public constructor with the first scattering's rapidity bounds
    // {recoil, peeled particle, beam sign} worked out once.
    TPropagatorMapping(
        const std::array<double, 3>& first_step_bounds,
        const std::vector<std::size_t>& integration_order,
        double invariant_power,
        const std::vector<double>& pt_min
    );

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

    // pt^2 of the outgoing particle at position i (0 if no pt cut).
    double pt2(std::size_t i) const;

    std::vector<std::size_t> _integration_order;
    std::vector<bool> _sample_sides;
    std::vector<double> _pt_min;
    bool _has_cut;
    bool _rapidity_window;
    Invariant _uniform_invariant;
    TwoToTwoParticleScattering _com_scattering;
    TwoToTwoParticleScattering _lab_scattering;
};

} // namespace madspace
