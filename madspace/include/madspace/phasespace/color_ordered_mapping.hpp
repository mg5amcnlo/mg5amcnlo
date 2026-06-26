#pragma once

#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/invariants.hpp"
#include "madspace/phasespace/three_particle.hpp"
#include "madspace/phasespace/topology.hpp"
#include "madspace/phasespace/two_particle.hpp"

namespace madspace {

class ColorOrderedMapping : public Mapping {
public:
    // color_order: 0-indexed permutation of {0, ..., n-1} (n = n_out + 2).
    // Particles 0 and 1 are the two incoming beams.
    ColorOrderedMapping(
        const std::vector<std::size_t>& color_order,
        double t_invariant_power = 0.8,
        double s_invariant_power = 0.8
    );

    std::size_t random_dim() const { return _random_dim; }

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

    // 0-indexed outgoing-particle indices (values in {0,...,n_out-1}).
    // _set1 contains the outgoing particles attached to beam 0's side,
    // _set2 those attached to beam 1's side, in peel order.
    std::vector<std::size_t> _set1;
    std::vector<std::size_t> _set2;
    std::size_t _n_out;
    std::size_t _random_dim;
    // True iff exactly one of (set1, set2) has size 1 (and the other >= 2).
    // In that case the central block is DoubleT instead of 2->2.
    bool _use_double_t;
    // True iff one of (set1, set2) is empty, i.e. particles 0 and 1 are
    // adjacent in the color order and all outgoing particles sit on one side.
    // In that case there is no central block at all: the full final state is
    // produced as a single t-channel chain seeded directly off the beams.
    bool _use_single_chain;

    Invariant _uniform_invariant;
    TwoToTwoParticleScattering _com_scattering;
    TwoToTwoParticleScattering _lab_scattering;
    TwoToThreeParticleScattering _two_to_three;
    DoubleT _double_t;
};

} // namespace madspace
