#pragma once

#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/invariants.hpp"
#include "madspace/phasespace/topology.hpp"
#include "madspace/phasespace/two_particle.hpp"

namespace madspace {

class TPropagatorMapping : public Mapping {
public:
    TPropagatorMapping(
        const std::vector<std::size_t>& integration_order,
        double invariant_power = 0.8,
        const std::vector<double>& pt_min = {}
    );
    std::size_t random_dim() const { return 3 * _integration_order.size() - 1; }

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

    // pt^2 of the outgoing particle at position i (0 if no pt cut).
    double pt2(std::size_t i) const;

    std::vector<std::size_t> _integration_order;
    std::vector<bool> _sample_sides;
    std::vector<double> _pt_min;
    bool _has_cut;
    Invariant _uniform_invariant;
    TwoToTwoParticleScattering _com_scattering;
    TwoToTwoParticleScattering _lab_scattering;
};

} // namespace madspace
