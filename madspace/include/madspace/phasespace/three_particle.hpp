#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/invariants.hpp"

namespace madspace {

class ThreeBodyDecay : public Mapping {
public:
    ThreeBodyDecay(bool com);
    std::size_t random_dim() const { return 5; }

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

    bool _com;
};

class TwoToThreeParticleScattering : public Mapping {
public:
    TwoToThreeParticleScattering(
        double t_invariant_power = 0,
        double t_mass = 0,
        double t_width = 0,
        double s_invariant_power = 0,
        double s_mass = 0,
        double s_width = 0
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

    Invariant _t_invariant;
    Invariant _s_invariant;
};

} // namespace madspace
