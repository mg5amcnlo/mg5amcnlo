#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/invariants.hpp"

namespace madspace {

class Luminosity : public Mapping {
public:
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
