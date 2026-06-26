#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

class Invariant : public Mapping {
public:
    Invariant(double power = 0, double mass = 0, double width = 0);

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

    double _power, _mass, _width;
};

} // namespace madspace
