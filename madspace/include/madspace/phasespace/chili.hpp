#pragma once

#include "madspace/phasespace/base.hpp"

#include <vector>

namespace madspace {

class ChiliMapping : public Mapping {
public:
    ChiliMapping(
        std::size_t n_particles,
        const std::vector<double>& y_max,
        const std::vector<double>& pt_min
    );

    std::size_t random_dim() const { return 3 * _n_particles - 2; }

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
    std::vector<double> _y_max;
    std::vector<double> _pt_min;
};

} // namespace madspace
