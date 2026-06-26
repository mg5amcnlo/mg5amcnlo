#pragma once

#include "madspace/driver/context.hpp"
#include "madspace/phasespace/base.hpp"

namespace madspace {

class VegasHistogram : public FunctionGenerator {
public:
    VegasHistogram(std::size_t dimension, std::size_t bin_count);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _bin_count;
};

class VegasMapping : public Mapping {
public:
    VegasMapping(
        std::size_t dimension, std::size_t bin_count, const std::string& prefix = ""
    );
    const std::string& grid_name() const { return _grid_name; }
    void initialize_globals(ContextPtr context) const;
    std::size_t dimension() const { return _dimension; }
    std::size_t bin_count() const { return _bin_count; }

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

    std::size_t _dimension;
    std::size_t _bin_count;
    std::string _grid_name;
};

void initialize_vegas_grid(ContextPtr context, const std::string& grid_name);

} // namespace madspace
