#pragma once

#include <format>
#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/phasespace.hpp"

namespace madspace {

class MultiChannelMapping : public Mapping {
public:
    MultiChannelMapping(const std::vector<std::shared_ptr<Mapping>>& mappings);

private:
    Result build_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions,
        bool inverse
    ) const;
    Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override {
        return build_impl(fb, inputs, conditions, false);
    }
    Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override {
        return build_impl(fb, inputs, conditions, true);
    }

    std::vector<std::shared_ptr<Mapping>> _mappings;
};

class MultiChannelFunction : public FunctionGenerator {
public:
    MultiChannelFunction(
        const std::vector<std::shared_ptr<FunctionGenerator>>& functions,
        bool return_batch_sizes = false
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<std::shared_ptr<FunctionGenerator>> _functions;
    bool _return_batch_sizes;
};

} // namespace madspace
