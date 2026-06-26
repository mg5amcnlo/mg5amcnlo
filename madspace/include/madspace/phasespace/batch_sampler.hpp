#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

class BatchSampler : public FunctionGenerator {
public:
    BatchSampler(const std::vector<NamedVector<Type>>& types);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<std::size_t> _channel_tensor_counts;
};

} // namespace madspace
