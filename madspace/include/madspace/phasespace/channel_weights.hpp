#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/topology.hpp"
#include "madspace/util.hpp"

namespace madspace {

class PropagatorChannelWeights : public FunctionGenerator {
public:
    PropagatorChannelWeights(
        const std::vector<Topology>& topologies,
        const nested_vector3<std::size_t>& permutations,
        const nested_vector2<std::size_t>& channel_indices
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    nested_vector2<double> _momentum_factors;
    nested_vector2<me_int_t> _invariant_indices;
    nested_vector2<double> _masses;
    nested_vector2<double> _widths;
};

class SubchannelWeights : public FunctionGenerator {
public:
    SubchannelWeights(
        const nested_vector2<Topology>& topologies,
        const nested_vector3<std::size_t>& permutations,
        const nested_vector2<std::size_t>& channel_indices
    );

    std::size_t channel_count() const { return _channel_indices.size(); }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    nested_vector2<double> _momentum_factors;
    nested_vector2<double> _masses;
    nested_vector2<double> _widths;
    nested_vector2<me_int_t> _invariant_indices;
    nested_vector2<me_int_t> _on_shell;
    std::vector<me_int_t> _group_sizes;
    std::vector<me_int_t> _channel_indices;
    std::vector<me_int_t> _subchannel_indices;
};

} // namespace madspace
