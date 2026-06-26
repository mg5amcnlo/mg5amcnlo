#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/mlp.hpp"

namespace madspace {

class DiscreteFlow : public Mapping {
public:
    DiscreteFlow(
        const std::vector<std::size_t>& option_counts,
        const std::string& prefix = "",
        const std::vector<std::size_t>& dims_with_prior = {},
        std::size_t condition_dim = 0,
        std::size_t subnet_hidden_dim = 32,
        std::size_t subnet_layers = 3,
        MLP::Activation subnet_activation = MLP::leaky_relu
    );
    const std::vector<std::size_t>& option_counts() const { return _option_counts; }
    std::size_t condition_dim() const { return _condition_dim; }
    void initialize_globals(ContextPtr context) const;

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
    Result build_transform(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions,
        bool inverse
    ) const;

    std::vector<std::size_t> _option_counts;
    std::size_t _condition_dim;
    std::optional<std::string> _first_prob_name;
    std::vector<MLP> _subnets;
    std::vector<bool> _dim_has_prior;
};

} // namespace madspace
