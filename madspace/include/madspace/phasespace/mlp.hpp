#pragma once

#include "madspace/driver/context.hpp"
#include "madspace/phasespace/base.hpp"

namespace madspace {

class MLP : public FunctionGenerator {
public:
    enum Activation { relu, leaky_relu, elu, gelu, sigmoid, softplus, linear };
    MLP(std::size_t input_dim,
        std::size_t output_dim,
        std::size_t hidden_dim = 32,
        std::size_t layers = 3,
        Activation activation = leaky_relu,
        const std::string& prefix = "");

    std::size_t input_dim() const { return _input_dim; }
    std::size_t output_dim() const { return _output_dim; }
    void initialize_globals(ContextPtr context) const;
    std::string last_layer_bias_name() const {
        return prefixed_name(_prefix, std::format("layer{}.bias", _layers));
    }
    std::vector<std::string> global_names() const;

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _input_dim;
    std::size_t _output_dim;
    std::size_t _hidden_dim;
    std::size_t _layers;
    Activation _activation;
    std::string _prefix;
};

} // namespace madspace
