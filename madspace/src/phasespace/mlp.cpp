#include "madspace/phasespace/mlp.hpp"

#include <format>
#include <random>

using namespace madspace;

namespace {

Value build_layer(
    FunctionBuilder& fb,
    Value input,
    int input_dim,
    int output_dim,
    MLP::Activation activation,
    const std::string& prefix,
    int layer_index
) {
    auto weight = fb.global(
        prefixed_name(prefix, std::format("layer{}.weight", layer_index)),
        DataType::dt_float,
        {output_dim, input_dim}
    );
    auto bias = fb.global(
        prefixed_name(prefix, std::format("layer{}.bias", layer_index)),
        DataType::dt_float,
        {output_dim}
    );
    auto linear_out = fb.matmul(input, weight, bias);
    switch (activation) {
    case MLP::relu:
        return fb.relu(linear_out);
    case MLP::leaky_relu:
        return fb.leaky_relu(linear_out);
    case MLP::elu:
        return fb.elu(linear_out);
    case MLP::gelu:
        return fb.gelu(linear_out);
    case MLP::sigmoid:
        return fb.sigmoid(linear_out);
    case MLP::softplus:
        return fb.softplus(linear_out);
    case MLP::linear:
        return linear_out;
    }
}

void initialize_layer(
    ContextPtr context,
    std::size_t input_dim,
    std::size_t output_dim,
    const std::string& prefix,
    int layer_index,
    std::mt19937& rand_gen,
    bool zeros
) {
    double bound = 1 / std::sqrt(input_dim);
    std::uniform_real_distribution<double> rand_dist(-bound, bound);
    auto weight_name =
        prefixed_name(prefix, std::format("layer{}.weight", layer_index));
    auto bias_name = prefixed_name(prefix, std::format("layer{}.bias", layer_index));
    auto weight_tensor_global = context->define_global(
        weight_name, DataType::dt_float, {output_dim, input_dim}, true
    );
    auto bias_tensor_global =
        context->define_global(bias_name, DataType::dt_float, {output_dim}, true);
    Tensor weight_tensor, bias_tensor;
    bool is_cpu = context->device() == cpu_device();
    if (is_cpu) {
        weight_tensor = weight_tensor_global;
        bias_tensor = bias_tensor_global;
    } else {
        weight_tensor = Tensor(DataType::dt_float, {1, output_dim, input_dim});
        bias_tensor = Tensor(DataType::dt_float, {1, output_dim});
    }

    auto weight_view = weight_tensor.view<double, 3>()[0];
    for (std::size_t i = 0; i < output_dim; ++i) {
        for (std::size_t j = 0; j < input_dim; ++j) {
            weight_view[i][j] = zeros ? 0. : rand_dist(rand_gen);
        }
    }
    auto bias_view = bias_tensor.view<double, 2>()[0];
    for (std::size_t i = 0; i < output_dim; ++i) {
        bias_view[i] = zeros ? 0. : rand_dist(rand_gen);
    }

    if (!is_cpu) {
        weight_tensor_global.copy_from(weight_tensor);
        bias_tensor_global.copy_from(bias_tensor);
    }
}

} // namespace

MLP::MLP(
    std::size_t input_dim,
    std::size_t output_dim,
    std::size_t hidden_dim,
    std::size_t layers,
    Activation activation,
    const std::string& prefix
) :
    FunctionGenerator(
        "MLP",
        {{"input", batch_float_array(input_dim)}},
        {{"output", batch_float_array(output_dim)}}
    ),
    _input_dim(input_dim),
    _output_dim(output_dim),
    _hidden_dim(hidden_dim),
    _layers(layers),
    _activation(activation),
    _prefix(prefix) {
    if (input_dim == 0) {
        throw std::invalid_argument("MLP input dimension cannot be 0");
    }
    if (output_dim == 0) {
        throw std::invalid_argument("MLP output dimension cannot be 0");
    }
};

NamedVector<Value>
MLP::build_function_impl(FunctionBuilder& fb, const NamedVector<Value>& args) const {
    std::size_t dim = _input_dim;
    Value x = args.at(0);
    for (std::size_t i = 1; i < _layers; ++i) {
        x = build_layer(fb, x, dim, _hidden_dim, _activation, _prefix, i);
        dim = _hidden_dim;
    }
    return {
        {"output", build_layer(fb, x, dim, _output_dim, MLP::linear, _prefix, _layers)}
    };
}

void MLP::initialize_globals(ContextPtr context) const {
    std::random_device rand_device;
    std::mt19937 rand_gen(rand_device());
    std::size_t dim = _input_dim;
    for (std::size_t i = 1; i < _layers; ++i) {
        initialize_layer(context, dim, _hidden_dim, _prefix, i, rand_gen, false);
        dim = _hidden_dim;
    }
    initialize_layer(context, dim, _output_dim, _prefix, _layers, rand_gen, true);
}

std::vector<std::string> MLP::global_names() const {
    std::vector<std::string> names;
    names.reserve(2 * _layers);
    for (std::size_t i = 1; i <= _layers; ++i) {
        names.push_back(prefixed_name(_prefix, std::format("layer{}.weight", i)));
        names.push_back(prefixed_name(_prefix, std::format("layer{}.bias", i)));
    }
    return names;
}
