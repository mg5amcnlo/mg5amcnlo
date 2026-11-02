#include "madspace/phasespace/mlp.hpp"

#include <format>

#include "madspace/driver/random.hpp"

using namespace madspace;

namespace {

Value build_layer(
    FunctionBuilder& fb,
    Value input,
    int input_dim,
    int output_dim,
    MLP::Activation activation,
    const std::string& prefix,
    int layer_index,
    DataType precision
) {
    auto weight = fb.global(
        prefixed_name(prefix, std::format("layer{}.weight", layer_index)),
        precision,
        {output_dim, input_dim}
    );
    auto bias = fb.global(
        prefixed_name(prefix, std::format("layer{}.bias", layer_index)),
        precision,
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
    std::optional<std::uint64_t> seed,
    bool zeros,
    DataType precision
) {
    auto weight_name =
        prefixed_name(prefix, std::format("layer{}.weight", layer_index));
    auto bias_name = prefixed_name(prefix, std::format("layer{}.bias", layer_index));
    auto weight_tensor_global =
        context->define_global(weight_name, precision, {output_dim, input_dim}, true);
    auto bias_tensor_global =
        context->define_global(bias_name, precision, {output_dim}, true);
    // filled in double precision on the CPU, then converted and copied
    Tensor weight_tensor(DataType::dt_float64, {1, output_dim, input_dim});
    Tensor bias_tensor(DataType::dt_float64, {1, output_dim});

    auto weight_view = weight_tensor.view<double, 3>()[0];
    auto bias_view = bias_tensor.view<double, 2>()[0];
    if (zeros) {
        for (std::size_t i = 0; i < output_dim; ++i) {
            for (std::size_t j = 0; j < input_dim; ++j) {
                weight_view[i][j] = 0.;
            }
            bias_view[i] = 0.;
        }
    } else {
        double bound = 1 / std::sqrt(input_dim);
        auto uniform = [&](MixMaxRandom& gen) {
            return bound * (2. * gen.generate_double() - 1.);
        };
        // Independently seeded per tensor, via a fresh unique_seed_index() rather
        // than a hash of the tensor's name.
        MixMaxRandom weight_rand_gen(
            DerivedSeed(seed, DerivedSeed::global_init, context->unique_seed_index())
        );
        for (std::size_t i = 0; i < output_dim; ++i) {
            for (std::size_t j = 0; j < input_dim; ++j) {
                weight_view[i][j] = uniform(weight_rand_gen);
            }
        }
        MixMaxRandom bias_rand_gen(
            DerivedSeed(seed, DerivedSeed::global_init, context->unique_seed_index())
        );
        for (std::size_t i = 0; i < output_dim; ++i) {
            bias_view[i] = uniform(bias_rand_gen);
        }
    }

    weight_tensor_global.copy_from(weight_tensor.cast(precision));
    bias_tensor_global.copy_from(bias_tensor.cast(precision));
}

} // namespace

MLP::MLP(
    std::size_t input_dim,
    std::size_t output_dim,
    std::size_t hidden_dim,
    std::size_t layers,
    Activation activation,
    const std::string& prefix,
    DataType precision
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
    _prefix(prefix),
    _precision(precision) {
    if (!is_float(precision)) {
        throw std::invalid_argument("MLP precision must be a float type");
    }
    if (input_dim == 0) {
        throw std::invalid_argument("MLP input dimension cannot be 0");
    }
    if (output_dim == 0) {
        throw std::invalid_argument("MLP output dimension cannot be 0");
    }
};

NamedVector<Value>
MLP::build_function_impl(FunctionBuilder& fb, const NamedVector<Value>& args) const {
    bool single = _precision == DataType::dt_float32;
    std::size_t dim = _input_dim;
    Value x = single ? fb.convert_float32(args.at(0)) : args.at(0);
    for (std::size_t i = 1; i < _layers; ++i) {
        x = build_layer(fb, x, dim, _hidden_dim, _activation, _prefix, i, _precision);
        dim = _hidden_dim;
    }
    Value output =
        build_layer(fb, x, dim, _output_dim, MLP::linear, _prefix, _layers, _precision);
    return {{"output", single ? fb.convert_float64(output) : output}};
}

void MLP::initialize_globals(
    ContextPtr context, std::optional<std::uint64_t> seed
) const {
    std::size_t dim = _input_dim;
    for (std::size_t i = 1; i < _layers; ++i) {
        initialize_layer(
            context, dim, _hidden_dim, _prefix, i, seed, false, _precision
        );
        dim = _hidden_dim;
    }
    initialize_layer(
        context, dim, _output_dim, _prefix, _layers, seed, true, _precision
    );
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
