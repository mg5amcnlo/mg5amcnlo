#include "madspace/phasespace/channel_weight_network.hpp"

using namespace madspace;

namespace {

std::size_t preproc_dim(std::size_t particle_count) {
    return 3 * (particle_count - 2) + 2;
}

} // namespace

MomentumPreprocessing::MomentumPreprocessing(std::size_t particle_count) :
    FunctionGenerator(
        "MomentumPreprocessing",
        {{"momenta", batch_four_vec_array(particle_count)},
         {"x1", batch_float},
         {"x2", batch_float}},
        {{"result", batch_float_array(preproc_dim(particle_count))}}
    ),
    _output_dim(3 * (particle_count - 2) + 2) {}

NamedVector<Value> MomentumPreprocessing::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    return {{"result", fb.pt_eta_phi_x(args["momenta"], args["x1"], args["x2"])}};
}

ChannelWeightNetwork::ChannelWeightNetwork(
    std::size_t channel_count,
    std::size_t particle_count,
    std::size_t hidden_dim,
    std::size_t layers,
    MLP::Activation activation,
    const std::string& prefix,
    bool include_preprocessing
) :
    FunctionGenerator(
        "ChannelWeightNetwork",
        {{"input", batch_float_array(preproc_dim(particle_count))},
         {"prior", batch_float_array(channel_count)}},
        {{"channel_weights", batch_float_array(channel_count)}}
    ),
    _preprocessing(particle_count),
    _mlp(
        _preprocessing.output_dim(),
        channel_count,
        hidden_dim,
        layers,
        activation,
        prefix
    ),
    _channel_count(channel_count),
    _mask_name(prefixed_name(prefix, "active_channels_mask")) {}

NamedVector<Value> ChannelWeightNetwork::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto mask =
        fb.global(_mask_name, DataType::dt_float, {static_cast<int>(_channel_count)});
    // auto net_input =
    //     _preprocessing.build_function(fb, {args["momenta"], args["x1"], args["x2"]});
    // auto net_output = _mlp.build_function(fb, net_input.values()).at(0);
    auto net_output = _mlp.build_function(fb, {args["input"]}).at(0);
    return {
        {"channel_weights", fb.softmax_prior(net_output, fb.mul(args["prior"], mask))}
    };
}

void ChannelWeightNetwork::initialize_globals(ContextPtr context) const {
    _mlp.initialize_globals(context);

    context->define_global(_mask_name, DataType::dt_float, {_channel_count});
    bool is_cpu = context->device() == cpu_device();
    auto mask_global = context->global(_mask_name);
    Tensor mask;
    if (is_cpu) {
        mask = mask_global;
    } else {
        mask = Tensor(DataType::dt_float, mask_global.shape());
    }
    auto mask_view = mask.view<double, 2>()[0];
    for (std::size_t i = 0; i < mask_view.size(); ++i) {
        mask_view[i] = 1.;
    }
    if (!is_cpu) {
        mask_global.copy_from(mask);
    }
}
