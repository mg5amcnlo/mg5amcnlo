#include "madspace/phasespace/madnis.hpp"

using namespace madspace;

MadnisLoss::MadnisLoss(
    const std::vector<std::shared_ptr<FunctionGenerator>>& functions,
    const std::optional<ChannelWeightNetwork>& cwnet,
    double softclip_threshold
) :
    FunctionGenerator(
        "MadnisLoss",
        [&] {
            NamedVector<Type> arg_types;
            for (std::size_t index = 0; auto& func : functions) {
                arg_types.push_back(
                    std::format("chan{}_integrand", index), batch_float
                );
                arg_types.push_back(
                    std::format("chan{}_sample_prob", index), batch_float
                );
                if (cwnet) {
                    arg_types.push_back(
                        std::format("chan{}_cwnet_inputs", index),
                        batch_float_array(cwnet->preprocessing().output_dim())
                    );
                    arg_types.push_back(
                        std::format("chan{}_channel_weights", index),
                        batch_float_array(cwnet->mlp().output_dim())
                    );
                    arg_types.push_back(
                        std::format("chan{}_channel_indices", index), batch_int
                    );
                }
                auto& func_arg_types = func->arg_types();
                for (auto [key, type] : zip(func_arg_types.keys(), func_arg_types)) {
                    arg_types.push_back(std::format("chan{}_{}", index, key), type);
                }
                ++index;
            }
            return arg_types;
        }(),
        {{"loss", single_float},
         {"abs_means", single_float_array(functions.size())},
         {"variances", single_float_array(functions.size())}}
    ),
    _functions(functions),
    _cwnet(cwnet),
    _softclip_threshold(softclip_threshold) {}

NamedVector<Value> MadnisLoss::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    ValueVec integrands, flow_probs, sample_probs, cwnet_inputs, cwnet_priors,
        chan_indices;
    std::size_t extra_args = _cwnet ? 5 : 2;
    for (std::size_t index = 0, arg_index = 0; auto& func : _functions) {
        std::size_t arg_index_end = arg_index + func->arg_types().size() + extra_args;
        integrands.push_back(args.at(arg_index));
        sample_probs.push_back(args.at(arg_index + 1));
        if (_cwnet) {
            cwnet_inputs.push_back(args.at(arg_index + 2));
            cwnet_priors.push_back(args.at(arg_index + 3));
            chan_indices.push_back(args.at(arg_index + 4));
        }
        ValueVec func_args(
            args.begin() + arg_index + extra_args, args.begin() + arg_index_end
        );
        if (_functions.size() > 1) {
            fb.set_current_stream(index + 1);
        }
        auto output = func->build_function(fb, func_args);
        flow_probs.push_back(output.at(0));
        ++index;
        arg_index = arg_index_end;
    }
    fb.set_current_stream(0);

    ValueVec chan_weights;
    if (_cwnet) {
        auto [cwnet_in_all, counts] = fb.batch_cat(cwnet_inputs);
        auto [cwnet_priors_all, counts_prior] = fb.batch_cat(cwnet_priors);
        auto [chan_indices_all, counts_idx] = fb.batch_cat(chan_indices);
        auto cwnet_out_all =
            _cwnet->build_function(fb, {cwnet_in_all, cwnet_priors_all});
        auto chan_weight_all = fb.gather(chan_indices_all, cwnet_out_all.at(0));
        chan_weights = fb.batch_split(chan_weight_all, counts);
    } else {
        chan_weights = ValueVec(_functions.size());
    }

    ValueVec chan_losses, chan_abs_means, chan_variances;
    for (std::size_t index = 0;
         auto [integrand, g, q, cw] :
         zip(integrands, flow_probs, sample_probs, chan_weights)) {
        if (_functions.size() > 1) {
            fb.set_current_stream(index + 1);
        }
        Value f = _cwnet ? fb.mul(cw, integrand) : integrand;
        Value abs_mean = fb.batch_reduce_mean(fb.madnis_abs_weight(f, q));
        if (_softclip_threshold != 0.) {
            f = fb.madnis_softclip(f, q, abs_mean, _softclip_threshold);
            abs_mean = fb.batch_reduce_mean(fb.madnis_abs_weight(f, q));
        }
        Value mean = fb.batch_reduce_mean_keepdim(fb.div(f, q));
        Value variance = fb.batch_reduce_mean(fb.madnis_variance(f, g, q, mean));
        chan_abs_means.push_back(abs_mean);
        chan_variances.push_back(variance);
        ++index;
    }
    fb.set_current_stream(0);

    if (_functions.size() == 1) {
        return {
            {"loss",
             fb.madnis_single_channel_variance(
                 chan_variances.at(0), chan_abs_means.at(0)
             )},
            {"abs_means", fb.unsqueeze(chan_abs_means.at(0))},
            {"variances", fb.unsqueeze(chan_variances.at(0))},
        };
    } else {
        Value loss = fb.madnis_multi_channel_variance(
            fb.stack(chan_variances), fb.stack(chan_abs_means)
        );
        return {
            {"loss", loss},
            {"abs_means", fb.stack(chan_abs_means)},
            {"variances", fb.stack(chan_variances)},
        };
    }
}
