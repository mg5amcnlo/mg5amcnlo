#include "madspace/phasespace/multichannel.hpp"

using namespace madspace;

MultiChannelMapping::MultiChannelMapping(
    const std::vector<std::shared_ptr<Mapping>>& mappings
) :
    Mapping(
        "MultiChannelMapping",
        mappings.at(0)->input_types(),
        mappings.at(0)->output_types(),
        [&] {
            auto condition_types = mappings.at(0)->condition_types();
            condition_types.push_back(
                "return_batch_sizes", multichannel_batch_size(_mappings.size())
            );
            return condition_types;
        }()
    ),
    _mappings(mappings) {
    auto& first_mapping = mappings.at(0);
    std::size_t input_count = first_mapping->input_types().size();
    std::size_t output_count = first_mapping->output_types().size();
    std::size_t condition_count = first_mapping->condition_types().size();
    for (auto& mapping : mappings) {
        if (mapping->input_types().size() != input_count ||
            mapping->output_types().size() != output_count ||
            mapping->condition_types().size() != condition_count) {
            throw std::invalid_argument(
                "All mappings must have the same number of inputs, outputs and "
                "conditions"
            );
        }
    }
}

Mapping::Result MultiChannelMapping::build_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions,
    bool inverse
) const {
    auto& counts = conditions.back();

    std::vector<ValueVec> split_inputs;
    for (auto& input : inputs) {
        split_inputs.push_back(fb.batch_split(input, counts));
    }
    std::vector<ValueVec> split_conditions;
    for (auto& condition : conditions) {
        if (&condition == &counts) {
            break;
        }
        split_conditions.push_back(fb.batch_split(condition, counts));
    }

    std::vector<ValueVec> split_outputs(output_types().size());
    ValueVec split_dets;
    std::size_t index = 0;
    for (auto& mapping : _mappings) {
        ValueVec in, cond;
        for (auto& input : split_inputs) {
            in.push_back(input.at(index));
        }
        for (auto& condition : split_conditions) {
            cond.push_back(condition.at(index));
        }
        fb.set_current_stream(index + 1);
        auto output = inverse
            ? mapping->build_inverse(fb, in, cond)
            : mapping->build_forward(fb, in, cond);
        for (auto [out, split_out] : zip(output, split_outputs)) {
            split_out.push_back(out);
        }
        split_dets.push_back(output["det"]);
        ++index;
    }
    fb.set_current_stream(0);
    ValueVec cat_outputs;
    for (auto& output : split_outputs) {
        auto [cat, _] = fb.batch_cat(output);
        cat_outputs.push_back(cat);
    }
    auto [det, _] = fb.batch_cat(split_dets);
    return {{output_types().keys(), cat_outputs}, det};
}

MultiChannelFunction::MultiChannelFunction(
    const std::vector<std::shared_ptr<FunctionGenerator>>& functions,
    bool return_batch_sizes
) :
    FunctionGenerator(
        "MultiChannelFunction",
        [&] {
            NamedVector<Type> arg_types;
            auto& first_types = functions.at(0)->arg_types();
            for (auto [key, arg_type] : zip(first_types.keys(), first_types)) {
                if (arg_type.dtype == DataType::batch_sizes) {
                    if (arg_type.batch_size_list.size() != 1) {
                        throw std::invalid_argument(
                            "Only batch size list arguments with size 1 accepted"
                        );
                    }
                } else {
                    arg_types.push_back(key, arg_type);
                }
            }
            arg_types.push_back(
                "batch_sizes", multichannel_batch_size(functions.size())
            );
            return arg_types;
        }(),
        [&] {
            NamedVector<Type> ret_types = functions.at(0)->return_types();
            if (return_batch_sizes) {
                ret_types.push_back(
                    "batch_sizes", multichannel_batch_size(functions.size())
                );
            }
            return ret_types;
        }()
    ),
    _functions(functions),
    _return_batch_sizes(return_batch_sizes) {
    auto& first_function = functions.at(0);
    std::size_t arg_count = first_function->arg_types().size();
    std::size_t return_count = first_function->return_types().size();
    for (auto& function : functions) {
        if (function->arg_types().size() != arg_count ||
            function->return_types().size() != return_count) {
            throw std::invalid_argument(
                "All functions must have the same number of inputs and outputs"
            );
        }
    }
}

NamedVector<Value> MultiChannelFunction::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto counts = args["batch_sizes"];

    auto arg_types = _functions.at(0)->arg_types();
    std::size_t arg_index = 0;
    std::vector<ValueVec> split_args;
    for (auto& arg_type : arg_types) {
        if (arg_type.dtype == DataType::batch_sizes) {
            split_args.push_back(fb.unstack_sizes(counts));
        } else {
            split_args.push_back(fb.batch_split(args.at(arg_index), counts));
            ++arg_index;
        }
    }

    std::vector<ValueVec> split_outputs(return_types().size() - _return_batch_sizes);
    std::size_t index = 0;
    for (auto& func : _functions) {
        ValueVec func_args;
        for (auto& arg : split_args) {
            func_args.push_back(arg.at(index));
        }
        fb.set_current_stream(index + 1);
        auto output = func->build_function(fb, func_args);
        for (auto [out, split_out] : zip(output, split_outputs)) {
            split_out.push_back(out);
        }
        ++index;
    }
    fb.set_current_stream(0);
    ValueVec cat_outputs;
    Value sizes_out;
    bool first = true;
    for (auto& output : split_outputs) {
        auto [cat, cat_sizes] = fb.batch_cat(output);
        cat_outputs.push_back(cat);
        if (first && _return_batch_sizes) {
            sizes_out = cat_sizes;
        }
        first = false;
    }
    if (_return_batch_sizes) {
        cat_outputs.push_back(sizes_out);
    }
    return {return_types().keys(), cat_outputs};
}
