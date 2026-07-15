#include "madspace/phasespace/unweighter.hpp"

using namespace madspace;

Unweighter::Unweighter(const NamedVector<Type>& types) :
    FunctionGenerator(
        "Unweighter",
        [&] {
            NamedVector<Type> arg_types = types;
            arg_types.push_back("max_weight", single_float);
            return arg_types;
        }(),
        types
    ) {}

NamedVector<Value> Unweighter::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value weights = args.at(0), max_weight = args.back();
    auto [uw_indices, uw_weights] = fb.unweight(weights, max_weight);
    ValueVec output{uw_weights};
    for (auto arg : std::span(args.begin() + 1, args.end() - 1)) {
        output.push_back(fb.batch_gather(uw_indices, arg));
    }
    return {return_types().keys(), output};
}

BufferUnweighter::BufferUnweighter(const NamedVector<Type>& types, double quantile) :
    FunctionGenerator("BufferUnweighter", types, types), _quantile(quantile) {}

NamedVector<Value> BufferUnweighter::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value max_weight = fb.quantile(args.at(0), _quantile);
    Value full_weight = args.at(0);
    auto [uw_indices, uw_weights] = fb.unweight(full_weight, max_weight);
    ValueVec output{uw_weights};
    for (auto arg : std::span(args.begin() + 1, args.end())) {
        output.push_back(fb.batch_gather(uw_indices, arg));
    }

    std::size_t rescale_index = args.index_map().at("adaptive_prob");
    Value orig_weights = fb.batch_gather(uw_indices, full_weight);
    Value& rescale_output = output.at(rescale_index);
    Value acc_factor = fb.div(orig_weights, uw_weights);
    Value acc_norm = fb.accept_norm(uw_indices, full_weight);
    rescale_output = fb.mul(fb.mul(acc_factor, rescale_output), acc_norm);
    return {return_types().keys(), output};
}
