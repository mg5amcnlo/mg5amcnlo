#include "madspace/phasespace/batch_sampler.hpp"

using namespace madspace;

BatchSampler::BatchSampler(const std::vector<NamedVector<Type>>& types) :
    FunctionGenerator(
        "batch_sampler",
        [&] {
            NamedVector<Type> arg_types;
            for (std::size_t i = 0; auto& chan_types : types) {
                for (auto [key, type] : zip(chan_types.keys(), chan_types)) {
                    arg_types.push_back(std::format("channel{}_in_{}", i, key), type);
                }
                ++i;
            }
            arg_types.push_back("batch_sizes", batch_size_array(types.size()));
            return arg_types;
        }(),
        [&] {
            NamedVector<Type> ret_types;
            for (std::size_t i = 0; auto& chan_types : types) {
                for (auto [key, type] : zip(chan_types.keys(), chan_types)) {
                    ret_types.push_back(std::format("channel{}_out_{}", i, key), type);
                }
                ++i;
            }
            return ret_types;
        }()
    ) {
    _channel_tensor_counts.reserve(types.size());
    for (auto& chan_types : types) {
        _channel_tensor_counts.push_back(chan_types.size());
    }
}

NamedVector<Value> BatchSampler::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    ValueVec outputs;
    auto batch_sizes = fb.unstack_sizes(args.back());
    std::size_t index = 0, offset = 0;
    for (auto [count, batch_size] : zip(_channel_tensor_counts, batch_sizes)) {
        fb.set_current_stream(index + 1);
        Value buffer_size = fb.batch_size({args.at(offset)});
        Value indices = fb.random_int(batch_size, buffer_size);
        for (std::size_t i = 0; i < count; ++i) {
            outputs.push_back(fb.batch_gather(indices, args.at(offset)));
            ++offset;
        }
        ++index;
    }
    fb.set_current_stream(0);
    return {return_types().keys(), outputs};
}
