#include "madspace/phasespace/discrete_flow.hpp"

#include "madspace/phasespace/discrete_sampler.hpp"
#include "madspace/util.hpp"

using namespace madspace;

DiscreteFlow::DiscreteFlow(
    const std::vector<std::size_t>& option_counts,
    const std::string& prefix,
    const std::vector<std::size_t>& dims_with_prior,
    std::size_t condition_dim,
    std::size_t subnet_hidden_dim,
    std::size_t subnet_layers,
    MLP::Activation subnet_activation
) :
    Mapping(
        "DiscreteFlow",
        [&] {
            NamedVector<Type> in_types;
            for (std::size_t dim = 0; dim < option_counts.size(); ++dim) {
                in_types.push_back(std::format("random_{}", dim), batch_float);
            }
            return in_types;
        }(),
        [&] {
            NamedVector<Type> out_types;
            for (std::size_t dim = 0; dim < option_counts.size(); ++dim) {
                out_types.push_back(std::format("index_{}", dim), batch_int);
            }
            return out_types;
        }(),
        [&] {
            NamedVector<Type> cond_types;
            if (condition_dim > 0) {
                cond_types.push_back("condition", batch_float_array(condition_dim));
            }
            for (std::size_t dim : dims_with_prior) {
                cond_types.push_back(
                    std::format("prior_{}", dim),
                    batch_float_array(option_counts.at(dim))
                );
            }
            return cond_types;
        }()
    ),
    _option_counts(option_counts),
    _dim_has_prior(option_counts.size()),
    _condition_dim(condition_dim) {
    std::size_t option_sum = 0, dim_index = 0;
    for (std::size_t option_count : option_counts) {
        std::size_t subnet_input_dim = option_sum + condition_dim;
        if (subnet_input_dim > 0) {
            _subnets.emplace_back(
                subnet_input_dim,
                option_count,
                subnet_hidden_dim,
                subnet_layers,
                subnet_activation,
                prefixed_name(prefix, std::format("subnet{}_", dim_index + 1))
            );
        } else {
            _first_prob_name = prefixed_name(prefix, "prob0");
        }
        option_sum += option_count;
        ++dim_index;
    }
    for (std::size_t dim : dims_with_prior) {
        _dim_has_prior.at(dim) = true;
    }
}

Mapping::Result DiscreteFlow::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    return build_transform(fb, inputs, conditions, false);
}

Mapping::Result DiscreteFlow::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    return build_transform(fb, inputs, conditions, true);
}

Mapping::Result DiscreteFlow::build_transform(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions,
    bool inverse
) const {
    Value subnet_input;
    std::size_t dim_index = 0, mlp_index = 0, condition_index = 0;
    me_int_t prev_option_count = 0;
    if (_condition_dim != 0) {
        subnet_input = conditions.at(0);
        ++condition_index;
    }

    Value prev_index;
    ValueVec outputs, dets;
    for (auto [option_count, input, has_prior] :
         zip(_option_counts, inputs, _dim_has_prior)) {
        if (dim_index > 0) {
            subnet_input =
                fb.cat({subnet_input, fb.one_hot(prev_index, prev_option_count)});
        }
        Value probs;
        if (dim_index == 0 && _first_prob_name) {
            probs = fb.global(
                _first_prob_name.value(),
                DataType::dt_float,
                {static_cast<int>(option_count)}
            );
        } else {
            probs = fb.softmax(
                _subnets.at(mlp_index).build_function(fb, {subnet_input}).at(0)
            );
            ++mlp_index;
        }
        if (has_prior) {
            probs = fb.mul(probs, conditions.at(condition_index));
            ++condition_index;
        }
        auto [output, det] = inverse
            ? fb.sample_discrete_probs_inverse(input, probs)
            : fb.sample_discrete_probs(input, probs);
        outputs.push_back(output);
        dets.push_back(det);
        prev_option_count = option_count;
        prev_index = inverse ? input : output;
        ++dim_index;
    }
    return {
        {(inverse ? input_types() : output_types()).keys(), outputs}, fb.product(dets)
    };
}

void DiscreteFlow::initialize_globals(ContextPtr context) const {
    if (_first_prob_name) {
        initialize_uniform_probs(
            context, _first_prob_name.value(), _option_counts.at(0)
        );
    }
    for (auto& subnet : _subnets) {
        subnet.initialize_globals(context);
    }
}
