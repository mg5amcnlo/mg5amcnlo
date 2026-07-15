#include "madspace/phasespace/base.hpp"

#include "madspace/util.hpp"

using namespace madspace;

namespace {

void check_types(
    const NamedVector<Value>& values,
    const NamedVector<Type>& types,
    const std::string& prefix
) {
    if (values.size() != types.size()) {
        throw std::runtime_error(
            std::format(
                "{}: Invalid number of values. Expected {}, got {}",
                prefix,
                types.size(),
                values.size()
            )
        );
    }
    std::size_t val_index = 1;
    for (auto [value, type] : zip(values, types)) {
        if (value.type.dtype != type.dtype) {
            throw std::runtime_error(
                std::format("{}, value {}: Invalid dtype", prefix, val_index)
            );
        }
        if (value.type.shape != type.shape) {
            std::string expected_shape, got_shape;
            for (auto& size : type.shape) {
                expected_shape += &size == &type.shape.back()
                    ? std::format("{}", size)
                    : std::format("{}, ", size);
            }
            for (auto& size : value.type.shape) {
                got_shape += &size == &value.type.shape.back()
                    ? std::format("{}", size)
                    : std::format("{}, ", size);
            }
            throw std::runtime_error(
                std::format(
                    "{}, value {}: Invalid shape, expected ({}), got ({})",
                    prefix,
                    val_index,
                    expected_shape,
                    got_shape
                )
            );
        }
        ++val_index;
    }
}

template <typename T>
[[noreturn]] void handle_error(const T& e, const std::string& name) {
    std::string message(e.what());
    if (auto in_pos = message.find("[in "); in_pos != std::string::npos) {
        message.insert(in_pos + 4, std::format("{} > ", name));
    } else {
        message.append(std::format("\n [in {}]", name));
    }
    throw T(message);
}

[[noreturn]] void handle_errors(const std::string& name) {
    try {
        throw;
    } catch (const std::invalid_argument& e) {
        handle_error(e, name);
    } catch (const std::domain_error& e) {
        handle_error(e, name);
    } catch (const std::length_error& e) {
        handle_error(e, name);
    } catch (const std::out_of_range& e) {
        handle_error(e, name);
    } catch (const std::logic_error& e) {
        handle_error(e, name);
    } catch (const std::range_error& e) {
        handle_error(e, name);
    } catch (const std::runtime_error& e) {
        handle_error(e, name);
    }
}

} // namespace

NamedVector<Value> Mapping::build_forward(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    try {
        NamedVector<Value> sorted_inputs = inputs.sort_like(_input_types.index_map());
        NamedVector<Value> sorted_conditions =
            conditions.sort_like(_condition_types.index_map());
        check_types(sorted_inputs, _input_types, "Input");
        check_types(sorted_conditions, _condition_types, "Condition");
        auto [outputs, det] = build_forward_impl(fb, sorted_inputs, sorted_conditions);
        NamedVector<Value> sorted_outputs =
            outputs.sort_like(_output_types.index_map());
        check_types(sorted_outputs, _output_types, "Output");
        check_types({{"det", det}}, {{"det", batch_float}}, "Determinant");
        outputs.push_back("det", det);
        return outputs;
    } catch (...) {
        handle_errors(name());
    }
}

NamedVector<Value> Mapping::build_inverse(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    try {
        NamedVector<Value> sorted_inputs = inputs.sort_like(_output_types.index_map());
        NamedVector<Value> sorted_conditions =
            conditions.sort_like(_condition_types.index_map());
        check_types(sorted_inputs, _output_types, "Input");
        check_types(sorted_conditions, _condition_types, "Condition");
        auto [outputs, det] = build_inverse_impl(fb, sorted_inputs, sorted_conditions);
        NamedVector<Value> sorted_outputs = outputs.sort_like(_input_types.index_map());
        check_types(sorted_outputs, _input_types, "Output");
        check_types({{"det", det}}, {{"det", batch_float}}, "Determinant");
        outputs.push_back("det", det);
        return outputs;
    } catch (...) {
        handle_errors(name());
    }
}

Function Mapping::forward_function() const {
    auto arg_types = _input_types;
    arg_types.insert_back(_condition_types);
    auto ret_types = _output_types;
    ret_types.push_back("det", batch_float);
    FunctionBuilder fb(arg_types, ret_types);
    auto n_inputs = _input_types.size();
    auto n_outputs = _output_types.size();
    auto [outputs, det] = build_forward_impl(
        fb,
        {_input_types.keys(), fb.input_range(0, n_inputs)},
        {_condition_types.keys(), fb.input_range(n_inputs, arg_types.size())}
    );
    NamedVector<Value> sorted_outputs = outputs.sort_like(_output_types.index_map());
    check_types(sorted_outputs, _output_types, "Output");
    fb.output_range(0, sorted_outputs.values());
    fb.output(n_outputs, det);
    return fb.function();
}

Function Mapping::inverse_function() const {
    auto arg_types = _output_types;
    arg_types.insert_back(_condition_types);
    auto ret_types = _input_types;
    ret_types.push_back("det", batch_float);
    FunctionBuilder fb(arg_types, ret_types);
    auto n_inputs = _input_types.size();
    auto n_outputs = _output_types.size();
    auto [outputs, det] = build_inverse_impl(
        fb,
        {_output_types.keys(), fb.input_range(0, n_outputs)},
        {_condition_types.keys(), fb.input_range(n_outputs, arg_types.size())}
    );
    NamedVector<Value> sorted_outputs = outputs.sort_like(_input_types.index_map());
    check_types(sorted_outputs, _input_types, "Output");
    fb.output_range(0, sorted_outputs.values());
    fb.output(n_inputs, det);
    return fb.function();
}

NamedVector<Value> FunctionGenerator::build_function(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    try {
        NamedVector<Value> sorted_args = args.sort_like(_arg_types.index_map());
        check_types(sorted_args, _arg_types, "Argument");
        auto outputs = build_function_impl(fb, sorted_args);
        NamedVector<Value> sorted_outputs =
            outputs.sort_like(_return_types.index_map());
        check_types(sorted_outputs, _return_types, "Output");
        return outputs;
    } catch (...) {
        handle_errors(name());
    }
}

Function FunctionGenerator::function() const {
    FunctionBuilder fb(_arg_types, _return_types);
    auto outputs = build_function_impl(
        fb, {_arg_types.keys(), fb.input_range(0, _arg_types.size())}
    );
    NamedVector<Value> sorted_outputs = outputs.sort_like(_return_types.index_map());
    check_types(sorted_outputs, _return_types, "Output");
    fb.output_range(0, sorted_outputs.values());
    return fb.function();
}
