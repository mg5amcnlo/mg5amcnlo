#pragma once

#include "madspace/compgraphs.hpp"
#include "madspace/util.hpp"

namespace madspace {

class Mapping {
public:
    using Result = std::tuple<NamedVector<Value>, Value>;

    Mapping(
        const std::string& name,
        const NamedVector<Type>& input_types,
        const NamedVector<Type>& output_types,
        const NamedVector<Type>& condition_types
    ) :
        _name(name),
        _input_types(input_types),
        _output_types(output_types),
        _condition_types(condition_types) {}
    virtual ~Mapping() = default;
    NamedVector<Value> build_forward(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions = {}
    ) const;
    NamedVector<Value> build_forward(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions = {}
    ) const {
        return build_forward(
            fb, {_input_types.keys(), inputs}, {_condition_types.keys(), conditions}
        );
    }
    NamedVector<Value> build_inverse(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions = {}
    ) const {
        return build_inverse(
            fb, {_output_types.keys(), inputs}, {_condition_types.keys(), conditions}
        );
    }
    NamedVector<Value> build_inverse(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions = {}
    ) const;
    Function forward_function() const;
    Function inverse_function() const;
    const NamedVector<Type>& input_types() const { return _input_types; }
    const NamedVector<Type>& output_types() const { return _output_types; }
    const NamedVector<Type>& condition_types() const { return _condition_types; }
    const std::string& name() const { return _name; }

protected:
    // TODO: make parameters const ref
    virtual Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const = 0;
    virtual Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const = 0;

private:
    std::string _name;
    NamedVector<Type> _input_types;
    NamedVector<Type> _output_types;
    NamedVector<Type> _condition_types;
};

class FunctionGenerator {
public:
    FunctionGenerator(
        const std::string& name,
        const NamedVector<Type>& arg_types,
        const NamedVector<Type>& return_types
    ) :
        _name(name), _arg_types(arg_types), _return_types(return_types) {}
    virtual ~FunctionGenerator() = default;
    NamedVector<Value>
    build_function(FunctionBuilder& fb, const NamedVector<Value>& args) const;
    NamedVector<Value> build_function(FunctionBuilder& fb, const ValueVec& args) const {
        return build_function(fb, {_arg_types.keys(), args});
    }
    Function function() const;
    const NamedVector<Type>& arg_types() const { return _arg_types; }
    const NamedVector<Type>& return_types() const { return _return_types; }
    const std::string& name() const { return _name; }

protected:
    virtual NamedVector<Value>
    build_function_impl(FunctionBuilder& fb, const NamedVector<Value>& args) const = 0;

private:
    std::string _name;
    NamedVector<Type> _arg_types;
    NamedVector<Type> _return_types;
};

} // namespace madspace
