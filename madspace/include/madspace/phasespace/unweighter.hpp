#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

class Unweighter : public FunctionGenerator {
public:
    Unweighter(const NamedVector<Type>& types);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;
};

class BufferUnweighter : public FunctionGenerator {
public:
    BufferUnweighter(const NamedVector<Type>& types, double quantile = 0.0);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    double _quantile;
};

} // namespace madspace
