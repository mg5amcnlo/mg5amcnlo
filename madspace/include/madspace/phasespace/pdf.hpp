#pragma once

#include "madspace/driver/context.hpp"
#include "madspace/phasespace/base.hpp"

namespace madspace {

struct PdfGrid {
    std::vector<double> x;
    std::vector<double> logx;
    std::vector<double> q;
    std::vector<double> logq2;
    std::vector<int> pids;
    std::vector<std::vector<double>> values;
    std::vector<std::size_t> region_sizes;

    PdfGrid(const std::string& file);
    std::size_t grid_point_count() const;
    std::size_t q_count() const;
    void initialize_coefficients(Tensor tensor) const;
    void initialize_logx(Tensor tensor) const;
    void initialize_logq2(Tensor tensor) const;
    std::vector<std::size_t> coefficients_shape(bool batch_dim = false) const;
    std::vector<std::size_t> logx_shape(bool batch_dim = false) const;
    std::vector<std::size_t> logq2_shape(bool batch_dim = false) const;
    void initialize_globals(ContextPtr context, const std::string& prefix = "") const;
};

class PartonDensity : public FunctionGenerator {
public:
    PartonDensity(
        const PdfGrid& grid,
        const std::vector<int>& pids,
        bool dynamic_pid = false,
        const std::string& prefix = ""
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<me_int_t> _pid_indices;
    bool _dynamic_pid;
    std::string _prefix;
    std::vector<std::size_t> _logx_shape;
    std::vector<std::size_t> _logq2_shape;
    std::vector<std::size_t> _coeffs_shape;
};

struct AlphaSGrid {
    std::vector<double> q;
    std::vector<double> logq2;
    std::vector<double> values;
    std::vector<std::size_t> region_sizes;

    AlphaSGrid(const std::string& file);
    std::size_t q_count() const;
    void initialize_coefficients(Tensor tensor) const;
    void initialize_logq2(Tensor tensor) const;
    std::vector<std::size_t> coefficients_shape(bool batch_dim = false) const;
    std::vector<std::size_t> logq2_shape(bool batch_dim = false) const;
    void initialize_globals(ContextPtr context, const std::string& prefix = "") const;
};

class RunningCoupling : public FunctionGenerator {
public:
    RunningCoupling(const AlphaSGrid& grid, const std::string& prefix = "");

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::string _prefix;
    std::vector<std::size_t> _logq2_shape;
    std::vector<std::size_t> _coeffs_shape;
};

} // namespace madspace
