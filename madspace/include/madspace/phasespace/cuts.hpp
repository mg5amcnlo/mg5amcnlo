#pragma once

#include "madspace/compgraphs.hpp"
#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/observable.hpp"

#include <functional>
#include <utility>
#include <vector>

namespace madspace {

class Cuts : public FunctionGenerator {
public:
    enum CutMode { any, all };
    struct CutItem {
        Observable observable;
        double min = -std::numeric_limits<double>::infinity();
        double max = std::numeric_limits<double>::infinity();
        CutMode mode = CutMode::all;
    };

    Cuts(const std::vector<CutItem>& cut_data);
    Cuts(std::size_t particle_count);
    double sqrt_s_min() const;
    std::vector<double> eta_max() const;
    std::vector<double> pt_min() const;
    std::vector<std::vector<double>> m_inv_min() const;
    std::vector<std::vector<double>> dr_min() const;

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<std::vector<double>> pairwise_min(
        Observable::ObservableOption obs,
        const std::function<
            std::vector<std::pair<std::size_t, std::size_t>>(const Observable&)>& pairs
    ) const;

    std::vector<CutItem> _cut_data;
};

} // namespace madspace
