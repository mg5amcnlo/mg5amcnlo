#include "madspace/driver/discrete_optimizer.hpp"

using namespace madspace;

void DiscreteOptimizer::add_data(const std::vector<Tensor>& values_and_counts) {
    if (_data.size() != _prob_names.size()) {
        _data.resize(_prob_names.size());
        for (auto [prob_name, data_item] : zip(_prob_names, _data)) {
            auto prob_global = _contexts.at(0)->global(prob_name);
            auto option_count = prob_global.size(1);
            auto& [weight_sums, counts] = data_item;
            weight_sums.resize(option_count);
            counts.resize(option_count);
        }
    }

    for (std::size_t i = 0; auto [prob_name, data_item] : zip(_prob_names, _data)) {
        auto& [weight_sums, counts] = data_item;
        auto values_cpu = values_and_counts.at(2 * i).cpu();
        auto counts_cpu = values_and_counts.at(2 * i + 1).cpu();
        auto values_view = values_cpu.view<double, 2>()[0];
        auto counts_view = counts_cpu.view<me_int_t, 2>()[0];
        for (std::size_t j = 0; j < values_view.size(); ++j) {
            weight_sums[j] += values_view[j];
            counts[j] += counts_view[j];
        }
        ++i;
    }
}

void DiscreteOptimizer::optimize() {
    // TODO: check shapes
    // std::size_t next_sample_count = _sample_count + weights_view.size();
    double prob_ratio = 1.; // static_cast<double>(_sample_count) / next_sample_count;
    //_sample_count = next_sample_count;
    for (auto [prob_name, data_item] : zip(_prob_names, _data)) {
        auto& [weight_sums, counts] = data_item;
        Tensor prob(DataType::dt_float, _contexts.at(0)->global(prob_name).shape());
        auto prob_view = prob.view<double, 2>()[0];

        double norm = 0.;
        for (std::size_t i = 0; auto [wsum, count] : zip(weight_sums, counts)) {
            if (count > 0) {
                wsum *= prob_view[i] / count;
            }
            norm += wsum;
            ++i;
        }

        for (std::size_t i = 0; double wsum : weight_sums) {
            // prob_view[i] = prob_view[i] * prob_ratio + wsum / norm * (1. -
            // prob_ratio);
            ++i;
        }

        for (auto& context : _contexts) {
            context->global(prob_name).copy_from(prob);
        }
        std::fill(weight_sums.begin(), weight_sums.end(), 0.);
        std::fill(counts.begin(), counts.end(), 0);
    }
}
