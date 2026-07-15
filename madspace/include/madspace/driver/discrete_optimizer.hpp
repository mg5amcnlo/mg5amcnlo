#pragma once

#include "madspace/compgraphs.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/driver/tensor.hpp"

namespace madspace {

class DiscreteOptimizer {
public:
    DiscreteOptimizer(
        const std::vector<ContextPtr>& contexts,
        const std::vector<std::string>& prob_names
    ) :
        _contexts(contexts), _prob_names(prob_names), _sample_count(7000) {}
    void add_data(const std::vector<Tensor>& values_and_counts);
    void optimize();

private:
    std::vector<ContextPtr> _contexts;
    std::vector<std::string> _prob_names;
    double _damping;
    std::size_t _sample_count;
    std::vector<std::tuple<std::vector<std::size_t>, std::vector<double>>> _data;
};

} // namespace madspace
