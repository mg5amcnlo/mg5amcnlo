#pragma once

#include "madspace/compgraphs.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/driver/tensor.hpp"

namespace madspace {

class VegasGridOptimizer {
public:
    VegasGridOptimizer(
        const std::vector<ContextPtr>& contexts,
        const std::string& grid_name,
        double damping
    ) :
        _contexts(contexts), _grid_name(grid_name), _damping(damping) {}
    void add_data(Tensor weights, Tensor inputs);
    void optimize();
    std::size_t input_dim() const;

private:
    std::vector<ContextPtr> _contexts;
    std::string _grid_name;
    double _damping;
    std::vector<std::tuple<std::vector<std::size_t>, std::vector<double>>> _data;
};

} // namespace madspace
