#pragma once

#include "madspace/compgraphs.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/driver/tensor.hpp"

namespace madspace {

/**
 * Refines a @ref VegasMapping grid from accumulated bin statistics.
 *
 * Counterpart of @ref VegasHistogram: @ref add_data feeds in the per-bin
 * weight sums and counts it accumulates, and @ref optimize rewrites the
 * `grid_name` global on every context in `contexts` with the refined grid.
 */
class VegasGridOptimizer {
public:
    /**
     * @param contexts   Contexts sharing the grid to refine.
     * @param grid_name  Global name of the VEGAS grid.
     * @param damping    Exponent of the damping applied to the normalized bin
     *                   values; smaller values damp more strongly.
     */
    VegasGridOptimizer(
        const std::vector<ContextPtr>& contexts,
        const std::string& grid_name,
        double damping
    ) :
        _contexts(contexts), _grid_name(grid_name), _damping(damping) {}
    /// Accumulate one batch of per-bin weight sums (`values`) and per-bin
    /// counts (`counts`) with shape [1, dims, bins]; see @ref VegasHistogram.
    void add_data(Tensor weights, Tensor inputs);
    /// Refine the grid from the accumulated data and write it back to every
    /// context.
    void optimize();
    /// Number of grid dimensions.
    std::size_t input_dim() const;

private:
    std::vector<ContextPtr> _contexts;
    std::string _grid_name;
    double _damping;
    std::vector<std::tuple<std::vector<std::size_t>, std::vector<double>>> _data;
};

} // namespace madspace
