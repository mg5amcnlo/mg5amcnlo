#pragma once

#include "madspace/driver/context.hpp"
#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * Per-bin weight accumulator that adapts a @ref VegasMapping grid.
 *
 * During the warm-up run it sums the sample weights falling in each bin of each
 * dimension. The accumulated `values` and `counts` are then used to refine the
 * VEGAS grid (Sec. 3.2.2 of [1]).
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `latent` – `float`, shape `(batch, dimension)` – the sampled coordinates.
 * - `weights` – `float`, shape `(batch,)` – the per-sample weights.
 *
 * **Returns**
 * - `values` – `float`, shape `(dimension, bin_count)` – summed weight per bin.
 * - `counts` – `int`, shape `(dimension, bin_count)` – sample count per bin.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.2)
 */
class VegasHistogram : public FunctionGenerator {
public:
    /// @param dimension  Number of dimensions.
    /// @param bin_count  Number of grid bins per dimension.
    VegasHistogram(std::size_t dimension, std::size_t bin_count);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _bin_count;
};

/**
 * VEGAS adaptive-sampling mapping.
 *
 * A separable piecewise-linear reparametrization of the unit hypercube, one
 * axis-aligned grid of @p bin_count bins per dimension, adapted to the
 * integrand during a warm-up run [1, 2]. The grid is a compute-graph global
 * named after @p prefix; call `initialize_globals(context)` once before use.
 *
 * `batch` is the leading batch dimension.
 *
 * **Inputs**
 * - `latent` – `float`, shape `(batch, dimension)` – uniform coordinates in
 *   `[0, 1)`.
 *
 * **Conditions**
 * - None.
 *
 * **Outputs**
 * - `data` – `float`, shape `(batch, dimension)` – the reparametrized
 *   coordinates.
 *
 * In addition every mapping returns a `det` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.2)
 * - [2] G. P. Lepage, "Adaptive multidimensional integration: VEGAS enhanced",
 *   https://arxiv.org/abs/2009.05112
 */
class VegasMapping : public Mapping {
public:
    /**
     * @param dimension  Number of dimensions.
     * @param bin_count  Number of grid bins per dimension.
     * @param prefix     Prefix for the grid global name.
     */
    VegasMapping(
        std::size_t dimension, std::size_t bin_count, const std::string& prefix = ""
    );
    /// Global name of the VEGAS grid.
    const std::string& grid_name() const { return _grid_name; }
    /// Define the grid global on @p context and fill it with a uniform grid.
    void initialize_globals(ContextPtr context) const;
    /// Number of dimensions.
    std::size_t dimension() const { return _dimension; }
    /// Number of grid bins per dimension.
    std::size_t bin_count() const { return _bin_count; }

private:
    Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;

    std::size_t _dimension;
    std::size_t _bin_count;
    std::string _grid_name;
};

/// Fill the already defined VEGAS grid global @p grid_name on @p context with a
/// uniform grid.
void initialize_vegas_grid(ContextPtr context, const std::string& grid_name);

} // namespace madspace
