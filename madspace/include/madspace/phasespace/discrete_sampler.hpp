#pragma once

#include "madspace/driver/context.hpp"
#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * Per-category weight accumulator that adapts a @ref DiscreteSampler.
 *
 * Sums the sample weights for each category of each discrete dimension. The
 * accumulated `values{d}` and `counts{d}` are then used
 * to refine the learned categorical distributions (Sec. 3.2.2 of [1]).
 *
 * `batch` is the leading batch dimension. `d` indexes the discrete dimensions.
 *
 * **Arguments**
 * - `index{d}` – `int`, shape `(batch,)` – the chosen category per dimension.
 * - `weight` – `float`, shape `(batch,)` – the per-sample weights.
 *
 * **Returns**
 * - `values{d}` – `float`, shape `(option_counts[d],)` – summed weight per
 *   category.
 * - `counts{d}` – `int`, shape `(option_counts[d],)` – sample count per category.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.2)
 */
class DiscreteHistogram : public FunctionGenerator {
public:
    /// @param option_counts  Number of categories for each discrete dimension.
    DiscreteHistogram(const std::vector<std::size_t>& option_counts);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<std::size_t> _option_counts;
};

/**
 * Adaptive sampler for discrete choices.
 *
 * Maps one uniform random number per discrete dimension to a category index,
 * using a learned categorical distribution per dimension [1]. The
 * probabilities are compute-graph globals named after @p prefix; call
 * `initialize_globals(context)` once before use. For a dimension listed in
 * @p dims_with_prior the probabilities are multiplied by a `prior{d}` input.
 *
 * `batch` is the leading batch dimension. `d` indexes the discrete dimensions.
 *
 * **Inputs**
 * - `random{d}` – `float`, shape `(batch,)` – one uniform number per dimension.
 *
 * **Conditions**
 * - `prior{d}` – `float`, shape `(batch, option_counts[d])` – prior category
 *   weights. Present only for the dimensions in @p dims_with_prior.
 *
 * **Outputs**
 * - `index{d}` – `int`, shape `(batch,)` – the chosen category per dimension.
 *
 * In addition every mapping returns a `det` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.2)
 */
class DiscreteSampler : public Mapping {
public:
    /**
     * @param option_counts    Number of categories for each discrete dimension.
     * @param prefix           Prefix for the probability global names.
     * @param dims_with_prior  Dimensions whose learned probabilities are
     *                         multiplied by a `prior{d}` input.
     */
    DiscreteSampler(
        const std::vector<std::size_t>& option_counts,
        const std::string& prefix = "",
        const std::vector<std::size_t>& dims_with_prior = {}
    );
    /// Number of categories per discrete dimension.
    const std::vector<std::size_t>& option_counts() const { return _option_counts; }
    /// Global names of the per-dimension probability vectors.
    const std::vector<std::string>& prob_names() const { return _prob_names; }
    /// Register the probability globals on @p context.
    void initialize_globals(ContextPtr context) const;

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
    Result build_transform(
        FunctionBuilder& fb,
        const ValueVec& inputs,
        const ValueVec& conditions,
        bool inverse
    ) const;

    std::vector<std::size_t> _option_counts;
    std::vector<bool> _dim_has_prior;
    std::vector<std::string> _prob_names;
};

/// Define the global @p name and fill it with uniform probabilities.
void initialize_uniform_probs(
    ContextPtr context, const std::string& name, std::size_t option_count
);

} // namespace madspace
