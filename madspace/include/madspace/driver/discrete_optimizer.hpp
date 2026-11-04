#pragma once

#include "madspace/compgraphs.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/driver/tensor.hpp"

namespace madspace {

/**
 * Refines the categorical probabilities of a @ref DiscreteSampler or @ref
 * DiscreteFlow from accumulated per-category statistics.
 *
 * Counterpart of @ref DiscreteHistogram: @ref add_data feeds in the
 * per-category weight sums and counts it accumulates, and @ref optimize
 * writes the probability tensor of every `prob_names` global to every context
 * in `contexts`. The probability update itself is unfinished (see the TODO in
 * `optimize`).
 */
class DiscreteOptimizer {
public:
    /**
     * @param contexts   Contexts sharing the probabilities to refine.
     * @param prob_names Global names of the probability tensors, one per
     *                   discrete dimension.
     */
    DiscreteOptimizer(
        const std::vector<ContextPtr>& contexts,
        const std::vector<std::string>& prob_names
    ) :
        _contexts(contexts), _prob_names(prob_names), _sample_count(7000) {}
    /// Accumulate one batch of `values` and `counts` for each dimension of
    /// `prob_names`, interleaved as `[values_0, counts_0, values_1, ...]`.
    void add_data(const std::vector<Tensor>& values_and_counts);
    /// Write the probabilities back to every context and reset the accumulated
    /// data.
    void optimize();

private:
    std::vector<ContextPtr> _contexts;
    std::vector<std::string> _prob_names;
    double _damping;
    std::size_t _sample_count;
    std::vector<std::tuple<std::vector<std::size_t>, std::vector<double>>> _data;
};

} // namespace madspace
