#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * Rejection-samples a weighted batch into an unweighted one.
 *
 * Keeps each event with probability `weight / max_weight` and returns the
 * surviving rows (Sec. 3.4 of [1]). The first argument and return value are the
 * weights; any further tensors listed in @p types are carried along and
 * gathered for the surviving events.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - the tensors named in @p types (the first is the event weight), plus
 * - `max_weight` – `float`, scalar – the weight to unweight against.
 *
 * **Returns**
 * - the tensors named in @p types, restricted to the accepted events.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.4)
 */
class Unweighter : public FunctionGenerator {
public:
    /// @param types  The per-event tensors, the first being the weight.
    Unweighter(const NamedVector<Type>& types);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;
};

/**
 * Partial unweighting of one batch, for the MadNIS replay buffer.
 *
 * The maximum weight is the @p quantile quantile of |w| over the batch (so a
 * fraction 1 - @p quantile of the events is over-weight). An event is kept
 * with probability min(1, |w| / w_max) and gets the weight
 * sign(w) * max(|w|, w_max), so the sign of a negative weight is kept. The
 * `adaptive_prob` of a kept event, the density it was sampled with, is
 * multiplied by its acceptance probability and by N / N_kept, so that training
 * on the buffered events stays unbiased (Sec. 3.4.1 of [1]).
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - the tensors named in @p types (the first is the event weight, and one
 *   must be named `adaptive_prob`).
 *
 * **Returns**
 * - the tensors named in @p types, partially unweighted, with `adaptive_prob`
 *   rescaled.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.4.1)
 */
class BufferUnweighter : public FunctionGenerator {
public:
    /// @param types     The per-event tensors, the first being the weight.
    /// @param quantile  Quantile of |w| taken as the maximum weight.
    BufferUnweighter(const NamedVector<Type>& types, double quantile = 0.0);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    double _quantile;
};

} // namespace madspace
