#pragma once

#include <format>
#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/phasespace.hpp"

namespace madspace {

/**
 * Applies one of several channel mappings to each slice of the batch.
 *
 * A multi-channel integrand covers the phase space with several mappings
 * @f$G_i@f$, each flattening a different structure of the integrand [1]. This
 * class combines them into a single @ref Mapping. The batch is partitioned by
 * the per-channel counts in the `return_batch_sizes` condition, each slice is
 * passed through its channel mapping, and the results are concatenated. All
 * channel mappings must share the same input, output and condition layout,
 * which this class then exposes. See Sec. 2.1.1 of [1], following [2, 3].
 *
 * `batch` is the leading batch dimension.
 *
 * **Inputs**
 * - the inputs of the channel mappings (identical across channels).
 *
 * **Conditions**
 * - the conditions of the channel mappings, plus
 * - `return_batch_sizes` – the number of batch entries routed to each channel.
 *
 * **Outputs**
 * - the outputs of the channel mappings (identical across channels).
 *
 * In addition every mapping returns a `det` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.1.1)
 * - [2] R. Kleiss, R. Pittau, "Weight optimization in multichannel Monte
 *   Carlo", https://arxiv.org/abs/hep-ph/9405257
 * - [3] F. Maltoni, T. Stelzer, "MadEvent: automatic event generation with
 *   MadGraph", https://arxiv.org/abs/hep-ph/0208156
 */
class MultiChannelMapping : public Mapping {
public:
    /// @param mappings  The channel mappings. All must have the same number of
    ///                  inputs, outputs and conditions.
    MultiChannelMapping(const std::vector<std::shared_ptr<Mapping>>& mappings);

private:
    Result build_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions,
        bool inverse
    ) const;
    Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override {
        return build_impl(fb, inputs, conditions, false);
    }
    Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override {
        return build_impl(fb, inputs, conditions, true);
    }

    std::vector<std::shared_ptr<Mapping>> _mappings;
};

/**
 * Applies one of several channel functions to each slice of the batch.
 *
 * The @ref FunctionGenerator counterpart of @ref MultiChannelMapping. The batch
 * is partitioned by the per-channel counts in the `batch_sizes` argument. Each
 * slice is passed through its channel function, and the results are
 * concatenated. All channel functions must share the same argument and return
 * layout, which this class then exposes (Sec. 2.1.1 of [1]). A batch-size
 * argument of the channel functions is replaced by the per-channel size.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - the arguments of the channel functions, plus
 * - `batch_sizes` – the number of batch entries routed to each channel.
 *
 * **Returns**
 * - the returns of the channel functions, plus
 * - `batch_sizes` – the per-channel sizes of the concatenated first return
 *   value, only when @p return_batch_sizes is true.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.1.1)
 */
class MultiChannelFunction : public FunctionGenerator {
public:
    /**
     * @param functions           The per-channel functions.
     * @param return_batch_sizes  If true, the per-channel batch sizes of the
     *                            result are also returned.
     */
    MultiChannelFunction(
        const std::vector<std::shared_ptr<FunctionGenerator>>& functions,
        bool return_batch_sizes = false
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<std::shared_ptr<FunctionGenerator>> _functions;
    bool _return_batch_sizes;
};

} // namespace madspace
