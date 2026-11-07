#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * Draws random batches from per-channel buffers.
 *
 * For each channel, picks `batch_sizes[i]` rows at random indices (with
 * replacement) of that channel's buffered tensors. Used to draw MadNIS training
 * batches from the replay buffer.
 *
 * `batch` is the leading batch dimension. `i` indexes the channels.
 *
 * **Arguments**
 * - `channel<i>_in_<key>` – the buffered tensors of channel `i`.
 * - `batch_sizes` – the number of rows to draw for each channel.
 *
 * **Returns**
 * - `channel<i>_out_<key>` – the drawn rows, one tensor per input tensor.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 */
class BatchSampler : public FunctionGenerator {
public:
    /// @param types  One tensor-type set per channel.
    BatchSampler(const std::vector<NamedVector<Type>>& types);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<std::size_t> _channel_tensor_counts;
};

} // namespace madspace
