#pragma once

#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/channel_weight_network.hpp"

namespace madspace {

/**
 * MadNIS training objective for the adaptive samplers.
 *
 * Combines the per-channel integrand and sampling-probability values into the
 * variance-based loss used to train the @ref Flow and @ref ChannelWeightNetwork
 * [1].
 *
 * `batch` is the leading batch dimension. `c` indexes the channels.
 *
 * **Arguments**
 * - `chan<c>_integrand` – `float`, shape `(batch,)` – integrand values.
 * - `chan<c>_sample_prob` – `float`, shape `(batch,)` – sampling probabilities.
 * - With a @ref ChannelWeightNetwork also `chan<c>_cwnet_inputs`,
 *   `chan<c>_channel_weights` (or the compressed pair
 *   `chan<c>_channel_weight_values` / `chan<c>_channel_weight_indices`) and
 *   `chan<c>_channel_indices`.
 * - `chan<c>_<name>` – the arguments of the c-th function of @p functions.
 *
 * **Returns**
 * - `loss` – `float`, scalar – the training loss.
 * - `abs_means` – `float`, shape `(c,)` – per-channel mean of `|integrand|`.
 * - `variances` – `float`, shape `(c,)` – per-channel integrand variance.
 *
 * **References**
 * - [1] T. Heimel et al., "MadNIS - Neural multi-channel importance sampling",
 *   https://arxiv.org/abs/2212.06172
 */
class MadnisLoss : public FunctionGenerator {
public:
    /**
     * @param functions                      The per-channel integrands; the
     *                                       first return value of each is the
     *                                       flow probability.
     * @param cwnet                          Optional channel-weight network to
     *                                       train jointly.
     * @param softclip_threshold             Soft-clipping threshold on the
     *                                       loss contribution; `0` disables it.
     * @param compressed_channel_weight_count Number of channel-weight entries
     *                                       kept per event.
     */
    MadnisLoss(
        const std::vector<std::shared_ptr<FunctionGenerator>>& functions,
        const std::optional<ChannelWeightNetwork>& cwnet,
        double softclip_threshold = 0.0,
        std::size_t compressed_channel_weight_count = 50
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<std::shared_ptr<FunctionGenerator>> _functions;
    std::optional<ChannelWeightNetwork> _cwnet;
    double _softclip_threshold;
    std::size_t _compressed_channel_weight_count;
};

} // namespace madspace
