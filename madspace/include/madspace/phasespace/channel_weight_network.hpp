#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/mlp.hpp"

namespace madspace {

/**
 * Turns event momenta into invariant features for @ref ChannelWeightNetwork.
 *
 * Computes @f$(p_\mathrm{T}, \eta, \phi)@f$ for each particle together with the
 * momentum fractions, a representation the channel-weight network can consume
 * directly [1].
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, particle_count, 4)` – the event momenta.
 * - `x1`, `x2` – `float`, shape `(batch,)` – the parton momentum fractions.
 *
 * **Returns**
 * - `result` – `float`, shape `(batch, output_dim())` – the feature vector.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 */
class MomentumPreprocessing : public FunctionGenerator {
public:
    /// @param particle_count  Number of external particles.
    MomentumPreprocessing(std::size_t particle_count);
    /// Length of the feature vector.
    std::size_t output_dim() const { return _output_dim; };

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _output_dim;
};

/**
 * Learned multi-channel weights from a neural network.
 *
 * The MadNIS approach to multi-channeling: an @ref MLP predicts the per-channel
 * weights @f$\alpha_i(x)@f$ from the @ref MomentumPreprocessing features and a
 * prior, replacing the fixed @ref PropagatorChannelWeights [1, 2]. The network
 * weights are compute-graph globals named after @p prefix. Call
 * `initialize_globals(context)` once before use.
 *
 * `batch` is the leading batch dimension. `c` is `channel_count`.
 *
 * **Arguments**
 * - `input` – `float`, shape `(batch, MomentumPreprocessing::output_dim())` –
 *   the invariant features.
 * - `prior` – `float`, shape `(batch, c)` – the prior per-channel weights.
 *
 * **Returns**
 * - `channel_weights` – `float`, shape `(batch, c)` – the learned per-channel
 *   weights.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.1.1)
 * - [2] T. Heimel et al., "MadNIS - Neural multi-channel importance sampling",
 *   https://arxiv.org/abs/2212.06172
 */
class ChannelWeightNetwork : public FunctionGenerator {
public:
    /**
     * @param channel_count          Number of channels.
     * @param particle_count          Number of external particles.
     * @param hidden_dim              Hidden width of the @ref MLP.
     * @param layers                  Number of @ref MLP layers.
     * @param activation              @ref MLP activation.
     * @param prefix                  Prefix for the trainable global names.
     * @param include_preprocessing   If true, apply @ref MomentumPreprocessing
     *                                to the input first.
     * @param precision               Float type of the @ref MLP.
     */
    ChannelWeightNetwork(
        std::size_t channel_count,
        std::size_t particle_count,
        std::size_t hidden_dim = 32,
        std::size_t layers = 3,
        MLP::Activation activation = MLP::leaky_relu,
        const std::string& prefix = "",
        bool include_preprocessing = true,
        DataType precision = DataType::dt_float64
    );

    /// The underlying network.
    const MLP& mlp() const { return _mlp; }
    /// The feature preprocessing.
    const MomentumPreprocessing& preprocessing() const { return _preprocessing; }
    /**
     * Register the network's trainable parameters on @p context.
     * @param context  Context receiving the globals.
     * @param seed     Seed for the random initialization. If unset (the
     *                 default), the initialization is non-deterministic.
     */
    void initialize_globals(
        ContextPtr context, std::optional<std::uint64_t> seed = std::nullopt
    ) const;
    /// Global name of the channel mask.
    const std::string& mask_name() const { return _mask_name; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    MomentumPreprocessing _preprocessing;
    MLP _mlp;
    std::size_t _channel_count;
    std::string _mask_name;
};

} // namespace madspace
