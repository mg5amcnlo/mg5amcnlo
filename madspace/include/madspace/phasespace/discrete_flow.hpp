#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/mlp.hpp"

namespace madspace {

/**
 * Autoregressive adaptive sampler for discrete choices.
 *
 * Like @ref DiscreteSampler, but the categorical distribution for each discrete
 * dimension is predicted by an @ref MLP subnetwork from the earlier choices and
 * an optional conditioning input [1]. The subnetwork weights are compute-graph
 * globals named after @p prefix; call `initialize_globals(context)` once before
 * use.
 *
 * `batch` is the leading batch dimension. `d` indexes the discrete dimensions.
 *
 * **Inputs**
 * - `random_d` – `float`, shape `(batch,)` – one uniform number per dimension.
 *
 * **Conditions**
 * - `condition` – `float`, shape `(batch, condition_dim)` – conditioning input.
 *   Present only when @p condition_dim is nonzero.
 * - `prior_d` – `float`, shape `(batch, option_counts[d])` – prior category
 *   weights. Present only for the dimensions in @p dims_with_prior.
 *
 * **Outputs**
 * - `index_d` – `int`, shape `(batch,)` – the chosen category per dimension.
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.2)
 */
class DiscreteFlow : public Mapping {
public:
    /**
     * @param option_counts     Number of categories for each discrete dimension.
     * @param prefix            Prefix for the trainable global names.
     * @param dims_with_prior   Dimensions conditioned on a `prior_d` input.
     * @param condition_dim     Width of the conditioning input; `0` for none.
     * @param subnet_hidden_dim Hidden width of the @ref MLP subnetworks.
     * @param subnet_layers     Number of layers in the subnetworks.
     * @param subnet_activation  Activation of the subnetworks.
     * @param subnet_precision  Float type of the subnetworks (see @ref MLP).
     */
    DiscreteFlow(
        const std::vector<std::size_t>& option_counts,
        const std::string& prefix = "",
        const std::vector<std::size_t>& dims_with_prior = {},
        std::size_t condition_dim = 0,
        std::size_t subnet_hidden_dim = 32,
        std::size_t subnet_layers = 3,
        MLP::Activation subnet_activation = MLP::leaky_relu,
        DataType subnet_precision = DataType::dt_float64
    );
    /// Number of categories per discrete dimension.
    const std::vector<std::size_t>& option_counts() const { return _option_counts; }
    /// Width of the conditioning input.
    std::size_t condition_dim() const { return _condition_dim; }
    /**
     * Register the flow's trainable parameters on @p context.
     * @param context  Context receiving the globals.
     * @param seed     Seed for the random initialization. If unset (the
     *                 default), the initialization is non-deterministic.
     */
    void initialize_globals(
        ContextPtr context, std::optional<std::uint64_t> seed = std::nullopt
    ) const;

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
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions,
        bool inverse
    ) const;

    std::vector<std::size_t> _option_counts;
    std::size_t _condition_dim;
    std::optional<std::string> _first_prob_name;
    std::vector<MLP> _subnets;
    std::vector<bool> _dim_has_prior;
};

} // namespace madspace
