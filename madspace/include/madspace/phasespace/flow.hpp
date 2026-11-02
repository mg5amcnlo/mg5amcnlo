#pragma once

#include <optional>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/mlp.hpp"

namespace madspace {

/**
 * Normalizing-flow adaptive-sampling mapping.
 *
 * A coupling-based normalizing flow built from rational-quadratic spline
 * transformations, whose bin positions are predicted by @ref MLP subnetworks
 * [1, 2]. It reparametrizes the unit hypercube into a learned sampling
 * distribution and is trained with MadNIS [3]. The subnetwork
 * weights are compute-graph globals named after @p prefix; call
 * `initialize_globals(context)` once before use.
 *
 * `batch` is the leading batch dimension.
 *
 * **Inputs**
 * - `latent` – `float`, shape `(batch, input_dim)` – uniform coordinates in
 *   `[0, 1)`.
 *
 * **Conditions**
 * - `c` – `float`, shape `(batch, condition_dim)` – conditioning input.
 *   Present only when @p condition_dim is nonzero.
 *
 * **Outputs**
 * - `data` – `float`, shape `(batch, input_dim)` – the transformed coordinates.
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.2)
 * - [2] C. Durkan et al., "Neural spline flows",
 *   https://arxiv.org/abs/1906.04032
 * - [3] T. Heimel et al., "MadNIS - Neural multi-channel importance sampling",
 *   https://arxiv.org/abs/2212.06172
 */
class Flow : public Mapping {
public:
    /**
     * @param input_dim         Number of dimensions to transform.
     * @param condition_dim     Width of the conditioning input; `0` for an
     *                          unconditional flow.
     * @param prefix            Prefix for the trainable global names.
     * @param bin_count         Number of spline bins per dimension.
     * @param subnet_hidden_dim Hidden width of the @ref MLP subnetworks.
     * @param subnet_layers     Number of layers in the subnetworks.
     * @param subnet_activation  Activation of the subnetworks.
     * @param invert_spline     Whether the spline is applied in the inverse
     *                          direction.
     * @param subnet_precision  Float type of the subnetworks (see @ref MLP).
     */
    Flow(
        std::size_t input_dim,
        std::size_t condition_dim = 0,
        const std::string& prefix = "",
        std::size_t bin_count = 10,
        std::size_t subnet_hidden_dim = 32,
        std::size_t subnet_layers = 3,
        MLP::Activation subnet_activation = MLP::leaky_relu,
        bool invert_spline = true,
        DataType subnet_precision = DataType::dt_float64
    );
    /// Number of transformed dimensions.
    std::size_t input_dim() const { return _input_dim; }
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
    /**
     * Initialize the flow to reproduce a trained VEGAS grid.
     * @param context    Context receiving the globals.
     * @param grid_name  Global name of the VEGAS grid to copy.
     * @param seed       Seed for the random initialization of the remaining
     *                   parameters. If unset (the default), that initialization
     *                   is non-deterministic.
     */
    void initialize_from_vegas(
        ContextPtr context,
        const std::string& grid_name,
        std::optional<std::uint64_t> seed = std::nullopt
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
        const ValueVec& inputs,
        const ValueVec& conditions,
        bool inverse
    ) const;

    struct CouplingBlock {
        MLP subnet1;
        MLP subnet2;
        std::vector<me_int_t> indices1;
        std::vector<me_int_t> indices2;
    };

    std::vector<CouplingBlock> _coupling_blocks;
    std::size_t _input_dim;
    std::size_t _condition_dim;
    std::size_t _bin_count;
    bool _invert_spline;
};

} // namespace madspace
