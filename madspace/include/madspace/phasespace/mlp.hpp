#pragma once

#include <cstdint>
#include <optional>

#include "madspace/driver/context.hpp"
#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * Fully connected neural network as a compute-graph function.
 *
 * A stack of @p layers dense layers of width @p hidden_dim with @p activation
 * between them and a linear output layer. Used as the subnetwork of the
 * @ref Flow and @ref DiscreteFlow adaptive samplers [1]. The weights and biases
 * are compute-graph globals named with @p prefix. Call
 * `initialize_globals(context)` once before use. With a single-precision
 * @p precision, the weights and the network evaluation are in float32, while the
 * input and output stay in float64.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `input` – `float`, shape `(batch, input_dim)` – the network input.
 *
 * **Returns**
 * - `output` – `float`, shape `(batch, output_dim)` – the network output.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.2)
 */
class MLP : public FunctionGenerator {
public:
    /// Activation applied between the hidden layers.
    enum Activation { relu, leaky_relu, elu, gelu, sigmoid, softplus, linear };
    /**
     * @param input_dim   Input width.
     * @param output_dim  Output width.
     * @param hidden_dim  Hidden-layer width.
     * @param layers      Number of layers.
     * @param activation  Activation between the hidden layers.
     * @param prefix      Prefix for the trainable global names.
     * @param precision   Float type of the weights and the evaluation,
     *                    `dt_float32` or `dt_float64`.
     */
    MLP(std::size_t input_dim,
        std::size_t output_dim,
        std::size_t hidden_dim = 32,
        std::size_t layers = 3,
        Activation activation = leaky_relu,
        const std::string& prefix = "",
        DataType precision = DataType::dt_float64);

    /// Input width.
    std::size_t input_dim() const { return _input_dim; }
    /// Output width.
    std::size_t output_dim() const { return _output_dim; }
    /// Float type of the weights and the evaluation.
    DataType precision() const { return _precision; }
    /**
     * Register the network's trainable parameters on @p context.
     * @param context  Context receiving the globals.
     * @param seed     Seed for the random initialization. If unset (the
     *                 default), the initialization is non-deterministic.
     */
    void initialize_globals(
        ContextPtr context, std::optional<std::uint64_t> seed = std::nullopt
    ) const;
    /// Global name of the final layer's bias vector.
    std::string last_layer_bias_name() const {
        return prefixed_name(_prefix, std::format("layer{}.bias", _layers));
    }
    /// Global names of all trainable parameters.
    std::vector<std::string> global_names() const;

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _input_dim;
    std::size_t _output_dim;
    std::size_t _hidden_dim;
    std::size_t _layers;
    Activation _activation;
    std::string _prefix;
    DataType _precision;
};

} // namespace madspace
