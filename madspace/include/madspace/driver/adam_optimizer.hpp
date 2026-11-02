#pragma once

#include "madspace/driver/backend.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/phasespace/base.hpp"

namespace madspace {

/// Compute-graph function that rescales a gradient tensor so its norm never
/// exceeds a threshold; the graph built internally by @ref AdamOptimizer for
/// `grad_clip_threshold`.
class GradientClipper : public FunctionGenerator {
public:
    /// @param dtype  Float type of the gradients; the norm is computed in float64.
    GradientClipper(DataType dtype = DataType::dt_float64);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;
};

/**
 * Adam gradient-descent optimizer for a compute-graph loss function [1].
 *
 * Wraps a loss @p function into a `Runtime`, runs it forward and backward on
 * each @ref step, and applies the Adam update (@p beta1 / @p beta2 moment
 * decay, @p eps for numerical stability, optional @p weight_decay) to its
 * `requires_grad` @ref Context globals. @p schedule optionally decays the
 * learning rate over @p step_count steps; @p grad_clip_threshold, if
 * non-zero, rescales the gradient norm before the update.
 *
 * **References**
 * - [1] D. P. Kingma, J. Ba, "Adam: A method for stochastic optimization",
 *   https://arxiv.org/abs/1412.6980
 */
class AdamOptimizer {
public:
    /// Learning-rate decay applied over the course of training.
    enum LRSchedule {
        /// Constant learning rate.
        none,
        /// Cosine decay from `learning_rate` to zero over `step_count` steps.
        cosine,
    };

    /**
     * @param function            The loss function to minimize.
     * @param context             Context whose `requires_grad` globals are
     *                            trained.
     * @param learning_rate       Initial (or, without `schedule`, constant)
     *                            learning rate.
     * @param schedule            Learning-rate decay over training.
     * @param step_count          Total number of steps, for `schedule`.
     * @param beta1               First-moment decay rate.
     * @param beta2               Second-moment decay rate.
     * @param eps                 Numerical-stability constant.
     * @param grad_clip_threshold Maximum gradient norm; `0` disables clipping.
     * @param weight_decay        L2 weight-decay coefficient.
     */
    AdamOptimizer(
        const Function& function,
        ContextPtr context,
        double learning_rate,
        LRSchedule schedule = LRSchedule::none,
        std::size_t step_count = 0,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double eps = 1e-8,
        double grad_clip_threshold = 0.0,
        double weight_decay = 0.0
    );
    /// Run one training step on `inputs` and apply the Adam update; returns
    /// `function`'s outputs.
    TensorVec step(const TensorVec& inputs);
    /// Replace the loss function, keeping the optimizer state.
    void replace_function(const Function& function);
    /// The current, possibly `schedule`-decayed, learning rate.
    double learning_rate() const;
    /// Input types of the loss function.
    const TypeVec& input_types() const { return _input_types; }
    /// The context being trained.
    ContextPtr context() const { return _context; }
    /// The trained globals, packed into one contiguous tensor.
    Tensor parameters() const { return _parameter; }
    /// Names of the trained globals, in the order they appear in @ref
    /// parameters.
    const std::vector<std::string>& param_names() const { return _param_names; }

private:
    ContextPtr _context;
    RuntimePtr _runtime;
    LRSchedule _schedule;
    double _learning_rate;
    std::size_t _step;
    std::size_t _step_count;
    double _beta1;
    double _beta2;
    double _eps;
    double _grad_clip_threshold;
    double _weight_decay;
    double _loss_mean;
    RuntimePtr _grad_clipper;
    Tensor _one;
    Tensor _parameter;
    Tensor _exp_avg;
    Tensor _exp_avg_sq;
    Tensor _threshold_tensor;
    TypeVec _input_types;
    std::vector<std::string> _param_names;
};

} // namespace madspace
