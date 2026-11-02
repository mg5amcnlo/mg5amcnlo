#include "madspace/driver/adam_optimizer.hpp"

#include "madspace/constants.hpp"

using namespace madspace;

GradientClipper::GradientClipper(DataType dtype) :
    FunctionGenerator(
        "Unweighter",
        {{"gradients_in", Type(dtype, batch_size, {})}, {"threshold", single_float}},
        {{"gradients_out", Type(dtype, batch_size, {})},
         {"gradient_norm", single_float}}
    ) {}

NamedVector<Value> GradientClipper::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    // only the norm is computed in double precision
    Value grads_in = args.at(0);
    bool single = grads_in.type.dtype == DataType::dt_float32;
    Value grads_sq = fb.square(single ? fb.convert_float64(grads_in) : grads_in);
    Value grad_norm = fb.sqrt(fb.batch_reduce_sum(grads_sq));
    Value factor = fb.min(fb.div(args.at(1), grad_norm), 1.0);
    Value grads_out = fb.mul(grads_in, single ? fb.convert_float32(factor) : factor);
    return {
        {"gradients_out", grads_out},
        {"gradient_norm", grad_norm},
    };
}

AdamOptimizer::AdamOptimizer(
    const Function& function,
    ContextPtr context,
    double learning_rate,
    LRSchedule schedule,
    std::size_t step_count,
    double beta1,
    double beta2,
    double eps,
    double grad_clip_threshold,
    double weight_decay
) :
    _context(context),
    _learning_rate(learning_rate),
    _schedule(schedule),
    _step(0),
    _step_count(step_count),
    _beta1(beta1),
    _beta2(beta2),
    _eps(eps),
    _grad_clip_threshold(grad_clip_threshold),
    _weight_decay(weight_decay),
    _one(1.0, context->device()),
    _loss_mean(std::numeric_limits<double>::quiet_NaN()) {
    DevicePtr device = context->device();
    for (auto& [name, value] : function.globals()) {
        if (context->global_requires_grad(name)) {
            _param_names.push_back(name);
        }
    }
    _parameter = context->reallocate_globals_contiguously(_param_names);
    _runtime = build_runtime(function, context);
    if (grad_clip_threshold > 0.) {
        _grad_clipper = build_runtime(
            GradientClipper(_parameter.dtype()).function(), context, false
        );
    }
    _exp_avg = Tensor(_parameter.dtype(), _parameter.shape(), _parameter.device());
    _exp_avg.zero();
    _exp_avg_sq = Tensor(_parameter.dtype(), _parameter.shape(), _parameter.device());
    _exp_avg_sq.zero();
    _threshold_tensor = Tensor(
        _grad_clip_threshold * std::sqrt(_parameter.size(0)), _parameter.device()
    );
    _input_types.reserve(function.inputs().size());
    for (auto& input : function.inputs()) {
        _input_types.push_back(input.type);
    }
}

TensorVec AdamOptimizer::step(const TensorVec& inputs) {
    double lr = learning_rate();
    ++_step;
    double bias_corr1 = 1 - std::pow(_beta1, _step);
    double bias_corr2 = 1 - std::pow(_beta2, _step);
    double step_size = lr / bias_corr1;
    double bias_corr2_sqrt = std::sqrt(bias_corr2);
    auto [outputs, stored_locals, eval_grad] =
        _runtime->run_with_grad(inputs, std::vector<bool>(inputs.size(), false));
    Tensor loss_cpu = outputs.at(0).cpu();
    double loss = loss_cpu.view<double, 1>()[0];
    // TODO: return loss as double
    if (std::isnan(loss)) {
        return outputs;
    }
    if (std::isnan(_loss_mean)) {
        _loss_mean = loss;
    } else {
        if (_step > 100 && loss > 20 * _loss_mean) {
            return outputs;
        }
        _loss_mean = 0.05 * loss + 0.95 * _loss_mean;
    }
    TensorVec output_grads(outputs.size());
    DevicePtr device = _context->device();
    output_grads.at(0) = _one;
    auto [input_grads, global_grads] =
        _runtime->run_backward(output_grads, stored_locals, eval_grad, true);

    if (_grad_clipper) {
        auto clip_result = _grad_clipper->run({global_grads.at(0), _threshold_tensor});
        global_grads = {clip_result.at(0)};
        // double norm = clip_result.at(1).cpu().view<double, 1>()[0];
    }

    device->adam_step(
        global_grads.at(0),
        _parameter,
        _exp_avg,
        _exp_avg_sq,
        step_size,
        _beta1,
        _beta2,
        _eps,
        bias_corr2_sqrt,
        lr * _weight_decay
    );
    return outputs;
}

void AdamOptimizer::replace_function(const Function& function) {
    std::unordered_map<std::string, std::pair<std::size_t, std::size_t>>
        offsets_and_sizes;
    for (std::size_t offset = 0; auto& name : _param_names) {
        std::size_t size = _context->global(name).shape().product();
        offsets_and_sizes[name] = {offset, size};
        offset += size;
    }
    _param_names.clear();
    for (auto& [name, value] : function.globals()) {
        if (_context->global_requires_grad(name)) {
            _param_names.push_back(name);
        }
    }
    _runtime.reset();
    _parameter = _context->reallocate_globals_contiguously(_param_names);
    _runtime = build_runtime(function, _context);
    if (_grad_clipper) {
        _grad_clipper = build_runtime(
            GradientClipper(_parameter.dtype()).function(), _context, false
        );
    }
    Tensor old_ea = _exp_avg, old_eas = _exp_avg_sq;
    _exp_avg = Tensor(_parameter.dtype(), _parameter.shape(), _parameter.device());
    _exp_avg_sq = Tensor(_parameter.dtype(), _parameter.shape(), _parameter.device());
    for (std::size_t offset = 0; auto& name : _param_names) {
        auto [old_offset, size] = offsets_and_sizes.at(name);
        _exp_avg.slice(0, offset, offset + size)
            .copy_from(old_ea.slice(0, old_offset, old_offset + size));
        _exp_avg_sq.slice(0, offset, offset + size)
            .copy_from(old_eas.slice(0, old_offset, old_offset + size));
        offset += size;
    }
    _threshold_tensor = Tensor(
        _grad_clip_threshold * std::sqrt(_parameter.size(0)), _parameter.device()
    );
}

double AdamOptimizer::learning_rate() const {
    switch (_schedule) {
    case none:
        return _learning_rate;
    case cosine:
        return 0.5 * _learning_rate * (1 + std::cos(_step * PI / _step_count));
    default:
        throw std::runtime_error("Invalid LR schedule");
    }
}
