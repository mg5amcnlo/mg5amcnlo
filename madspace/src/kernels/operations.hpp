#pragma once

#include "madspace/compgraphs/function.hpp"
#include "madspace/driver/tensor.hpp"
#include "madspace/util.hpp"

namespace madspace {
namespace kernels {

template <auto foreach_func, int n_in, int n_out, typename I, typename D>
void batch_foreach(const I& instruction, TensorVec& locals, D& device) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    std::array<const Tensor*, n_in> inputs;
    for (int i = 0; i < n_in; ++i) {
        inputs[i] = &locals[instruction.input_indices[i]];
    }

    std::array<Tensor*, n_out> outputs;
    for (int i = 0; i < n_out; ++i) {
        auto& output = locals[instruction.output_indices[i]];
        auto& output_shape = instruction.output_shapes[i];
        Sizes shape(output_shape.size() + 1);
        shape[0] = batch_size;
        std::copy(output_shape.begin(), output_shape.end(), shape.begin() + 1);
        output = Tensor(
            instruction.output_dtypes[i],
            shape,
            device,
            instruction.output_alloc_hints[i]
        );
        outputs[i] = &output;
    }

    foreach_func(inputs, outputs, batch_size, device);
}

template <
    auto foreach_func,
    int n_in,
    int n_out,
    int n_in_stored,
    int n_out_stored,
    typename I,
    typename D>
void backward_batch_foreach(
    const I& instruction,
    TensorVec& locals,
    TensorVec& local_grads,
    std::array<std::size_t, n_in_stored> in_stored_indices,
    std::array<std::size_t, n_out_stored> out_stored_indices,
    const D& device
) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    constexpr int n_args = n_in_stored + n_out_stored + n_out;
    std::array<const Tensor*, n_args> args;
    std::array<Tensor*, n_in> input_grads;
    for (int i = 0; i < n_in_stored; ++i) {
        args[i] = &locals[instruction.input_indices[in_stored_indices[i]]];
    }
    for (int i = 0; i < n_out_stored; ++i) {
        args[n_in_stored + i] =
            &locals[instruction.output_indices[out_stored_indices[i]]];
    }
    for (int i = 0; i < n_out; ++i) {
        args[n_in_stored + n_out_stored + i] =
            &local_grads[instruction.output_indices[i]];
    }
    for (int i = 0; i < n_in; ++i) {
        auto input_index = instruction.input_indices[i];
        auto& input_grad = local_grads[input_index];
        if (!input_grad) {
            auto& input = locals[input_index];
            /*if (input.size(0) != batch_size && input.dtype() == DataType::dt_float) {
                throw std::runtime_error("backward not possible for broadcasting args");
            }*/
            input_grad = Tensor(
                input.dtype(),
                input.shape(),
                device,
                instruction.input_grad_alloc_hints[i]
            );
            // input_grad.zero(device);
        }
        input_grads[i] = &input_grad;
    }

    device.sync_barrier();
    foreach_func(args, input_grads, batch_size, device);
}

template <typename I, typename D>
void op_stack(const I& instruction, TensorVec& locals, const D& device) {
    auto& first_shape = locals[instruction.input_indices[0]].shape();
    Sizes shape(first_shape.size() + 1);
    shape[0] = locals[instruction.batch_size_index].size(0);
    shape[1] = instruction.input_indices.size();
    std::copy(first_shape.begin() + 1, first_shape.end(), shape.begin() + 2);
    Tensor output(
        instruction.output_dtypes.front(),
        shape,
        device,
        instruction.output_alloc_hints[0]
    );
    std::size_t index = 0;
    for (auto input_index : instruction.input_indices) {
        output.select(1, index).copy_from(locals[input_index], device);
        ++index;
    }
    locals[instruction.output_indices[0]] = output;
}

template <typename I, typename D>
void backward_op_stack(
    const I& instruction, TensorVec& locals, TensorVec& local_grads, const D& device
) {
    // TODO: differentiate integer and float here (also other backwards)
    auto& output_grad = local_grads[instruction.output_indices[0]];
    for (std::size_t i = 0; std::size_t input_index : instruction.input_indices) {
        auto& input_grad = local_grads[input_index];
        if (!input_grad) {
            input_grad = Tensor(
                DataType::dt_float,
                locals[input_index].shape(),
                device,
                instruction.input_grad_alloc_hints[i]
            );
            // input_grad.zero(device);
        }
        ++i;
    }

    device.sync_barrier();
    for (std::size_t index = 0; auto input_index : instruction.input_indices) {
        auto& input_grad = local_grads[input_index];
        input_grad.add(output_grad.select(1, index), device);
        ++index;
    }
}

template <typename I, typename D>
void op_unstack(const I& instruction, TensorVec& locals, const D& device) {
    auto tensors = locals[instruction.input_indices[0]].unstack(1);
    for (auto [tensor, output_index] : zip(tensors, instruction.output_indices)) {
        locals[output_index] = tensor;
    }
}

template <typename I, typename D>
void op_stack_sizes(const I& instruction, TensorVec& locals, const D& device) {
    SizeVec sizes;
    for (std::size_t index : instruction.input_indices) {
        sizes.push_back(locals[index].batch_sizes().at(0));
    }
    locals[instruction.output_indices[0]] = Tensor(sizes);
}

template <typename I, typename D>
void op_unstack_sizes(const I& instruction, TensorVec& locals, const D& device) {
    auto sizes = locals[instruction.input_indices[0]].batch_sizes();
    for (auto [size, output_index] : zip(sizes, instruction.output_indices)) {
        locals[output_index] = Tensor({size});
    }
}

template <typename I, typename D>
void backward_op_unstack(
    const I& instruction, TensorVec& locals, TensorVec& local_grads, const D& device
) {
    auto input_index = instruction.input_indices[0];
    auto& input_grad = local_grads[input_index];
    if (!input_grad) {
        input_grad = Tensor(
            DataType::dt_float,
            locals[input_index].shape(),
            device,
            instruction.input_grad_alloc_hints[0]
        );
        // input_grad.zero(device);
    }
    device.sync_barrier();
    auto unstacked_grads = input_grad.unstack(1);
    for (auto [grad, output_index] : zip(unstacked_grads, instruction.output_indices)) {
        grad.add(local_grads[output_index], device);
    }
}

template <typename I, typename D>
void op_pop(const I& instruction, TensorVec& locals, const D& device) {
    auto input = locals[instruction.input_indices[0]];
    std::size_t last_index = input.size(1) - 1;
    locals[instruction.output_indices[0]] = input.slice(1, 0, last_index);
    locals[instruction.output_indices[1]] = input.select(1, last_index);
}

template <typename I, typename D>
void backward_op_pop(
    const I& instruction, TensorVec& locals, TensorVec& local_grads, const D& device
) {
    auto input_index = instruction.input_indices[0];
    auto input_grad = local_grads[input_index];
    if (!input_grad) {
        input_grad = Tensor(
            DataType::dt_float,
            locals[input_index].shape(),
            device,
            instruction.input_grad_alloc_hints[0]
        );
        // input_grad.zero(device);
    }
    device.sync_barrier();
    std::size_t last_index = input_grad.size(1) - 1;
    input_grad.slice(1, 0, last_index)
        .add(local_grads[instruction.output_indices[0]], device);
    input_grad.select(1, last_index)
        .add(local_grads[instruction.output_indices[1]], device);
}

template <typename I, typename D>
void op_batch_cat(const I& instruction, TensorVec& locals, const D& device) {
    std::size_t batch_size = 0;
    SizeVec sizes;
    for (auto input_index : instruction.input_indices) {
        auto size = locals[input_index].size(0);
        sizes.push_back(size);
        batch_size += size;
    }
    auto shape = locals[instruction.input_indices.front()].shape();
    shape[0] = batch_size;
    Tensor output(
        instruction.output_dtypes.front(),
        shape,
        device,
        instruction.output_alloc_hints[0]
    );
    std::size_t offset = 0;
    for (auto input_index : instruction.input_indices) {
        auto& input = locals[input_index];
        auto next_offset = offset + input.size(0);
        output.slice(0, offset, next_offset).copy_from(input, device);
        offset = next_offset;
    }

    locals[instruction.output_indices[0]] = output;
    locals[instruction.output_indices[1]] = Tensor(sizes);
}

template <typename I, typename D>
void backward_op_batch_cat(
    const I& instruction, TensorVec& locals, TensorVec& local_grads, const D& device
) {
    auto output_grad = local_grads[instruction.output_indices[0]];
    for (std::size_t i = 0; std::size_t input_index : instruction.input_indices) {
        auto& input_grad = local_grads[input_index];
        if (!input_grad) {
            input_grad = Tensor(
                DataType::dt_float,
                locals[input_index].shape(),
                device,
                instruction.input_grad_alloc_hints[i]
            );
            // input_grad.zero(device);
        }
        ++i;
    }
    device.sync_barrier();
    for (std::size_t offset = 0; auto input_index : instruction.input_indices) {
        auto& input_grad = local_grads[input_index];
        auto next_offset = offset + input_grad.size(0);
        input_grad.add(output_grad.slice(0, offset, next_offset), device);
        offset = next_offset;
    }
}

template <typename I, typename D>
void op_batch_split(const I& instruction, TensorVec& locals, const D& device) {
    auto& sizes = locals[instruction.input_indices[1]].batch_sizes();
    auto tensors = locals[instruction.input_indices[0]].split(0, sizes);
    for (auto [tensor, output_index] : zip(tensors, instruction.output_indices)) {
        locals[output_index] = tensor;
    }
}

template <typename I, typename D>
void backward_op_batch_split(
    const I& instruction, TensorVec& locals, TensorVec& local_grads, const D& device
) {
    auto& sizes = locals[instruction.input_indices[1]].batch_sizes();
    auto input_index = instruction.input_indices[0];
    auto& input_grad = local_grads[input_index];
    if (!input_grad) {
        input_grad = Tensor(
            DataType::dt_float,
            locals[input_index].shape(),
            device,
            instruction.input_grad_alloc_hints[0]
        );
        // input_grad.zero(device);
    }
    device.sync_barrier();
    auto split_grads = input_grad.split(0, sizes);
    for (auto [tensor, output_index] : zip(split_grads, instruction.output_indices)) {
        tensor.add(local_grads[output_index], device);
    }
}

template <typename I, typename D>
void op_cat(const I& instruction, TensorVec& locals, const D& device) {
    std::size_t cat_size = 0;
    for (auto input_index : instruction.input_indices) {
        cat_size += locals[input_index].size(1);
    }

    auto& first_shape = locals[instruction.input_indices[0]].shape();
    Sizes shape(first_shape.size());
    shape[0] = locals[instruction.batch_size_index].size(0);
    shape[1] = cat_size;
    std::copy(first_shape.begin() + 2, first_shape.end(), shape.begin() + 2);

    Tensor output(
        instruction.output_dtypes.front(),
        shape,
        device,
        instruction.output_alloc_hints[0]
    );
    std::size_t offset = 0;
    for (auto input_index : instruction.input_indices) {
        auto& input = locals[input_index];
        auto next_offset = offset + input.size(1);
        output.slice(1, offset, next_offset).copy_from(input, device);
        offset = next_offset;
    }
    locals[instruction.output_indices[0]] = output;
}

template <typename I, typename D>
void backward_op_cat(
    const I& instruction, TensorVec& locals, TensorVec& local_grads, const D& device
) {
    auto& output_grad = local_grads[instruction.output_indices[0]];
    for (std::size_t i = 0; std::size_t input_index : instruction.input_indices) {
        auto& input_grad = local_grads[input_index];
        if (!input_grad) {
            input_grad = Tensor(
                DataType::dt_float,
                locals[input_index].shape(),
                device,
                instruction.input_grad_alloc_hints[i]
            );
            // input_grad.zero(device);
        }
        ++i;
    }

    device.sync_barrier();
    for (std::size_t offset = 0; auto input_index : instruction.input_indices) {
        auto& input_grad = local_grads[input_index];
        auto next_offset = offset + input_grad.size(1);
        input_grad.add(output_grad.slice(1, offset, next_offset), device);
        offset = next_offset;
    }
}

template <typename I, typename D>
void op_batch_size(const I& instruction, TensorVec& locals, const D& device) {
    SizeVec batch_size{locals[instruction.batch_size_index].size(0)};
    locals[instruction.output_indices[0]] = Tensor(batch_size);
}

template <typename I, typename D>
void op_full(const I& instruction, TensorVec& locals, const D& device) {
    auto& input = locals[instruction.input_indices[0]];
    std::size_t batch_size = locals[instruction.input_indices[1]].batch_sizes().at(0);
    auto& out_shape = instruction.output_shapes[0];
    Sizes shape(out_shape.size() + 1);
    shape[0] = batch_size;
    std::copy(out_shape.begin(), out_shape.end(), shape.begin() + 1);
    locals[instruction.output_indices[0]] = input.expand(shape);
}

template <typename I, typename D>
void op_squeeze(const I& instruction, TensorVec& locals, const D& device) {
    auto tensor = locals[instruction.input_indices[0]].select(1, 0);
    locals[instruction.output_indices[0]] = tensor;
}

template <typename I, typename D>
void backward_op_squeeze(
    const I& instruction, TensorVec& locals, TensorVec& local_grads, const D& device
) {
    auto output_grad = local_grads[instruction.output_indices[0]].unsqueeze(1);
    auto& input_grad = local_grads[instruction.input_indices[0]];
    if (input_grad) {
        input_grad.add(output_grad, device);
    } else {
        input_grad = Tensor(
            output_grad.dtype(),
            output_grad.shape(),
            device,
            instruction.input_grad_alloc_hints[0]
        );
        input_grad.copy_from(output_grad, device);
    }
}

template <typename I, typename D>
void op_unsqueeze(const I& instruction, TensorVec& locals, const D& device) {
    auto tensor = locals[instruction.input_indices[0]].unsqueeze(1);
    locals[instruction.output_indices[0]] = tensor;
}

template <typename I, typename D>
void backward_op_unsqueeze(
    const I& instruction, TensorVec& locals, TensorVec& local_grads, const D& device
) {
    auto output_grad = local_grads[instruction.output_indices[0]].select(1, 0);
    auto& input_grad = local_grads[instruction.input_indices[0]];
    if (input_grad) {
        input_grad.add(output_grad, device);
    } else {
        input_grad = Tensor(
            output_grad.dtype(),
            output_grad.shape(),
            device,
            instruction.input_grad_alloc_hints[0]
        );
        input_grad.copy_from(output_grad, device);
    }
}

template <typename I, typename D>
void op_accept_norm(const I& instruction, TensorVec& locals, const D& device) {
    double size_acc = locals[instruction.input_indices[0]].size(0);
    double size_full = locals[instruction.input_indices[1]].size(0);
    locals[instruction.output_indices[0]] =
        Tensor(size_full / size_acc, device, instruction.output_alloc_hints[0]);
}

template <typename I, typename D>
void op_rqs_reshape(const I& instruction, TensorVec& locals, const D& device) {
    auto n_dims = instruction.output_shapes[0][0];
    auto n_bins = instruction.output_shapes[0][1];
    auto input = locals[instruction.input_indices[0]].factor_dim(1, n_dims);
    locals[instruction.output_indices[0]] = input.slice(2, 0, n_bins);
    locals[instruction.output_indices[1]] = input.slice(2, n_bins, 2 * n_bins);
    locals[instruction.output_indices[2]] = input.slice(2, 2 * n_bins, 3 * n_bins + 1);
}

template <typename I, typename D>
void backward_op_rqs_reshape(
    const I& instruction, TensorVec& locals, TensorVec& local_grads, const D& device
) {
    auto n_dims = instruction.output_shapes[0][0];
    auto n_bins = instruction.output_shapes[0][1];
    auto input_index = instruction.input_indices[0];
    auto& input_grad = local_grads[input_index];
    if (!input_grad) {
        input_grad = Tensor(
            DataType::dt_float,
            locals[input_index].shape(),
            device,
            instruction.input_grad_alloc_hints[0]
        );
        // input_grad.zero(device);
    }
    device.sync_barrier();
    auto input_grad_reshaped = input_grad.factor_dim(1, n_dims);
    input_grad_reshaped.slice(2, 0, n_bins)
        .add(local_grads[instruction.output_indices[0]], device);
    input_grad_reshaped.slice(2, n_bins, 2 * n_bins)
        .add(local_grads[instruction.output_indices[1]], device);
    input_grad_reshaped.slice(2, 2 * n_bins, 3 * n_bins + 1)
        .add(local_grads[instruction.output_indices[2]], device);
}

} // namespace kernels
} // namespace madspace
