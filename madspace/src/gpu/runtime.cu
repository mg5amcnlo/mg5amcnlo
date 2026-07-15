#include "runtime.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <functional>
#include <random>
#include <tuple>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>

#include "../kernels/kernels.hpp"
#include "../kernels/operations.hpp"
#include "device.hpp"
#include "madspace/util.hpp"
#include "tensor.cuh"

using namespace madspace;
using namespace madspace::gpu;
using namespace madspace::kernels;

namespace {

void op_matrix_element(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    std::size_t me_index = locals[instruction.input_indices[0]].index_value();
    std::size_t input_count = locals[instruction.input_indices[1]].index_value();
    std::size_t output_count = locals[instruction.input_indices[2]].index_value();
    TensorVec contiguous_inputs(input_count);
    std::vector<UmamiInputKey> input_keys(input_count + 1);
    std::vector<UmamiOutputKey> output_keys(output_count);
    std::vector<void*> input_ptrs(input_count), output_ptrs(output_count + 1);
    for (std::size_t i = 0; i < input_count; ++i) {
        input_keys[i] = static_cast<UmamiInputKey>(
            locals[instruction.input_indices[3 + 2 * i]].index_value()
        );
        contiguous_inputs[i] =
            locals[instruction.input_indices[3 + 2 * i + 1]].contiguous(
                batch_size, device, AllocHint::temporary
            );
        input_ptrs[i] = contiguous_inputs[i].data();
    }
    std::size_t output_offset = 3 + 2 * input_count;
    for (std::size_t i = 0; i < output_count; ++i) {
        output_keys[i] = static_cast<UmamiOutputKey>(
            locals[instruction.input_indices[output_offset + 2 * i]].index_value()
        );
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
        output_ptrs[i] = output.data();
        if (me_index == 0xBADCAFE) {
            // flat dummy matrix element for testing purposes
            // implemented at LUMI hackathon where the coffee was indeed terrible
            switch (output_keys[i]) {
            case UMAMI_OUT_MATRIX_ELEMENT:
                thrust::fill_n(
                    thrust_par.on(device.stream()),
                    thrust::device_pointer_cast(static_cast<double*>(output_ptrs[i])),
                    batch_size,
                    1.0
                );
                break;
            case UMAMI_OUT_DIAGRAM_AMP2:
                thrust::fill_n(
                    thrust_par.on(device.stream()),
                    thrust::device_pointer_cast(static_cast<double*>(output_ptrs[i])),
                    batch_size * shape[1],
                    1. / shape[1]
                );
                break;
            default:
                output.zero(device);
                break;
            }
        }
    }
    output_keys[output_count] = UMAMI_OUT_GPU_STREAM;
    output_ptrs[output_count] = device.stream();
    if (me_index == 0xBADCAFE || batch_size == 0) {
        return;
    }
    auto& matrix_element = instruction.runtime.context().matrix_element(me_index);
    if (matrix_element.device_type() != GpuDevice::gpu_device_type) {
        throw std::runtime_error("Matrix element has incompatible device");
    }
    device.sync_barrier();

    matrix_element.call(
        matrix_element.process_instance(),
        batch_size,
        batch_size,
        0,
        input_count,
        input_keys.data(),
        input_ptrs.data(),
        output_count,
        output_keys.data(),
        output_ptrs.data()
    );
    for (auto& input : contiguous_inputs) {
        input.reset(device);
    }
}

void op_matmul(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto input =
        locals[instruction.input_indices[0]].contiguous(device, AllocHint::temporary);
    auto weight =
        locals[instruction.input_indices[1]].contiguous(device, AllocHint::temporary);
    auto bias =
        locals[instruction.input_indices[2]].contiguous(device, AllocHint::temporary);
    auto& output = locals[instruction.output_indices[0]];
    std::size_t batch_size = input.size(0);
    std::size_t dims_in = input.size(1);
    std::size_t dims_out = weight.size(1);
    output = Tensor(
        DataType::dt_float,
        {batch_size, dims_out},
        device,
        instruction.output_alloc_hints[0]
    );
    output.copy_from(bias, device);
    if (batch_size == 0) {
        return;
    }

    gpublasHandle_t handle = instruction.runtime.gpublas_handle();
    check_error(gpublasSetStream(handle, device.stream()));
    double alpha = 1., beta = 1.;
    check_error(gpublasDgemm(
        handle,
        GPUBLAS_OP_N,
        GPUBLAS_OP_T,
        batch_size,
        dims_out,
        dims_in,
        &alpha,
        static_cast<double*>(input.data()),
        batch_size,
        static_cast<double*>(weight.data()),
        dims_out,
        &beta,
        static_cast<double*>(output.data()),
        batch_size
    ));
    input.reset(device);
    weight.reset(device);
    bias.reset(device);
}

__global__ void
kernel_one(std::size_t batch_size, GpuTensorView<double, 1, true> output) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        output[i] = 1.;
    }
}

void backward_op_matmul(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    TensorVec& local_grads,
    const AsyncGpuDevice& device
) {
    auto input =
        locals[instruction.input_indices[0]].contiguous(device, AllocHint::temporary);
    auto weight =
        locals[instruction.input_indices[1]].contiguous(device, AllocHint::temporary);
    auto output_grad = local_grads[instruction.output_indices[0]].contiguous(
        device, AllocHint::temporary
    );
    auto& input_grad = local_grads[instruction.input_indices[0]];
    auto& weight_grad = local_grads[instruction.input_indices[1]];
    auto& bias_grad = local_grads[instruction.input_indices[2]];
    std::size_t batch_size = input.size(0);
    std::size_t dims_in = input.size(1);
    std::size_t dims_out = weight.size(1);

    if (!input_grad) {
        input_grad = Tensor(
            DataType::dt_float,
            input.shape(),
            device,
            instruction.input_grad_alloc_hints[0]
        );
        // input_grad.zero(device);
    }
    if (!weight_grad) {
        weight_grad = Tensor(
            DataType::dt_float,
            weight.shape(),
            device,
            instruction.input_grad_alloc_hints[1]
        );
        // weight_grad.zero(device);
    }
    if (!bias_grad) {
        bias_grad = Tensor(
            DataType::dt_float,
            {1, dims_out},
            device,
            instruction.input_grad_alloc_hints[2]
        );
        // bias_grad.zero(device);
    }
    if (batch_size == 0) {
        input.reset(device);
        weight.reset(device);
        output_grad.reset(device);
        return;
    }

    double alpha = 1., beta = 1.;
    gpublasHandle_t handle = instruction.runtime.gpublas_handle();
    gpuStream_t stream = device.stream();
    check_error(gpublasSetStream(handle, stream));

    // compute input_grad += output_grad * weight
    check_error(gpublasDgemm(
        handle,
        GPUBLAS_OP_N,
        GPUBLAS_OP_N,
        batch_size,
        dims_in,
        dims_out,
        &alpha,
        static_cast<double*>(output_grad.data()),
        batch_size,
        static_cast<double*>(weight.data()),
        dims_out,
        &beta,
        static_cast<double*>(input_grad.data()),
        batch_size
    ));

    // compute weight_grad += output_grad.T * input
    check_error(gpublasDgemm(
        handle,
        GPUBLAS_OP_T,
        GPUBLAS_OP_N,
        dims_out,
        dims_in,
        batch_size,
        &alpha,
        static_cast<double*>(output_grad.data()),
        batch_size,
        static_cast<double*>(input.data()),
        batch_size,
        &beta,
        static_cast<double*>(weight_grad.data()),
        dims_out
    ));

    // compute bias_grad += sum_i output_grad_ij
    Tensor ones(DataType::dt_float, {batch_size}, device, AllocHint::temporary);
    launch_kernel(
        kernel_one, batch_size, device.stream(), batch_size, ones.view<double, 1>()
    );
    check_error(gpublasDgemv(
        handle,
        GPUBLAS_OP_T,
        batch_size,
        dims_out,
        &alpha,
        static_cast<double*>(output_grad.data()),
        batch_size,
        static_cast<double*>(ones.data()),
        1,
        &beta,
        static_cast<double*>(bias_grad.data()),
        1
    ));
    ones.reset(device);
    input.reset(device);
    weight.reset(device);
    output_grad.reset(device);
}

struct NotMinusOne {
    __device__ bool operator()(me_int_t val) { return val != -1; }
};

__global__ void kernel_nonzero(
    std::size_t batch_size,
    GpuTensorView<double, 1, true> input,
    GpuTensorView<me_int_t, 1, true> output
) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        output[i] = input[i] == 0. ? -1 : i;
    }
}

void op_nonzero(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto& input = locals[instruction.input_indices[0]];
    auto batch_size = input.size(0);
    auto& output = locals[instruction.output_indices[0]];
    Tensor indices_tmp(DataType::dt_int, {batch_size}, device, AllocHint::temporary);
    Tensor output_tmp(
        DataType::dt_int, {batch_size}, device, instruction.output_alloc_hints[0]
    );
    launch_kernel(
        kernel_nonzero,
        batch_size,
        device.stream(),
        batch_size,
        input.view<double, 1>(),
        indices_tmp.view<me_int_t, 1>()
    );

    auto indices_ptr =
        thrust::device_pointer_cast(static_cast<me_int_t*>(indices_tmp.data()));
    auto output_ptr =
        thrust::device_pointer_cast(static_cast<me_int_t*>(output_tmp.data()));
    auto count =
        thrust::copy_if(
            thrust_par.on(device.stream()),
            indices_ptr,
            indices_ptr + batch_size,
            output_ptr,
            NotMinusOne()
        ) -
        output_ptr; // TODO: use stream
    output = output_tmp.slice(0, 0, count);
    indices_tmp.reset(device);
    output_tmp.reset(device);
}

template <int dim>
__global__ void batch_gather_kernel(
    std::size_t batch_size,
    GpuTensorView<me_int_t, 1, true> indices,
    GpuTensorView<double, dim, true> values,
    GpuTensorView<double, dim, true> selection
) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        recursive_for<kernel_copy<GpuTypes>, dim - 1>(values[indices[i]], selection[i]);
    }
}

template <int dim>
__global__ void batch_gather_kernel_int(
    std::size_t batch_size,
    GpuTensorView<me_int_t, 1, true> indices,
    GpuTensorView<me_int_t, dim, true> values,
    GpuTensorView<me_int_t, dim, true> selection
) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        recursive_for<kernel_copy_int<GpuTypes>, dim - 1>(
            values[indices[i]], selection[i]
        );
    }
}

template <int dim>
void batch_gather_impl(
    Tensor& indices,
    Tensor& values,
    Tensor& selection,
    AllocHint hint,
    const AsyncGpuDevice& device
) {
    auto batch_size = indices.size(0);
    Sizes out_shape = values.shape();
    out_shape[0] = batch_size;

    if (values.dtype() == DataType::dt_float) {
        selection = Tensor(DataType::dt_float, out_shape, device, hint);
        launch_kernel(
            batch_gather_kernel<dim>,
            batch_size,
            device.stream(),
            batch_size,
            indices.view<me_int_t, 1>(),
            values.view<double, dim>(),
            selection.view<double, dim>()
        );
    } else if (values.dtype() == DataType::dt_int) {
        selection = Tensor(DataType::dt_int, out_shape, device, hint);
        launch_kernel(
            batch_gather_kernel_int<dim>,
            batch_size,
            device.stream(),
            batch_size,
            indices.view<me_int_t, 1>(),
            values.view<me_int_t, dim>(),
            selection.view<me_int_t, dim>()
        );
    } else {
        throw std::runtime_error("invalid dtype in batch_gather");
    }
}

void op_batch_gather(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto& indices = locals[instruction.input_indices[0]];
    auto& values = locals[instruction.input_indices[1]];
    auto& selection = locals[instruction.output_indices[0]];
    AllocHint hint = instruction.output_alloc_hints[0];
    switch (values.shape().size()) {
    case 1:
        batch_gather_impl<1>(indices, values, selection, hint, device);
        break;
    case 2:
        batch_gather_impl<2>(indices, values, selection, hint, device);
        break;
    case 3:
        batch_gather_impl<3>(indices, values, selection, hint, device);
        break;
    case 4:
        batch_gather_impl<4>(indices, values, selection, hint, device);
        break;
    default:
        throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

template <int dim>
__global__ void batch_scatter_kernel(
    std::size_t batch_size,
    GpuTensorView<me_int_t, 1, true> indices,
    GpuTensorView<double, dim, true> source,
    GpuTensorView<double, dim, true> output
) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        recursive_for<kernel_copy<GpuTypes>, dim - 1>(source[i], output[indices[i]]);
    }
}

template <int dim>
__global__ void batch_scatter_kernel_int(
    std::size_t batch_size,
    GpuTensorView<me_int_t, 1, true> indices,
    GpuTensorView<me_int_t, dim, true> source,
    GpuTensorView<me_int_t, dim, true> output
) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        recursive_for<kernel_copy_int<GpuTypes>, dim - 1>(
            source[i], output[indices[i]]
        );
    }
}

template <int dim>
void batch_scatter_impl(
    Tensor& indices, Tensor& source, Tensor& output, const AsyncGpuDevice& device
) {
    if (source.dtype() == DataType::dt_float) {
        auto batch_size = indices.size(0);
        launch_kernel(
            batch_scatter_kernel<dim>,
            batch_size,
            device.stream(),
            batch_size,
            indices.view<me_int_t, 1>(),
            source.view<double, dim>(),
            output.view<double, dim>()
        );
    } else if (source.dtype() == DataType::dt_int) {
        auto batch_size = indices.size(0);
        launch_kernel(
            batch_scatter_kernel_int<dim>,
            batch_size,
            device.stream(),
            batch_size,
            indices.view<me_int_t, 1>(),
            source.view<me_int_t, dim>(),
            output.view<me_int_t, dim>()
        );
    } else {
        throw std::runtime_error("invalid dtype in batch_scatter");
    }
}

void op_batch_scatter(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto& indices = locals[instruction.input_indices[0]];
    auto& target = locals[instruction.input_indices[1]];
    auto& source = locals[instruction.input_indices[2]];

    auto& output = locals[instruction.output_indices[0]];
    output = target.copy(device, instruction.output_alloc_hints[0]);
    switch (target.shape().size()) {
    case 1:
        batch_scatter_impl<1>(indices, source, output, device);
        break;
    case 2:
        batch_scatter_impl<2>(indices, source, output, device);
        break;
    case 3:
        batch_scatter_impl<3>(indices, source, output, device);
        break;
    case 4:
        batch_scatter_impl<4>(indices, source, output, device);
        break;
    default:
        throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

__global__ void kernel_div_batch_size(
    std::size_t batch_size,
    GpuTensorView<double, 1, true> input,
    GpuTensorView<double, 1, true> output
) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        output[i] = input[i] / batch_size;
    }
}

void batch_reduce_mean_impl(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device,
    bool keepdim
) {
    auto input = locals[instruction.input_indices[0]].contiguous(device);
    auto& output = locals[instruction.output_indices[0]];
    AllocHint hint = instruction.output_alloc_hints[0];
    std::size_t batch_size = input.size(0);

    Tensor out(DataType::dt_float, {1}, device, hint);
    std::size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(
        nullptr,
        temp_storage_bytes,
        static_cast<double*>(input.data()),
        static_cast<double*>(out.data()),
        batch_size,
        device.stream()
    );
    Tensor temp(
        DataType::dt_float, {(temp_storage_bytes + 7) / 8}, device, AllocHint::temporary
    );
    cub::DeviceReduce::Sum(
        temp.data(),
        temp_storage_bytes,
        static_cast<double*>(input.data()),
        static_cast<double*>(out.data()),
        batch_size,
        device.stream()
    );
    temp.reset(device);
    input.reset(device);
    launch_kernel(
        kernel_div_batch_size,
        1,
        device.stream(),
        batch_size,
        out.view<double, 1>(),
        out.view<double, 1>()
    );
    if (keepdim) {
        output = out.expand({batch_size});
    } else {
        output = out;
    }
}

void batch_reduce_mean_backward_impl(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    TensorVec& local_grads,
    const AsyncGpuDevice& device,
    bool keepdim
) {
    auto& input = locals[instruction.input_indices[0]];
    auto& input_grad = local_grads[instruction.input_indices[0]];
    if (!input_grad) {
        input_grad = Tensor(
            DataType::dt_float,
            input.shape(),
            device,
            instruction.input_grad_alloc_hints[0]
        );
        // input_grad.zero(device);
    }

    Tensor grad(DataType::dt_float, {1}, device, AllocHint::temporary);
    if (keepdim) {
        auto output_grad =
            local_grads[instruction.output_indices[0]].contiguous(device);
        std::size_t batch_size = output_grad.size(0);
        std::size_t temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(
            nullptr,
            temp_storage_bytes,
            static_cast<double*>(output_grad.data()),
            static_cast<double*>(grad.data()),
            batch_size,
            device.stream()
        );
        Tensor temp(
            DataType::dt_float,
            {(temp_storage_bytes + 7) / 8},
            device,
            AllocHint::temporary
        );
        cub::DeviceReduce::Sum(
            temp.data(),
            temp_storage_bytes,
            static_cast<double*>(output_grad.data()),
            static_cast<double*>(grad.data()),
            batch_size,
            device.stream()
        );
        launch_kernel(
            kernel_div_batch_size,
            1,
            device.stream(),
            output_grad.size(0),
            grad.view<double, 1>(),
            grad.view<double, 1>()
        );
        temp.reset(device);
        output_grad.reset(device);
    } else {
        auto& output_grad = local_grads[instruction.output_indices[0]];
        launch_kernel(
            kernel_div_batch_size,
            1,
            device.stream(),
            output_grad.size(0),
            output_grad.view<double, 1>(),
            grad.view<double, 1>()
        );
    }
    input_grad.add(grad, device);
    grad.reset(device);
}

void op_batch_reduce_mean(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    batch_reduce_mean_impl(instruction, locals, device, false);
}

void backward_op_batch_reduce_mean(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    TensorVec& local_grads,
    const AsyncGpuDevice& device
) {
    batch_reduce_mean_backward_impl(instruction, locals, local_grads, device, false);
}

void op_batch_reduce_mean_keepdim(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    batch_reduce_mean_impl(instruction, locals, device, true);
}

void backward_op_batch_reduce_mean_keepdim(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    TensorVec& local_grads,
    const AsyncGpuDevice& device
) {
    batch_reduce_mean_backward_impl(instruction, locals, local_grads, device, true);
}

void op_offset_indices(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto& sizes_offset = locals[instruction.input_indices[0]].batch_sizes();
    auto& sizes_out = locals[instruction.input_indices[1]].batch_sizes();
    std::size_t total_size = std::accumulate(sizes_out.begin(), sizes_out.end(), 0);
    auto& output = locals[instruction.output_indices[0]];
    output = Tensor(
        DataType::dt_int, {total_size}, device, instruction.output_alloc_hints[0]
    );
    std::size_t sum_offset = 0, sum_out = 0;
    for (auto [size_offset, size_out] : zip(sizes_offset, sizes_out)) {
        thrust::fill_n(
            thrust_par.on(device.stream()),
            thrust::device_pointer_cast(
                static_cast<me_int_t*>(output.data()) + sum_out
            ),
            size_out,
            sum_offset
        );
        sum_offset += size_offset;
        sum_out += size_out;
    }
}

__global__ void kernel_gather_quantile(
    std::size_t batch_size,
    GpuTensorView<double, 1, true> q,
    GpuTensorView<double, 1, true> input,
    GpuTensorView<double, 1, true> output
) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i == 0) {
        double q_scaled = q[0] < 0. ? 0. : q[0] * batch_size;
        std::size_t q_index = static_cast<std::size_t>(q_scaled);
        if (q_index >= batch_size) {
            q_index = batch_size - 1;
        }
        output[0] = input[q_index];
    }
}

void op_quantile(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto& input = locals[instruction.input_indices[0]];
    auto batch_size = input.size(0);
    auto& q = locals[instruction.input_indices[1]];
    auto& output = locals[instruction.output_indices[0]];

    if (batch_size == 0) {
        output = Tensor(0., device, instruction.output_alloc_hints[0]);
        return;
    }

    Tensor tmp(DataType::dt_float, {batch_size}, device, AllocHint::temporary);
    tmp.copy_from(input, device);
    std::size_t temp_storage_bytes;
    cub::DeviceMergeSort::SortKeys(
        nullptr,
        temp_storage_bytes,
        static_cast<double*>(tmp.data()),
        batch_size,
        gpu_less_double{},
        device.stream()
    );
    Tensor tmp_sort(
        DataType::dt_float, {(temp_storage_bytes + 7) / 8}, device, AllocHint::temporary
    );
    cub::DeviceMergeSort::SortKeys(
        tmp_sort.data(),
        temp_storage_bytes,
        static_cast<double*>(tmp.data()),
        batch_size,
        gpu_less_double{},
        device.stream()
    );
    tmp_sort.reset(device);

    output = Tensor(DataType::dt_float, {1}, device, instruction.output_alloc_hints[0]);
    launch_kernel(
        kernel_gather_quantile,
        1,
        device.stream(),
        batch_size,
        q.view<double, 1>(),
        tmp.view<double, 1>(),
        output.view<double, 1>()
    );
    tmp.reset(device);
}

void op_random(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto batch_size = locals[instruction.input_indices[0]].batch_sizes()[0];
    auto& output = locals[instruction.output_indices[0]];
    auto dim = instruction.output_shapes[0][0];
    output = Tensor(
        DataType::dt_float, {batch_size, dim}, device, instruction.output_alloc_hints[0]
    );
    gpurandGenerator_t generator = instruction.runtime.gpurand_generator();
    check_error(gpurandSetStream(generator, device.stream()));
    check_error(gpurandGenerateUniformDouble(
        generator, static_cast<double*>(output.data()), batch_size * dim
    ));
}

__global__ void kernel_double_to_int_range(
    std::size_t batch_size,
    me_int_t max_val,
    GpuTensorView<double, 1, true> double_in,
    GpuTensorView<me_int_t, 1, true> int_out
) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= batch_size) {
        return;
    }
    me_int_t rand_int = double_in[i] * max_val;
    if (rand_int >= max_val) {
        rand_int = max_val - 1;
    }
    int_out[i] = rand_int;
}

void op_random_int(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto batch_size = locals[instruction.input_indices[0]].batch_sizes()[0];
    auto max_val = locals[instruction.input_indices[1]].batch_sizes()[0];
    auto& output = locals[instruction.output_indices[0]];
    Tensor tmp(DataType::dt_float, {batch_size}, device, AllocHint::temporary);
    output = Tensor(
        DataType::dt_int, {batch_size}, device, instruction.output_alloc_hints[0]
    );
    gpurandGenerator_t generator = instruction.runtime.gpurand_generator();
    check_error(gpurandSetStream(generator, device.stream()));
    check_error(gpurandGenerateUniformDouble(
        generator, static_cast<double*>(tmp.data()), batch_size
    ));
    launch_kernel(
        kernel_double_to_int_range,
        batch_size,
        device.stream(),
        batch_size,
        max_val,
        tmp.view<double, 1>(),
        output.view<me_int_t, 1>()
    );
    tmp.reset(device);
}

__global__ void kernel_unweight(
    std::size_t batch_size,
    GpuTensorView<double, 1, true> rand_in,
    GpuTensorView<double, 1, true> weights_in,
    GpuTensorView<double, 1, true> max_weights_in,
    GpuTensorView<double, 1, true> weights_out,
    GpuTensorView<me_int_t, 1, true> indices_out
) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= batch_size) {
        return;
    }

    auto rand = rand_in[i], weight = weights_in[i], max_weight = max_weights_in[i];
    bool accepted = max_weight * rand < weight;
    auto weight_clipped = weight < max_weight ? max_weight : weight;
    weights_out[i] = accepted ? weight_clipped : 0.;
    indices_out[i] = accepted ? i : -1;
}

void op_unweight(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto& weights = locals[instruction.input_indices[0]];
    auto& max_weight = locals[instruction.input_indices[1]];
    auto& indices = locals[instruction.output_indices[0]];
    auto& uw_weights = locals[instruction.output_indices[1]];
    auto batch_size = weights.size(0);
    gpuStream_t stream = device.stream();

    if (batch_size == 0) {
        indices =
            Tensor(DataType::dt_float, {0}, device, instruction.output_alloc_hints[0]);
        uw_weights =
            Tensor(DataType::dt_float, {0}, device, instruction.output_alloc_hints[1]);
        return;
    }

    Tensor rand(DataType::dt_float, {batch_size}, device, AllocHint::temporary);
    gpurandGenerator_t generator = instruction.runtime.gpurand_generator();
    check_error(gpurandSetStream(generator, stream));
    check_error(gpurandGenerateUniformDouble(
        generator, static_cast<double*>(rand.data()), batch_size
    ));

    Tensor indices_tmp(DataType::dt_int, {batch_size}, device, AllocHint::temporary);
    Tensor uw_weights_tmp(
        DataType::dt_float, {batch_size}, device, AllocHint::temporary
    );
    launch_kernel(
        kernel_unweight,
        batch_size,
        stream,
        batch_size,
        rand.view<double, 1>(),
        weights.view<double, 1>(),
        max_weight.view<double, 1>(),
        uw_weights_tmp.view<double, 1>(),
        indices_tmp.view<me_int_t, 1>()
    );

    Tensor indices_compacted(
        DataType::dt_int, {batch_size}, device, instruction.output_alloc_hints[0]
    );
    auto ptr_all =
        thrust::device_pointer_cast(static_cast<me_int_t*>(indices_tmp.data()));
    auto ptr_compacted =
        thrust::device_pointer_cast(static_cast<me_int_t*>(indices_compacted.data()));
    auto ptr_compacted_end = thrust::copy_if(
        thrust_par.on(device.stream()),
        ptr_all,
        ptr_all + batch_size,
        ptr_compacted,
        NotMinusOne()
    );

    std::size_t count = ptr_compacted_end - ptr_compacted;
    indices = indices_compacted.slice(0, 0, count);
    uw_weights =
        Tensor(DataType::dt_float, {count}, device, instruction.output_alloc_hints[1]);
    auto ptr_all_weights =
        thrust::device_pointer_cast(static_cast<double*>(uw_weights_tmp.data()));
    auto ptr_uw_weights =
        thrust::device_pointer_cast(static_cast<double*>(uw_weights.data()));
    thrust::gather(
        thrust_par.on(stream),
        ptr_compacted,
        ptr_compacted_end,
        ptr_all_weights,
        ptr_uw_weights
    );
    rand.reset(device);
    indices_tmp.reset(device);
    uw_weights_tmp.reset(device);
    indices_compacted.reset(device);
}

struct Decrement {
    __device__ void operator()(me_int_t& val) { --val; }
};

void histogram_common(
    const AsyncGpuDevice& device,
    std::size_t padded_size,
    std::size_t n_dims,
    std::size_t n_bins,
    Tensor& indices_tmp,
    Tensor& weights_tmp,
    Tensor& counts,
    Tensor& values
) {
    auto policy = thrust_par.on(device.stream());
    Tensor reduce_tmp(
        DataType::dt_float, {n_dims * n_bins}, device, AllocHint::temporary
    );
    auto indices_ptr =
        thrust::device_pointer_cast(static_cast<me_int_t*>(indices_tmp.data()));
    auto weights_ptr =
        thrust::device_pointer_cast(static_cast<double*>(weights_tmp.data()));
    auto counts_ptr =
        thrust::device_pointer_cast(static_cast<me_int_t*>(counts.data()));
    auto values_ptr = thrust::device_pointer_cast(static_cast<double*>(values.data()));
    auto reduce_tmp_ptr =
        thrust::device_pointer_cast(static_cast<double*>(reduce_tmp.data()));

    std::size_t flat_size = padded_size * n_dims;
    thrust::sort_by_key(policy, indices_ptr, indices_ptr + flat_size, weights_ptr);
    thrust::reduce_by_key(
        policy,
        indices_ptr,
        indices_ptr + flat_size,
        thrust::constant_iterator<me_int_t>(1),
        reduce_tmp_ptr,
        counts_ptr
    );
    thrust::for_each_n(policy, counts_ptr, n_dims * n_bins, Decrement{});
    thrust::reduce_by_key(
        policy,
        indices_ptr,
        indices_ptr + flat_size,
        weights_ptr,
        reduce_tmp_ptr,
        values_ptr
    );
    reduce_tmp.reset(device);
}

__global__ void kernel_prepare_vegas_hist(
    std::size_t batch_size,
    std::size_t n_bins,
    GpuTensorView<double, 2, true> input,
    GpuTensorView<double, 1, true> weights_in,
    GpuTensorView<me_int_t, 2, true> indices,
    GpuTensorView<double, 2, true> weights_out
) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= batch_size + n_bins) {
        return;
    }
    std::size_t n_dims = input.size(1);
    double bin_count_f = n_bins;
    double w2 = i < batch_size ? weights_in[i] * weights_in[i] : 0.;
    for (std::size_t j = 0; j < n_dims; ++j) {
        indices[i][j] = j +
            n_dims *
                (i < batch_size ? static_cast<me_int_t>(input[i][j] * bin_count_f)
                                : i - batch_size);
        weights_out[i][j] = w2;
    }
}

void op_vegas_histogram(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto& input = locals[instruction.input_indices[0]];
    auto& weights = locals[instruction.input_indices[1]];
    auto& values = locals[instruction.output_indices[0]];
    auto& counts = locals[instruction.output_indices[1]];

    auto out_shape = instruction.output_shapes[0];
    Sizes shape(out_shape.size() + 1);
    shape[0] = 1;
    std::copy(out_shape.begin(), out_shape.end(), shape.begin() + 1);
    values =
        Tensor(DataType::dt_float, shape, device, instruction.output_alloc_hints[0]);
    counts = Tensor(DataType::dt_int, shape, device, instruction.output_alloc_hints[1]);

    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    std::size_t n_dims = input.size(1);
    std::size_t n_bins = values.size(2);

    std::size_t padded_size = batch_size + n_bins;
    Tensor indices_tmp(
        DataType::dt_int, {padded_size, n_dims}, device, AllocHint::temporary
    );
    Tensor weights_tmp(
        DataType::dt_float, {padded_size, n_dims}, device, AllocHint::temporary
    );
    launch_kernel(
        kernel_prepare_vegas_hist,
        padded_size,
        device.stream(),
        batch_size,
        n_bins,
        input.view<double, 2>(),
        weights.view<double, 1>(),
        indices_tmp.view<me_int_t, 2>(),
        weights_tmp.view<double, 2>()
    );
    histogram_common(
        device, padded_size, n_dims, n_bins, indices_tmp, weights_tmp, counts, values
    );
    indices_tmp.reset(device);
    weights_tmp.reset(device);
}

__global__ void kernel_prepare_discrete_hist(
    std::size_t batch_size,
    std::size_t n_opts,
    GpuTensorView<me_int_t, 1, true> input,
    GpuTensorView<double, 1, true> weights_in,
    GpuTensorView<me_int_t, 1, true> indices,
    GpuTensorView<double, 1, true> weights_out
) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= batch_size + n_opts) {
        return;
    }
    indices[i] = i < batch_size ? input[i] : i - batch_size;
    weights_out[i] = i < batch_size ? weights_in[i] : 0.;
}

void op_discrete_histogram(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto& input = locals[instruction.input_indices[0]];
    auto& weights = locals[instruction.input_indices[1]];
    auto& values = locals[instruction.output_indices[0]];
    auto& counts = locals[instruction.output_indices[1]];

    auto out_shape = instruction.output_shapes[0];
    Sizes shape(out_shape.size() + 1);
    shape[0] = 1;
    std::copy(out_shape.begin(), out_shape.end(), shape.begin() + 1);
    values =
        Tensor(DataType::dt_float, shape, device, instruction.output_alloc_hints[0]);
    counts = Tensor(DataType::dt_int, shape, device, instruction.output_alloc_hints[1]);

    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    std::size_t n_opts = values.size(1);

    std::size_t padded_size = batch_size + n_opts;
    Tensor indices_tmp(DataType::dt_int, {padded_size}, device, AllocHint::temporary);
    Tensor weights_tmp(DataType::dt_float, {padded_size}, device, AllocHint::temporary);
    launch_kernel(
        kernel_prepare_discrete_hist,
        padded_size,
        device.stream(),
        batch_size,
        n_opts,
        input.view<me_int_t, 1>(),
        weights.view<double, 1>(),
        indices_tmp.view<me_int_t, 1>(),
        weights_tmp.view<double, 1>()
    );
    histogram_common(
        device, padded_size, 1, n_opts, indices_tmp, weights_tmp, counts, values
    );
    indices_tmp.reset(device);
    weights_tmp.reset(device);
}

__global__ void kernel_prepare_hist(
    std::size_t batch_size,
    std::size_t n_bins,
    GpuTensorView<double, 1, true> input,
    GpuTensorView<double, 1, true> min,
    GpuTensorView<double, 1, true> max,
    GpuTensorView<double, 1, true> weights_in,
    GpuTensorView<me_int_t, 1, true> indices,
    GpuTensorView<double, 1, true> weights_out,
    GpuTensorView<double, 1, true> square_weights_out
) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= batch_size + n_bins + 2) {
        return;
    }
    double bin_count_f = n_bins;
    double w;
    me_int_t index;
    if (i < batch_size) {
        w = weights_in[i];
        me_int_t index_rounded = (input[i] - min[i]) / (max[i] - min[i]) * bin_count_f;
        if (index_rounded < 0) {
            index = 0;
        } else if (index_rounded >= n_bins) {
            index = n_bins + 1;
        } else {
            index = index_rounded + 1;
        }
    } else {
        w = 0.;
        index = i - batch_size;
    }
    indices[i] = index;
    weights_out[i] = w;
    square_weights_out[i] = w * w;
}

void op_histogram(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    Tensor input = locals[instruction.input_indices[0]].contiguous(device);
    auto& weights = locals[instruction.input_indices[1]];
    auto& hist_min = locals[instruction.input_indices[2]];
    auto& hist_max = locals[instruction.input_indices[3]];
    auto& values = locals[instruction.output_indices[0]];
    auto& square_values = locals[instruction.output_indices[1]];

    auto out_shape = instruction.output_shapes[0];
    Sizes shape(out_shape.size() + 1);
    shape[0] = 1;
    std::copy(out_shape.begin(), out_shape.end(), shape.begin() + 1);
    values =
        Tensor(DataType::dt_float, shape, device, instruction.output_alloc_hints[0]);
    square_values =
        Tensor(DataType::dt_float, shape, device, instruction.output_alloc_hints[1]);

    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    std::size_t n_bins = values.size(1) - 2;
    std::size_t padded_size = batch_size + n_bins + 2;
    Tensor indices_tmp(DataType::dt_int, {padded_size}, device, AllocHint::temporary);
    Tensor weights_tmp(DataType::dt_float, {padded_size}, device, AllocHint::temporary);
    Tensor square_weights_tmp(
        DataType::dt_float, {padded_size}, device, AllocHint::temporary
    );
    Tensor reduce_tmp(DataType::dt_float, {n_bins}, device, AllocHint::temporary);

    launch_kernel(
        kernel_prepare_hist,
        padded_size,
        device.stream(),
        batch_size,
        n_bins,
        input.view<double, 1>(),
        hist_min.view<double, 1>(),
        hist_max.view<double, 1>(),
        weights.view<double, 1>(),
        indices_tmp.view<me_int_t, 1>(),
        weights_tmp.view<double, 1>(),
        square_weights_tmp.view<double, 1>()
    );

    auto policy = thrust_par.on(device.stream());
    auto indices_ptr =
        thrust::device_pointer_cast(static_cast<me_int_t*>(indices_tmp.data()));
    auto reduce_tmp_ptr =
        thrust::device_pointer_cast(static_cast<double*>(reduce_tmp.data()));

    auto weights_ptr =
        thrust::device_pointer_cast(static_cast<double*>(weights_tmp.data()));
    auto values_ptr = thrust::device_pointer_cast(static_cast<double*>(values.data()));
    thrust::sort_by_key(policy, indices_ptr, indices_ptr + padded_size, weights_ptr);
    thrust::reduce_by_key(
        policy,
        indices_ptr,
        indices_ptr + padded_size,
        weights_ptr,
        reduce_tmp_ptr,
        values_ptr
    );

    auto square_weights_ptr =
        thrust::device_pointer_cast(static_cast<double*>(square_weights_tmp.data()));
    auto square_values_ptr =
        thrust::device_pointer_cast(static_cast<double*>(square_values.data()));
    thrust::sort_by_key(
        policy, indices_ptr, indices_ptr + padded_size, square_weights_ptr
    );
    thrust::reduce_by_key(
        policy,
        indices_ptr,
        indices_ptr + padded_size,
        square_weights_ptr,
        reduce_tmp_ptr,
        square_values_ptr
    );

    input.reset(device);
    reduce_tmp.reset(device);
    indices_tmp.reset(device);
    weights_tmp.reset(device);
    square_weights_tmp.reset(device);
}

class SyncTracker {
public:
    SyncTracker(std::size_t stream_count) :
        _stream_count(stream_count), _sync_matrix(stream_count * stream_count) {
        reset();
    }

    bool is_in_sync_with(std::size_t this_stream, std::size_t other_stream) const {
        return _sync_matrix.at(this_stream * _stream_count + other_stream);
    }
    void desynchronize(std::size_t this_stream) {
        for (std::size_t other_stream = 0; other_stream < _stream_count;
             ++other_stream) {
            if (this_stream != other_stream) {
                _sync_matrix.at(other_stream * _stream_count + this_stream) = false;
            }
        }
    }
    void synchronize(std::size_t this_stream, std::size_t other_stream) {
        for (std::size_t i = 0; i < _stream_count; ++i) {
            if (is_in_sync_with(other_stream, i)) {
                _sync_matrix.at(this_stream * _stream_count + i) = true;
            }
        }
    }
    void reset() {
        for (std::size_t i = 0; i < _stream_count; ++i) {
            for (std::size_t j = 0; j < _stream_count; ++j) {
                _sync_matrix.at(i * _stream_count + j) = i == j;
            }
        }
    }

private:
    std::size_t _stream_count;
    std::vector<bool> _sync_matrix;
};

} // namespace

GpuRuntime::GpuRuntime(const Function& function_arg, ContextPtr context) :
    _context(context),
    _input_count(function_arg.inputs().size()),
    _gpublas_handle(
        context->thread_pool(),
        []() {
            gpublasHandle_t handle;
            check_error(gpublasCreate(&handle));
            return handle;
        },
        [](gpublasHandle_t handle) { check_error(gpublasDestroy(handle)); }
    ),
    _gpurand_generator(
        context->thread_pool(),
        []() {
            gpurandGenerator_t handle;
            check_error(gpurandCreateGenerator(&handle, GPURAND_RNG_PSEUDO_DEFAULT));
            std::random_device rand_dev;
            check_error(gpurandSetPseudoRandomGeneratorSeed(handle, rand_dev()));
            return handle;
        },
        [](gpurandGenerator_t handle) { check_error(gpurandDestroyGenerator(handle)); }
    ),
    _prev_caches(context->thread_pool(), []() { return TensorVec{}; }),
    _prev_caches_backward(context->thread_pool(), []() { return TensorVec{}; }) {
    if (context->device()->device_type() != GpuDevice::gpu_device_type) {
        throw std::runtime_error("Context has incompatible device");
    }
    auto& gpu_device = *static_cast<const GpuDevice*>(_context->device());
    gpu_device.activate();

    // cudaMemPool_t pool;
    // check_error(cudaDeviceGetMemPool(&pool, 0));
    // uint64_t thresh = UINT64_MAX;
    // check_error(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
    // &thresh));

    Function function = sort_breadth_first(function_arg);

    _locals_init.resize(function.locals().size());
    _requires_grad_init.resize(function.locals().size());
    LastUseOfLocals last_use(function);
    InstructionDependencies dependencies(function);

    std::size_t stream_count = 0, event_count = 0, backward_event_count = 0;
    for (auto& instr : function.instructions()) {
        if (instr.stream_index >= stream_count) {
            stream_count = instr.stream_index + 1;
        }
    }
    SyncTracker sync_tracker(stream_count);
    std::vector<int> local_source_streams(function.locals().size(), -1);
    SizeVec last_stream_instrs(stream_count);
    nested_vector2<std::size_t> backward_wait_events(function.instructions().size());
    std::vector<int> backward_record_events(function.instructions().size(), -1);

    auto update_sync_backward =
        [&](std::size_t local_index, std::size_t stream_index, SizeVec& wait_events) {
            int source_stream = local_source_streams.at(local_index);
            if (source_stream == -1) {
                return;
            }
            if (!sync_tracker.is_in_sync_with(stream_index, source_stream)) {
                int& event =
                    backward_record_events.at(last_stream_instrs.at(source_stream));
                if (event == -1) {
                    event = backward_event_count;
                    ++backward_event_count;
                }
                wait_events.push_back(event);
                sync_tracker.synchronize(stream_index, source_stream);
            }
        };

    for (std::size_t instr_index = 0;
         auto [instr, bw_wait_events] :
         zip(std::views::reverse(function.instructions()),
             std::views::reverse(backward_wait_events))) {
        for (auto& out : instr.outputs) {
            update_sync_backward(out.local_index, instr.stream_index, bw_wait_events);
        }
        for (auto& in : instr.inputs) {
            local_source_streams.at(in.local_index) = instr.stream_index;
        }
        last_stream_instrs.at(instr.stream_index) = instr_index;
        sync_tracker.desynchronize(instr.stream_index);
        ++instr_index;
    }
    for (auto& in : function.inputs()) {
        update_sync_backward(in.local_index, 0, _backward_wait_events);
    }

    sync_tracker.reset();
    std::fill(local_source_streams.begin(), local_source_streams.end(), -1);
    nested_vector2<std::size_t> local_consumer_streams(function.locals().size());

    auto update_sync =
        [&](std::size_t local_index, std::size_t stream_index, SizeVec& wait_events) {
            int source_stream = local_source_streams.at(local_index);
            if (source_stream == -1) {
                return;
            }
            auto& consumer_streams = local_consumer_streams.at(local_index);
            if (std::find(
                    consumer_streams.begin(), consumer_streams.end(), stream_index
                ) == consumer_streams.end()) {
                consumer_streams.push_back(stream_index);
            }
            if (!sync_tracker.is_in_sync_with(stream_index, source_stream)) {
                int& event =
                    _instructions.at(last_stream_instrs.at(source_stream)).record_event;
                if (event == -1) {
                    event = event_count;
                    ++event_count;
                }
                wait_events.push_back(event);
                sync_tracker.synchronize(stream_index, source_stream);
            }
        };

    std::vector<bool> is_input(function.locals().size());
    std::vector<bool> is_output(function.locals().size());
    std::vector<bool> is_global(function.locals().size());
    for (auto& input : function.inputs()) {
        is_input.at(input.local_index) = true;
    }
    for (auto& output : function.outputs()) {
        is_output.at(output.local_index) = true;
    }
    for (auto& [name, global] : function.globals()) {
        is_global.at(global.local_index) = true;
    }

    SizeVec free_queue;
    for (std::size_t instr_index = 0;
         auto [instr, bw_wait_events, bw_record_event] :
         zip(function.instructions(), backward_wait_events, backward_record_events)) {
        SizeVec input_indices, wait_events;
        std::size_t batch_size_index = instr.inputs.at(0).local_index;
        std::vector<AllocHint> input_grad_alloc_hints;
        for (auto& in : instr.inputs) {
            input_indices.push_back(in.local_index);
            if (is_input.at(in.local_index)) {
                input_grad_alloc_hints.push_back(AllocHint::input_grad);
            } else if (is_global.at(in.local_index)) {
                input_grad_alloc_hints.push_back(AllocHint::global_grad);
            } else {
                input_grad_alloc_hints.push_back(AllocHint::local_grad);
            }

            if (in.type.batch_size != BatchSize::one) {
                batch_size_index = in.local_index;
            }
            update_sync(in.local_index, instr.stream_index, wait_events);
        }
        SizeVec output_indices;
        std::vector<AllocHint> output_alloc_hints;
        std::vector<DataType> output_dtypes;
        std::vector<SizeVec> output_shapes;
        for (auto& out : instr.outputs) {
            output_indices.push_back(out.local_index);
            output_dtypes.push_back(out.type.dtype);
            output_shapes.push_back({out.type.shape.begin(), out.type.shape.end()});
            if (is_output.at(out.local_index)) {
                output_alloc_hints.push_back(AllocHint::output);
            } else {
                output_alloc_hints.push_back(AllocHint::local);
            }
            local_source_streams.at(out.local_index) = instr.stream_index;
        }

        sync_tracker.desynchronize(instr.stream_index);
        last_stream_instrs.at(instr.stream_index) = _instructions.size();
        _instructions.push_back({
            instr.instruction->opcode(),
            input_indices,
            input_grad_alloc_hints,
            output_indices,
            output_alloc_hints,
            output_dtypes,
            output_shapes,
            batch_size_index,
            *this,
            instr.instruction->differentiable(),
            instr.stream_index,
            wait_events,
            -1,
            bw_wait_events,
            bw_record_event,
        });

        SizeVec locals_to_free = last_use.local_indices(instr_index);
        free_queue.insert(
            free_queue.end(), locals_to_free.begin(), locals_to_free.end()
        );
        free_queue.erase(
            std::remove_if(
                free_queue.begin(),
                free_queue.end(),
                [&](std::size_t local_index) {
                    for (std::size_t consumer_stream :
                         local_consumer_streams.at(local_index)) {
                        if (!sync_tracker.is_in_sync_with(
                                instr.stream_index, consumer_stream
                            )) {
                            return false;
                        }
                    }
                    _instructions.push_back(
                        {-1,
                         {local_index},
                         {},
                         {},
                         {},
                         {},
                         {},
                         0,
                         *this,
                         false,
                         instr.stream_index,
                         {},
                         -1,
                         {},
                         -1}
                    );
                    return true;
                }
            ),
            free_queue.end()
        );

        ++instr_index;
    }

    _grad_global_total_size = 0;
    for (auto& [name, value] : function.globals()) {
        Tensor global = context->global(name);
        auto& global_shape = value.type.shape;
        Sizes full_shape(global_shape.size() + 1);
        full_shape[0] = 1;
        std::copy(global_shape.begin(), global_shape.end(), full_shape.begin() + 1);
        if (value.type.dtype != global.dtype() || full_shape != global.shape()) {
            throw std::invalid_argument(
                std::format("Global {} has wrong dtype or shape", name)
            );
        }
        _locals_init.at(value.local_index) = global;
        if (context->global_requires_grad(name)) {
            _requires_grad_init.at(value.local_index) = true;
            _grad_global_indices.push_back(value.local_index);
            _grad_global_shapes.push_back(global.shape());
            _grad_global_total_size += global.shape().product();
        }
    }

    for (auto& local : function.locals()) {
        std::visit(
            Overloaded{
                [&](auto val) {
                    Tensor tensor(val, static_cast<DevicePtr>(&gpu_device));
                    _locals_init[local.local_index] = tensor;
                },
                [](std::monostate val) {}
            },
            local.literal_value
        );
    }

    for (auto& out : function.outputs()) {
        _output_indices.push_back(out.local_index);
        update_sync(out.local_index, 0, _wait_events);
    }

    _streams = ThreadResource<std::vector<gpuStream_t>>(
        context->thread_pool(),
        [stream_count]() {
            std::vector<gpuStream_t> streams(stream_count);
            for (auto& item : streams) {
                check_error(gpuStreamCreate(&item));
            }
            return streams;
        },
        [](auto& streams) {
            for (auto item : streams) {
                check_error(gpuStreamDestroy(item));
            }
        }
    );
    std::size_t max_event_count = std::max(event_count, backward_event_count);
    _events = ThreadResource<std::vector<gpuEvent_t>>(
        context->thread_pool(),
        [max_event_count]() {
            std::vector<gpuEvent_t> events(max_event_count);
            for (auto& item : events) {
                check_error(gpuEventCreate(&item));
            }
            return events;
        },
        [](auto& events) {
            for (auto item : events) {
                check_error(gpuEventDestroy(item));
            }
        }
    );
}

TensorVec GpuRuntime::run(const TensorVec& inputs) {
    auto& gpu_device = *static_cast<const GpuDevice*>(_context->device());
    auto& streams = _streams.get();
    auto& events = _events.get();
    gpu_device.activate();
    auto locals = _locals_init;
    std::copy(inputs.begin(), inputs.end(), locals.begin());
    gpuStream_t main_stream = streams.at(0);
    MemPool mem_pool(gpu_device, load_pool_size_cache(false), main_stream);

    for (auto& instr : _instructions) {
        gpuStream_t stream = streams.at(instr.stream);
        AsyncGpuDevice device(gpu_device, stream, instr.stream, &mem_pool);
        for (auto event : instr.wait_events) {
            check_error(gpuStreamWaitEvent(stream, events.at(event)));
        }
        switch (instr.opcode) {
        case -1: // free memory
            locals[instr.input_indices[0]].reset(device);
            break;
#include "runtime_mixin.inc"
        }
        if (instr.record_event != -1) {
            check_error(gpuEventRecord(events.at(instr.record_event), stream));
        }
    }
    for (auto event : _wait_events) {
        check_error(gpuStreamWaitEvent(main_stream, events.at(event)));
    }
    update_pool_size_cache(mem_pool.total_sizes(), false);
    update_cached_tensors(mem_pool.reset(main_stream), false);
    TensorVec outputs;
    for (auto index : _output_indices) {
        outputs.push_back(locals[index]);
    }
    check_error(gpuStreamSynchronize(main_stream));
    return outputs;
}

std::tuple<TensorVec, TensorVec, std::vector<bool>> GpuRuntime::run_with_grad(
    const TensorVec& inputs, const std::vector<bool>& input_requires_grad
) {
    auto& gpu_device = *static_cast<const GpuDevice*>(_context->device());
    auto& streams = _streams.get();
    auto& events = _events.get();
    gpu_device.activate();
    auto locals = _locals_init;
    auto requires_grad = _requires_grad_init;
    std::vector<bool> store_local(locals.size());
    std::vector<bool> eval_grad(_instructions.size());
    std::copy(inputs.begin(), inputs.end(), locals.begin());
    std::copy(
        input_requires_grad.begin(), input_requires_grad.end(), requires_grad.begin()
    );
    gpuStream_t main_stream = streams.at(0);
    MemPool mem_pool(gpu_device, load_pool_size_cache(false), main_stream);

    for (auto [instr, instr_eval_grad] : zip(_instructions, eval_grad)) {
        gpuStream_t stream = streams.at(instr.stream);
        AsyncGpuDevice device(gpu_device, stream, instr.stream, &mem_pool);
        if (instr.differentiable) {
            for (auto input_index : instr.input_indices) {
                if (requires_grad[input_index]) {
                    instr_eval_grad = true;
                    break;
                }
            }
            if (instr_eval_grad) {
                // TODO: only store necessary
                for (auto input_index : instr.input_indices) {
                    store_local[input_index] = true;
                }
                for (auto output_index : instr.output_indices) {
                    store_local[output_index] = true;
                    requires_grad[output_index] = true;
                }
            }
        }
        for (auto event : instr.wait_events) {
            check_error(gpuStreamWaitEvent(stream, events.at(event)));
        }
        switch (instr.opcode) {
        case -1: { // free memory
            auto input_index = instr.input_indices[0];
            if (!store_local[input_index]) {
                locals[input_index].reset(device);
            }
            break;
        }
#include "runtime_mixin.inc"
        }
        if (instr.record_event != -1) {
            check_error(gpuEventRecord(events.at(instr.record_event), stream));
        }
    }
    for (auto event : _wait_events) {
        check_error(gpuStreamWaitEvent(main_stream, events.at(event)));
    }
    update_pool_size_cache(mem_pool.total_sizes(), false);
    update_cached_tensors(mem_pool.reset(main_stream), false);
    TensorVec outputs;
    for (auto index : _output_indices) {
        outputs.push_back(locals[index]);
    }
    check_error(gpuStreamSynchronize(main_stream));
    return {outputs, locals, eval_grad};
}

std::pair<TensorVec, TensorVec> GpuRuntime::run_backward(
    const TensorVec& output_grads,
    const TensorVec& stored_locals,
    const std::vector<bool>& eval_grad,
    bool return_contiguous_grads
) {
    auto& gpu_device = *static_cast<const GpuDevice*>(_context->device());
    auto& streams = _streams.get();
    auto& events = _events.get();
    gpu_device.activate();
    TensorVec local_grads(stored_locals.size());
    TensorVec locals(stored_locals);
    for (auto [index, grad] : zip(_output_indices, output_grads)) {
        local_grads[index] = grad;
    }
    gpuStream_t main_stream = streams.at(0);
    MemPool mem_pool(gpu_device, load_pool_size_cache(true), main_stream);

    AsyncGpuDevice init_device(gpu_device, main_stream, 0, &mem_pool);
    Tensor all_global_grads(
        DataType::dt_float,
        {_grad_global_total_size},
        init_device,
        AllocHint::global_grad
    );
    // all_global_grads.zero(init_device);
    TensorVec global_grads = all_global_grads.split_and_reshape(_grad_global_shapes);
    for (auto [index, grad] : zip(_grad_global_indices, global_grads)) {
        local_grads[index] = grad;
    }

    for (auto [instr, instr_eval_grad] :
         zip(std::views::reverse(_instructions), std::views::reverse(eval_grad))) {
        /*gpuStream_t stream = streams.at(instr.stream);
        for (auto event : instr.backward_wait_events) {
            check_error(gpuStreamWaitEvent(stream, events.at(event)));
        }*/
        if (instr_eval_grad) {
            // AsyncGpuDevice device(gpu_device, stream, &mem_pool);
            AsyncGpuDevice device(gpu_device, main_stream, 0, &mem_pool);
            // static_cast<const void*>(&mem_pool));
            for (auto [output_index, output_dtype] :
                 zip(instr.output_indices, instr.output_dtypes)) {
                auto& grad = local_grads[output_index];
                if (!grad && output_dtype == DataType::dt_float) {
                    grad = Tensor(
                        DataType::dt_float,
                        locals[output_index].shape(),
                        device,
                        AllocHint::local_grad
                    );
                    // grad.zero(device);
                }
            }
            switch (instr.opcode) {
#include "runtime_backward_mixin.inc"
            }
        }
        /*if (instr.backward_record_event != -1) {
            check_error(gpuEventRecord(events.at(instr.backward_record_event), stream));
        }*/
    }
    /*for (auto event : _backward_wait_events) {
        check_error(gpuStreamWaitEvent(main_stream, events.at(event)));
    }*/
    update_pool_size_cache(mem_pool.total_sizes(), true);
    update_cached_tensors(mem_pool.reset(main_stream), true);
    check_error(gpuStreamSynchronize(main_stream));
    return {
        {local_grads.begin(), local_grads.begin() + _input_count},
        return_contiguous_grads ? TensorVec{all_global_grads} : global_grads
    };
}

std::vector<std::tuple<std::size_t, std::size_t, Tensor, bool>>
GpuRuntime::load_pool_size_cache(bool backward) {
    auto cache = backward ? _pool_size_cache_backward.load() : _pool_size_cache.load();
    std::vector<std::tuple<std::size_t, std::size_t, Tensor, bool>> ret;
    if (cache) {
        auto& thread_prev_caches =
            backward ? _prev_caches_backward.get() : _prev_caches.get();
        for (auto [pool_index, size] : *cache) {
            Tensor new_cache;
            if (pool_index < thread_prev_caches.size()) {
                Tensor& prev_cache = thread_prev_caches.at(pool_index);
                if (prev_cache && prev_cache.is_only_reference()) {
                    new_cache = prev_cache;
                }
            }
            ret.push_back(
                {pool_index,
                 size,
                 new_cache,
                 needs_zero_init(static_cast<AllocHint>(pool_index + 1))}
            );
        }
    }
    return ret;
}

void GpuRuntime::update_pool_size_cache(
    const std::vector<std::pair<std::size_t, std::size_t>>& total_sizes, bool backward
) {
    auto cache = backward ? _pool_size_cache_backward.load() : _pool_size_cache.load();
    auto new_cache = cache
        ? std::make_shared<std::unordered_map<std::size_t, std::size_t>>(*cache)
        : std::make_shared<std::unordered_map<std::size_t, std::size_t>>();
    for (auto [pool_index, size] : total_sizes) {
        auto& cache_size = (*new_cache)[pool_index];
        if (size > cache_size) {
            // if the cache needs to be resized, add some padding to prevent frequent
            // resizing
            cache_size = size * 4 / 3;
        }
    }
    if (backward) {
        _pool_size_cache_backward.store(new_cache);
    } else {
        _pool_size_cache.store(new_cache);
    }
}

void GpuRuntime::update_cached_tensors(
    const std::vector<std::pair<std::size_t, Tensor>>& tensors, bool backward
) {
    auto& thread_prev_caches =
        backward ? _prev_caches_backward.get() : _prev_caches.get();
    for (auto& [pool_index, tensor] : tensors) {
        if (pool_index >= thread_prev_caches.size()) {
            thread_prev_caches.resize(pool_index + 1);
        }
        thread_prev_caches.at(pool_index) = tensor;
    }
}

extern "C" Runtime*
build_runtime(const Function& function, ContextPtr context, bool concurrent) {
    return new GpuRuntime(function, context);
}
