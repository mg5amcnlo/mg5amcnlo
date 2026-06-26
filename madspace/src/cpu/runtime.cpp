#include "runtime.hpp"

#include <algorithm>
#include <random>
#include <ranges>
#include <tuple>

#include "../kernels/kernels.hpp"
#include "../kernels/operations.hpp"
#include "device.hpp"
#include "madspace/compgraphs/optimizer.hpp"
#include "madspace/util.hpp"

extern "C" void dgemm_(
    char* transa,
    char* transb,
    int* m,
    int* n,
    int* k,
    double* alpha,
    double* a,
    int* lda,
    double* b,
    int* ldb,
    double* beta,
    double* c,
    int* ldc
);

extern "C" void dgemv_(
    char* trans,
    int* m,
    int* n,
    double* alpha,
    double* a,
    int* lda,
    double* x,
    int* incx,
    double* beta,
    double* y,
    int* incy
);

using namespace madspace;
using namespace madspace::cpu;
using namespace madspace::kernels;

namespace {

template <typename D>
void op_matrix_element(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const D& device
) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    std::size_t me_index = locals[instruction.input_indices[0]].index_value();
    std::size_t input_count = locals[instruction.input_indices[1]].index_value();
    std::size_t output_count = locals[instruction.input_indices[2]].index_value();
    TensorVec contiguous_inputs(input_count);
    std::vector<UmamiInputKey> input_keys(input_count);
    std::vector<UmamiOutputKey> output_keys(output_count);
    std::vector<void*> input_ptrs(input_count), output_ptrs(output_count);
    for (std::size_t i = 0; i < input_count; ++i) {
        input_keys[i] = static_cast<UmamiInputKey>(
            locals[instruction.input_indices[3 + 2 * i]].index_value()
        );
        contiguous_inputs[i] =
            locals[instruction.input_indices[3 + 2 * i + 1]].contiguous(
                batch_size, device
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
        output = Tensor(instruction.output_dtypes[i], shape, device);
        output_ptrs[i] = output.data();
        if (me_index == 0xBADCAFE) {
            // flat dummy matrix element for testing purposes
            // implemented at LUMI hackathon where the coffee was indeed terrible
            switch (output_keys[i]) {
            case UMAMI_OUT_MATRIX_ELEMENT:
                device.submit([=]() mutable {
                    double* ptr = static_cast<double*>(output_ptrs[i]);
                    std::fill(ptr, ptr + batch_size, 1.);
                });
                break;
            case UMAMI_OUT_DIAGRAM_AMP2:
                device.submit([=]() mutable {
                    double* ptr = static_cast<double*>(output_ptrs[i]);
                    std::fill(ptr, ptr + batch_size * shape[1], 1. / shape[1]);
                });
                break;
            default:
                output.zero(device);
                break;
            }
        }
    }
    if (me_index == 0xBADCAFE || batch_size == 0) {
        return;
    }
    auto& matrix_element = instruction.runtime.context().matrix_element(me_index);
    if (matrix_element.device_type() != DeviceType::cpu) {
        throw std::runtime_error("Matrix element has incompatible device");
    }
    device.sync_barrier();

    device.foreach (
        batch_size,
        [contiguous_inputs,
         input_count,
         input_keys,
         input_ptrs,
         output_count,
         output_keys,
         output_ptrs,
         &matrix_element,
         batch_size](std::size_t count, std::size_t offset) {
            matrix_element.call(
                matrix_element.process_instance(),
                count,
                batch_size,
                offset,
                input_count,
                input_keys.data(),
                input_ptrs.data(),
                output_count,
                output_keys.data(),
                output_ptrs.data()
            );
        }
    );
}

template <typename D>
void op_matmul(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const D& device
) {
    auto input = locals[instruction.input_indices[0]].contiguous(device);
    auto weight = locals[instruction.input_indices[1]].contiguous(device);
    auto bias_orig = locals[instruction.input_indices[2]];
    auto bias = bias_orig.contiguous(device);
    auto& output = locals[instruction.output_indices[0]];
    std::size_t batch_size = input.size(0);
    std::size_t dims_in = input.size(1);
    std::size_t dims_out = weight.size(1);
    output = Tensor(DataType::dt_float, {batch_size, dims_out}, device);
    output.copy_from(bias_orig, device);
    if (batch_size == 0) {
        return;
    }

    device.sync_barrier();
    device.submit([=]() mutable {
        char transa = 'N', transb = 'T';
        int m = batch_size, n = dims_out, k = dims_in;
        double alpha = 1., beta = 1.;
        int lda = batch_size, ldb = dims_out, ldc = batch_size;
        double* a = static_cast<double*>(input.data());
        double* b = static_cast<double*>(weight.data());
        double* c = static_cast<double*>(output.data());
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    });
}

template <typename D>
void backward_op_matmul(
    const CpuRuntime::Instruction& instruction,
    TensorVec& locals,
    TensorVec& local_grads,
    const D& device
) {
    auto input = locals[instruction.input_indices[0]].contiguous(device);
    auto weight = locals[instruction.input_indices[1]].contiguous(device);
    auto output_grad = local_grads[instruction.output_indices[0]].contiguous(device);
    auto& input_grad = local_grads[instruction.input_indices[0]];
    auto& weight_grad = local_grads[instruction.input_indices[1]];
    auto& bias_grad = local_grads[instruction.input_indices[2]];
    std::size_t batch_size = input.size(0);
    std::size_t dims_in = input.size(1);
    std::size_t dims_out = weight.size(1);

    if (!input_grad) {
        input_grad =
            Tensor(DataType::dt_float, input.shape(), device, AllocHint::local_grad);
        // input_grad.zero(device);
    }
    if (!weight_grad) {
        weight_grad =
            Tensor(DataType::dt_float, weight.shape(), device, AllocHint::local_grad);
        // weight_grad.zero(device);
    }
    if (!bias_grad) {
        bias_grad =
            Tensor(DataType::dt_float, {1, dims_out}, device, AllocHint::local_grad);
        // bias_grad.zero(device);
    }
    if (batch_size == 0) {
        return;
    }
    device.sync_barrier();

    // compute input_grad += output_grad * weight
    device.submit([=]() mutable {
        char transa = 'N', transb = 'N';
        int m = batch_size, n = dims_in, k = dims_out;
        double alpha = 1., beta = 1.;
        int lda = batch_size, ldb = dims_out, ldc = batch_size;
        double* a = static_cast<double*>(output_grad.data());
        ;
        double* b = static_cast<double*>(weight.data());
        double* c = static_cast<double*>(input_grad.data());
        ;
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    });

    // compute weight_grad += output_grad.T * input
    device.submit([=]() mutable {
        char transa = 'T', transb = 'N';
        int m = dims_out, n = dims_in, k = batch_size;
        double alpha = 1., beta = 1.;
        int lda = batch_size, ldb = batch_size, ldc = dims_out;
        double* a = static_cast<double*>(output_grad.data());
        double* b = static_cast<double*>(input.data());
        double* c = static_cast<double*>(weight_grad.data());
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    });

    // compute bias_grad += sum_i output_grad_ij
    device.submit([=]() mutable {
        // TODO: we should probably do this differently...
        std::vector<double> ones(batch_size, 1.);
        char trans = 'T';
        int m = batch_size, n = dims_out;
        double alpha = 1., beta = 1.;
        int lda = batch_size, incx = 1, incy = 1;
        double* a = static_cast<double*>(output_grad.data());
        double* x = ones.data();
        double* y = static_cast<double*>(bias_grad.data());
        dgemv_(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    });
}

template <typename D>
void op_nonzero(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const D& device
) {
    // TODO: not parallelized for now
    auto& input = locals[instruction.input_indices[0]];
    auto batch_size = input.size(0);
    auto& output = locals[instruction.output_indices[0]];
    Tensor output_tmp(DataType::dt_int, {batch_size});
    auto input_view_flat = input.flat_view<double, 1>(0);
    auto output_view_flat = output_tmp.flat_view<me_int_t, 1>(0);
    device.submit(
        [&output, output_tmp, batch_size, input_view_flat, output_view_flat]() mutable {
            TensorView<double, 1> input_view(input_view_flat);
            TensorView<me_int_t, 1> output_view(output_view_flat);
            std::size_t count = 0;
            for (std::size_t i = 0; i < batch_size; ++i) {
                if (input_view[i] != 0.) {
                    output_view[count] = i;
                    ++count;
                }
            }
            output = output_tmp.slice(0, 0, count);
        }
    );
}

template <auto kernel, int dim, typename T, typename D>
void batch_gather_impl_body(
    Tensor& indices, Tensor& values, Tensor& selection, const D& device
) {
    device.foreach (
        indices.size(0),
        [&](std::size_t count, std::size_t offset) {
            auto indices_view = indices.view<me_int_t, 1>();
            auto values_view = values.view<T, dim>();
            auto selection_view = selection.view<T, dim>();
            for (std::size_t i = 0; i < count; ++i) {
                nested_for_nobatch<kernel, dim - 1>(
                    values_view[indices_view[i]], selection_view[i]
                );
            }
        },
        true
    );
}

template <int dim, typename D>
void batch_gather_impl(
    Tensor& indices, Tensor& values, Tensor& selection, const D& device
) {
    Sizes out_shape = values.shape();
    out_shape[0] = indices.size(0);
    if (values.dtype() == DataType::dt_float) {
        selection = Tensor(DataType::dt_float, out_shape, device);
        batch_gather_impl_body<kernel_copy<CpuTypes>, dim, double>(
            indices, values, selection, device
        );
    } else if (values.dtype() == DataType::dt_int) {
        selection = Tensor(DataType::dt_int, out_shape, device);
        batch_gather_impl_body<kernel_copy_int<CpuTypes>, dim, me_int_t>(
            indices, values, selection, device
        );
    } else {
        throw std::runtime_error("invalid dtype in batch_gather");
    }
}

template <typename D>
void op_batch_gather(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const D& device
) {
    auto& indices = locals[instruction.input_indices[0]];
    auto& values = locals[instruction.input_indices[1]];
    auto& selection = locals[instruction.output_indices[0]];
    switch (values.shape().size()) {
    case 1:
        batch_gather_impl<1>(indices, values, selection, device);
        break;
    case 2:
        batch_gather_impl<2>(indices, values, selection, device);
        break;
    case 3:
        batch_gather_impl<3>(indices, values, selection, device);
        break;
    case 4:
        batch_gather_impl<4>(indices, values, selection, device);
        break;
    default:
        throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

template <auto kernel, int dim, typename T, typename D>
void batch_scatter_impl_body(
    Tensor& indices, Tensor& source, Tensor& output, const D& device
) {
    device.foreach (
        indices.size(0),
        [&](std::size_t count, std::size_t offset) {
            auto indices_view = indices.view<me_int_t, 1>();
            auto source_view = source.view<T, dim>();
            auto output_view = output.view<T, dim>();
            for (std::size_t i = 0; i < count; ++i) {
                nested_for_nobatch<kernel, dim - 1>(
                    source_view[i], output_view[indices_view[i]]
                );
            }
        },
        true
    );
}

template <int dim, typename D>
void batch_scatter_impl(
    Tensor& indices, Tensor& source, Tensor& output, const D& device
) {
    if (source.dtype() == DataType::dt_float) {
        batch_scatter_impl_body<kernel_copy<CpuTypes>, dim, double>(
            indices, source, output, device
        );
    } else if (source.dtype() == DataType::dt_int) {
        batch_scatter_impl_body<kernel_copy_int<CpuTypes>, dim, me_int_t>(
            indices, source, output, device
        );
    } else {
        throw std::runtime_error("invalid dtype in batch_scatter");
    }
}

template <typename D>
void op_batch_scatter(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const D& device
) {
    auto& indices = locals[instruction.input_indices[0]];
    auto& target = locals[instruction.input_indices[1]];
    auto& source = locals[instruction.input_indices[2]];

    auto& output = locals[instruction.output_indices[0]];
    output = target.copy(device);
    device.sync_barrier();
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

template <typename D>
void batch_reduce_mean_impl(
    const CpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const D& device,
    bool keepdim
) {
    auto& input = locals[instruction.input_indices[0]];
    auto& output = locals[instruction.output_indices[0]];
    std::size_t batch_size = input.size(0);
    output = Tensor(DataType::dt_float, {keepdim ? batch_size : 1}, device);
    device.sync_barrier();

    auto input_view_flat = input.flat_view<double, 1>(0);
    auto output_view_flat = output.flat_view<double, 1>(0);
    device.submit([keepdim, input_view_flat, output_view_flat, batch_size]() mutable {
        TensorView<double, 1> input_view(input_view_flat);
        TensorView<double, 1> output_view(output_view_flat);
        double sum = 0.;
        for (std::size_t i = 0; i < batch_size; ++i) {
            sum += input_view[i];
        }
        if (keepdim) {
            for (std::size_t i = 0; i < batch_size; ++i) {
                output_view[i] = sum / batch_size;
            }
        } else {
            output_view[0] = sum / batch_size;
        }
    });
}

template <typename D>
void batch_reduce_mean_backward_impl(
    const CpuRuntime::Instruction& instruction,
    TensorVec& locals,
    TensorVec& local_grads,
    const D& device,
    bool keepdim
) {
    auto& input = locals[instruction.input_indices[0]];
    auto& input_grad = local_grads[instruction.input_indices[0]];
    auto& output_grad = local_grads[instruction.output_indices[0]];
    if (!input_grad) {
        input_grad =
            Tensor(DataType::dt_float, input.shape(), device, AllocHint::local_grad);
        // input_grad.zero(device);
    }
    device.sync_barrier();

    auto input_grad_view_flat = input_grad.flat_view<double, 1>(0);
    auto output_grad_view_flat = output_grad.flat_view<double, 1>(0);
    std::size_t batch_size = input.size(0);
    device.submit(
        [keepdim, input_grad_view_flat, output_grad_view_flat, batch_size]() mutable {
            TensorView<double, 1> input_grad_view(input_grad_view_flat);
            TensorView<double, 1> output_grad_view(output_grad_view_flat);
            double grad = 0.0;
            if (keepdim) {
                for (std::size_t i = 0; i < batch_size; ++i) {
                    grad += output_grad_view[i];
                }
                grad /= batch_size;
            } else {
                grad = output_grad_view[0] / batch_size;
            }
            for (std::size_t i = 0; i < batch_size; ++i) {
                input_grad_view[i] += grad;
            }
        }
    );
}

template <typename D>
void op_batch_reduce_mean(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const D& device
) {
    batch_reduce_mean_impl(instruction, locals, device, false);
}

template <typename D>
void backward_op_batch_reduce_mean(
    const CpuRuntime::Instruction& instruction,
    TensorVec& locals,
    TensorVec& local_grads,
    const D& device
) {
    batch_reduce_mean_backward_impl(instruction, locals, local_grads, device, false);
}

template <typename D>
void op_batch_reduce_mean_keepdim(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const D& device
) {
    batch_reduce_mean_impl(instruction, locals, device, true);
}

template <typename D>
void backward_op_batch_reduce_mean_keepdim(
    const CpuRuntime::Instruction& instruction,
    TensorVec& locals,
    TensorVec& local_grads,
    const D& device
) {
    batch_reduce_mean_backward_impl(instruction, locals, local_grads, device, true);
}

template <typename D>
void backward_op_full(
    const CpuRuntime::Instruction& instruction,
    TensorVec& locals,
    TensorVec& local_grads,
    const D& device
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

template <typename D>
void op_offset_indices(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const D& device
) {
    auto& sizes_offset = locals[instruction.input_indices[0]].batch_sizes();
    auto& sizes_out = locals[instruction.input_indices[1]].batch_sizes();
    std::size_t total_size = std::accumulate(sizes_out.begin(), sizes_out.end(), 0);
    auto& output = locals[instruction.output_indices[0]];
    output = Tensor(DataType::dt_int, {total_size}, device);
    auto flat_view = output.flat_view<me_int_t, 1>(0);

    std::size_t sum_offset = 0, sum_out = 0;
    for (auto [size_offset, size_out] : zip(sizes_offset, sizes_out)) {
        device.foreach (
            size_out,
            [flat_view,
             sum_offset,
             sum_out](std::size_t job_count, std::size_t job_offset) mutable {
                auto output_view = TensorView<me_int_t, 1>(flat_view);
                for (std::size_t i = sum_out + job_offset;
                     i < sum_out + job_offset + job_count;
                     ++i) {
                    output_view[i] = sum_offset;
                }
            }
        );
        sum_offset += size_offset;
        sum_out += size_out;
    }
}

template <typename D>
void op_quantile(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const D& device
) {
    auto& input = locals[instruction.input_indices[0]];
    auto& q = locals[instruction.input_indices[1]];
    auto& output = locals[instruction.output_indices[0]];
    if (input.size(0) == 0) {
        output = Tensor(0., device);
        return;
    }

    output = Tensor(DataType::dt_float, {1}, device);
    device.foreach (
        input.size(0),
        [&](std::size_t count, std::size_t offset) {
            std::size_t batch_size = input.size(0);
            auto input_view = input.view<double, 1>();
            double q_val = q.view<double, 1>()[0];
            std::size_t q_index = std::min(
                static_cast<std::size_t>(std::max(q_val, 0.0) * batch_size),
                batch_size - 1
            );
            std::vector<std::size_t> indices(batch_size);
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), [&](auto i, auto j) {
                return input_view[i] < input_view[j];
            });
            output.view<double, 1>()[0] = input_view[indices.at(q_index)];
        },
        true
    );
}

template <typename D>
void op_random(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const D& device
) {
    auto batch_size = locals[instruction.input_indices[0]].batch_sizes()[0];
    auto& output = locals[instruction.output_indices[0]];
    auto dim = instruction.output_shapes[0][0];
    output = Tensor(DataType::dt_float, {batch_size, dim}, device);
    auto flat_view = output.flat_view<double, 1>(2);
    auto& runtime = instruction.runtime;
    device.foreach (
        flat_view.shape[0],
        [flat_view, &runtime](std::size_t count, std::size_t offset) mutable {
            auto output_view = TensorView<double, 1>(flat_view);
            std::uniform_real_distribution<double> dist;
            auto& rand_gen = runtime.rand_gen();
            for (std::size_t i = offset; i < offset + count; ++i) {
                output_view[i] = dist(rand_gen);
            }
        }
    );
}

template <typename D>
void op_random_int(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const D& device
) {
    auto batch_size = locals[instruction.input_indices[0]].batch_sizes()[0];
    auto max_val = locals[instruction.input_indices[1]].batch_sizes()[0];
    auto& output = locals[instruction.output_indices[0]];
    output = Tensor(DataType::dt_float, {batch_size}, device);
    auto flat_view = output.flat_view<me_int_t, 1>(0);
    auto& runtime = instruction.runtime;
    device.foreach (
        flat_view.shape[0],
        [flat_view, max_val, &runtime](std::size_t count, std::size_t offset) mutable {
            auto output_view = TensorView<me_int_t, 1>(flat_view);
            std::uniform_int_distribution<me_int_t> dist(0, max_val - 1);
            auto& rand_gen = runtime.rand_gen();
            for (std::size_t i = offset; i < offset + count; ++i) {
                output_view[i] = dist(rand_gen);
            }
        }
    );
}

template <typename D>
void op_unweight(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const D& device
) {
    // TODO: not parallelized for now
    auto& weights = locals[instruction.input_indices[0]];
    auto& max_weight = locals[instruction.input_indices[1]];
    auto& indices = locals[instruction.output_indices[0]];
    auto& uw_weights = locals[instruction.output_indices[1]];

    auto batch_size = weights.size(0);
    Tensor indices_tmp(DataType::dt_int, {batch_size});
    Tensor uw_weights_tmp(DataType::dt_float, {batch_size});

    auto weights_view_flat = weights.flat_view<double, 1>(0);
    auto max_weight_view_flat = max_weight.flat_view<double, 1>(0);
    auto indices_view_flat = indices_tmp.flat_view<me_int_t, 1>(0);
    auto uw_weights_view_flat = uw_weights_tmp.flat_view<double, 1>(0);
    auto& runtime = instruction.runtime;

    device.submit(
        [weights_view_flat,
         max_weight_view_flat,
         indices_view_flat,
         uw_weights_view_flat,
         &runtime,
         batch_size,
         &indices,
         &uw_weights,
         indices_tmp,
         uw_weights_tmp]() mutable {
            TensorView<double, 1> weights_view(weights_view_flat);
            TensorView<double, 1> max_weight_view(max_weight_view_flat);
            TensorView<me_int_t, 1> indices_view(indices_view_flat);
            TensorView<double, 1> uw_weights_view(uw_weights_view_flat);
            std::uniform_real_distribution<double> dist;
            auto& rand_gen = runtime.rand_gen();
            std::size_t count = 0;
            for (std::size_t i = 0; i < batch_size; ++i) {
                double w = weights_view[i], w_max = max_weight_view[i];
                if (w != 0. && w > dist(rand_gen) * w_max) {
                    indices_view[count] = i;
                    uw_weights_view[count] = w > w_max ? w : w_max;
                    ++count;
                }
            }

            indices = indices_tmp.slice(0, 0, count);
            uw_weights = uw_weights_tmp.slice(0, 0, count);
        }
    );
}

template <typename D>
void op_histogram(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const D& device
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
    values = Tensor(DataType::dt_float, shape, device);
    square_values = Tensor(DataType::dt_float, shape, device);
    device.sync_barrier();

    auto weights_view_flat = weights.flat_view<double, 1>(0);
    auto input_view_flat = input.flat_view<double, 1>(input.shape().size());
    auto min_view_flat = hist_min.flat_view<double, 1>(0);
    auto max_view_flat = hist_max.flat_view<double, 1>(0);
    auto values_view_flat = values.flat_view<double, 2>(0);
    auto square_values_view_flat = square_values.flat_view<double, 2>(0);

    std::size_t batch_size = locals[instruction.batch_size_index].size(0);

    device.submit(
        [input,
         input_view_flat,
         weights_view_flat,
         min_view_flat,
         max_view_flat,
         values_view_flat,
         square_values_view_flat,
         batch_size]() mutable {
            TensorView<double, 1> input_view(input_view_flat);
            TensorView<double, 1> weights_view(weights_view_flat);
            TensorView<double, 1> min_view(min_view_flat);
            TensorView<double, 1> max_view(max_view_flat);
            TensorView<double, 2> values_view(values_view_flat);
            TensorView<double, 2> square_values_view(square_values_view_flat);

            std::size_t n_bins = values_view.size(1) - 2;

            auto bin_values = values_view[0];
            auto bin_square_values = square_values_view[0];
            for (std::size_t i_bin = 0; i_bin < n_bins + 2; ++i_bin) {
                bin_values[i_bin] = 0.;
                bin_square_values[i_bin] = 0.;
            }
            for (std::size_t i_sample = 0; i_sample < batch_size; ++i_sample) {
                int i_bin_rounded = (input_view[i_sample] - min_view[i_sample]) /
                    (max_view[i_sample] - min_view[i_sample]) * n_bins;
                int i_bin;
                if (i_bin_rounded < 0) {
                    i_bin = 0;
                } else if (i_bin_rounded >= n_bins) {
                    i_bin = n_bins + 1;
                } else {
                    i_bin = i_bin_rounded + 1;
                }
                double w = weights_view[i_sample];
                bin_values[i_bin] += w;
                bin_square_values[i_bin] += w * w;
            }
        }
    );
}

template <typename D>
void op_vegas_histogram(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const D& device
) {
    auto& input = locals[instruction.input_indices[0]];
    auto& weights = locals[instruction.input_indices[1]];
    auto& values = locals[instruction.output_indices[0]];
    auto& counts = locals[instruction.output_indices[1]];

    auto out_shape = instruction.output_shapes[0];
    Sizes shape(out_shape.size() + 1);
    shape[0] = 1;
    std::copy(out_shape.begin(), out_shape.end(), shape.begin() + 1);
    values = Tensor(DataType::dt_float, shape, device);
    counts = Tensor(DataType::dt_int, shape, device);
    device.sync_barrier();

    auto input_view_flat = input.flat_view<double, 2>(0);
    auto weights_view_flat = weights.flat_view<double, 1>(0);
    auto values_view_flat = values.flat_view<double, 3>(0);
    auto counts_view_flat = counts.flat_view<me_int_t, 3>(0);

    std::size_t batch_size = locals[instruction.batch_size_index].size(0);

    device.submit(
        [input_view_flat,
         weights_view_flat,
         values_view_flat,
         counts_view_flat,
         batch_size]() mutable {
            TensorView<double, 2> input_view(input_view_flat);
            TensorView<double, 1> weights_view(weights_view_flat);
            TensorView<double, 3> values_view(values_view_flat);
            TensorView<me_int_t, 3> counts_view(counts_view_flat);

            std::size_t n_dims = input_view.size(1);
            std::size_t n_bins = values_view.size(2);

            for (std::size_t i_dim = 0; i_dim < n_dims; ++i_dim) {
                auto bin_values = values_view[0][i_dim];
                auto bin_counts = counts_view[0][i_dim];
                for (std::size_t i_bin = 0; i_bin < n_bins; ++i_bin) {
                    bin_values[i_bin] = 0.;
                    bin_counts[i_bin] = 0;
                }
                for (std::size_t i_sample = 0; i_sample < batch_size; ++i_sample) {
                    int i_bin = input_view[i_sample][i_dim] * n_bins;
                    if (i_bin < 0 || i_bin >= n_bins) {
                        continue;
                    }
                    double w = weights_view[i_sample];
                    bin_values[i_bin] += w * w;
                    bin_counts[i_bin] += 1;
                }
            }
        }
    );
}

template <typename D>
void op_discrete_histogram(
    const CpuRuntime::Instruction& instruction, TensorVec& locals, const D& device
) {
    auto& input = locals[instruction.input_indices[0]];
    auto& weights = locals[instruction.input_indices[1]];
    auto& values = locals[instruction.output_indices[0]];
    auto& counts = locals[instruction.output_indices[1]];

    auto out_shape = instruction.output_shapes[0];
    Sizes shape(out_shape.size() + 1);
    shape[0] = 1;
    std::copy(out_shape.begin(), out_shape.end(), shape.begin() + 1);
    values = Tensor(DataType::dt_float, shape, device);
    counts = Tensor(DataType::dt_int, shape, device);

    std::size_t batch_size = locals[instruction.batch_size_index].size(0);

    auto input_view_flat = input.flat_view<me_int_t, 1>(0);
    auto weights_view_flat = weights.flat_view<double, 1>(0);
    auto values_view_flat = values.flat_view<double, 2>(0);
    auto counts_view_flat = counts.flat_view<me_int_t, 2>(0);

    device.submit(
        [input_view_flat,
         weights_view_flat,
         values_view_flat,
         counts_view_flat,
         batch_size]() mutable {
            TensorView<me_int_t, 1> input_view(input_view_flat);
            TensorView<double, 1> weights_view(weights_view_flat);
            TensorView<double, 2> values_view(values_view_flat);
            TensorView<me_int_t, 2> counts_view(counts_view_flat);
            std::size_t n_opts = values_view.size(1);
            for (std::size_t i_opt = 0; i_opt < n_opts; ++i_opt) {
                values_view[0][i_opt] = 0.;
                counts_view[0][i_opt] = 0;
            }
            for (std::size_t i = 0; i < batch_size; ++i) {
                auto w = weights_view[i];
                std::size_t index = input_view[i];
                values_view[0][index] += w;
                counts_view[0][index] += 1;
            }
        }
    );
}

} // namespace

CpuRuntime::CpuRuntime(const Function& function, ContextPtr context, bool concurrent) :
    _context(context),
    _input_count(function.inputs().size()),
    _rand_gens(
        context->thread_pool(),
        []() {
            std::random_device rand_device;
            return std::mt19937(rand_device());
        }
    ),
    _concurrent(concurrent) {
    if (context->device()->device_type() != DeviceType::cpu) {
        throw std::runtime_error("Context has incompatible device");
    }
    std::size_t instr_count = function.instructions().size();
    std::vector<int> local_sources_backward(function.locals().size(), -1);
    std::vector<SizeVec> local_uses_backward(function.locals().size());
    std::vector<std::size_t> dep_counts_backward(instr_count);
    std::vector<SizeVec> dep_instrs_backward(instr_count);
    for (std::size_t instr_index = instr_count;
         auto& instr : std::views::reverse(function.instructions())) {
        --instr_index;
        bool is_ready = true;
        std::size_t dependency_count = 0;
        for (auto& out : instr.outputs) {
            local_uses_backward.at(out.local_index).push_back(instr_index);
            std::size_t source_instr = local_sources_backward.at(out.local_index);
            if (source_instr != -1) {
                is_ready = false;
                auto& source_deps = dep_instrs_backward.at(source_instr);
                if (std::find(source_deps.begin(), source_deps.end(), instr_index) ==
                    source_deps.end()) {
                    source_deps.push_back(instr_index);
                    ++dependency_count;
                }
            }
        }
        dep_counts_backward.at(instr_index) = dependency_count;
        if (is_ready) {
            _ready_instructions_backward_init.push_back(instr_index);
        }

        for (auto& in : instr.inputs) {
            local_sources_backward.at(in.local_index) = instr_index;
        }
    }

    _locals_init.resize(function.locals().size());
    _requires_grad_init.resize(function.locals().size());
    LastUseOfLocals last_use(function);
    std::vector<int> local_sources(function.locals().size(), -1);
    std::vector<SizeVec> local_uses(function.locals().size());
    SizeVec instr_index_map;

    for (std::size_t instr_index = 0;
         auto [instr, dep_instrs_bw, dep_count_bw] :
         zip(function.instructions(), dep_instrs_backward, dep_counts_backward)) {
        std::size_t runtime_instr_index = _instructions.size();
        SizeVec input_indices;
        std::size_t batch_size_index = instr.inputs.at(0).local_index;
        bool is_ready = true;
        std::size_t dependency_count = 0;
        for (auto& in : instr.inputs) {
            input_indices.push_back(in.local_index);
            if (in.type.batch_size != BatchSize::one) {
                batch_size_index = in.local_index;
            }

            local_uses.at(in.local_index).push_back(runtime_instr_index);
            std::size_t source_instr = local_sources.at(in.local_index);
            if (source_instr != -1) {
                is_ready = false;
                auto& source_deps =
                    _instructions.at(source_instr).dependent_instructions;
                if (std::find(
                        source_deps.begin(), source_deps.end(), runtime_instr_index
                    ) == source_deps.end()) {
                    source_deps.push_back(runtime_instr_index);
                    ++dependency_count;
                }
            }
        }
        if (is_ready) {
            _ready_instructions_init.push_back(runtime_instr_index);
        }

        SizeVec output_indices;
        std::vector<DataType> output_dtypes;
        std::vector<SizeVec> output_shapes;
        for (auto& out : instr.outputs) {
            output_indices.push_back(out.local_index);
            output_dtypes.push_back(out.type.dtype);
            output_shapes.push_back({out.type.shape.begin(), out.type.shape.end()});
            local_sources.at(out.local_index) = runtime_instr_index;
        }

        instr_index_map.push_back(runtime_instr_index);
        _instructions.push_back(
            {instr.instruction->opcode(),
             input_indices,
             output_indices,
             output_dtypes,
             output_shapes,
             batch_size_index,
             *this,
             instr.instruction->differentiable(),
             {},
             dependency_count,
             dep_instrs_bw,
             dep_count_bw}
        );

        for (std::size_t local_index : last_use.local_indices(instr_index)) {
            std::size_t free_dep_count = 0;
            std::size_t free_instr_index = _instructions.size();
            for (auto use_index : local_uses.at(local_index)) {
                auto& use_deps = _instructions.at(use_index).dependent_instructions;
                if (std::find(use_deps.begin(), use_deps.end(), free_instr_index) ==
                    use_deps.end()) {
                    use_deps.push_back(free_instr_index);
                    ++free_dep_count;
                }
            }
            _instructions.push_back(
                {-1,
                 {local_index},
                 {},
                 {},
                 {},
                 0,
                 *this,
                 false,
                 {},
                 free_dep_count,
                 {},
                 0}
            );
        }
        ++instr_index;
    }

    // rewrite indices for backward pass to account for the added "free" instructions
    for (auto& instr : _instructions) {
        for (auto& dep : instr.dependent_instructions_backward) {
            dep = instr_index_map.at(dep);
        }
    }
    for (auto& index : _ready_instructions_backward_init) {
        index = instr_index_map.at(index);
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
                    Tensor tensor(val, cpu_device());
                    _locals_init[local.local_index] = tensor;
                },
                [](std::monostate val) {}
            },
            local.literal_value
        );
    }

    for (auto& out : function.outputs()) {
        _output_indices.push_back(out.local_index);
    }
}

TensorVec CpuRuntime::run(const TensorVec& inputs) {
    if (_concurrent && _context->thread_pool().thread_count() > 1) {
        auto [outputs, locals, eval_grad] = run_concurrent(inputs, {}, false);
        return outputs;
    } else {
        return run_single(inputs);
    }
}

std::tuple<TensorVec, TensorVec, std::vector<bool>> CpuRuntime::run_with_grad(
    const TensorVec& inputs, const std::vector<bool>& input_requires_grad
) {
    if (_concurrent && _context->thread_pool().thread_count() > 1) {
        return run_concurrent(inputs, input_requires_grad, true);
    } else {
        return run_with_grad_single(inputs, input_requires_grad);
    }
}

std::pair<TensorVec, TensorVec> CpuRuntime::run_backward(
    const TensorVec& output_grads,
    const TensorVec& stored_locals,
    const std::vector<bool>& eval_grad,
    bool return_contiguous_grads
) {
    if (_concurrent && _context->thread_pool().thread_count() > 1) {
        return run_backward_concurrent(
            output_grads, stored_locals, eval_grad, return_contiguous_grads
        );
    } else {
        return run_backward_single(
            output_grads, stored_locals, eval_grad, return_contiguous_grads
        );
    }
}

TensorVec CpuRuntime::run_single(const TensorVec& inputs) const {
    auto& device = CpuDevice::instance();
    auto locals = _locals_init;
    std::copy(inputs.begin(), inputs.end(), locals.begin());

    for (auto& instr : _instructions) {
        using DeviceType = CpuDevice;
        switch (instr.opcode) {
        case -1: // free memory
            locals[instr.input_indices[0]].reset(device);
            break;
#include "runtime_mixin.inc"
        }
    }
    TensorVec outputs;
    for (auto index : _output_indices) {
        outputs.push_back(locals[index]);
    }
    return outputs;
}

std::tuple<TensorVec, TensorVec, std::vector<bool>> CpuRuntime::run_with_grad_single(
    const TensorVec& inputs, const std::vector<bool>& input_requires_grad
) const {
    auto& device = CpuDevice::instance();
    auto locals = _locals_init;
    auto requires_grad = _requires_grad_init;
    std::vector<bool> store_local(locals.size());
    std::vector<bool> eval_grad(_instructions.size());
    std::copy(inputs.begin(), inputs.end(), locals.begin());
    std::copy(
        input_requires_grad.begin(), input_requires_grad.end(), requires_grad.begin()
    );

    for (auto [instr, instr_eval_grad] : zip(_instructions, eval_grad)) {
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
        using DeviceType = CpuDevice;
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
    }
    TensorVec outputs;
    for (auto index : _output_indices) {
        outputs.push_back(locals[index]);
    }
    return {outputs, locals, eval_grad};
}

std::pair<TensorVec, TensorVec> CpuRuntime::run_backward_single(
    const TensorVec& output_grads,
    const TensorVec& stored_locals,
    const std::vector<bool>& eval_grad,
    bool return_contiguous_grads
) const {
    auto& device = CpuDevice::instance();
    TensorVec local_grads(stored_locals.size());
    TensorVec locals(stored_locals);
    for (auto [index, grad] : zip(_output_indices, output_grads)) {
        local_grads[index] = grad;
    }

    Tensor all_global_grads(
        DataType::dt_float, {_grad_global_total_size}, AllocHint::global_grad
    );
    // all_global_grads.zero();
    TensorVec global_grads = all_global_grads.split_and_reshape(_grad_global_shapes);
    for (auto [index, grad] : zip(_grad_global_indices, global_grads)) {
        local_grads[index] = grad;
    }

    for (auto [instr, instr_eval_grad] :
         zip(std::views::reverse(_instructions), std::views::reverse(eval_grad))) {
        if (!instr_eval_grad) {
            continue;
        }
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
        using DeviceType = CpuDevice;
        switch (instr.opcode) {
#include "runtime_backward_mixin.inc"
        }
    }
    return {
        {local_grads.begin(), local_grads.begin() + _input_count},
        return_contiguous_grads ? TensorVec{all_global_grads} : global_grads
    };
}

std::tuple<TensorVec, TensorVec, std::vector<bool>> CpuRuntime::run_concurrent(
    const TensorVec& inputs,
    const std::vector<bool>& input_requires_grad,
    bool with_grad
) const {
    auto& thread_pool = _context->thread_pool();
    auto locals = _locals_init;
    auto requires_grad = _requires_grad_init;
    std::vector<bool> store_local(with_grad ? locals.size() : 0);
    std::vector<bool> eval_grad(with_grad ? _instructions.size() : 0);
    std::copy(inputs.begin(), inputs.end(), locals.begin());
    if (with_grad) {
        std::copy(
            input_requires_grad.begin(),
            input_requires_grad.end(),
            requires_grad.begin()
        );
    }

    std::size_t instr_count = _instructions.size();
    SizeVec job_counts(instr_count);
    SizeVec ready_input_count(instr_count);
    SizeVec ready_instructions(_ready_instructions_init);
    SizeVec next_ready_instructions;
    std::vector<std::vector<std::function<std::size_t()>>> funcs_after_barrier(
        instr_count
    );
    while (true) {
        while (ready_instructions.size() > 0) {
            for (std::size_t instr_index : ready_instructions) {
                auto& instr = _instructions[instr_index];
                using DeviceType = AsyncCpuDevice;
                bool barrier_state = false;
                std::size_t& job_count = job_counts[instr_index];

                if (with_grad && instr.differentiable) {
                    auto instr_eval_grad = eval_grad[instr_index];
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

                AsyncCpuDevice device(
                    instr_index,
                    job_count,
                    barrier_state,
                    funcs_after_barrier[instr_index],
                    _context->thread_pool()
                );
                switch (instr.opcode) {
                case -1: { // free memory
                    auto input_index = instr.input_indices[0];
                    if (with_grad && !store_local[input_index]) {
                        locals[input_index].reset(device);
                    }
                    break;
                }
#include "runtime_mixin.inc"
                }

                if (job_count == 0) {
                    for (std::size_t dep_index : instr.dependent_instructions) {
                        auto& ready_count = ready_input_count[dep_index];
                        ++ready_count;
                        if (ready_count == _instructions[dep_index].dependency_count) {
                            next_ready_instructions.push_back(dep_index);
                        }
                    }
                }
            }
            ready_instructions = next_ready_instructions;
            next_ready_instructions.clear();
        }

        if (auto job = thread_pool.wait()) {
            std::size_t instr_index = *job;
            auto& job_count = job_counts[instr_index];
            --job_count;
            if (job_count > 0) {
                continue;
            }

            auto& extra_funcs = funcs_after_barrier[instr_index];
            if (extra_funcs.size() > 0) {
                for (auto& func : extra_funcs) {
                    thread_pool.submit(func);
                }
                job_count = extra_funcs.size();
                extra_funcs.clear();
                continue;
            }

            auto& instr = _instructions[instr_index];
            for (std::size_t dep_index : instr.dependent_instructions) {
                auto& ready_count = ready_input_count[dep_index];
                ++ready_count;
                if (ready_count == _instructions[dep_index].dependency_count) {
                    ready_instructions.push_back(dep_index);
                }
            }
        } else if (ready_instructions.size() == 0) {
            break;
        }
    }

    TensorVec outputs;
    for (auto index : _output_indices) {
        outputs.push_back(locals[index]);
    }
    if (with_grad) {
        return {outputs, locals, eval_grad};
    } else {
        return {outputs, {}, {}};
    }
}

std::pair<TensorVec, TensorVec> CpuRuntime::run_backward_concurrent(
    const TensorVec& output_grads,
    const TensorVec& stored_locals,
    const std::vector<bool>& eval_grad,
    bool return_contiguous_grads
) const {
    auto& thread_pool = _context->thread_pool();
    TensorVec local_grads(stored_locals.size());
    TensorVec locals(stored_locals);
    for (auto [index, grad] : zip(_output_indices, output_grads)) {
        local_grads[index] = grad;
    }

    Tensor all_global_grads(
        DataType::dt_float, {_grad_global_total_size}, AllocHint::global_grad
    );
    // all_global_grads.zero();
    TensorVec global_grads = all_global_grads.split_and_reshape(_grad_global_shapes);
    for (auto [index, grad] : zip(_grad_global_indices, global_grads)) {
        local_grads[index] = grad;
    }

    std::size_t instr_count = _instructions.size();
    SizeVec job_counts(instr_count);
    SizeVec ready_input_count(instr_count);
    SizeVec ready_instructions(_ready_instructions_backward_init);
    SizeVec next_ready_instructions;
    std::vector<std::vector<std::function<std::size_t()>>> funcs_after_barrier(
        instr_count
    );
    while (true) {
        while (ready_instructions.size() > 0) {
            for (std::size_t instr_index : ready_instructions) {
                auto& instr = _instructions[instr_index];
                using DeviceType = AsyncCpuDevice;
                bool barrier_state = false;
                std::size_t& job_count = job_counts[instr_index];

                if (eval_grad[instr_index]) {
                    AsyncCpuDevice device(
                        instr_index,
                        job_count,
                        barrier_state,
                        funcs_after_barrier[instr_index],
                        _context->thread_pool()
                    );

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

                if (job_count == 0) {
                    for (std::size_t dep_index :
                         instr.dependent_instructions_backward) {
                        auto& ready_count = ready_input_count[dep_index];
                        ++ready_count;
                        if (ready_count ==
                            _instructions[dep_index].dependency_count_backward) {
                            next_ready_instructions.push_back(dep_index);
                        }
                    }
                }
            }
            ready_instructions = next_ready_instructions;
            next_ready_instructions.clear();
        }

        if (auto job = thread_pool.wait()) {
            std::size_t instr_index = *job;
            auto& job_count = job_counts[instr_index];
            --job_count;
            if (job_count > 0) {
                continue;
            }

            auto& extra_funcs = funcs_after_barrier[instr_index];
            if (extra_funcs.size() > 0) {
                for (auto& func : extra_funcs) {
                    thread_pool.submit(func);
                }
                job_count = extra_funcs.size();
                extra_funcs.clear();
                continue;
            }

            auto& instr = _instructions[instr_index];
            for (std::size_t dep_index : instr.dependent_instructions_backward) {
                auto& ready_count = ready_input_count[dep_index];
                ++ready_count;
                if (ready_count == _instructions[dep_index].dependency_count_backward) {
                    ready_instructions.push_back(dep_index);
                }
            }
        } else if (ready_instructions.size() == 0) {
            break;
        }
    }

    return {
        {local_grads.begin(), local_grads.begin() + _input_count},
        return_contiguous_grads ? TensorVec{all_global_grads} : global_grads
    };
}

extern "C" Runtime*
build_runtime(const Function& function, ContextPtr context, bool concurrent) {
    return new CpuRuntime(function, context, concurrent);
}
