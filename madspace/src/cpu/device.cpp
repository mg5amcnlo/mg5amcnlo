#include "device.hpp"
#include "../kernels/kernels.hpp"
#include "tensor.hpp"

using namespace madspace;
using namespace madspace::cpu;
using namespace madspace::kernels;

namespace {

// calls func with the scalar and SIMD kernel policies of a float dtype
template <typename Func>
void dispatch_float(DataType dtype, const char* op, Func&& func) {
    switch (dtype) {
    case DataType::dt_float64:
        func.template operator()<CpuTypes, SimdTypes>();
        break;
    case DataType::dt_float32:
        func.template operator()<CpuTypes32, SimdTypes32>();
        break;
    default:
        throw std::runtime_error(std::string("invalid dtype in ") + op);
    }
}

template <typename D>
void tensor_copy_impl(const Tensor& source, Tensor& target, const D& device) {
    if (source.dtype() != target.dtype()) {
        throw std::runtime_error("invalid dtype in copy");
    }
    if (source.dtype() == DataType::dt_int32) {
        tensor_foreach_dynamic_single<
            kernel_copy_int<CpuTypes>,
            kernel_copy_int<SimdTypes>,
            1,
            1>({&source}, {&target}, target.size(0), device);
        return;
    }
    dispatch_float(source.dtype(), "copy", [&]<typename C, typename S>() {
        tensor_foreach_dynamic_single<kernel_copy<C>, kernel_copy<S>, 1, 1>(
            {&source}, {&target}, target.size(0), device
        );
    });
}

template <typename D>
void tensor_zero_impl(Tensor& tensor, const D& device) {
    if (tensor.dtype() == DataType::dt_int32) {
        tensor_foreach_dynamic_single<
            kernel_zero_int<CpuTypes>,
            kernel_zero_int<SimdTypes>,
            1,
            1>({&tensor}, {&tensor}, tensor.size(0), device);
        return;
    }
    dispatch_float(tensor.dtype(), "zero", [&]<typename C, typename S>() {
        tensor_foreach_dynamic_single<kernel_zero<C>, kernel_zero<S>, 1, 1>(
            {&tensor}, {&tensor}, tensor.size(0), device
        );
    });
}

template <typename D>
void tensor_add_impl(const Tensor& source, Tensor& target, const D& device) {
    if (source.dtype() != target.dtype()) {
        throw std::runtime_error("invalid dtype in add");
    }
    // a broadcast target (e.g. the gradient of a literal or global) accumulates
    // the whole source batch
    std::size_t batch_size = std::max(source.size(0), target.size(0));
    dispatch_float(source.dtype(), "add", [&]<typename C, typename S>() {
        tensor_foreach_dynamic_single<
            kernel_add_inplace<C>,
            kernel_add_inplace<S>,
            1,
            1>({&source}, {&target}, batch_size, device);
    });
}

} // namespace

void CpuDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    tensor_copy_impl(source, target, *this);
}

void CpuDevice::tensor_zero(Tensor& tensor) const { tensor_zero_impl(tensor, *this); }

void CpuDevice::tensor_add(const Tensor& source, Tensor& target) const {
    tensor_add_impl(source, target, *this);
}

void CpuDevice::adam_step(
    const Tensor& gradient,
    Tensor& parameter,
    Tensor& exp_avg,
    Tensor& exp_avg_sq,
    double step_size,
    double beta1,
    double beta2,
    double eps,
    double bias_corr2_sqrt,
    double weight_decay
) const {
    dispatch_float(parameter.dtype(), "adam_step", [&]<typename C, typename S>() {
        using F = typename C::FVal;
        tensor_foreach_dynamic<kernel_adam_step<C>, kernel_adam_step<S>, 1, 3>(
            {&gradient},
            {&parameter, &exp_avg, &exp_avg_sq},
            gradient.size(0),
            *this,
            static_cast<F>(step_size),
            static_cast<F>(beta1),
            static_cast<F>(beta2),
            static_cast<F>(eps),
            static_cast<F>(bias_corr2_sqrt),
            static_cast<F>(weight_decay)
        );
    });
}

void AsyncCpuDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    tensor_copy_impl(source, target, *this);
}

void AsyncCpuDevice::tensor_zero(Tensor& tensor) const {
    tensor_zero_impl(tensor, *this);
}

void AsyncCpuDevice::tensor_add(const Tensor& source, Tensor& target) const {
    tensor_add_impl(source, target, *this);
}

extern "C" int device_count() { return 1; }
extern "C" DevicePtr get_device(int index) { return &CpuDevice::instance(); }
