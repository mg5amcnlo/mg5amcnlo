#pragma once

#include "gpu_abstraction.cuh"
#include "tensor.cuh"

#define KERNELSPEC __device__ //__forceinline__

namespace madspace {
namespace kernels {

template <typename F>
struct GpuTypesT {
    template <int dim>
    using FIn = const gpu::GpuTensorView<F, dim>;
    template <int dim>
    using IIn = const gpu::GpuTensorView<me_int_t, dim>;
    template <int dim>
    using FOut = gpu::GpuTensorView<F, dim>;
    template <int dim>
    using IOut = gpu::GpuTensorView<me_int_t, dim>;
    // single inputs are broadcast through a zero batch stride on the GPU
    template <int dim>
    using FSIn = const gpu::GpuTensorView<F, dim>;
    template <int dim>
    using ISIn = const gpu::GpuTensorView<me_int_t, dim>;
    using FVal = F;
    using IVal = me_int_t;
    using BVal = bool;
    // the same policy with a different float type, for conversion kernels
    using F32 = GpuTypesT<float>;
    using F64 = GpuTypesT<double>;
};
using GpuTypes = GpuTypesT<double>;
using GpuTypes32 = GpuTypesT<float>;

inline __device__ double where(bool condition, double val_true, double val_false) {
    return condition ? val_true : val_false;
}
inline __device__ float where(bool condition, float val_true, float val_false) {
    return condition ? val_true : val_false;
}
inline __device__ me_int_t
where(bool condition, me_int_t val_true, me_int_t val_false) {
    return condition ? val_true : val_false;
}
inline __device__ double min(double arg1, double arg2) {
    return arg1 < arg2 ? arg1 : arg2;
}
inline __device__ double max(double arg1, double arg2) {
    return arg1 > arg2 ? arg1 : arg2;
}
inline __device__ float min(float arg1, float arg2) {
    return arg1 < arg2 ? arg1 : arg2;
}
inline __device__ float max(float arg1, float arg2) {
    return arg1 > arg2 ? arg1 : arg2;
}
inline __device__ std::size_t single_index(me_int_t arg) { return arg; }

using ::acos;
using ::asinh;
using ::atan;
using ::atan2;
using ::atanh;
using ::cos;
using ::cosh;
using ::erf;
using ::exp;
using ::expm1;
using ::fabs;
using ::fma;
using ::log;
using ::log1p;
using ::pow;
using ::sin;
using ::sinh;
using ::sqrt;
using ::tan;
using ::tanh;

using ::isnan;

} // namespace kernels
} // namespace madspace
