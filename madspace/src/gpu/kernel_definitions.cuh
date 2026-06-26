#pragma once

#include "gpu_abstraction.cuh"
#include "tensor.cuh"

#define KERNELSPEC __device__ //__forceinline__

namespace madspace {
namespace kernels {

struct GpuTypes {
    template <int dim>
    using FIn = const gpu::GpuTensorView<double, dim>;
    template <int dim>
    using IIn = const gpu::GpuTensorView<me_int_t, dim>;
    template <int dim>
    using FOut = gpu::GpuTensorView<double, dim>;
    template <int dim>
    using IOut = gpu::GpuTensorView<me_int_t, dim>;
    using FVal = double;
    using IVal = me_int_t;
    using BVal = bool;
};

inline __device__ double where(bool condition, double val_true, double val_false) {
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

using ::isnan;

} // namespace kernels
} // namespace madspace
