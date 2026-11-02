#include <arm_neon.h>
#include <cstddef>
#include <sleef.h>

constexpr int simd_vec_size = 2;
// float32 vectors have twice the lanes of the float64 ones
constexpr int simd_vec_size_f32 = 4;

struct FVec {
    static constexpr int lanes = simd_vec_size;
    FVec() = default;
    FVec(float64x2_t _v) : v(_v) {};
    FVec(double _v) : v(vdupq_n_f64(_v)) {};
    explicit FVec(int64x2_t _v) : v(vcvtq_f64_s64(_v)) {};
    operator float64x2_t() { return v; }
    FVec operator+=(FVec _v) {
        v = vaddq_f64(v, _v);
        return v;
    }
    float64x2_t v;
};

struct IVec {
    static constexpr int lanes = simd_vec_size;
    IVec() = default;
    IVec(int64x2_t _v) : v(_v) {};
    IVec(int _v) : v(vdupq_n_s64(_v)) {};
    explicit IVec(float64x2_t _v) : v(vcvtq_s64_f64(_v)) {};
    operator int64x2_t() { return v; }
    IVec operator+=(IVec _v) {
        v = vaddq_s64(v, _v);
        return v;
    }
    int64x2_t v;
};

struct BVec {
    static constexpr int lanes = simd_vec_size;
    BVec() = default;
    BVec(uint64x2_t _v) : v(_v) {};
    BVec(bool _v) : v(vdupq_n_u64(_v ? ~0ULL : 0)) {};
    operator uint64x2_t() { return v; }
    uint64x2_t v;
};

inline FVec vgather(
    VecTag<FVec>,
    double* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec indices
) {
    FVec ret;
    IVec strided_indices = indices * IVec(index_stride);
    ret = vsetq_lane_f64(base_ptr[vgetq_lane_s64(strided_indices, 0)], ret, 0);
    ret = vsetq_lane_f64(
        base_ptr[vgetq_lane_s64(strided_indices, 1) + batch_stride], ret, 1
    );
    return ret;
}

inline IVec vgather(
    VecTag<IVec>,
    int* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec indices
) {
    IVec ret;
    IVec strided_indices = indices * IVec(index_stride);
    ret = vsetq_lane_s64(base_ptr[vgetq_lane_s64(strided_indices, 0)], ret, 0);
    ret = vsetq_lane_s64(
        base_ptr[vgetq_lane_s64(strided_indices, 1) + batch_stride], ret, 1
    );
    return ret;
}

// contiguous loads and stores of the lanes of one batch element
inline FVec vload(VecTag<FVec>, double* base_ptr) { return vld1q_f64(base_ptr); }

inline IVec vload(VecTag<IVec>, int* base_ptr) { return vmovl_s32(vld1_s32(base_ptr)); }

// the same value in every lane
inline FVec vload_broadcast(VecTag<FVec>, double* base_ptr) {
    return vld1q_dup_f64(base_ptr);
}

inline IVec vload_broadcast(VecTag<IVec>, int* base_ptr) {
    return vdupq_n_s64(*base_ptr);
}

inline void vscatter(
    double* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec indices,
    FVec values
) {
    IVec strided_indices = indices * IVec(index_stride);
    base_ptr[vgetq_lane_s64(strided_indices, 0)] = vgetq_lane_f64(values, 0);
    base_ptr[vgetq_lane_s64(strided_indices, 1) + batch_stride] =
        vgetq_lane_f64(values, 1);
}

inline void vscatter(
    int* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec indices,
    IVec values
) {
    IVec strided_indices = indices * IVec(index_stride);
    base_ptr[vgetq_lane_s64(strided_indices, 0)] = vgetq_lane_s64(values, 0);
    base_ptr[vgetq_lane_s64(strided_indices, 1) + batch_stride] =
        vgetq_lane_s64(values, 1);
}

inline void vstore(double* base_ptr, FVec values) { vst1q_f64(base_ptr, values); }

inline void vstore(int* base_ptr, IVec values) {
    vst1_s32(base_ptr, vmovn_s64(values));
}

inline FVec where(BVec arg1, FVec arg2, FVec arg3) {
    return vbslq_f64(arg1, arg2, arg3);
}
inline IVec where(BVec arg1, IVec arg2, IVec arg3) {
    return vbslq_s64(arg1, arg2, arg3);
}
inline std::size_t single_index(IVec arg) { return vgetq_lane_s64(arg, 0); }
inline FVec min(FVec arg1, FVec arg2) { return where(arg1 < arg2, arg1, arg2); }
inline FVec max(FVec arg1, FVec arg2) { return where(arg1 > arg2, arg1, arg2); }

inline BVec operator==(FVec arg1, FVec arg2) { return vceqq_f64(arg1, arg2); }
inline BVec operator!=(FVec arg1, FVec arg2) {
    return vceqzq_u64(vceqq_f64(arg1, arg2));
}
inline BVec operator>(FVec arg1, FVec arg2) { return vcgtq_f64(arg1, arg2); }
inline BVec operator<(FVec arg1, FVec arg2) { return vcltq_f64(arg1, arg2); }
inline BVec operator>=(FVec arg1, FVec arg2) { return vcgeq_f64(arg1, arg2); }
inline BVec operator<=(FVec arg1, FVec arg2) { return vcleq_f64(arg1, arg2); }

inline BVec operator==(IVec arg1, IVec arg2) { return vceqq_s64(arg1, arg2); }
inline BVec operator!=(IVec arg1, IVec arg2) {
    return vceqzq_u64(vceqq_s64(arg1, arg2));
}
inline BVec operator>(IVec arg1, IVec arg2) { return vcgtq_s64(arg1, arg2); }
inline BVec operator<(IVec arg1, IVec arg2) { return vcltq_s64(arg1, arg2); }
inline BVec operator>=(IVec arg1, IVec arg2) { return vcgeq_s64(arg1, arg2); }
inline BVec operator<=(IVec arg1, IVec arg2) { return vcleq_s64(arg1, arg2); }

inline BVec operator&(BVec arg1, BVec arg2) { return vandq_u64(arg1, arg2); }
inline BVec operator|(BVec arg1, BVec arg2) { return vorrq_u64(arg1, arg2); }
inline BVec operator!(BVec arg1) { return vceqzq_u64(arg1); }
inline FVec operator-(FVec arg1) { return vnegq_f64(arg1); }
inline FVec operator+(FVec arg1, FVec arg2) { return vaddq_f64(arg1, arg2); }
inline FVec operator-(FVec arg1, FVec arg2) { return vsubq_f64(arg1, arg2); }
inline FVec operator*(FVec arg1, FVec arg2) { return vmulq_f64(arg1, arg2); }
inline FVec operator/(FVec arg1, FVec arg2) { return vdivq_f64(arg1, arg2); }
inline IVec operator-(IVec arg1) { return vnegq_s64(arg1); }
inline IVec operator+(IVec arg1, IVec arg2) { return vaddq_s64(arg1, arg2); }
inline IVec operator-(IVec arg1, IVec arg2) { return vsubq_s64(arg1, arg2); }

inline BVec isnan(FVec arg) { return arg != arg; }

inline FVec sqrt(FVec arg1) { return Sleef_sqrtd2_u05(arg1); }
inline FVec sin(FVec arg1) { return Sleef_sind2_u10(arg1); }
inline FVec cos(FVec arg1) { return Sleef_cosd2_u10(arg1); }
inline FVec sinh(FVec arg1) { return Sleef_sinhd2_u10(arg1); }
inline FVec asinh(FVec arg1) { return Sleef_asinhd2_u10(arg1); }
inline FVec cosh(FVec arg1) { return Sleef_coshd2_u10(arg1); }
inline FVec tanh(FVec arg1) { return Sleef_tanhd2_u10(arg1); }
inline FVec atan2(FVec arg1, FVec arg2) { return Sleef_atan2d2_u10(arg1, arg2); }
inline FVec pow(FVec arg1, FVec arg2) { return Sleef_powd2_u10(arg1, arg2); }
inline FVec fabs(FVec arg1) { return Sleef_fabsd2(arg1); }
inline FVec log(FVec arg1) { return Sleef_logd2_u10(arg1); }
inline FVec tan(FVec arg1) { return Sleef_tand2_u10(arg1); }
inline FVec atan(FVec arg1) { return Sleef_atand2_u10(arg1); }
inline FVec acos(FVec arg1) { return Sleef_acosd2_u10(arg1); }
inline FVec atanh(FVec arg1) { return Sleef_atanhd2_u10(arg1); }
inline FVec exp(FVec arg1) { return Sleef_expd2_u10(arg1); }
inline FVec log1p(FVec arg1) { return Sleef_log1pd2_u10(arg1); }
inline FVec expm1(FVec arg1) { return Sleef_expm1d2_u10(arg1); }
inline FVec erf(FVec arg1) { return Sleef_erfd2_u10(arg1); }
inline FVec fma(FVec arg1, FVec arg2, FVec arg3) { return vfmaq_f64(arg3, arg1, arg2); }

// float32 family: four lanes, with int32 indices and masks of the same width

struct IVec32;

struct FVec32 {
    static constexpr int lanes = simd_vec_size_f32;
    FVec32() = default;
    FVec32(float32x4_t _v) : v(_v) {};
    FVec32(double _v) : v(vdupq_n_f32(static_cast<float>(_v))) {};
    explicit FVec32(int32x4_t _v) : v(vcvtq_f32_s32(_v)) {};
    operator float32x4_t() { return v; }
    FVec32 operator+=(FVec32 _v) {
        v = vaddq_f32(v, _v);
        return v;
    }
    float32x4_t v;
};

struct IVec32 {
    static constexpr int lanes = simd_vec_size_f32;
    IVec32() = default;
    IVec32(int32x4_t _v) : v(_v) {};
    IVec32(int _v) : v(vdupq_n_s32(_v)) {};
    explicit IVec32(float32x4_t _v) : v(vcvtq_s32_f32(_v)) {};
    operator int32x4_t() { return v; }
    IVec32 operator+=(IVec32 _v) {
        v = vaddq_s32(v, _v);
        return v;
    }
    int32x4_t v;
};

struct BVec32 {
    static constexpr int lanes = simd_vec_size_f32;
    BVec32() = default;
    BVec32(uint32x4_t _v) : v(_v) {};
    BVec32(bool _v) : v(vdupq_n_u32(_v ? ~0U : 0)) {};
    operator uint32x4_t() { return v; }
    uint32x4_t v;
};

inline IVec32 operator*(IVec32 arg1, IVec32 arg2) { return vmulq_s32(arg1, arg2); }

inline FVec32 vgather(
    VecTag<FVec32>,
    float* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec32 indices
) {
    int32x4_t idx = indices * IVec32(index_stride);
    float values[4];
    for (int i = 0; i < 4; ++i) {
        values[i] = base_ptr[idx[i] + batch_stride * i];
    }
    return vld1q_f32(values);
}

inline IVec32 vgather(
    VecTag<IVec32>,
    int* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec32 indices
) {
    int32x4_t idx = indices * IVec32(index_stride);
    int values[4];
    for (int i = 0; i < 4; ++i) {
        values[i] = base_ptr[idx[i] + batch_stride * i];
    }
    return vld1q_s32(values);
}

inline FVec32 vload(VecTag<FVec32>, float* base_ptr) { return vld1q_f32(base_ptr); }

inline IVec32 vload(VecTag<IVec32>, int* base_ptr) { return vld1q_s32(base_ptr); }

inline FVec32 vload_broadcast(VecTag<FVec32>, float* base_ptr) {
    return vld1q_dup_f32(base_ptr);
}

inline IVec32 vload_broadcast(VecTag<IVec32>, int* base_ptr) {
    return vld1q_dup_s32(base_ptr);
}

inline void vscatter(
    float* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec32 indices,
    FVec32 values
) {
    int32x4_t idx = indices * IVec32(index_stride);
    float32x4_t vals = values;
    for (int i = 0; i < 4; ++i) {
        base_ptr[idx[i] + batch_stride * i] = vals[i];
    }
}

inline void vscatter(
    int* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec32 indices,
    IVec32 values
) {
    int32x4_t idx = indices * IVec32(index_stride);
    int32x4_t vals = values;
    for (int i = 0; i < 4; ++i) {
        base_ptr[idx[i] + batch_stride * i] = vals[i];
    }
}

inline void vstore(float* base_ptr, FVec32 values) { vst1q_f32(base_ptr, values); }

inline void vstore(int* base_ptr, IVec32 values) { vst1q_s32(base_ptr, values); }

inline FVec32 where(BVec32 arg1, FVec32 arg2, FVec32 arg3) {
    return vbslq_f32(arg1, arg2, arg3);
}
inline IVec32 where(BVec32 arg1, IVec32 arg2, IVec32 arg3) {
    return vbslq_s32(arg1, arg2, arg3);
}
inline std::size_t single_index(IVec32 arg) { return vgetq_lane_s32(arg, 0); }

inline BVec32 operator==(FVec32 arg1, FVec32 arg2) { return vceqq_f32(arg1, arg2); }
inline BVec32 operator!=(FVec32 arg1, FVec32 arg2) {
    return vmvnq_u32(vceqq_f32(arg1, arg2));
}
inline BVec32 operator>(FVec32 arg1, FVec32 arg2) { return vcgtq_f32(arg1, arg2); }
inline BVec32 operator<(FVec32 arg1, FVec32 arg2) { return vcltq_f32(arg1, arg2); }
inline BVec32 operator>=(FVec32 arg1, FVec32 arg2) { return vcgeq_f32(arg1, arg2); }
inline BVec32 operator<=(FVec32 arg1, FVec32 arg2) { return vcleq_f32(arg1, arg2); }

inline BVec32 operator==(IVec32 arg1, IVec32 arg2) { return vceqq_s32(arg1, arg2); }
inline BVec32 operator!=(IVec32 arg1, IVec32 arg2) {
    return vmvnq_u32(vceqq_s32(arg1, arg2));
}
inline BVec32 operator>(IVec32 arg1, IVec32 arg2) { return vcgtq_s32(arg1, arg2); }
inline BVec32 operator<(IVec32 arg1, IVec32 arg2) { return vcltq_s32(arg1, arg2); }
inline BVec32 operator>=(IVec32 arg1, IVec32 arg2) { return vcgeq_s32(arg1, arg2); }
inline BVec32 operator<=(IVec32 arg1, IVec32 arg2) { return vcleq_s32(arg1, arg2); }

inline BVec32 operator&(BVec32 arg1, BVec32 arg2) { return vandq_u32(arg1, arg2); }
inline BVec32 operator|(BVec32 arg1, BVec32 arg2) { return vorrq_u32(arg1, arg2); }
inline BVec32 operator!(BVec32 arg1) { return vmvnq_u32(arg1); }

inline FVec32 min(FVec32 arg1, FVec32 arg2) { return where(arg1 < arg2, arg1, arg2); }
inline FVec32 max(FVec32 arg1, FVec32 arg2) { return where(arg1 > arg2, arg1, arg2); }

inline FVec32 operator-(FVec32 arg1) { return vnegq_f32(arg1); }
inline FVec32 operator+(FVec32 arg1, FVec32 arg2) { return vaddq_f32(arg1, arg2); }
inline FVec32 operator-(FVec32 arg1, FVec32 arg2) { return vsubq_f32(arg1, arg2); }
inline FVec32 operator*(FVec32 arg1, FVec32 arg2) { return vmulq_f32(arg1, arg2); }
inline FVec32 operator/(FVec32 arg1, FVec32 arg2) { return vdivq_f32(arg1, arg2); }
inline IVec32 operator-(IVec32 arg1) { return vnegq_s32(arg1); }
inline IVec32 operator+(IVec32 arg1, IVec32 arg2) { return vaddq_s32(arg1, arg2); }
inline IVec32 operator-(IVec32 arg1, IVec32 arg2) { return vsubq_s32(arg1, arg2); }

inline BVec32 isnan(FVec32 arg) { return arg != arg; }

inline FVec32 sqrt(FVec32 arg1) { return Sleef_sqrtf4_u05(arg1); }
inline FVec32 sin(FVec32 arg1) { return Sleef_sinf4_u10(arg1); }
inline FVec32 cos(FVec32 arg1) { return Sleef_cosf4_u10(arg1); }
inline FVec32 sinh(FVec32 arg1) { return Sleef_sinhf4_u10(arg1); }
inline FVec32 asinh(FVec32 arg1) { return Sleef_asinhf4_u10(arg1); }
inline FVec32 cosh(FVec32 arg1) { return Sleef_coshf4_u10(arg1); }
inline FVec32 tanh(FVec32 arg1) { return Sleef_tanhf4_u10(arg1); }
inline FVec32 atan2(FVec32 arg1, FVec32 arg2) { return Sleef_atan2f4_u10(arg1, arg2); }
inline FVec32 pow(FVec32 arg1, FVec32 arg2) { return Sleef_powf4_u10(arg1, arg2); }
inline FVec32 fabs(FVec32 arg1) { return Sleef_fabsf4(arg1); }
inline FVec32 log(FVec32 arg1) { return Sleef_logf4_u10(arg1); }
inline FVec32 tan(FVec32 arg1) { return Sleef_tanf4_u10(arg1); }
inline FVec32 atan(FVec32 arg1) { return Sleef_atanf4_u10(arg1); }
inline FVec32 acos(FVec32 arg1) { return Sleef_acosf4_u10(arg1); }
inline FVec32 atanh(FVec32 arg1) { return Sleef_atanhf4_u10(arg1); }
inline FVec32 exp(FVec32 arg1) { return Sleef_expf4_u10(arg1); }
inline FVec32 log1p(FVec32 arg1) { return Sleef_log1pf4_u10(arg1); }
inline FVec32 expm1(FVec32 arg1) { return Sleef_expm1f4_u10(arg1); }
inline FVec32 erf(FVec32 arg1) { return Sleef_erff4_u10(arg1); }
inline FVec32 fma(FVec32 arg1, FVec32 arg2, FVec32 arg3) {
    return vfmaq_f32(arg3, arg1, arg2);
}
