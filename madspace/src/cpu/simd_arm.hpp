#include <arm_neon.h>
#include <cstddef>
#include <sleef.h>

constexpr int simd_vec_size = 2;

struct FVec {
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
    BVec() = default;
    BVec(uint64x2_t _v) : v(_v) {};
    BVec(bool _v) : v(vceqzq_u64(vdupq_n_u64(_v))) {};
    operator uint64x2_t() { return v; }
    uint64x2_t v;
};

inline FVec vgather(
    double* base_ptr, std::size_t batch_stride, std::size_t index_stride, IVec indices
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
    int* base_ptr, std::size_t batch_stride, std::size_t index_stride, IVec indices
) {
    IVec ret;
    IVec strided_indices = indices * IVec(index_stride);
    ret = vsetq_lane_s64(base_ptr[vgetq_lane_s64(strided_indices, 0)], ret, 0);
    ret = vsetq_lane_s64(
        base_ptr[vgetq_lane_s64(strided_indices, 1) + batch_stride], ret, 1
    );
    return ret;
}

inline FVec vload(double* base_ptr, std::size_t stride) {
    FVec ret;
    ret = vsetq_lane_f64(base_ptr[0], ret, 0);
    ret = vsetq_lane_f64(base_ptr[stride], ret, 1);
    return ret;
}

inline IVec vload(int* base_ptr, std::size_t stride) {
    IVec ret;
    ret = vsetq_lane_s64(base_ptr[0], ret, 0);
    ret = vsetq_lane_s64(base_ptr[stride], ret, 1);
    return ret;
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

inline void vstore(double* base_ptr, std::size_t stride, FVec values) {
    base_ptr[0] = vgetq_lane_f64(values, 0);
    base_ptr[stride] = vgetq_lane_f64(values, 1);
}

inline void vstore(int* base_ptr, std::size_t stride, IVec values) {
    base_ptr[0] = vgetq_lane_s64(values, 0);
    base_ptr[stride] = vgetq_lane_s64(values, 1);
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
inline FVec fma(FVec arg1, FVec arg2, FVec arg3) { return vfmaq_f64(arg1, arg2, arg3); }
