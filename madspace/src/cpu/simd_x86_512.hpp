#include <cstddef>
#include <immintrin.h>
#include <sleef.h>

constexpr int simd_vec_size = 8;

struct FVec {
    FVec() = default;
    FVec(__m512d _v) : v(_v) {};
    FVec(double _v) : v(_mm512_set1_pd(_v)) {};
    explicit FVec(__m512i _v) { v = _mm512_cvtepi32_pd(_mm512_cvtepi64_epi32(_v)); }
    operator __m512d() { return v; }
    FVec operator+=(FVec _v) {
        v = _mm512_add_pd(v, _v);
        return v;
    }
    __m512d v;
};

struct IVec {
    IVec() = default;
    IVec(__m512i _v) : v(_v) {};
    IVec(int _v) : v(_mm512_set1_epi64(_v)) {};
    explicit IVec(__m512d _v) { v = _mm512_cvtepi32_epi64(_mm512_cvtpd_epi32(_v)); }
    operator __m512i() { return v; }
    IVec operator+=(IVec _v) {
        v = _mm512_add_epi64(v, _v);
        return v;
    }
    __m512i v;
};

struct BVec {
    BVec() = default;
    BVec(bool _v) : v(_v ? 0xFF : 0) {};
    BVec(__mmask8 _v) : v(_v) {};
    operator __mmask8() { return v; }
    __mmask8 v;
};

inline __m256i stride_seq(std::size_t stride) {
    return _mm256_mullo_epi32(
        _mm256_set1_epi32(stride), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0)
    );
}

inline __m256i
mem_indices(std::size_t batch_stride, std::size_t index_stride, IVec indices) {
    return _mm256_add_epi32(
        _mm256_mullo_epi32(
            _mm512_cvtepi64_epi32(indices), _mm256_set1_epi32(index_stride)
        ),
        stride_seq(batch_stride)
    );
}

inline FVec vgather(
    double* base_ptr, std::size_t batch_stride, std::size_t index_stride, IVec indices
) {
    return _mm512_i32gather_pd(
        mem_indices(batch_stride, index_stride, indices), base_ptr, 8
    );
}

inline IVec vgather(
    int* base_ptr, std::size_t batch_stride, std::size_t index_stride, IVec indices
) {
    return _mm512_cvtepi32_epi64(_mm256_i32gather_epi32(
        base_ptr, mem_indices(batch_stride, index_stride, indices), 4
    ));
}

inline FVec vload(double* base_ptr, std::size_t stride) {
    return _mm512_i32gather_pd(stride_seq(stride), base_ptr, 8);
}

inline IVec vload(int* base_ptr, std::size_t stride) {
    return _mm512_cvtepi32_epi64(
        _mm256_i32gather_epi32(base_ptr, stride_seq(stride), 4)
    );
}

inline void vscatter(
    double* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec indices,
    FVec values
) {
    _mm512_i32scatter_pd(
        base_ptr, mem_indices(batch_stride, index_stride, indices), values, 8
    );
}

inline void vscatter(
    int* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec indices,
    IVec values
) {
    union {
        int scalar[8];
        __m256i vec;
    } values_buf;
    union {
        long long scalar[8];
        __m512i vec;
    } indices_buf;
    _mm256_store_si256(&values_buf.vec, _mm512_cvtepi64_epi32(values));
    _mm512_store_si512(&indices_buf.vec, indices);
    for (int i = 0; i < 8; ++i) {
        base_ptr[index_stride * indices_buf.scalar[i] + batch_stride * i] =
            values_buf.scalar[i];
    }
}

inline void vstore(double* base_ptr, std::size_t stride, FVec values) {
    _mm512_i32scatter_pd(base_ptr, stride_seq(stride), values, 8);
}

inline void vstore(int* base_ptr, std::size_t stride, IVec values) {
    union {
        int scalar[8];
        __m256i vec;
    } values_buf;
    _mm256_store_si256(&values_buf.vec, _mm512_cvtepi64_epi32(values));
    for (int i = 0; i < 8; ++i) {
        base_ptr[i * stride] = values_buf.scalar[i];
    }
}

inline FVec where(BVec arg1, FVec arg2, FVec arg3) {
    return _mm512_mask_blend_pd(arg1, arg3, arg2);
}
inline IVec where(BVec arg1, IVec arg2, IVec arg3) {
    return _mm512_mask_blend_epi64(arg1, arg3, arg2);
}
inline std::size_t single_index(IVec arg) {
    return _mm256_cvtsi256_si32(_mm512_cvtepi64_epi32(arg));
}
inline FVec min(FVec arg1, FVec arg2) { return _mm512_min_pd(arg1, arg2); }
inline FVec max(FVec arg1, FVec arg2) { return _mm512_max_pd(arg1, arg2); }

inline BVec operator==(FVec arg1, FVec arg2) {
    return _mm512_cmp_pd_mask(arg1, arg2, _CMP_EQ_OQ);
}
inline BVec operator!=(FVec arg1, FVec arg2) {
    return _mm512_cmp_pd_mask(arg1, arg2, _CMP_NEQ_UQ);
}
inline BVec operator>(FVec arg1, FVec arg2) {
    return _mm512_cmp_pd_mask(arg1, arg2, _CMP_GT_OQ);
}
inline BVec operator<(FVec arg1, FVec arg2) {
    return _mm512_cmp_pd_mask(arg1, arg2, _CMP_LT_OQ);
}
inline BVec operator>=(FVec arg1, FVec arg2) {
    return _mm512_cmp_pd_mask(arg1, arg2, _CMP_GE_OQ);
}
inline BVec operator<=(FVec arg1, FVec arg2) {
    return _mm512_cmp_pd_mask(arg1, arg2, _CMP_LE_OQ);
}

inline BVec operator&(BVec arg1, BVec arg2) {
    return static_cast<uint8_t>((arg1.v & arg2.v) & 0xFF);
}
inline BVec operator|(BVec arg1, BVec arg2) {
    return static_cast<uint8_t>((arg1.v | arg2.v) & 0xFF);
}
inline BVec operator!(BVec arg1) { return static_cast<uint8_t>(~arg1.v & 0xFF); }

inline BVec operator==(IVec arg1, IVec arg2) {
    return _mm512_cmpeq_epi64_mask(arg1, arg2);
}
inline BVec operator!=(IVec arg1, IVec arg2) {
    return _mm512_cmpneq_epi64_mask(arg1, arg2);
}
inline BVec operator>(IVec arg1, IVec arg2) {
    return _mm512_cmpgt_epi64_mask(arg1, arg2);
}
inline BVec operator>=(IVec arg1, IVec arg2) {
    return _mm512_cmpge_epi64_mask(arg1, arg2);
}
inline BVec operator<(IVec arg1, IVec arg2) {
    return _mm512_cmplt_epi64_mask(arg1, arg2);
}
inline BVec operator<=(IVec arg1, IVec arg2) {
    return _mm512_cmple_epi64_mask(arg1, arg2);
}

inline FVec operator-(FVec arg1) { return _mm512_sub_pd(_mm512_set1_pd(0.), arg1); }
inline FVec operator+(FVec arg1, FVec arg2) { return _mm512_add_pd(arg1, arg2); }
inline FVec operator-(FVec arg1, FVec arg2) { return _mm512_sub_pd(arg1, arg2); }
inline FVec operator*(FVec arg1, FVec arg2) { return _mm512_mul_pd(arg1, arg2); }
inline FVec operator/(FVec arg1, FVec arg2) { return _mm512_div_pd(arg1, arg2); }
inline IVec operator-(IVec arg1) {
    return _mm512_sub_epi64(_mm512_set1_epi64(0), arg1);
}
inline IVec operator+(IVec arg1, IVec arg2) { return _mm512_add_epi64(arg1, arg2); }
inline IVec operator-(IVec arg1, IVec arg2) { return _mm512_sub_epi64(arg1, arg2); }

inline BVec isnan(FVec arg) { return arg != arg; }

inline FVec sqrt(FVec arg1) { return Sleef_sqrtd8_u05avx512f(arg1); }
inline FVec sin(FVec arg1) { return Sleef_sind8_u10avx512f(arg1); }
inline FVec cos(FVec arg1) { return Sleef_cosd8_u10avx512f(arg1); }
inline FVec sinh(FVec arg1) { return Sleef_sinhd8_u10avx512f(arg1); }
inline FVec asinh(FVec arg1) { return Sleef_asinhd8_u10avx512f(arg1); }
inline FVec cosh(FVec arg1) { return Sleef_coshd8_u10avx512f(arg1); }
inline FVec atan2(FVec arg1, FVec arg2) { return Sleef_atan2d8_u10avx512f(arg1, arg2); }
inline FVec pow(FVec arg1, FVec arg2) { return Sleef_powd8_u10avx512f(arg1, arg2); }
inline FVec fabs(FVec arg1) { return Sleef_fabsd8_avx512f(arg1); }
inline FVec log(FVec arg1) { return Sleef_logd8_u10avx512f(arg1); }
inline FVec tan(FVec arg1) { return Sleef_tand8_u10avx512f(arg1); }
inline FVec atan(FVec arg1) { return Sleef_atand8_u10avx512f(arg1); }
inline FVec acos(FVec arg1) { return Sleef_acosd8_u10avx512f(arg1); }
inline FVec atanh(FVec arg1) { return Sleef_atanhd8_u10avx512f(arg1); }
inline FVec exp(FVec arg1) { return Sleef_expd8_u10avx512f(arg1); }
inline FVec log1p(FVec arg1) { return Sleef_log1pd8_u10avx512f(arg1); }
inline FVec expm1(FVec arg1) { return Sleef_expm1d8_u10avx512f(arg1); }
inline FVec erf(FVec arg1) { return Sleef_erfd8_u10avx512f(arg1); }
inline FVec fma(FVec arg1, FVec arg2, FVec arg3) {
    return _mm512_fmadd_pd(arg1, arg2, arg3);
}
