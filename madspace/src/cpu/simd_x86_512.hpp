#include <cstddef>
#include <immintrin.h>
#include <sleef.h>

constexpr int simd_vec_size = 8;
// float32 vectors have twice the lanes of the float64 ones
constexpr int simd_vec_size_f32 = 16;

struct FVec {
    static constexpr int lanes = simd_vec_size;
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
    static constexpr int lanes = simd_vec_size;
    IVec() = default;
    IVec(__m512i _v) : v(_v) {};
    IVec(int _v) : v(_mm512_set1_epi64(_v)) {};
    explicit IVec(__m512d _v) { v = _mm512_cvtepi32_epi64(_mm512_cvttpd_epi32(_v)); }
    operator __m512i() { return v; }
    IVec operator+=(IVec _v) {
        v = _mm512_add_epi64(v, _v);
        return v;
    }
    __m512i v;
};

struct BVec {
    static constexpr int lanes = simd_vec_size;
    BVec() = default;
    BVec(bool _v) : v(_v ? 0xFF : 0) {};
    BVec(__mmask8 _v) : v(_v) {};
    operator __mmask8() { return v; }
    __mmask8 v;
};

// 64-bit element offsets index * index_stride + lane * batch_stride of eight lanes,
// from the low 32 bits of the 64-bit indices (indices >= 0, strides < 2^32)
inline __m512i mem_indices(
    std::size_t batch_stride,
    std::size_t index_stride,
    __m512i indices,
    long long first_lane = 0
) {
    return _mm512_add_epi64(
        _mm512_mul_epu32(indices, _mm512_set1_epi64(index_stride)),
        _mm512_mul_epu32(
            _mm512_set1_epi64(batch_stride),
            _mm512_add_epi64(
                _mm512_set1_epi64(first_lane), _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0)
            )
        )
    );
}

inline FVec vgather(
    VecTag<FVec>,
    double* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec indices
) {
    return _mm512_i64gather_pd(
        mem_indices(batch_stride, index_stride, indices), base_ptr, 8
    );
}

inline IVec vgather(
    VecTag<IVec>,
    int* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec indices
) {
    return _mm512_cvtepi32_epi64(_mm512_i64gather_epi32(
        mem_indices(batch_stride, index_stride, indices), base_ptr, 4
    ));
}

// contiguous loads and stores of the lanes of one batch element
inline FVec vload(VecTag<FVec>, double* base_ptr) { return _mm512_loadu_pd(base_ptr); }

inline IVec vload(VecTag<IVec>, int* base_ptr) {
    return _mm512_cvtepi32_epi64(
        _mm256_loadu_si256(reinterpret_cast<__m256i*>(base_ptr))
    );
}

// the same value in every lane
inline FVec vload_broadcast(VecTag<FVec>, double* base_ptr) {
    return _mm512_set1_pd(*base_ptr);
}

inline IVec vload_broadcast(VecTag<IVec>, int* base_ptr) {
    return _mm512_set1_epi64(*base_ptr);
}

inline void vscatter(
    double* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec indices,
    FVec values
) {
    _mm512_i64scatter_pd(
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

inline void vstore(double* base_ptr, FVec values) {
    _mm512_storeu_pd(base_ptr, values);
}

inline void vstore(int* base_ptr, IVec values) {
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(base_ptr), _mm512_cvtepi64_epi32(values)
    );
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
inline FVec tanh(FVec arg1) { return Sleef_tanhd8_u10avx512f(arg1); }
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

// float32 family: sixteen lanes, with int32 indices and masks of the same width

struct FVec32 {
    static constexpr int lanes = simd_vec_size_f32;
    FVec32() = default;
    FVec32(__m512 _v) : v(_v) {};
    FVec32(double _v) : v(_mm512_set1_ps(static_cast<float>(_v))) {};
    explicit FVec32(__m512i _v) : v(_mm512_cvtepi32_ps(_v)) {};
    operator __m512() { return v; }
    FVec32 operator+=(FVec32 _v) {
        v = _mm512_add_ps(v, _v);
        return v;
    }
    __m512 v;
};

struct IVec32 {
    static constexpr int lanes = simd_vec_size_f32;
    IVec32() = default;
    IVec32(__m512i _v) : v(_v) {};
    IVec32(int _v) : v(_mm512_set1_epi32(_v)) {};
    explicit IVec32(__m512 _v) : v(_mm512_cvttps_epi32(_v)) {};
    operator __m512i() { return v; }
    IVec32 operator+=(IVec32 _v) {
        v = _mm512_add_epi32(v, _v);
        return v;
    }
    __m512i v;
};

struct BVec32 {
    static constexpr int lanes = simd_vec_size_f32;
    BVec32() = default;
    BVec32(bool _v) : v(_v ? 0xFFFF : 0) {};
    BVec32(__mmask16 _v) : v(_v) {};
    operator __mmask16() { return v; }
    __mmask16 v;
};

// 64-bit offsets of the lower (half = 0) or upper eight of the sixteen lanes
template <int half>
inline __m512i
mem_indices32(std::size_t batch_stride, std::size_t index_stride, IVec32 indices) {
    return mem_indices(
        batch_stride,
        index_stride,
        _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(indices, half)),
        8 * half
    );
}

// the halves of a float32 vector (no AVX512DQ for the 32-bit inserts and extracts)
inline __m512 join_halves(__m256 low, __m256 high) {
    return _mm512_castpd_ps(_mm512_insertf64x4(
        _mm512_castps_pd(_mm512_castps256_ps512(low)), _mm256_castps_pd(high), 1
    ));
}

template <int half>
inline __m256 get_half(__m512 v) {
    return _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), half));
}

inline FVec32 vgather(
    VecTag<FVec32>,
    float* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec32 indices
) {
    return join_halves(
        _mm512_i64gather_ps(
            mem_indices32<0>(batch_stride, index_stride, indices), base_ptr, 4
        ),
        _mm512_i64gather_ps(
            mem_indices32<1>(batch_stride, index_stride, indices), base_ptr, 4
        )
    );
}

inline IVec32 vgather(
    VecTag<IVec32>,
    int* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec32 indices
) {
    return _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_i64gather_epi32(
            mem_indices32<0>(batch_stride, index_stride, indices), base_ptr, 4
        )),
        _mm512_i64gather_epi32(
            mem_indices32<1>(batch_stride, index_stride, indices), base_ptr, 4
        ),
        1
    );
}

inline FVec32 vload(VecTag<FVec32>, float* base_ptr) {
    return _mm512_loadu_ps(base_ptr);
}

inline IVec32 vload(VecTag<IVec32>, int* base_ptr) {
    return _mm512_loadu_si512(base_ptr);
}

inline FVec32 vload_broadcast(VecTag<FVec32>, float* base_ptr) {
    return _mm512_set1_ps(*base_ptr);
}

inline IVec32 vload_broadcast(VecTag<IVec32>, int* base_ptr) {
    return _mm512_set1_epi32(*base_ptr);
}

inline void vscatter(
    float* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec32 indices,
    FVec32 values
) {
    _mm512_i64scatter_ps(
        base_ptr,
        mem_indices32<0>(batch_stride, index_stride, indices),
        get_half<0>(values),
        4
    );
    _mm512_i64scatter_ps(
        base_ptr,
        mem_indices32<1>(batch_stride, index_stride, indices),
        get_half<1>(values),
        4
    );
}

inline void vscatter(
    int* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec32 indices,
    IVec32 values
) {
    _mm512_i64scatter_epi32(
        base_ptr,
        mem_indices32<0>(batch_stride, index_stride, indices),
        _mm512_extracti64x4_epi64(values, 0),
        4
    );
    _mm512_i64scatter_epi32(
        base_ptr,
        mem_indices32<1>(batch_stride, index_stride, indices),
        _mm512_extracti64x4_epi64(values, 1),
        4
    );
}

inline void vstore(float* base_ptr, FVec32 values) {
    _mm512_storeu_ps(base_ptr, values);
}

inline void vstore(int* base_ptr, IVec32 values) {
    _mm512_storeu_si512(base_ptr, values);
}

inline FVec32 where(BVec32 arg1, FVec32 arg2, FVec32 arg3) {
    return _mm512_mask_blend_ps(arg1, arg3, arg2);
}
inline IVec32 where(BVec32 arg1, IVec32 arg2, IVec32 arg3) {
    return _mm512_mask_blend_epi32(arg1, arg3, arg2);
}
inline std::size_t single_index(IVec32 arg) {
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(arg));
}
inline FVec32 min(FVec32 arg1, FVec32 arg2) { return _mm512_min_ps(arg1, arg2); }
inline FVec32 max(FVec32 arg1, FVec32 arg2) { return _mm512_max_ps(arg1, arg2); }

inline BVec32 operator==(FVec32 arg1, FVec32 arg2) {
    return _mm512_cmp_ps_mask(arg1, arg2, _CMP_EQ_OQ);
}
inline BVec32 operator!=(FVec32 arg1, FVec32 arg2) {
    return _mm512_cmp_ps_mask(arg1, arg2, _CMP_NEQ_UQ);
}
inline BVec32 operator>(FVec32 arg1, FVec32 arg2) {
    return _mm512_cmp_ps_mask(arg1, arg2, _CMP_GT_OQ);
}
inline BVec32 operator<(FVec32 arg1, FVec32 arg2) {
    return _mm512_cmp_ps_mask(arg1, arg2, _CMP_LT_OQ);
}
inline BVec32 operator>=(FVec32 arg1, FVec32 arg2) {
    return _mm512_cmp_ps_mask(arg1, arg2, _CMP_GE_OQ);
}
inline BVec32 operator<=(FVec32 arg1, FVec32 arg2) {
    return _mm512_cmp_ps_mask(arg1, arg2, _CMP_LE_OQ);
}

inline BVec32 operator&(BVec32 arg1, BVec32 arg2) {
    return static_cast<__mmask16>(arg1.v & arg2.v);
}
inline BVec32 operator|(BVec32 arg1, BVec32 arg2) {
    return static_cast<__mmask16>(arg1.v | arg2.v);
}
inline BVec32 operator!(BVec32 arg1) { return static_cast<__mmask16>(~arg1.v); }

inline BVec32 operator==(IVec32 arg1, IVec32 arg2) {
    return _mm512_cmpeq_epi32_mask(arg1, arg2);
}
inline BVec32 operator!=(IVec32 arg1, IVec32 arg2) {
    return _mm512_cmpneq_epi32_mask(arg1, arg2);
}
inline BVec32 operator>(IVec32 arg1, IVec32 arg2) {
    return _mm512_cmpgt_epi32_mask(arg1, arg2);
}
inline BVec32 operator>=(IVec32 arg1, IVec32 arg2) {
    return _mm512_cmpge_epi32_mask(arg1, arg2);
}
inline BVec32 operator<(IVec32 arg1, IVec32 arg2) {
    return _mm512_cmplt_epi32_mask(arg1, arg2);
}
inline BVec32 operator<=(IVec32 arg1, IVec32 arg2) {
    return _mm512_cmple_epi32_mask(arg1, arg2);
}

inline FVec32 operator-(FVec32 arg1) {
    return _mm512_sub_ps(_mm512_set1_ps(0.f), arg1);
}
inline FVec32 operator+(FVec32 arg1, FVec32 arg2) { return _mm512_add_ps(arg1, arg2); }
inline FVec32 operator-(FVec32 arg1, FVec32 arg2) { return _mm512_sub_ps(arg1, arg2); }
inline FVec32 operator*(FVec32 arg1, FVec32 arg2) { return _mm512_mul_ps(arg1, arg2); }
inline FVec32 operator/(FVec32 arg1, FVec32 arg2) { return _mm512_div_ps(arg1, arg2); }
inline IVec32 operator-(IVec32 arg1) {
    return _mm512_sub_epi32(_mm512_setzero_si512(), arg1);
}
inline IVec32 operator+(IVec32 arg1, IVec32 arg2) {
    return _mm512_add_epi32(arg1, arg2);
}
inline IVec32 operator-(IVec32 arg1, IVec32 arg2) {
    return _mm512_sub_epi32(arg1, arg2);
}

inline BVec32 isnan(FVec32 arg) { return arg != arg; }

inline FVec32 sqrt(FVec32 arg1) { return Sleef_sqrtf16_u05avx512f(arg1); }
inline FVec32 sin(FVec32 arg1) { return Sleef_sinf16_u10avx512f(arg1); }
inline FVec32 cos(FVec32 arg1) { return Sleef_cosf16_u10avx512f(arg1); }
inline FVec32 sinh(FVec32 arg1) { return Sleef_sinhf16_u10avx512f(arg1); }
inline FVec32 asinh(FVec32 arg1) { return Sleef_asinhf16_u10avx512f(arg1); }
inline FVec32 cosh(FVec32 arg1) { return Sleef_coshf16_u10avx512f(arg1); }
inline FVec32 tanh(FVec32 arg1) { return Sleef_tanhf16_u10avx512f(arg1); }
inline FVec32 atan2(FVec32 arg1, FVec32 arg2) {
    return Sleef_atan2f16_u10avx512f(arg1, arg2);
}
inline FVec32 pow(FVec32 arg1, FVec32 arg2) {
    return Sleef_powf16_u10avx512f(arg1, arg2);
}
inline FVec32 fabs(FVec32 arg1) { return Sleef_fabsf16_avx512f(arg1); }
inline FVec32 log(FVec32 arg1) { return Sleef_logf16_u10avx512f(arg1); }
inline FVec32 tan(FVec32 arg1) { return Sleef_tanf16_u10avx512f(arg1); }
inline FVec32 atan(FVec32 arg1) { return Sleef_atanf16_u10avx512f(arg1); }
inline FVec32 acos(FVec32 arg1) { return Sleef_acosf16_u10avx512f(arg1); }
inline FVec32 atanh(FVec32 arg1) { return Sleef_atanhf16_u10avx512f(arg1); }
inline FVec32 exp(FVec32 arg1) { return Sleef_expf16_u10avx512f(arg1); }
inline FVec32 log1p(FVec32 arg1) { return Sleef_log1pf16_u10avx512f(arg1); }
inline FVec32 expm1(FVec32 arg1) { return Sleef_expm1f16_u10avx512f(arg1); }
inline FVec32 erf(FVec32 arg1) { return Sleef_erff16_u10avx512f(arg1); }
inline FVec32 fma(FVec32 arg1, FVec32 arg2, FVec32 arg3) {
    return _mm512_fmadd_ps(arg1, arg2, arg3);
}
