#include <cstddef>
#include <immintrin.h>
#include <sleef.h>

constexpr int simd_vec_size = 4;
// float32 vectors have twice the lanes of the float64 ones
constexpr int simd_vec_size_f32 = 8;

inline __m128i truncate_i64_to_i32(__m256i arg) {
    __m256 x = _mm256_castsi256_ps(arg);
    return _mm_castps_si128(_mm_shuffle_ps(
        _mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1), _MM_SHUFFLE(2, 0, 2, 0)
    ));
}

struct FVec {
    static constexpr int lanes = simd_vec_size;
    FVec() = default;
    FVec(__m256d _v) : v(_v) {};
    FVec(double _v) : v(_mm256_set1_pd(_v)) {};
    explicit FVec(__m256i _v) { v = _mm256_cvtepi32_pd(truncate_i64_to_i32(_v)); }
    operator __m256d() { return v; }
    FVec operator+=(FVec _v) {
        v = _mm256_add_pd(v, _v);
        return v;
    }
    __m256d v;
};

struct IVec {
    static constexpr int lanes = simd_vec_size;
    IVec() = default;
    IVec(__m256i _v) : v(_v) {};
    IVec(int _v) : v(_mm256_set1_epi64x(_v)) {};
    explicit IVec(__m256d _v) { v = _mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(_v)); }
    operator __m256i() { return v; }
    IVec operator+=(IVec _v) {
        v = _mm256_add_epi64(v, _v);
        return v;
    }
    __m256i v;
};

struct BVec {
    static constexpr int lanes = simd_vec_size;
    BVec() = default;
    BVec(__m256d _v) : v(_v) {};
    BVec(__m256i _v) : v(_mm256_castsi256_pd(_v)) {};
    BVec(bool _v) : v(_mm256_castsi256_pd(_mm256_set1_epi64x(_v ? -1 : 0))) {};
    operator __m256d() { return v; }
    operator __m256i() { return _mm256_castpd_si256(v); }
    __m256d v;
};

// 64-bit element offsets index * index_stride + lane * batch_stride of four lanes,
// from the low 32 bits of the 64-bit indices (indices >= 0, strides < 2^32)
inline __m256i mem_indices(
    std::size_t batch_stride,
    std::size_t index_stride,
    __m256i indices,
    long long first_lane = 0
) {
    return _mm256_add_epi64(
        _mm256_mul_epu32(indices, _mm256_set1_epi64x(index_stride)),
        _mm256_mul_epu32(
            _mm256_set1_epi64x(batch_stride),
            _mm256_add_epi64(
                _mm256_set1_epi64x(first_lane), _mm256_set_epi64x(3, 2, 1, 0)
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
    return _mm256_i64gather_pd(
        base_ptr, mem_indices(batch_stride, index_stride, indices), 8
    );
}

inline IVec vgather(
    VecTag<IVec>,
    int* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec indices
) {
    return _mm256_cvtepi32_epi64(_mm256_i64gather_epi32(
        base_ptr, mem_indices(batch_stride, index_stride, indices), 4
    ));
}

// contiguous loads and stores of the lanes of one batch element
inline FVec vload(VecTag<FVec>, double* base_ptr) { return _mm256_loadu_pd(base_ptr); }

inline IVec vload(VecTag<IVec>, int* base_ptr) {
    return _mm256_cvtepi32_epi64(_mm_loadu_si128(reinterpret_cast<__m128i*>(base_ptr)));
}

// the same value in every lane
inline FVec vload_broadcast(VecTag<FVec>, double* base_ptr) {
    return _mm256_broadcast_sd(base_ptr);
}

inline IVec vload_broadcast(VecTag<IVec>, int* base_ptr) {
    return _mm256_set1_epi64x(*base_ptr);
}

inline void vscatter(
    double* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec indices,
    FVec values
) {
    double values_buf[4];
    union {
        long long scalar[4];
        __m256i vec;
    } indices_buf;
    _mm256_storeu_pd(values_buf, values);
    _mm256_store_si256(&indices_buf.vec, indices);
    for (int i = 0; i < 4; ++i) {
        base_ptr[index_stride * indices_buf.scalar[i] + batch_stride * i] =
            values_buf[i];
    }
}

inline void vscatter(
    int* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec indices,
    IVec values
) {
    union {
        int scalar[4];
        __m128i vec;
    } values_buf;
    union {
        long long scalar[4];
        __m256i vec;
    } indices_buf;
    _mm_store_si128(&values_buf.vec, truncate_i64_to_i32(values));
    _mm256_store_si256(&indices_buf.vec, indices);
    for (int i = 0; i < 4; ++i) {
        base_ptr[index_stride * indices_buf.scalar[i] + batch_stride * i] =
            values_buf.scalar[i];
    }
}

inline void vstore(double* base_ptr, FVec values) {
    _mm256_storeu_pd(base_ptr, values);
}

inline void vstore(int* base_ptr, IVec values) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(base_ptr), truncate_i64_to_i32(values));
}

inline FVec where(BVec arg1, FVec arg2, FVec arg3) {
    return _mm256_blendv_pd(arg3, arg2, arg1);
}
inline IVec where(BVec arg1, IVec arg2, IVec arg3) {
    return _mm256_blendv_epi8(arg3, arg2, arg1);
}
inline std::size_t single_index(IVec arg) { return _mm256_extract_epi32(arg, 0); }
inline FVec min(FVec arg1, FVec arg2) { return _mm256_min_pd(arg1, arg2); }
inline FVec max(FVec arg1, FVec arg2) { return _mm256_max_pd(arg1, arg2); }

inline BVec operator==(FVec arg1, FVec arg2) {
    return _mm256_cmp_pd(arg1, arg2, _CMP_EQ_OQ);
}
inline BVec operator!=(FVec arg1, FVec arg2) {
    return _mm256_cmp_pd(arg1, arg2, _CMP_NEQ_UQ);
}
inline BVec operator>(FVec arg1, FVec arg2) {
    return _mm256_cmp_pd(arg1, arg2, _CMP_GT_OQ);
}
inline BVec operator<(FVec arg1, FVec arg2) {
    return _mm256_cmp_pd(arg1, arg2, _CMP_LT_OQ);
}
inline BVec operator>=(FVec arg1, FVec arg2) {
    return _mm256_cmp_pd(arg1, arg2, _CMP_GE_OQ);
}
inline BVec operator<=(FVec arg1, FVec arg2) {
    return _mm256_cmp_pd(arg1, arg2, _CMP_LE_OQ);
}

inline BVec operator&(BVec arg1, BVec arg2) { return _mm256_and_si256(arg1, arg2); }
inline BVec operator|(BVec arg1, BVec arg2) { return _mm256_or_si256(arg1, arg2); }
inline BVec operator!(BVec arg1) {
    return _mm256_xor_si256(arg1, _mm256_cmpeq_epi64(arg1, arg1));
}

inline BVec operator==(IVec arg1, IVec arg2) { return _mm256_cmpeq_epi64(arg1, arg2); }
inline BVec operator!=(IVec arg1, IVec arg2) { return !(arg1 == arg2); }
inline BVec operator>(IVec arg1, IVec arg2) { return _mm256_cmpgt_epi64(arg1, arg2); }
inline BVec operator>=(IVec arg1, IVec arg2) { return (arg1 > arg2) | (arg1 == arg2); }
inline BVec operator<(IVec arg1, IVec arg2) { return !(arg1 >= arg2); }
inline BVec operator<=(IVec arg1, IVec arg2) { return !(arg1 > arg2); }

inline FVec operator-(FVec arg1) { return _mm256_sub_pd(_mm256_set1_pd(0.), arg1); }
inline FVec operator+(FVec arg1, FVec arg2) { return _mm256_add_pd(arg1, arg2); }
inline FVec operator-(FVec arg1, FVec arg2) { return _mm256_sub_pd(arg1, arg2); }
inline FVec operator*(FVec arg1, FVec arg2) { return _mm256_mul_pd(arg1, arg2); }
inline FVec operator/(FVec arg1, FVec arg2) { return _mm256_div_pd(arg1, arg2); }
inline IVec operator-(IVec arg1) {
    return _mm256_sub_epi64(_mm256_setzero_si256(), arg1);
}
inline IVec operator+(IVec arg1, IVec arg2) { return _mm256_add_epi64(arg1, arg2); }
inline IVec operator-(IVec arg1, IVec arg2) { return _mm256_sub_epi64(arg1, arg2); }

inline BVec isnan(FVec arg) { return arg != arg; }

inline FVec sqrt(FVec arg1) { return Sleef_sqrtd4_u05avx2(arg1); }
inline FVec sin(FVec arg1) { return Sleef_sind4_u10avx2(arg1); }
inline FVec cos(FVec arg1) { return Sleef_cosd4_u10avx2(arg1); }
inline FVec sinh(FVec arg1) { return Sleef_sinhd4_u10avx2(arg1); }
inline FVec asinh(FVec arg1) { return Sleef_asinhd4_u10avx2(arg1); }
inline FVec cosh(FVec arg1) { return Sleef_coshd4_u10avx2(arg1); }
inline FVec tanh(FVec arg1) { return Sleef_tanhd4_u10avx2(arg1); }
inline FVec atan2(FVec arg1, FVec arg2) { return Sleef_atan2d4_u10avx2(arg1, arg2); }
inline FVec pow(FVec arg1, FVec arg2) { return Sleef_powd4_u10avx2(arg1, arg2); }
inline FVec fabs(FVec arg1) { return Sleef_fabsd4_avx2(arg1); }
inline FVec log(FVec arg1) { return Sleef_logd4_u10avx2(arg1); }
inline FVec tan(FVec arg1) { return Sleef_tand4_u10avx2(arg1); }
inline FVec atan(FVec arg1) { return Sleef_atand4_u10avx2(arg1); }
inline FVec acos(FVec arg1) { return Sleef_acosd4_u10avx2(arg1); }
inline FVec atanh(FVec arg1) { return Sleef_atanhd4_u10avx2(arg1); }
inline FVec exp(FVec arg1) { return Sleef_expd4_u10avx2(arg1); }
inline FVec log1p(FVec arg1) { return Sleef_log1pd4_u10avx2(arg1); }
inline FVec expm1(FVec arg1) { return Sleef_expm1d4_u10avx2(arg1); }
inline FVec erf(FVec arg1) { return Sleef_erfd4_u10avx2(arg1); }
inline FVec fma(FVec arg1, FVec arg2, FVec arg3) {
    return _mm256_fmadd_pd(arg1, arg2, arg3);
}

// float32 family: eight lanes, with int32 indices and masks of the same width

struct FVec32 {
    static constexpr int lanes = simd_vec_size_f32;
    FVec32() = default;
    FVec32(__m256 _v) : v(_v) {};
    FVec32(double _v) : v(_mm256_set1_ps(static_cast<float>(_v))) {};
    explicit FVec32(__m256i _v) : v(_mm256_cvtepi32_ps(_v)) {};
    operator __m256() { return v; }
    FVec32 operator+=(FVec32 _v) {
        v = _mm256_add_ps(v, _v);
        return v;
    }
    __m256 v;
};

struct IVec32 {
    static constexpr int lanes = simd_vec_size_f32;
    IVec32() = default;
    IVec32(__m256i _v) : v(_v) {};
    IVec32(int _v) : v(_mm256_set1_epi32(_v)) {};
    explicit IVec32(__m256 _v) : v(_mm256_cvttps_epi32(_v)) {};
    operator __m256i() { return v; }
    IVec32 operator+=(IVec32 _v) {
        v = _mm256_add_epi32(v, _v);
        return v;
    }
    __m256i v;
};

struct BVec32 {
    static constexpr int lanes = simd_vec_size_f32;
    BVec32() = default;
    BVec32(__m256 _v) : v(_v) {};
    BVec32(__m256i _v) : v(_mm256_castsi256_ps(_v)) {};
    BVec32(bool _v) : v(_mm256_castsi256_ps(_mm256_set1_epi32(_v ? -1 : 0))) {};
    operator __m256() { return v; }
    operator __m256i() { return _mm256_castps_si256(v); }
    __m256 v;
};

// 64-bit offsets of the lower (half = 0) or upper four of the eight lanes
template <int half>
inline __m256i
mem_indices32(std::size_t batch_stride, std::size_t index_stride, IVec32 indices) {
    return mem_indices(
        batch_stride,
        index_stride,
        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(indices, half)),
        4 * half
    );
}

inline FVec32 vgather(
    VecTag<FVec32>,
    float* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec32 indices
) {
    return _mm256_set_m128(
        _mm256_i64gather_ps(
            base_ptr, mem_indices32<1>(batch_stride, index_stride, indices), 4
        ),
        _mm256_i64gather_ps(
            base_ptr, mem_indices32<0>(batch_stride, index_stride, indices), 4
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
    return _mm256_set_m128i(
        _mm256_i64gather_epi32(
            base_ptr, mem_indices32<1>(batch_stride, index_stride, indices), 4
        ),
        _mm256_i64gather_epi32(
            base_ptr, mem_indices32<0>(batch_stride, index_stride, indices), 4
        )
    );
}

inline FVec32 vload(VecTag<FVec32>, float* base_ptr) {
    return _mm256_loadu_ps(base_ptr);
}

inline IVec32 vload(VecTag<IVec32>, int* base_ptr) {
    return _mm256_loadu_si256(reinterpret_cast<__m256i*>(base_ptr));
}

inline FVec32 vload_broadcast(VecTag<FVec32>, float* base_ptr) {
    return _mm256_broadcast_ss(base_ptr);
}

inline IVec32 vload_broadcast(VecTag<IVec32>, int* base_ptr) {
    return _mm256_set1_epi32(*base_ptr);
}

inline void vscatter(
    float* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec32 indices,
    FVec32 values
) {
    float values_buf[8];
    int indices_buf[8];
    _mm256_storeu_ps(values_buf, values);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(indices_buf), indices);
    for (int i = 0; i < 8; ++i) {
        base_ptr[index_stride * indices_buf[i] + batch_stride * i] = values_buf[i];
    }
}

inline void vscatter(
    int* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec32 indices,
    IVec32 values
) {
    int values_buf[8];
    int indices_buf[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(values_buf), values);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(indices_buf), indices);
    for (int i = 0; i < 8; ++i) {
        base_ptr[index_stride * indices_buf[i] + batch_stride * i] = values_buf[i];
    }
}

inline void vstore(float* base_ptr, FVec32 values) {
    _mm256_storeu_ps(base_ptr, values);
}

inline void vstore(int* base_ptr, IVec32 values) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(base_ptr), values);
}

inline FVec32 where(BVec32 arg1, FVec32 arg2, FVec32 arg3) {
    return _mm256_blendv_ps(arg3, arg2, arg1);
}
inline IVec32 where(BVec32 arg1, IVec32 arg2, IVec32 arg3) {
    return _mm256_blendv_epi8(arg3, arg2, arg1);
}
inline std::size_t single_index(IVec32 arg) {
    return _mm_cvtsi128_si32(_mm256_castsi256_si128(arg));
}
inline FVec32 min(FVec32 arg1, FVec32 arg2) { return _mm256_min_ps(arg1, arg2); }
inline FVec32 max(FVec32 arg1, FVec32 arg2) { return _mm256_max_ps(arg1, arg2); }

inline BVec32 operator==(FVec32 arg1, FVec32 arg2) {
    return _mm256_cmp_ps(arg1, arg2, _CMP_EQ_OQ);
}
inline BVec32 operator!=(FVec32 arg1, FVec32 arg2) {
    return _mm256_cmp_ps(arg1, arg2, _CMP_NEQ_UQ);
}
inline BVec32 operator>(FVec32 arg1, FVec32 arg2) {
    return _mm256_cmp_ps(arg1, arg2, _CMP_GT_OQ);
}
inline BVec32 operator<(FVec32 arg1, FVec32 arg2) {
    return _mm256_cmp_ps(arg1, arg2, _CMP_LT_OQ);
}
inline BVec32 operator>=(FVec32 arg1, FVec32 arg2) {
    return _mm256_cmp_ps(arg1, arg2, _CMP_GE_OQ);
}
inline BVec32 operator<=(FVec32 arg1, FVec32 arg2) {
    return _mm256_cmp_ps(arg1, arg2, _CMP_LE_OQ);
}

inline BVec32 operator&(BVec32 arg1, BVec32 arg2) {
    return _mm256_and_si256(arg1, arg2);
}
inline BVec32 operator|(BVec32 arg1, BVec32 arg2) {
    return _mm256_or_si256(arg1, arg2);
}
inline BVec32 operator!(BVec32 arg1) {
    return _mm256_xor_si256(arg1, _mm256_cmpeq_epi32(arg1, arg1));
}

inline BVec32 operator==(IVec32 arg1, IVec32 arg2) {
    return _mm256_cmpeq_epi32(arg1, arg2);
}
inline BVec32 operator!=(IVec32 arg1, IVec32 arg2) { return !(arg1 == arg2); }
inline BVec32 operator>(IVec32 arg1, IVec32 arg2) {
    return _mm256_cmpgt_epi32(arg1, arg2);
}
inline BVec32 operator>=(IVec32 arg1, IVec32 arg2) {
    return (arg1 > arg2) | (arg1 == arg2);
}
inline BVec32 operator<(IVec32 arg1, IVec32 arg2) { return !(arg1 >= arg2); }
inline BVec32 operator<=(IVec32 arg1, IVec32 arg2) { return !(arg1 > arg2); }

inline FVec32 operator-(FVec32 arg1) {
    return _mm256_sub_ps(_mm256_set1_ps(0.f), arg1);
}
inline FVec32 operator+(FVec32 arg1, FVec32 arg2) { return _mm256_add_ps(arg1, arg2); }
inline FVec32 operator-(FVec32 arg1, FVec32 arg2) { return _mm256_sub_ps(arg1, arg2); }
inline FVec32 operator*(FVec32 arg1, FVec32 arg2) { return _mm256_mul_ps(arg1, arg2); }
inline FVec32 operator/(FVec32 arg1, FVec32 arg2) { return _mm256_div_ps(arg1, arg2); }
inline IVec32 operator-(IVec32 arg1) {
    return _mm256_sub_epi32(_mm256_setzero_si256(), arg1);
}
inline IVec32 operator+(IVec32 arg1, IVec32 arg2) {
    return _mm256_add_epi32(arg1, arg2);
}
inline IVec32 operator-(IVec32 arg1, IVec32 arg2) {
    return _mm256_sub_epi32(arg1, arg2);
}

inline BVec32 isnan(FVec32 arg) { return arg != arg; }

inline FVec32 sqrt(FVec32 arg1) { return Sleef_sqrtf8_u05avx2(arg1); }
inline FVec32 sin(FVec32 arg1) { return Sleef_sinf8_u10avx2(arg1); }
inline FVec32 cos(FVec32 arg1) { return Sleef_cosf8_u10avx2(arg1); }
inline FVec32 sinh(FVec32 arg1) { return Sleef_sinhf8_u10avx2(arg1); }
inline FVec32 asinh(FVec32 arg1) { return Sleef_asinhf8_u10avx2(arg1); }
inline FVec32 cosh(FVec32 arg1) { return Sleef_coshf8_u10avx2(arg1); }
inline FVec32 tanh(FVec32 arg1) { return Sleef_tanhf8_u10avx2(arg1); }
inline FVec32 atan2(FVec32 arg1, FVec32 arg2) {
    return Sleef_atan2f8_u10avx2(arg1, arg2);
}
inline FVec32 pow(FVec32 arg1, FVec32 arg2) { return Sleef_powf8_u10avx2(arg1, arg2); }
inline FVec32 fabs(FVec32 arg1) { return Sleef_fabsf8_avx2(arg1); }
inline FVec32 log(FVec32 arg1) { return Sleef_logf8_u10avx2(arg1); }
inline FVec32 tan(FVec32 arg1) { return Sleef_tanf8_u10avx2(arg1); }
inline FVec32 atan(FVec32 arg1) { return Sleef_atanf8_u10avx2(arg1); }
inline FVec32 acos(FVec32 arg1) { return Sleef_acosf8_u10avx2(arg1); }
inline FVec32 atanh(FVec32 arg1) { return Sleef_atanhf8_u10avx2(arg1); }
inline FVec32 exp(FVec32 arg1) { return Sleef_expf8_u10avx2(arg1); }
inline FVec32 log1p(FVec32 arg1) { return Sleef_log1pf8_u10avx2(arg1); }
inline FVec32 expm1(FVec32 arg1) { return Sleef_expm1f8_u10avx2(arg1); }
inline FVec32 erf(FVec32 arg1) { return Sleef_erff8_u10avx2(arg1); }
inline FVec32 fma(FVec32 arg1, FVec32 arg2, FVec32 arg3) {
    return _mm256_fmadd_ps(arg1, arg2, arg3);
}
