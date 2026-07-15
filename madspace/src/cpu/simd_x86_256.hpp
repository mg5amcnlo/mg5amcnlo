#include <cstddef>
#include <immintrin.h>
#include <sleef.h>

constexpr int simd_vec_size = 4;

inline __m128i truncate_i64_to_i32(__m256i arg) {
    __m256 x = _mm256_castsi256_ps(arg);
    return _mm_castps_si128(_mm_shuffle_ps(
        _mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1), _MM_SHUFFLE(2, 0, 2, 0)
    ));
}

struct FVec {
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
    IVec() = default;
    IVec(__m256i _v) : v(_v) {};
    IVec(int _v) : v(_mm256_set1_epi64x(_v)) {};
    explicit IVec(__m256d _v) { v = _mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(_v)); }
    operator __m256i() { return v; }
    IVec operator+=(IVec _v) {
        v = _mm256_add_epi64(v, _v);
        return v;
    }
    __m256i v;
};

struct BVec {
    BVec() = default;
    BVec(__m256d _v) : v(_v) {};
    BVec(__m256i _v) : v(_mm256_castsi256_pd(_v)) {};
    BVec(bool _v) : v(_mm256_set1_pd(-1 * static_cast<int>(_v))) {};
    operator __m256d() { return v; }
    operator __m256i() { return _mm256_castpd_si256(v); }
    __m256d v;
};

inline __m128i stride_seq(std::size_t stride) {
    return _mm_mullo_epi32(_mm_set1_epi32(stride), _mm_set_epi32(3, 2, 1, 0));
}

inline __m128i
mem_indices(std::size_t batch_stride, std::size_t index_stride, IVec indices) {
    return _mm_add_epi32(
        _mm_mullo_epi32(truncate_i64_to_i32(indices), _mm_set1_epi32(index_stride)),
        stride_seq(batch_stride)
    );
}

inline FVec vgather(
    double* base_ptr, std::size_t batch_stride, std::size_t index_stride, IVec indices
) {
    return _mm256_i32gather_pd(
        base_ptr, mem_indices(batch_stride, index_stride, indices), 8
    );
}

inline IVec vgather(
    int* base_ptr, std::size_t batch_stride, std::size_t index_stride, IVec indices
) {
    return _mm256_cvtepi32_epi64(_mm_i32gather_epi32(
        base_ptr, mem_indices(batch_stride, index_stride, indices), 4
    ));
}

inline FVec vload(double* base_ptr, std::size_t stride) {
    return _mm256_i32gather_pd(base_ptr, stride_seq(stride), 8);
}

inline IVec vload(int* base_ptr, std::size_t stride) {
    return _mm256_cvtepi32_epi64(_mm_i32gather_epi32(base_ptr, stride_seq(stride), 4));
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
    _mm256_store_pd(values_buf, values);
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

inline void vstore(double* base_ptr, std::size_t stride, FVec values) {
    double values_buf[4];
    _mm256_store_pd(values_buf, values);
    for (int i = 0; i < 4; ++i) {
        base_ptr[i * stride] = values_buf[i];
    }
}

inline void vstore(int* base_ptr, std::size_t stride, IVec values) {
    union {
        int scalar[4];
        __m128i vec;
    } values_buf;
    _mm_store_si128(&values_buf.vec, truncate_i64_to_i32(values));
    for (int i = 0; i < 4; ++i) {
        base_ptr[i * stride] = values_buf.scalar[i];
    }
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
    return _mm256_sub_epi32(_mm256_set1_epi32(0), arg1);
}
inline IVec operator+(IVec arg1, IVec arg2) { return _mm256_add_epi32(arg1, arg2); }
inline IVec operator-(IVec arg1, IVec arg2) { return _mm256_sub_epi32(arg1, arg2); }

inline BVec isnan(FVec arg) { return arg != arg; }

inline FVec sqrt(FVec arg1) { return Sleef_sqrtd4_u05avx2(arg1); }
inline FVec sin(FVec arg1) { return Sleef_sind4_u10avx2(arg1); }
inline FVec cos(FVec arg1) { return Sleef_cosd4_u10avx2(arg1); }
inline FVec sinh(FVec arg1) { return Sleef_sinhd4_u10avx2(arg1); }
inline FVec asinh(FVec arg1) { return Sleef_asinhd4_u10avx2(arg1); }
inline FVec cosh(FVec arg1) { return Sleef_coshd4_u10avx2(arg1); }
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
