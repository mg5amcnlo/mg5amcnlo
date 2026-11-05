#pragma once

// Selects the vector type for loads and gathers, which can not be overloaded on
// the return type alone (e.g. int loads into the float64 or float32 family)
template <class V>
struct VecTag {};

#ifdef USE_SIMD

#ifdef USE_SIMD_NEON
#include "simd_arm.hpp"
#endif

#ifdef USE_SIMD_AVX2
#include "simd_x86_256.hpp"
#endif

#ifdef USE_SIMD_AVX512Y
#include "simd_x86_256y.hpp"
#endif

#ifdef USE_SIMD_AVX512
#include "simd_x86_512.hpp"
#endif

#else // USE_SIMD

constexpr int simd_vec_size = 1;
constexpr int simd_vec_size_f32 = 1;

#endif // USE_SIMD

// the largest lane count, used to align jobs and pad literals
constexpr int simd_vec_size_max = simd_vec_size_f32;
