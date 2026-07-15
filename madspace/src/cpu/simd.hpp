#pragma once

#ifdef USE_SIMD

#ifdef USE_SIMD_NEON
#include "simd_arm.hpp"
#endif

#ifdef USE_SIMD_AVX2
#include "simd_x86_256.hpp"
#endif

#ifdef USE_SIMD_AVX512
#include "simd_x86_512.hpp"
#endif

#else // USE_SIMD

constexpr int simd_vec_size = 1;

#endif // USE_SIMD
