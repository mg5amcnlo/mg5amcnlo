// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jul 2020) for the MadGraph7 CUDACPP plugin.
// Further modified by: S. Hageboeck, O. Mattelaer, S. Roiser, J. Teig, A. Valassi (2020-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MGONGPUCONFIG_H
#define MGONGPUCONFIG_H 1

#include <sstream>
#include <string>

// This build ships mgOnGpuVectors.h (the SIMD vector types). The headers every backend
// shares (CPPProcess.h, Parameters.h, HelAmps_<model>.h) include it on this macro, not on
// whether some mgOnGpuVectors.h happens to be reachable on the include path.
#define MGONGPU_HAS_VECTORS_H 1

// simd backend: always built with a plain host compiler, never nvcc/hipcc, so the
// GPU-backend selection macro (see gpu/mgOnGpuConfig.h) is deliberately never
// defined here - single-file, all-backend headers rely on that to pick branches.

// Choose floating point precision (for everything but color algebra #537)
// If set from outside with e.g. -DMGONGPU_FPTYPE_FLOAT, nothing happens (#167)
#if not defined MGONGPU_FPTYPE_DOUBLE and not defined MGONGPU_FPTYPE_FLOAT
#define MGONGPU_FPTYPE_DOUBLE 1 // default
//#define MGONGPU_FPTYPE_FLOAT 1 // 2x faster
#endif

// Choose floating point precision (for color algebra alone #537)
#if not defined MGONGPU_FPTYPE2_DOUBLE and not defined MGONGPU_FPTYPE2_FLOAT
#define MGONGPU_FPTYPE2_DOUBLE 1 // default
//#define MGONGPU_FPTYPE2_FLOAT 1 // 2x faster
#endif

// Choose whether to inline all HelAmps functions (can gain ~4x, issue #229)
// By default off; set from outside with -DMGONGPU_INLINE_HELAMPS
//#define MGONGPU_INLINE_HELAMPS 1

// Choose whether to hardcode cIPD physics parameters instead of reading user cards
// By default off; set from outside with -DMGONGPU_HARDCODE_PARAM
//#define MGONGPU_HARDCODE_PARAM 1

// Complex type in C++: cxsmpl by default, or std::complex (CHOOSE ONLY ONE)
//#define MGONGPU_CPPCXTYPE_STDCOMPLEX 1 // ~8% slower on float, same on double
#define MGONGPU_CPPCXTYPE_CXSMPL 1 // default

// No BLAS on the simd backend (cuBLAS/hipBLAS are GPU-only)
#define MGONGPU_HAS_NO_BLAS 1

// nsight compute (ncu) debugging is CUDA-only; always off here
#undef MGONGPU_NSIGHT_DEBUG

// SANITY CHECKS
#if defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE_FLOAT
#error You must CHOOSE (ONE AND) ONLY ONE of MGONGPU_FPTYPE_DOUBLE or MGONGPU_FPTYPE_FLOAT
#endif
#if defined MGONGPU_FPTYPE2_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
#error You must CHOOSE (ONE AND) ONLY ONE of MGONGPU_FPTYPE2_DOUBLE or MGONGPU_FPTYPE2_FLOAT
#endif
#if defined MGONGPU_FPTYPE2_DOUBLE and defined MGONGPU_FPTYPE_FLOAT
#error You cannot use double precision for color algebra and single precision elsewhere
#endif
#if defined MGONGPU_CPPCXTYPE_STDCOMPLEX and defined MGONGPU_CPPCXTYPE_CXSMPL
#error You must CHOOSE (ONE AND) ONLY ONE of MGONGPU_CPPCXTYPE_STDCOMPLEX or MGONGPU_CPPCXTYPE_CXSMPL for C++
#endif

// NB: namespace mgOnGpu includes types which are defined in exactly the same way for CPU and GPU builds (see #318 and #725)
namespace mgOnGpu
{
  // Floating point type (for everything but color algebra #537): fptype
#if defined MGONGPU_FPTYPE_DOUBLE
  typedef double fptype; // double precision (8 bytes, fp64)
#elif defined MGONGPU_FPTYPE_FLOAT
  typedef float fptype;  // single precision (4 bytes, fp32)
#endif

  // Floating point type (for color algebra alone #537): fptype2
#if defined MGONGPU_FPTYPE2_DOUBLE
  typedef double fptype2; // double precision (8 bytes, fp64)
#elif defined MGONGPU_FPTYPE2_FLOAT
  typedef float fptype2; // single precision (4 bytes, fp32)
#endif

  // --- Mixed-precision stage types (3 independent precisions) ---
  // fptype       (== fptype_amp): wavefunctions, helicity amplitudes, ME, vertex
  //                               and polarization computation (MGONGPU_FPTYPE_*)
  // fptype_momenta (== fptype_denom): momenta storage and in-vertex denominators
  //                               (MGONGPU_FPTYPE_MOMENTA_*, default fptype)
  // fptype2      (== fptype_colour): color algebra alone (MGONGPU_FPTYPE2_*)

#if defined MGONGPU_FPTYPE_MOMENTA_DOUBLE
  typedef double fptype_momenta;
#elif defined MGONGPU_FPTYPE_MOMENTA_FLOAT
  typedef float fptype_momenta;
#else
  typedef fptype fptype_momenta;
#endif
  typedef fptype_momenta fptype_denom; // denominator precision == momenta precision
  typedef fptype fptype_amp;           // amplitudes/wavefunctions == fptype
  typedef fptype2 fptype_colour;       // color algebra == fptype2

  // Valid precision ordering: colour <= amp <= momenta (4=fp32, 8=fp64)
  static_assert( sizeof( fptype_colour ) <= sizeof( fptype_amp ), "colour precision must not exceed amp precision" );
  static_assert( sizeof( fptype_amp ) <= sizeof( fptype_momenta ), "amp precision must not exceed momenta precision" );

  // Maximum number of threads per block
  const int ntpbMAX = 1024; // NB: 512 is ok, but 1024 does fail with "too many resources requested for launch"

  // Alignment requirement for using reinterpret_cast with SIMD vectorized code
  constexpr int cppAlign = 64; // 64-byte i.e. 512-bit

  // Retrieve the compiler that was used to build this module
  inline std::string
  getCompiler()
  {
    std::stringstream out;
    // HIP version (HIPCC)
    // [Use __HIPCC__ instead of MGONGPUCPP_GPUIMPL here!]
    // [This tests if 'hipcc' was used even to build a .cc file, even if not necessarily 'nvcc -x cu' for a .cu file]
    // [Check 'hipcc -dM -E -x hip -I ../../src CPPProcess.cc | grep HIP']
#ifdef __HIPCC__
#if defined HIP_VERSION_MAJOR && defined HIP_VERSION_MINOR && defined HIP_VERSION_PATCH
    out << "hipcc " << HIP_VERSION_MAJOR << "." << HIP_VERSION_MINOR << "." << HIP_VERSION_PATCH;
#else
    out << "hipcc UNKNOWN";
#endif
    out << " (";
#endif
    // CUDA version (NVCC)
    // [Use __NVCC__ instead of MGONGPUCPP_GPUIMPL here!]
    // [This tests if 'nvcc' was used even to build a .cc file, even if not necessarily 'nvcc -x cu' for a .cu file]
    // [Check 'nvcc --compiler-options -dM -E dummy.c | grep CUDA': see https://stackoverflow.com/a/53713712]
#ifdef __NVCC__
#if defined __CUDACC_VER_MAJOR__ && defined __CUDACC_VER_MINOR__ && defined __CUDACC_VER_BUILD__
    out << "nvcc " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__;
#else
    out << "nvcc UNKNOWN";
#endif
    out << " (";
#endif
    // ICX version (either as CXX or as host compiler inside NVCC)
#if defined __INTEL_COMPILER
#error "icc is no longer supported: please use icx"
#elif defined __INTEL_LLVM_COMPILER // alternative: __INTEL_CLANG_COMPILER
    out << "icx " << __INTEL_LLVM_COMPILER;
#ifdef __NVCC__
    out << ", ";
#else
    out << " (";
#endif
#endif
    // CLANG version (either as CXX or as host compiler inside NVCC or inside ICX)
#if defined __clang__
#if defined __clang_major__ && defined __clang_minor__ && defined __clang_patchlevel__
#ifdef __APPLE__
    out << "Apple clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#else
    out << "clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#endif
#else
    out << "clang UNKNOWKN";
#endif
#else
    // GCC version (either as CXX or as host compiler inside NVCC)
#if defined __GNUC__ && defined __GNUC_MINOR__ && defined __GNUC_PATCHLEVEL__
    out << "gcc " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#else
    out << "gcc UNKNOWKN";
#endif
#endif
#if defined __HIPCC__ or defined __NVCC__ or defined __INTEL_LLVM_COMPILER
    out << ")";
#endif
    return out.str();
  }
}

// Expose typedefs outside the namespace
using mgOnGpu::fptype;
using mgOnGpu::fptype2;
using mgOnGpu::fptype_momenta;
using mgOnGpu::fptype_denom;
using mgOnGpu::fptype_amp;
using mgOnGpu::fptype_colour;

// Undefine ARM_NEON (hack for the 'scalar' backend on Apple silicon ARM)
#ifdef MGONGPU_NOARMNEON
#undef __ARM_NEON
#endif

// C++ SIMD vectorization width (this will be used to set neppV)
#if defined __AVX512VL__ && defined MGONGPU_PVW512 // "512z" AVX512 512-bit: 8 (DOUBLE) or 16 (FLOAT)
#ifdef MGONGPU_FPTYPE_DOUBLE
#define MGONGPU_CPPSIMD 8
#else
#define MGONGPU_CPPSIMD 16
#endif
#elif defined __AVX512VL__ // "512y" AVX512 256-bit: 4 (DOUBLE) or 8 (FLOAT) [gcc default]
#ifdef MGONGPU_FPTYPE_DOUBLE
#define MGONGPU_CPPSIMD 4
#else
#define MGONGPU_CPPSIMD 8
#endif
#elif defined __AVX2__ // "avx2" 256-bit: 4 (DOUBLE) or 8 (FLOAT) [clang default]
#ifdef MGONGPU_FPTYPE_DOUBLE
#define MGONGPU_CPPSIMD 4
#else
#define MGONGPU_CPPSIMD 8
#endif
#elif defined __SSE4_2__ // "sse4" SSE4.2 128-bit: 2 (DOUBLE) or 4 (FLOAT) [Power9 default]
#ifdef MGONGPU_FPTYPE_DOUBLE
#define MGONGPU_CPPSIMD 2
#else
#define MGONGPU_CPPSIMD 4
#endif
#elif defined __ARM_NEON // ARM NEON 128-bit: 2 (DOUBLE) or 4 (FLOAT) [ARM default]
#ifdef MGONGPU_FPTYPE_DOUBLE
#define MGONGPU_CPPSIMD 2
#else
#define MGONGPU_CPPSIMD 4
#endif
#else // "none" i.e. no SIMD
#undef MGONGPU_CPPSIMD
#endif

// macro for computing denom twice to fill rest of SIMD lane for rest
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_FLOAT and defined MGONGPU_FPTYPE_MOMENTA_DOUBLE
#define MGONGPU_SIMD_DENOM64 1
#endif

// No-op debug macros (nsight-based debugging is CUDA-only, unused here)
#define mgDebugDeclare() /*noop*/
#define mgDebugInitialise() /*noop*/
#define mgDebug( code, text ) /*noop*/
#define mgDebugFinalise() /*noop*/

// Define empty CUDA/HIP declaration specifiers for C++
#define __global__
#define __host__
#define __device__

// For SANITY CHECKS: check that neppR, neppM, neppV... are powers of two
inline constexpr bool
ispoweroftwo( int n )
{
  return ( n > 0 ) && !( n & ( n - 1 ) );
}

// Compiler version support (#96)
#if defined __clang__
#if( __clang_major__ < 11 )
#error Unsupported clang version: please use clang >= 11
#endif
#elif defined __GNUC__
#if( __GNUC__ < 9 ) || ( __GNUC__ == 9 && __GNUC_MINOR__ < 3 )
#error Unsupported gcc version: please gcc >= 9.3
#endif
#endif

#endif // MGONGPUCONFIG_H
