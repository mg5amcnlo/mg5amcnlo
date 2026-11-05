// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: F.Stloukal (May 2026) for the MadGraph7 MadMatrix.
// Further modified by: F.Stloukal  

#ifndef BASICOPS_H
#define BASICOPS_H

#include <limits>

#if defined( __CUDACC__ )
#include <cuda_runtime.h>
#endif


namespace MG_ARITHM{

#if defined( __CUDACC__ )
#define __mgDWA_inline__ __forceinline__
#elif defined( _MSC_VER )
#define __mgDWA_inline__ __forceinline
#elif defined( __GNUC__ ) || defined( __clang__ )
#define __mgDWA_inline__ __attribute__( ( always_inline ) ) inline
#else
#define __mgDWA_inline__ inline
#endif

#if defined( __CUDACC__ )
#define __cuda_callable__ \
__device__             \
__host__
#else
#define __cuda_callable__
#endif

#ifdef __CADNA__
template <typename T>
concept GoodFloatType = std::is_same_v<T, double_st> || std::is_same_v<T, float_st>;
template <typename T>
concept SingleFloatType = std::is_same_v<T, float_st>;
template <typename T>
concept DoubleFloatType = std::is_same_v<T, double_st>;
#else
template <typename T>
concept GoodFloatType = std::is_same_v<T, double> || std::is_same_v<T, float>;
template <typename T>
concept SingleFloatType = std::is_same_v<T, float>;
template <typename T>
concept DoubleFloatType = std::is_same_v<T, double>;
#endif


template <GoodFloatType T>
__cuda_callable__
static constexpr __mgDWA_inline__ T
add_rn( const T x, const T y )
{
#if defined __CUDA_ARCH__
   if constexpr( std::is_same_v< T, double > ) {
      return __dadd_rn( x, y );
   }
   else if constexpr( std::is_same_v< T, float > ) {
      return __fadd_rn( x, y );
   }
#else
   return x + y;
#endif
}

template<GoodFloatType T>
__cuda_callable__
static constexpr __mgDWA_inline__ T
mul_rn( const T x, const T y )
{
#if defined __CUDA_ARCH__
   if constexpr( std::is_same_v< T, double > ) {
      return __dmul_rn( x, y );
   }
   else if constexpr( std::is_same_v< T, float > ) {
      return __fmul_rn( x, y );
   }
#else
   return x * y;
#endif
}

template<GoodFloatType T>
__cuda_callable__
static constexpr __mgDWA_inline__ T
div_rn( const T x, const T y )
{
#if defined __CUDA_ARCH__
   if constexpr( std::is_same_v< T, double > ) {
      return __ddiv_rn( x, y );
   }
   else if constexpr( std::is_same_v< T, float > ) {
      return __fdiv_rn( x, y );
   }
#else
   return x / y;
#endif
}

template<GoodFloatType T>
__cuda_callable__
static constexpr __mgDWA_inline__ T
fma_rn( const T x, const T y, const T z )
{
#if defined __CUDA_ARCH__
   if constexpr( std::is_same_v< T, double > ) {
      return __fma_rn( x, y, z );
   }
   else if constexpr( std::is_same_v< T, float > ) {
      return __fmaf_rn( x, y, z );
   }
#else
   #ifdef FP_FAST_FMA
   return fma( x, y, z );
   #else
   printf( "There is no FMA. Do not enable FP_FAST_FMA e.g. with -mfma." );
   return std::numeric_limits< T >::quiet_NaN();
   #endif
#endif
}

}

#endif  //BASICOPS_H
