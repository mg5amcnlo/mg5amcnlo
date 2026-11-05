// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2022) for the MadGraph7 CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MGONGPUFPTYPES_H
#define MGONGPUFPTYPES_H 1

#include "mgOnGpuConfig.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

//One namespace. Split ber backend.
namespace madmatrix
{
  //==========================================================================


  //------------------------------
  // Floating point types - Cuda
  //------------------------------

  /*
  inline __host__ __device__ fptype
  fpmax( const fptype& a, const fptype& b )
  {
    return max( a, b );
  }

  inline __host__ __device__ fptype
  fpmin( const fptype& a, const fptype& b )
  {
    return min( a, b );
  }
  */

  template<typename FP>
  inline __host__ __device__ const FP&
  fpmax( const FP& a, const FP& b )
  {
    return ( ( b < a ) ? a : b );
  }

  template<typename FP>
  inline __host__ __device__ const FP&
  fpmin( const FP& a, const FP& b )
  {
    return ( ( a < b ) ? a : b );
  }

  // Non-template overloads 
  inline __host__ __device__ const fptype_amp&
  fpmax( const fptype_amp& a, const fptype_amp& b ) { return ( ( b < a ) ? a : b ); }

  inline __host__ __device__ const fptype_amp&
  fpmin( const fptype_amp& a, const fptype_amp& b ) { return ( ( a < b ) ? a : b ); }

  template<typename FP>
  inline __host__ __device__ FP
  fpsqrt( FP f )
  {
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html
    if constexpr( std::is_same_v<std::remove_cv_t<FP>, float> )
      return sqrtf( f );
    else
      return sqrt( f );
  }

  template<typename FP>
  inline __host__ __device__ bool
  fpsignbit( FP f ) { return signbit( f ); }

  //==========================================================================


  //==========================================================================

} // end namespace madmatrix

#endif // MGONGPUFPTYPES_H
