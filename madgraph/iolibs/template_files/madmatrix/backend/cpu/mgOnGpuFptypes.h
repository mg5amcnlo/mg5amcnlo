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


  //==========================================================================


  //------------------------------
  // Floating point types - C++
  //------------------------------

  template<typename FP>
  inline const FP&
  fpmax( const FP& a, const FP& b )
  {
    return std::max( a, b );
  }

  template<typename FP>
  inline const FP&
  fpmin( const FP& a, const FP& b )
  {
    return std::min( a, b );
  }

  // Non-template overloads
  inline const fptype_amp&
  fpmax( const fptype_amp& a, const fptype_amp& b ) { return std::max( a, b ); }

  inline const fptype_amp&
  fpmin( const fptype_amp& a, const fptype_amp& b ) { return std::min( a, b ); }

  template<typename FP>
  inline FP
  fpsqrt( FP f )
  {
    return std::sqrt( f );
  }

  template<typename FP>
  inline bool
  fpsignbit( FP f ) { return std::signbit( f ); }

  //==========================================================================

} // end namespace madmatrix

#endif // MGONGPUFPTYPES_H
