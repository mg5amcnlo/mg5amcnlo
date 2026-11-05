// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2022) for the MadGraph7 CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MemoryAccessVectors_H
#define MemoryAccessVectors_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuVectors.h"

namespace madmatrix // this is only needed for CPU SIMD vectorization
{

  //--------------------------------------------------------------------------

  // Cast one non-const fptype_v reference (one vector of neppV fptype values) from one non-const fptype reference (#435),
  // assuming that "pointer(evt#0)+1" indicates "pointer(evt#1)", and that the arrays are aligned
  inline fptype_v& fptypevFromAlignedArray( fptype& ref )
  {
    return *reinterpret_cast<fptype_sv*>( &ref );
  }

  inline uint_v& uintvFromAlignedArray( unsigned int& ref )
  {
    return *reinterpret_cast<uint_sv*>( &ref );
  }

  // Cast one const fptype_v reference (one vector of neppV fptype values) from one const fptype reference,
  // assuming that "pointer(evt#0)+1" indicates "pointer(evt#1)", and that the arrays are aligned
  inline const fptype_v& fptypevFromAlignedArray( const fptype& ref )
  {
    return *reinterpret_cast<const fptype_sv*>( &ref );
  }

  inline const uint_v& uintvFromAlignedArray( const unsigned int& ref )
  {
    return *reinterpret_cast<const uint_sv*>( &ref );
  }

  // Build one fptype_v (one vector of neppV fptype values) from one fptype reference,
  // assuming that "pointer(evt#0)+1" indicates "pointer(evt#1)", but that the arrays are not aligned
  inline fptype_v fptypevFromUnalignedArray( const fptype& ref )
  {
    return fptype_v{ *( &ref ), // explicit initialization of all array elements (2)
                     *( &ref + 1 ) };
  }

#ifdef MGONGPU_SIMD_DENOM64
  // assume load from one page return high and low
  inline fptype_denom_sv momvFromContiguousArray( const fptype_momenta& ref )
  {
    fptype_denom_sv out;
    __builtin_memcpy( &out, &ref, sizeof( out ) );
    return out;
  }
#endif

  // Build one fptype_v (one vector of neppV fptype values) from one fptype reference,
  // with no a priori assumption on how the input fptype array should be decoded
  template<typename Functor>
  inline fptype_v fptypevFromArbitraryArray( Functor decoderIeppv )
  {
    return fptype_v{ decoderIeppv( 0 ), // explicit initialization of all array elements (2)
                     decoderIeppv( 1 ) };
  }

  //--------------------------------------------------------------------------

} // end namespace

#endif // MemoryAccessVectors_H
