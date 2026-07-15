// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2022) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MemoryAccessVectors_H
#define MemoryAccessVectors_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuVectors.h"

#ifndef MGONGPUCPP_GPUIMPL
namespace mg5amcCpu // this is only needed for CPU SIMD vectorization
{

#ifdef MGONGPU_CPPSIMD
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
#if MGONGPU_CPPSIMD == 2
    return fptype_v{ *( &ref ), // explicit initialization of all array elements (2)
                     *( &ref + 1 ) };
#elif MGONGPU_CPPSIMD == 4
    return fptype_v{ *( &ref ), // explicit initialization of all array elements (4)
                     *( &ref + 1 ),
                     *( &ref + 2 ),
                     *( &ref + 3 ) };
#elif MGONGPU_CPPSIMD == 8
    return fptype_v{ *( &ref ), // explicit initialization of all array elements (8)
                     *( &ref + 1 ),
                     *( &ref + 2 ),
                     *( &ref + 3 ),
                     *( &ref + 4 ),
                     *( &ref + 5 ),
                     *( &ref + 6 ),
                     *( &ref + 7 ) };
#elif MGONGPU_CPPSIMD == 16
    return fptype_v{ *( &ref ), // explicit initialization of all array elements (16)
                     *( &ref + 1 ),
                     *( &ref + 2 ),
                     *( &ref + 3 ),
                     *( &ref + 4 ),
                     *( &ref + 5 ),
                     *( &ref + 6 ),
                     *( &ref + 7 ),
                     *( &ref + 8 ),
                     *( &ref + 9 ),
                     *( &ref + 10 ),
                     *( &ref + 11 ),
                     *( &ref + 12 ),
                     *( &ref + 13 ),
                     *( &ref + 14 ),
                     *( &ref + 15 ) };
#else
#error Internal error! Unknown MGONGPU_CPPSIMD value
#endif
  }

  // Build one fptype_v (one vector of neppV fptype values) from one fptype reference,
  // with no a priori assumption on how the input fptype array should be decoded
  template<typename Functor>
  inline fptype_v fptypevFromArbitraryArray( Functor decoderIeppv )
  {
#if MGONGPU_CPPSIMD == 2
    return fptype_v{ decoderIeppv( 0 ), // explicit initialization of all array elements (2)
                     decoderIeppv( 1 ) };
#elif MGONGPU_CPPSIMD == 4
    return fptype_v{ decoderIeppv( 0 ), // explicit initialization of all array elements (4)
                     decoderIeppv( 1 ),
                     decoderIeppv( 2 ),
                     decoderIeppv( 3 ) };
#elif MGONGPU_CPPSIMD == 8
    return fptype_v{ decoderIeppv( 0 ), // explicit initialization of all array elements (8)
                     decoderIeppv( 1 ),
                     decoderIeppv( 2 ),
                     decoderIeppv( 3 ),
                     decoderIeppv( 4 ),
                     decoderIeppv( 5 ),
                     decoderIeppv( 6 ),
                     decoderIeppv( 7 ) };
#elif MGONGPU_CPPSIMD == 16
    return fptype_v{ decoderIeppv( 0 ), // explicit initialization of all array elements (16)
                     decoderIeppv( 1 ),
                     decoderIeppv( 2 ),
                     decoderIeppv( 3 ),
                     decoderIeppv( 4 ),
                     decoderIeppv( 5 ),
                     decoderIeppv( 6 ),
                     decoderIeppv( 7 ),
                     decoderIeppv( 8 ),
                     decoderIeppv( 9 ),
                     decoderIeppv( 10 ),
                     decoderIeppv( 11 ),
                     decoderIeppv( 12 ),
                     decoderIeppv( 13 ),
                     decoderIeppv( 14 ),
                     decoderIeppv( 15 ) };
#else
#error Internal error! Unknown MGONGPU_CPPSIMD value
#endif
  }

  //--------------------------------------------------------------------------
#endif

} // end namespace
#endif

#endif // MemoryAccessVectors_H
