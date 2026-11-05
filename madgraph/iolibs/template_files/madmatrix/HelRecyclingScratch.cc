// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
//==========================================================================
// --hel_recycling: the per-thread scratch area of the recycled build (see
// HelRecyclingScratch.h).
//==========================================================================
#include "HelRecyclingScratch.h"

#include <cstdlib>
#include <cstring>
#include <new>

namespace madmatrix
{
  namespace
  {
    struct HrScratch
    {
      void* data = nullptr;
      std::size_t size = 0;
      const void* owner = nullptr;
      ~HrScratch() { std::free( data ); }
    };
  }

  void*
  hr_thread_scratch( std::size_t bytes, std::size_t zeroed, const void* owner )
  {
    static thread_local HrScratch scratch;
    if( bytes > scratch.size )
    {
      std::free( scratch.data );
      const std::size_t size = ( bytes + 63 ) / 64 * 64; // aligned_alloc wants a multiple of the alignment
      scratch.data = std::aligned_alloc( 64, size );
      if( !scratch.data ) throw std::bad_alloc();
      scratch.size = size;
      scratch.owner = nullptr; // what was there is gone
    }
    if( scratch.owner != owner )
    {
      std::memset( scratch.data, 0, zeroed < scratch.size ? zeroed : scratch.size );
      scratch.owner = owner;
    }
    return scratch.data;
  }
}
