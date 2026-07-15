// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Dec 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2021-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MemoryAccessHelpers_H
#define MemoryAccessHelpers_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuFptypes.h"

//----------------------------------------------------------------------------

// A templated helper class that includes the boilerplate code for MemoryAccess classes
template<class T, typename FT = fptype>
class MemoryAccessHelper
{
public:

  //--------------------------------------------------------------------------

  // Locate an event record (output) in a memory buffer (input) from the given event number (input)
  // [Signature (non-const) ===> fptype* ieventAccessRecord( fptype* buffer, const int ievt ) <===]
  static constexpr auto ieventAccessRecord = T::ieventAccessRecord;

  //--------------------------------------------------------------------------

  // Locate an event record (output) in a memory buffer (input) from the given event number (input)
  // [Signature (const) ===> const fptype* ieventAccessRecordConst( const fptype* buffer, const int ievt ) <===]
  static __host__ __device__ inline const FT*
  ieventAccessRecordConst( const FT* buffer,
                           const int ievt )
  {
    return ieventAccessRecord( const_cast<FT*>( buffer ), ievt );
  }

  //--------------------------------------------------------------------------

  // Locate a field (output) of an event record (input) from the given field indexes (input)
  // [Signature (non-const) ===> fptype& decodeRecord( fptype* buffer, Ts... args ) <===]
  static constexpr auto decodeRecord = T::decodeRecord;

  //--------------------------------------------------------------------------

  // Locate a field (output) of an event record (input) from the given field indexes (input)
  // [Signature (const) ===> const fptype& decodeRecordConst( const fptype* buffer, Ts... args ) <===]
  template<class... Ts>
  static __host__ __device__ inline const FT&
  decodeRecordConst( const FT* buffer,
                     Ts... args ) // variadic template
  {
    return T::decodeRecord( const_cast<FT*>( buffer ), args... );
  }

  //--------------------------------------------------------------------------

  // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
  // [Signature (non-const) ===> fptype& ieventAccessField( fptype* buffer, const ievt, Ts... args ) <===]
  template<class... Ts>
  static __host__ __device__ inline FT&
  ieventAccessField( FT* buffer,
                     const int ievt,
                     Ts... args ) // variadic template
  {
    // NB all KernelLaunchers assume that memory access can be decomposed as "accessField = decodeRecord( accessRecord )"
    // (in other words: first locate the event record for a given event, then locate an element in that record)
    return T::decodeRecord( T::ieventAccessRecord( buffer, ievt ), args... );
  }

  //--------------------------------------------------------------------------

  // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
  // [Signature (const) ===> const fptype& ieventAccessFieldConst( const fptype* buffer, const ievt, Ts... args ) <===]
  template<class... Ts>
  static __host__ __device__ inline const FT&
  ieventAccessFieldConst( const FT* buffer,
                          const int ievt,
                          Ts... args ) // variadic template
  {
    return ieventAccessField( const_cast<FT*>( buffer ), ievt, args... );
  }
};

//----------------------------------------------------------------------------

// A templated helper class that includes the boilerplate code for KernelAccess classes
template<class T, bool onDevice, typename FT = fptype>
class KernelAccessHelper : public MemoryAccessHelper<T, FT>
{
public:

  //--------------------------------------------------------------------------

  // Locate an event record (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal)
  // [Signature (non-const) ===> fptype* kernelAccessRecord( fptype* buffer ) <===]
  static __host__ __device__ inline FT*
  kernelAccessRecord( FT* buffer )
  {
    if constexpr( !onDevice ) // requires c++17 also in CUDA (#333)
    {
      // FIXME #436: clarify that buffer includes all events on device, and only the record for an event subset on host!
      // FIXME #436: am I not assuming that the following line is always identical to buffer for all access classes T?
      return T::ieventAccessRecord( buffer, 0 );
    }
    else
    {
#ifdef MGONGPUCPP_GPUIMPL
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread) in grid
      //printf( "kernelAccessRecord: ievt=%d threadId=%d\n", ievt, threadIdx.x );
      return T::ieventAccessRecord( buffer, ievt ); // NB fptype and fptype_sv coincide for CUDA
#else
      throw std::runtime_error( "kernelAccessRecord on device is only implemented in CUDA" );
#endif
    }
  }

  //--------------------------------------------------------------------------

  // Locate an event record (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal)
  // [Signature (const) ===> const fptype* kernelAccessRecordConst( const fptype* buffer ) <===]
  static __host__ __device__ inline const FT*
  kernelAccessRecordConst( const FT* buffer )
  {
    return kernelAccessRecord( const_cast<FT*>( buffer ) );
  }

  //--------------------------------------------------------------------------

  // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
  // [Signature (non-const) ===> fptype& kernelAccessField( fptype* buffer, Ts... args ) <===]
  template<class... Ts>
  static __host__ __device__ inline FT&
  kernelAccessField( FT* buffer,
                     Ts... args ) // variadic template
  {
    // NB all KernelLaunchers assume that memory access can be decomposed as "accessField = decodeRecord( accessRecord )"
    // (in other words: first locate the event record for a given event, then locate an element in that record)
    return T::decodeRecord( kernelAccessRecord( buffer ), args... );
  }

  //--------------------------------------------------------------------------

  // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
  // [Signature (const) ===> const fptype& kernelAccessFieldConst( const fptype* buffer, Ts... args ) <===]
  template<class... Ts>
  static __host__ __device__ inline const FT&
  kernelAccessFieldConst( const FT* buffer,
                          Ts... args ) // variadic template
  {
    return kernelAccessField( const_cast<FT*>( buffer ), args... );
  }

  //--------------------------------------------------------------------------
};

#endif // MemoryAccessHelpers_H
