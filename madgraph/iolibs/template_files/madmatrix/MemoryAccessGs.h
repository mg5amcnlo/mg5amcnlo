// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Dec 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: S. Roiser, A. Valassi (2021-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MemoryAccessGs_H
#define MemoryAccessGs_H 1

#include "mgOnGpuConfig.h"

#include "MemoryAccessHelpers.h"
#include "MemoryAccessVectors.h"
#include "MemoryBuffers.h" // for HostBufferMatrixElements::isaligned

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //----------------------------------------------------------------------------

  // A class describing the internal layout of memory buffers for Gs
  // This implementation uses a plain ARRAY[nevt]
  // [If many implementations are used, a suffix _ARRAYv1 should be appended to the class name]
  class MemoryAccessGsBase //_ARRAYv1
  {
  private:

    friend class MemoryAccessHelper<MemoryAccessGsBase>;
    friend class KernelAccessHelper<MemoryAccessGsBase, true>;
    friend class KernelAccessHelper<MemoryAccessGsBase, false>;

    //--------------------------------------------------------------------------
    // NB all KernelLaunchers assume that memory access can be decomposed as "accessField = decodeRecord( accessRecord )"
    // (in other words: first locate the event record for a given event, then locate an element in that record)
    //--------------------------------------------------------------------------

    // Locate an event record (output) in a memory buffer (input) from the given event number (input)
    // [Signature (non-const) ===> fptype* ieventAccessRecord( fptype* buffer, const int ievt ) <===]
    static __host__ __device__ inline fptype*
    ieventAccessRecord( fptype* buffer,
                        const int ievt )
    {
      return &( buffer[ievt] ); // ARRAY[nevt]
    }

    //--------------------------------------------------------------------------

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (non-const) ===> fptype& decodeRecord( fptype* buffer, Ts... args ) <===]
    // [NB: expand variadic template "Ts... args" to empty and rename "Field" as empty]
    static __host__ __device__ inline fptype&
    decodeRecord( fptype* buffer )
    {
      constexpr int ievt = 0;
      return buffer[ievt]; // ARRAY[nevt]
    }
  };

  //----------------------------------------------------------------------------

  // A class providing access to memory buffers for a given event, based on explicit event numbers
  // Its methods use the MemoryAccessHelper templates - note the use of the template keyword in template function instantiations
  class MemoryAccessGs : public MemoryAccessGsBase
  {
  public:

    // Locate an event record (output) in a memory buffer (input) from the given event number (input)
    // [Signature (non-const) ===> fptype* ieventAccessRecord( fptype* buffer, const int ievt ) <===]
    static constexpr auto ieventAccessRecord = MemoryAccessHelper<MemoryAccessGsBase>::ieventAccessRecord;

    // Locate an event record (output) in a memory buffer (input) from the given event number (input)
    // [Signature (const) ===> const fptype* ieventAccessRecordConst( const fptype* buffer, const int ievt ) <===]
    static constexpr auto ieventAccessRecordConst = MemoryAccessHelper<MemoryAccessGsBase>::ieventAccessRecordConst;

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (non-const) ===> fptype& decodeRecord( fptype* buffer ) <===]
    static constexpr auto decodeRecord = MemoryAccessHelper<MemoryAccessGsBase>::decodeRecord;

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (const) ===> const fptype& decodeRecordConst( const fptype* buffer ) <===]
    static constexpr auto decodeRecordConst =
      MemoryAccessHelper<MemoryAccessGsBase>::template decodeRecordConst<>;

    // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
    // [Signature (non-const) ===> fptype& ieventAccess( fptype* buffer, const ievt ) <===]
    static constexpr auto ieventAccess =
      MemoryAccessHelper<MemoryAccessGsBase>::template ieventAccessField<>;

    // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
    // [Signature (const) ===> const fptype& ieventAccessConst( const fptype* buffer, const ievt ) <===]
    static constexpr auto ieventAccessConst =
      MemoryAccessHelper<MemoryAccessGsBase>::template ieventAccessFieldConst<>;
  };

  //----------------------------------------------------------------------------

  // A class providing access to memory buffers for a given event, based on implicit kernel rules
  // Its methods use the KernelAccessHelper template - note the use of the template keyword in template function instantiations
  template<bool onDevice>
  class KernelAccessGs
  {
  public:

    // Expose selected functions from MemoryAccessGs
    static constexpr auto ieventAccessRecord = MemoryAccessGs::ieventAccessRecord;

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (non-const, SCALAR) ===> fptype& kernelAccess( fptype* buffer ) <===]
    static constexpr auto kernelAccess_s =
      KernelAccessHelper<MemoryAccessGsBase, onDevice>::template kernelAccessField<>; // requires cuda 11.4

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal)
    // [Signature (non-const, SCALAR OR VECTOR) ===> fptype_sv& kernelAccess( fptype* buffer ) <===]
    static __host__ __device__ inline fptype_sv&
    kernelAccess( fptype* buffer )
    {
      fptype& out = kernelAccess_s( buffer );
#ifndef MGONGPU_CPPSIMD
      return out;
#else
      // NB: derived from MemoryAccessMomenta, restricting the implementation to contiguous aligned arrays (#435)
      static_assert( mg5amcCpu::HostBufferGs::isaligned() ); // ASSUME ALIGNED ARRAYS (reinterpret_cast will segfault otherwise!)
      //assert( (size_t)( buffer ) % mgOnGpu::cppAlign == 0 ); // ASSUME ALIGNED ARRAYS (reinterpret_cast will segfault otherwise!)
      return mg5amcCpu::fptypevFromAlignedArray( out ); // SIMD bulk load of neppV, use reinterpret_cast
#endif
    }

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal)
    // [Signature (SCALAR OR VECTOR) ===> fptype_sv* kernelAccess( fptype* buffer ) <===]
    static __host__ __device__ inline fptype_sv*
    kernelAccessP( fptype* buffer )
    {
      return reinterpret_cast<fptype_sv*>( buffer );
    }

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (const, SCALAR) ===> const fptype& kernelAccessConst( const fptype* buffer ) <===]
    static constexpr auto kernelAccessConst_s =
      KernelAccessHelper<MemoryAccessGsBase, onDevice>::template kernelAccessFieldConst<>; // requires cuda 11.4

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal)
    // [Signature (const, SCALAR OR VECTOR) ===> const fptype_sv& kernelAccess( const fptype* buffer ) <===]
    static __host__ __device__ inline const fptype_sv&
    kernelAccessConst( const fptype* buffer )
    {
      const fptype& out = kernelAccessConst_s( buffer );
#ifndef MGONGPU_CPPSIMD
      return out;
#else
      // NB: derived from MemoryAccessMomenta, restricting the implementation to contiguous aligned arrays (#435)
      static_assert( mg5amcCpu::HostBufferGs::isaligned() ); // ASSUME ALIGNED ARRAYS (reinterpret_cast will segfault otherwise!)
      //assert( (size_t)( buffer ) % mgOnGpu::cppAlign == 0 ); // ASSUME ALIGNED ARRAYS (reinterpret_cast will segfault otherwise!)
      return mg5amcCpu::fptypevFromAlignedArray( out ); // SIMD bulk load of neppV, use reinterpret_cast
#endif
    }
  };

  //----------------------------------------------------------------------------

  typedef KernelAccessGs<false> HostAccessGs;
  typedef KernelAccessGs<true> DeviceAccessGs;

  //----------------------------------------------------------------------------

} // end namespace mg5amcGpu/mg5amcCpu

#endif // MemoryAccessGs_H
