// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Dec 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2021-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MemoryAccessWeights_H
#define MemoryAccessWeights_H 1

#include "mgOnGpuConfig.h"

#include "MemoryAccessHelpers.h"

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //----------------------------------------------------------------------------

  // A class describing the internal layout of memory buffers for weights
  // This implementation uses a plain ARRAY[nevt]
  // [If many implementations are used, a suffix _ARRAYv1 should be appended to the class name]
  class MemoryAccessWeightsBase //_ARRAYv1
  {
  private:

    friend class MemoryAccessHelper<MemoryAccessWeightsBase>;
    friend class KernelAccessHelper<MemoryAccessWeightsBase, true>;
    friend class KernelAccessHelper<MemoryAccessWeightsBase, false>;

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
  class MemoryAccessWeights : public MemoryAccessWeightsBase
  {
  public:

    // Locate an event record (output) in a memory buffer (input) from the given event number (input)
    // [Signature (non-const) ===> fptype* ieventAccessRecord( fptype* buffer, const int ievt ) <===]
    static constexpr auto ieventAccessRecord = MemoryAccessHelper<MemoryAccessWeightsBase>::ieventAccessRecord;

    // Locate an event record (output) in a memory buffer (input) from the given event number (input)
    // [Signature (const) ===> const fptype* ieventAccessRecordConst( const fptype* buffer, const int ievt ) <===]
    static constexpr auto ieventAccessRecordConst = MemoryAccessHelper<MemoryAccessWeightsBase>::ieventAccessRecordConst;

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (non-const) ===> fptype& decodeRecord( fptype* buffer ) <===]
    static constexpr auto decodeRecord = MemoryAccessHelper<MemoryAccessWeightsBase>::decodeRecord;

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (const) ===> const fptype& decodeRecordConst( const fptype* buffer ) <===]
    static constexpr auto decodeRecordConst =
      MemoryAccessHelper<MemoryAccessWeightsBase>::template decodeRecordConst<>;

    // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
    // [Signature (non-const) ===> fptype& ieventAccess( fptype* buffer, const ievt ) <===]
    static constexpr auto ieventAccess =
      MemoryAccessHelper<MemoryAccessWeightsBase>::template ieventAccessField<>;

    // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
    // [Signature (const) ===> const fptype& ieventAccessConst( const fptype* buffer, const ievt ) <===]
    static constexpr auto ieventAccessConst =
      MemoryAccessHelper<MemoryAccessWeightsBase>::template ieventAccessFieldConst<>;
  };

  //----------------------------------------------------------------------------

  // A class providing access to memory buffers for a given event, based on implicit kernel rules
  // Its methods use the KernelAccessHelper template - note the use of the template keyword in template function instantiations
  template<bool onDevice>
  class KernelAccessWeights
  {
  public:

    /*
  // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
  // [Signature (non-const) ===> fptype& kernelAccess( fptype* buffer ) <===]
  // FINAL IMPLEMENTATION FOR CUDA 11.4
  static constexpr auto kernelAccess =
    KernelAccessHelper<MemoryAccessWeightsBase, onDevice>::template kernelAccessField<>;
    */

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (non-const) ===> fptype& kernelAccess( fptype* buffer ) <===]
    // TEMPORARY HACK FOR CUDA 11.1
    static __host__ __device__ inline fptype&
    kernelAccess( fptype* buffer )
    {
      return KernelAccessHelper<MemoryAccessWeightsBase, onDevice>::template kernelAccessField<>( buffer );
    }

    /*
  // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
  // [Signature (const) ===> const fptype& kernelAccessConst( const fptype* buffer ) <===]
  // FINAL IMPLEMENTATION FOR CUDA 11.4
  static constexpr auto kernelAccessConst =
    KernelAccessHelper<MemoryAccessWeightsBase, onDevice>::template kernelAccessFieldConst<>;
    */

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (const) ===> const fptype& kernelAccessConst( const fptype* buffer ) <===]
    // TEMPORARY HACK FOR CUDA 11.1
    static __host__ __device__ inline const fptype&
    kernelAccessConst( const fptype* buffer )
    {
      return KernelAccessHelper<MemoryAccessWeightsBase, onDevice>::template kernelAccessFieldConst<>( buffer );
    }
  };

  //----------------------------------------------------------------------------

  typedef KernelAccessWeights<false> HostAccessWeights;
  typedef KernelAccessWeights<true> DeviceAccessWeights;

  //----------------------------------------------------------------------------

} // end namespace mg5amcGpu/mg5amcCpu

#endif // MemoryAccessWeights_H
