// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2022) for the MadGraph7 CUDACPP plugin.
// Further modified by: A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MemoryAccessAmplitudes_H
#define MemoryAccessAmplitudes_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuCxtypes.h"

#include "MemoryAccessHelpers.h"

#define MGONGPU_TRIVIAL_AMPLITUDES 1

//One namespace. Split ber backend.
namespace madmatrix
{
  //----------------------------------------------------------------------------

#ifndef MGONGPU_TRIVIAL_AMPLITUDES

  // A class describing the internal layout of memory buffers for amplitudes
  // This implementation uses an AOSOA[npagA][nx2][neppA] where nevt=npagA*neppA
  // [If many implementations are used, a suffix _AOSOAv1 should be appended to the class name]
  class MemoryAccessAmplitudesBase //_AOSOAv1
  {
  public:

    // Number of Events Per Page in the amplitude AOSOA memory buffer layout
    static constexpr int neppA = 1; // AOS (just a test...)

  private:

    friend class MemoryAccessHelper<MemoryAccessAmplitudesBase, fptype_amp>;
    friend class KernelAccessHelper<MemoryAccessAmplitudesBase, true, fptype_amp>;
    friend class KernelAccessHelper<MemoryAccessAmplitudesBase, false, fptype_amp>;

    // The number of floating point components of a complex number
    static constexpr int nx2 = mgOnGpu::nx2;

    //--------------------------------------------------------------------------
    // NB all KernelLaunchers assume that memory access can be decomposed as "accessField = decodeRecord( accessRecord )"
    // (in other words: first locate the event record for a given event, then locate an element in that record)
    //--------------------------------------------------------------------------

    // Locate an event record (output) in a memory buffer (input) from the given event number (input)
    // [Signature (non-const) ===> fptype_amp* ieventAccessRecord( fptype_amp* buffer, const int ievt ) <===]
    static __host__ __device__ inline fptype_amp*
    ieventAccessRecord( fptype_amp* buffer,
                        const int ievt )
    {
      const int ipagA = ievt / neppA; // #event "A-page"
      const int ieppA = ievt % neppA; // #event in the current event A-page
      constexpr int ix2 = 0;
      return &( buffer[ipagA * nx2 * neppA + ix2 * neppA + ieppA] ); // AOSOA[ipagA][ix2][ieppA]
    }

    //--------------------------------------------------------------------------

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (non-const) ===> fptype_amp& decodeRecord( fptype_amp* buffer, Ts... args ) <===]
    // [NB: expand variadic template "Ts... args" to "const int ix2" and rename "Field" as "Ix2"]
    static __host__ __device__ inline fptype_amp&
    decodeRecord( fptype_amp* buffer,
                  const int ix2 )
    {
      constexpr int ipagA = 0;
      constexpr int ieppA = 0;
      return buffer[ipagA * nx2 * neppA + ix2 * neppA + ieppA]; // AOSOA[ipagA][ix2][ieppA]
    }
  };

  //----------------------------------------------------------------------------

  // A class providing access to memory buffers for a given event, based on explicit event numbers
  // Its methods use the MemoryAccessHelper templates - note the use of the template keyword in template function instantiations
  class MemoryAccessAmplitudes : public MemoryAccessAmplitudesBase
  {
  public:

    // Locate an event record (output) in a memory buffer (input) from the given event number (input)
    // [Signature (non-const) ===> fptype_amp* ieventAccessRecord( fptype_amp* buffer, const int ievt ) <===]
    static constexpr auto ieventAccessRecord = MemoryAccessHelper<MemoryAccessAmplitudesBase, fptype_amp>::ieventAccessRecord;

    // Locate an event record (output) in a memory buffer (input) from the given event number (input)
    // [Signature (const) ===> const fptype_amp* ieventAccessRecordConst( const fptype_amp* buffer, const int ievt ) <===]
    static constexpr auto ieventAccessRecordConst = MemoryAccessHelper<MemoryAccessAmplitudesBase, fptype_amp>::ieventAccessRecordConst;

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (non-const) ===> fptype_amp& decodeRecord( fptype_amp* buffer, const int ix2 ) <===]
    static constexpr auto decodeRecordIx2 = MemoryAccessHelper<MemoryAccessAmplitudesBase, fptype_amp>::decodeRecord;

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (const) ===> const fptype_amp& decodeRecordConst( const fptype_amp* buffer, const int ix2 ) <===]
    static constexpr auto decodeRecordIx2Const =
      MemoryAccessHelper<MemoryAccessAmplitudesBase, fptype_amp>::template decodeRecordConst<int>;

    // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
    // [Signature (non-const) ===> fptype_amp& ieventAccessIx2( fptype_amp* buffer, const ievt, const int ix2 ) <===]
    static constexpr auto ieventAccessIx2 =
      MemoryAccessHelper<MemoryAccessAmplitudesBase, fptype_amp>::template ieventAccessField<int>;

    // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
    // [Signature (const) ===> const fptype_amp& ieventAccessIx2Const( const fptype_amp* buffer, const ievt, const int ix2 ) <===]
    static constexpr auto ieventAccessIx2Const =
      MemoryAccessHelper<MemoryAccessAmplitudesBase, fptype_amp>::template ieventAccessFieldConst<int>;
  };

#endif // #ifndef MGONGPU_TRIVIAL_AMPLITUDES

  //----------------------------------------------------------------------------

  // A class providing access to memory buffers for a given event, based on implicit kernel rules
  // Its methods use the KernelAccessHelper template - note the use of the template keyword in template function instantiations
  template<bool onDevice>
  class KernelAccessAmplitudes
  {
  public:

#ifndef MGONGPU_TRIVIAL_AMPLITUDES

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (non-const) ===> fptype_amp& kernelAccessIx2( fptype_amp* buffer, const int ix2 ) <===]
    static constexpr auto kernelAccessIx2 =
      KernelAccessHelper<MemoryAccessAmplitudesBase, onDevice, fptype_amp>::template kernelAccessField<int>;

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (const) ===> const fptype_amp& kernelAccessIx2Const( const fptype_amp* buffer, const int ix2 ) <===]
    static constexpr auto kernelAccessIx2Const =
      KernelAccessHelper<MemoryAccessAmplitudesBase, onDevice, fptype_amp>::template kernelAccessFieldConst<int>;

#else

    static __host__ __device__ inline cxtype_amp_sv*
    kernelAccess( fptype_amp* buffer )
    {
      return reinterpret_cast<cxtype_amp_sv*>( buffer );
    }

    static __host__ __device__ inline const cxtype_amp_sv*
    kernelAccessConst( const fptype_amp* buffer )
    {
      return reinterpret_cast<const cxtype_amp_sv*>( buffer );
    }

#endif // #ifndef MGONGPU_TRIVIAL_AMPLITUDES
  };

  //----------------------------------------------------------------------------

  typedef KernelAccessAmplitudes<false> HostAccessAmplitudes;
  typedef KernelAccessAmplitudes<true> DeviceAccessAmplitudes;

  //----------------------------------------------------------------------------

} // end namespace madmatrix

#endif // MemoryAccessAmplitudes_H
