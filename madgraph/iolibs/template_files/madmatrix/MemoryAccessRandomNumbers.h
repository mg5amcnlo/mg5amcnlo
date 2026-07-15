// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Dec 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2021-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MemoryAccessRandomNumbers_H
#define MemoryAccessRandomNumbers_H 1

#include "mgOnGpuConfig.h"

#include "CPPProcess.h"
#include "MemoryAccessHelpers.h"

#ifdef MGONGPUCPP_GPUIMPL
using mg5amcGpu::CPPProcess;
#else
using mg5amcCpu::CPPProcess;
#endif

//----------------------------------------------------------------------------

// A class describing the internal layout of memory buffers for random numbers
// This implementation uses an AOSOA[npagR][nparf][np4][neppR] where nevt=npagR*neppR
// [If many implementations are used, a suffix _AOSOAv1 should be appended to the class name]
class MemoryAccessRandomNumbersBase //_AOSOAv1
{
public: /* clang-format off */

  // Number of Events Per Page in the random number AOSOA memory buffer layout
  // *** NB Different values of neppR lead to different physics results: the ***
  // *** same 1d array is generated, but it is interpreted in different ways ***
  static constexpr int neppR = 8; // HARDCODED TO GIVE ALWAYS THE SAME PHYSICS RESULTS!
  //static constexpr int neppR = 1; // AOS (tests of sectors/requests)

private: /* clang-format on */

  friend class MemoryAccessHelper<MemoryAccessRandomNumbersBase>;
  friend class KernelAccessHelper<MemoryAccessRandomNumbersBase, true>;
  friend class KernelAccessHelper<MemoryAccessRandomNumbersBase, false>;

  // The number of components of a 4-momentum
  static constexpr int np4 = CPPProcess::np4;

  // The number of final state particles in this physics process
  static constexpr int nparf = CPPProcess::nparf;

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
    const int ipagR = ievt / neppR; // #event "R-page"
    const int ieppR = ievt % neppR; // #event in the current event R-page
    constexpr int ip4 = 0;
    constexpr int iparf = 0;
    return &( buffer[ipagR * nparf * np4 * neppR + iparf * np4 * neppR + ip4 * neppR + ieppR] ); // AOSOA[ipagR][iparf][ip4][ieppR]
  }

  //--------------------------------------------------------------------------

  // Locate a field (output) of an event record (input) from the given field indexes (input)
  // [Signature (non-const) ===> fptype& decodeRecord( fptype* buffer, Ts... args ) <===]
  // [NB: expand variadic template "Ts... args" to "const int ip4, const int iparf" and rename "Field" as "Ip4Iparf"]
  static __host__ __device__ inline fptype&
  decodeRecord( fptype* buffer,
                const int ip4,
                const int iparf )
  {
    constexpr int ipagR = 0;
    constexpr int ieppR = 0;
    return buffer[ipagR * nparf * np4 * neppR + iparf * np4 * neppR + ip4 * neppR + ieppR]; // AOSOA[ipagR][iparf][ip4][ieppR]
  }
};

//----------------------------------------------------------------------------

// A class providing access to memory buffers for a given event, based on explicit event numbers
// Its methods use the MemoryAccessHelper templates - note the use of the template keyword in template function instantiations
class MemoryAccessRandomNumbers : public MemoryAccessRandomNumbersBase
{
public:

  // Locate an event record (output) in a memory buffer (input) from the given event number (input)
  // [Signature (non-const) ===> fptype* ieventAccessRecord( fptype* buffer, const int ievt ) <===]
  static constexpr auto ieventAccessRecord = MemoryAccessHelper<MemoryAccessRandomNumbersBase>::ieventAccessRecord;

  // Locate an event record (output) in a memory buffer (input) from the given event number (input)
  // [Signature (const) ===> const fptype* ieventAccessRecordConst( const fptype* buffer, const int ievt ) <===]
  static constexpr auto ieventAccessRecordConst = MemoryAccessHelper<MemoryAccessRandomNumbersBase>::ieventAccessRecordConst;

  // Locate a field (output) of an event record (input) from the given field indexes (input)
  // [Signature (non-const) ===> fptype& decodeRecord( fptype* buffer, const int ipar, const int iparf ) <===]
  static constexpr auto decodeRecordIp4Iparf = MemoryAccessHelper<MemoryAccessRandomNumbersBase>::decodeRecord;

  // Locate a field (output) of an event record (input) from the given field indexes (input)
  // [Signature (const) ===> const fptype& decodeRecordConst( const fptype* buffer, const int ipar, const int iparf ) <===]
  static constexpr auto decodeRecordIp4IparfConst =
    MemoryAccessHelper<MemoryAccessRandomNumbersBase>::template decodeRecordConst<int, int>;

  // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
  // [Signature (non-const) ===> fptype& ieventAccessIp4Iparf( fptype* buffer, const ievt, const int ipar, const int iparf ) <===]
  static constexpr auto ieventAccessIp4Iparf =
    MemoryAccessHelper<MemoryAccessRandomNumbersBase>::template ieventAccessField<int, int>;

  // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
  // [Signature (const) ===> const fptype& ieventAccessIp4IparfConst( const fptype* buffer, const ievt, const int ipar, const int iparf ) <===]
  static constexpr auto ieventAccessIp4IparfConst =
    MemoryAccessHelper<MemoryAccessRandomNumbersBase>::template ieventAccessFieldConst<int, int>;
};

//----------------------------------------------------------------------------

// A class providing access to memory buffers for a given event, based on implicit kernel rules
// Its methods use the KernelAccessHelper template - note the use of the template keyword in template function instantiations
template<bool onDevice>
class KernelAccessRandomNumbers
{
public:

  // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
  // [Signature (non-const) ===> fptype& kernelAccessIp4Iparf( fptype* buffer, const int ipar, const int iparf ) <===]
  static constexpr auto kernelAccessIp4Iparf =
    KernelAccessHelper<MemoryAccessRandomNumbersBase, onDevice>::template kernelAccessField<int, int>;

  // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
  // [Signature (const) ===> const fptype& kernelAccessIp4IparfConst( const fptype* buffer, const int ipar, const int iparf ) <===]
  static constexpr auto kernelAccessIp4IparfConst =
    KernelAccessHelper<MemoryAccessRandomNumbersBase, onDevice>::template kernelAccessFieldConst<int, int>;
};

//----------------------------------------------------------------------------

typedef KernelAccessRandomNumbers<false> HostAccessRandomNumbers;
typedef KernelAccessRandomNumbers<true> DeviceAccessRandomNumbers;

//----------------------------------------------------------------------------

#endif // MemoryAccessRandomNumbers_H
