// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Dec 2021) for the MadGraph7 CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2021-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MemoryAccessMomenta_H
#define MemoryAccessMomenta_H 1

#include "mgOnGpuConfig.h"

#include "ProcessData.h"
#include "MemoryAccessHelpers.h"
#include "mgOnGpuCxtypes.h"

//One namespace. Split ber backend.
namespace madmatrix
{
  //----------------------------------------------------------------------------

  // A class describing the internal layout of memory buffers for momenta
  // This implementation uses an AOSOA[npagM][npar][np4][neppM] where nevt=npagM*neppM
  // [If many implementations are used, a suffix _AOSOAv1 should be appended to the class name]
  class MemoryAccessMomentaBase //_AOSOAv1
  {
  public:

    // Number of Events Per Page in the momenta AOSOA memory buffer layout
    // (these are all best kept as a compile-time constants: see issue #23)
    // -----------------------------------------------------------------------------------------------
    // --- CPUs: neppM is best set equal to the number of fptype's (neppV) in a vector register
    // --- This is relevant to ensure faster access to momenta from C++ memory cache lines
    // --- However, neppM is now decoupled from neppV (issue #176) and can be separately hardcoded
    // --- In practice, neppR, neppM and neppV could now (in principle) all be different
    // -----------------------------------------------------------------------------------------------
    static constexpr int neppM = 1; // (DEFAULT) neppM=neppV for optimal performance (NB: this is equivalent to AOS)

    // SANITY CHECK: check that neppM is a power of two
    static_assert( ispoweroftwo( neppM ), "neppM is not a power of 2" );

  private:

    friend class MemoryAccessHelper<MemoryAccessMomentaBase, fptype_momenta>;
    friend class KernelAccessHelper<MemoryAccessMomentaBase, true, fptype_momenta>;
    friend class KernelAccessHelper<MemoryAccessMomentaBase, false, fptype_momenta>;

    // The number of components of a 4-momentum
    static constexpr int np4 = ProcessData::np4;

    // The number of particles in this physics process
    static constexpr int npar = ProcessData::npar;

    //--------------------------------------------------------------------------
    // NB all KernelLaunchers assume that memory access can be decomposed as "accessField = decodeRecord( accessRecord )"
    // (in other words: first locate the event record for a given event, then locate an element in that record)
    //--------------------------------------------------------------------------

    // Locate an event record (output) in a memory buffer (input) from the given event number (input)
    // [Signature (non-const) ===> fptype_momenta* ieventAccessRecord( fptype_momenta* buffer, const int ievt ) <===]
    static __host__ __device__ inline fptype_momenta*
    ieventAccessRecord( fptype_momenta* buffer,
                        const int ievt )
    {
      const int ipagM = ievt / neppM; // #event "M-page"
      const int ieppM = ievt % neppM; // #event in the current event M-page
      constexpr int ip4 = 0;
      constexpr int ipar = 0;
      return &( buffer[ipagM * npar * np4 * neppM + ipar * np4 * neppM + ip4 * neppM + ieppM] ); // AOSOA[ipagM][ipar][ip4][ieppM]
    }

    //--------------------------------------------------------------------------

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (non-const) ===> fptype_momenta& decodeRecord( fptype_momenta* buffer, Ts... args ) <===]
    // [NB: expand variadic template "Ts... args" to "const int ip4, const int ipar" and rename "Field" as "Ip4Ipar"]
    static __host__ __device__ inline fptype_momenta&
    decodeRecord( fptype_momenta* buffer,
                  const int ip4,
                  const int ipar )
    {
      constexpr int ipagM = 0;
      constexpr int ieppM = 0;
      return buffer[ipagM * npar * np4 * neppM + ipar * np4 * neppM + ip4 * neppM + ieppM]; // AOSOA[ipagM][ipar][ip4][ieppM]
    }
  };

  //----------------------------------------------------------------------------

  // A class providing access to memory buffers for a given event, based on explicit event numbers
  // Its methods use the MemoryAccessHelper templates - note the use of the template keyword in template function instantiations
  class MemoryAccessMomenta : public MemoryAccessMomentaBase
  {
  public:

    // Locate an event record (output) in a memory buffer (input) from the given event number (input)
    // [Signature (non-const) ===> fptype_momenta* ieventAccessRecord( fptype_momenta* buffer, const int ievt ) <===]
    static constexpr auto ieventAccessRecord = MemoryAccessHelper<MemoryAccessMomentaBase, fptype_momenta>::ieventAccessRecord;

    // Locate an event record (output) in a memory buffer (input) from the given event number (input)
    // [Signature (const) ===> const fptype_momenta* ieventAccessRecordConst( const fptype_momenta* buffer, const int ievt ) <===]
    static constexpr auto ieventAccessRecordConst = MemoryAccessHelper<MemoryAccessMomentaBase, fptype_momenta>::ieventAccessRecordConst;

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (non-const) ===> fptype_momenta& decodeRecord( fptype_momenta* buffer, const int ipar, const int ipar ) <===]
    static constexpr auto decodeRecordIp4Ipar = MemoryAccessHelper<MemoryAccessMomentaBase, fptype_momenta>::decodeRecord;

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (const) ===> const fptype_momenta& decodeRecordConst( const fptype_momenta* buffer, const int ipar, const int ipar ) <===]
    static constexpr auto decodeRecordIp4IparConst =
      MemoryAccessHelper<MemoryAccessMomentaBase, fptype_momenta>::template decodeRecordConst<int, int>;

    // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
    // [Signature (non-const) ===> fptype_momenta& ieventAccessIp4Ipar( fptype_momenta* buffer, const ievt, const int ipar, const int ipar ) <===]
    static constexpr auto ieventAccessIp4Ipar =
      MemoryAccessHelper<MemoryAccessMomentaBase, fptype_momenta>::template ieventAccessField<int, int>;

    // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
    // [Signature (const) ===> const fptype_momenta& ieventAccessIp4IparConst( const fptype_momenta* buffer, const ievt, const int ipar, const int ipar ) <===]
    // DEFAULT VERSION
    static constexpr auto ieventAccessIp4IparConst =
      MemoryAccessHelper<MemoryAccessMomentaBase, fptype_momenta>::template ieventAccessFieldConst<int, int>;

    /*
    // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
    // [Signature (const) ===> const fptype_momenta& ieventAccessIp4IparConst( const fptype_momenta* buffer, const ievt, const int ipar, const int ipar ) <===]
    // DEBUG VERSION WITH PRINTOUTS
    static __host__ __device__ inline const fptype_momenta&
    ieventAccessIp4IparConst( const fptype_momenta* buffer,
                                            const int ievt,
                                            const int ip4,
                                            const int ipar )
    {
      const fptype_momenta& out = MemoryAccessHelper<MemoryAccessMomentaBase, fptype_momenta>::template ieventAccessFieldConst<int, int>( buffer, ievt, ip4, ipar );
      printf( "ipar=%2d ip4=%2d ievt=%8d out=%8.3f\n", ipar, ip4, ievt, out );
      return out;
    }
    */
  };

  //----------------------------------------------------------------------------

  // A class providing access to memory buffers for a given event, based on implicit kernel rules
  // Its methods use the KernelAccessHelper template - note the use of the template keyword in template function instantiations
  template<bool onDevice>
  class KernelAccessMomenta
  {
  public:

    // Expose selected functions from MemoryAccessMomenta
    static constexpr auto ieventAccessRecordConst = MemoryAccessMomenta::ieventAccessRecordConst;

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (non-const, SCALAR) ===> fptype_momenta& kernelAccessIp4Ipar( fptype_momenta* buffer, const int ipar, const int ipar ) <===]
    static constexpr auto kernelAccessIp4Ipar =
      KernelAccessHelper<MemoryAccessMomentaBase, onDevice, fptype_momenta>::template kernelAccessField<int, int>;

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (const, SCALAR) ===> const fptype_momenta& kernelAccessIp4IparConst( const fptype_momenta* buffer, const int ipar, const int ipar ) <===]
    // DEFAULT VERSION
    static constexpr auto kernelAccessIp4IparConst_s =
      KernelAccessHelper<MemoryAccessMomentaBase, onDevice, fptype_momenta>::template kernelAccessFieldConst<int, int>;

    /*
    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (const, SCALAR) ===> const fptype_momenta& kernelAccessIp4IparConst( const fptype_momenta* buffer, const int ipar, const int ipar ) <===]
    // DEBUG VERSION WITH PRINTOUTS
    static __host__ __device__ inline const fptype_momenta&
    kernelAccessIp4IparConst_s( const fptype_momenta* buffer,
                                const int ip4,
                                const int ipar )
    {
      const fptype_momenta& out = KernelAccessHelper<MemoryAccessMomentaBase, onDevice, fptype_momenta>::template kernelAccessFieldConst<int, int>( buffer, ip4, ipar );
      printf( "ipar=%2d ip4=%2d ievt='kernel' out=%8.3f\n", ipar, ip4, out );
      return out;
    }
    */

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (const, SCALAR OR VECTOR) ===> fptype_momenta_sv kernelAccessIp4IparConst( const fptype_momenta* buffer, const int ipar, const int ipar ) <===]
    // FIXME? Eventually return by const reference and support aligned arrays only?
    // FIXME? Currently return by value to support also unaligned and arbitrary arrays
    static __host__ __device__ inline fptype_momenta_sv
    kernelAccessIp4IparConst( const fptype_momenta* buffer,
                              const int ip4,
                              const int ipar )
    {
      const fptype_momenta& out = kernelAccessIp4IparConst_s( buffer, ip4, ipar );
      return out;
    }

    // Is this a HostAccess or DeviceAccess class?
    // [this is only needed for a warning printout in rambo.h for nparf==1 #358]
    static __host__ __device__ inline constexpr bool
    isOnDevice()
    {
      return onDevice;
    }
  };

  //----------------------------------------------------------------------------

  typedef KernelAccessMomenta<false> HostAccessMomenta;
  typedef KernelAccessMomenta<true> DeviceAccessMomenta;

  //----------------------------------------------------------------------------

} // end namespace madmatrix

#endif // MemoryAccessMomenta_H
