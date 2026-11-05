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
#include "MemoryAccessVectors.h"

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
    static constexpr int neppM = MGONGPU_CPPSIMD; // (DEFAULT) neppM=neppV for optimal performance
    //static constexpr int neppM = 64/sizeof(fptype_momenta); // maximum CPU vector width (512 bits): 8 (DOUBLE) or 16 (FLOAT)
    //static constexpr int neppM = 32/sizeof(fptype_momenta); // lower CPU vector width (256 bits): 4 (DOUBLE) or 8 (FLOAT)
    //static constexpr int neppM = 1; // *** NB: this is equivalent to AOS *** (slower: 4.66E6 instead of 5.09E9 in eemumu)
    //static constexpr int neppM = MGONGPU_CPPSIMD*2; // FOR TESTS

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
#ifdef MGONGPU_SIMD_DENOM64
      // assume load from one page return high and low
      static_assert( MemoryAccessMomentaBase::neppM == neppV, "SIMD denom64 (FPTYPE=v) assumes neppM == neppV" );
      return madmatrix::momvFromContiguousArray( out );
#else
      constexpr int neppM = MemoryAccessMomentaBase::neppM;
      constexpr bool useContiguousEventsIfPossible = true; // DEFAULT
      //constexpr bool useContiguousEventsIfPossible = false; // FOR PERFORMANCE TESTS (treat as arbitrary array even if it is an AOSOA)
      // Use c++17 "if constexpr": compile-time branching
      if constexpr( useContiguousEventsIfPossible && ( neppM >= neppV ) && ( neppM % neppV == 0 ) )
      {
        //constexpr bool skipAlignmentCheck = true; // FASTEST (SEGFAULTS IF MISALIGNED ACCESS, NEEDS A SANITY CHECK ELSEWHERE!)
        constexpr bool skipAlignmentCheck = false; // DEFAULT: A BIT SLOWER BUT SAFER [ALLOWS MISALIGNED ACCESS]
        if constexpr( skipAlignmentCheck )
        {
          //static bool first=true; if( first ){ std::cout << "WARNING! assume aligned AOSOA, skip check" << std::endl; first=false; } // SLOWER (5.06E6)
          // FASTEST? (5.09E6 in eemumu 512y)
          // This assumes alignment for momenta1d without checking - causes segmentation fault in reinterpret_cast if not aligned!
          return madmatrix::fptypevFromAlignedArray( out ); // use reinterpret_cast
        }
        else if( (size_t)( buffer ) % mgOnGpu::cppAlign == 0 )
        {
          //static bool first=true; if( first ){ std::cout << "WARNING! aligned AOSOA, reinterpret cast" << std::endl; first=false; } // SLOWER (5.00E6)
          // DEFAULT! A tiny bit (<1%) slower because of the alignment check (5.07E6 in eemumu 512y)
          // This explicitly checks buffer alignment to avoid segmentation faults in reinterpret_cast
          return madmatrix::fptypevFromAlignedArray( out ); // SIMD bulk load of neppV, use reinterpret_cast
        }
        else
        {
          //static bool first=true; if( first ){ std::cout << "WARNING! AOSOA but no reinterpret cast" << std::endl; first=false; } // SLOWER (4.93E6)
          // A bit (1%) slower (5.05E6 in eemumu 512y)
          // This does not require buffer alignment, but it requires AOSOA with neppM>=neppV and neppM%neppV==0
          return madmatrix::fptypevFromUnalignedArray( out ); // SIMD bulk load of neppV, do not use reinterpret_cast (fewer SIMD operations)
        }
      }
      else
      {
        //static bool first=true; if( first ){ std::cout << "WARNING! arbitrary array" << std::endl; first=false; } // SLOWER (5.08E6)
        // ?!Used to be much slower, now a tiny bit faster for AOSOA?! (5.11E6 for AOSOA, 4.64E6 for AOS in eemumu 512y)
        // This does not even require AOSOA with neppM>=neppV and neppM%neppV==0 (e.g. can be used with AOS neppM==1)
        constexpr int ievt0 = 0; // just make it explicit in the code that buffer refers to a given ievt0 and decoderIeppV fetches event ievt0+ieppV
        auto decoderIeppv = [buffer, ip4, ipar]( int ieppV )
          -> const fptype_momenta&
        { return MemoryAccessMomenta::ieventAccessIp4IparConst( buffer, ievt0 + ieppV, ip4, ipar ); };
        return madmatrix::fptypevFromArbitraryArray( decoderIeppv ); // iterate over ieppV in neppV (no SIMD)
      }
#endif
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
