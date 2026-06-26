// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Dec 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2021-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MemoryAccessMomenta_H
#define MemoryAccessMomenta_H 1

#include "mgOnGpuConfig.h"

#include "CPPProcess.h"
#include "MemoryAccessHelpers.h"
#include "MemoryAccessVectors.h"

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
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
#ifdef MGONGPUCPP_GPUIMPL /* clang-format off */
    // -----------------------------------------------------------------------------------------------
    // --- GPUs: neppM is best set to a power of 2 times the number of fptype's in a 32-byte cacheline
    // --- This is relevant to ensure coalesced access to momenta in global memory
    // --- Note that neppR is hardcoded and may differ from neppM and neppV on some platforms
    // -----------------------------------------------------------------------------------------------
    //static constexpr int neppM = 64/sizeof(fptype); // 2x 32-byte GPU cache lines (512 bits): 8 (DOUBLE) or 16 (FLOAT)
    static constexpr int neppM = 32/sizeof(fptype); // (DEFAULT) 32-byte GPU cache line (256 bits): 4 (DOUBLE) or 8 (FLOAT)
    //static constexpr int neppM = 1; // *** NB: this is equivalent to AOS *** (slower: 1.03E9 instead of 1.11E9 in eemumu)
#else
    // -----------------------------------------------------------------------------------------------
    // --- CPUs: neppM is best set equal to the number of fptype's (neppV) in a vector register
    // --- This is relevant to ensure faster access to momenta from C++ memory cache lines
    // --- However, neppM is now decoupled from neppV (issue #176) and can be separately hardcoded
    // --- In practice, neppR, neppM and neppV could now (in principle) all be different
    // -----------------------------------------------------------------------------------------------
#ifdef MGONGPU_CPPSIMD
    static constexpr int neppM = MGONGPU_CPPSIMD; // (DEFAULT) neppM=neppV for optimal performance
    //static constexpr int neppM = 64/sizeof(fptype); // maximum CPU vector width (512 bits): 8 (DOUBLE) or 16 (FLOAT)
    //static constexpr int neppM = 32/sizeof(fptype); // lower CPU vector width (256 bits): 4 (DOUBLE) or 8 (FLOAT)
    //static constexpr int neppM = 1; // *** NB: this is equivalent to AOS *** (slower: 4.66E6 instead of 5.09E9 in eemumu)
    //static constexpr int neppM = MGONGPU_CPPSIMD*2; // FOR TESTS
#else
    static constexpr int neppM = 1; // (DEFAULT) neppM=neppV for optimal performance (NB: this is equivalent to AOS)
#endif
#endif /* clang-format on */

    // SANITY CHECK: check that neppM is a power of two
    static_assert( ispoweroftwo( neppM ), "neppM is not a power of 2" );

  private:

    friend class MemoryAccessHelper<MemoryAccessMomentaBase>;
    friend class KernelAccessHelper<MemoryAccessMomentaBase, true>;
    friend class KernelAccessHelper<MemoryAccessMomentaBase, false>;

    // The number of components of a 4-momentum
    static constexpr int np4 = CPPProcess::np4;

    // The number of particles in this physics process
    static constexpr int npar = CPPProcess::npar;

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
      const int ipagM = ievt / neppM; // #event "M-page"
      const int ieppM = ievt % neppM; // #event in the current event M-page
      constexpr int ip4 = 0;
      constexpr int ipar = 0;
      return &( buffer[ipagM * npar * np4 * neppM + ipar * np4 * neppM + ip4 * neppM + ieppM] ); // AOSOA[ipagM][ipar][ip4][ieppM]
    }

    //--------------------------------------------------------------------------

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (non-const) ===> fptype& decodeRecord( fptype* buffer, Ts... args ) <===]
    // [NB: expand variadic template "Ts... args" to "const int ip4, const int ipar" and rename "Field" as "Ip4Ipar"]
    static __host__ __device__ inline fptype&
    decodeRecord( fptype* buffer,
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
    // [Signature (non-const) ===> fptype* ieventAccessRecord( fptype* buffer, const int ievt ) <===]
    static constexpr auto ieventAccessRecord = MemoryAccessHelper<MemoryAccessMomentaBase>::ieventAccessRecord;

    // Locate an event record (output) in a memory buffer (input) from the given event number (input)
    // [Signature (const) ===> const fptype* ieventAccessRecordConst( const fptype* buffer, const int ievt ) <===]
    static constexpr auto ieventAccessRecordConst = MemoryAccessHelper<MemoryAccessMomentaBase>::ieventAccessRecordConst;

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (non-const) ===> fptype& decodeRecord( fptype* buffer, const int ipar, const int ipar ) <===]
    static constexpr auto decodeRecordIp4Ipar = MemoryAccessHelper<MemoryAccessMomentaBase>::decodeRecord;

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (const) ===> const fptype& decodeRecordConst( const fptype* buffer, const int ipar, const int ipar ) <===]
    static constexpr auto decodeRecordIp4IparConst =
      MemoryAccessHelper<MemoryAccessMomentaBase>::template decodeRecordConst<int, int>;

    // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
    // [Signature (non-const) ===> fptype& ieventAccessIp4Ipar( fptype* buffer, const ievt, const int ipar, const int ipar ) <===]
    static constexpr auto ieventAccessIp4Ipar =
      MemoryAccessHelper<MemoryAccessMomentaBase>::template ieventAccessField<int, int>;

    // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
    // [Signature (const) ===> const fptype& ieventAccessIp4IparConst( const fptype* buffer, const ievt, const int ipar, const int ipar ) <===]
    // DEFAULT VERSION
    static constexpr auto ieventAccessIp4IparConst =
      MemoryAccessHelper<MemoryAccessMomentaBase>::template ieventAccessFieldConst<int, int>;

    /*
    // Locate a field (output) in a memory buffer (input) from the given event number (input) and the given field indexes (input)
    // [Signature (const) ===> const fptype& ieventAccessIp4IparConst( const fptype* buffer, const ievt, const int ipar, const int ipar ) <===]
    // DEBUG VERSION WITH PRINTOUTS
    static __host__ __device__ inline const fptype&
    ieventAccessIp4IparConst( const fptype* buffer,
                                            const int ievt,
                                            const int ip4,
                                            const int ipar )
    {
      const fptype& out = MemoryAccessHelper<MemoryAccessMomentaBase>::template ieventAccessFieldConst<int, int>( buffer, ievt, ip4, ipar );
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
    // [Signature (non-const, SCALAR) ===> fptype& kernelAccessIp4Ipar( fptype* buffer, const int ipar, const int ipar ) <===]
    static constexpr auto kernelAccessIp4Ipar =
      KernelAccessHelper<MemoryAccessMomentaBase, onDevice>::template kernelAccessField<int, int>;

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (const, SCALAR) ===> const fptype& kernelAccessIp4IparConst( const fptype* buffer, const int ipar, const int ipar ) <===]
    // DEFAULT VERSION
    static constexpr auto kernelAccessIp4IparConst_s =
      KernelAccessHelper<MemoryAccessMomentaBase, onDevice>::template kernelAccessFieldConst<int, int>;

    /*
    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (const, SCALAR) ===> const fptype& kernelAccessIp4IparConst( const fptype* buffer, const int ipar, const int ipar ) <===]
    // DEBUG VERSION WITH PRINTOUTS
    static __host__ __device__ inline const fptype&
    kernelAccessIp4IparConst_s( const fptype* buffer,
                                const int ip4,
                                const int ipar )
    {
      const fptype& out = KernelAccessHelper<MemoryAccessMomentaBase, onDevice>::template kernelAccessFieldConst<int, int>( buffer, ip4, ipar );
      printf( "ipar=%2d ip4=%2d ievt='kernel' out=%8.3f\n", ipar, ip4, out );
      return out;
    }
    */

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (const, SCALAR OR VECTOR) ===> fptype_sv kernelAccessIp4IparConst( const fptype* buffer, const int ipar, const int ipar ) <===]
    // FIXME? Eventually return by const reference and support aligned arrays only?
    // FIXME? Currently return by value to support also unaligned and arbitrary arrays
    static __host__ __device__ inline fptype_sv
    kernelAccessIp4IparConst( const fptype* buffer,
                              const int ip4,
                              const int ipar )
    {
      const fptype& out = kernelAccessIp4IparConst_s( buffer, ip4, ipar );
#ifndef MGONGPU_CPPSIMD
      return out;
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
          return mg5amcCpu::fptypevFromAlignedArray( out ); // use reinterpret_cast
        }
        else if( (size_t)( buffer ) % mgOnGpu::cppAlign == 0 )
        {
          //static bool first=true; if( first ){ std::cout << "WARNING! aligned AOSOA, reinterpret cast" << std::endl; first=false; } // SLOWER (5.00E6)
          // DEFAULT! A tiny bit (<1%) slower because of the alignment check (5.07E6 in eemumu 512y)
          // This explicitly checks buffer alignment to avoid segmentation faults in reinterpret_cast
          return mg5amcCpu::fptypevFromAlignedArray( out ); // SIMD bulk load of neppV, use reinterpret_cast
        }
        else
        {
          //static bool first=true; if( first ){ std::cout << "WARNING! AOSOA but no reinterpret cast" << std::endl; first=false; } // SLOWER (4.93E6)
          // A bit (1%) slower (5.05E6 in eemumu 512y)
          // This does not require buffer alignment, but it requires AOSOA with neppM>=neppV and neppM%neppV==0
          return mg5amcCpu::fptypevFromUnalignedArray( out ); // SIMD bulk load of neppV, do not use reinterpret_cast (fewer SIMD operations)
        }
      }
      else
      {
        //static bool first=true; if( first ){ std::cout << "WARNING! arbitrary array" << std::endl; first=false; } // SLOWER (5.08E6)
        // ?!Used to be much slower, now a tiny bit faster for AOSOA?! (5.11E6 for AOSOA, 4.64E6 for AOS in eemumu 512y)
        // This does not even require AOSOA with neppM>=neppV and neppM%neppV==0 (e.g. can be used with AOS neppM==1)
        constexpr int ievt0 = 0; // just make it explicit in the code that buffer refers to a given ievt0 and decoderIeppV fetches event ievt0+ieppV
        auto decoderIeppv = [buffer, ip4, ipar]( int ieppV )
          -> const fptype&
        { return MemoryAccessMomenta::ieventAccessIp4IparConst( buffer, ievt0 + ieppV, ip4, ipar ); };
        return mg5amcCpu::fptypevFromArbitraryArray( decoderIeppv ); // iterate over ieppV in neppV (no SIMD)
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

} // end namespace mg5amcGpu/mg5amcCpu

#endif // MemoryAccessMomenta_H
