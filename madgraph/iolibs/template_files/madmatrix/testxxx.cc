// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Apr 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2021-2024).
// Integrated with the MadGraph7 project in Feb 2026.
// ----------------------------------------------------------------------------
// Use ./runTest.exe --gtest_filter=*xxx to run only testxxx.cc tests
// ----------------------------------------------------------------------------

#include "mgOnGpuConfig.h"

#include "CPPProcess.h"
#include "HelAmps_%(model_name)s.h"
#include "MemoryAccessMomenta.h"
#include "MemoryAccessWavefunctions.h"
#include "MemoryBuffers.h"
#include "epoch_process_id.h"

#include <gtest/gtest.h>

#include <array>
#include <cassert>
#include <cfenv> // for signal and SIGFPE (see https://stackoverflow.com/a/17473528)
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#ifdef MGONGPUCPP_GPUIMPL
#define TESTID( s ) s##_GPU_XXX
#else
#define TESTID( s ) s##_CPU_XXX
#endif

#define XTESTID( s ) TESTID( s )

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  std::string fpeHandlerMessage = "unknown";
  int fpeHandlerIevt = -1;
  inline void fpeHandlerTestxxx( int /*sig*/ )
  {
#ifdef MGONGPUCPP_GPUIMPL
    std::cerr << "Floating Point Exception (GPU): '" << fpeHandlerMessage << "' ievt=" << fpeHandlerIevt << std::endl;
#else
    std::cerr << "Floating Point Exception (CPU neppV=" << neppV << "): '" << fpeHandlerMessage << "' ievt=" << fpeHandlerIevt << std::endl;
#endif
    exit( 1 );
  }
}

TEST( XTESTID( MG_EPOCH_PROCESS_ID ), testxxx )
{
#ifdef MGONGPUCPP_GPUIMPL
  using namespace mg5amcGpu;
#else
  using namespace mg5amcCpu;
#endif
#ifndef __APPLE__ // test #701 (except on MacOS where feenableexcept is not defined #730)
  auto fpeHandlerDefault = signal( SIGFPE, fpeHandlerTestxxx );
#endif
  constexpr bool dumpEvents = false;       // dump the expected output of the test?
  constexpr bool testEvents = !dumpEvents; // run the test?
  constexpr fptype toleranceXXXs = std::is_same<fptype, double>::value ? 1.E-15 : 1.E-5;
  // Constant parameters
  constexpr int neppM = MemoryAccessMomenta::neppM; // AOSOA layout
  constexpr int np4 = CPPProcess::np4;
  const int nevt = 32;         // 12 independent tests plus 20 duplicates (need a multiple of 16 for floats '512z')
  assert( nevt %% neppM == 0 ); // nevt must be a multiple of neppM
  assert( nevt %% neppV == 0 ); // nevt must be a multiple of neppV
  // Fill in the input momenta
#ifdef MGONGPUCPP_GPUIMPL
  mg5amcGpu::PinnedHostBufferMomenta hstMomenta( nevt ); // AOSOA[npagM][npar=4][np4=4][neppM]
#else
  mg5amcCpu::HostBufferMomenta hstMomenta( nevt ); // AOSOA[npagM][npar=4][np4=4][neppM]
#endif /* clang-format off */
  // NB NEW TESTS FOR DEBUGGING #701: KEEP TWO SEPARATE SETS (16-SIMD-VECTORS!) OF TESTS FOR M==0 AND M!=0!
  const fptype par0[np4 * nevt] = // AOS[nevt][np4]
    {
      500, 0, 0, 500,      // #0  (m=0 pT=0 E=pz>0)
      500, 0, 0, -500,     // #1  (m=0 pT=0 -E=pz<0)
      500, 300, 400, 0,    // #2  (m=0 pT>0 pz=0)
      500, 180, 240, 400,  // #3  (m=0 pT>0 pz>0)
      500, 180, 240, -400, // #4  (m=0 pT>0 pz<0)
      500, 0, 0, 500,      // #5  DUPLICATE == #0 (m=0 pT=0 E=pz>0)
      500, 0, 0, -500,     // #6  DUPLICATE == #1 (m=0 pT=0 -E=pz<0)
      500, 300, 400, 0,    // #7  DUPLICATE == #2 (m=0 pT>0 pz=0)
      500, 180, 240, 400,  // #8  DUPLICATE == #3 (m=0 pT>0 pz>0)
      500, 180, 240, -400, // #9  DUPLICATE == #4 (m=0 pT>0 pz<0)
      500, 0, 0, 500,      // #10 DUPLICATE == #0 (m=0 pT=0 E=pz>0)
      500, 0, 0, -500,     // #11 DUPLICATE == #1 (m=0 pT=0 -E=pz<0)
      500, 300, 400, 0,    // #12 DUPLICATE == #2 (m=0 pT>0 pz=0)
      500, 180, 240, 400,  // #13 DUPLICATE == #3 (m=0 pT>0 pz>0)
      500, 180, 240, -400, // #14 DUPLICATE == #4 (m=0 pT>0 pz<0)
      500, 0, 0, 500,      // #15 DUPLICATE == #0 (m=0 pT=0 E=pz>0)
      500, 0, 0, 0,        // #16 (m=50>0 pT=0 pz=0)
      500, 0, 0, 300,      // #17 (m=40>0 pT=0 pz>0)
      500, 0, 0, -300,     // #18 (m=40>0 pT=0 pz<0)
      500, 180, 240, 0,    // #19 (m=40>0 pT>0 pz=0)
      500, -240, -180, 0,  // #20 (m=40>0 pT>0 pz=0)
      500, 180, 192, 144,  // #21 (m=40>0 pT>0 pz>0)
      500, 180, 192, -144, // #22 (m=40>0 pT>0 pz<0)
      500, 0, 0, 0,        // #23 DUPLICATE == #16 (m=50>0 pT=0 pz=0)
      500, 0, 0, 300,      // #24 DUPLICATE == #17 (m=40>0 pT=0 pz>0)
      500, 0, 0, -300,     // #25 DUPLICATE == #18 (m=40>0 pT=0 pz<0)
      500, 180, 240, 0,    // #26 DUPLICATE == #19 (m=40>0 pT>0 pz=0)
      500, -240, -180, 0,  // #27 DUPLICATE == #20 (m=40>0 pT>0 pz=0)
      500, 180, 192, 144,  // #28 DUPLICATE == #21 (m=40>0 pT>0 pz>0)
      500, 180, 192, -144, // #29 DUPLICATE == #22 (m=40>0 pT>0 pz<0)
      500, 0, 0, 0,        // #30 DUPLICATE == #16 (m=50>0 pT=0 pz=0)
      500, 0, 0, 300       // #31 DUPLICATE == #17 (m=40>0 pT=0 pz>0)
    }; /* clang-format on */
  // Array initialization: zero-out as "{0}" (C and C++) or as "{}" (C++ only)
  // See https://en.cppreference.com/w/c/language/array_initialization#Notes
  fptype mass0[nevt] = {};
  bool ispzgt0[nevt] = {};
  bool ispzlt0[nevt] = {};
  bool isptgt0[nevt] = {};
  for( int ievt = 0; ievt < nevt; ievt++ )
  {
    const fptype p0 = par0[ievt * np4 + 0];
    const fptype p1 = par0[ievt * np4 + 1];
    const fptype p2 = par0[ievt * np4 + 2];
    const fptype p3 = par0[ievt * np4 + 3];
    volatile fptype m2 = fpmax( p0 * p0 - p1 * p1 - p2 * p2 - p3 * p3, 0 ); // see #736
    if( m2 > 0 )
      mass0[ievt] = fpsqrt( (fptype)m2 );
    else
      mass0[ievt] = 0;
    ispzgt0[ievt] = ( p3 > 0 );
    ispzlt0[ievt] = ( p3 < 0 );
    isptgt0[ievt] = ( p1 != 0 ) || ( p2 != 0 );
  }
  const int ipar0 = 0; // use only particle0 for this test
  for( int ievt = 0; ievt < nevt; ievt++ )
  {
    for( int ip4 = 0; ip4 < np4; ip4++ )
    {
      MemoryAccessMomenta::ieventAccessIp4Ipar( hstMomenta.data(), ievt, ip4, ipar0 ) = par0[ievt * np4 + ip4]; // AOS to AOSOA
    }
  }
  // Expected output wavefunctions
  std::vector<std::array<fptype, 12>> expwfs;
#include "testxxx_cc_ref.txt" // expwfs.push_back( {...} );
  std::string dumpFileName = "testxxx_cc_ref.txt.new";
  // Compute the output wavefunctions
  // Dump new reference file if requested
  constexpr int nw6 = CPPProcess::nw6; // dimensions of each wavefunction (HELAS KEK 91-11): e.g. 6 for e+ e- -> mu+ mu- (fermions and vectors)
  int itest = 0;                       // index on the expected output vector
  std::ofstream dumpFile;
  if( dumpEvents )
  {
    dumpFile.open( dumpFileName, std::ios::trunc );
    dumpFile << "  // Copyright (C) 2020-2024 CERN and UCLouvain." << std::endl
             << "  // Licensed under the GNU Lesser General Public License (version 3 or later)." << std::endl
             << "  // Created by: A. Valassi (Apr 2021) for the MG5aMC CUDACPP plugin." << std::endl
             << "  // Further modified by: A. Valassi (2021-2024) for the MG5aMC CUDACPP plugin." << std::endl;
  }
  // Lambda function for dumping wavefunctions
  auto dumpwf6 = [&]( std::ostream& out, const cxtype_sv wf[6], const char* xxx, int ievt, int nsp, fptype mass )
  {
    out << std::setprecision( 15 ) << std::scientific;
    out << "  expwfs.push_back( {";
    out << "                                   // ---------" << std::endl;
    for( int iw6 = 0; iw6 < nw6; iw6++ )
    {
#ifdef MGONGPU_CPPSIMD
      const int ieppV = ievt %% neppV; // #event in the current event vector in this iteration
#ifdef MGONGPU_HAS_CPPCXTYPEV_BRK
      out << std::setw( 26 ) << cxreal( wf[iw6][ieppV] ) << ", ";
      out << std::setw( 22 ) << cximag( wf[iw6][ieppV] );
#else
      out << std::setw( 26 ) << wf[iw6].real()[ieppV] << ", ";
      out << std::setw( 22 ) << wf[iw6].imag()[ieppV];
#endif
#else
      out << std::setw( 26 ) << wf[iw6].real();
      out << ", " << std::setw( 22 ) << wf[iw6].imag();
#endif
      if( iw6 < nw6 - 1 )
        out << ",    ";
      else
        out << " } );";
      out << " // itest=" << itest << ": " << xxx << "#" << ievt;
      out << " nsp=" << nsp << " mass=" << (int)mass << std::endl;
    }
    out << std::defaultfloat;
  };
  // Lambda function for testing wavefunctions (1)
  auto testwf6 = [&]( const cxtype_sv wf[6], const char* xxx, int ievt, int nsp, fptype mass )
  {
    if( dumpEvents ) dumpwf6( dumpFile, wf, xxx, ievt, nsp, mass );
    if( testEvents )
    {
      std::array<fptype, 12>& expwf = expwfs[itest];
      //std::cout << "Testing " << std::setw(3) << itest << ": " << xxx << " #" << ievt << std::endl;
      ////for ( int iw6 = 0; iw6<nw6; iw6++ ) std::cout << wf[iw6] << std::endl;
      ////std::cout << "against" << std::endl;
      ////for ( int iw6 = 0; iw6<nw6; iw6++ )
      ////  std::cout << "[" << expwf[iw6*2] << "," << expwf[iw6*2+1] << "]" << std::endl; // NB: expwf[iw6*2], expwf[iw6*2+1] are fp
      for( int iw6 = 0; iw6 < nw6; iw6++ )
      {
        const fptype expReal = expwf[iw6 * 2];
        const fptype expImag = expwf[iw6 * 2 + 1];
        if( true )
        {
#ifdef MGONGPU_CPPSIMD
          const int ieppV = ievt %% neppV; // #event in the current event vector in this iteration
#ifdef MGONGPU_HAS_CPPCXTYPEV_BRK
          EXPECT_NEAR( cxreal( wf[iw6][ieppV] ), expReal, std::abs( expReal * toleranceXXXs ) )
            << " itest=" << itest << ": " << xxx << "#" << ievt;
          EXPECT_NEAR( cximag( wf[iw6][ieppV] ), expImag, std::abs( expImag * toleranceXXXs ) )
            << " itest=" << itest << ": " << xxx << "#" << ievt;
#else
          EXPECT_NEAR( wf[iw6].real()[ieppV], expReal, std::abs( expReal * toleranceXXXs ) )
            << " itest=" << itest << ": " << xxx << "#" << ievt;
          EXPECT_NEAR( wf[iw6].imag()[ieppV], expImag, std::abs( expImag * toleranceXXXs ) )
            << " itest=" << itest << ": " << xxx << "#" << ievt;
#endif
#else
          EXPECT_NEAR( cxreal( wf[iw6] ), expReal, std::abs( expReal * toleranceXXXs ) )
            << " itest=" << itest << ": " << xxx << "#" << ievt;
          EXPECT_NEAR( cximag( wf[iw6] ), expImag, std::abs( expImag * toleranceXXXs ) )
            << " itest=" << itest << ": " << xxx << "#" << ievt;
#endif
        }
      }
    }
    itest++;
  };
  // Lambda function for testing wavefunctions (2)
  auto testwf6two = [&]( const cxtype_sv wf[6], const cxtype_sv expwf[6], const char* xxx, int ievt )
  {
    if( testEvents )
    {
      const std::string xxxFull( xxx[0] == 'i' ? "ixxxxx" : "oxxxxx" );
      //std::cout << "Testing " << std::setw(3) << itest << ": ";
      //std::cout << xxx << " #" << ievt << " against " << xxxFull << std::endl;
      ////for ( int iw6 = 0; iw6<nw6; iw6++ ) std::cout << wf[iw6] << std::endl;
      ////std::cout << "against" << std::endl;
      ////for ( int iw6 = 0; iw6<nw6; iw6++ ) std::cout << expwf[iw6] << std::endl; // NB: expwf[iw6] is cx
      for( int iw6 = 0; iw6 < nw6; iw6++ )
      {
        if( true )
        {
#ifdef MGONGPU_CPPSIMD
          const int ieppV = ievt %% neppV; // #event in the current event vector in this iteration
#ifdef MGONGPU_HAS_CPPCXTYPEV_BRK
          const fptype expReal = cxreal( expwf[iw6][ieppV] );
          const fptype expImag = cximag( expwf[iw6][ieppV] );
          EXPECT_NEAR( cxreal( wf[iw6][ieppV] ), expReal, std::abs( expReal * toleranceXXXs ) )
            << " itest=" << itest << ": " << xxx << "#" << ievt << " against " << xxxFull;
          EXPECT_NEAR( cximag( wf[iw6][ieppV] ), expImag, std::abs( expImag * toleranceXXXs ) )
            << " itest=" << itest << ": " << xxx << "#" << ievt << " against " << xxxFull;
#else
          const fptype expReal = expwf[iw6].real()[ieppV];
          const fptype expImag = expwf[iw6].imag()[ieppV];
          EXPECT_NEAR( wf[iw6].real()[ieppV], expReal, std::abs( expReal * toleranceXXXs ) )
            << " itest=" << itest << ": " << xxx << "#" << ievt << " against " << xxxFull;
          EXPECT_NEAR( wf[iw6].imag()[ieppV], expImag, std::abs( expImag * toleranceXXXs ) )
            << " itest=" << itest << ": " << xxx << "#" << ievt << " against " << xxxFull;
#endif
#else
          const fptype expReal = cxreal( expwf[iw6] );
          const fptype expImag = cximag( expwf[iw6] );
          EXPECT_NEAR( cxreal( wf[iw6] ), expReal, std::abs( expReal * toleranceXXXs ) )
            << " itest=" << itest << ": " << xxx << "#" << ievt << " against " << xxxFull;
          EXPECT_NEAR( cximag( wf[iw6] ), expImag, std::abs( expImag * toleranceXXXs ) )
            << " itest=" << itest << ": " << xxx << "#" << ievt << " against " << xxxFull;
#endif
        }
      }
    }
  };
  // Lambda function for resetting hstMomenta to the values of par0
  // This is needed in each test because hstMomenta may have been modified to ensure a function like ipzxxx can be used (#701)
  auto resetHstMomentaToPar0 = [&]()
  {
    for( int ievt = 0; ievt < nevt; ievt++ )
      for( int ip4 = 0; ip4 < np4; ip4++ )
        MemoryAccessMomenta::ieventAccessIp4Ipar( hstMomenta.data(), ievt, ip4, ipar0 ) = par0[ievt * np4 + ip4]; // AOS to AOSOA
  };
  // Lambda function for preparing the test of one specific function
  const bool debug = false;
  auto prepareTest = [&]( const char* xxx, int ievt )
  {
    if( debug ) std::cout << "Prepare test " << xxx << " ievt=" << ievt << std::endl;
    resetHstMomentaToPar0();
    fpeHandlerMessage = xxx;
    fpeHandlerIevt = ievt;
    if( std::string( xxx ) == "ipzxxx" || std::string( xxx ) == "opzxxx" || std::string( xxx ) == "imzxxx" || std::string( xxx ) == "omzxxx" || std::string( xxx ) == "ixzxxx" || std::string( xxx ) == "oxzxxx" )
    {
      // Modify hstMomenta so that ALL events have the momenta of a single ievt
      // This ensures that a function like ipzxxx (which assumes pZ>0) can be used without triggering FPEs (#701)
      // This is done by filling the full SIMD vector with the value of ievt, which was already tested to respect the relevant assumptions
      for( int jevt = 0; jevt < nevt; jevt++ )
        for( int ip4 = 0; ip4 < np4; ip4++ )
          MemoryAccessMomenta::ieventAccessIp4Ipar( hstMomenta.data(), jevt, ip4, ipar0 ) = par0[ievt * np4 + ip4]; // AOS to AOSOA
    }
  };
  // Array initialization: zero-out as "{0}" (C and C++) or as "{}" (C++ only)
  // See https://en.cppreference.com/w/c/language/array_initialization#Notes
  cxtype_sv outwfI[6] = {}; // last result of ixxxxx (mass==0)
  cxtype_sv outwfO[6] = {}; // last result of oxxxxx (mass==0)
  cxtype_sv outwf[6] = {};
  cxtype_sv outwf3[6] = {};                                // NB: only 3 are filled by sxxxxx, but 6 are compared!
  fptype* fp_outwfI = reinterpret_cast<fptype*>( outwfI ); // proof of concept for using fptype* in the interface
  fptype* fp_outwfO = reinterpret_cast<fptype*>( outwfO ); // proof of concept for using fptype* in the interface
  fptype* fp_outwf = reinterpret_cast<fptype*>( outwf );   // proof of concept for using fptype* in the interface
  fptype* fp_outwf3 = reinterpret_cast<fptype*>( outwf3 ); // proof of concept for using fptype* in the interface
  const int nhel = 1;
  // *** START OF TESTING LOOP
  for( auto nsp: { -1, +1 } ) // antifermion/fermion (or initial/final for scalar and vector)
  {
    for( int ievt = 0; ievt < nevt; ievt++ )
    {
#ifdef MGONGPUCPP_GPUIMPL
      using namespace mg5amcGpu;
#else
      using namespace mg5amcCpu;
#endif
      if( debug )
      {
        std::cout << std::endl;
        std::cout << "nsp=" << nsp << " ievt=" << ievt << ": ";
        for( int ip4 = 0; ip4 < np4; ip4++ ) std::cout << par0[ievt * np4 + ip4] << ", ";
        std::cout << std::endl;
      }
      const int ipagV = ievt / neppV; // #event vector in this iteration
      const fptype* ievt0Momenta = MemoryAccessMomenta::ieventAccessRecordConst( hstMomenta.data(), ipagV * neppV );
      // Test ixxxxx - NO ASSUMPTIONS
      {
        prepareTest( "ixxxxx", ievt );
        const fptype fmass = mass0[ievt];
        ixxxxx<HostAccessMomenta, HostAccessWavefunctions>( ievt0Momenta, fmass, nhel, nsp, fp_outwfI, ipar0 );
        testwf6( outwfI, "ixxxxx", ievt, nsp, fmass );
        ixxxxx<HostAccessMomenta, HostAccessWavefunctions>( ievt0Momenta, -fmass, nhel, nsp, fp_outwfI, ipar0 );
        testwf6( outwfI, "ixxxxx", ievt, nsp, -fmass );
      }
      // Test ipzxxx - ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == +PZ > 0)
      if( mass0[ievt] == 0 && !isptgt0[ievt] && ispzgt0[ievt] )
      {
        prepareTest( "ipzxxx", ievt );
        ipzxxx<HostAccessMomenta, HostAccessWavefunctions>( ievt0Momenta, nhel, nsp, fp_outwf, ipar0 );
        testwf6two( outwf, outwfI, "ipzxxx", ievt );
        testwf6( outwf, "ipzxxx", ievt, nsp, 0 );
      }
      // Test imzxxx - ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == -PZ > 0)
      if( mass0[ievt] == 0 && !isptgt0[ievt] && ispzlt0[ievt] )
      {
        prepareTest( "imzxxx", ievt );
        imzxxx<HostAccessMomenta, HostAccessWavefunctions>( ievt0Momenta, nhel, nsp, fp_outwf, ipar0 );
        testwf6two( outwf, outwfI, "imzxxx", ievt );
        testwf6( outwf, "imzxxx", ievt, nsp, 0 );
      }
      // Test ixzxxx - ASSUMPTIONS: (FMASS == 0) and (PT > 0)
      if( mass0[ievt] == 0 && isptgt0[ievt] )
      {
        prepareTest( "ixzxxx", ievt );
        ixzxxx<HostAccessMomenta, HostAccessWavefunctions>( ievt0Momenta, nhel, nsp, fp_outwf, ipar0 );
        testwf6two( outwf, outwfI, "ixzxxx", ievt );
        testwf6( outwf, "ixzxxx", ievt, nsp, 0 );
      }
      // Test vxxxxx - NO ASSUMPTIONS
      {
        prepareTest( "vxxxxx", ievt );
        const fptype vmass = mass0[ievt];
        vxxxxx<HostAccessMomenta, HostAccessWavefunctions>( ievt0Momenta, vmass, nhel, nsp, fp_outwf, ipar0 );
        testwf6( outwf, "vxxxxx", ievt, nsp, vmass );
        vxxxxx<HostAccessMomenta, HostAccessWavefunctions>( ievt0Momenta, -vmass, nhel, nsp, fp_outwf, ipar0 );
        testwf6( outwf, "vxxxxx", ievt, nsp, -vmass );
      }
      // Test sxxxxx - NO ASSUMPTIONS
      {
        prepareTest( "sxxxxx", ievt );
        const fptype smass = mass0[ievt];
        sxxxxx<HostAccessMomenta, HostAccessWavefunctions>( ievt0Momenta, nsp, fp_outwf3, ipar0 ); // no mass, no helicity (was "smass>0")
        testwf6( outwf3, "sxxxxx", ievt, nsp, smass );
        sxxxxx<HostAccessMomenta, HostAccessWavefunctions>( ievt0Momenta, nsp, fp_outwf3, ipar0 ); // no mass, no helicity (was "smass<0")
        testwf6( outwf3, "sxxxxx", ievt, nsp, -smass );
      }
      // Test oxxxxx - NO ASSUMPTIONS
      {
        prepareTest( "oxxxxx", ievt );
        const fptype fmass = mass0[ievt];
        oxxxxx<HostAccessMomenta, HostAccessWavefunctions>( ievt0Momenta, fmass, nhel, nsp, fp_outwfO, ipar0 );
        testwf6( outwfO, "oxxxxx", ievt, nsp, fmass );
        oxxxxx<HostAccessMomenta, HostAccessWavefunctions>( ievt0Momenta, -fmass, nhel, nsp, fp_outwfO, ipar0 );
        testwf6( outwfO, "oxxxxx", ievt, nsp, -fmass );
      }
      // Test opzxxx - ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == +PZ > 0)
      if( mass0[ievt] == 0 && !isptgt0[ievt] && ispzgt0[ievt] )
      {
        prepareTest( "opzxxx", ievt );
        opzxxx<HostAccessMomenta, HostAccessWavefunctions>( ievt0Momenta, nhel, nsp, fp_outwf, ipar0 );
        testwf6two( outwf, outwfO, "opzxxx", ievt );
        testwf6( outwf, "opzxxx", ievt, nsp, 0 );
      }
      // Test omzxxx - ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == -PZ > 0)
      if( mass0[ievt] == 0 && !isptgt0[ievt] && ispzlt0[ievt] )
      {
        prepareTest( "omzxxx", ievt );
        omzxxx<HostAccessMomenta, HostAccessWavefunctions>( ievt0Momenta, nhel, nsp, fp_outwf, ipar0 );
        testwf6two( outwf, outwfO, "omzxxx", ievt );
        testwf6( outwf, "omzxxx", ievt, nsp, 0 );
      }
      // Test oxzxxx - ASSUMPTIONS: (FMASS == 0) and (PT > 0)
      if( mass0[ievt] == 0 && isptgt0[ievt] )
      {
        prepareTest( "oxzxxx", ievt );
        oxzxxx<HostAccessMomenta, HostAccessWavefunctions>( ievt0Momenta, nhel, nsp, fp_outwf, ipar0 );
        testwf6two( outwf, outwfO, "oxzxxx", ievt );
        testwf6( outwf, "oxzxxx", ievt, nsp, 0 );
      }
    }
  }
  // *** END OF TESTING LOOP
  if( dumpEvents )
  {
    dumpFile.close();
    std::cout << "INFO: New reference data dumped to file '" << dumpFileName << "'" << std::endl;
  }
#ifndef __APPLE__ // test #701 (except on MacOS where feenableexcept is not defined #730)
  signal( SIGFPE, fpeHandlerDefault );
#endif
}

//==========================================================================

// Reset the GPU after ALL tests have gone out of scope
// (This was needed to avoid leaks in profilers, but compute-sanitizer reports no leaks, is it STILL needed?)
// ========= NB: resetting the GPU too early causes segfaults that are very difficult to debug #907 =========
// Try to use atexit (https://stackoverflow.com/a/14610501) but this still crashes!
// ********* FIXME? avoid CUDA API calls in destructors? (see https://stackoverflow.com/a/16982503) *********
void
myexit()
{
#ifdef MGONGPUCPP_GPUIMPL
  //checkGpu( gpuDeviceReset() ); // FIXME??? this still crashes! should systematically avoid CUDA calls in all destructors?
#endif
}

// Main function (see https://google.github.io/googletest/primer.html#writing-the-main-function)
// (NB: the test executables are now separate for C++ and CUDA, therefore main must be included all the time)
// (NB: previously, '#ifndef MGONGPUCPP_GPUIMPL' was ensuring that main was only included once while linking both C++ and CUDA tests)
int
main( int argc, char** argv )
{
  atexit( myexit );
  testing::InitGoogleTest( &argc, argv );
  int status = RUN_ALL_TESTS();
  return status;
}
