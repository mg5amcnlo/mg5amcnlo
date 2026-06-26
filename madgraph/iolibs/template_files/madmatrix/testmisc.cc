// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2022) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.
// ----------------------------------------------------------------------------
// Use ./runTest.exe --gtest_filter=*misc to run only testmisc.cc tests
// ----------------------------------------------------------------------------

#include "mgOnGpuConfig.h"

#include "mgOnGpuVectors.h"

#include "constexpr_math.h"
#include "epoch_process_id.h"
#include "valgrind.h"

#include <gtest/gtest.h>

//#include <quadmath.h>
//#include <format> // needs C++20... https://stackoverflow.com/a/65347016
#include <iomanip>
#include <sstream>
#include <typeinfo>

#ifdef MGONGPUCPP_GPUIMPL
#define TESTID( s ) s##_GPU_MISC
#else
#define TESTID( s ) s##_CPU_MISC
#endif

#define XTESTID( s ) TESTID( s )

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
#ifdef MGONGPU_CPPSIMD /* clang-format off */
#define EXPECT_TRUE_sv( cond ) { bool_v mask( cond ); EXPECT_TRUE( maskand( mask ) ); }
#else
#define EXPECT_TRUE_sv( cond ) { EXPECT_TRUE( cond ); }
#endif /* clang-format on */

  inline const std::string
  boolTF( const bool& b )
  {
    return ( b ? "T" : "F" );
  }

#ifdef MGONGPU_CPPSIMD
  inline const std::string
  boolTF( const bool_v& v )
  {
    std::stringstream out;
    out << "{ " << ( v[0] ? "T" : "F" );
    for( int i = 1; i < neppV; i++ ) out << ", " << ( v[i] ? "T" : "F" );
    out << " }";
    return out.str();
  }
#endif
}

TEST( XTESTID( MG_EPOCH_PROCESS_ID ), testmisc )
{
#ifdef MGONGPUCPP_GPUIMPL
  using namespace mg5amcGpu;
#else
  using namespace mg5amcCpu;
#endif

  //--------------------------------------------------------------------------

  EXPECT_TRUE( true );

  //--------------------------------------------------------------------------

  // Vector initialization for fptype_sv
  {
    fptype_sv f{ 0 };
    EXPECT_TRUE_sv( f == 0 );
  }
  {
    fptype_sv f = fptype_sv{ 0 };
    EXPECT_TRUE_sv( f == 0 );
  }

  // Vector initialization for fptype_sv - demonstrate bug #339 in older cxmake implementation
  {
    fptype_sv f{ 1 };
    //std::cout << f << std::endl << boolTF( f == 1 ) << std::endl;
    //EXPECT_TRUE_sv( f == 1 ); // this fails for vectors! TFFF
#ifndef MGONGPU_CPPSIMD
    EXPECT_TRUE_sv( f == 1 ); // this succeds: T
#else
    EXPECT_TRUE( ( f == 1 )[0] ); // this succeds: TFFF[0]
    EXPECT_TRUE( ( f[0] == 1 ) );
    for( int i = 1; i < neppV; i++ )
    {
      EXPECT_TRUE( !( ( f == 1 )[i] ) ); // this succeds: FTTT[i>=1]
      EXPECT_TRUE( ( f[i] == 0 ) );      // equals 0, not 1
    }
#endif
  }

#ifdef MGONGPU_CPPSIMD
  // Vector initialization for cxtype_sv - demonstrate fix for bug #339
  {
    fptype_sv f1 = fptype_v{ 0 } + 1;
    EXPECT_TRUE_sv( f1 == 1 );
    cxtype_v c12 = cxmake( f1, 2 );
    //std::cout << c12 << std::endl << boolTF( c12.real() == 1 ) << std::endl << boolTF( c12.imag() == 2 ) << std::endl;
    EXPECT_TRUE_sv( c12.real() == 1 );
    EXPECT_TRUE_sv( c12.imag() == 2 );
    cxtype_v c21 = cxmake( 2, f1 );
    //std::cout << c21 << std::endl << boolTF( c21.real() == 2 ) << std::endl << boolTF( c21.imag() == 1 ) << std::endl;
    EXPECT_TRUE_sv( c21.real() == 2 );
    EXPECT_TRUE_sv( c21.imag() == 1 );
  }
#endif

  // Vector initialization for cxtype_sv
  {
    cxtype_sv c = cxzero_sv();
    EXPECT_TRUE_sv( c.real() == 0 );
    EXPECT_TRUE_sv( c.imag() == 0 );
  }
  {
    cxtype_sv c = cxmake( 1, fptype_sv{ 0 } ); // here was a bug #339
    EXPECT_TRUE_sv( c.real() == 1 );
    EXPECT_TRUE_sv( c.imag() == 0 );
  }
  {
    cxtype_sv c = cxmake( fptype_sv{ 0 }, 1 ); // here was a bug #339
    EXPECT_TRUE_sv( c.real() == 0 );
    EXPECT_TRUE_sv( c.imag() == 1 );
  }

  // Array initialization for cxtype_sv array (example: jamp_sv in CPPProcess.cc)
  {
    cxtype_sv array[2] = {}; // all zeros (NB: vector cxtype_v IS initialized to 0, but scalar cxype is NOT, if "= {}" is missing!)
    //std::cout << array[0].real() << std::endl; std::cout << boolTF( array[0].real() == 0 ) << std::endl;
    EXPECT_TRUE_sv( array[0].real() == 0 );
    EXPECT_TRUE_sv( array[0].imag() == 0 );
    EXPECT_TRUE_sv( array[1].real() == 0 );
    EXPECT_TRUE_sv( array[1].imag() == 0 );
  }

  // Alternative array initialization for cxtype_sv array (example: was used for outwf in testxxx.cc)
  {
    cxtype_sv array[2]{}; // all zeros (NB: vector cxtype_v IS initialized to 0, but scalar cxype is NOT, if "{}" is missing!)
    //std::cout << array[0].real() << std::endl; std::cout << boolTF( array[0].real() == 0 ) << std::endl;
    EXPECT_TRUE_sv( array[0].real() == 0 );
    EXPECT_TRUE_sv( array[0].imag() == 0 );
    EXPECT_TRUE_sv( array[1].real() == 0 );
    EXPECT_TRUE_sv( array[1].imag() == 0 );
  }

  //--------------------------------------------------------------------------

  // Scalar complex references
  {
    using namespace mgOnGpu;
    // Refs to f1, f2
    fptype f1 = 1;
    fptype f2 = 2;
    cxtype_ref r12( f1, f2 ); // copy refs
    //cxtype_ref r12a( r12 ); //deleted
    cxtype_ref r12a( cxtype_ref( f1, f2 ) ); // copy refs
    //cxtype_ref r12b = r12; // deleted
    cxtype_ref r12b = cxtype_ref( f1, f2 ); // copy refs
    EXPECT_TRUE( cxtype( r12 ).real() == 1 );
    EXPECT_TRUE( cxtype( r12 ).imag() == 2 );
    EXPECT_TRUE( cxtype( r12a ).real() == 1 );
    EXPECT_TRUE( cxtype( r12a ).imag() == 2 );
    EXPECT_TRUE( cxtype( r12b ).real() == 1 );
    EXPECT_TRUE( cxtype( r12b ).imag() == 2 );
    // Refs to f1c, f2c
    fptype f1c = 0;
    fptype f2c = 0;
    cxtype_ref r12c( f1c, f2c );
    EXPECT_TRUE( cxtype( r12c ).real() == 0 );
    EXPECT_TRUE( cxtype( r12c ).imag() == 0 );
    //r12c = r12; // deleted
    r12c = cxtype( r12 ); // copy values
    EXPECT_TRUE( cxtype( r12c ).real() == 1 );
    EXPECT_TRUE( cxtype( r12c ).imag() == 2 );
    // Update f1, f2
    f1 = 10;
    f2 = 20;
    EXPECT_TRUE( cxtype( r12 ).real() == 10 );
    EXPECT_TRUE( cxtype( r12 ).imag() == 20 );
    EXPECT_TRUE( cxtype( r12a ).real() == 10 );
    EXPECT_TRUE( cxtype( r12a ).imag() == 20 );
    EXPECT_TRUE( cxtype( r12b ).real() == 10 );
    EXPECT_TRUE( cxtype( r12b ).imag() == 20 );
    EXPECT_TRUE( cxtype( r12c ).real() == 1 ); // points to f1c, not to f1
    EXPECT_TRUE( cxtype( r12c ).imag() == 2 ); // points to f2c, not to f2
  }

  // Vector complex references
  {
    using namespace mgOnGpu;
    // Refs to f1, f2
    fptype_sv f1 = fptype_sv{ 0 } + 1;
    fptype_sv f2 = fptype_sv{ 0 } + 2;
    cxtype_sv_ref r12( f1, f2 ); // copy refs
    //cxtype_sv_ref r12a( r12 ); //deleted
    cxtype_sv_ref r12a( cxtype_sv_ref( f1, f2 ) ); // copy refs
    //cxtype_sv_ref r12b = r12; // deleted
    cxtype_sv_ref r12b = cxtype_sv_ref( f1, f2 ); // copy refs
    EXPECT_TRUE_sv( cxtype_sv( r12 ).real() == 1 );
    EXPECT_TRUE_sv( cxtype_sv( r12 ).imag() == 2 );
    EXPECT_TRUE_sv( cxtype_sv( r12a ).real() == 1 );
    EXPECT_TRUE_sv( cxtype_sv( r12a ).imag() == 2 );
    EXPECT_TRUE_sv( cxtype_sv( r12b ).real() == 1 );
    EXPECT_TRUE_sv( cxtype_sv( r12b ).imag() == 2 );
    // Refs to f1c, f2c
    fptype_sv f1c = fptype_sv{ 0 };
    fptype_sv f2c = fptype_sv{ 0 };
    cxtype_sv_ref r12c( f1c, f2c );
    EXPECT_TRUE_sv( cxtype_sv( r12c ).real() == 0 );
    EXPECT_TRUE_sv( cxtype_sv( r12c ).imag() == 0 );
    //r12c = r12; // deleted
    r12c = cxtype_sv( r12 ); // copy values
    EXPECT_TRUE_sv( cxtype_sv( r12c ).real() == 1 );
    EXPECT_TRUE_sv( cxtype_sv( r12c ).imag() == 2 );
    // Update f1, f2
    f1 = fptype_sv{ 0 } + 10;
    f2 = fptype_sv{ 0 } + 20;
    EXPECT_TRUE_sv( cxtype_sv( r12 ).real() == 10 );
    EXPECT_TRUE_sv( cxtype_sv( r12 ).imag() == 20 );
    EXPECT_TRUE_sv( cxtype_sv( r12a ).real() == 10 );
    EXPECT_TRUE_sv( cxtype_sv( r12a ).imag() == 20 );
    EXPECT_TRUE_sv( cxtype_sv( r12b ).real() == 10 );
    EXPECT_TRUE_sv( cxtype_sv( r12b ).imag() == 20 );
    EXPECT_TRUE_sv( cxtype_sv( r12c ).real() == 1 ); // points to f1c, not to f1
    EXPECT_TRUE_sv( cxtype_sv( r12c ).imag() == 2 ); // points to f2c, not to f2
  }

  //--------------------------------------------------------------------------

  // Boolean vector (mask) times FP vector
  /*
  // From https://github.com/madgraph5/madgraph4gpu/issues/765#issuecomment-1853672838
  channelids_sv = CHANNEL_ACCESS::kernelAccess( pchannelIds ); // the 4 channels in the SIMD vector
  bool_sv mask_sv = ( channelids_sv == 1 );
  numerators_sv += mask_sv * cxabs2( amp_sv[0] );
  if( pchannelIds != nullptr ) denominators_sv += cxabs2( amp_sv[0] );
  */
  {
    typedef bool_sv test_int_sv;  // defined as scalar_or_vector of long int (FPTYPE=double) or int (FPTYPE=float)
    test_int_sv channelids0_sv{}; // mimic CHANNEL_ACCESS::kernelAccess( pchannelIds )
    test_int_sv channelids1_sv{}; // mimic CHANNEL_ACCESS::kernelAccess( pchannelIds )
    fptype_sv absamp0_sv{};       // mimic cxabs2( amp_sv[0] )
    fptype_sv absamp1_sv{};       // mimic cxabs2( amp_sv[0] )
#ifdef MGONGPU_CPPSIMD
    for( int i = 0; i < neppV; i++ )
    {
      channelids0_sv[i] = i;   // 0123
      channelids1_sv[i] = i;   // 1234
      absamp0_sv[i] = 10. + i; // 10. 11. 12. 13.
      absamp1_sv[i] = 11. + i; // 11. 12. 13. 14.
    }
#else
    channelids0_sv = 0;
    channelids1_sv = 1;
    absamp0_sv = 10.;
    absamp1_sv = 11.;
#endif
    bool_sv mask0_sv = ( channelids0_sv % 2 == 0 ); // even channels 0123 -> TFTF (1010)
    bool_sv mask1_sv = ( channelids1_sv % 2 == 0 ); // even channels 1234 -> FTFT (0101)
    constexpr fptype_sv fpZERO_sv{};                // 0000
    //fptype_sv numerators0_sv = mask0_sv * absamp0_sv; // invalid operands to binary * ('__vector(4) long int' and '__vector(4) double')
    fptype_sv numerators0_sv = fpternary( mask0_sv, absamp0_sv, fpZERO_sv ); // equivalent to "mask0_sv * absamp0_sv"
    fptype_sv numerators1_sv = fpternary( mask1_sv, absamp1_sv, fpZERO_sv ); // equivalent to "mask1_sv * absamp1_sv"
#ifdef MGONGPU_CPPSIMD
    //std::cout << "numerators0_sv: " << numerators0_sv << std::endl;
    //std::cout << "numerators1_sv: " << numerators1_sv << std::endl;
    for( int i = 0; i < neppV; i++ )
    {
      // Values of numerators0_sv: 10.*1 11.*0 12.*1 13.*0
      if( channelids0_sv[i] % 2 == 0 ) // even channels
        EXPECT_TRUE( numerators0_sv[i] == ( 10. + i ) );
      else // odd channels
        EXPECT_TRUE( numerators0_sv[i] == 0. );
      // Values of numerators1_sv: 11.*0 12.*1 13.*0 14.*1
      if( channelids1_sv[i] % 2 == 0 ) // even channels
        EXPECT_TRUE( numerators1_sv[i] == ( 11. + i ) );
      else // odd channels
        EXPECT_TRUE( numerators1_sv[i] == 0. );
    }
#else
    // Values of numerators0_sv: 10.*1
    EXPECT_TRUE( numerators0_sv == 10. );
    // Values of numerators1_sv: 11.*0
    EXPECT_TRUE( numerators1_sv == 0. );
#endif
  }

  //--------------------------------------------------------------------------

  // Test constexpr floor
  EXPECT_TRUE( constexpr_floor( 1.5 ) == 1 );
  EXPECT_TRUE( constexpr_floor( 0.5 ) == 0 );
  EXPECT_TRUE( constexpr_floor( -0.5 ) == -1 );
  EXPECT_TRUE( constexpr_floor( -1.5 ) == -2 );

  // Test constexpr pow
  EXPECT_TRUE( constexpr_pow( 10, 0 ) == 1 );
  EXPECT_TRUE( constexpr_pow( 10, 1 ) == 10 );
  EXPECT_TRUE( constexpr_pow( 10, 2 ) == 100 );
  EXPECT_NEAR( constexpr_pow( 10, -1 ), 0.1, 0.1 * 1E-14 )
    << std::setprecision( 40 ) << "constexpr_pow( 10, -1 ) = " << constexpr_pow( 10, -1 );
  EXPECT_NEAR( constexpr_pow( 10, -2 ), 0.01, 0.01 * 1E-14 )
    << std::setprecision( 40 ) << "constexpr_pow( 10, -2 ) = " << constexpr_pow( 10, -2 );
  EXPECT_NEAR( constexpr_pow( 100, 0.5 ), 10, 10 * 1E-14 )
    << std::setprecision( 40 ) << "constexpr_pow( 100, 0.5 ) = " << constexpr_pow( 100, 0.5 );
  EXPECT_NEAR( constexpr_pow( 100, -0.5 ), 0.1, 0.1 * 1E-14 )
    << std::setprecision( 40 ) << "constexpr_pow( 100, -0.5 ) = " << constexpr_pow( 100, -0.5 );
  EXPECT_NEAR( constexpr_pow( 10000, 0.25 ), 10, 10 * 1E-14 )
    << std::setprecision( 40 ) << "constexpr_pow( 10000, 0.25 ) = " << constexpr_pow( 10000, 0.25 );
  EXPECT_NEAR( constexpr_pow( 10000, -0.25 ), 0.1, 0.1 * 1E-14 )
    << std::setprecision( 40 ) << "constexpr_pow( 10000, -0.25 ) = " << constexpr_pow( 10000, -0.25 );

#ifndef __aarch64__ // TO BE UNDERSTOOD? DISABLE CONSTEXPR_SQRT TESTS ON AARCH64 (#1064)
  // Distance from the horizontal or vertical axis (i.e. from 0, pi/2, pi, or 3pi/2)
  auto distance4 = []( const long double xx )
  {
    const long double xx2 = mapIn0to2Pi( xx );                                                    // in [0,2*pi)
    const long double xx3 = xx2 - constexpr_floor( xx2 / constexpr_pi_by_2 ) * constexpr_pi_by_2; // in [0,pi/2)
    const long double d0 = xx3;                                                                   // distance from 0
    const long double d1 = constexpr_pi_by_2 - xx3;                                               // distance from pi/2
    return ( d0 < d1 ? d0 : d1 );
  };

  // Test constexpr sin, cos, tan - specific, problematic, points
  auto testSinCosTanX = []( const long double xx, const double tolerance0, const bool debug = false, const long long istep = -999999999 )
  {
    const double x = (double)xx;
    const double tolerance = tolerance0 * ( !RUNNING_ON_VALGRIND ? 1 : 1100 ); // higher tolerance when running through valgrind #906
    if( debug )
    {
      //std::cout << std::setprecision(40) << "testSinCosTanX: xx= " << xx << std::endl;
      //std::cout << std::setprecision(40) << "                x=  " << x << std::endl;
    }
    //std::cout << std::setprecision(40) << "xx - 3pi/2 " << xx - 3 * constexpr_pi_by_2 << std::endl;
    //int width = 46;
    //char buf[128];
    //quadmath_snprintf( buf, sizeof( buf ), "%+-#*.40Qe", width, (__float128)xx );
    //std::cout << std::setprecision(40) << "testSinCosTanX: xx=" << buf << std::endl;
    //quadmath_snprintf( buf, sizeof( buf ), "%+-#*.40Qe", width, (__float128)x );
    //std::cout << std::setprecision(40) << "                x= " << buf << std::endl;
    EXPECT_NEAR( std::sin( x ), constexpr_sin( x ), std::abs( std::sin( x ) * tolerance ) )
      << "x=" << x << ", x(0to2Pi)=" << mapIn0to2Pi( x ) << ", istep=" << istep;
    EXPECT_NEAR( std::cos( x ), constexpr_cos( x ), std::abs( std::cos( x ) * tolerance ) )
      << "x=" << x << ", x(0to2Pi)=" << mapIn0to2Pi( x ) << ", istep=" << istep;
#ifndef __aarch64__
    if( !RUNNING_ON_VALGRIND )
    {
      EXPECT_NEAR( std::tan( x ), constexpr_tan( x ), std::abs( std::tan( x ) * tolerance ) )
        << "x=" << x << ", x(0to2Pi)=" << mapIn0to2Pi( x ) << ", istep=" << istep;
    }
    else
#endif
    {
      // Higher tolerance when running through valgrind #906 (or on aarch64 #1064)
      const long double ctanx = constexpr_tan( x );
      const long double taninf = 4E14; // declare tan(x) as "infinity" if above this threshold
      if( ctanx > -taninf && ctanx < taninf )
        EXPECT_NEAR( std::tan( x ), ctanx, std::abs( std::tan( x ) * tolerance ) )
          << "x=" << x << ", x(0to2Pi)=" << mapIn0to2Pi( x ) << ", istep=" << istep;
      else
      {
        // Allow tan(x)=-inf if ctan(x)=+inf and viceversa
        EXPECT_GT( std::abs( std::tan( x ) ), taninf )
          << "x=" << x << ", x(0to2Pi)=" << mapIn0to2Pi( x ) << ", istep=" << istep;
        /*
        // Require tan(x)=+inf if ctan(x)=+inf and similarly for -inf (this fails around 3*pi/2)
        if( ctanx > 0 )
          EXPECT_GT( std::tan( x ), taninf )
            << "x=" << x << ", x(0to2Pi)=" << mapIn0to2Pi( x ) << ", istep=" << istep;
        else
          EXPECT_LT( std::tan( x ), -taninf )
            << "x=" << x << ", x(0to2Pi)=" << mapIn0to2Pi( x ) << ", istep=" << istep;
        */
      }
    }
    std::cout << std::setprecision( 6 ); // default
  };
  testSinCosTanX( constexpr_pi, 1E-3, true );                                         // from math.h
  testSinCosTanX( (long double)3.141592653589793238462643383279502884L, 1E-3, true ); // from math.h
  testSinCosTanX( 4.712388980384687897640105802565813064575L, 1E-3, true );           // from 100 steps n [-4*pi,6*pi]... succeeds? (note x==xx)
  testSinCosTanX( 3 * constexpr_pi_by_2 - 1.96e-15L, 1E-3, true );                    // from 100 steps n [-4*pi,6*pi]... succeeds? (note x!=xx)
  testSinCosTanX( 3 * constexpr_pi_by_2 - 1.9601e-15L, 1E-3, true );                  // from 100 steps n [-4*pi,6*pi]... succeeds? (note x==xx)

  // Test constexpr sin, cos, tan - 8 points on (or close to) the boundaries of the 8 sectors of [0,2*pi]
  auto testSinCosTan8 = [testSinCosTanX]( const double deltax, const double tolerance )
  {
    for( int ioff = -1; ioff < 2; ioff++, ioff++ ) // -1, 1
    {
      const bool debug = false;
      const int nstep = 8;
      for( int istep = 0; istep < nstep + 1; istep++ )
      {
        long double x0 = deltax * ioff;
        long double x1 = deltax * ioff + 2 * constexpr_pi;
        double x = x0 + istep * ( x1 - x0 ) / nstep; // test this for double (else std::cos and std::sin use long double)
        testSinCosTanX( x, tolerance, debug, istep );
      }
    }
  };

  // Use much lower tolerance when testing on the boundaries of the 8 sectors of [0,2*pi]
  // Use progressively stricter tolerances as you move away from the boundaries of the 8 sectors of [0,2*pi]
  testSinCosTan8( 0, 1E-03 );     // fails with 1E-04 - DANGEROUS ANYWAY...
  testSinCosTan8( 1E-15, 1E-03 ); // fails with 1E-04 - DANGEROUS ANYWAY...
  testSinCosTan8( 1E-14, 1E-04 ); // fails with 1E-05
  testSinCosTan8( 1E-12, 1E-06 ); // fails with 1E-07
  testSinCosTan8( 1E-09, 1E-09 ); // fails with 1E-10
  testSinCosTan8( 1E-06, 1E-12 ); // fails with 1E-13
  testSinCosTan8( 1E-03, 1E-14 ); // fails with 1E-16: could use 1E-14 but keep it at 1E-14 (avoid 'EXPECT_NEAR equivalent to EXPECT_EQUAL' on Mac)
  testSinCosTan8( 1E-02, 1E-14 ); // never fails? could use 1E-99(?) but keep it at 1E-14 (avoid 'EXPECT_NEAR equivalent to EXPECT_EQUAL' on Mac)

  // Test constexpr sin, cos, tan - N points almost randomly with a varying tolerance
  auto testSinCosTanN = [distance4]( const int nstep, const double x0, const double x1 )
  {
    auto toleranceForX = [distance4]( const double x )
    {
      const double d4 = distance4( x );
      if( d4 < 1E-14 )
        return 1E-03; // NB: absolute distance limited to 1E-14 anyway even if relative tolerance is 1E-3...
      else if( d4 < 1E-13 )
        return 1E-04;
      else if( d4 < 1E-12 )
        return 1E-05;
      else if( d4 < 1E-11 )
        return 1E-06;
      else if( d4 < 1E-10 )
        return 1E-07;
      else if( d4 < 1E-09 )
        return 1E-08;
      else if( d4 < 1E-08 )
        return 1E-09;
      else if( d4 < 1E-07 )
        return 1E-10;
      else if( d4 < 1E-06 )
        return 1E-11;
      else if( d4 < 1E-05 )
        return 1E-12;
      else if( d4 < 1E-04 )
        return 1E-13;
      else
        return 1E-14; // play it safe even if the agreement might even be better?
    };
    for( int istep = 0; istep < nstep + 1; istep++ )
    {
      double x = x0 + istep * ( x1 - x0 ) / nstep; // test this for double (else std::cos and std::sin use long double)
      const double tolerance0 = toleranceForX( x );
      const double tolerance = tolerance0 * ( !RUNNING_ON_VALGRIND ? 1 : 1100 ); // higher tolerance when running through valgrind #906
      EXPECT_NEAR( std::sin( x ), constexpr_sin( x ), std::max( std::abs( std::sin( x ) * tolerance ), 3E-15 ) )
        << std::setprecision( 40 ) << "x=" << x << ", x(0to2Pi)=" << mapIn0to2Pi( x ) << ",\n istep=" << istep << ", distance4=" << distance4( x );
      EXPECT_NEAR( std::cos( x ), constexpr_cos( x ), std::max( std::abs( std::cos( x ) * tolerance ), 3E-15 ) )
        << std::setprecision( 40 ) << "x=" << x << ", x(0to2Pi)=" << mapIn0to2Pi( x ) << ",\n istep=" << istep << ", distance4=" << distance4( x );
#ifndef __aarch64__
      if( !RUNNING_ON_VALGRIND )
      {
        EXPECT_NEAR( std::tan( x ), constexpr_tan( x ), std::max( std::abs( std::tan( x ) * tolerance ), 3E-15 ) )
          << std::setprecision( 40 ) << "x=" << x << ", x(0to2Pi)=" << mapIn0to2Pi( x ) << ",\n istep=" << istep << ", distance4=" << distance4( x );
      }
      else
#endif
      {
        // Higher tolerance when running through valgrind #906 (or on aarch64 #1064)
        const long double ctanx = constexpr_tan( x );
        const long double taninf = 4E14; // declare tan(x) as "infinity if above this threshold
        if( ctanx > -taninf && ctanx < taninf )
          EXPECT_NEAR( std::tan( x ), constexpr_tan( x ), std::max( std::abs( std::tan( x ) * tolerance ), 3E-15 ) )
            << std::setprecision( 40 ) << "x=" << x << ", x(0to2Pi)=" << mapIn0to2Pi( x ) << ",\n istep=" << istep << ", distance4=" << distance4( x );
        else
        {
          // Allow tan(x)=-inf if ctan(x)=+inf and viceversa
          EXPECT_GT( std::abs( std::tan( x ) ), taninf )
            << std::setprecision( 40 ) << "x=" << x << ", x(0to2Pi)=" << mapIn0to2Pi( x ) << ",\n istep=" << istep << ", distance4=" << distance4( x );
          /*
          // Require tan(x)=+inf if ctan(x)=+inf and similarly for -inf (this fails around 3*pi/2)
          if( ctanx > 0 )
            EXPECT_GT( std::tan( x ), taninf )
              << std::setprecision( 40 ) << "x=" << x << ", x(0to2Pi)=" << mapIn0to2Pi( x ) << ",\n istep=" << istep << ", distance4=" << distance4( x );
          else
            EXPECT_LT( std::tan( x ), -taninf )
              << std::setprecision( 40 ) << "x=" << x << ", x(0to2Pi)=" << mapIn0to2Pi( x ) << ",\n istep=" << istep << ", distance4=" << distance4( x );
          */
        }
      }
    }
  };
  testSinCosTanN( 100, -4 * constexpr_pi, 6 * constexpr_pi ); // this was failing at 3*pi/2 (now fixed by absolute tolerance 3E-15)
  testSinCosTanN( 10000, -constexpr_pi_by_2, 5 * constexpr_pi_by_2 );

  // Test constexpr atan
  {
    const double tolerance = 1E-12;
    const int nstep = 1000;
    for( int istep = 0; istep < nstep + 1; istep++ )
    {
      long double x0 = -5, x1 = +5;
      double x = x0 + istep * ( x1 - x0 ) / nstep; // test this for double (else std::cos and std::sin use long double)
      EXPECT_NEAR( std::atan( x ), constexpr_atan( x ), std::abs( std::atan( x ) * tolerance ) )
        << "x=" << x << ", istep=" << istep;
    }
  }
#endif
  //--------------------------------------------------------------------------
}
