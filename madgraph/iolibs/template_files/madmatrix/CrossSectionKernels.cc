// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2022) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#include "CrossSectionKernels.h"

#include "GpuAbstraction.h"
#include "MemoryAccessMatrixElements.h"
#include "MemoryAccessWeights.h"
#include "MemoryBuffers.h"

#include <sstream>

// ******************************************************************************************
// *** NB: Disabling fast math is essential here, otherwise results are undefined         ***
// *** NB: This file CrossSectionKernels.cc IS BUILT WITH -fno-fast-math in the Makefile! ***
// *** NB: Attempts with __attribute__((optimize("-fno-fast-math"))) were unsatisfactory  ***
// ******************************************************************************************

inline bool
fp_is_nan( const fptype& fp )
{
  //#pragma clang diagnostic push
  //#pragma clang diagnostic ignored "-Wtautological-compare" // for icpx2021/clang13 (https://stackoverflow.com/a/15864661)
  return std::isnan( fp ); // always false for clang in fast math mode (tautological compare)?
  //#pragma clang diagnostic pop
}

inline bool
fp_is_abnormal( const fptype& fp )
{
  if( fp_is_nan( fp ) ) return true;
  if( fp != fp ) return true;
  return false;
}

inline bool
fp_is_zero( const fptype& fp )
{
  if( fp == 0 ) return true;
  return false;
}

// See https://en.cppreference.com/w/cpp/numeric/math/FP_categories
inline const char*
fp_show_class( const fptype& fp )
{
  switch( std::fpclassify( fp ) )
  {
    case FP_INFINITE: return "Inf";
    case FP_NAN: return "NaN";
    case FP_NORMAL: return "normal";
    case FP_SUBNORMAL: return "subnormal";
    case FP_ZERO: return "zero";
    default: return "unknown";
  }
}

inline void
debug_me_is_abnormal( const fptype& me, size_t ievtALL )
{
  std::cout << "DEBUG[" << ievtALL << "]"
            << " ME=" << me
            << " fpisabnormal=" << fp_is_abnormal( me )
            << " fpclass=" << fp_show_class( me )
            << " (me==me)=" << ( me == me )
            << " (me==me+1)=" << ( me == me + 1 )
            << " isnan=" << fp_is_nan( me )
            << " isfinite=" << std::isfinite( me )
            << " isnormal=" << std::isnormal( me )
            << " is0=" << ( me == 0 )
            << " is1=" << ( me == 1 )
            << " abs(ME)=" << std::abs( me )
            << " isnan=" << fp_is_nan( std::abs( me ) )
            << std::endl;
}

//============================================================================

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //--------------------------------------------------------------------------

  void flagAbnormalMEs( fptype* hstMEs, unsigned int nevt )
  {
    for( unsigned int ievt = 0; ievt < nevt; ievt++ )
    {
      if( fp_is_abnormal( hstMEs[ievt] ) )
      {
        std::cout << "WARNING! flagging abnormal ME for ievt=" << ievt << std::endl;
        hstMEs[ievt] = std::sqrt( -1. );
      }
    }
  }

  //--------------------------------------------------------------------------

  CrossSectionKernelHost::CrossSectionKernelHost( const BufferWeights& samplingWeights,       // input: sampling weights
                                                  const BufferMatrixElements& matrixElements, // input: matrix elements
                                                  EventStatistics& stats,                     // output: event statistics
                                                  const size_t nevt )
    : CrossSectionKernelBase( samplingWeights, matrixElements, stats )
    , NumberOfEvents( nevt )
  {
    if( m_samplingWeights.isOnDevice() ) throw std::runtime_error( "CrossSectionKernelHost: samplingWeights must be a host array" );
    if( m_matrixElements.isOnDevice() ) throw std::runtime_error( "CrossSectionKernelHost: matrixElements must be a host array" );
    if( this->nevt() != m_samplingWeights.nevt() ) throw std::runtime_error( "CrossSectionKernelHost: nevt mismatch with samplingWeights" );
    if( this->nevt() != m_matrixElements.nevt() ) throw std::runtime_error( "CrossSectionKernelHost: nevt mismatch with matrixElements" );
  }

  //--------------------------------------------------------------------------

  void CrossSectionKernelHost::updateEventStatistics( const bool debug )
  {
    EventStatistics stats; // new statistics for the new nevt events
    // FIRST PASS: COUNT ALL/ABN/ZERO EVENTS, COMPUTE MIN/MAX, COMPUTE REFS AS MEANS OF SIMPLE SUMS
    for( size_t ievt = 0; ievt < nevt(); ++ievt ) // Loop over all events in this iteration
    {
      const fptype& me = MemoryAccessMatrixElements::ieventAccessConst( m_matrixElements.data(), ievt );
      const fptype& wg = MemoryAccessWeights::ieventAccessConst( m_samplingWeights.data(), ievt );
      const size_t ievtALL = m_iter * nevt() + ievt;
      // The following events are abnormal in a run with "-p 2048 256 12 -d"
      // - check.exe/commonrand: ME[310744,451171,3007871,3163868,4471038,5473927] with fast math
      // - check.exe/curand: ME[578162,1725762,2163579,5407629,5435532,6014690] with fast math
      // - gcheck.exe/curand: ME[596016,1446938] with fast math
      // Debug NaN/abnormal issues
      //if ( ievtALL == 310744 ) // this ME is abnormal both with and without fast math
      //  debug_me_is_abnormal( me, ievtALL );
      //if ( ievtALL == 5473927 ) // this ME is abnormal only with fast math
      //  debug_me_is_abnormal( me, ievtALL );
      stats.nevtALL++;
      if( fp_is_abnormal( me ) )
      {
        if( debug ) // only printed out with "-p -d" (matrixelementALL is not filled without -p)
          std::cout << "WARNING! ME[" << ievtALL << "] is NaN/abnormal" << std::endl;
        stats.nevtABN++;
        continue;
      }
      if( fp_is_zero( me ) ) stats.nevtZERO++;
      stats.minME = std::min( stats.minME, (double)me );
      stats.maxME = std::max( stats.maxME, (double)me );
      stats.minWG = std::min( stats.minWG, (double)wg );
      stats.maxWG = std::max( stats.maxWG, (double)wg );
      stats.sumMEdiff += me; // NB stats.refME is 0 here
      stats.sumWGdiff += wg; // NB stats.refWG is 0 here
    }
    stats.refME = stats.meanME(); // draft ref
    stats.refWG = stats.meanWG(); // draft ref
    stats.sumMEdiff = 0;
    stats.sumWGdiff = 0;
    // SECOND PASS: IMPROVE MEANS FROM SUMS OF DIFFS TO PREVIOUS REF, UPDATE REF
    for( size_t ievt = 0; ievt < nevt(); ++ievt ) // Loop over all events in this iteration
    {
      const fptype& me = MemoryAccessMatrixElements::ieventAccessConst( m_matrixElements.data(), ievt );
      const fptype& wg = MemoryAccessWeights::ieventAccessConst( m_samplingWeights.data(), ievt );
      if( fp_is_abnormal( me ) ) continue;
      stats.sumMEdiff += ( me - stats.refME );
      stats.sumWGdiff += ( wg - stats.refWG );
    }
    stats.refME = stats.meanME(); // final ref
    stats.refWG = stats.meanWG(); // final ref
    stats.sumMEdiff = 0;
    stats.sumWGdiff = 0;
    // THIRD PASS: COMPUTE STDDEV FROM SQUARED SUMS OF DIFFS TO REF
    for( size_t ievt = 0; ievt < nevt(); ++ievt ) // Loop over all events in this iteration
    {
      const fptype& me = MemoryAccessMatrixElements::ieventAccessConst( m_matrixElements.data(), ievt );
      const fptype& wg = MemoryAccessWeights::ieventAccessConst( m_samplingWeights.data(), ievt );
      if( fp_is_abnormal( me ) ) continue;
      stats.sqsMEdiff += std::pow( me - stats.refME, 2 );
      stats.sqsWGdiff += std::pow( wg - stats.refWG, 2 );
    }
    // FOURTH PASS: UPDATE THE OVERALL STATS BY ADDING THE NEW STATS
    m_stats += stats;
    // Increment the iterations counter
    m_iter++;
  }

  //--------------------------------------------------------------------------
}

//============================================================================

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
{

  /*
  //--------------------------------------------------------------------------

  CrossSectionKernelDevice::CrossSectionKernelDevice( const BufferWeights& samplingWeights,       // input: sampling weights
                                                      const BufferMatrixElements& matrixElements, // input: matrix elements
                                                      EventStatistics& stats,                     // output: event statistics
                                                      const size_t gpublocks,
                                                      const size_t gputhreads )
    : CrossSectionKernelBase( samplingWeights, matrixElements, stats )
    , NumberOfEvents( gpublocks*gputhreads )
    , m_gpublocks( gpublocks )
    , m_gputhreads( gputhreads )
  {
    if ( ! m_samplingWeights.isOnDevice() ) throw std::runtime_error( "CrossSectionKernelDevice: samplingWeights must be a device array" );
    if ( ! m_matrixElements.isOnDevice() ) throw std::runtime_error( "CrossSectionKernelDevice: matrixElements must be a device array" );
    if ( m_gpublocks == 0 ) throw std::runtime_error( "CrossSectionKernelDevice: gpublocks must be > 0" );
    if ( m_gputhreads == 0 ) throw std::runtime_error( "CrossSectionKernelDevice: gputhreads must be > 0" );
    if ( this->nevt() != m_samplingWeights.nevt() ) throw std::runtime_error( "CrossSectionKernelDevice: nevt mismatch with samplingWeights" );
    if ( this->nevt() != m_matrixElements.nevt() ) throw std::runtime_error( "CrossSectionKernelDevice: nevt mismatch with matrixElements" );
  }

  //--------------------------------------------------------------------------

  void CrossSectionKernelDevice::setGrid( const size_t gpublocks, const size_t gputhreads )
  {
    if ( m_gpublocks == 0 ) throw std::runtime_error( "CrossSectionKernelDevice: gpublocks must be > 0 in setGrid" );
    if ( m_gputhreads == 0 ) throw std::runtime_error( "CrossSectionKernelDevice: gputhreads must be > 0 in setGrid" );
    if ( this->nevt() != m_gpublocks * m_gputhreads ) throw std::runtime_error( "CrossSectionKernelDevice: nevt mismatch in setGrid" );
  }

  //--------------------------------------------------------------------------

  void CrossSectionKernelDevice::updateEventStatistics( const bool debug )
  {
    // Increment the iterations counter
    m_iter++;
  }

  //--------------------------------------------------------------------------
  */

}
#endif

//============================================================================
