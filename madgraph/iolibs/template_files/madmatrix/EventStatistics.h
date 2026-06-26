// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2022) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef EventStatistics_H
#define EventStatistics_H 1

#include "mgOnGpuConfig.h"

#include "CPPProcess.h" // for npar (meGeVexponent)

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //--------------------------------------------------------------------------

  // The EventStatistics struct is used to accumulate running aggregates of event statistics.
  // This will eventually include the process cross section and the process maximum weight:
  // one important case of EventStatistics will then be the "gridpack" result set, which is
  // the output of the "integration" step and the input to "unweighted event generation" step.
  // The current implementation only includes statistics for matrix elements (ME) and sampling weights (WG);
  // in first approximation, the process cross section and maximum weight are just the mean ME and maximum ME,
  // but eventually the sampling weights WG (e.g. from Rambo) must also be taken into account in the calculation.
  // The implementation uses differences to reference values to improve numerical precision.
  struct EventStatistics
  {
  public:
    size_t nevtALL;   // total number of events used
    size_t nevtABN;   // number of events used, where ME is abnormal (nevtABN <= nevtALL)
    size_t nevtZERO;  // number of not-abnormal events used, where ME is zero (nevtZERO <= nevtOK)
    double minME;     // minimum matrix element
    double maxME;     // maximum matrix element
    double minWG;     // minimum sampling weight
    double maxWG;     // maximum sampling weight
    double refME;     // "reference" matrix element (normally the current mean)
    double refWG;     // "reference" sampling weight (normally the current mean)
    double sumMEdiff; // sum of diff to ref for matrix element
    double sumWGdiff; // sum of diff to ref for sampling weight
    double sqsMEdiff; // squared sum of diff to ref for matrix element
    double sqsWGdiff; // squared sum of diff to ref for sampling weight
    std::string tag;  // a text tag for printouts
    // Number of events used, where ME is not abnormal
    size_t nevtOK() const { return nevtALL - nevtABN; }
    // Mean matrix element
    // [x = ref+d => mean(x) = sum(x)/n = ref+sum(d)/n]
    double meanME() const
    {
      return refME + ( nevtOK() > 0 ? sumMEdiff / nevtOK() : 0 );
    }
    // Mean sampling weight
    // [x = ref+d => mean(x) = sum(x)/n = ref+sum(d)/n]
    double meanWG() const
    {
      return refWG + ( nevtOK() > 0 ? sumWGdiff / nevtOK() : 0 );
    }
    // Variance matrix element
    // [x = ref+d => n*var(x) = sum((x-mean(x))^2) = sum((ref+d-ref-sum(d)/n)^2) = sum((d-sum(d)/n)^2)/n = sum(d^2)-(sum(d))^2/n]
    double varME() const { return ( sqsMEdiff - std::pow( sumMEdiff, 2 ) / nevtOK() ) / nevtOK(); }
    // Variance sampling weight
    // [x = ref+d => n*var(x) = sum((x-mean(x))^2) = sum((ref+d-ref-sum(d)/n)^2) = sum((d-sum(d)/n)^2)/n = sum(d^2)-(sum(d))^2/n]
    double varWG() const { return ( sqsWGdiff - std::pow( sumWGdiff, 2 ) / nevtOK() ) / nevtOK(); }
    // Standard deviation matrix element
    double stdME() const { return std::sqrt( varME() ); }
    // Standard deviation sampling weight
    double stdWG() const { return std::sqrt( varWG() ); }
    // Update reference matrix element
    void updateRefME( const double newRef )
    {
      const double deltaRef = refME - newRef;
      sqsMEdiff += deltaRef * ( 2 * sumMEdiff + nevtOK() * deltaRef );
      sumMEdiff += deltaRef * nevtOK();
      refME = newRef;
    }
    // Update reference sampling weight
    void updateRefWG( const double newRef )
    {
      const double deltaRef = refWG - newRef;
      sqsWGdiff += deltaRef * ( 2 * sumWGdiff + nevtOK() * deltaRef );
      sumWGdiff += deltaRef * nevtOK();
      refWG = newRef;
    }
    // Constructor
    EventStatistics()
      : nevtALL( 0 )
      , nevtABN( 0 )
      , nevtZERO( 0 )
      , minME( std::numeric_limits<double>::max() )
      , maxME( std::numeric_limits<double>::lowest() )
      , minWG( std::numeric_limits<double>::max() )
      , maxWG( std::numeric_limits<double>::lowest() )
      , refME( 0 )
      , refWG( 0 )
      , sumMEdiff( 0 )
      , sumWGdiff( 0 )
      , sqsMEdiff( 0 )
      , sqsWGdiff( 0 )
      , tag( "" ) {}
    // Combine two EventStatistics
#ifdef __clang__
    // Disable optimizations for this function in HIP (work around FPE crash #1003: originally using #if __HIP_CLANG_ONLY__)
    // Disable optimizations for this function in clang tout court (work around FPE crash #1005: now using #ifdef __clang__)
    // See https://clang.llvm.org/docs/LanguageExtensions.html#extensions-for-selectively-disabling-optimization
    __attribute__( ( optnone ) )
#endif
    EventStatistics&
    operator+=( const EventStatistics& stats )
    {
      EventStatistics s1 = *this; // temporary copy
      EventStatistics s2 = stats; // temporary copy
      EventStatistics& sum = *this;
      sum.nevtALL = s1.nevtALL + s2.nevtALL;
      sum.nevtABN = s1.nevtABN + s2.nevtABN;
      sum.nevtZERO = s1.nevtZERO + s2.nevtZERO;
      sum.minME = std::min( s1.minME, s2.minME );
      sum.maxME = std::max( s1.maxME, s2.maxME );
      sum.minWG = std::min( s1.minWG, s2.minWG );
      sum.maxWG = std::max( s1.maxWG, s2.maxWG );
      sum.refME = ( s1.meanME() * s1.nevtOK() + s2.meanME() * s2.nevtOK() ) / sum.nevtOK(); // new mean ME
      s1.updateRefME( sum.refME );
      s2.updateRefME( sum.refME );
      sum.sumMEdiff = s1.sumMEdiff + s2.sumMEdiff;
      sum.sqsMEdiff = s1.sqsMEdiff + s2.sqsMEdiff;
      sum.refWG = ( s1.meanWG() * s1.nevtOK() + s2.meanWG() * s2.nevtOK() ) / sum.nevtOK(); // new mean WG
      s1.updateRefWG( sum.refWG );
      s2.updateRefWG( sum.refWG );
      sum.sumWGdiff = s1.sumWGdiff + s2.sumWGdiff;
      sum.sqsWGdiff = s1.sqsWGdiff + s2.sqsWGdiff;
      return sum;
    }
    // Printout
    void printout( std::ostream& out ) const
    {
      const EventStatistics& s = *this;
      constexpr int meGeVexponent = -( 2 * CPPProcess::npar - 8 );
      out << s.tag << "NumMatrixElems(notAbnormal) = " << s.nevtOK() << std::endl
          << std::scientific // fixed format: affects all floats (default precision: 6)
          << s.tag << "MeanMatrixElemValue         = ( " << s.meanME()
          << " +- " << s.stdME() / std::sqrt( s.nevtOK() ) << " )  GeV^" << meGeVexponent << std::endl // standard error
          << s.tag << "[Min,Max]MatrixElemValue    = [ " << s.minME
          << " ,  " << s.maxME << " ]  GeV^" << meGeVexponent << std::endl
          << s.tag << "StdDevMatrixElemValue       = ( " << s.stdME()
          << std::string( 16, ' ' ) << " )  GeV^" << meGeVexponent << std::endl
          << s.tag << "MeanWeight                  = ( " << s.meanWG()
          << " +- " << s.stdWG() / std::sqrt( s.nevtOK() ) << std::endl // standard error
          << s.tag << "[Min,Max]Weight             = [ " << s.minWG
          << " ,  " << s.maxWG << " ]" << std::endl
          << s.tag << "StdDevWeight                = ( " << s.stdWG()
          << std::string( 16, ' ' ) << " )" << std::endl
          << std::defaultfloat; // default format: affects all floats
    }
  };

  //--------------------------------------------------------------------------

  inline std::ostream& operator<<( std::ostream& out, const EventStatistics& s )
  {
    s.printout( out );
    return out;
  }

  //--------------------------------------------------------------------------
}

#endif // EventStatistics_H
