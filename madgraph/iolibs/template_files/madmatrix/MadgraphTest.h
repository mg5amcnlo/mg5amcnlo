// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: S. Hageboeck (Dec 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: S. Hageboeck, J. Teig, A. Valassi (2020-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MADGRAPHTEST_H_
#define MADGRAPHTEST_H_ 1

#include "mgOnGpuConfig.h"

#include "CPPProcess.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
//#ifdef __HIPCC__
//#include <experimental/filesystem> // see https://rocm.docs.amd.com/en/docs-5.4.3/CHANGELOG.html#id79
//#else
//#include <filesystem> // bypass this completely to ease portability on LUMI #803
//#endif
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef MGONGPUCPP_GPUIMPL
using mg5amcGpu::CPPProcess;
#else
using mg5amcCpu::CPPProcess;
#endif

namespace
{
  struct ReferenceData
  {
    std::vector<std::vector<std::array<fptype, CPPProcess::np4>>> momenta;
    std::vector<fptype> MEs;
    std::vector<int> ChanIds;
    std::vector<int> SelHels;
    std::vector<int> SelCols;
  };

  /// Read batches of reference data from a file and store them in a map.
  std::map<unsigned int, ReferenceData> readReferenceData( const std::string& refFileName )
  {
    std::cout << "INFO: Opening reference file " << refFileName << std::endl;
    std::ifstream referenceFile( refFileName.c_str() );
    EXPECT_TRUE( referenceFile.is_open() ) << refFileName;
    std::map<unsigned int, ReferenceData> referenceData;
    unsigned int evtNo;
    unsigned int batchNo;
    for( std::string line; std::getline( referenceFile, line ); )
    {
      std::stringstream lineStr( line );
      if( line.empty() || line[0] == '#' )
      {
        continue;
      }
      else if( line.find( "Event" ) != std::string::npos )
      {
        std::string dummy;
        lineStr >> dummy >> evtNo >> dummy >> batchNo;
      }
      else if( line.find( "ME" ) != std::string::npos )
      {
        if( evtNo <= referenceData[batchNo].MEs.size() )
          referenceData[batchNo].MEs.resize( evtNo + 1 );
        std::string dummy;
        lineStr >> dummy >> referenceData[batchNo].MEs[evtNo];
      }
      else if( line.find( "ChanId" ) != std::string::npos )
      {
        if( evtNo <= referenceData[batchNo].ChanIds.size() )
          referenceData[batchNo].ChanIds.resize( evtNo + 1 );
        std::string dummy;
        lineStr >> dummy >> referenceData[batchNo].ChanIds[evtNo];
        referenceData[batchNo].ChanIds[evtNo] = 0; // disable ChanId comparison if multichannel is not supported #976
      }
      else if( line.find( "SelHel" ) != std::string::npos )
      {
        if( evtNo <= referenceData[batchNo].SelHels.size() )
          referenceData[batchNo].SelHels.resize( evtNo + 1 );
        std::string dummy;
        lineStr >> dummy >> referenceData[batchNo].SelHels[evtNo];
      }
      else if( line.find( "SelCol" ) != std::string::npos )
      {
        if( evtNo <= referenceData[batchNo].SelCols.size() )
          referenceData[batchNo].SelCols.resize( evtNo + 1 );
        std::string dummy;
        lineStr >> dummy >> referenceData[batchNo].SelCols[evtNo];
      }
      else
      {
        unsigned int particleIndex;
        lineStr >> particleIndex;
        if( evtNo <= referenceData[batchNo].momenta.size() )
          referenceData[batchNo].momenta.resize( evtNo + 1 );
        if( particleIndex <= referenceData[batchNo].momenta[evtNo].size() )
          referenceData[batchNo].momenta[evtNo].resize( particleIndex + 1 );
        auto& fourVec = referenceData[batchNo].momenta[evtNo][particleIndex];
        for( unsigned int i = 0; i < fourVec.size(); ++i )
        {
          EXPECT_TRUE( lineStr.good() );
          lineStr >> fourVec[i];
        }
        EXPECT_TRUE( lineStr.eof() );
      }
    }
    return referenceData;
  }

}

/**
 * Test driver providing a common interface for testing different implementations.
 * Users need to implement:
 * - Functions to retrieve matrix element and 4-momenta. These are used in the tests.
 * - Driver functions that run the madgraph workflow.
 */
class TestDriverBase
{
  std::string m_refFileName;
public:
  const unsigned int nparticle;
  static constexpr unsigned int niter = 2;
  static constexpr unsigned int gpublocks = 2;
  static constexpr unsigned int gputhreads = 128;
  static constexpr unsigned int nevt = gpublocks * gputhreads;

  TestDriverBase( unsigned int npart, const std::string& refFileName )
    : m_refFileName( refFileName )
    , nparticle( npart )
  {
  }
  TestDriverBase() = delete;
  virtual ~TestDriverBase() {}
  const std::string& getRefFileName() { return m_refFileName; }

  // ------------------------------------------------
  // Interface for retrieving info from madgraph
  // ------------------------------------------------
  virtual fptype getMomentum( std::size_t evtNo, unsigned int particleNo, unsigned int component ) const = 0;
  virtual fptype getMatrixElement( std::size_t evtNo ) const = 0;
  virtual int getChannelId( std::size_t ievt ) const = 0;
  virtual int getSelectedHelicity( std::size_t ievt ) const = 0;
  virtual int getSelectedColor( std::size_t ievt ) const = 0;

  // ------------------------------------------------
  // Interface for steering madgraph run
  // ------------------------------------------------
  virtual void prepareRandomNumbers( unsigned int iiter ) = 0;
  virtual void prepareMomenta( fptype energy ) = 0;
  virtual void runSigmaKin( std::size_t iiter ) = 0;

  /// Print the requested event into the stream. If the reference data has enough events, it will be printed as well.
  void dumpParticles( std::ostream& stream, std::size_t ievt, unsigned int numParticles, unsigned int nDigit, const ReferenceData& referenceData ) const
  {
    const auto width = nDigit + 8;
    for( unsigned int ipar = 0; ipar < numParticles; ipar++ )
    {
      // NB: 'setw' affects only the next field (of any type)
      stream << std::scientific // fixed format: affects all floats (default nDigit: 6)
             << std::setprecision( nDigit )
             << std::setw( 4 ) << ipar
             << std::setw( width ) << getMomentum( ievt, ipar, 0 )
             << std::setw( width ) << getMomentum( ievt, ipar, 1 )
             << std::setw( width ) << getMomentum( ievt, ipar, 2 )
             << std::setw( width ) << getMomentum( ievt, ipar, 3 )
             << "\n";
      if( ievt < referenceData.momenta.size() )
      {
        stream << "ref" << ipar;
        stream << std::setw( width ) << referenceData.momenta[ievt][ipar][0]
               << std::setw( width ) << referenceData.momenta[ievt][ipar][1]
               << std::setw( width ) << referenceData.momenta[ievt][ipar][2]
               << std::setw( width ) << referenceData.momenta[ievt][ipar][3]
               << "\n\n";
      }
      stream << std::flush << std::defaultfloat; // default format: affects all floats
    }
  }
};

/**
 * Test class that's defining all tests to run with a Madgraph workflow.
 */
class MadgraphTest
{
public:
  MadgraphTest( TestDriverBase& testDriverRef )
    : testDriver( &testDriverRef ) {}
  ~MadgraphTest() {}
  void CompareMomentaAndME( testing::Test& googleTest ) const; // NB: googleTest is ONLY needed for the HasFailure method...
private:
  TestDriverBase* testDriver; // non-owning pointer
};

void
MadgraphTest::CompareMomentaAndME( testing::Test& googleTest ) const
{
  const fptype toleranceMomenta = std::is_same<double, fptype>::value ? 1.E-10 : 4.E-2; // see #735
#ifdef __APPLE__
  const fptype toleranceMEs = std::is_same<double, fptype>::value ? 1.E-6 : 3.E-2; // see #583
#else
  //const fptype toleranceMEs = std::is_same<double, fptype>::value ? 1.E-6 : 2.E-3; // fails smeft/hip #843
  const fptype toleranceMEs = std::is_same<double, fptype>::value ? 1.E-6 : 3.E-3;
#endif
  constexpr fptype energy = 1500; // historical default, Ecms = 1500 GeV = 1.5 TeV (above the Z peak)
  // Dump events to a new reference file?
  const char* dumpEventsC = getenv( "CUDACPP_RUNTEST_DUMPEVENTS" );
  const bool dumpEvents = ( dumpEventsC != 0 ) && ( std::string( dumpEventsC ) != "" );
  const std::string refFileName = testDriver->getRefFileName();
  /*
#ifdef __HIPCC__
  const std::string dumpFileName = std::experimental::filesystem::path( refFileName ).filename();
#else
  const std::string dumpFileName = std::filesystem::path( refFileName ).filename();
#endif
  */
  const std::string dumpFileName = refFileName; // bypass std::filesystem #803
  std::ofstream dumpFile;
  if( dumpEvents )
  {
    dumpFile.open( dumpFileName, std::ios::trunc );
  }
  // Read reference data
  std::map<unsigned int, ReferenceData> referenceData;
  if( !dumpEvents )
  {
    referenceData = readReferenceData( refFileName );
  }
  ASSERT_FALSE( googleTest.HasFailure() ); // It doesn't make any sense to continue if we couldn't read the reference file.
  // **************************************
  // *** START MAIN LOOP ON #ITERATIONS ***
  // **************************************
  for( unsigned int iiter = 0; iiter < testDriver->niter; ++iiter )
  {
    testDriver->prepareRandomNumbers( iiter );
    testDriver->prepareMomenta( energy );
    testDriver->runSigmaKin( iiter );
    // --- Run checks on all events produced in this iteration
    for( std::size_t ievt = 0; ievt < testDriver->nevt && !googleTest.HasFailure(); ++ievt )
    {
      if( dumpEvents )
      {
        ASSERT_TRUE( dumpFile.is_open() ) << dumpFileName;
        dumpFile << "Event " << std::setw( 8 ) << ievt << "  "
                 << "Batch " << std::setw( 4 ) << iiter << "\n";
        testDriver->dumpParticles( dumpFile, ievt, testDriver->nparticle, 15, ReferenceData() );
        // Dump matrix element
        dumpFile << std::setw( 4 ) << "ME" << std::scientific << std::setw( 15 + 8 )
                 << testDriver->getMatrixElement( ievt ) << "\n"
                 << std::defaultfloat;
        // Dump channelId
        dumpFile << "ChanId" << std::setw( 8 ) << testDriver->getChannelId( ievt ) << "\n";
        // Dump selected helicity and color
        dumpFile << "SelHel" << std::setw( 8 ) << testDriver->getSelectedHelicity( ievt ) << "\n";
        dumpFile << "SelCol" << std::setw( 8 ) << testDriver->getSelectedColor( ievt ) << "\n"
                 << std::endl; // leave one line between events
        continue;
      }
      // Check that we have the required reference data
      ASSERT_GT( referenceData.size(), iiter )
        << "Don't have enough reference data for iteration " << iiter << ". Ref file:" << refFileName;
      ASSERT_GT( referenceData[iiter].MEs.size(), ievt )
        << "Don't have enough reference MEs for iteration " << iiter << " event " << ievt << ".\nRef file: " << refFileName;
      ASSERT_GT( referenceData[iiter].ChanIds.size(), ievt )
        << "Don't have enough reference ChanIds for iteration " << iiter << " event " << ievt << ".\nRef file: " << refFileName;
      ASSERT_GT( referenceData[iiter].SelHels.size(), ievt )
        << "Don't have enough reference SelHels for iteration " << iiter << " event " << ievt << ".\nRef file: " << refFileName;
      ASSERT_GT( referenceData[iiter].SelCols.size(), ievt )
        << "Don't have enough reference SelCols for iteration " << iiter << " event " << ievt << ".\nRef file: " << refFileName;
      ASSERT_GT( referenceData[iiter].momenta.size(), ievt )
        << "Don't have enough reference momenta for iteration " << iiter << " event " << ievt << ".\nRef file: " << refFileName;
      ASSERT_GE( referenceData[iiter].momenta[ievt].size(), testDriver->nparticle )
        << "Don't have enough reference particles for iteration " << iiter << " event " << ievt << ".\nRef file: " << refFileName;
      // This trace will help to understand the event that is being checked.
      // It will only be printed in case of failures:
      std::stringstream eventTrace;
      eventTrace << "In comparing event " << ievt << " from iteration " << iiter << "\n";
      testDriver->dumpParticles( eventTrace, ievt, testDriver->nparticle, 15, referenceData[iiter] );
      eventTrace << std::setw( 4 ) << "ME" << std::scientific << std::setw( 15 + 8 )
                 << testDriver->getMatrixElement( ievt ) << "\n"
                 << std::setw( 4 ) << "r.ME" << std::scientific << std::setw( 15 + 8 )
                 << referenceData[iiter].MEs[ievt] << std::endl
                 << std::defaultfloat;
      eventTrace << std::setw( 8 ) << "ChanId" << std::setw( 8 ) << testDriver->getChannelId( ievt ) << "\n"
                 << std::setw( 8 ) << "r.ChanId" << std::setw( 8 ) << referenceData[iiter].ChanIds[ievt] << std::endl;
      eventTrace << std::setw( 8 ) << "SelHel" << std::setw( 8 ) << testDriver->getSelectedHelicity( ievt ) << "\n"
                 << std::setw( 8 ) << "r.SelHel" << std::setw( 8 ) << referenceData[iiter].SelHels[ievt] << std::endl;
      eventTrace << std::setw( 8 ) << "SelCol" << std::setw( 8 ) << testDriver->getSelectedColor( ievt ) << "\n"
                 << std::setw( 8 ) << "r.SelCol" << std::setw( 8 ) << referenceData[iiter].SelCols[ievt] << std::endl;
      SCOPED_TRACE( eventTrace.str() );
      // Compare Momenta
      for( unsigned int ipar = 0; ipar < testDriver->nparticle; ++ipar )
      {
        std::stringstream momentumErrors;
        for( unsigned int icomp = 0; icomp < CPPProcess::np4; ++icomp )
        {
          const fptype pMadg = testDriver->getMomentum( ievt, ipar, icomp );
          const fptype pOrig = referenceData[iiter].momenta[ievt][ipar][icomp];
          //const fptype relDelta = fabs( ( pMadg - pOrig ) / pOrig ); // computing relDelta may lead to FPEs
          const fptype delta = fabs( pMadg - pOrig );
          if( delta > toleranceMomenta * fabs( pOrig ) ) // better than "relDelta > toleranceMomenta"
          {
            momentumErrors << std::setprecision( 15 ) << std::scientific << "\nparticle " << ipar << "\tcomponent " << icomp
                           << "\n\t madGraph:  " << std::setw( 22 ) << pMadg
                           << "\n\t reference: " << std::setw( 22 ) << pOrig
                           << "\n\t relative delta exceeds tolerance of " << toleranceMomenta;
          }
        }
        ASSERT_TRUE( momentumErrors.str().empty() ) << momentumErrors.str();
      }
      // Compare ME:
      EXPECT_NEAR( testDriver->getMatrixElement( ievt ),
                   referenceData[iiter].MEs[ievt],
                   toleranceMEs * referenceData[iiter].MEs[ievt] );
      // Compare channelId
      EXPECT_EQ( testDriver->getChannelId( ievt ),
                 referenceData[iiter].ChanIds[ievt] );
      // Compare selected helicity and color
      EXPECT_EQ( testDriver->getSelectedHelicity( ievt ),
                 referenceData[iiter].SelHels[ievt] );
      EXPECT_EQ( testDriver->getSelectedColor( ievt ),
                 referenceData[iiter].SelCols[ievt] );
    }
  }
  if( dumpEvents )
  {
    std::cout << "Event dump written to " << dumpFileName << std::endl;
  }
}

#endif /* MADGRAPHTEST_H_ */
