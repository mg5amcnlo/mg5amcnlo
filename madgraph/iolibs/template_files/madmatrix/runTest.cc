// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: S. Hageboeck (Nov 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: S. Hageboeck, O. Mattelaer, S. Roiser, J. Teig, A. Valassi (2020-2024).
// Integrated with the MadGraph7 project in Feb 2026.
// ----------------------------------------------------------------------------
// Use ./runTest.exe --gtest_filter=*xxx to run only testxxx.cc tests
// ----------------------------------------------------------------------------

#include "mgOnGpuConfig.h"

#include "CPPProcess.h"
#include "MadgraphTest.h"
#include "MatrixElementKernels.h"
#include "MemoryAccessChannelIds.h"
#include "MemoryAccessMatrixElements.h"
#include "MemoryAccessMomenta.h"
#include "MemoryBuffers.h"
#include "RamboSamplingKernels.h"
#include "RandomNumberKernels.h"
#include "coloramps.h"
#include "epoch_process_id.h"

#include <memory>

#ifdef MGONGPUCPP_GPUIMPL
using namespace mg5amcGpu;
#else
using namespace mg5amcCpu;
#endif

struct CUDA_CPU_TestBase : public TestDriverBase
{
  static constexpr int neppM = MemoryAccessMomenta::neppM; // AOSOA layout
  static constexpr int np4 = CPPProcess::np4;
  static constexpr int npar = CPPProcess::npar;
  static_assert( gputhreads % neppM == 0, "ERROR! #threads/block should be a multiple of neppM" );
  static_assert( gputhreads <= mgOnGpu::ntpbMAX, "ERROR! #threads/block should be <= ntpbMAX" );
  CUDA_CPU_TestBase( const std::string& refFileName )
    : TestDriverBase( npar, refFileName ) {}
  // Does this test use channelIds?
  virtual bool useChannelIds() const = 0;
  // Set channelId array (in the same way for CUDA and CPU tests)
  static constexpr unsigned int warpSize = 32; // FIXME: add a sanity check in madevent that this is the minimum? (would need to expose this from cudacpp to madevent)
  static void setChannelIds( BufferChannelIds& hstChannelIds, std::size_t iiter )
  {
    static const char* debugC = getenv( "CUDACPP_RUNTEST_DEBUG" );
    static const bool debug = ( debugC != 0 ) && ( std::string( debugC ) != "" );
    // Fill channelIds for multi-channel tests #896
    // (NB: these are only used if useChannelIds == true)
    // TEMPORARY(0): debug multichannel tests with channelId=1 for all events
    //for( unsigned int i = 0; i < nevt; ++i ) hstChannelIds[i] = 1;
    // TEMPORARY(1): debug multichannel tests with channelId=1,2,..,ndiag,1,2,..ndiag,... (every event gets a different channel, no warps)
    //for( unsigned int i = 0; i < nevt; ++i ) hstChannelIds[i] = 1 + i % CPPProcess::ndiagrams;
    // ALMOST FINAL test implementation: 1111222233331111... (every 32-event warp gets a different channel)
    // FINAL(?) test implementation: 2222333344442222... (every 32-event warp gets a different channel, skip those without associated iconfig #917)
    static_assert( nevt % warpSize == 0, "ERROR! nevt should be a multiple of warpSize" );
    constexpr unsigned int nWarp = nevt / warpSize;
    for( unsigned int iWarp = 0; iWarp < nWarp; ++iWarp )
    {
      //const unsigned int channelId = 1 + ( iWarp + iiter * nWarp ) % CPPProcess::ndiagrams; // bug #917
      const int iconfig = 1 + ( iWarp + iiter * nWarp ) % mgOnGpu::nconfigSDE;
      unsigned int channelId = 0;
      //for( unsigned int idiagram = 1; idiagram < CPPProcess::ndiagrams; idiagram++ ) // two bugs #920 and #919
      for( unsigned int idiagram = 0; idiagram < mgOnGpu::nchannels; idiagram++ ) // fix #920 and work around #919
      {
        if( mgOnGpu::hostChannel2iconfig[idiagram] == iconfig )
        {
          channelId = idiagram + 1; // fix #917 (NB add +1 because channelId uses F indexing)
          break;
        }
      }
      assert( channelId > 0 ); // sanity check that the channelId for the given iconfig was found
      if( debug ) std::cout << "CUDA_CPU_TestBase::setChannelIds: iWarp=" << iWarp << ", iconfig=" << iconfig << ", channelId=" << channelId << std::endl;
      for( unsigned int i = 0; i < warpSize; ++i )
        hstChannelIds[iWarp * warpSize + i] = channelId;
    }
  }
};

#ifndef MGONGPUCPP_GPUIMPL
struct CPUTest : public CUDA_CPU_TestBase
{
  // Struct data members (process, and memory structures for random numbers, momenta, matrix elements and weights on host and device)
  // [NB the hst/dev memory arrays must be initialised in the constructor, see issue #290]
  CPPProcess process;
  HostBufferRndNumMomenta hstRndMom;
  HostBufferChannelIds hstChannelIds;
  HostBufferMomenta hstMomenta;
  HostBufferGs hstGs;
  HostBufferRndNumHelicity hstRndHel;
  HostBufferRndNumColor hstRndCol;
  HostBufferWeights hstWeights;
  HostBufferMatrixElements hstMatrixElements;
  HostBufferSelectedHelicity hstSelHel;
  HostBufferSelectedColor hstSelCol;
  HostBufferHelicityMask hstIsGoodHel;
  std::unique_ptr<MatrixElementKernelBase> pmek;

  // Create a process object
  // Read param_card and set parameters
  // ** WARNING EVIL EVIL **
  // The CPPProcess constructor has side effects on the globals Proc::cHel, which is needed in ME calculations.
  // Don't remove!
  CPUTest( const std::string& refFileName )
    : CUDA_CPU_TestBase( refFileName )
    , process( /*verbose=*/false )
    , hstRndMom( nevt )
    , hstChannelIds( nevt )
    , hstMomenta( nevt )
    , hstGs( nevt )
    , hstRndHel( nevt )
    , hstRndCol( nevt )
    , hstWeights( nevt )
    , hstMatrixElements( nevt )
    , hstSelHel( nevt )
    , hstSelCol( nevt )
    , hstIsGoodHel( CPPProcess::ncomb )
    , pmek( new MatrixElementKernelHost( hstMomenta, hstGs, hstRndHel, hstRndCol, hstChannelIds, hstMatrixElements, hstSelHel, hstSelCol, nevt ) )
  {
    // FIXME: the process instance can happily go out of scope because it is only needed to read parameters?
    // FIXME: the CPPProcess should really be a singleton?
    process.initProc( "../../Cards/param_card.dat" );
  }

  virtual ~CPUTest() {}

  void prepareRandomNumbers( unsigned int iiter ) override
  {
    // Random numbers for momenta
    CommonRandomNumberKernel rnk( hstRndMom );
    rnk.seedGenerator( 1337 + iiter );
    rnk.generateRnarray();
    // Random numbers for helicity and color selection (fix #931)
    CommonRandomNumberKernel rnk2( hstRndHel );
    rnk2.seedGenerator( 1338 + iiter );
    rnk2.generateRnarray();
    CommonRandomNumberKernel rnk3( hstRndCol );
    rnk3.seedGenerator( 1339 + iiter );
    rnk3.generateRnarray();
  }

  void prepareMomenta( fptype energy ) override
  {
    RamboSamplingKernelHost rsk( energy, hstRndMom, hstMomenta, hstWeights, nevt );
    // --- 2a. Fill in momenta of initial state particles on the device
    rsk.getMomentaInitial();
    // --- 2b. Fill in momenta of final state particles using the RAMBO algorithm on the device
    // (i.e. map random numbers to final-state particle momenta for each of nevt events)
    rsk.getMomentaFinal();
  }

  void runSigmaKin( std::size_t iiter ) override
  {
    constexpr fptype fixedG = 1.2177157847767195; // fixed G for aS=0.118 (hardcoded for now in check_sa.cc, fcheck_sa.f, runTest.cc)
    for( unsigned int i = 0; i < nevt; ++i ) hstGs[i] = fixedG;
    setChannelIds( hstChannelIds, iiter ); // fill channelIds for multi-channel tests #896
    if( iiter == 0 ) pmek->computeGoodHelicities();
    pmek->computeMatrixElements( useChannelIds() );
  }

  fptype getMomentum( std::size_t ievt, unsigned int ipar, unsigned int ip4 ) const override
  {
    assert( ipar < npar );
    assert( ip4 < np4 );
    return MemoryAccessMomenta::ieventAccessIp4IparConst( hstMomenta.data(), ievt, ip4, ipar );
  }

  fptype getMatrixElement( std::size_t ievt ) const override
  {
    return MemoryAccessMatrixElements::ieventAccessConst( hstMatrixElements.data(), ievt );
  }

  int getChannelId( std::size_t ievt ) const override
  {
    return MemoryAccessChannelIds::ieventAccessConst( hstChannelIds.data(), ievt );
  }

  int getSelectedHelicity( std::size_t ievt ) const override
  {
    //return MemoryAccessSelectedHelicity::ieventAccessConst( hstSelHel.data(), ievt ); // does not exist yet...
    return hstSelHel.data()[ievt];
  }

  int getSelectedColor( std::size_t ievt ) const override
  {
    //return MemoryAccessSelectedColor::ieventAccessConst( hstSelCol.data(), ievt ); // does not exist yet...
    return hstSelCol.data()[ievt];
  }
};

// Old test with multi-channel disabled #466
struct CPUTestNoMultiChannel : public CPUTest
{
  // Does this test use channelIds?
  bool useChannelIds() const override final { return false; }

  // Constructor
  CPUTestNoMultiChannel( const std::string& refFileName )
    : CPUTest( refFileName ) {} // suffix .txt

  // Destructor
  virtual ~CPUTestNoMultiChannel() {}
};

// New test with multi-channel enabled #896
struct CPUTestMultiChannel : public CPUTest
{
  // Does this test use channelIds?
  bool useChannelIds() const override final { return true; }

  // Constructor
  CPUTestMultiChannel( const std::string& refFileName )
    : CPUTest( refFileName + "2" ) {} // suffix .txt2

  // Destructor
  virtual ~CPUTestMultiChannel() {}
};
#endif

#ifdef MGONGPUCPP_GPUIMPL
struct CUDATest : public CUDA_CPU_TestBase
{
  // Struct data members (process, and memory structures for random numbers, momenta, matrix elements and weights on host and device)
  // [NB the hst/dev memory arrays must be initialised in the constructor, see issue #290]
  CPPProcess process;
  PinnedHostBufferRndNumMomenta hstRndMom;
  PinnedHostBufferMomenta hstMomenta;
  PinnedHostBufferGs hstGs;
  PinnedHostBufferRndNumHelicity hstRndHel;
  PinnedHostBufferRndNumColor hstRndCol;
  PinnedHostBufferWeights hstWeights;
  PinnedHostBufferChannelIds hstChannelIds;
  PinnedHostBufferMatrixElements hstMatrixElements;
  PinnedHostBufferSelectedHelicity hstSelHel;
  PinnedHostBufferSelectedColor hstSelCol;
  PinnedHostBufferHelicityMask hstIsGoodHel;
  DeviceBufferRndNumMomenta devRndMom;
  DeviceBufferChannelIds devChannelIds;
  DeviceBufferMomenta devMomenta;
  DeviceBufferGs devGs;
  DeviceBufferRndNumHelicity devRndHel;
  DeviceBufferRndNumColor devRndCol;
  DeviceBufferWeights devWeights;
  DeviceBufferMatrixElements devMatrixElements;
  DeviceBufferSelectedHelicity devSelHel;
  DeviceBufferSelectedColor devSelCol;
  DeviceBufferHelicityMask devIsGoodHel;
  std::unique_ptr<MatrixElementKernelBase> pmek;

  // Create a process object
  // Read param_card and set parameters
  // ** WARNING EVIL EVIL **
  // The CPPProcess constructor has side effects on the globals Proc::cHel, which is needed in ME calculations.
  // Don't remove!
  CUDATest( const std::string& refFileName )
    : CUDA_CPU_TestBase( refFileName )
    , process( /*verbose=*/false )
    , hstRndMom( nevt )
    , hstChannelIds( nevt )
    , hstMomenta( nevt )
    , hstGs( nevt )
    , hstRndHel( nevt )
    , hstRndCol( nevt )
    , hstWeights( nevt )
    , hstMatrixElements( nevt )
    , hstSelHel( nevt )
    , hstSelCol( nevt )
    , hstIsGoodHel( CPPProcess::ncomb )
    , devRndMom( nevt )
    , devChannelIds( nevt )
    , devMomenta( nevt )
    , devGs( nevt )
    , devRndHel( nevt )
    , devRndCol( nevt )
    , devWeights( nevt )
    , devMatrixElements( nevt )
    , devSelHel( nevt )
    , devSelCol( nevt )
    , devIsGoodHel( CPPProcess::ncomb )
    , pmek( new MatrixElementKernelDevice( devMomenta, devGs, devRndHel, devRndCol, devChannelIds, devMatrixElements, devSelHel, devSelCol, gpublocks, gputhreads ) )
  {
    // FIXME: the process instance can happily go out of scope because it is only needed to read parameters?
    // FIXME: the CPPProcess should really be a singleton?
    process.initProc( "../../Cards/param_card.dat" );
  }

  virtual ~CUDATest() {}

  void prepareRandomNumbers( unsigned int iiter ) override
  {
    // Random numbers for momenta
    CommonRandomNumberKernel rnk( hstRndMom );
    rnk.seedGenerator( 1337 + iiter );
    rnk.generateRnarray();
    copyDeviceFromHost( devRndMom, hstRndMom );
    // Random numbers for helicity and color selection (fix #931)
    CommonRandomNumberKernel rnk2( hstRndHel );
    rnk2.seedGenerator( 1338 + iiter );
    rnk2.generateRnarray();
    copyDeviceFromHost( devRndHel, hstRndHel );
    CommonRandomNumberKernel rnk3( hstRndCol );
    rnk3.seedGenerator( 1339 + iiter );
    rnk3.generateRnarray();
    copyDeviceFromHost( devRndCol, hstRndCol );
  }

  void prepareMomenta( fptype energy ) override
  {
    RamboSamplingKernelDevice rsk( energy, devRndMom, devMomenta, devWeights, gpublocks, gputhreads );
    // --- 2a. Fill in momenta of initial state particles on the device
    rsk.getMomentaInitial();
    // --- 2b. Fill in momenta of final state particles using the RAMBO algorithm on the device
    // (i.e. map random numbers to final-state particle momenta for each of nevt events)
    rsk.getMomentaFinal();
    // --- 2c. CopyDToH Weights
    copyHostFromDevice( hstWeights, devWeights );
    // --- 2d. CopyDToH Momenta
    copyHostFromDevice( hstMomenta, devMomenta );
  }

  void runSigmaKin( std::size_t iiter ) override
  {
    constexpr fptype fixedG = 1.2177157847767195; // fixed G for aS=0.118 (hardcoded for now in check_sa.cc, fcheck_sa.f, runTest.cc)
    for( unsigned int i = 0; i < nevt; ++i ) hstGs[i] = fixedG;
    copyDeviceFromHost( devGs, hstGs );    // BUG FIX #566
    setChannelIds( hstChannelIds, iiter ); // fill channelIds for multi-channel tests #896
    copyDeviceFromHost( devChannelIds, hstChannelIds );
    if( iiter == 0 ) pmek->computeGoodHelicities();
    pmek->computeMatrixElements( useChannelIds() );
    copyHostFromDevice( hstMatrixElements, devMatrixElements );
    copyHostFromDevice( hstSelHel, devSelHel );
    copyHostFromDevice( hstSelCol, devSelCol );
  }

  fptype getMomentum( std::size_t ievt, unsigned int ipar, unsigned int ip4 ) const override
  {
    assert( ipar < npar );
    assert( ip4 < np4 );
    return MemoryAccessMomenta::ieventAccessIp4IparConst( hstMomenta.data(), ievt, ip4, ipar );
  }

  fptype getMatrixElement( std::size_t ievt ) const override
  {
    return MemoryAccessMatrixElements::ieventAccessConst( hstMatrixElements.data(), ievt );
  }

  int getChannelId( std::size_t ievt ) const override
  {
    return MemoryAccessChannelIds::ieventAccessConst( hstChannelIds.data(), ievt );
  }

  int getSelectedHelicity( std::size_t ievt ) const override
  {
    //return MemoryAccessSelectedHelicity::ieventAccessConst( hstSelHel.data(), ievt ); // does not exist yet...
    return hstSelHel.data()[ievt];
  }

  int getSelectedColor( std::size_t ievt ) const override
  {
    //return MemoryAccessSelectedColor::ieventAccessConst( hstSelCol.data(), ievt ); // does not exist yet...
    return hstSelCol.data()[ievt];
  }
};

// Old test with multi-channel disabled #466
struct CUDATestNoMultiChannel : public CUDATest
{
  // Does this test use channelIds?
  bool useChannelIds() const override final { return false; }

  // Constructor
  CUDATestNoMultiChannel( const std::string& refFileName )
    : CUDATest( refFileName ) {} // suffix .txt

  // Destructor
  virtual ~CUDATestNoMultiChannel() {}
};

// New test with multi-channel enabled #896
struct CUDATestMultiChannel : public CUDATest
{
  // Does this test use channelIds?
  bool useChannelIds() const override final { return true; }

  // Constructor
  CUDATestMultiChannel( const std::string& refFileName )
    : CUDATest( refFileName + "2" ) {} // suffix .txt2

  // Destructor
  virtual ~CUDATestMultiChannel() {}
};
#endif /* clang-format off */

// AV July 2024 much simpler class structure without the presently-unnecessary googletest templates
// This is meant as a workaround to prevent not-understood segfault #907 when adding a second test
// Note: instantiate test2 first and test1 second to ensure that the channelid printout from the dtors comes from test1 first and test2 second
#ifdef MGONGPUCPP_GPUIMPL
// CUDA test drivers
CUDATestMultiChannel driver2( MG_EPOCH_REFERENCE_FILE_NAME );
#define TESTID2( s ) s##_GPU_MULTICHANNEL
CUDATestNoMultiChannel driver1( MG_EPOCH_REFERENCE_FILE_NAME );
#define TESTID1( s ) s##_GPU_NOMULTICHANNEL
#else
// CPU test drivers
CPUTestMultiChannel driver2( MG_EPOCH_REFERENCE_FILE_NAME );
#define TESTID2( s ) s##_CPU_MULTICHANNEL
CPUTestNoMultiChannel driver1( MG_EPOCH_REFERENCE_FILE_NAME );
#define TESTID1( s ) s##_CPU_NOMULTICHANNEL
#endif
// Madgraph tests
MadgraphTest mgTest2( driver2 );
MadgraphTest mgTest1( driver1 );
// Instantiate Google test 1
#define XTESTID1( s ) TESTID1( s )
TEST( XTESTID1( MG_EPOCH_PROCESS_ID ), compareMomAndME )
{
#ifdef MGONGPU_CHANNELID_DEBUG
  driver1.pmek->setTagForNevtProcessedByChannel( "(no multichannel)" );
#endif
  mgTest1.CompareMomentaAndME( *this );
}
// Instantiate Google test 2
#define XTESTID2( s ) TESTID2( s )
TEST( XTESTID2( MG_EPOCH_PROCESS_ID ), compareMomAndME )
{
#ifdef MGONGPU_CHANNELID_DEBUG
  driver2.pmek->setTagForNevtProcessedByChannel( "(channelid array)" );
#endif
  mgTest2.CompareMomentaAndME( *this );
}
/* clang-format on */
