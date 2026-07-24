#include "mgOnGpuConfig.h"
#include "mgOnGpuTypes.h"

#include "CommonRandomNumbers.h"
#include "gCPPProcess.h"
#include "Memory.h"
#ifdef __CUDACC__
#include "grambo.cu"
#else
#include "rambo.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <unistd.h>

#include <gtest/gtest.h>


struct ReferenceData {
  std::vector< std::array<std::array<fptype,mgOnGpu::np4>, mgOnGpu::npar> > momenta;
  std::vector<fptype> MEs;
};

std::map<unsigned int, ReferenceData> readReferenceData(const std::string& refFileName);

#ifndef __CUDACC__
std::map<unsigned int, ReferenceData> readReferenceData(const std::string& refFileName)
{
  std::ifstream referenceFile(refFileName.c_str());
  EXPECT_TRUE(referenceFile.is_open()) << refFileName;
  std::map<unsigned int, ReferenceData> referenceData;
  unsigned int evtNo;
  unsigned int batchNo;

  for (std::string line; std::getline(referenceFile, line); )
  {
    std::stringstream lineStr(line);
    if (line.empty())
    {
      continue;
    }
    else if (line.find("Event") != std::string::npos)
    {
      std::string dummy;
      lineStr >> dummy >> evtNo >> dummy >> batchNo;
    }
    else if (line.find("ME") != std::string::npos)
    {
      if (evtNo <= referenceData[batchNo].MEs.size())
        referenceData[batchNo].MEs.resize(evtNo + 1);

      std::string dummy;
      lineStr >> dummy >> referenceData[batchNo].MEs[evtNo];
    }
    else
    {
      unsigned int particleIndex;
      lineStr >> particleIndex;

      if (evtNo <= referenceData[batchNo].momenta.size())
        referenceData[batchNo].momenta.resize(evtNo + 1);

      for (unsigned int i=0; i < mgOnGpu::np4; ++i) {
        EXPECT_TRUE(lineStr.good());
        lineStr >> referenceData[batchNo].momenta[evtNo][particleIndex][i];
      }
      EXPECT_TRUE(lineStr.eof());
    }
  }
  return referenceData;
}
#endif

class BaseTest : public ::testing::Test {
 protected:

  static constexpr unsigned niter = 2;
  static constexpr unsigned gpublocks = 2;
  static constexpr unsigned gputhreads = 32;
  static constexpr std::size_t nevt = gpublocks * gputhreads;

  const std::size_t nRnarray; // (NB: ASA layout with nevt=npagR*neppR events per iteration)
  const std::size_t nMomenta; // (NB: nevt=npagM*neppM for ASA layouts)
  const std::size_t nWeights;
  const std::size_t nMEs;

  BaseTest() :
    nRnarray{ mgOnGpu::np4 * mgOnGpu::nparf * nevt }, // (NB: ASA layout with nevt=npagR*neppR events per iteration)
    nMomenta{ mgOnGpu::np4 * mgOnGpu::npar  * nevt },// (NB: nevt=npagM*neppM for ASA layouts)
    nWeights{ nevt },
    nMEs    { nevt }
  { }

  virtual void prepareRandomNumbers(int iiter) = 0;
  virtual void prepareMomenta(fptype energy) = 0;
  virtual void runSigmaKin(std::size_t iiter) = 0;
};


#ifndef __CUDACC__
struct CPUTest : public BaseTest {
  Proc::CPPProcess process;

  unique_ptr_host<fptype> hstRnarray;
  unique_ptr_host<fptype> hstMomenta;
  unique_ptr_host<bool  > hstIsGoodHel;
  unique_ptr_host<fptype> hstWeights;
  unique_ptr_host<fptype> hstMEs;

  // Create a process object
  // Read param_card and set parameters
  // ** WARNING EVIL EVIL **
  // The CPPProcess constructor has side effects on the globals Proc::cHel, which is needed in ME calculations.
  // Don't remove!
  CPUTest() :
  BaseTest(),
  process(niter, gpublocks, gputhreads, /*verbose=*/false)
  {
    process.initProc("../../Cards/param_card.dat");

    // --- 0b. Allocate memory structures
    // Memory structures for random numbers, momenta, matrix elements and weights on host and device
    hstRnarray   = hstMakeUnique<fptype>( nRnarray ); // AOSOA[npagR][nparf][np4][neppR] (NB: nevt=npagR*neppR)
    hstMomenta   = hstMakeUnique<fptype>( nMomenta ); // AOSOA[npagM][npar][np4][neppM] (previously was: lp)
    hstIsGoodHel = hstMakeUnique<bool  >( mgOnGpu::ncomb );
    hstWeights   = hstMakeUnique<fptype>( nWeights ); // (previously was: meHostPtr)
    hstMEs       = hstMakeUnique<fptype>( nMEs ); // (previously was: meHostPtr)
  }
  virtual ~CPUTest() { }


  void prepareRandomNumbers(int iiter) override {
    std::vector<fptype> rnd = CommonRandomNumbers::generate<fptype>(nRnarray, 1337 + iiter);
    std::copy(rnd.begin(), rnd.end(), hstRnarray.get());
  }


  void prepareMomenta(fptype energy) override {
    // --- 2a. Fill in momenta of initial state particles on the device
    rambo2toNm0::getMomentaInitial( energy, hstMomenta.get(), nevt );

    // --- 2b. Fill in momenta of final state particles using the RAMBO algorithm on the device
    // (i.e. map random numbers to final-state particle momenta for each of nevt events)
    rambo2toNm0::getMomentaFinal( energy, hstRnarray.get(), hstMomenta.get(), hstWeights.get(), nevt );
  }


  void runSigmaKin(std::size_t /*iiter*/) override {
    // --- 3a. SigmaKin
    Proc::sigmaKin(hstMomenta.get(), hstMEs.get(), nevt);
  }
};
#endif

#ifdef __CUDACC__
struct CUDATest : public BaseTest {
  // Reset the device when our test goes out of scope. Note that this should happen after
  // the frees, i.e. be declared before the pointers to device memory.
  struct DeviceReset {
    ~DeviceReset() {
      checkCuda( cudaDeviceReset() ); // this is needed by cuda-memcheck --leak-check full
    }
  } deviceResetter;

  unique_ptr_host<fptype> hstRnarray;
  unique_ptr_host<fptype> hstMomenta;
  unique_ptr_host<bool  > hstIsGoodHel;
  unique_ptr_host<fptype> hstWeights;
  unique_ptr_host<fptype> hstMEs;

  unique_ptr_dev<fptype> devRnarray;
  unique_ptr_dev<fptype> devMomenta;
  unique_ptr_dev<bool  > devIsGoodHel;
  unique_ptr_dev<fptype> devWeights;
  unique_ptr_dev<fptype> devMEs;

  gProc::CPPProcess process;

  // Create a process object
  // Read param_card and set parameters
  // ** WARNING EVIL EVIL **
  // The CPPProcess constructor has side effects on the globals Proc::cHel, which is needed in ME calculations.
  // Don't remove!
  CUDATest() :
  BaseTest(),
  process(niter, gpublocks, gputhreads, /*verbose=*/false)
  {
    process.initProc("../../Cards/param_card.dat");

    checkCuda( cudaFree( 0 ) ); // SLOW!

    // --- 0b. Allocate memory structures
    // Memory structures for random numbers, momenta, matrix elements and weights on host and device
    hstRnarray   = hstMakeUnique<fptype>( nRnarray ); // AOSOA[npagR][nparf][np4][neppR] (NB: nevt=npagR*neppR)
    hstMomenta   = hstMakeUnique<fptype>( nMomenta ); // AOSOA[npagM][npar][np4][neppM] (previously was: lp)
    hstIsGoodHel = hstMakeUnique<bool  >( mgOnGpu::ncomb );
    hstWeights   = hstMakeUnique<fptype>( nWeights ); // (previously was: meHostPtr)
    hstMEs       = hstMakeUnique<fptype>( nMEs ); // (previously was: meHostPtr)

    devRnarray   = devMakeUnique<fptype>( nRnarray ); // AOSOA[npagR][nparf][np4][neppR] (NB: nevt=npagR*neppR)
    devMomenta   = devMakeUnique<fptype>( nMomenta ); // (previously was: allMomenta)
    devIsGoodHel = devMakeUnique<bool  >( mgOnGpu::ncomb );
    devWeights   = devMakeUnique<fptype>( nWeights ); // (previously was: meDevPtr)
    devMEs       = devMakeUnique<fptype>( nMEs ); // (previously was: meDevPtr)
  }

  virtual ~CUDATest() { }

  void prepareRandomNumbers(int iiter) override {
    std::vector<fptype> rnd = CommonRandomNumbers::generate<fptype>(nRnarray, 1337 + iiter);
    std::copy(rnd.begin(), rnd.end(), hstRnarray.get());
    checkCuda( cudaMemcpy( devRnarray.get(), hstRnarray.get(), nRnarray * sizeof(decltype(devRnarray)::element_type), cudaMemcpyHostToDevice ) );
  }


  void prepareMomenta(fptype energy) override {
    // --- 2a. Fill in momenta of initial state particles on the device
    grambo2toNm0::getMomentaInitial<<<gpublocks, gputhreads>>>( energy, devMomenta.get() );

    // --- 2b. Fill in momenta of final state particles using the RAMBO algorithm on the device
    // (i.e. map random numbers to final-state particle momenta for each of nevt events)
    grambo2toNm0::getMomentaFinal<<<gpublocks, gputhreads>>>( energy, devRnarray.get(), devMomenta.get(), devWeights.get() );

    // --- 2c. CopyDToH Weights
    checkCuda( cudaMemcpy( hstWeights.get(), devWeights.get(), nWeights * sizeof(decltype(hstWeights)::element_type), cudaMemcpyDeviceToHost ) );

    // --- 2d. CopyDToH Momenta
    checkCuda( cudaMemcpy( hstMomenta.get(), devMomenta.get(), nMomenta * sizeof(decltype(hstMomenta)::element_type), cudaMemcpyDeviceToHost ) );
  }

  void runSigmaKin(std::size_t iiter) override {
    // --- 0d. SGoodHel
    if ( iiter == 0 )
    {
      // ... 0d1. Compute good helicity mask on the device
      gProc::sigmaKin_getGoodHel<<<gpublocks, gputhreads>>>(devMomenta.get(), devIsGoodHel.get());
      checkCuda( cudaPeekAtLastError() );
      // ... 0d2. Copy back good helicity mask to the host
      checkCuda( cudaMemcpy( hstIsGoodHel.get(), devIsGoodHel.get(), mgOnGpu::ncomb * sizeof(decltype(hstIsGoodHel)::element_type), cudaMemcpyDeviceToHost ) );
      // ... 0d3. Copy back good helicity list to constant memory on the device
      gProc::sigmaKin_setGoodHel(hstIsGoodHel.get());
    }

    // --- 3a. SigmaKin
#ifndef MGONGPU_NSIGHT_DEBUG
    gProc::sigmaKin<<<gpublocks, gputhreads>>>(devMomenta.get(), devMEs.get());
#else
    gProc::sigmaKin<<<gpublocks, gputhreads, ntpbMAX*sizeof(float)>>>(devMomenta.get(), devMEs.get());
#endif
    checkCuda( cudaPeekAtLastError() );

    // --- 3b. CopyDToH MEs
    checkCuda( cudaMemcpy( hstMEs.get(), devMEs.get(), nMEs * sizeof(decltype(hstMEs)::element_type), cudaMemcpyDeviceToHost ) );
  }

};
#endif


#ifdef __CUDACC__
TEST_F(CUDATest, eemumu)
#else
TEST_F(CPUTest, eemumu)
#endif
{
  // Set to dump events:
  constexpr bool dumpEvents = false;
  const std::string dumpFileName = dumpEvents ?
      std::string("dump_") + testing::UnitTest::GetInstance()->current_test_info()->test_suite_name() + "." + testing::UnitTest::GetInstance()->current_test_info()->name() + ".txt" :
      "";
  const std::string refFileName = "dump_CPUTest.eemumu.txt";

  const int neppR = mgOnGpu::neppR; // ASA layout: constant at compile-time
  static_assert( gputhreads%neppR == 0, "ERROR! #threads/block should be a multiple of neppR" );

  const int neppM = mgOnGpu::neppM; // ASA layout: constant at compile-time
  static_assert( gputhreads%neppM == 0, "ERROR! #threads/block should be a multiple of neppM" );

  using mgOnGpu::ntpbMAX;
  static_assert( gputhreads <= ntpbMAX, "ERROR! #threads/block should be <= ntpbMAX" );

  std::ofstream dumpFile;
  if ( !dumpFileName.empty() )
  {
    dumpFile.open(dumpFileName, std::ios::trunc);
  }

  std::map<unsigned int, ReferenceData> referenceData = readReferenceData(refFileName);
  ASSERT_FALSE(HasFailure()); // It doesn't make any sense to continue if we couldn't read the reference file.

  constexpr fptype energy = 1500; // historical default, Ecms = 1500 GeV = 1.5 TeV (above the Z peak)


  // **************************************
  // *** START MAIN LOOP ON #ITERATIONS ***
  // **************************************

  for (unsigned int iiter = 0; iiter < niter; ++iiter)
  {
    prepareRandomNumbers(iiter);

    prepareMomenta(energy);

    runSigmaKin(iiter);

    // --- Run checks on all events produced in this iteration
    for (std::size_t ievt = 0; ievt < nevt && !HasFailure(); ++ievt)
    {
      auto getMomentum = [&](std::size_t evtNo, int particle, int component)
      {
        assert(component < mgOnGpu::np4);
        assert(particle  < mgOnGpu::npar);
        const auto page  = evtNo / neppM; // #eventpage in this iteration
        const auto ieppM = evtNo % neppM; // #event in the current eventpage in this iteration
        return hstMomenta[page * mgOnGpu::npar*mgOnGpu::np4*neppM + particle * neppM*mgOnGpu::np4 + component * neppM + ieppM];
      };
      auto dumpParticles = [&](std::ostream& stream, std::size_t evtNo, unsigned precision, bool dumpReference)
      {
        const auto width = precision + 8;
        for (int ipar = 0; ipar < mgOnGpu::npar; ipar++)
        {
          // NB: 'setw' affects only the next field (of any type)
          stream << std::scientific // fixed format: affects all floats (default precision: 6)
                 << std::setprecision(precision)
                 << std::setw(4) << ipar
                 << std::setw(width) << getMomentum(ievt, ipar, 0)
                 << std::setw(width) << getMomentum(ievt, ipar, 1)
                 << std::setw(width) << getMomentum(ievt, ipar, 2)
                 << std::setw(width) << getMomentum(ievt, ipar, 3)
                 << "\n";
          if (dumpReference) {
            stream << "ref" << ipar;
            if (ievt < referenceData[iiter].momenta.size()) {
              stream << std::setw(width) << referenceData[iiter].momenta[ievt][ipar][0]
                     << std::setw(width) << referenceData[iiter].momenta[ievt][ipar][1]
                     << std::setw(width) << referenceData[iiter].momenta[ievt][ipar][2]
                     << std::setw(width) << referenceData[iiter].momenta[ievt][ipar][3]
                     << "\n\n";
            } else {
              stream << "  --- No reference ---\n\n";
            }
          }
          stream << std::flush << std::defaultfloat; // default format: affects all floats
        }
      };

      if (dumpFile.is_open()) {
        dumpFile << "Event " << std::setw(8) << ievt << "  "
                 << "Batch " << std::setw(4) << iiter << "\n";
        dumpParticles(dumpFile, ievt, 15, false);
        // Dump matrix element
        dumpFile << std::setw(4) << "ME" << std::scientific << std::setw(15+8) << hstMEs[ievt] << "\n" << std::endl << std::defaultfloat;
        continue;
      }

      // This trace will only be printed in case of failures:
      std::stringstream eventTrace;
      eventTrace << "In comparing event " << ievt << " from iteration " << iiter << "\n";
      dumpParticles(eventTrace, ievt, 15, true);
      eventTrace << std::setw(4) << "ME"   << std::scientific << std::setw(15+8) << hstMEs[ievt] << "\n"
                 << std::setw(4) << "r.ME" << std::scientific << std::setw(15+8) << referenceData[iiter].MEs[ievt] << std::endl << std::defaultfloat;
      SCOPED_TRACE(eventTrace.str());

      ASSERT_LT( ievt, referenceData[iiter].momenta.size() ) << "Don't have enough events in reference file #ref=" << referenceData[iiter].momenta.size();


      // Compare Momenta
      const fptype toleranceMomenta = 200. * std::pow(10., -std::numeric_limits<fptype>::digits10);
      for (unsigned int ipar = 0; ipar < mgOnGpu::npar; ++ipar) {
        std::stringstream momentumErrors;
        for (unsigned int icomp = 0; icomp < mgOnGpu::np4; ++icomp) {
          const double pMadg = getMomentum(ievt, ipar, icomp);
          const double pOrig = referenceData[iiter].momenta[ievt][ipar][icomp];
          const double relDelta = fabs( (pMadg - pOrig)/pOrig );
          if (relDelta > toleranceMomenta) {
            momentumErrors << std::setprecision(15) << std::scientific << "\nparticle " << ipar << "\tcomponent " << icomp
                << "\n\t madGraph:  " << std::setw(22) << pMadg
                << "\n\t reference: " << std::setw(22) << pOrig
                << "\n\t rel delta: " << std::setw(22) << relDelta << " exceeds tolerance of " << toleranceMomenta;
          }
        }
        ASSERT_TRUE(momentumErrors.str().empty()) << momentumErrors.str();
      }

      // Compare ME:
      const fptype toleranceMEs = 500. * std::pow(10., -std::numeric_limits<fptype>::digits10);
      EXPECT_NEAR(hstMEs[ievt], referenceData[iiter].MEs[ievt], toleranceMEs * referenceData[iiter].MEs[ievt]);
    }


  }
}
