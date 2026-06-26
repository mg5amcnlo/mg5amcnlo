// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Feb 2022) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#include "mgOnGpuConfig.h"

#include "Bridge.h"
#include "CPPProcess.h"
#include "MemoryBuffers.h"
#include "RamboSamplingKernels.h"
#include "RandomNumberKernels.h"

//--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  template<typename FORTRANFPTYPE>
  class Sampler final : public CppObjectInFortran
  {
  public:
    // Constructor
    // @param nevtF (VECSIZE_USED, vector.inc) number of events in Fortran arrays
    // @param nparF (NEXTERNAL, nexternal.inc) number of external particles in Fortran arrays (KEPT FOR SANITY CHECKS ONLY: remove it?)
    // @param np4F number of momenta components, usually 4, in Fortran arrays (KEPT FOR SANITY CHECKS ONLY: remove it?)
    Sampler( int nevtF, int nparF, int np4F );
    // Destructor
    virtual ~Sampler() {}
    // Delete copy/move constructors and assignment operators
    Sampler( const Sampler& ) = delete;
    Sampler( Sampler&& ) = delete;
    Sampler& operator=( const Sampler& ) = delete;
    Sampler& operator=( Sampler&& ) = delete;
    // Draw random numbers and convert them to momenta in C++, then transpose them to Fortran momenta
    void samplerHostSequence( FORTRANFPTYPE* fortranMomenta );
  private:
    const int m_nevt; // The number of events in each iteration
    int m_iiter;      // The iteration counter (for random number seeding)
#ifndef MGONGPUCPP_GPUIMPL
    HostBufferRndNumMomenta m_hstRndmom; // Memory buffers for random numbers
    HostBufferMomenta m_hstMomenta;      // Memory buffers for momenta
    HostBufferWeights m_hstWeights;      // Memory buffers for sampling weights
#else
    PinnedHostBufferRndNumMomenta m_hstRndmom; // Memory buffers for random numbers
    PinnedHostBufferMomenta m_hstMomenta;      // Memory buffers for momenta
    PinnedHostBufferWeights m_hstWeights;      // Memory buffers for sampling weights
#endif
    std::unique_ptr<RandomNumberKernelBase> m_prnk; // The appropriate RandomNumberKernel
    std::unique_ptr<SamplingKernelBase> m_prsk;     // The appropriate SamplingKernel
    // HARDCODED DEFAULTS
    static constexpr fptype energy = 1500; // historical default, Ecms = 1500 GeV = 1.5 TeV (above the Z peak)
  };

  template<typename FORTRANFPTYPE>
  Sampler<FORTRANFPTYPE>::Sampler( int nevtF, int nparF, int np4F )
    : m_nevt( nevtF )
    , m_iiter( 0 )
    , m_hstRndmom( nevtF )
    , m_hstMomenta( nevtF )
    , m_hstWeights( nevtF )
    , m_prnk( new CommonRandomNumberKernel( m_hstRndmom ) )
    , m_prsk( new RamboSamplingKernelHost( energy, m_hstRndmom, m_hstMomenta, m_hstWeights, nevtF ) )
  {
    if( nparF != CPPProcess::npar ) throw std::runtime_error( "Sampler constructor: npar mismatch" );
    if( np4F != CPPProcess::np4 ) throw std::runtime_error( "Sampler constructor: np4 mismatch" );
    std::cout << "WARNING! Instantiate host Sampler (nevt=" << m_nevt << ")" << std::endl;
  }

  // Draw random numbers and convert them to momenta in C++, then transpose them to Fortran momenta
  template<typename FORTRANFPTYPE>
  void Sampler<FORTRANFPTYPE>::samplerHostSequence( FORTRANFPTYPE* fortranMomenta )
  {
    std::cout << "Iteration #" << m_iiter + 1 << std::endl;
    // === STEP 1 OF 3
    // --- 1a. Seed rnd generator (to get same results on host and device in curand)
    // [NB This should not be necessary using the host API: "Generation functions
    // can be called multiple times on the same generator to generate successive
    // blocks of results. For pseudorandom generators, multiple calls to generation
    // functions will yield the same result as a single call with a large size."]
    // *** NB! REMEMBER THAT THE FORTRAN SAMPLER ALWAYS USES COMMON RANDOM NUMBERS! ***
    constexpr unsigned long long seed = 20200805;
    m_prnk->seedGenerator( seed + m_iiter );
    m_iiter++;
    // --- 1b. Generate all relevant numbers to build nevt events (i.e. nevt phase space points) on the host
    m_prnk->generateRnarray();
    //std::cout << "Got random numbers" << std::endl;
    // === STEP 2 OF 3
    // --- 2a. Fill in momenta of initial state particles on the device
    m_prsk->getMomentaInitial();
    //std::cout << "Got initial momenta" << std::endl;
    // --- 2b. Fill in momenta of final state particles using the RAMBO algorithm on the device
    // (i.e. map random numbers to final-state particle momenta for each of nevt events)
    m_prsk->getMomentaFinal();
    //std::cout << "Got final momenta" << std::endl;
    // --- 2c. TransposeC2F
    hst_transposeMomentaC2F( m_hstMomenta.data(), fortranMomenta, m_nevt );
  }
}

//--------------------------------------------------------------------------

extern "C"
{
#ifdef MGONGPUCPP_GPUIMPL
  using namespace mg5amcGpu;
#else
  using namespace mg5amcCpu;
#endif

  /**
   * The floating point precision used in Fortran arrays.
   * This is presently hardcoded to double precision (REAL*8).
   */
  using FORTRANFPTYPE = double; // for Fortran double precision (REAL*8) arrays
  //using FORTRANFPTYPE = float; // for Fortran single precision (REAL*4) arrays

  /**
   * Create a Sampler and return its pointer.
   * This is a C symbol that should be called from the Fortran code (in auto_dsig1.f).
   *
   * @param ppsampler the pointer to the Sampler pointer (the Sampler pointer is handled in Fortran as an INTEGER*8 variable)
   * @param nevtF the pointer to the number of events in the Fortran arrays
   * @param nparF the pointer to the number of external particles in the Fortran arrays (KEPT FOR SANITY CHECKS ONLY)
   * @param np4F the pointer to the number of momenta components, usually 4, in the Fortran arrays (KEPT FOR SANITY CHECKS ONLY)
   */
  void fsamplercreate_( CppObjectInFortran** ppsampler, const int* pnevtF, const int* pnparF, const int* pnp4F )
  {
    *ppsampler = new Sampler<FORTRANFPTYPE>( *pnevtF, *pnparF, *pnp4F );
  }

  /**
   * Delete a Sampler.
   * This is a C symbol that should be called from the Fortran code (in auto_dsig1.f).
   *
   * @param ppsampler the pointer to the Sampler pointer (the Sampler pointer is handled in Fortran as an INTEGER*8 variable)
   */
  void fsamplerdelete_( CppObjectInFortran** ppsampler )
  {
    Sampler<FORTRANFPTYPE>* psampler = dynamic_cast<Sampler<FORTRANFPTYPE>*>( *ppsampler );
    if( psampler == 0 ) throw std::runtime_error( "fsamplerdelete_: invalid Sampler address" );
    delete psampler;
  }

  /**
   * Execute the matrix-element calculation "sequence" via a Sampler on GPU/CUDA or CUDA/C++.
   * This is a C symbol that should be called from the Fortran code (in auto_dsig1.f).
   *
   * @param ppsampler the pointer to the Sampler pointer (the Sampler pointer is handled in Fortran as an INTEGER*8 variable)
   * @param momenta the pointer to the input 4-momenta
   * @param mes the pointer to the output matrix elements
   */
  void fsamplersequence_( CppObjectInFortran** ppsampler, FORTRANFPTYPE* momenta )
  {
    Sampler<FORTRANFPTYPE>* psampler = dynamic_cast<Sampler<FORTRANFPTYPE>*>( *ppsampler );
    if( psampler == 0 ) throw std::runtime_error( "fsamplersequence_: invalid Sampler address" );
    // Use the host/CPU implementation (there is no device implementation)
    psampler->samplerHostSequence( momenta );
  }
}

//--------------------------------------------------------------------------
