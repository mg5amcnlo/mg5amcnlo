// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Dec 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2021-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#include "RamboSamplingKernels.h"

#include "GpuRuntime.h"
#include "MemoryAccessMomenta.h"
#include "MemoryAccessRandomNumbers.h"
#include "MemoryAccessWeights.h"
#include "MemoryBuffers.h"
#include "rambo.h" // inline classic (massive) RAMBO, ported from standalone_cpp
#include "massless_rambo.h" // inline implementation of massless RAMBO algorithms and kernels

#include <sstream>

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //--------------------------------------------------------------------------

  MasslessRamboSamplingKernelHost::MasslessRamboSamplingKernelHost( const fptype energy,               // input: energy
                                                    const BufferRndNumMomenta& rndmom, // input: random numbers in [0,1]
                                                    BufferMomenta& momenta,            // output: momenta
                                                    BufferWeights& weights,            // output: weights
                                                    const size_t nevt )
    : SamplingKernelBase( energy, rndmom, momenta, weights )
    , NumberOfEvents( nevt )
  {
    if( m_rndmom.isOnDevice() ) throw std::runtime_error( "MasslessRamboSamplingKernelHost: rndmom must be a host array" );
    if( m_momenta.isOnDevice() ) throw std::runtime_error( "MasslessRamboSamplingKernelHost: momenta must be a host array" );
    if( m_weights.isOnDevice() ) throw std::runtime_error( "MasslessRamboSamplingKernelHost: weights must be a host array" );
    if( this->nevt() != m_rndmom.nevt() ) throw std::runtime_error( "MasslessRamboSamplingKernelHost: nevt mismatch with rndmom" );
    if( this->nevt() != m_momenta.nevt() ) throw std::runtime_error( "MasslessRamboSamplingKernelHost: nevt mismatch with momenta" );
    if( this->nevt() != m_weights.nevt() ) throw std::runtime_error( "MasslessRamboSamplingKernelHost: nevt mismatch with weights" );
    // Sanity checks for memory access (momenta buffer)
    constexpr int neppM = MemoryAccessMomenta::neppM; // AOSOA layout
    static_assert( ispoweroftwo( neppM ), "neppM is not a power of 2" );
    if( nevt % neppM != 0 )
    {
      std::ostringstream sstr;
      sstr << "MasslessRamboSamplingKernelHost: nevt should be a multiple of neppM=" << neppM;
      throw std::runtime_error( sstr.str() );
    }
    // Sanity checks for memory access (random number buffer)
    constexpr int neppR = MemoryAccessRandomNumbers::neppR; // AOSOA layout
    static_assert( ispoweroftwo( neppR ), "neppR is not a power of 2" );
    if( nevt % neppR != 0 )
    {
      std::ostringstream sstr;
      sstr << "MasslessRamboSamplingKernelHost: nevt should be a multiple of neppR=" << neppR;
      throw std::runtime_error( sstr.str() );
    }
  }

  //--------------------------------------------------------------------------

  void
  MasslessRamboSamplingKernelHost::getMomentaInitial()
  {
    constexpr auto getMomentaInitial = massless_rambo::ramboGetMomentaInitial<HostAccessMomenta>;
    // ** START LOOP ON IEVT **
    for( size_t ievt = 0; ievt < nevt(); ++ievt )
    {
      // NB all KernelLaunchers assume that memory access can be decomposed as "accessField = decodeRecord( accessRecord )"
      fptype* ievtMomenta = MemoryAccessMomenta::ieventAccessRecord( m_momenta.data(), ievt );
      getMomentaInitial( m_energy, ievtMomenta );
    }
    // ** END LOOP ON IEVT **
  }

  //--------------------------------------------------------------------------

  void
  MasslessRamboSamplingKernelHost::getMomentaFinal()
  {
    constexpr auto getMomentaFinal = massless_rambo::ramboGetMomentaFinal<HostAccessRandomNumbers, HostAccessMomenta, HostAccessWeights>;
    // ** START LOOP ON IEVT **
    for( size_t ievt = 0; ievt < nevt(); ++ievt )
    {
      // NB all KernelLaunchers assume that memory access can be decomposed as "accessField = decodeRecord( accessRecord )"
      const fptype* ievtRndmom = MemoryAccessRandomNumbers::ieventAccessRecordConst( m_rndmom.data(), ievt );
      fptype* ievtMomenta = MemoryAccessMomenta::ieventAccessRecord( m_momenta.data(), ievt );
      fptype* ievtWeights = MemoryAccessWeights::ieventAccessRecord( m_weights.data(), ievt );
      getMomentaFinal( m_energy, ievtRndmom, ievtMomenta, ievtWeights );
    }
    // ** END LOOP ON IEVT **
  }

  //--------------------------------------------------------------------------

  RamboSamplingKernelHost::RamboSamplingKernelHost( const fptype energy,               // input: energy
                                                                  const BufferRndNumMomenta& rndmom, // input: random [0,1] UNUSED
                                                                  const std::vector<fptype>& masses, // input: external-leg masses
                                                                  const int ninitial,                // input: #initial-state particles
                                                                  const size_t nevt,                 // input: #events
                                                                  BufferMomenta& momenta,            // output: momenta
                                                                  BufferWeights& weights )           // output: weights
    : SamplingKernelBase( energy, rndmom, momenta, weights )
    , NumberOfEvents( nevt )
    , m_masses( masses.begin(), masses.end() )
    , m_ninitial( ninitial )
  {
    if( m_momenta.isOnDevice() ) throw std::runtime_error( "RamboSamplingKernelHost: momenta must be a host array" );
    if( m_weights.isOnDevice() ) throw std::runtime_error( "RamboSamplingKernelHost: weights must be a host array" );
    if( this->nevt() != m_momenta.nevt() ) throw std::runtime_error( "RamboSamplingKernelHost: nevt mismatch with momenta" );
    if( this->nevt() != m_weights.nevt() ) throw std::runtime_error( "RamboSamplingKernelHost: nevt mismatch with weights" );

    constexpr int neppM = MemoryAccessMomenta::neppM; // AOSOA layout

    static_assert( ispoweroftwo( neppM ), "neppM is not a power of 2" );
    if( nevt % neppM != 0 )
    {
      std::ostringstream sstr;
      sstr << "RamboSamplingKernelHost: nevt should be a multiple of neppM=" << neppM;
      throw std::runtime_error( sstr.str() );
    }
  }

  //--------------------------------------------------------------------------

  void
  RamboSamplingKernelHost::getMomentaInitial()
  {
     // NOOP
  }

  //--------------------------------------------------------------------------

  void
  RamboSamplingKernelHost::getMomentaFinal()
  {
    const int npar = (int)m_masses.size();
    // ** START LOOP ON IEVT **
    for( size_t ievt = 0; ievt < nevt(); ++ievt )
    {
      // Clas. RAMBO returns [E,px,py,pz] vector per ex. particle
      // own RNG, intial final once
      // For reproducibility betwn fptype = FP32/FP64 generation in FP64
      double wgt = 0.;
      const std::vector<std::vector<double>> point =
        rambo::get_momenta( m_ninitial, (double)m_energy, m_masses, wgt );
      for( int ipar = 0; ipar < npar; ++ipar )
        for( int ip4 = 0; ip4 < 4; ++ip4 )
          MemoryAccessMomenta::ieventAccessIp4Ipar( m_momenta.data(), ievt, ip4, ipar ) = (fptype)point[ipar][ip4];
      MemoryAccessWeights::ieventAccess( m_weights.data(), ievt ) = (fptype)wgt;
    }
    // ** END LOOP ON IEVT **
  }

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  MasslessRamboSamplingKernelDevice::MasslessRamboSamplingKernelDevice( const fptype energy,               // input: energy
                                                        const BufferRndNumMomenta& rndmom, // input: random numbers in [0,1]
                                                        BufferMomenta& momenta,            // output: momenta
                                                        BufferWeights& weights,            // output: weights
                                                        const size_t gpublocks,
                                                        const size_t gputhreads )
    : SamplingKernelBase( energy, rndmom, momenta, weights )
    , NumberOfEvents( gpublocks * gputhreads )
    , m_gpublocks( gpublocks )
    , m_gputhreads( gputhreads )
  {
    if( !m_rndmom.isOnDevice() ) throw std::runtime_error( "MasslessRamboSamplingKernelDevice: rndmom must be a device array" );
    if( !m_momenta.isOnDevice() ) throw std::runtime_error( "MasslessRamboSamplingKernelDevice: momenta must be a device array" );
    if( !m_weights.isOnDevice() ) throw std::runtime_error( "MasslessRamboSamplingKernelDevice: weights must be a device array" );
    if( m_gpublocks == 0 ) throw std::runtime_error( "MasslessRamboSamplingKernelDevice: gpublocks must be > 0" );
    if( m_gputhreads == 0 ) throw std::runtime_error( "MasslessRamboSamplingKernelDevice: gputhreads must be > 0" );
    if( this->nevt() != m_rndmom.nevt() ) throw std::runtime_error( "MasslessRamboSamplingKernelDevice: nevt mismatch with rndmom" );
    if( this->nevt() != m_momenta.nevt() ) throw std::runtime_error( "MasslessRamboSamplingKernelDevice: nevt mismatch with momenta" );
    if( this->nevt() != m_weights.nevt() ) throw std::runtime_error( "MasslessRamboSamplingKernelDevice: nevt mismatch with weights" );
    // Sanity checks for memory access (momenta buffer)
    constexpr int neppM = MemoryAccessMomenta::neppM; // AOSOA layout
    static_assert( ispoweroftwo( neppM ), "neppM is not a power of 2" );
    if( m_gputhreads % neppM != 0 )
    {
      std::ostringstream sstr;
      sstr << "MasslessRamboSamplingKernelHost: gputhreads should be a multiple of neppM=" << neppM;
      throw std::runtime_error( sstr.str() );
    }
    // Sanity checks for memory access (random number buffer)
    constexpr int neppR = MemoryAccessRandomNumbers::neppR; // AOSOA layout
    static_assert( ispoweroftwo( neppR ), "neppR is not a power of 2" );
    if( m_gputhreads % neppR != 0 )
    {
      std::ostringstream sstr;
      sstr << "MasslessRamboSamplingKernelDevice: gputhreads should be a multiple of neppR=" << neppR;
      throw std::runtime_error( sstr.str() );
    }
  }
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  __global__ void
  getMomentaInitialDevice( const fptype energy,
                           fptype* momenta )
  {
    constexpr auto getMomentaInitial = massless_rambo::ramboGetMomentaInitial<DeviceAccessMomenta>;
    return getMomentaInitial( energy, momenta );
  }
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  void
  MasslessRamboSamplingKernelDevice::getMomentaInitial()
  {
    gpuLaunchKernel( getMomentaInitialDevice, m_gpublocks, m_gputhreads, m_energy, m_momenta.data() );
  }
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  __global__ void
  getMomentaFinalDevice( const fptype energy,
                         const fptype* rndmom,
                         fptype* momenta,
                         fptype* wgts )
  {
    constexpr auto getMomentaFinal = massless_rambo::ramboGetMomentaFinal<DeviceAccessRandomNumbers, DeviceAccessMomenta, DeviceAccessWeights>;
    return getMomentaFinal( energy, rndmom, momenta, wgts );
  }
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  void
  MasslessRamboSamplingKernelDevice::getMomentaFinal()
  {
    gpuLaunchKernel( getMomentaFinalDevice, m_gpublocks, m_gputhreads, m_energy, m_rndmom.data(), m_momenta.data(), m_weights.data() );
  }
#endif

  //--------------------------------------------------------------------------
}
