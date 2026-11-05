// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2022) for the MadGraph7 CUDACPP plugin.
// Further modified by: D. Massaro, J. Teig, A. Thete, A. Valassi, Z. Wettersten (2022-2025).
// Integrated with the MadGraph7 project in Feb 2026.

#include "MatrixElementKernels.h"

#include "CPPProcess.h" // for CPPProcess::ncolor_flow
#include "ProcessData.h"
#include "SigmaKin.h"
#include "GpuRuntime.h" // Includes the abstraction for Nvidia/AMD compilation
#include "MemoryAccessMomenta.h"
#include "MemoryBuffers.h"
#include "color_sum.h" // for blasColorSumTmpSize

#include <algorithm> // for std::max
#include <cfenv> // for fetestexcept
#include <iostream>
#include <sstream>

//============================================================================

namespace madmatrix
{
  //--------------------------------------------------------------------------

  MatrixElementKernelBase::MatrixElementKernelBase( ProcessParameters& params,            // input: parameters of the process (output: good helicities)
                                                    const BufferMomenta& momenta,         // input: momenta
                                                    const BufferGs& gs,                   // input: gs for alphaS
                                                    const BufferIflavorVec& iflavorVec,   // input: flavor indices for the flavor combination
                                                    const BufferRndNumHelicity& rndhel,   // input: random numbers for helicity selection
                                                    const BufferRndNumColor& rndcol,      // input: random numbers for color selection
                                                    const BufferChannelIds& channelIds,   // input: channel ids for single-diagram enhancement
                                                    BufferMatrixElements& matrixElements, // output: matrix elements
                                                    BufferSelectedHelicity& selhel,       // output: helicity selection
                                                    BufferSelectedColor& selcol )         // output: color selection
    : m_params( params )
    , m_momenta( momenta )
    , m_gs( gs )
    , m_iflavorVec( iflavorVec )
    , m_rndhel( rndhel )
    , m_rndcol( rndcol )
    , m_channelIds( channelIds )
    , m_matrixElements( matrixElements )
    , m_selhel( selhel )
    , m_selcol( selcol )
#ifdef MGONGPU_CHANNELID_DEBUG
    , m_nevtProcessedByChannel()
    , m_tag()
#endif
  {
    //std::cout << "DEBUG: MatrixElementKernelBase ctor " << this << std::endl;
#ifdef MGONGPU_CHANNELID_DEBUG
    for( size_t channelId = 0; channelId < ProcessData::ndiagrams + 1; channelId++ ) // [0...ndiagrams] (TEMPORARY: 0=multichannel)
      m_nevtProcessedByChannel[channelId] = 0;
#endif
  }

  //--------------------------------------------------------------------------

  MatrixElementKernelBase::~MatrixElementKernelBase()
  {
    //std::cout << "DEBUG: MatrixElementKernelBase dtor " << this << std::endl;
#ifdef MGONGPU_CHANNELID_DEBUG
    MatrixElementKernelBase::dumpNevtProcessedByChannel();
#endif
#ifdef MGONGPUCPP_VERBOSE
    MatrixElementKernelBase::dumpSignallingFPEs();
#endif
  }

  //--------------------------------------------------------------------------

#ifdef MGONGPU_CHANNELID_DEBUG
  void MatrixElementKernelBase::updateNevtProcessedByChannel( const unsigned int* pHstChannelIds, const size_t nevt )
  {
    if( pHstChannelIds != nullptr )
    {
      //std::cout << "DEBUG " << this << ": not nullptr " << nevt << std::endl;
      for( unsigned int ievt = 0; ievt < nevt; ievt++ )
      {
        const size_t channelId = pHstChannelIds[ievt]; // Fortran indexing
        //assert( channelId > 0 );
        //assert( channelId < ProcessData::ndiagrams );
        m_nevtProcessedByChannel[channelId]++;
      }
    }
    else
    {
      //std::cout << "DEBUG " << this << ": nullptr " << std::endl;
      m_nevtProcessedByChannel[0] += nevt;
    }
  }
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPU_CHANNELID_DEBUG
  void MatrixElementKernelBase::dumpNevtProcessedByChannel()
  {
    size_t nevtProcessed = 0;
    for( size_t channelId = 0; channelId < ProcessData::ndiagrams + 1; channelId++ ) // [0...ndiagrams] (TEMPORARY: 0=multichannel)
      nevtProcessed += m_nevtProcessedByChannel[channelId];
    std::ostringstream sstr;
    sstr << " {";
    for( size_t channelId = 0; channelId < ProcessData::ndiagrams + 1; channelId++ ) // [0...ndiagrams] (TEMPORARY: 0=multichannel)
    {
      if( m_nevtProcessedByChannel[channelId] > 0 )
      {
        if( sstr.str() != " {" ) sstr << ",";
        if( channelId == 0 )
          sstr << " no-multichannel";
        else
          sstr << " " << channelId;
        sstr << " : " << m_nevtProcessedByChannel[channelId];
      }
    }
    sstr << " }";
    std::cout << "DEBUG: MEK " << this;
    if( m_tag != "" ) std::cout << " " << m_tag;
    std::cout << " processed " << nevtProcessed << " events across " << ProcessData::ndiagrams << " channels" << sstr.str() << std::endl;
  }
#endif

  //--------------------------------------------------------------------------

  void MatrixElementKernelBase::dumpSignallingFPEs()
  {
    // New strategy for issue #831: add a final report of FPEs
    // Note: normally only underflow will be reported here (inexact is switched off because it would almost always signal;
    // divbyzero, invalid and overflow are configured by feenablexcept to send a SIGFPE signal, and are normally fixed in the code)
    // Note: this is now called in the individual destructors of MEK classes rather than in that of MatrixElementKernelBase(#837)
    std::string fpes;
    if( std::fetestexcept( FE_DIVBYZERO ) ) fpes += " FE_DIVBYZERO";
    if( std::fetestexcept( FE_INVALID ) ) fpes += " FE_INVALID";
    if( std::fetestexcept( FE_OVERFLOW ) ) fpes += " FE_OVERFLOW";
    if( std::fetestexcept( FE_UNDERFLOW ) ) fpes += " FE_UNDERFLOW";
    //if( std::fetestexcept( FE_INEXACT ) ) fpes += " FE_INEXACT"; // do not print this out: this would almost always signal!
    if( fpes == "" )
      std::cout << "INFO: No Floating Point Exceptions have been reported" << std::endl;
    else
      std::cerr << "INFO: The following Floating Point Exceptions have been reported:" << fpes << std::endl;
  }

  //--------------------------------------------------------------------------
}

//============================================================================


//============================================================================

namespace madmatrix
{

  //--------------------------------------------------------------------------

  MatrixElementKernelDevice::MatrixElementKernelDevice( ProcessParameters& params,            // input: parameters of the process (output: good helicities)
                                                        const BufferMomenta& momenta,         // input: momenta
                                                        const BufferGs& gs,                   // input: gs for alphaS
                                                        const BufferIflavorVec& iflavorVec,   // input: flavor indices for the flavor combination
                                                        const BufferRndNumHelicity& rndhel,   // input: random numbers for helicity selection
                                                        const BufferRndNumColor& rndcol,      // input: random numbers for color selection
                                                        const BufferChannelIds& channelIds,   // input: channel ids for single-diagram enhancement
                                                        BufferMatrixElements& matrixElements, // output: matrix elements
                                                        BufferSelectedHelicity& selhel,       // output: helicity selection
                                                        BufferSelectedColor& selcol,          // output: color selection
                                                        const size_t gpublocks,
                                                        const size_t gputhreads)
    : MatrixElementKernelBase( params, momenta, gs, iflavorVec, rndhel, rndcol, channelIds, matrixElements, selhel, selcol )
    , NumberOfEvents( gpublocks * gputhreads )
    , m_deviceParams()
    , m_couplings( this->nevt() )
    , m_pHelMEs()
    , m_pHelJamps()
    , m_pHelNumerators()
    , m_pHelDenominators()
    , m_colJamp2s( CPPProcess::ncolor_flow * this->nevt() )
#ifdef MGONGPU_CHANNELID_DEBUG
    , m_hstChannelIds( this->nevt() )
#endif
#ifndef MGONGPU_HAS_NO_BLAS
    , m_blasColorSum( false )
    , m_blasTf32Tensor( false )
    , m_pHelBlasTmp()
    , m_blasHandle()
#endif
    , m_helStreams()
    , m_gpublocks( gpublocks )
    , m_gputhreads( gputhreads )
  {
    //std::cout << "DEBUG: MatrixElementKernelDevice::ctor " << this << std::endl;
    if( !m_momenta.isOnDevice() ) throw std::runtime_error( "MatrixElementKernelDevice: momenta must be a device array" );
    if( !m_matrixElements.isOnDevice() ) throw std::runtime_error( "MatrixElementKernelDevice: matrixElements must be a device array" );
    if( !m_channelIds.isOnDevice() ) throw std::runtime_error( "MatrixElementKernelDevice: channelIds must be a device array" ); // FIXME?!
    if( !m_iflavorVec.isOnDevice() ) throw std::runtime_error( "MatrixElementKernelDevice: iflavorVec must be a device array" );
    if( m_gpublocks == 0 ) throw std::runtime_error( "MatrixElementKernelDevice: gpublocks must be > 0" );
    if( m_gputhreads == 0 ) throw std::runtime_error( "MatrixElementKernelDevice: gputhreads must be > 0" );
    if( this->nevt() != m_momenta.nevt() ) throw std::runtime_error( "MatrixElementKernelDevice: nevt mismatch with momenta" );
    if( this->nevt() != m_matrixElements.nevt() ) throw std::runtime_error( "MatrixElementKernelDevice: nevt mismatch with matrixElements" );
    if( this->nevt() != m_channelIds.nevt() ) throw std::runtime_error( "MatrixElementKernelDevice: nevt mismatch with channelIds" );
    if( this->nevt() != m_iflavorVec.nevt() ) throw std::runtime_error( "MatrixElementKernelDevice: nevt mismatch with iflavorVec" );
    // Sanity checks for memory access (momenta buffer)
    constexpr int neppM = MemoryAccessMomenta::neppM; // AOSOA layout
    static_assert( ispoweroftwo( neppM ), "neppM is not a power of 2" );
    if( m_gputhreads % neppM != 0 )
    {
      std::ostringstream sstr;
      sstr << "MatrixElementKernelHost: gputhreads should be a multiple of neppM=" << neppM;
      throw std::runtime_error( sstr.str() );
    }
    // Create the "one-helicity" jamp buffer that will be used for helicity filtering
    m_pHelJamps.reset( new DeviceBufferAmp( ProcessData::njampso * mgOnGpu::nx2 * this->nevt() ) );
    // Create the "one-helicity" numerator and denominator buffers that will be used for helicity filtering
    m_pHelNumerators.reset( new DeviceBufferSimple( this->nevt() * ProcessData::ndiagrams ) );
    m_pHelDenominators.reset( new DeviceBufferSimple( this->nevt() ) );
    // Decide at runtime whether to use BLAS for color sums
    // Decide at runtime whether TF32TENSOR math should be used in cuBLAS
    static bool first = true;
    if( first )
    {
      first = false;
      // Analyse environment variable CUDACPP_RUNTIME_BLASCOLORSUM
      const char* blasEnv = getenv( "CUDACPP_RUNTIME_BLASCOLORSUM" );
      if( blasEnv && std::string( blasEnv ) != "" )
      {
#ifndef MGONGPU_HAS_NO_BLAS
        if constexpr( ProcessData::nampso > 1 )
        {
          // the BLAS color sum contracts one ncolor jamp vector per helicity: it cannot pair
          // the jamps of several amplitude split orders, which color_sum_kernel does
          std::cout << "WARNING! Env variable CUDACPP_RUNTIME_BLASCOLORSUM is set and non-empty, but the BLAS color sum does not pair split amplitude orders: disable BLAS" << std::endl;
        }
        else
        {
          m_blasColorSum = true; // fixme? eventually set default=true and decode "Y" and "N" choices?
          std::cout << "INFO: Env variable CUDACPP_RUNTIME_BLASCOLORSUM is set and non-empty: enable BLAS" << std::endl;
        }
#else
        throw std::runtime_error( "Env variable CUDACPP_RUNTIME_BLASCOLORSUM is set and non-empty, but BLAS was disabled at build time" );
#endif
      }
      else
      {
#ifndef MGONGPU_HAS_NO_BLAS
        std::cout << "INFO: Env variable CUDACPP_RUNTIME_BLASCOLORSUM is empty or not set: disable BLAS" << std::endl;
#else
        std::cout << "INFO: BLAS was disabled at build time" << std::endl;
#endif
      }
#ifndef MGONGPU_HAS_NO_BLAS
#ifdef __CUDACC__ // this must be __CUDACC__
      // Analyse environment variable CUDACPP_RUNTIME_CUBLASTF32TENSOR
      const char* blasEnv2 = getenv( "CUDACPP_RUNTIME_CUBLASTF32TENSOR" );
      if( blasEnv2 && std::string( blasEnv2 ) != "" )
      {
        if( m_blasColorSum )
        {
#ifdef MGONGPU_FPTYPE2_FLOAT
          m_blasTf32Tensor = true;
          std::cout << "INFO: Env variable CUDACPP_RUNTIME_CUBLASTF32TENSOR is set and non-empty: enable CUBLAS_TF32_TENSOR_OP_MATH" << std::endl;
#else
          std::cout << "WARNING! Env variable CUDACPP_RUNTIME_CUBLASTF32TENSOR is set and non-empty, but color sums use FP64" << std::endl;
#endif
        }
        else
          std::cout << "WARNING! Env variable CUDACPP_RUNTIME_CUBLASTF32TENSOR is set and non-empty, but BLAS was disabled at runtime" << std::endl;
      }
#ifdef MGONGPU_FPTYPE2_FLOAT
      else
      {
        if( m_blasColorSum )
          std::cout << "INFO: Env variable CUDACPP_RUNTIME_CUBLASTF32TENSOR is empty or not set: keep cuBLAS math defaults" << std::endl;
      }
#endif
#endif
#endif
    }
  }

  //--------------------------------------------------------------------------

  MatrixElementKernelDevice::~MatrixElementKernelDevice()
  {
    //std::cout << "DEBUG: MatrixElementKernelDevice::dtor " << this << std::endl;
#ifndef MGONGPU_HAS_NO_BLAS
    if( m_blasHandle ) gpuBlasDestroy( m_blasHandle );
#endif
    for( int ihel = 0; ihel < ProcessData::ncomb; ihel++ )
    {
      if( m_helStreams[ihel] ) gpuStreamDestroy( m_helStreams[ihel] ); // do not destroy if nullptr
    }
  }

  //--------------------------------------------------------------------------

  // FIXME! The relevance of this function should be reassessed (#543 and #902)
  void MatrixElementKernelDevice::setGrid( const int /*gpublocks*/, const int /*gputhreads*/ )
  {
    if( m_gpublocks == 0 ) throw std::runtime_error( "MatrixElementKernelDevice: gpublocks must be > 0 in setGrid" );
    if( m_gputhreads == 0 ) throw std::runtime_error( "MatrixElementKernelDevice: gputhreads must be > 0 in setGrid" );
    if( this->nevt() != m_gpublocks * m_gputhreads ) throw std::runtime_error( "MatrixElementKernelDevice: nevt mismatch in setGrid" );
  }

  //--------------------------------------------------------------------------

  int MatrixElementKernelDevice::computeGoodHelicities()
  {
    PinnedHostBufferHelicityMask hstIsGoodHel( ProcessData::ncomb );
    // ... 0d1. Compute good helicity mask (a host variable) on the device
    m_deviceParams.update( m_params );
    gpuLaunchKernel( computeDependentCouplings, m_gpublocks, m_gputhreads, m_deviceParams.arg( m_params ), m_gs.data(), m_couplings.data() );
    const int nevt = m_gpublocks * m_gputhreads;
    sigmaKin_getGoodHel( m_deviceParams.arg( m_params ), m_momenta.data(), m_couplings.data(), m_iflavorVec.data(), m_matrixElements.data(), m_pHelNumerators->data(), m_pHelDenominators->data(), m_pHelJamps->data(), hstIsGoodHel.data(), nevt );
    // ... 0d3. Set good helicity list in the parameters of the process (and on the device)
    int nGoodHel = sigmaKin_setGoodHel( m_params, hstIsGoodHel.data() );
    m_deviceParams.update( m_params );
    // nGoodHel can be 0 (a subprocess whose |M|^2 vanishes identically): sigmaKin then skips the
    // helicity loop and returns |M|^2 = 0, as on the CPU. The per-helicity buffers below are
    // sized for one helicity at least, so that they are never empty (null) allocations.
    const int nGoodHelAlloc = std::max( nGoodHel, 1 );
    // Create one GPU stream for each good helicity
    for( int ighel = 0; ighel < nGoodHel; ighel++ )
      gpuStreamCreate( &m_helStreams[ighel] );
#ifndef MGONGPU_HAS_NO_BLAS
    // Create one cuBLAS/hipBLAS handle for each good helicity (attached to the default stream)
    if( m_blasColorSum )
    {
      checkGpuBlas( gpuBlasCreate( &m_blasHandle ) );
#ifdef __CUDACC__ // this must be __CUDACC__
      if( m_blasTf32Tensor )
        checkGpuBlas( cublasSetMathMode( m_blasHandle, CUBLAS_TF32_TENSOR_OP_MATH ) ); // enable TF32 tensor cores
#endif
    }
#endif
    // ... Create the "many-helicity" super-buffer of nGoodHel ME buffers (dynamically allocated because nGoodHel is determined at runtime)
    m_pHelMEs.reset( new DeviceBufferSimple( nGoodHelAlloc * nevt ) );
    // ... Create the "many-helicity" super-buffer of nGoodHel ME buffers (dynamically allocated because nGoodHel is determined at runtime)
    // ... (calling reset here deletes the previously created "one-helicity" buffers used for helicity filtering)
    m_pHelJamps.reset( new DeviceBufferAmp( nGoodHelAlloc * ProcessData::njampso * mgOnGpu::nx2 * nevt ) );
    // ... Create the numerator and denominator buffers. These no longer carry a helicity dimension:
    // ... the numerators are accumulated in place over all good helicities via atomicAdd in calculate_jamps
    // ... ([nevt][ndiagrams]) and the denominators are derived from them ([nevt]).
    m_pHelNumerators.reset( new DeviceBufferSimple( ProcessData::ndiagrams * nevt ) );
    m_pHelDenominators.reset( new DeviceBufferSimple( nevt ) );
#ifndef MGONGPU_HAS_NO_BLAS
    // Create the "many-helicity" super-buffer of temporary buffers for the cuBLAS/hipBLAS intermediate
    // results in color_sum_blas, and in mixed precision mode for the converted jamps too
    // (see blasColorSumTmpSize in color_sum.h, which is where the size is defined)
    if( m_blasColorSum ) m_pHelBlasTmp.reset( new DeviceBufferSimple2( blasColorSumTmpSize( nGoodHelAlloc, nevt ) ) );
#endif
    // Return the number of good helicities
    return nGoodHel;
  }

  //--------------------------------------------------------------------------

  void MatrixElementKernelDevice::computeMatrixElements( const bool useChannelIds )
  {
    gpuLaunchKernel( computeDependentCouplings, m_gpublocks, m_gputhreads, m_deviceParams.arg( m_params ), m_gs.data(), m_couplings.data() );
#ifndef MGONGPU_HAS_NO_BLAS
    fptype2* ghelAllBlasTmp = ( m_blasColorSum ? m_pHelBlasTmp->data() : nullptr );
    gpuBlasHandle_t* pBlasHandle = ( m_blasColorSum ? &m_blasHandle : nullptr );
#else
    fptype2* ghelAllBlasTmp = nullptr;
    gpuBlasHandle_t* pBlasHandle = nullptr;
#endif
    const unsigned int* pChannelIds = ( useChannelIds ? m_channelIds.data() : nullptr );
    sigmaKin( m_params, m_deviceParams.arg( m_params ), m_momenta.data(), m_couplings.data(), m_iflavorVec.data(), m_rndhel.data(), m_rndcol.data(), pChannelIds, nullptr, m_matrixElements.data(), m_selhel.data(), m_selcol.data(), m_colJamp2s.data(), m_pHelNumerators->data(), m_pHelDenominators->data(), nullptr, true, m_pHelMEs->data(), m_pHelJamps->data(), ghelAllBlasTmp, pBlasHandle, m_helStreams, false, m_gpublocks, m_gputhreads );
#ifdef MGONGPU_CHANNELID_DEBUG
    //std::cout << "DEBUG: MatrixElementKernelDevice::computeMatrixElements " << this << " " << ( useChannelIds ? "T" : "F" ) << " " << nevt() << std::endl;
    copyHostFromDevice( m_hstChannelIds, m_channelIds ); // FIXME?!
    const unsigned int* pHstChannelIds = ( useChannelIds ? m_hstChannelIds.data() : nullptr );
    MatrixElementKernelBase::updateNevtProcessedByChannel( pHstChannelIds, nevt() );
#endif
    checkGpu( gpuPeekAtLastError() );   // is this needed?
    checkGpu( gpuDeviceSynchronize() ); // probably not needed? but it avoids errors in sigmaKin above from appearing later on in random places...
  }

  //--------------------------------------------------------------------------

}

//============================================================================
