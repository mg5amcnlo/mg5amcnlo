// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2022) for the MG5aMC CUDACPP plugin.
// Further modified by: D. Massaro, J. Teig, A. Thete, A. Valassi, Z. Wettersten (2022-2025).
// Integrated with the MadGraph7 project in Feb 2026.

#include "MatrixElementKernels.h"

#include "CPPProcess.h"
#include "GpuRuntime.h" // Includes the abstraction for Nvidia/AMD compilation
#include "MemoryAccessMomenta.h"
#include "MemoryBuffers.h"

#include <cfenv> // for fetestexcept
#include <iostream>
#include <sstream>

//============================================================================

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //--------------------------------------------------------------------------

  MatrixElementKernelBase::MatrixElementKernelBase( const BufferMomenta& momenta,         // input: momenta
                                                    const BufferGs& gs,                   // input: gs for alphaS
                                                    const BufferIflavorVec& iflavorVec,   // input: flavor indices for the flavor combination
                                                    const BufferRndNumHelicity& rndhel,   // input: random numbers for helicity selection
                                                    const BufferRndNumColor& rndcol,      // input: random numbers for color selection
                                                    const BufferChannelIds& channelIds,   // input: channel ids for single-diagram enhancement
                                                    BufferMatrixElements& matrixElements, // output: matrix elements
                                                    BufferSelectedHelicity& selhel,       // output: helicity selection
                                                    BufferSelectedColor& selcol )         // output: color selection
    : m_momenta( momenta )
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
    for( size_t channelId = 0; channelId < CPPProcess::ndiagrams + 1; channelId++ ) // [0...ndiagrams] (TEMPORARY: 0=multichannel)
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
        //assert( channelId < CPPProcess::ndiagrams );
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
    for( size_t channelId = 0; channelId < CPPProcess::ndiagrams + 1; channelId++ ) // [0...ndiagrams] (TEMPORARY: 0=multichannel)
      nevtProcessed += m_nevtProcessedByChannel[channelId];
    std::ostringstream sstr;
    sstr << " {";
    for( size_t channelId = 0; channelId < CPPProcess::ndiagrams + 1; channelId++ ) // [0...ndiagrams] (TEMPORARY: 0=multichannel)
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
    std::cout << " processed " << nevtProcessed << " events across " << CPPProcess::ndiagrams << " channels" << sstr.str() << std::endl;
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

#ifndef MGONGPUCPP_GPUIMPL
namespace mg5amcCpu
{

  //--------------------------------------------------------------------------

  MatrixElementKernelHost::MatrixElementKernelHost( const BufferMomenta& momenta,         // input: momenta
                                                    const BufferGs& gs,                   // input: gs for alphaS
                                                    const BufferIflavorVec& iflavorVec,   // input: flavor indices for the flavor combination
                                                    const BufferRndNumHelicity& rndhel,   // input: random numbers for helicity selection
                                                    const BufferRndNumColor& rndcol,      // input: random numbers for color selection
                                                    const BufferChannelIds& channelIds,   // input: channel ids for single-diagram enhancement
                                                    BufferMatrixElements& matrixElements, // output: matrix elements
                                                    BufferSelectedHelicity& selhel,       // output: helicity selection
                                                    BufferSelectedColor& selcol,          // output: color selection
                                                    const size_t nevt )
    : MatrixElementKernelBase( momenta, gs, iflavorVec, rndhel, rndcol, channelIds, matrixElements, selhel, selcol )
    , NumberOfEvents( nevt )
    , m_couplings( nevt )
    , m_numerators( nevt * CPPProcess::ndiagrams )
    , m_denominators( nevt )
  {
    //std::cout << "DEBUG: MatrixElementKernelHost::ctor " << this << std::endl;
    if( m_momenta.isOnDevice() ) throw std::runtime_error( "MatrixElementKernelHost: momenta must be a host array" );
    if( m_matrixElements.isOnDevice() ) throw std::runtime_error( "MatrixElementKernelHost: matrixElements must be a host array" );
    if( m_channelIds.isOnDevice() ) throw std::runtime_error( "MatrixElementKernelHost: channelIds must be a device array" );
    if( this->nevt() != m_momenta.nevt() ) throw std::runtime_error( "MatrixElementKernelHost: nevt mismatch with momenta" );
    if( this->nevt() != m_matrixElements.nevt() ) throw std::runtime_error( "MatrixElementKernelHost: nevt mismatch with matrixElements" );
    if( this->nevt() != m_channelIds.nevt() ) throw std::runtime_error( "MatrixElementKernelHost: nevt mismatch with channelIds" );
    if( this->nevt() != m_iflavorVec.nevt() ) throw std::runtime_error( "MatrixElementKernelHost: nevt mismatch with iflavorVec" );
    // Sanity checks for memory access (momenta buffer)
    constexpr int neppM = MemoryAccessMomenta::neppM; // AOSOA layout
    static_assert( ispoweroftwo( neppM ), "neppM is not a power of 2" );
    if( nevt % neppM != 0 )
    {
      std::ostringstream sstr;
      sstr << "MatrixElementKernelHost: nevt should be a multiple of neppM=" << neppM;
      throw std::runtime_error( sstr.str() );
    }
    // Fail gently and avoid "Illegal instruction (core dumped)" if the host does not support the SIMD used in the ME calculation
    // Note: this prevents a crash on pmpe04 but not on some github CI nodes?
    // [NB: SIMD vectorization in mg5amc C++ code is only used in the ME calculation below MatrixElementKernelHost!]
    if( !MatrixElementKernelHost::hostSupportsSIMD() )
      throw std::runtime_error( "Host does not support the SIMD implementation of MatrixElementKernelsHost" );
  }

  //--------------------------------------------------------------------------

  MatrixElementKernelHost::~MatrixElementKernelHost()
  {
    //std::cout << "DEBUG: MatrixElementKernelBase::dtor " << this << std::endl;
  }

  //--------------------------------------------------------------------------

  int MatrixElementKernelHost::computeGoodHelicities()
  {
    HostBufferHelicityMask hstIsGoodHel( CPPProcess::ncomb );
    // ... 0d1. Compute good helicity mask on the host
    computeDependentCouplings( m_gs.data(), m_couplings.data(), m_gs.size() );
    sigmaKin_getGoodHel( m_momenta.data(), m_couplings.data(), m_iflavorVec.data(), m_matrixElements.data(), m_numerators.data(), m_denominators.data(), hstIsGoodHel.data(), nevt() );
    // ... 0d2. Copy good helicity list to static memory on the host
    // [FIXME! REMOVE THIS STATIC THAT BREAKS MULTITHREADING?]
    return sigmaKin_setGoodHel( hstIsGoodHel.data() );
  }

  //--------------------------------------------------------------------------

  void MatrixElementKernelHost::computeMatrixElements( const bool useChannelIds )
  {
    computeDependentCouplings( m_gs.data(), m_couplings.data(), m_gs.size() );
    const unsigned int* pChannelIds = ( useChannelIds ? m_channelIds.data() : nullptr );
    sigmaKin( m_momenta.data(), m_couplings.data(), m_iflavorVec.data(), m_rndhel.data(), m_rndcol.data(), pChannelIds, nullptr, m_matrixElements.data(), m_selhel.data(), m_selcol.data(), m_numerators.data(), m_denominators.data(), nullptr, true, nevt() );
#ifdef MGONGPU_CHANNELID_DEBUG
    //std::cout << "DEBUG: MatrixElementKernelHost::computeMatrixElements " << this << " " << ( useChannelIds ? "T" : "F" ) << " " << nevt() << std::endl;
    MatrixElementKernelBase::updateNevtProcessedByChannel( pChannelIds, nevt() );
#endif
  }

  //--------------------------------------------------------------------------

  // Does this host system support the SIMD used in the matrix element calculation?
  bool MatrixElementKernelHost::hostSupportsSIMD( const bool verbose )
  {
#if defined __AVX512VL__
    bool known = true;
    bool ok = __builtin_cpu_supports( "avx512vl" );
    const std::string tag = "skylake-avx512 (AVX512VL)";
#elif defined __AVX2__
    bool known = true;
    bool ok = __builtin_cpu_supports( "avx2" );
    const std::string tag = "haswell (AVX2)";
#elif defined __SSE4_2__
#ifdef __PPC__
    // See https://gcc.gnu.org/onlinedocs/gcc/Basic-PowerPC-Built-in-Functions-Available-on-all-Configurations.html
    bool known = true;
    bool ok = __builtin_cpu_supports( "vsx" );
    const std::string tag = "powerpc vsx (128bit as in SSE4.2)";
#elif defined( __x86_64__ ) || defined( __i386__ )
    bool known = true;
    bool ok = __builtin_cpu_supports( "sse4.2" );
    const std::string tag = "nehalem (SSE4.2)";
#else // AV FIXME! Added by OM for Mac, should identify the correct __xxx__ flag that should be targeted
    // DM now we have an explicit NEON target for ARM
    bool known = false; // __builtin_cpu_supports is not supported
    bool ok = true;     // this is just an assumption!
    const std::string tag = "simd arch not defined";
#endif
#elif defined __ARM_NEON // consider using __BUILTIN_CPU_SUPPORTS__
    bool known = false; // __builtin_cpu_supports is not supported
    // See https://stackoverflow.com/q/62783908
    // See https://community.arm.com/arm-community-blogs/b/operating-systems-blog/posts/runtime-detection-of-cpu-features-on-an-armv8-a-cpu
    bool ok = true; // this is just an assumption!
    const std::string tag = "arm neon (128bit as in SSE4.2)";
#else
    bool known = true;
    bool ok = true;
    const std::string tag = "none";
#endif
    if( verbose )
    {
      if( tag == "none" )
        std::cout << "INFO: The application does not require the host to support any AVX feature" << std::endl;
      else if( ok && known )
        std::cout << "INFO: The application is built for " << tag << " and the host supports it" << std::endl;
      else if( ok )
        std::cout << "WARNING: The application is built for " << tag << " but it is unknown if the host supports it" << std::endl;
      else
        std::cout << "ERROR! The application is built for " << tag << " but the host does not support it" << std::endl;
    }
    return ok;
  }

  //--------------------------------------------------------------------------

}
#endif

//============================================================================

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
{

  //--------------------------------------------------------------------------

  MatrixElementKernelDevice::MatrixElementKernelDevice( const BufferMomenta& momenta,         // input: momenta
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
    : MatrixElementKernelBase( momenta, gs, iflavorVec, rndhel, rndcol, channelIds, matrixElements, selhel, selcol )
    , NumberOfEvents( gpublocks * gputhreads )
    , m_couplings( this->nevt() )
    , m_pHelMEs()
    , m_pHelJamps()
    , m_pHelNumerators()
    , m_pHelDenominators()
    , m_colJamp2s( CPPProcess::ncolor * this->nevt() )
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
    m_pHelJamps.reset( new DeviceBufferSimple( CPPProcess::ncolor * mgOnGpu::nx2 * this->nevt() ) );
    // Create the "one-helicity" numerator and denominator buffers that will be used for helicity filtering
    m_pHelNumerators.reset( new DeviceBufferSimple( this->nevt() * CPPProcess::ndiagrams ) );
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
        m_blasColorSum = true; // fixme? eventually set default=true and decode "Y" and "N" choices?
        std::cout << "INFO: Env variable CUDACPP_RUNTIME_BLASCOLORSUM is set and non-empty: enable BLAS" << std::endl;
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
#ifdef __CUDACC__ // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)
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
    for( int ihel = 0; ihel < CPPProcess::ncomb; ihel++ )
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
    PinnedHostBufferHelicityMask hstIsGoodHel( CPPProcess::ncomb );
    // ... 0d1. Compute good helicity mask (a host variable) on the device
    gpuLaunchKernel( computeDependentCouplings, m_gpublocks, m_gputhreads, m_gs.data(), m_couplings.data() );
    const int nevt = m_gpublocks * m_gputhreads;
    sigmaKin_getGoodHel( m_momenta.data(), m_couplings.data(), m_iflavorVec.data(), m_matrixElements.data(), m_pHelJamps->data(), m_pHelNumerators->data(), m_pHelDenominators->data(), hstIsGoodHel.data(), nevt );
    // ... 0d3. Set good helicity list in host static memory
    int nGoodHel = sigmaKin_setGoodHel( hstIsGoodHel.data() );
    assert( nGoodHel > 0 ); // SANITY CHECK: there should be at least one good helicity
    // Create one GPU stream for each good helicity
    for( int ighel = 0; ighel < nGoodHel; ighel++ )
      gpuStreamCreate( &m_helStreams[ighel] );
#ifndef MGONGPU_HAS_NO_BLAS
    // Create one cuBLAS/hipBLAS handle for each good helicity (attached to the default stream)
    if( m_blasColorSum )
    {
      checkGpuBlas( gpuBlasCreate( &m_blasHandle ) );
#ifdef __CUDACC__ // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)
      if( m_blasTf32Tensor )
        checkGpuBlas( cublasSetMathMode( m_blasHandle, CUBLAS_TF32_TENSOR_OP_MATH ) ); // enable TF32 tensor cores
#endif
    }
#endif
    // ... Create the "many-helicity" super-buffer of nGoodHel ME buffers (dynamically allocated because nGoodHel is determined at runtime)
    m_pHelMEs.reset( new DeviceBufferSimple( nGoodHel * nevt ) );
    // ... Create the "many-helicity" super-buffer of nGoodHel ME buffers (dynamically allocated because nGoodHel is determined at runtime)
    // ... (calling reset here deletes the previously created "one-helicity" buffers used for helicity filtering)
    m_pHelJamps.reset( new DeviceBufferSimple( nGoodHel * CPPProcess::ncolor * mgOnGpu::nx2 * nevt ) );
    // ... Create the "many-helicity" super-buffers of nGoodHel numerator and denominator buffers (dynamically allocated)
    // ... (calling reset here deletes the previously created "one-helicity" buffers used for helicity filtering)
    m_pHelNumerators.reset( new DeviceBufferSimple( nGoodHel * CPPProcess::ndiagrams * nevt ) );
    m_pHelDenominators.reset( new DeviceBufferSimple( nGoodHel * nevt ) );
#ifndef MGONGPU_HAS_NO_BLAS
    // Create the "many-helicity" super-buffers of real/imag ncolor*nevt temporary buffers for cuBLAS/hipBLAS intermediate results in color_sum_blas
#if defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    // Mixed precision mode: need two fptype2[ncolor*2*nevt] buffers and one fptype2[nevt] buffer per good helicity
    if( m_blasColorSum ) m_pHelBlasTmp.reset( new DeviceBufferSimple2( nGoodHel * ( 2 * CPPProcess::ncolor * mgOnGpu::nx2 + 1 ) * nevt ) );
#else
    // Standard single/double precision mode: need one fptype2[ncolor*2*nevt] buffer per good helicity
    if( m_blasColorSum ) m_pHelBlasTmp.reset( new DeviceBufferSimple2( nGoodHel * CPPProcess::ncolor * mgOnGpu::nx2 * nevt ) );
#endif
#endif
    // Return the number of good helicities
    return nGoodHel;
  }

  //--------------------------------------------------------------------------

  void MatrixElementKernelDevice::computeMatrixElements( const bool useChannelIds )
  {
    gpuLaunchKernel( computeDependentCouplings, m_gpublocks, m_gputhreads, m_gs.data(), m_couplings.data() );
#ifndef MGONGPU_HAS_NO_BLAS
    fptype2* ghelAllBlasTmp = ( m_blasColorSum ? m_pHelBlasTmp->data() : nullptr );
    gpuBlasHandle_t* pBlasHandle = ( m_blasColorSum ? &m_blasHandle : nullptr );
#else
    fptype2* ghelAllBlasTmp = nullptr;
    gpuBlasHandle_t* pBlasHandle = nullptr;
#endif
    const unsigned int* pChannelIds = ( useChannelIds ? m_channelIds.data() : nullptr );
    sigmaKin( m_momenta.data(), m_couplings.data(), m_iflavorVec.data(), m_rndhel.data(), m_rndcol.data(), pChannelIds, nullptr, m_matrixElements.data(), m_selhel.data(), m_selcol.data(), m_colJamp2s.data(), m_pHelNumerators->data(), m_pHelDenominators->data(), nullptr, true, m_pHelMEs->data(), m_pHelJamps->data(), ghelAllBlasTmp, pBlasHandle, m_helStreams, false, m_gpublocks, m_gputhreads );
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
#endif

//============================================================================
