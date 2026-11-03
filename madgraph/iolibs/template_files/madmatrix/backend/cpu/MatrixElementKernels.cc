// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2022) for the MadGraph7 CUDACPP plugin.
// Further modified by: D. Massaro, J. Teig, A. Thete, A. Valassi, Z. Wettersten (2022-2025).
// Integrated with the MadGraph7 project in Feb 2026.

#include "MatrixElementKernels.h"

#include "ProcessData.h"
#include "SigmaKin.h"
#include "MemoryAccessMomenta.h"
#include "MemoryBuffers.h"
#include "color_sum.h" // for blasColorSumTmpSize

#include <cassert>
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

namespace madmatrix
{

  //--------------------------------------------------------------------------

  MatrixElementKernelHost::MatrixElementKernelHost( ProcessParameters& params,            // input: parameters of the process (output: good helicities)
                                                    const BufferMomenta& momenta,         // input: momenta
                                                    const BufferGs& gs,                   // input: gs for alphaS
                                                    const BufferIflavorVec& iflavorVec,   // input: flavor indices for the flavor combination
                                                    const BufferRndNumHelicity& rndhel,   // input: random numbers for helicity selection
                                                    const BufferRndNumColor& rndcol,      // input: random numbers for color selection
                                                    const BufferChannelIds& channelIds,   // input: channel ids for single-diagram enhancement
                                                    BufferMatrixElements& matrixElements, // output: matrix elements
                                                    BufferSelectedHelicity& selhel,       // output: helicity selection
                                                    BufferSelectedColor& selcol,          // output: color selection
                                                    const size_t nevt )
    : MatrixElementKernelBase( params, momenta, gs, iflavorVec, rndhel, rndcol, channelIds, matrixElements, selhel, selcol )
    , NumberOfEvents( nevt )
    , m_couplings( nevt )
    , m_numerators( nevt * ProcessData::ndiagrams )
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
    HostBufferHelicityMask hstIsGoodHel( ProcessData::ncomb );
    // ... 0d1. Compute good helicity mask on the host
    computeDependentCouplings( m_params, m_gs.data(), m_couplings.data(), m_gs.size() );
    sigmaKin_getGoodHel( m_params, m_momenta.data(), m_couplings.data(), m_iflavorVec.data(), m_matrixElements.data(), m_numerators.data(), m_denominators.data(), hstIsGoodHel.data(), nevt() );
    // ... 0d2. Copy good helicity list to the parameters of the process
    return sigmaKin_setGoodHel( m_params, hstIsGoodHel.data() );
  }

  //--------------------------------------------------------------------------

  void MatrixElementKernelHost::computeMatrixElements( const bool useChannelIds )
  {
    computeDependentCouplings( m_params, m_gs.data(), m_couplings.data(), m_gs.size() );
    const unsigned int* pChannelIds = ( useChannelIds ? m_channelIds.data() : nullptr );
    sigmaKin( m_params, m_momenta.data(), m_couplings.data(), m_iflavorVec.data(), m_rndhel.data(), m_rndcol.data(), pChannelIds, nullptr, m_matrixElements.data(), m_selhel.data(), m_selcol.data(), m_numerators.data(), m_denominators.data(), nullptr, true, nevt() );
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

//============================================================================


//============================================================================
