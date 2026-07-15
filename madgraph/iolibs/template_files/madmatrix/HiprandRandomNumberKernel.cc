// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2024) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2024).
// Integrated with the MadGraph7 project in Feb 2026.

#include "mgOnGpuConfig.h"

#include "GpuRuntime.h"
#include "MemoryBuffers.h"
#include "RandomNumberKernels.h"

#include <cassert>

#ifndef MGONGPU_HAS_NO_HIPRAND /* clang-format off */
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1 // enable hiprand for AMD (rocrand)
#endif
#include <hiprand/hiprand.h>
#define checkHiprand( code ){ assertHiprand( code, __FILE__, __LINE__ ); }
inline void assertHiprand( hiprandStatus_t code, const char *file, int line, bool abort = true )
{
  if ( code != HIPRAND_STATUS_SUCCESS )
  {
    printf( "HiprandAssert: %s:%d code=%d\n", file, line, code );
    if ( abort ) assert( code == HIPRAND_STATUS_SUCCESS );
  }
}
#endif /* clang-format on */

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //--------------------------------------------------------------------------
#ifndef MGONGPU_HAS_NO_HIPRAND
  HiprandRandomNumberKernel::HiprandRandomNumberKernel( BufferRndNumMomenta& rnarray, const bool onDevice )
    : RandomNumberKernelBase( rnarray )
    , m_isOnDevice( onDevice )
  {
    if( m_isOnDevice )
    {
#ifdef MGONGPUCPP_GPUIMPL
      if( !m_rnarray.isOnDevice() )
        throw std::runtime_error( "HiprandRandomNumberKernel on device with a host random number array" );
#else
      throw std::runtime_error( "HiprandRandomNumberKernel does not support HiprandDevice on CPU host" );
#endif
    }
    else
    {
      if( m_rnarray.isOnDevice() )
        throw std::runtime_error( "HiprandRandomNumberKernel on host with a device random number array" );
    }
    createGenerator();
  }

  //--------------------------------------------------------------------------

  HiprandRandomNumberKernel::~HiprandRandomNumberKernel()
  {
    destroyGenerator();
  }

  //--------------------------------------------------------------------------

  void HiprandRandomNumberKernel::seedGenerator( const unsigned int seed )
  {
    if( m_isOnDevice )
    {
      destroyGenerator(); // workaround for #429
      createGenerator();  // workaround for #429
    }
    //printf( "seedGenerator: seed %d\n", seed );
    checkHiprand( hiprandSetPseudoRandomGeneratorSeed( m_rnGen, seed ) );
  }

  //--------------------------------------------------------------------------

  void HiprandRandomNumberKernel::createGenerator()
  {
    //const hiprandRngType_t type = HIPRAND_RNG_PSEUDO_DEFAULT;
    //const hiprandRngType_t type = HIPRAND_RNG_PSEUDO_XORWOW;
    //const hiprandRngType_t type = HIPRAND_RNG_PSEUDO_MRG32K3A;
    const hiprandRngType_t type = HIPRAND_RNG_PSEUDO_MTGP32; // same as curand; not implemented yet (code=1000) in host code
    //const hiprandRngType_t type = HIPRAND_RNG_PSEUDO_MT19937;
    //const hiprandRngType_t type = HIPRAND_RNG_PSEUDO_PHILOX4_32_10;
    if( m_isOnDevice )
    {
      checkHiprand( hiprandCreateGenerator( &m_rnGen, type ) );
    }
    else
    {
      // See https://github.com/ROCm/hipRAND/issues/76
      throw std::runtime_error( "HiprandRandomNumberKernel on host is not supported yet (hiprandCreateGeneratorHost is not implemented yet)" );
      //checkHiprand( hiprandCreateGeneratorHost( &m_rnGen, type ) ); // ALWAYS FAILS WITH CODE=1000
    }
    // FIXME: hiprand ordering is not implemented yet
    // See https://github.com/ROCm/hipRAND/issues/75
    /*
    //checkHiprand( hiprandSetGeneratorOrdering( *&m_rnGen, HIPRAND_ORDERING_PSEUDO_LEGACY ) );
    checkHiprand( hiprandSetGeneratorOrdering( *&m_rnGen, HIPRAND_ORDERING_PSEUDO_BEST ) );
    //checkHiprand( hiprandSetGeneratorOrdering( *&m_rnGen, HIPRAND_ORDERING_PSEUDO_DYNAMIC ) );
    //checkHiprand( hiprandSetGeneratorOrdering( *&m_rnGen, HIPRAND_ORDERING_PSEUDO_SEEDED ) );
    */
  }

  //--------------------------------------------------------------------------

  void HiprandRandomNumberKernel::destroyGenerator()
  {
    checkHiprand( hiprandDestroyGenerator( m_rnGen ) );
  }

  //--------------------------------------------------------------------------

  void HiprandRandomNumberKernel::generateRnarray()
  {
#if defined MGONGPU_FPTYPE_DOUBLE
    checkHiprand( hiprandGenerateUniformDouble( m_rnGen, m_rnarray.data(), m_rnarray.size() ) );
#elif defined MGONGPU_FPTYPE_FLOAT
    checkHiprand( hiprandGenerateUniform( m_rnGen, m_rnarray.data(), m_rnarray.size() ) );
#endif
    /*
    printf( "\nHiprandRandomNumberKernel::generateRnarray size = %d\n", (int)m_rnarray.size() );
    fptype* data = m_rnarray.data();
#ifdef MGONGPUCPP_GPUIMPL
    if( m_rnarray.isOnDevice() )
    {
      data = new fptype[m_rnarray.size()]();
      checkCuda( cudaMemcpy( data, m_rnarray.data(), m_rnarray.bytes(), cudaMemcpyDeviceToHost ) );
    }
#endif
    for( int i = 0; i < ( (int)m_rnarray.size() / 4 ); i++ )
      printf( "[%4d] %f %f %f %f\n", i * 4, data[i * 4], data[i * 4 + 2], data[i * 4 + 2], data[i * 4 + 3] );
#ifdef MGONGPUCPP_GPUIMPL
    if( m_rnarray.isOnDevice() ) delete[] data;
#endif
    */
  }

  //--------------------------------------------------------------------------
#endif
}
