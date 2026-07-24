//==========================================================================
// This file has been automatically generated for C++ Standalone by
%(info_lines)s
//==========================================================================

%(helamps_h)s

#ifndef MG5_%(process_file_name)s_H
#define MG5_%(process_file_name)s_H

#include <complex>
#include <vector>
#include <cassert>
#include <iostream>

#include "mgOnGpuConfig.h"
#include "mgOnGpuTypes.h"


#include "Parameters_%(model_name)s.h"

//--------------------------------------------------------------------------

#ifdef __CUDACC__

#define checkCuda( code ) { assertCuda( code, __FILE__, __LINE__ ); }

inline void assertCuda( cudaError_t code, const char *file, int line, bool abort = true )
{
  if ( code != cudaSuccess )
  {
    printf( "GPUassert: %%s %%s:%%d\n", cudaGetErrorString(code), file, line );
    if ( abort ) assert( code == cudaSuccess );
  }
}

#endif
//--------------------------------------------------------------------------

#ifdef __CUDACC__
namespace gProc
#else
namespace Proc
#endif
{

%(process_class_definitions)s


  //--------------------------------------------------------------------------
#ifdef __CUDACC__
  __global__
  void sigmaKin_getGoodHel( const fptype* allmomenta, // input: momenta as AOSOA[npagM][npar][4][neppM] with nevt=npagM*neppM
                            bool* isGoodHel );        // output: isGoodHel[ncomb] - device array
#endif

  //--------------------------------------------------------------------------

#ifdef __CUDACC__
  void sigmaKin_setGoodHel( const bool* isGoodHel ); // input: isGoodHel[ncomb] - host array
#endif

  //--------------------------------------------------------------------------

  __global__
  void sigmaKin( const fptype* allmomenta, // input: momenta as AOSOA[npagM][npar][4][neppM] with nevt=npagM*neppM
                 fptype* allMEs            // output: allMEs[nevt], final |M|^2 averaged over all helicities
#ifndef __CUDACC__
                 , const int nevt          // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
#endif
                 );

  //--------------------------------------------------------------------------
}

#endif // MG5_%(process_file_name)s_H