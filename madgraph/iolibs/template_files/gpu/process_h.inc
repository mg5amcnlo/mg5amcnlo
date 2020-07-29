//==========================================================================
// This file has been automatically generated for C++ Standalone by
%(info_lines)s
//==========================================================================

#ifndef MG5_%(process_file_name)s_H
#define MG5_%(process_file_name)s_H

#include <complex>
#include <vector>
#include <cassert>

%(include_for_complex)s

#include "Parameters_%(model_name)s.h"

#include <thrust/complex.h>

#define gpuErrchk3( code ) { gpuAssert3( code, __FILE__, __LINE__ ); }
	
inline void gpuAssert3( cudaError_t code, const char *file, int line, bool abort = true )
{
  if ( code != cudaSuccess )
  {
    printf( "GPUassert: %%s %%s %%d\n", cudaGetErrorString(code), file, line );
    if ( abort ) assert( code == cudaSuccess );
  }
}


__global__ void sigmaKin(double * allmomenta, double * output);

%(process_class_definitions)s

#endif // MG5_%(process_file_name)s_H