//==========================================================================
// This file has been automatically generated for C++ Standalone by
%(info_lines)s
//==========================================================================

#ifndef MG5_%(process_file_name)s_H
#define MG5_%(process_file_name)s_H

#include <complex>
#include <vector>
%(include_for_complex)s

#include "Parameters_%(model_name)s.h"

#include <thrust/complex.h>


__global__ void sigmaKin(cudaPitchedPtr tp, double *meDevPtr, size_t mePitch);

%(process_class_definitions)s

#endif // MG5_%(process_file_name)s_H