//==========================================================================
// This file has been automatically generated for %(output_name)s
%(info_lines)s
// GPU Template
//==========================================================================

#ifndef HelAmps_%(model_name)s_H
#define HelAmps_%(model_name)s_H

#include <cmath> 
#include "mgOnGpuConfig.h"
#include "mgOnGpuTypes.h"

using namespace std;

namespace %(namespace)s_%(model_name)s {
%(function_declarations)s
} // end namespace %(namespace)s_%(model_name)s

#endif // HelAmps_%(model_name)s_H