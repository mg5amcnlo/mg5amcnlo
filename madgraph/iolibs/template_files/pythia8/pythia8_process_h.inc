//==========================================================================
// This file has been automatically generated for Pythia 8
%(info_lines)s
//==========================================================================

#ifndef Pythia8_%(process_file_name)s_H
#define Pythia8_%(process_file_name)s_H

#include <complex>

#include "%(include_prefix)sSigmaProcess.h"
#include "Parameters_%(model_name)s.h"

using namespace std;

namespace Pythia8 {
%(process_class_definitions)s
} // end namespace Pythia8

#endif // Pythia8_%(process_file_name)s_H
