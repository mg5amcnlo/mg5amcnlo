//==========================================================================
// This file has been automatically generated for C++
%(info_lines)s
//==========================================================================

#ifndef Parameters_%(model_name)s_H
#define Parameters_%(model_name)s_H

#include <complex>

#include "read_slha.h"
using namespace std;

class Parameters_%(model_name)s
{
public:

static Parameters_%(model_name)s* getInstance();

// Define "zero"
double zero, ZERO;
%(independent_parameters)s
%(independent_couplings)s
%(dependent_parameters)s
%(dependent_couplings)s

// Set parameters that are unchanged during the run
void setIndependentParameters(SLHAReader& slha);
// Set couplings that are unchanged during the run
void setIndependentCouplings();
// Set parameters that are changed event by event
void setDependentParameters();
// Set couplings that are changed event by event
void setDependentCouplings();

// Print parameters that are unchanged during the run
void printIndependentParameters();
// Print couplings that are unchanged during the run
void printIndependentCouplings();
// Print parameters that are changed event by event
void printDependentParameters();
// Print couplings that are changed event by event
void printDependentCouplings();


  private:
    static Parameters_%(model_name)s* instance;
};

#endif // Parameters_%(model_name)s_H
