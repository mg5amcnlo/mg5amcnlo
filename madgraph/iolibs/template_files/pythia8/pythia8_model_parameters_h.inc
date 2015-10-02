//==========================================================================
// This file has been automatically generated for Pythia 8
%(info_lines)s
//==========================================================================

#ifndef Pythia8_parameters_%(model_name)s_H
#define Pythia8_parameters_%(model_name)s_H

#include <complex>

#include "%(include_prefix)sParticleData.h"
#include "%(include_prefix)sStandardModel.h"
#include "%(include_prefix)sSusyLesHouches.h"

using namespace std;

namespace Pythia8 {

class Parameters_%(model_name)s
{
public:

static Parameters_%(model_name)s* getInstance();

%(independent_parameters)s
%(independent_couplings)s
%(dependent_parameters)s
%(dependent_couplings)s

// Set parameters that are unchanged during the run
void setIndependentParameters(ParticleData*& pd, Couplings*& csm, SusyLesHouches*& slhaPtr);
// Set couplings that are unchanged during the run
void setIndependentCouplings();
// Set parameters that are changed event by event
void setDependentParameters(ParticleData*& pd, Couplings*& csm, SusyLesHouches*& slhaPtr, double alpS);
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

} // end namespace Pythia8
#endif // Pythia8_parameters_%(model_name)s_H
