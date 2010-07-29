//==========================================================================
// This file has been automatically generated for Pythia 8 by
%(info_lines)s
//==========================================================================

#include "Parameters_%(model_name)s.h"
#include "PythiaStdlib.h"

namespace Pythia8 {

    // Initialize static instance
    Parameters_%(model_name)s* Parameters_%(model_name)s::instance = 0;

    // Function to get static instance - only one instance per program
    Parameters_%(model_name)s* Parameters_%(model_name)s::getInstance(){
	if (instance == 0)
	    instance = new Parameters_%(model_name)s();

	return instance;
    }

    void Parameters_%(model_name)s::setIndependentParameters(ParticleData*& pd, CoupSM*& csm){
	// First set "zero"
	zero=0;
	ZERO=0;
	%(set_independent_parameters)s
    }
    void Parameters_%(model_name)s::setIndependentCouplings(ParticleData*& pd, CoupSM*& csm){
	%(set_independent_couplings)s
    }
    void Parameters_%(model_name)s::setDependentParameters(ParticleData*& pd, CoupSM*& csm, double alpS){
	%(set_dependent_parameters)s
    }
    void Parameters_%(model_name)s::setDependentCouplings(ParticleData*& pd, CoupSM*&csm){
	%(set_dependent_couplings)s
    }

} // end namespace Pythia8
