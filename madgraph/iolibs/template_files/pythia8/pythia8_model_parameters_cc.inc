//==========================================================================
// This file has been automatically generated for Pythia 8 by
%(info_lines)s
//==========================================================================

#include <iostream>
#include "Parameters_%(model_name)s.h"
#include "%(include_prefix)sPythiaStdlib.h"

namespace Pythia8 {

    // Initialize static instance
    Parameters_%(model_name)s* Parameters_%(model_name)s::instance = 0;

    // Function to get static instance - only one instance per program
    Parameters_%(model_name)s* Parameters_%(model_name)s::getInstance(){
	if (instance == 0)
	    instance = new Parameters_%(model_name)s();

	return instance;
    }

    void Parameters_%(model_name)s::setIndependentParameters(ParticleData*& pd, Couplings*& csm, SusyLesHouches*& slhaPtr){
	%(set_independent_parameters)s
    }
    void Parameters_%(model_name)s::setIndependentCouplings(){
	%(set_independent_couplings)s
    }
    void Parameters_%(model_name)s::setDependentParameters(ParticleData*& pd, Couplings*& csm, SusyLesHouches*& slhaPtr, double alpS){
	%(set_dependent_parameters)s
    }
    void Parameters_%(model_name)s::setDependentCouplings(){
	%(set_dependent_couplings)s
    }

    // Routines for printing out parameters
    void Parameters_%(model_name)s::printIndependentParameters(){
	cout << "%(model_name)s model parameters independent of event kinematics:" << endl;
	%(print_independent_parameters)s
    }
    void Parameters_%(model_name)s::printIndependentCouplings(){
	cout << "%(model_name)s model couplings independent of event kinematics:" << endl;
	%(print_independent_couplings)s
    }
    void Parameters_%(model_name)s::printDependentParameters(){
	cout << "%(model_name)s model parameters dependent on event kinematics:" << endl;
	%(print_dependent_parameters)s
    }
    void Parameters_%(model_name)s::printDependentCouplings(){
	cout << "%(model_name)s model couplings dependent on event kinematics:" << endl;
	%(print_dependent_couplings)s
    }

} // end namespace Pythia8
