//==========================================================================
// This file has been automatically generated for C++ by
%(info_lines)s
//==========================================================================

#include <iostream>
#include <iomanip>
#include "Parameters_%(model_name)s.h"

    // Initialize static instance
    Parameters_%(model_name)s* Parameters_%(model_name)s::instance = 0;

    // Function to get static instance - only one instance per program
    Parameters_%(model_name)s* Parameters_%(model_name)s::getInstance(){
	if (instance == 0)
	    instance = new Parameters_%(model_name)s();

	return instance;
    }

    void Parameters_%(model_name)s::setIndependentParameters(SLHAReader& slha){
	// Define "zero"
	zero=0;
	ZERO=0;
	// Prepare a vector for indices 
	vector<int> indices(2,0);
	%(set_independent_parameters)s
    }
    void Parameters_%(model_name)s::setIndependentCouplings(){
	%(set_independent_couplings)s
    }
    void Parameters_%(model_name)s::setDependentParameters(){
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

